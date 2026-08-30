"""
Trainer, by definition, takes care of anything related to training,
Trainer should have all information for a training process
e.g., logging, epoch count, inference, model, criterion, optimizer, scheduler, dataloader,
      channel (in SemCom we need to model the channel), etc.

Trainer is also the unit we do tests on, for example, for trainers related to data reconstruction, 
trainer itself already specifies the system model to transmit this data in (ref. ReconstructionTrainer.transmit())
This is why you'd see a lot of trainer related stuff in the src/test folder: in that case, the trainer is to specify the system model,
instead of actual training.
"""

import torch
from torch import nn
from pathlib import Path
from typing import *
import logging
import numpy as np
import random
import sys

import datetime
from types import ModuleType

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.autograd import Variable

from tqdm import tqdm

from ..utils import toColor, get_system_info, get_scheduler_str, get_git_commit_hash, str_type, to_device

class BaseTrainer:
    """
        Trainer that does basic per-epoch logging, saving and loading.

        only train() is public, and when devising your own trainer, you should consider
        overloading most of the below methods:
        - Initializing: __init__()
        - Training: _train_epoch(), _eval()
        - Postprocessing: _on_epoch_finish(), _on_train_finish()
        - Checkpoint: _save_checkpoint(), _resume_checkpoint()

        you will have access to the following variables in these functions:
            logger, save_dir, checkpoint_dir, global_step, start_epoch, n_epoch,
            display_interval, with_cuda, gpus, device, metric
    """
    def __init__(self, 
                 # pass by program
                 logger: logging.Logger,
                 model: nn.Module, criterion: nn.Module | None,
                 optimizer: Optimizer | dict | None, lr_scheduler: Optimizer | dict | None, 
                 # pass by config
                 save_dir: str | Path, display_interval: int,
                 n_epoch: int, gpus: list[int], seed: Optional[int] = None,
                 resume_checkpoint: Optional[str | Path] = None, 
                 weights_init: Union[Callable[[nn.Module], None], None] = None
                 ):

        """
            Args:
                logger: will use its .info() method for logging
                model: the model to train on.
                       will help you do .to(device) and saving / resuming checkpoint
                       if you have a lot of models that you need to train in one session (i.e., your loss function is related to mulitple models), 
                       make them all be in one nn.Module. we only accept one and only one nn.Module.
                optimizer, lr_scheduler: can give class or a config dict for them
                                         (config should have key ['type', 'args'], ref. self._initialize)
                                         optimizer can be None if you only wanna use the trainer for testing, but make sure to not call .train()
                                         lr_scheduler can be None
                save_dir: the main saving directory, most things will store in this directory
                          including checkpoints and other things the subclass think is necessary
                display_interval: every `display_interval` batches it will log the batch info
                                  implemented in subclass's self._train_epoch()
                n_epoch: the count of epoch to train
                seed: the seed to set torch, VERY RECOMMEND to set (if give None then skip all set seed related function)
                gpus: the list of gpus to use, if len(gpus) == 0 (or gpu unavailable) then use cpu
                      (currently only support one gpu, i.e., len(gpus) == 1)
                resume_checkpoint: a model (.pth) file to continue from. contains more state related
                                   to training and other things, ref. self._save_checkpoint() and self._resume_checkpoint
                weights_init: the function to initialize model if not resuming from checkpoint, 
                              will model.apply()
            
            If you wanna support resume_checkpoint in the subclass, you should set self.[model, optimizer, scheduler] first
            before calling super().__init__

            Note that some things may be loaded during _resume_checkpoint (e.g., self.metrics),
            make sure subclass's __init__ don't wipe them out
        """
        
        # checkpoint paths
        self.logger = logger
        self.logger.info(f"Command line: {sys.argv}")
        self.logger.info(f"Current Git commit version: {get_git_commit_hash()}")
        self.logger.info(f'save_dir = {str(save_dir)}')
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = self.save_dir / 'checkpoint'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # step-related setting
        self.global_step = 0
        self.start_epoch = 1
        self.n_epochs = n_epoch    # inclusive: train for [start_epoch, n_epoch]
        self.display_interval = display_interval

        # set seed
        if seed is not None:
            self.logger.info(f'seed = {seed}')
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # set device
        self.logger.info(f'System info:')
        self.logger.info(get_system_info())
        self.logger.info(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
        self.logger.info(f'torch.cuda.device_count() = {torch.cuda.device_count()}')

        if len(gpus) > 0 and torch.cuda.is_available():

            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.logger.info(f'train with gpu {gpus} and pytorch {torch.__version__}')

            self.gpus = {i: item for i, item in enumerate(gpus)}
            self.device = torch.device(f"cuda:{','.join(str(gpu_id) for gpu_id in gpus)}")

            if seed is not None:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        else:
            self.with_cuda = False
            self.logger.info(f'train with cpu and pytorch {torch.__version__}')
            self.device = torch.device("cpu")

        self.logger.info('device = {}'.format(self.device))
        
        # set model related stuff
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        if optimizer is not None and not isinstance(optimizer, Optimizer):
            self.optimizer = self._initialize(self.optimizer, torch.optim, model.parameters())
        if lr_scheduler is not None and not isinstance(lr_scheduler, LRScheduler):
            self.scheduler = self._initialize(lr_scheduler, torch.optim.lr_scheduler, self.optimizer)

        self.logger.info(f'model = {model}')
        self.logger.info(f'criterion = {criterion}')
        self.logger.info(f'optimizer = {optimizer}')
        self.logger.info(f'scheduler = {get_scheduler_str(self.scheduler)}')

        # kinda a placeholder, this will be saved and loaded in _save_checkpoint and _resume_checkpoint
        # do whatever you want
        self.metrics = {}
        
        # resume checkpoint or initialize weight
        # might overwrite some variables above, should be placed at the very end!
        if resume_checkpoint:
            self._resume_checkpoint(resume_checkpoint)
        else:
            if weights_init is not None:
                self.model.apply(weights_init)
        
    def train(self) -> None:
        """
            The main training function
        """

        # log the trainer variables
        # try:
        trainer_var = {**vars(self)}
        if 'metrics' in trainer_var: 
            del trainer_var['metrics']
        self.logger.info('Training started')
        self.logger.info(f'Trainer: {str_type(trainer_var, indent=4, array_limit_items=20)}')
        # except Exception as e:
        #     self.logger.info(f'Error when logging trainer variables: {e}')

        # train
        progress_bar = tqdm(range(self.start_epoch, self.n_epochs + 1), leave=False, dynamic_ncols=True)
        for epoch in progress_bar:
            progress_bar.set_description(f'Epoch {epoch}/{self.n_epochs}')
            try:
                # train result
                self.epoch_result = self._train_epoch(epoch)

                # epoch finished
                self._eval(epoch)
                self._on_epoch_finish(epoch)
                self._log_memory_usage()

            except torch.cuda.CudaError:
                self._log_memory_usage()
            
        # train finished
        self._on_train_finish()
    
    def _train_epoch(self, epoch: int) -> dict:
        """
            main logic to train one epoch, don't step optimizer
            remember to call model.train() in this function
            
            Args:
                epoch: current epoch number, 1-indexed
            
            Returns:
                a dictionary of result, will set as self.epoch_result for later (_on_epoch_finish) uses.
                self.epoch_result['train_loss'] will be used for stepping scheduler.
        """
        raise NotImplementedError

    def _eval(self, epoch: int) -> None:
        """
            main logic to do validation, will do after an epoch of training finished, but before _on_epoch_finish()
            you can store something in class to use on _on_epoch_finish
            remember to call model.eval() or torch.no_grad() in this function

            Args:
                epoch: current epoch number, 1-indexed
        """
        raise NotImplementedError

    def _on_epoch_finish(self, epoch: int) -> None:
        """
            main logic for when an epoch finishes, will do after _eval()
            you can put your store model logic and step epoch scheduler here.
            you have access to self.epoch_result and things you stored during _eval()

            Args:
                epoch: current epoch number, 1-indexed
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.epoch_result['train_loss'])
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass
            else:
                self.scheduler.step()

    def _on_train_finish(self) -> None:
        """
            this is called after all training has finished (i.e., all epochs done)
        """
        pass

    def _log_memory_usage(self) -> None:
        """
            logs memory usage to logger, with... questionable practicality.
            called after _on_epoch_finish() and when there is CudaError happening

            after pytorch 2.1 there is a memory snapshot tool.
            ref. 
            - https://pytorch.org/docs/stable/torch_cuda_memory.html
            - https://pytorch.org/blog/understanding-gpu-memory-1/
        """
        if not self.with_cuda:
            return

        template = """Memory Usage: \n{}"""
        usage = []
        for deviceID, device in self.gpus.items():
            deviceID = int(deviceID)
            allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(deviceID) / (1024 * 1024)

            usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

        content = ''.join(usage)
        content = template.format(content)

        self.logger.debug(content)

    def _get_checkpoint_state(self, epoch) -> dict:
        """
            get the checkpoint state, this is called in _save_checkpoint
            you can overload this function if you want to save more stuff
            or change the format of the checkpoint
            remember to use super()._get_checkpoint_state() to get the base class state
            and add your own state to it

            Returns:
                a dictionary of states to be saved in checkpoint
        """
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state
    
    def _save_checkpoint(self, epoch: int, file_name: Path) -> None:
        """
        Save checkpoints. 
        This serves as an example implementation

        Args:
            epoch: current epoch number, 1-indexed
            log: logging information of the epoch
            save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = self._get_checkpoint_state(epoch)
            
        filename = self.checkpoint_dir / file_name
        torch.save(state, str(filename))
        self.logger.info("Saving checkpoint: {}".format(filename))

    def _load_checkpoint_state(self, checkpoint: dict) -> None:
        """
            Load the state from a checkpoint state, this is called in _resume_checkpoint
            you can overload this function if you want to load more stuff
            or change the format of the checkpoint
            remember to use super()._load_checkpoint_state() to load the base class state
        """
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']

        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            for key in list(checkpoint["state_dict"].keys()):
                new_key = key.replace('module.', '')
                checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(key)
            self.model.load_state_dict(checkpoint["state_dict"])

        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']

        if self.with_cuda:
            for checkpoint in self.optimizer.state.values():
                for k, v in checkpoint.items():
                    if isinstance(v, torch.Tensor):
                        checkpoint[k] = v.to(self.device)

    def _resume_checkpoint(self, resume_path: Path) -> None:
        """
        Resume from saved checkpoints. We may call this during __init__.
        This serves as an example. Should pair with _save_checkpoint if you are overloading that

        Args:
            resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)

        self._load_checkpoint_state(checkpoint)

        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def _load_model(self, resume_path: Path) -> None:
        """
            just load model, nothing more
            keeps everything else intact (e.g., n_epoch, scheduler, etc)
        """
        resume_path = str(resume_path)
        self.logger.info("Loading model: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            for key in list(checkpoint["state_dict"].keys()):
                new_key = key.replace('module.', '')
                checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(key)
            self.model.load_state_dict(checkpoint["state_dict"])

    @classmethod
    def load_model(cls, checkpoint_path: Path, model: nn.Module, map_location: torch.DeviceObjType) -> None:
        """
            load the model parameters in `checkpoint_path` to `model` 
        """
        checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            for key in list(checkpoint["state_dict"].keys()):
                new_key = key.replace('module.', '')
                checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(key)
            model.load_state_dict(checkpoint["state_dict"])

    @staticmethod
    def _initialize(config: dict, module: ModuleType, *args, **kwargs):
        """
            Initialize a class with config. Useful if you're using a config file to specify a class to use.

            e.g., if you wanna make / initialize an optimizer like:
                {
                    "type": "Adam",
                    "args": {
                        "lr": 0.0003,
                        "weight_decay": 0,
                        "amsgrad": true
                    }
                }
                you can do:
                self.optimizer = self._initialize(optimizer_config, torch.optim, model.parameters())
            
            Args:
                config: dict containing keys ['type', 'args'].
                    - type: str, the class we access the module for, i.e., the class is `module.type`
                    - args: dict, the kwargs given to initialize said class
                args, kwargs: additional arguments to initialize the class.
            Returns:
                the class that is constructed and initialized.

            NOTE:
                we use the module to do getattr(), so technically it doesn't have to be a module, and can
                be your self-defined type or dataclass.
                do note that we aren't accessing it like dict, the stuff you wanna access has to be an attribute.
        """
        module_name = config['type']
        module_args = config['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

# -------

from timeit import default_timer

class TaskOrientTrainer(BaseTrainer):
    """
        For semcom: multiple transmitter (user) use a common channel to send signal to one receiver.

        metric[epoch: int] = {          # epoch is one-indexed
            batch_train_loss: list[float],
            batch_val_loss: list[float], 
            avg_train_loss: float, 
            avg_val_loss: float,
            avg_train_acc: float
            avg_val_acc: float
            train_epoch_time: float,    # seconds
            val_epoch_time: float,
        }
    """
    def __init__(self, 
                 train_dataloader, val_dataloader, channel, acc_calculator: Callable, *args, 
                 accumulate_batch: int = 1,
                 model_saving_policy: Literal['only_best', 'every_min_val_loss', 'all'] = 'every_min_val_loss', **kwargs):
        """
            Args:
                train_dataloader, val_dataloader: for training & validation
                    expect the dataloaders to yield ()
                channel: should be a channel that supports D2D
                acc_calculator: should be a function to calculate the accuracy of model prediction
                accumulate_batch: will only do loss.backward() after this many batches
                                  if your dataloader's batch size is `bs`, the effective
                                  batch size will be `bs * accumulative_batch`
                                  will NOT loss.backward() at the last batch if
                                  it hasn't accumulated `accumulate_batch` amount of batch yet.
                model_saving_policy: decides when to save model
                    'only_best': only save the best (min val loss) model, i.e. only one model is saved
                    'every_min_val_loss': save every time there is an update to min val loss
                    'all': save all model from every epoch
            
            If only loading (and not training), the below things can be None:
                train_dataloader, val_dataloader, model_saving_policy
        """
        super().__init__(*args, **kwargs)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.channel = channel
        self.acc_calculator = acc_calculator
        self.accumulate_batch = accumulate_batch
        self.logger.info(f'channel = {self.channel}')
        self.logger.info(f'accumulate_batch = {accumulate_batch}')

        self.min_val_loss = float('inf')
        self.model_saving_policy = model_saving_policy
        self.logger.info(f'model_saving_policy = {self.model_saving_policy}')

    def transmit(self, inputs):
        """
            Does the transmission, i.e., start from transmitter(s) getting `inputs`,
            send to receiver(s) and inference task `results`

            targets and results should have the same shape, and will later calculate loss
            by `loss = self.criterion(targets, results)`

            The transmit process may be different for every paper / model / D2D, uplink, downlink / channel,
            try to deal with all of that here. All things mentioned above should be defined in the derived class
        """
        pass

    def _train_epoch(self, epoch: int) -> dict:
        self.logger.info(f'Train epoch {epoch}/{self.n_epochs}:')
        self.model.train()
        epoch_start = default_timer()

        train_loss_ls = []
        total_train_loss = 0.
        train_acc_ls = []
        total_train_acc  = 0.
        
        self.optimizer.zero_grad(set_to_none=True)

        n_batch = len(self.train_dataloader)
        progress_bar = tqdm(enumerate(self.train_dataloader), leave=False, desc='Train', total=n_batch, dynamic_ncols=True)
        for batch_i, (inputs, targets) in progress_bar:
            batch_start = default_timer()
            log_prefix = toColor(f'[train ep={epoch}/{self.n_epochs} batch={batch_i+1}/{n_batch}]', 'yellow')

            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            batch_size = targets.shape[0]
            
            # do prediction
            inputs = to_device(inputs, self.device)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.transmit(inputs)
            
            # if targets.max() >= 9:
            #     print(name)
            # calculate loss & back propagation
            loss = self.criterion(outputs, targets)
            # loss = loss * 8 ### Note
            loss.backward()
            train_loss_ls.append(loss.item())
            total_train_loss += loss.item()

            # calculate accuracy
            acc = self.acc_calculator(outputs, targets)
            train_acc_ls.append(float(acc))
            total_train_acc  += float(acc)

            if (batch_i + 1) % self.accumulate_batch == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                
            # do record
            if (batch_i + 1) % self.display_interval == 0:
                batch_time = default_timer() - batch_start
                self.logger.info(
                    f'{log_prefix} global_step: {self.global_step}, '
                    f'lr: {lr:.6}, batch_time: {batch_time:.2f} sec, batch loss = {loss.item()}'
                )
                self.logger.info(
                    f'{log_prefix} estimated remaining time of train epoch: {datetime.timedelta(seconds=(n_batch - batch_i - 1) * batch_time)}'
                )

            if isinstance(loss, torch.Tensor) and torch.any(torch.isnan(loss)).item():
                self.logger.info('[x] NaN happened!')
                exit()

        self.optimizer.zero_grad(set_to_none=True)
        epoch_time = default_timer() - epoch_start
        log_prefix = toColor(f'[train ep={epoch}/{self.n_epochs}]', 'yellow')
        self.logger.info(f'{log_prefix} Epoch time: {datetime.timedelta(seconds=epoch_time)}')

        train_result = {
            'epoch': epoch,
            'train_loss': total_train_loss / len(train_loss_ls), 
            'train_acc': total_train_acc / len(train_acc_ls),
            'lr': lr, 
            'time': epoch_time,
        }
        self.logger.info(f'{log_prefix} train_result = {train_result}')

        self.metrics[epoch] = {
            'batch_train_loss': train_loss_ls,
            'batch_train_acc': train_acc_ls,
            'avg_train_loss': total_train_loss / n_batch,
            'avg_train_acc': total_train_acc / n_batch,
            'train_epoch_time': epoch_time,
        }

        return train_result

    def _eval(self, epoch: int) -> None:
        self.logger.info(f'Val epoch {epoch}/{self.n_epochs}:')
        self.model.eval()

        epoch_start = default_timer()
        val_loss_ls = []
        self.val_loss = 0.
        val_acc_ls = []
        self.val_acc  = 0.

        batch_len = len(self.val_dataloader)
        progress_bar = tqdm(enumerate(self.val_dataloader), leave=False, desc='Validation', total=batch_len, dynamic_ncols=True)
        with torch.no_grad():
            for batch_i, (inputs, targets) in progress_bar:
                batch_start = default_timer()
                log_prefix = toColor(f'[val ep={epoch}/{self.n_epochs} batch={batch_i+1}/{batch_len}]', 'cyan')

                batch_size = targets.shape[0]
            
                # do prediction
                inputs = to_device(inputs, self.device)
                targets = targets.to(self.device)
                outputs = self.transmit(inputs)
            
                loss = self.criterion(outputs, targets)
                val_loss_ls.append(loss.item())
                self.val_loss += loss.item()

                acc = self.acc_calculator(outputs, targets)
                val_acc_ls.append(float(acc))
                self.val_acc  += float(acc)

                # do record
                if (batch_i + 1) % self.display_interval == 0:
                    batch_time = default_timer() - batch_start
                    self.logger.info(
                        f'{log_prefix} global_step: {self.global_step}, '
                        f', batch_time: {batch_time:.2f} sec, batch loss = {loss.item()}'
                    )
                    self.logger.info(
                        f'{log_prefix} estimated remaining time of val epoch: {datetime.timedelta(seconds=(batch_len - batch_i - 1) * batch_time)}'
                    )

                if isinstance(loss, torch.Tensor) and torch.any(torch.isnan(loss)).item():
                    self.logger.info('[x] NaN happened!')
                    exit()
        
        self.val_loss /= batch_len
        epoch_time = default_timer() - epoch_start
        log_prefix = toColor(f'[val ep={epoch}/{self.n_epochs}]', 'cyan')
        self.logger.info(f'{log_prefix} Epoch time: {datetime.timedelta(seconds=epoch_time)}, test_loss = {self.val_loss}, test_acc = {self.val_acc}')

        self.metrics[epoch].update({
            'batch_val_loss': val_loss_ls,
            'batch_val_acc': val_acc_ls,
            'avg_val_loss': self.val_loss,
            'avg_val_acc': self.val_acc,
            'val_epoch_time': epoch_time,
        })
            
    def _on_epoch_finish(self, epoch: int) -> None:
        """
            main logic for when an epoch finishes, will do after _eval()
            you can put your store model logic and step epoch scheduler here.
            you have access to self.epoch_result and things you stored during _eval()

            Args:
                epoch: current epoch number, 1-indexed
        """
        is_val_loss_updated = (self.val_loss < self.min_val_loss)
        self.min_val_loss = min(self.val_loss, self.min_val_loss)

        # store latest epoch in any circumstances
        self._save_checkpoint(epoch, 'model_latest.pth')
            
        model_filename = f'model_ep{epoch}_{self.val_loss:.4f}.pth'
        if self.model_saving_policy == 'only_best':
            model_filename = f'model_best.pth'

        if is_val_loss_updated:
            self._save_checkpoint(epoch, model_filename)
        elif self.model_saving_policy == 'all':
            self._save_checkpoint(epoch, model_filename)
    
    def _on_train_finish(self):
        self.logger.info('train finished.')

    # def _log_memory_usage(self) -> None:
    #     pass

    # def _save_checkpoint(self, epoch: int, file_name: Path):
    #     pass

    # def _resume_checkpoint(self, resume_path):
    #     pass
    
def get_best_checkpoint(folder: Path, before_epoch: int=None, use_latest: bool=True) -> Path:
    """
    get the biggest model_ep{number}_* in the folder
    Args:
        folder: the folder to search for checkpoints,
            if you use BaseTrainer, this should be named something like `checkpoint/`
        before_epoch: if given, only return checkpoints before this epoch (inclusive)
        use_latest: whether to use the latest checkpoint
            if True, return model_latest.pth if it exists, otherwise raise an exception
            (also ignores before_epoch if True)
            if False, return the checkpoint with the largest epoch number checkpoint
            (that is less than or equal to before_epoch if given)
    Returns:
        the path to the checkpoint file, which is the one with the largest epoch number
        (or model_latest.pth if use_latest is True)

    Note:
        if you use option 'every_min_val_loss' in BaseTrainer, 
        then get_best_checkpoint(folder, None, False) will return the checkpoint with the smallest validation loss
    """
    # if use_latest:
    #     raise Exception('YOU USED USE_LATEST!!!!!')

    if use_latest:
        latest = folder / 'model_latest.pth'
        if latest.exists():
            return latest
        raise Exception(f'Folder {folder} does not have model_latest.pth')
    
    import re
    pattern = re.compile('model_ep(\d+)_*')
    
    paths = {}
    for p in folder.glob('*'):
        m = re.match(pattern, p.name)
        if m is None:
            continue
        paths[int(m.group(1))] = p
    if len(paths) == 0:
        raise Exception(f'Folder {folder} does not have any viable model')
    if before_epoch is not None:
        paths = {ep: path for ep, path in paths.items() if ep <= before_epoch}
    ep, path = max(paths.items(), key=lambda a: a[0])
    return path