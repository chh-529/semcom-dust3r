import argparse
import torch


import torch.nn.functional as F
import datetime
from pathlib import Path
import itertools
import logging
from ..log import get_logger
from .. import utils
from ..util.util import *
from ..trainer.trainer import BaseTrainer

from ..model.cif import ChannelFusion, MFNet
from torch.optim.lr_scheduler import ConstantLR, StepLR
from ..dataset import cif as chfusion_dataset
from ..channel import AWGNMultiUplinkChannel, RayleighFadingMultiD2DChannel, RayleighFadingMultiUplinkChannel, AWGNSingleChannel
from ..trainer.trainer_CIF import ChannelFusionTrainer, MFNetTrainer
import pickle
torch.autograd.set_detect_anomaly(True)

def parse_chfusion_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        help="The name of the model, will store the model at ./checkpoint/{model_name}. "
                             "The model will be stored at its checkpoint directory (e.g., ./checkpoint/{model_name}/test_1/checkpoint). ",
                        required=True)
    parser.add_argument('--gpu', 
                        help="The GPU used. Input integer list like \"0\" or \"0,1,2\". Default \"0\".",
                        type=str,
                        default="0")
    parser.add_argument('--interfere_mode', 
                        help="The interference mode used for the channel. 'all' for superimpose, 'none' for individual (i.e., separated training). "
                             "Default 'all'",
                        choices=['all', 'self'],
                        default='all')
    parser.add_argument('--seed', 
                        help="The seed used. If not given, will still randomly pick a seed for the trainer.",
                        type=int,
                        default=None)
    parser.add_argument('--do_torch_normalize', 
                        help="Whether to do transform.Normalize(0.5, 0.5) on the dataset. If not set, will only use ToTensor() to turn the range into (0, 1)",
                        action="store_true")
    parser.add_argument('--fading_mode', 
                        help="The fading mode of the channel, 'fast' or 'slow'. Default 'slow'.",
                        type=str,
                        default='slow')
    
    args = parser.parse_args()
    
    return args

def get_chfusion_trainer(
        logger: logging.Logger = None,
        save_dir: Path = Path('./checkpoint'),
        resume_checkpoint: Path = None,
        gpus: list[int] = [0],
        seed: int = 0,
        display_interval: int = 10,
        trainer_class: type[BaseTrainer]=ChannelFusionTrainer,
        num_workers: int=4,
        interfere_mode: str='all',
        do_torch_normalize: bool=False,
        fading_mode: str='slow',
        save_arguments: bool=True
    ) -> BaseTrainer:

    # log the arguments
    saved_args = {**locals()}
    print(f'{saved_args = }')

    time_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

    if save_arguments:
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(saved_args, open(save_dir / f'args.pkl', 'wb'))

    # basic setting
    args = {
        'batch_size': 16,
        'n_user': 2,
        'n_epoch': 150,
        'lr_start': 0.02,
        'lr_decay': 0.99,   
    }
    n_t = 2
    n_r = 4

    data_dir = './data/MF'
    channel_gain_var = 1.0
    channel_gain_var_ls = [[1], [1]]
    snr_ls = [1] * args['n_user']
    power_constraint = [2] * args['n_user']
    n_class = 9
    

    if logger is None:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(save_dir) / f'channel_fusion_{time_str}.ansi'
        logger = get_logger('channel_fusion', str(log_path), stdout=True)

    logger.info('channel_fusion')
    logger.info(f'argument for trainer function: {saved_args}')
    
    # load dataset
    train_dataloader, val_dataloader = chfusion_dataset.make_channelfusion_MF_dataloaders(
        batch_size=args['batch_size'], root=data_dir)

    # make channel
    # channel = AWGNMultiUplinkChannel(
    #         args['n_user'],
    #         snr_db = [0.0],
    #         interfere_mode='all'
    #     )
    
    channel = RayleighFadingMultiUplinkChannel(
            args['n_user'], 
            snr_db=[0], 
            divide_gain=False, 
            channel_gain_var=channel_gain_var_ls, 
            interfere_mode='all', 
            fading_mode='slow',
        )

    # make model stuff
    model = ChannelFusion(in_h=480, in_w=640, n_t=n_t, n_r=n_r, channel_gain_var=channel_gain_var, n_class=n_class)
    criterion = torch.nn.CrossEntropyLoss()
    acc_function = calculate_accuracy
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr_start'], momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = StepLR(optimizer, step_size=1, gamma=args['lr_decay'])
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=100000)

    # make trainer
    trainer = trainer_class(
        logger=logger,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        display_interval=display_interval,
        n_epoch=args['n_epoch'],
        gpus=gpus,
        seed=seed,
        resume_checkpoint=resume_checkpoint,
        weights_init=None,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        channel=channel,
        acc_calculator=acc_function,
        model_saving_policy='every_min_val_loss',
        power_constraint=power_constraint,
        n_t=n_t,
        n_r=n_r
    )

    return trainer

def get_MFNet_trainer(
        logger: logging.Logger = None,
        save_dir: Path = Path('./checkpoint'),
        resume_checkpoint: Path = None,
        gpus: list[int] = [0],
        seed: int = 0,
        display_interval: int = 10,
        trainer_class: type[BaseTrainer]=MFNetTrainer,
        num_workers: int=4,
        interfere_mode: str='all',
        do_torch_normalize: bool=False,
        fading_mode: str='slow',
        save_arguments: bool=True
    ) -> BaseTrainer:

    # log the arguments
    saved_args = {**locals()}
    print(f'{saved_args = }')

    time_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

    if save_arguments:
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(saved_args, open(save_dir / f'args.pkl', 'wb'))

    # basic setting
    args = {
        'batch_size': 16,
        'n_epoch': 100,
        'lr_start': 0.01,
        'lr_decay': 0.94,   
    }

    data_dir = './data/MF'
    n_class = 9
    

    if logger is None:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(save_dir) / f'mfnet_{time_str}.ansi'
        logger = get_logger('mfnet', str(log_path), stdout=True)

    logger.info('mfnet')
    logger.info(f'argument for trainer function: {saved_args}')
    
    # load dataset
    train_dataloader, val_dataloader = chfusion_dataset.make_channelfusion_MF_dataloaders(
        batch_size=args['batch_size'], root=data_dir)

    # make channel
    channel = AWGNSingleChannel(snr_db=0)

    # make model stuff
    model = MFNet(n_class=n_class)
    criterion = torch.nn.CrossEntropyLoss()
    acc_function = calculate_accuracy
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr_start'], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=100000) # keep it constant for now

    # make trainer
    trainer = trainer_class(
        logger=logger,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        display_interval=display_interval,
        n_epoch=args['n_epoch'],
        gpus=gpus,
        seed=seed,
        resume_checkpoint=resume_checkpoint,
        weights_init=None,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        channel=channel,
        acc_calculator=acc_function,
        model_saving_policy='every_min_val_loss',
    )

    return trainer

def get_best_checkpoint(folder: Path, before_epoch: int=None) -> Path:
    """
        get the biggest model_ep{number}_* in the folder
    """
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


if __name__ == '__main__':
    args = parse_chfusion_args()
    print(args.gpu)

    if args.seed is None:
        import random
        args.seed = random.randint(1, 1000000)
    
    args.gpu = [int(g) for g in args.gpu.split(',')]
    
    trainer = get_chfusion_trainer(
        save_dir=Path(f'./checkpoint/{args.model_name}/'),
        seed=args.seed,
        gpus=args.gpu,
        interfere_mode=args.interfere_mode,
        do_torch_normalize=args.do_torch_normalize,
        fading_mode=args.fading_mode
    )
    # trainer = get_MFNet_trainer(
    #     save_dir=Path(f'./checkpoint/{args.model_name}/'),
    #     seed=args.seed,
    #     gpus=args.gpu,
    #     interfere_mode=args.interfere_mode,
    #     do_torch_normalize=args.do_torch_normalize,
    #     fading_mode=args.fading_mode
    # )
    trainer.train()