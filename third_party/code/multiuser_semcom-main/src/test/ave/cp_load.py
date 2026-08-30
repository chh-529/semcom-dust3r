from pathlib import Path

from ... import utils
from ...train.train_UDeepSC import get_best_checkpoint, get_udeepsc_ave_trainer, get_udeepscoma_ave_trainer
from ...trainer.trainer import BaseTrainer 
from ...trainer.trainer_udeepsc import UDeepSCNoSICTrainer_Msa

from pathlib import Path
from typing import *
import pickle

def get_trainer_from_checkpoint(logger, checkpoint_path: Path, args_path: Path, before_epoch: int=None, isOMA: bool = False, **kwargs) -> BaseTrainer:
    """
        load the checkpoint and get the trainer 
        the parameters is fetched from checkpoint_path

        logger: the logger for trainer
        checkpoint_path: the .pth file for the checkpoint.
                         if the path is a directory, will use get_best_checkpoint() to get the best checkpoint instead
        args_path: the argument path for the trainer
                   if the path is a directory, will use `args_path / 'args.pkl'` instead
                   will override its resume_checkpoint for checkpoint_path and anything else in **kwargs
        before_epoch: get the best checkpoint before this epoch
        **kwargs: the parameters for the trainer to override the argument with
    """
    if checkpoint_path.is_dir():
        checkpoint_path = get_best_checkpoint(checkpoint_path, before_epoch)
    
    trainer_args = pickle.load(open(args_path, 'rb'))
    trainer_args.update(kwargs)

    trainer_args['resume_checkpoint'] = checkpoint_path
    trainer_args['logger'] = logger
    trainer_args['save_arguments'] = False
    
    if isOMA:
        return get_udeepscoma_ave_trainer(**trainer_args)
    
    return get_udeepsc_ave_trainer(**trainer_args)