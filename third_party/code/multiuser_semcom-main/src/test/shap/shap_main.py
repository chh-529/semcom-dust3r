import torch
import numpy as np
from pathlib import Path
from typing import *
from tqdm import tqdm
import random
import argparse
import matplotlib.pyplot as plt

from ...log import get_logger
from ... import utils

from ...channel import *
from ..udeepsc import cp_load as msa_cpld
from ..ave import cp_load as ave_cpld

from ..ave import model_test as ave_test
from ..udeepsc import model_test as msa_test

from ...channel import *
from ...utils import to_device, pad_tensor_batch
import shap

def parse_shap_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path',
                        help='Path to model checkpoint',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('-m','--method', 
                        help="The position of the FSM put, 'features' for putting after semantic encoder, 'signal' for putting after channel encoder",
                        type=str,
                        choices=['features', 'signals'],
                        default='features')
    parser.add_argument('-t','--task', 
                        help="Target task to perform shap analysis",
                        type=str, choices=['msa', 'ave'],
                        default='msa')
    parser.add_argument('--save_dir',
                        help="directory to save shap value results",
                        type=str,
                        default=None,
                        required=True)
    
    args = parser.parse_args()
    
    return args
        
def main_test_SHAP_KernelExplainer(args):
    logger = get_logger('NoSIC_shap_test', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    
    ckpt_path = Path(args.ckpt_path)
    
    n_tests = 3000
    method = args.method
    task = args.task
    is_JSCC = method == 'signals'
    
    model_name = ckpt_path.name
    result_main_folder = Path(args.save_dir)
    result_main_folder.mkdir(parents=True, exist_ok=True)

    result_path = result_main_folder / f'{task}' / f'kernel_shap_{n_tests}_{method}' / model_name
    
    if task == 'msa':
        trainer = msa_cpld.get_trainer_from_checkpoint(
            logger=logger,
            checkpoint_path=ckpt_path / 'checkpoint',
            args_path=ckpt_path / 'args.pkl',
            gpus=[1]
        )
        msa_test.test_SHAP_KernelExplainer_Masker(trainer, result_dir = result_path, save_result=True, n_test=n_tests, is_JSCC=is_JSCC, bg_size=10)
    elif task == 'ave':
        trainer = ave_cpld.get_trainer_from_checkpoint(
            logger=logger,
            checkpoint_path=ckpt_path / 'checkpoint',
            args_path=ckpt_path / 'args.pkl',
            gpus=[1]
        )
        ave_test.test_SHAP_KernelExplainer_Masker(trainer, result_dir = result_path, save_result=True, n_test=n_tests, is_JSCC=is_JSCC, bg_size=10)
    
if __name__ == '__main__':
    args = parse_shap_args()
    main_test_SHAP_KernelExplainer(args)