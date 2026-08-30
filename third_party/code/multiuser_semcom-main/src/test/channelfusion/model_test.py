import torch
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import *
from PIL import Image
from tqdm import tqdm
import random
import pickle

from ...log import get_logger
from ... import utils

from ...channel import *
from .cp_load import *

from ...trainer.trainer_CIF import ChannelFusionTrainer
from ...dataset.cif import MF_dataset, make_channelfusion_MF_testdataloader

from ...model.cif import ChannelFusion
from ...channel import *
from ...util.util import calculate_accuracy, calculate_result, visualize

def test_dataset(trainer, test_loader, n_class, n_batch, result_folder: Path, save_predict: bool = False, save_per_nbatch: int = 10):
    """
        Args:
            trainer: the trainer object
            test_loader: the dataloader to test on, n_user should be the same as trainer
            n_batch: number of batch to test 
                     (so the total signal count will be n_batch * n_user)
            result_folder: the folder to store the result
    """
    result_folder.mkdir(parents=True, exist_ok=True)
    if save_predict:
        (result_folder / 'images').mkdir(parents=True, exist_ok=True)

    with open(result_folder / 'trainer.txt', 'w') as fout:
        trainer_var = vars(trainer)
        if 'metrics' in trainer_var: 
            del trainer_var['metrics']
        print(utils.str_type(trainer_var, indent=4, array_limit_items=20), file=fout)

    cf = np.zeros((n_class, n_class))

    trainer.model.eval()
    total_acc  = 0
    with torch.no_grad():
        for it, (inputs, targets, names) in tqdm(zip(range(n_batch), test_loader)):
            fpath = []
            # print(str_type(names))
            for n in names:
                # print(n)
                fpath.append(result_folder / ('images/' + n + '_pred.png'))

            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)
            results = trainer.transmit(inputs) # (batch_size, n_class, h, w)

            targets, results = targets.to('cpu'), results.to('cpu')

            acc = calculate_accuracy(results, targets)
            total_acc += float(acc)

            trainer.logger.info('test iter %s/%s.  acc: %.4f' \
                    % (it + 1, n_batch, float(acc)))
            
            predictions = results.argmax(1)
            for gtcid in range(n_class): 
                for pcid in range(n_class):
                    gt_mask      = targets == gtcid 
                    pred_mask    = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())

            if save_predict:
                if (it + 1) % save_per_nbatch == 0:
                    visualize(fpath, predictions)
    
    overall_acc, acc, IoU = calculate_result(cf)

    stat = {
        'overall accuracy': overall_acc,
        'accuracy of each class': acc,
        'class accuracy avg': acc.mean(),
        'IoU': IoU,
        'class IoU avg': IoU.mean()
    }

    with open(result_folder / 'metric.txt', 'w') as fout:
        for name in stat.keys():
            print(f"{name}:{stat[name]}", file=fout)

    

def main_test_20250306():
    logger = get_logger('ch_fusion_test', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('/home/ldap/hansliu/semcom/checkpoint/20250522_chfusion_test/checkpoint'),
        args_path=Path('/home/ldap/hansliu/semcom/checkpoint/20250522_chfusion_test/args.pkl'),
        gpus=[0]
    )

    args = pickle.load(open(Path('/home/ldap/hansliu/semcom/checkpoint/20250522_chfusion_test/args.pkl'), 'rb'))

    # data_dir = './data/MF'
    n_class = 9
    test_dataloader = make_channelfusion_MF_testdataloader(batch_size=10)
    test_dataloader.n_iter = len(test_dataloader)

    test_dataset(
        trainer, test_dataloader, n_class, n_batch = len(test_dataloader), result_folder = Path('./tmp/20250522_chfusion_test'), save_predict = True, save_per_nbatch = 15
    )

def main_mfnet_test_20250323():
    logger = get_logger('mfnet_test', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_MFNet_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('/home/ldap/hansliu/semcom/checkpoint/20250323_mfnet_test/checkpoint'),
        args_path=Path('/home/ldap/hansliu/semcom/checkpoint/20250323_mfnet_test/args.pkl'),
        gpus=[0]
    )

    args = pickle.load(open(Path('/home/ldap/hansliu/semcom/checkpoint/20250323_mfnet_test/args.pkl'), 'rb'))

    # data_dir = './data/MF'
    n_class = 9
    test_dataloader = make_channelfusion_MF_testdataloader(batch_size=10)
    test_dataloader.n_iter = len(test_dataloader)

    test_dataset(
        trainer, test_dataloader, n_class, n_batch = len(test_dataloader), result_folder = Path('./tmp/20250323_mfnet_test'), save_predict = True
    )

def main_test_20250316_different_SNR():
    logger = get_logger('ch_fusion_test', log_file_path=None, stdout=True, stdout_tqdm_write=True)
    trainer = get_trainer_from_checkpoint(
        logger=logger,
        checkpoint_path=Path('/home/ldap/hansliu/semcom/checkpoint/20250316_chfusion_test_SNR20/checkpoint'),
        args_path=Path('/home/ldap/hansliu/semcom/checkpoint/20250316_chfusion_test_SNR20/args.pkl'),
        gpus=[0]
    )

    args = pickle.load(open(Path('/home/ldap/hansliu/semcom/checkpoint/20250316_chfusion_test_SNR20/args.pkl'), 'rb'))

    # data_dir = './data/MF'
    n_class = 9
    test_dataloader = make_channelfusion_MF_testdataloader(batch_size=2)
    test_dataloader.n_iter = len(test_dataloader)

    rayleigh_channel = RayleighFadingMultiUplinkChannel(
            n_user=2, 
            snr_db=[10], 
            divide_gain=False, 
            interfere_mode='all', 
            fading_mode='slow'
        )

    trainer.channel = rayleigh_channel
    test_dataset(
        trainer, test_dataloader, n_class, n_batch = 5, result_folder = Path('./tmp/20250316_chfusion_test2'), save_predict = True
    )

if __name__ == '__main__':
    main_test_20250306()
    # main_mfnet_test_20250323()
    # main_test_20250316_different_SNR()