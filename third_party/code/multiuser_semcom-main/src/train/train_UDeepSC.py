import argparse
import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR, LinearLR

import datetime
from pathlib import Path
import itertools
import logging
from ..log import get_logger
from .. import utils
from ..trainer.trainer import BaseTrainer
from functools import partial
import pickle
import json
import os

from .optim_factory import create_optimizer
from ..model.udeepsc import (
    UDeepSCNoSIC_msa, UDeepSCNoSIC_ave, UDeepSCNoSIC, UDeepSCOMA_msa, UDeepSCSIC_msa, UDeepSCOMA_ave, UDeepSCSIC_ave
)
from ..dataset.udeepsc import make_udeepsc_msa_dataloader
from ..dataset.ave import make_AVE_dataloaders
from ..channel import AWGNMultiUplinkChannel, RayleighFadingMultiD2DChannel, RayleighFadingMultiUplinkChannel, AWGNSingleChannel
from ..trainer.trainer_udeepsc import *

torch.autograd.set_detect_anomaly(True)

today = datetime.date.today().strftime("%Y%m%d")

def parse_udeepsc_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        help="The name of the model, will store the model at ./checkpoint/{model_name}. "
                             "The model will be stored at its checkpoint directory (e.g., ./checkpoint/{model_name}/test_1/checkpoint). ",
                        required=True)
    parser.add_argument('--gpu', 
                        help="The GPU used. Input integer list like \"0\" or \"0,1,2\". Default \"0\".",
                        type=str,
                        default="0")
    parser.add_argument('--seed', 
                        help="The seed used. If not given, will still randomly pick a seed for the trainer.",
                        type=int,
                        default=None)
    parser.add_argument('--task',
                        help="The task to be perform. Support msa and ave so far",
                        type=str,
                        choices=['msa', 'ave'],
                        default='msa'
                        )
    channel_group = parser.add_argument_group(
        title='Channel hyperparameters',
        description='Channel related hyperparameters'
    )
    channel_group.add_argument('--interfere_mode', 
                            help="The interference mode used for the channel. 'all' for superimpose, 'self' for orthogonal multiple access (i.e., TDMA). "
                                    "Default 'all'",
                            choices=['all', 'self'],
                            default='all')
    channel_group.add_argument('--channel_type', 
                               help="The channel type. 'rayleigh' or 'awgn'. Default 'awgn'.",
                               type=str,
                               default='awgn')
    channel_group.add_argument('--fading_mode', 
                               help="The fading mode of the channel, 'fast' or 'slow'. Default 'slow'.",
                               type=str,
                               default='slow')
    channel_group.add_argument('--power', 
                               help="The power of the AP. Default 1.",
                               type=float,
                               default=1)
    channel_group.add_argument('--snr_db', 
                               help="The SNR. You can type e.g., '10' to fix the SNR to 10dB, or '5,10' to randomly pick a SNR between 5 and 10dB. Default '5,10'.",
                               type=str,
                               default='5,10')
    model_group = parser.add_argument_group(
        title='Model hyperparameters',
        description='Model related hyperparameters')
    model_group.add_argument('--num_symbols', 
                                help="transmit number of symbols. Default 32",
                                type=int,
                                default=None)
    
    args = parser.parse_args()
    
    return args

def _get_model_name(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id=None) -> str:
    """
        a unified, standard model name... at least for our test results
        used basically anywhere from storing training checkpoints to plotting results
        these information in the name should be enough to identify a unique model

        Args:
            channel_type: the channel type in which it is trained in, e.g., 'awgn', 'rayleigh'
            snr_db: the channel SNR in which this model is trained in. Unit is dB, e.g., 0, 5, 10, 15, 20
            model_type: the model architecture type, e.g., 'udeepsc', 'cif', etc.
            encoder_out_dims: the number of output channels for the encoder
                this is also called e.g., encoder_output_dim, etc.
                this should be architecture dependent, but overall, this number is twice the number of signal symbols, e.g., 2 * num_symbols
            dataset_type: the dataset type this model is trained in, e.g., 'cmu-mosei', 'ave', 'mf', etc.
            model_id: the model id, if there are multiple models trained for the same setting,
                this is used to distinguish them, 1-indexed. e.g., 1, 2, etc.
        Returns:
            a string representing the model name,
            e.g., 'awgn_0_udeepsc_msa_symbols_32_cmu-mosei' (if only specifying the model settings)
            or 'awgn_0_udeepsc_msa_symbols_32_cmu-mosei_1' (if also specifying the model_id)
        """
    name = f'{channel_type}_{snr_db}_{model_type}_symbols_{encoder_out_dims // 2}_{dataset_type}'
    if model_id is not None:
        name += f'_{model_id}'
    return name

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

def get_udeepsc_msa_trainer(
        logger: logging.Logger = None,
        save_dir: Path = Path('./checkpoint'),
        resume_checkpoint: Path = None,
        gpus: list[int] = [0],
        seed: int = 0,
        display_interval: int = 10,
        trainer_class: type[BaseTrainer]=UDeepSCNoSICTrainer_Msa,
        model_class: type[UDeepSCNoSIC] =UDeepSCNoSIC_msa,
        num_workers: int=4,
        channel_type: Literal['awgn', 'rayleigh'] = 'rayleigh',
        interfere_mode: str='all',
        fading_mode: str='slow',
        snr_db: float=12.0,
        num_symbols: int=32,
        batch_size: int=50,
        n_epoch: int=150,
        power_constraint: list[float] = [1, 1, 1],
        divide_gain: bool=False,
        save_arguments: bool=True,
        other_kwargs: dict={}
    ) -> BaseTrainer:
    
    # log the arguments
    saved_args = {**locals()}
    print(f'{saved_args = }')

    time_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

    if save_arguments:
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(saved_args, open(save_dir / f'args.pkl', 'wb'))

    
    if logger is None:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(save_dir) / f'UdeepSC_{time_str}.ansi'
        logger = get_logger('UdeepSC', str(log_path), stdout=True)

    logger.info('UdeepSC')
    logger.info(f'argument for trainer function: {saved_args}')

    # basic setting
    args = {
        'train_samples': 13052,
        'warmup_ep': 1,
        'n_user': 3,
        'ta_perform': 'msa', 
    }

    opt_args_dict = {
        'opt': 'adamw',
        'lr': 3e-5,
        'weight_decay': 0.02,
        'opt_eps': 1e-8,
        'opt_betas': [0.95, 0.99],
    }
    # opt_args = argparse.Namespace(**opt_args_dict)
    logger.info(f"optimizer settings: {opt_args_dict}")

    data_dir = 'data/msadata'
    channel_gain_var = 1.0
    channel_gain_var_ls = [[1], [1], [1]]
    snr_ls = [1]
    # power_constraint = [1] * args['n_user']

    # load dataset
    train_dataloader, val_dataloader = make_udeepsc_msa_dataloader(
        batch_size=batch_size,
        root=data_dir, 
        num_workers=num_workers,
        pin_memory = True,
    )
    
    # make channel
    if channel_type == 'rayleigh':
        channel = RayleighFadingMultiUplinkChannel(
            args['n_user'], 
            snr_db=[snr_db], 
            divide_gain=divide_gain, 
            noise_power_density_dBm=-90,    # ref. ISSNOMATrainer's note
            reference_distance=1,
            reference_path_loss=pow(10, -30/10),
            path_loss_exponent=4,
            distance=torch.Tensor([33, 83, 133]).reshape(3, 1),
            channel_gain_var=channel_gain_var_ls, 
            interfere_mode=interfere_mode, 
            fading_mode=fading_mode,
        )
    elif channel_type == 'awgn':
        channel = AWGNMultiUplinkChannel(
                args['n_user'],
                snr_db = [snr_db],
                interfere_mode=interfere_mode
            )
    
    

    # make model stuff
    model = model_class(
        num_symbols=num_symbols,
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )

    criterion = torch.nn.MSELoss()
    acc_function = utils.calc_metrics
    # optimizer = torch.optim.AdamW(model.parameters(), **opt_args)
    optimizer = create_optimizer(opt_args_dict, model)

    num_iters_per_ep = len(train_dataloader)
    warmup_iters = num_iters_per_ep * args['warmup_ep']
    total_iters = n_epoch * num_iters_per_ep - warmup_iters
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5)
    
    lr_scheduler = SequentialLR(
        optimizer, schedulers=[
            LinearLR(               # warmup: from start_factor to lr
                optimizer,
                start_factor=1e-10 / opt_args_dict['lr'],  # cannot be 0
                end_factor=1,
                total_iters=warmup_iters
            ),
            CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5) # main: from lr to eta_min
        ], milestones=[warmup_iters])

    # make trainer
    trainer = trainer_class(
        logger=logger,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        display_interval=display_interval,
        n_epoch=n_epoch,
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
        ta_perform=args['ta_perform'],
        **other_kwargs,
    )

    return trainer

def get_udeepscoma_msa_trainer(
        logger: logging.Logger = None,
        save_dir: Path = Path('./checkpoint'),
        resume_checkpoint: Path = None,
        gpus: list[int] = [0],
        seed: int = 0,
        display_interval: int = 10,
        trainer_class: type[BaseTrainer]=UDeepSCOMATrainer_Msa,
        model_class: type[UDeepSCNoSIC]=UDeepSCOMA_msa,
        num_workers: int=4,
        channel_type: Literal['awgn', 'rayleigh'] = 'rayleigh',
        interfere_mode: str='all',
        fading_mode: str='slow',
        snr_db: float=12.0,
        num_symbols: int=32,
        batch_size: int=50,
        n_epoch: int=150,
        power_constraint: list[float] = [1, 1, 1],
        save_arguments: bool=True,
    ) -> BaseTrainer:
    
    # log the arguments
    saved_args = {**locals()}
    print(f'{saved_args = }')

    time_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

    if save_arguments:
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(saved_args, open(save_dir / f'args.pkl', 'wb'))

    
    if logger is None:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(save_dir) / f'UdeepSCOMA_{time_str}.ansi'
        logger = get_logger('UdeepSCOMA', str(log_path), stdout=True)

    logger.info('UdeepSCOMA')
    logger.info(f'argument for trainer function: {saved_args}')

    # basic setting
    args = {
        'train_samples': 13052,
        'warmup_ep': 1,
        'n_user': 3,
        'ta_perform': 'msa', 
    }

    opt_args_dict = {
        'opt': 'adamw',
        'lr': 3e-5,
        'weight_decay': 0.02,
        'opt_eps': 1e-8,
        'opt_betas': [0.95, 0.99],
    }
    # opt_args = argparse.Namespace(**opt_args_dict)
    logger.info(f"optimizer settings: {opt_args_dict}")

    data_dir = 'data/msadata'
    channel_gain_var = 1.0
    power_constraint = [1] * args['n_user']

    # load dataset
    train_dataloader, val_dataloader = make_udeepsc_msa_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        root=data_dir, 
        pin_memory = True,
    )
    
    # make channel
    if channel_type == 'rayleigh':
        channel = RayleighFadingSingleChannel(
            snr_db=snr_db, 
            divide_gain=True, 
            channel_gain_var=channel_gain_var, 
            fading_mode=fading_mode,
        )
    elif channel_type == 'awgn':
        channel = AWGNSingleChannel(
                snr_db = snr_db,
            )
    
    

    # make model stuff
    model = model_class(
        num_symbols=num_symbols,
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )

    criterion = torch.nn.MSELoss()
    acc_function = utils.calc_metrics
    # optimizer = torch.optim.AdamW(model.parameters(), **opt_args)
    optimizer = create_optimizer(opt_args_dict, model)

    num_iters_per_ep = len(train_dataloader)
    warmup_iters = num_iters_per_ep * args['warmup_ep']
    total_iters = n_epoch * num_iters_per_ep - warmup_iters
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5)
    
    lr_scheduler = SequentialLR(
        optimizer, schedulers=[
            LinearLR(               # warmup: from start_factor to lr
                optimizer,
                start_factor=1e-10 / opt_args_dict['lr'],  # cannot be 0
                end_factor=1,
                total_iters=warmup_iters
            ),
            CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5) # main: from lr to eta_min
        ], milestones=[warmup_iters])

    # make trainer
    trainer = trainer_class(
        logger=logger,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        display_interval=display_interval,
        n_epoch=n_epoch,
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
        ta_perform=args['ta_perform'],
    )

    return trainer

def get_udeepsc_ave_trainer(
        logger: logging.Logger = None,
        save_dir: Path = Path('./checkpoint'),
        resume_checkpoint: Path = None,
        gpus: list[int] = [0],
        seed: int = 0,
        display_interval: int = 10,
        trainer_class: type[BaseTrainer]=UDeepSCNoSICTrainer_Msa,
        model_class: type[UDeepSCNoSIC] =UDeepSCNoSIC_msa,
        num_workers: int=4,
        channel_type: Literal['awgn', 'rayleigh'] = 'rayleigh',
        interfere_mode: str='all',
        fading_mode: str='slow',
        snr_db: float=12.0,
        num_symbols: int=32,
        batch_size: int=50,
        n_epoch: int=150,
        power_constraint: list[float] = [1, 1],
        save_arguments: bool=True,
        other_kwargs: dict={}
    ) -> BaseTrainer:
    
    # log the arguments
    saved_args = {**locals()}
    print(f'{saved_args = }')

    time_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

    if save_arguments:
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(saved_args, open(save_dir / f'args.pkl', 'wb'))

    
    if logger is None:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(save_dir) / f'UdeepSC_ave_{time_str}.ansi'
        logger = get_logger('UdeepSC_ave', str(log_path), stdout=True)

    logger.info('UdeepSC_ave')
    logger.info(f'argument for trainer function: {saved_args}')

    # basic setting
    args = {
        'batch_size': 64,
        'n_epoch': 200,
        'warmup_ep': 1,
        'n_user': 2,
        'ta_perform': 'ave', 
    }

    opt_args_dict = {
        'opt': 'adamw',
        'lr': 3e-5,
        'weight_decay': 0.02,
        'opt_eps': 1e-8,
        'opt_betas': [0.95, 0.99],
    }
    
    logger.info(f"optimizer settings: {opt_args_dict}")

    data_dir = 'data/avedata'
    channel_gain_var = 1.0
    channel_gain_var_ls = [[1], [1]]
    snr_ls = [1] * args['n_user']

    # load dataset
    train_dataloader, val_dataloader = make_AVE_dataloaders(batch_size=batch_size, root=data_dir, num_workers=num_workers)
    
    # make channel
    if channel_type == 'rayleigh':
        channel = RayleighFadingMultiUplinkChannel(
            args['n_user'], 
            snr_db=[snr_db], 
            divide_gain=False, 
            channel_gain_var=channel_gain_var_ls, 
            interfere_mode=interfere_mode, 
            fading_mode=fading_mode,
        )
    elif channel_type == 'awgn':
        channel = AWGNMultiUplinkChannel(
                args['n_user'],
                snr_db = [snr_db],
                interfere_mode=interfere_mode
            )
    
    

    # make model stuff
    model = model_class(
        num_symbols=num_symbols,
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )

    criterion = torch.nn.MultiLabelSoftMarginLoss()
    acc_function = utils.compute_acc_AVE
    # optimizer = torch.optim.AdamW(model.parameters(), **opt_args)
    optimizer = create_optimizer(opt_args_dict, model)

    num_iters_per_ep = len(train_dataloader)
    warmup_iters = num_iters_per_ep * args['warmup_ep']
    total_iters = n_epoch * num_iters_per_ep - warmup_iters
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5)
    
    lr_scheduler = SequentialLR(
        optimizer, schedulers=[
            LinearLR(               # warmup: from start_factor to lr
                optimizer,
                start_factor=1e-10 / opt_args_dict['lr'],  # cannot be 0
                end_factor=1,
                total_iters=warmup_iters
            ),
            CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5) # main: from lr to eta_min
        ], milestones=[warmup_iters])
    
    # make trainer
    trainer = trainer_class(
        logger=logger,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        display_interval=display_interval,
        n_epoch=n_epoch,
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
        ta_perform=args['ta_perform'],
        **other_kwargs,
    )

    return trainer

def get_udeepscoma_ave_trainer(
        logger: logging.Logger = None,
        save_dir: Path = Path('./checkpoint'),
        resume_checkpoint: Path = None,
        gpus: list[int] = [0],
        seed: int = 0,
        display_interval: int = 10,
        trainer_class: type[BaseTrainer]=UDeepSCOMATrainer_Ave,
        model_class: type[UDeepSCNoSIC] =UDeepSCOMA_ave,
        num_workers: int=4,
        channel_type: Literal['awgn', 'rayleigh'] = 'awgn',
        interfere_mode: str='all',
        fading_mode: str='slow',
        snr_db: float=12.0,
        num_symbols: int=32,
        batch_size: int=50,
        n_epoch: int=150,
        save_arguments: bool=True,
        other_kwargs: dict={}
    ) -> BaseTrainer:
    
    # log the arguments
    saved_args = {**locals()}
    print(f'{saved_args = }')

    time_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

    if save_arguments:
        save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(saved_args, open(save_dir / f'args.pkl', 'wb'))

    
    if logger is None:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(save_dir) / f'UdeepSCOMA_ave_{time_str}.ansi'
        logger = get_logger('UdeepSCOMA_ave', str(log_path), stdout=True)

    logger.info('UdeepSCOMA_ave')
    logger.info(f'argument for trainer function: {saved_args}')

    # basic setting
    args = {
        'warmup_ep': 1,
        'n_user': 2,
        'ta_perform': 'ave', 
    }

    opt_args_dict = {
        'opt': 'adamw',
        'lr': 3e-5,
        'weight_decay': 0.02,
        'opt_eps': 1e-8,
        'opt_betas': [0.95, 0.99],
    }
    
    logger.info(f"optimizer settings: {opt_args_dict}")

    data_dir = 'data/avedata'
    channel_gain_var = 1.0
    channel_gain_var_ls = [[1], [1], [1]]
    snr_ls = [1] * args['n_user']
    power_constraint = [1] * args['n_user']

    # load dataset
    train_dataloader, val_dataloader = make_AVE_dataloaders(batch_size=batch_size, root=data_dir, num_workers=num_workers, pin_memory=True)
    
    # make channel
    if channel_type == 'rayleigh':
        channel = RayleighFadingSingleChannel(
            snr_db=snr_db, 
            divide_gain=False, 
            channel_gain_var=channel_gain_var, 
            fading_mode=fading_mode,
        )
    elif channel_type == 'awgn':
        channel = AWGNSingleChannel(
                snr_db = snr_db,
            )
    
    
    # make model stuff
    model = model_class(
        num_symbols=num_symbols,
        mode='small',
        img_size=32,
        patch_size=4,
        img_embed_dim=128,
        text_embed_dim=512,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )

    criterion = torch.nn.MultiLabelSoftMarginLoss()
    acc_function = utils.compute_acc_AVE
    # optimizer = torch.optim.AdamW(model.parameters(), **opt_args)
    optimizer = create_optimizer(opt_args_dict, model)

    num_iters_per_ep = len(train_dataloader)
    warmup_iters = num_iters_per_ep * args['warmup_ep']
    total_iters = n_epoch * num_iters_per_ep - warmup_iters
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5)
    
    lr_scheduler = SequentialLR(
        optimizer, schedulers=[
            LinearLR(               # warmup: from start_factor to lr
                optimizer,
                start_factor=1e-10 / opt_args_dict['lr'],  # cannot be 0
                end_factor=1,
                total_iters=warmup_iters
            ),
            CosineAnnealingLR(optimizer, T_max=total_iters ,eta_min=1e-5) # main: from lr to eta_min
        ], milestones=[warmup_iters])
    
    # make trainer
    trainer = trainer_class(
        logger=logger,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        display_interval=display_interval,
        n_epoch=n_epoch,
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
        ta_perform=args['ta_perform'],
        **other_kwargs,
    )

    return trainer

def main_msa_train(config, model_id=1):
    # set seed
    import random
    
    seed = config.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
    
    # set save_dir
    model_name = _get_model_name(config["channel"]["channel_type"], config["channel"]["snr_db"], config["model_type"], config["model"]["encoder_out_dims"], config["dataset"]["name"], model_id)
    save_dir = Path(config["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # make trainer
    trainer = get_udeepsc_msa_trainer(
        logger=None,
        save_dir=save_dir,
        resume_checkpoint=None,
        gpus=config["gpu"],
        seed=seed,
        display_interval=10,
        trainer_class=UDeepSCNoSICTrainer_Msa,
        model_class=UDeepSCNoSIC_msa,
        num_workers=4,
        channel_type=config["channel"]["channel_type"],
        interfere_mode=config["channel"]["interfere_mode"],
        fading_mode=config["channel"]["fading_mode"],
        num_symbols=config["model"]["encoder_out_dims"],
        snr_db=config["channel"]["snr_db"],
        batch_size=config["batch_size"],
        n_epoch=config["epochs"],
        power_constraint=[config["channel"]['power']] * 3,
        divide_gain=True,
        save_arguments=True
    )
    
    from timeit import default_timer
    start = default_timer()
    
    # start training
    trainer.train()
    
    train_time = default_timer() - start
    trainer.logger.info(f'Total train time: {train_time}')
    
def main_oma_msa_train(config, model_id=1):
    import random
    
    seed = config.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
    
    # set save_dir
    model_name = _get_model_name(config["channel"]["channel_type"], config["channel"]["snr_db"], config["model_type"], config["model"]["encoder_out_dims_OMA"], config["dataset"]["name"], model_id)
    save_dir = Path(config["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # make trainer
    trainer = get_udeepscoma_msa_trainer(
        logger=None,
        save_dir=save_dir,
        resume_checkpoint=None,
        gpus=config["gpu"],
        seed=seed,
        display_interval=10,
        trainer_class=UDeepSCOMATrainer_Msa,
        model_class=UDeepSCOMA_msa,
        num_workers=4,
        channel_type=config["channel"]["channel_type"],
        fading_mode=config["channel"]["fading_mode"],
        num_symbols=config["model"]["encoder_out_dims_OMA"],
        snr_db=config["channel"]["snr_db"],
        batch_size=config["batch_size"],
        n_epoch=config["epochs"],
        save_arguments=True
    )
    
    from timeit import default_timer
    start = default_timer()
    
    # start training
    trainer.train()
    
    train_time = default_timer() - start
    trainer.logger.info(f'Total train time: {train_time}')
    
def main_sic_msa_train(config, model_id=1):
    # set seed
    import random
    
    seed = config.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
    
    # set save_dir
    model_name = _get_model_name(config["channel"]["channel_type"], config["channel"]["snr_db"], config["model_type"], config["model"]["encoder_out_dims"], config["dataset"]["name"], model_id)
    save_dir = Path(config["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # make trainer
    trainer = get_udeepsc_msa_trainer(
        logger=None,
        save_dir=save_dir,
        resume_checkpoint=None,
        gpus=config["gpu"],
        seed=seed,
        display_interval=10,
        trainer_class=UDeepSCSICTrainer_Msa,
        model_class=UDeepSCSIC_msa,
        num_workers=4,
        channel_type=config["channel"]["channel_type"],
        interfere_mode=config["channel"]["interfere_mode"],
        fading_mode=config["channel"]["fading_mode"],
        num_symbols=config["model"]["encoder_out_dims"],
        snr_db=config["channel"]["snr_db"],
        batch_size=config["batch_size"],
        n_epoch=config["epochs"],
        power_constraint=[0.8, 0.4, 1.8],
        divide_gain=False,
        save_arguments=True,
        other_kwargs={"channel_type": config["channel"]["channel_type"]}
    )
    
    from timeit import default_timer
    start = default_timer()
    
    # start training
    trainer.train()
    
    train_time = default_timer() - start
    trainer.logger.info(f'Total train time: {train_time}')
    
def main_ave_train(config, model_id=1):
    # set seed
    import random
    
    seed = config.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
    
    # set save_dir
    model_name = _get_model_name(config["channel"]["channel_type"], config["channel"]["snr_db"], config["model_type"], config["model"]["encoder_out_dims"], config["dataset"]["name"], model_id)
    save_dir = Path(config["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # make trainer
    trainer = get_udeepsc_ave_trainer(
        logger=None,
        save_dir=save_dir,
        resume_checkpoint=None,
        gpus=config["gpu"],
        seed=seed,
        display_interval=10,
        trainer_class=UDeepSCNoSICATrainer_AVE,
        model_class=UDeepSCNoSIC_ave,
        num_workers=4,
        channel_type=config["channel"]["channel_type"],
        fading_mode=config["channel"]["fading_mode"],
        num_symbols=config["model"]["encoder_out_dims"],
        snr_db=config["channel"]["snr_db"],
        batch_size=config["batch_size"],
        n_epoch=config["epochs"],
        power_constraint=[config["channel"]['power']] * 2,
        save_arguments=True
    )
    
    from timeit import default_timer
    start = default_timer()
    
    # start training
    trainer.train()
    
    train_time = default_timer() - start
    trainer.logger.info(f'Total train time: {train_time}')
    
def main_oma_ave_train(config, model_id=1):
    # set seed
    import random
    
    seed = config.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
    
    # set save_dir
    model_name = _get_model_name(config["channel"]["channel_type"], config["channel"]["snr_db"], config["model_type"], config["model"]["encoder_out_dims_OMA"], config["dataset"]["name"], model_id)
    save_dir = Path(config["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # make trainer
    trainer = get_udeepscoma_ave_trainer(
        logger=None,
        save_dir=save_dir,
        resume_checkpoint=None,
        gpus=config["gpu"],
        seed=seed,
        display_interval=10,
        trainer_class=UDeepSCOMATrainer_Ave,
        model_class=UDeepSCOMA_ave,
        num_workers=4,
        channel_type=config["channel"]["channel_type"],
        fading_mode=config["channel"]["fading_mode"],
        num_symbols=config["model"]["encoder_out_dims_OMA"],
        snr_db=config["channel"]["snr_db"],
        batch_size=config["batch_size"],
        n_epoch=config["epochs"],
        save_arguments=True
    )
    
    from timeit import default_timer
    start = default_timer()
    
    # start training
    trainer.train()
    
    train_time = default_timer() - start
    trainer.logger.info(f'Total train time: {train_time}')
    
def main_sic_ave_train(config, model_id=1):
    # set seed
    import random
    
    seed = config.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
    
    # set save_dir
    model_name = _get_model_name(config["channel"]["channel_type"], config["channel"]["snr_db"], config["model_type"], config["model"]["encoder_out_dims"], config["dataset"]["name"], model_id)
    save_dir = Path(config["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # make trainer
    trainer = get_udeepsc_ave_trainer(
        logger=None,
        save_dir=save_dir,
        resume_checkpoint=None,
        gpus=config["gpu"],
        seed=seed,
        display_interval=10,
        trainer_class=UDeepSCSICTrainer_Ave,
        model_class=UDeepSCSIC_ave,
        num_workers=4,
        channel_type=config["channel"]["channel_type"],
        interfere_mode=config["channel"]["interfere_mode"],
        fading_mode=config["channel"]["fading_mode"],
        num_symbols=config["model"]["encoder_out_dims"],
        snr_db=config["channel"]["snr_db"],
        batch_size=config["batch_size"],
        n_epoch=config["epochs"],
        power_constraint=[1.5, 0.5],
        save_arguments=True,
        other_kwargs={'channel_type': config["channel"]["channel_type"]},
    )
    
    from timeit import default_timer
    start = default_timer()
    
    # start training
    trainer.train()
    
    train_time = default_timer() - start
    trainer.logger.info(f'Total train time: {train_time}')


if __name__ == '__main__':
    # args = parse_udeepsc_args()

    # print(args.gpu)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='configuration file')
    parser.add_argument('--model_id', type=int, default=None, help='target model id to re-train (if there are multiple models with same setting)')
    args = parser.parse_args()
    
    # set gpus
    # args.gpu = [int(i) for i in args.gpu.split(',')]
    
    # load config
    with open(args.config) as f:
        config = json.load(f)

    task = config["task"]
    scheme = config["channel"]["scheme"]
    n_models = config["n_models"]
    model_start_id = config.get("model_start_id", 1)
    
    if args.model_id is not None:
        if task == "msa":
            if scheme == "oma":
                main_oma_msa_train(config, args.model_id)
            elif scheme == "sic":
                main_sic_msa_train(config, args.model_id)
            elif scheme == "nosic":
                main_msa_train(config, args.model_id)
            else:
                raise ValueError(f"Unknown scheme type: {scheme}")
        elif task == "ave":
            if scheme == "oma":
                main_oma_ave_train(config, args.model_id)
            elif scheme == "sic":
                main_sic_ave_train(config, args.model_id)
            elif scheme == "nosic":
                main_ave_train(config, args.model_id)
            else:
                raise ValueError(f"Unknown scheme type: {scheme}")
        else:
            raise ValueError(f"Unknown task: {task}")
    else:
        for i in range(model_start_id, n_models + 1):
            if task == "msa":
                if scheme == "oma":
                    main_oma_msa_train(config, i)
                elif scheme == "sic":
                    main_sic_msa_train(config, i)
                elif scheme == "nosic":
                    main_msa_train(config, i)
                else:
                    raise ValueError(f"Unknown scheme type: {scheme}")
            elif task == "ave":
                if scheme == "oma":
                    main_oma_ave_train(config, i)
                elif scheme == "sic":
                    main_sic_ave_train(config, i)
                elif scheme == "nosic":
                    main_ave_train(config, i)
                else:
                    raise ValueError(f"Unknown scheme type: {scheme}")
            else:
                raise ValueError(f"Unknown task: {task}")
    