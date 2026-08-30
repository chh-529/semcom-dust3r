from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import *
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import functools
import random

from ...log import get_logger
from ... import utils
from ...channel import *
from ..udeepsc.cp_load import *
from ...channel import *
from ...utils import (
    str_type, tensor_complex2real, tensor_real2complex, to_device, calc_metrics
)
from ..experiment.modality import (
    UplinkInference, UplinkExperiment20250911, test_general, plot_general, UplinkExperiment20250914, test_OMA_case, MultiModalSC
)

from ..experiment.method import (
    UITestSuite
)

from ..experiment.regression import (
    _model_infos_from_config
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PreliminaryTestSuite:
    ue_cls = UplinkExperiment20250914
    # result_main_folder = ue_cls.result_main_folder / 'prelim'
    
    @classmethod
    def _result_root(cls) -> Path:
        return cls.ue_cls.result_main_folder / 'prelim'
    
    @classmethod
    def _get_stuff(cls, ue_cls: type[UplinkExperiment20250911], batch_size: int, model_info: dict | tuple, shuffle: bool, device: str, m_combine: Optional[MultiModalSC.SupportedModalCombin]=None):
        """
        use ue_cls to get the related model
        returns:
            th: TestHelper20250426 instance
            edp: EDP instance for the model, made from th
            dataloader: InfiniteDataLoader instance for the model, made from th
        """
        
        model_info = ue_cls._normalize_model_info(model_info, tuple)
        channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id = model_info
        
        th = ue_cls.TestHelper20250911(
            device=device,
            path=ue_cls._get_model_path(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id), 
            model_type=model_type, encoder_out_dims=encoder_out_dims, 
            channel_type=channel_type, dataset_type=dataset_type,
            use_latest_checkpoint=False, modal_combin=m_combine
        )
        sc = th.get_multimodal_sc()
        dataloader = th.get_dataloader(1, batch_size, False, shuffle=shuffle)

        return th, sc, dataloader

    @classmethod
    def _test(cls, ui_cls: UplinkInference, ui_kwargs: dict,
             ue_cls: UplinkExperiment20250911, model_infos: list[tuple | dict],
             uplink_method: UplinkInference.SupportedUplinkMethods, 
             n_user, m_combine: MultiModalSC.SupportedModalCombin, 
             snr_range: Optional[list[int]] = None,
             divide_gain: bool=True,
             batch_size: int = 50, n_batch: int = 20, device: Optional[str] = 'cuda:0'
             ):
        """
        Test the given UplinkInference class with the given model info and method.

        Args:
            ui_cls: UplinkInference class to test
            ui_kwargs: kwargs to pass to the ui_cls constructor
                except the ones from UplinkInference: ['edps', 'power_constraint', 'channel', 'device']
            ue_cls: UplinkExperiment20250911 class to use for getting the models
            model_info: a list of tuple or dict, model info for each model to test, should be
                ('channel_type', snr_db, 'model_type', encoder_out_channels, 'dataset_type', model_id)
            uplink_method: the uplink method to use
                one of UplinkInference.SupportedUplinkMethods
            batch_size: batch size to use for the dataloaders, default is 50
            n_batch: number of batches to test, default is 20
            get_data_batch: the batch index to get the data from, default is 0
            device: the device to use for the ui_cls, default is 'cuda:0'
        
        Note:
            For model commutative, we assume that any uplink method is commutative, so this commutative property is only for the UI
            
            We assume the model from model_info should be the same, except the model_id
        """
        
        th_ls = []
        md_if = list(model_infos[0])
        md_if.pop()
        channel_type = md_if[0]
        
        for model_info in model_infos:
            model_info = ue_cls._normalize_model_info(model_info, tuple)

            # get the necessary stuff
            th, _, _ = cls._get_stuff(ue_cls, batch_size, model_info, shuffle=True, device=device, m_combine=m_combine)
            
            th_ls.append(th)

        partial_ui_cls = functools.partial(ui_cls, **ui_kwargs)

        # check if the result file already exists
        result_path = (
            cls._result_root() / 
            m_combine / 
            uplink_method / 
            ue_cls._get_model_name(*md_if) /  
            'result.pkl'
        )

        if result_path.exists():
            print(f"[*] Result file {result_path} already exists, skipping test.")
            return
        
        result_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Result Path: {result_path}')
        
        test_general(
            th_ls, result_path, uplink_method,
            batch_size=batch_size, n_batch=n_batch, 
            snr_range=snr_range,
            n_user=n_user,
            divide_gain=divide_gain,
            channel_type=channel_type,
            ui_cls=partial_ui_cls
        )
    @classmethod
    def _test_oma(cls, ui_cls: UplinkInference, ui_kwargs: dict,
             ue_cls: UplinkExperiment20250911, model_infos: list[tuple | dict],
             uplink_method: UplinkInference.SupportedUplinkMethods, 
             n_user, m_combine: MultiModalSC.SupportedModalCombin, 
             snr_range: Optional[list[int]] = None,
             batch_size: int = 50, n_batch: int = 20, device: Optional[str] = 'cuda:0'
             ):
        """
        Test the given UplinkInference class with the given model info and method.

        Args:
            ui_cls: UplinkInference class to test
            ui_kwargs: kwargs to pass to the ui_cls constructor
                except the ones from UplinkInference: ['edps', 'power_constraint', 'channel', 'device']
            ue_cls: UplinkExperiment20250911 class to use for getting the models
            model_info: tuple or dict, model info for the first model, should be
                ('channel_type', snr_db, 'model_type', encoder_out_channels, 'dataset_type', model_id)
            uplink_method: the uplink method to use
                one of UplinkInference.SupportedUplinkMethods
            batch_size: batch size to use for the dataloaders, default is 50
            n_batch: number of batches to test, default is 20
            get_data_batch: the batch index to get the data from, default is 0
            device: the device to use for the ui_cls, default is 'cuda:0'
        
        Note:
            For model commutative, we assume that any uplink method is commutative, so this commutative property is only for the UI
            
            We assume the model from model_info should be the same, except the model_id
        """
        
        th_ls = []
        md_if = list(model_infos[0])
        md_if.pop()
        for model_info in model_infos:
            model_info = ue_cls._normalize_model_info(model_info, tuple)

            # get the necessary stuff
            th, _, _ = cls._get_stuff(ue_cls, batch_size, model_info, shuffle=True, device=device, m_combine=m_combine)
            
            th_ls.append(th)

        partial_ui_cls = functools.partial(ui_cls, **ui_kwargs)

        # check if the result file already exists
        result_path = (
            cls._result_root() / 
            m_combine / 
            uplink_method / 
            ue_cls._get_model_name(*md_if) /  
            'result.pkl'
        )

        if result_path.exists():
            print(f"[*] Result file {result_path} already exists, skipping test.")
            return
        
        result_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Result Path: {result_path}')
        
        test_OMA_case(
            th_ls, result_path, uplink_method,
            batch_size=batch_size, n_batch=n_batch, 
            snr_range=snr_range,
            ui_cls=partial_ui_cls
        )
    
    
    @classmethod
    def prelim_test(cls):
        config = getattr(cls, 'config', None)
        if config is None:
            raise RuntimeError("OptimizeTestSuit.config was not set. Set it before calling test().")
        
        device = f"cuda:{config['gpu'][0]}"

        mi = config['model_info']
        task = 'msa'
        # snr_range = np.linspace(-6, 12, 10).tolist()
        snr_range = list(range(-6, 13))
        
        def _noFSM(n_user, m_combine):
            method = 'no_FSM'
            model_infos = [(mi["channel_type"], mi["snr_db"], f'udeepsc_{task}', mi["encoder_out_dims"], mi["dataset_type"], i) for i in range(1, config['model_nums']["nosic"] + 1)]

            return cls._test(
                UplinkInference, 
                {'power_constraint': [1] * n_user},
                cls.ue_cls, model_infos, method, n_user=n_user, m_combine=m_combine,
                divide_gain=True,batch_size=30, n_batch=None, device=device, snr_range=snr_range
            )
        def _noFSM_oma(n_user, m_combine):
            method = 'no_FSM_oma'
            model_infos = [(mi["channel_type"], mi["snr_db"], f'udeepscOMA_{task}', mi["encoder_out_dims_OMA"], mi["dataset_type"], i) for i in range(1, config['model_nums']["oma"] + 1)]
            
            return cls._test_oma(
                UplinkInference, 
                {'power_constraint': [1] * n_user},
                cls.ue_cls, model_infos, method, batch_size=30, n_user=n_user, m_combine=m_combine, n_batch=None, device=device, snr_range=snr_range
            )
        def _noFSM_sic(n_user, m_combine):
            method = 'no_FSM_sic'
            model_infos = [(mi["channel_type"], mi["snr_db"], f'udeepscSIC_{task}', mi["encoder_out_dims"], mi["dataset_type"], i) for i in range(1, config['model_nums']["sic"] + 1)]
            power_constraint = [1.5, 0.5]

            return cls._test(
                UplinkInference, 
                {'power_constraint': power_constraint}, 
                cls.ue_cls, model_infos, method, n_user=n_user, m_combine=m_combine, divide_gain=False, batch_size=30, n_batch=None, device=device, snr_range=snr_range
            )
        
        
        def prelim_test_all():
            UITestSuite.config = config
            UITestSuite.result_main_folder = Path(config['result_dir']) / 'prelim' / 'all'
            
            UITestSuite.test()
        
        def prelim_test_Text_Img():
            n_user = 2
            m_combine = 'textimg'
            _noFSM(n_user, m_combine)
            _noFSM_oma(3, m_combine)
            _noFSM_sic(n_user, m_combine)
        
        def prelim_test_Text_Spe():
            n_user = 2
            m_combine = 'textspe'
            _noFSM(n_user, m_combine)
            _noFSM_oma(3, m_combine)
            _noFSM_sic(n_user, m_combine)
        
        def prelim_test_Img_Spe():
            n_user = 2
            m_combine = 'imgspe'
            _noFSM(n_user, m_combine)
            _noFSM_oma(3, m_combine)
            _noFSM_sic(n_user, m_combine)
            
            
        prelim_test_all()
        prelim_test_Text_Img()
        prelim_test_Text_Spe()
        prelim_test_Img_Spe()
        
class Plotting():
    ue_cls = UplinkExperiment20250914
    
    @classmethod
    def _result_root(cls) -> Path:
        return PreliminaryTestSuite._result_root()
    
    # result_plot_folder = cls._result_root() / '..'
    
    @classmethod
    def plots(cls):
        config = getattr(cls, 'config', None)
        if config is None:
            raise RuntimeError("OptimizeTestSuit.config was not set. Set it before calling test().")
        
        mi = config["model_info"]
        channel = mi['channel_type']
        
        def plot_all():
            model_name = f"{channel}_{mi['snr_db']}_udeepsc_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            oma_model_name = f"{channel}_{mi['snr_db']}_udeepscOMA_msa_symbols_{mi['encoder_out_dims_OMA'] // 2}_{mi['dataset_type']}"
            sic_model_name = f"{channel}_{mi['snr_db']}_udeepscSIC_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            
            result_path = cls._result_root() / 'all' / 'plots' /  model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'all' / 'ui' / 'no_FSM' / model_name / 'result.pkl'
            oma_model = cls._result_root() / 'all' / 'ui' / 'no_FSM_oma' / oma_model_name / 'result.pkl'
            
            sic_model = cls._result_root() / 'all' / 'ui' / 'no_FSM_sic' / sic_model_name / 'result.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [oma_model, sic_model, nofsm], 
                [
                    {'marker': 'o', 'linestyle': '-', 'color': 'tab:blue', 'label': 'UDeepSC'},
                    {'marker': 'D', 'linestyle': '-', 'color': 'forestgreen', 'label': 'UDeepSC NOMA (with SIC)'},
                    {'marker': 'o', 'linestyle': '-', 'color': 'red', 'label': 'UDeepSC NO (w/o SIC)'},
                ], 
                legend_kwargs={'loc': 'outside upper center', 'ncols': 2},
                override_title='AWGN Channel',
                task_type='msa', channel_type=channel, y_limited=[55, 85, 7]
            )
            
        def plot_TI():
            model_name = f"{channel}_{mi['snr_db']}_udeepsc_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            oma_model_name = f"{channel}_{mi['snr_db']}_udeepscOMA_msa_symbols_{mi['encoder_out_dims_OMA'] // 2}_{mi['dataset_type']}"
            sic_model_name = f"{channel}_{mi['snr_db']}_udeepscSIC_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            
            result_path = cls._result_root() / 'textimg' / 'plots' /  model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'textimg' / 'no_FSM' / model_name / 'result.pkl'
            oma_model = cls._result_root() / 'textimg' / 'no_FSM_oma' / oma_model_name / 'result.pkl'
            
            sic_model = cls._result_root() / 'textimg' / 'no_FSM_sic' / sic_model_name / 'result.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [oma_model, sic_model, nofsm], 
                [
                    {'marker': 'o', 'linestyle': '-', 'color': 'tab:blue', 'label': 'UDeepSC'},
                    {'marker': 'D', 'linestyle': '-', 'color': 'forestgreen', 'label': 'UDeepSC NOMA (with SIC)'},
                    {'marker': 'o', 'linestyle': '-', 'color': 'red', 'label': 'UDeepSC NO (w/o SIC)'},
                ], 
                legend_kwargs={'loc': 'outside upper center', 'ncols': 2},
                override_title='AWGN Channel',
                upper_bound_info={'Upper bound': (83.1, 'k'), 'Upper bound (Only 2 modal)': (83.3, 'brown')},
                task_type='msa', channel_type=channel, y_limited=[55, 85, 7]
            )
        def plot_TS():
            model_name = f"{channel}_{mi['snr_db']}_udeepsc_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            oma_model_name = f"{channel}_{mi['snr_db']}_udeepscOMA_msa_symbols_{mi['encoder_out_dims_OMA'] // 2}_{mi['dataset_type']}"
            sic_model_name = f"{channel}_{mi['snr_db']}_udeepscSIC_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            
            result_path = cls._result_root() / 'textspe' / 'plots' /  model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'textspe' / 'no_FSM' / model_name / 'result.pkl'
            oma_model = cls._result_root() / 'textspe' / 'no_FSM_oma' / oma_model_name / 'result.pkl'
            
            sic_model = cls._result_root() / 'textspe' / 'no_FSM_sic' / sic_model_name / 'result.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [oma_model, sic_model, nofsm], 
                [
                    {'marker': 'o', 'linestyle': '-', 'color': 'tab:blue', 'label': 'UDeepSC'},
                    {'marker': 'D', 'linestyle': '-', 'color': 'forestgreen', 'label': 'UDeepSC NOMA (with SIC)'},
                    {'marker': 'o', 'linestyle': '-', 'color': 'red', 'label': 'UDeepSC NO (w/o SIC)'},
                ], 
                legend_kwargs={'loc': 'outside upper center', 'ncols': 2},
                override_title='AWGN Channel',
                upper_bound_info={'Upper bound': (83.1, 'k'), 'Upper bound (Only 2 modal)': (83, 'brown')},
                task_type='msa', channel_type=channel, y_limited=[55, 85, 7]
            )
        def plot_IS():
            model_name = f"{channel}_{mi['snr_db']}_udeepsc_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            oma_model_name = f"{channel}_{mi['snr_db']}_udeepscOMA_msa_symbols_{mi['encoder_out_dims_OMA'] // 2}_{mi['dataset_type']}"
            sic_model_name = f"{channel}_{mi['snr_db']}_udeepscSIC_msa_symbols_{mi['encoder_out_dims'] // 2}_{mi['dataset_type']}"
            
            result_path = cls._result_root() / 'imgspe' / 'plots' /  model_name
            print(result_path)
            
            nofsm = cls._result_root() / 'imgspe' / 'no_FSM' / model_name / 'result.pkl'
            oma_model = cls._result_root() / 'imgspe' / 'no_FSM_oma' / oma_model_name / 'result.pkl'
            
            sic_model = cls._result_root() / 'imgspe' / 'no_FSM_sic' / sic_model_name / 'result.pkl'
            
            # print(nofsm)
            # print(fsm_poly)
            
            result_path.mkdir(parents=True, exist_ok=True)
            
            plot_general(
                result_path, 'Accuracy (%)', 'acc',
                [oma_model, sic_model, nofsm], 
                [
                    {'marker': 'o', 'linestyle': '-', 'color': 'tab:blue', 'label': 'UDeepSC'},
                    {'marker': 'D', 'linestyle': '-', 'color': 'forestgreen', 'label': 'UDeepSC NOMA (with SIC)'},
                    {'marker': 'o', 'linestyle': '-', 'color': 'red', 'label': 'UDeepSC NO (w/o SIC)'},
                ], 
                legend_kwargs={'loc': 'outside upper center', 'ncols': 2},
                override_title='AWGN Channel',
                upper_bound_info={'Upper bound': (83.1, 'k'), 'Upper bound (Only 2 modal)': (56, 'brown')},
                task_type='msa', channel_type=channel
            )
            
        plot_all()
        plot_TI()
        plot_TS()
        plot_IS()


if __name__ == '__main__':
    fixed_args = {
        "cp_main_folder": "checkpoint/20251105",
        "result_dir": "tmp/20251231",
        "task": "msa",
        "seed": 2001,
        "gpu": [1],
        "model_info": {
            "channel_type": "awgn",
            "snr_db": 12,
            "model_type": "udeepsc_msa",
            "dataset_type": "cmu-mosei",
            "encoder_out_dims": 48,
            "encoder_out_dims_OMA": 16
        },
        "model_nums": {
            "nosic": 5,
            "sic": 5,
            "oma": 5
        },
        "modalities": ["text", "image", "speech"],
    }
    
    seed = fixed_args.get("seed", None)
    if seed is None:
        seed = random.randint(0, 10000)
        
    set_seed(seed)
    
    UplinkExperiment20250914.result_main_folder = Path(fixed_args['result_dir'])
    UITestSuite.ue_cls.cp_main_folder = Path(fixed_args['cp_main_folder'])
    
    PreliminaryTestSuite.config = fixed_args
    Plotting.config = fixed_args
    
    PreliminaryTestSuite.prelim_test()
    Plotting.plots()