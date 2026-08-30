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

from ...log import get_logger
from ... import utils
from ...channel import *
from ...model.component import power_normalize, signal_power_per_complex_symbol, signal_power
from ..udeepsc.cp_load import *
from ...channel import *
from ...utils import (
    str_type, tensor_complex2real, tensor_real2complex, to_device, calc_metrics
)
from .modality import (
    UplinkInference, UplinkExperiment20250911, test_general, plot_general, UplinkExperiment20250914, test_OMA_case
)

class FSM:
    def __init__(self, modalities:list[str], m_ratios:List[float]):
        self.modalities = modalities
        self.m_ratios = m_ratios  
    
    def mask_gen(self, feature_scores, m_ratio: float=1):
        """
        Generate a 0-1 mask for feature selection.
         Args:
            feature_scores (np.ndarray): Importance scores of shape (d,), larger means more important.
            m_ratio (float): Selection ratio between 0 and 1.
        
        Returns:
            mask (np.ndarray): Binary mask of shape (d,), 1 for selected features, 0 otherwise.
        """
        d = feature_scores.shape[0]
        k = max(1, int(np.ceil(m_ratio * d)))  # number of features to select
        
         # get indices of top-k features
        topk_indices = np.argsort(feature_scores)[-k:]
        
        # build mask
        mask = np.zeros(d, dtype=np.int32)
        mask[topk_indices] = 1
        
        return mask
        
    def _apply_mask(self, features: torch.Tensor, feature_scores, m_ratio):
        """
        Select the features to be transmitted
        Args:
            features: tensor of shape (B, d) where d is the feature dim
            feature_scores (np.ndarray): Importance scores of shape (d,), larger means more important.
            m_ratio (float): Selection ratio between 0 and 1.
        Returns:
            features: tensor of shape (B, d) where d is the feature dim, after feature selection
            mask: tensor of shape (1, d), the mask used for feature selection
        """
        mask = self.mask_gen(feature_scores, m_ratio)
        # Determine the number of leading dimensions before the feature dim
        expand_shape = [1] * (features.dim() - 1) + [features.shape[-1]]
        mask = torch.from_numpy(mask).to(features.device).reshape(expand_shape)
        mask = mask.expand_as(features)
        # Optionally, scale features by m_ratio if needed
        features = features * mask
        
        return features, mask
    
    def feature_selection_all(self, inputs):
        """
        Select features to be transmitted for each user
        Args:
            inputs: 
        """
        raise NotImplementedError
    
class FSM_Msa(FSM):
    def __init__(self, feature_scores_all:Dict[str, np.ndarray], *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Args:
            feature_scores_all: A dict containing feature importance scores for each modality in form:
                {"modality_name": scores in np.ndarray of shape (dim,)}
        """
        self.feature_scores = feature_scores_all
    
    def feature_selection_all(self, inputs:Tuple):
        """
        Select features to be transmitted for each user
        Args:
            inputs: A Tuple of tensors, each is a real tensor of shape (batch, frames, d) where d is the feature dim. 
                In msa, assume modality order is [text, image, speech]
            m_ratios: A list of float number, each is the selection ratio for the corresponding modality
        Returns:
            A Tuple of tensors, each is a real tensor of shape (batch, frames, d) where d is the feature dim, after feature selection.
            List of masks used for feature selection.
        """
        masked_features = []
        masks = []
        
        for i, (mod, m_ratio) in enumerate(zip(self.modalities, self.m_ratios)):
            masked, mask = self._apply_mask(inputs[i], self.feature_scores[mod], m_ratio=m_ratio)
            masked_features.append(masked)
            masks.append(mask)

        return tuple(masked_features), masks
    
class FSM_Msa_random(FSM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _apply_mask(self, features: torch.Tensor, feature_scores, m_ratio):
        """
        Select the features to be transmitted
        Args:
            features: tensor of shape (B, d) where d is the feature dim
            feature_scores (np.ndarray): Importance scores of shape (d,), larger means more important.
            m_ratio (float): Selection ratio between 0 and 1.
        Returns:
            features: tensor of shape (B, d) where d is the feature dim, after feature selection
            mask: tensor of shape (1, d), the mask used for feature selection
        """
        dim = features.shape[-1]
        num_ones = int(m_ratio * dim)
        mask = torch.cat([torch.ones(num_ones), torch.zeros(dim - num_ones)])
        mask = mask[torch.randperm(dim)]
        # Determine the number of leading dimensions before the feature dim
        expand_shape = [1] * (features.dim() - 1) + [features.shape[-1]]
        mask = mask.to(features.device).reshape(expand_shape)
        mask = mask.expand(*features.shape)
        # Optionally, scale features by m_ratio if needed
        features = features * mask
        
        return features, mask    
    
    def feature_selection_all(self, inputs:Tuple):
        """
        Select features to be transmitted for each user
        Args:
            inputs: A Tuple of tensors, each is a real tensor of shape (batch, frames, d) where d is the feature dim. 
            In msa, assume modality order is [text, image, speech]
            m_ratios: A list of float number, each is the selection ratio for the corresponding modality
        Returns:
            A Tuple of tensors, each is a real tensor of shape (batch, frames, d) where d is the feature dim, after feature selection.
        """
        masked_features = []
        masks = []
        
        for i, (mod, m_ratio) in enumerate(zip(self.modalities, self.m_ratios)):
            masked, mask = self._apply_mask(inputs[i], None, m_ratio=m_ratio)
            masked_features.append(masked)
            masks.append(mask)

        return tuple(masked_features), masks

class FSM_Ave(FSM):
    def __init__(self, feature_scores_all:Dict[str, np.ndarray], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_scores = feature_scores_all
    
    def feature_selection_all(self, inputs:Tuple):
        """
        Select features to be transmitted for each user
        Args:
            inputs: A Tuple of tensors, each is a real tensor of shape (batch, frames, d) where d is the feature dim. 
                In ave, assume modality order is [image, speech]
            m_ratios: A list of float number, each is the selection ratio for the corresponding modality
        Returns:
            A Tuple of tensors, each is a real tensor of shape (batch, frames, d) where d is the feature dim, after feature selection.
        """
        masked_features = []
        masks = []
        
        for i, (mod, m_ratio) in enumerate(zip(self.modalities, self.m_ratios)):
            masked, mask = self._apply_mask(inputs[i], self.feature_scores[mod], m_ratio=m_ratio)
            masked_features.append(masked)
            masks.append(mask)

        return tuple(masked_features), masks

class UplinkInference_FeatureSelection(UplinkInference):
    """
    A class that extends UplinkInference to support the system model with Feature Selection (FS).

    Specifically, this would "modify" the uplink code from the parent clas to do something like this:

    user 1 data -> encoder (MultiModalSC) -> FS Module (pre_process) -> |         | 
                                                                        | channel | -> decoder (MultiModalSC)
    user 2 data -> encoder (MultiModalSC) -> FS Module (pre_process) -> |         | 

    Note that:
    1. pre_process() := FS Module
    2. setup_overhead() -> test_uplink() -> [make features/signals] -> pre_process() -> uplink()
    """
    class OverrideChannel:
        def __init__(self, ui, original_channel=None):
            self.ui = ui
            self.original_channel = original_channel
        def interfere(self, signal: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            signal = self.ui.pre_process(signal)
            signal = self.original_channel.interfere(signal, *args, **kwargs)
            # signal = self.ui.post_process(signal)
            return signal
        def get_original_channel(self):
            return self.original_channel
        def __getattr__(self, name):
            return getattr(self.original_channel, name)
    
    def __init__(self, modalities:List[str], m_ratios:List[float], *args, **kwargs):
        """
        for the subclasses:
            1. please don't do too much setup in here, put the setup in setup_overhead() instead
            2. please store all args / kwargs with the same name in the class,
        """
        super().__init__(*args, **kwargs)
        # self.channel = self.OverrideChannel(self, self.channel)
        if len(modalities) != len(m_ratios):
            raise ValueError(f"number of modalities {len(modalities)} does not match number of m_ratios {len(m_ratios)}")
        
        self.modalities = modalities
        self.m_ratios = m_ratios
        
    def test_uplink(self, dataloader: DataLoader, method: UplinkInference.SupportedUplinkMethods, n_batch: int = 1):
        """
            test the uplink method
            Args:
                dataloader: dataloader with multiple modalities data
                            will yield tuple of data
                method: the method to use for uplink, ref. uplink()
                n_batch: number of batches to test
                
            Return:
                metrics: result of task performance()
        """

        self.setup_overhead(dataloader, method, n_batch)        
        return super().test_uplink(dataloader, method, n_batch)
    
    def uplink(self, inputs: Tuple[torch.Tensor], targets: torch.Tensor, method: UplinkInference.SupportedUplinkMethods):
        # just to make pre-processing and post-processing have access to the uplink arguments
        self.uplink_args = {
            'inputs': inputs, 
            'method': method, 
        }
        
        features = self.model.make_features(tuple(inputs))
        if method == 'signals':
            features = self.model.make_signals(features)
        
        masked_inputs = self.pre_process(features)
        return super().uplink(masked_inputs, targets, method)
    

    def setup_overhead(self, dataloader: DataLoader, method: UplinkInference.SupportedUplinkMethods, n_batch: int = 1):   
        """
        Corresponds to the setup overhead steps in the uplink process.
        Before test_uplink(), you can do some setup like how users and BS does before the multiple access process
        This will only be called once per test_uplink()

        Store the related setup parameter in this class (self) and use them in FS (pre_process and post_process) methods.

        Args:
            dataloader: A DataLoader consist all modalities data coresponding to the task
                (note that this is the dataloader that's passed to test_uplink(), so this is the test datasets!
                 if the base station wanna adopt some strategies that needs data, they shouldn't really have this,
                 this is only for if e.g., you wanna get the size of the image of the test dataset, etc)
            method: the uplink method to use, one of UplinkInference.SupportedUplinkMethods
            n_batch: number of batches to process
                (this is for test_uplink() btw)
            get_data_batch: the index of the data batch to get from the dataloader, default is 0
        Returns:
            None, but you can store the related setup parameter in this class (self) and use them in the FSM process
        """
        pass  # no setup overhead for now, but you can implement it if needed

    def pre_process(self, features):
        """
        Do pre-processing on the features, this is the FSM part of the uplink process.
        This will be called everytime the underlying uplink method uses channel.interfere()

        Args:
            features: input tensor, shape (batch_size, user_dim, *dim, feature_dim)
                
        Returns:
            features or signals: pre-processed features/signals that will be put on channel, same shape as input features
                the returned signals no need do power normalized, this will be done in the parent uplink method
                
                We do: pre_process() -> uplink()

        Note:
            if you need parameters of test_uplink, you can access them via self.uplink_args
            note that according to the uplink method implemented, you might need to keep some additional variables in here,

            e.g., since pn_repeat_splice method does 2 transmissions (1 for each user), you might need to keep track of the current user
            by self.next_user, which is initialized to 1 in __init__ method.

        """
        return features
    
    @classmethod
    def _get(cls, ui_spec, name):
        return ui_spec[name] if isinstance(ui_spec, dict) else getattr(ui_spec, name, None)
    
    @classmethod
    def _get_all(cls, ui_spec, *names):
        return [ui_spec[name] if isinstance(ui_spec, dict) else getattr(ui_spec, name, None)
                for name in names]

    @classmethod
    def ui_folder_name(cls, ui_spec: UplinkInference | dict) -> str:
        """
        Return the name for this UplinkInference instance, which will be used as a folder name later
        to store related results of the feature selection (FS) method implemented by this UI
        This should somewhat specify the general settings of the FS method... but also not too long because
        this will act as a folder name
        e.g., (for rotation) what block size is used, etc
        """
        return f'ui_fs'
    
    @classmethod
    def parse_ui_folder_name(self, folder_name: str) -> dict | None:
        """
        Parse the folder name
        if this matches with the folder name of this class, return a dict with the specific settings of this FS method
        if this folder name isn't generated by this class ('s ui_folder_name), return None
        """
        return {} if folder_name == 'ui_fs' else None
    

class UplinkInference_SHAPbasedSelection(UplinkInference_FeatureSelection):
        """
        Select features according to its shapley values magnitude (contributions)
        """
        def __init__(self, shap_val_path:Path, fsm_class:Callable=FSM, *args, **kwargs):
            """
            Args:
                shap_values: A dictionary containing SHAP values for each modality
            """
            super().__init__(*args, **kwargs)

            self.shap_val_path = Path(shap_val_path)
            self.fsm_class = fsm_class
            self.masks = []

        def setup_overhead(self, dataloader: DataLoader, method: UplinkInference.SupportedUplinkMethods, n_batch: int = 1):
            """
            Corresponds to the setup overhead steps in the uplink process.
            Before test_uplink(), you can do some setup like how users and BS does before the multiple access process
            This will only be called once per test_uplink()

            Store the related setup parameter in this class (self) and use them in FS (pre_process and post_process) methods.
            """
            from ..shap.shap_helper import read_shap_from_file
            data = read_shap_from_file(self.shap_val_path)
            shap_vals_mod = data['shap_mod'].item()
            data.close()
            
            self.fsm = self.fsm_class(shap_vals_mod, self.modalities, self.m_ratios)

        def pre_process(self, features):  
            masked_features, masks = self.fsm.feature_selection_all(features)   
            self.masks = masks
            return masked_features   
        
        def get_masks(self):
            return self.masks
        
        @classmethod
        def ui_folder_name(cls, ui_spec: UplinkInference | dict) -> str:
            mods = cls._get(ui_spec, 'modalities')
            return f'ui_fs_shap_{"".join([m.upper()[0] for m in mods])}'

        @classmethod
        def parse_ui_folder_name(cls, folder_name: str) -> dict | None:
            # use re to match
            pattern = r'ui_fs_shap(_(\d+))+'
            match = re.match(pattern, folder_name)
            # if match:
            #     m_ratios_str = match.group(1).split('_')[1:]  # skip the first empty string
            #     m_ratios = [float(mr) / 10 if len(mr) == 1 else float(mr[:-1] + '.' + mr[-1]) for mr in m_ratios_str]
            #     return {'m_ratios': m_ratios}
            
class UplinkInference_RandomSelection(UplinkInference_FeatureSelection):
        """
        Select features according to its shapley values magnitude (contributions)
        """
        def __init__(self, fsm_class:Callable=FSM, *args, **kwargs):
            """
            Args:
                shap_values: A dictionary containing SHAP values for each modality
            """
            super().__init__(*args, **kwargs)

            self.fsm_class = fsm_class
            self.masks = []

        def setup_overhead(self, dataloader: DataLoader, method: UplinkInference.SupportedUplinkMethods, n_batch: int = 1):
            self.fsm = self.fsm_class(self.modalities, self.m_ratios)

        def pre_process(self, features):  
            masked_features, masks = self.fsm.feature_selection_all(features)   
            self.masks = masks
            return masked_features   
        
        def _get_masks(self):
            return self.masks
        
        @classmethod
        def ui_folder_name(cls, ui_spec: UplinkInference | dict) -> str:
            mods = cls._get(ui_spec, 'modalities')
            return f'ui_fs_rand_{"".join([m.upper()[0] for m in mods])}'

        @classmethod
        def parse_ui_folder_name(cls, folder_name: str) -> dict | None:
            # use re to match
            pattern = r'ui_fs_rand_(+)'
            match = re.match(pattern, folder_name)
            # if match:
            #     m_ratios_str = match.group(1).split('_')[1:]  # skip the first empty string
            #     m_ratios = [float(mr) / 10 if len(mr) == 1 else float(mr[:-1] + '.' + mr[-1]) for mr in m_ratios_str]
            #     return {'m_ratios': m_ratios}

class UITestSuite:
    """
    Given a UplinkInference class, this class will test the class with different models and such

    we will use the following file tree:
    - cls.result_main_folder/
        - ui_fs/
            - ui_folder_name/uplink_method/model_name/result.pkl
        - result/
            - ui_folder_name/uplink_method/model_name/<PLOT_RELATED_FILES>
    """
    ue_cls = UplinkExperiment20250914
    result_main_folder = ue_cls.result_main_folder / 'ui_test'
    shap_file_path = Path('./tmp/20250915/kernel_shap_6000_features/awgn_12_udeepsc_msa_symbols_24_cmu-mosei/feat_contribs.npz')

    @classmethod
    def _get_stuff(cls, ue_cls: type[UplinkExperiment20250911], batch_size: int, model_info: dict | tuple, shuffle: bool, device: str):
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
            use_latest_checkpoint=False
        )
        sc = th.get_multimodal_sc()
        dataloader = th.get_dataloader(1, batch_size, False, shuffle=shuffle)

        return th, sc, dataloader

    @classmethod
    def _test(cls, ui_cls: UplinkInference, ui_kwargs: dict,
             ue_cls: UplinkExperiment20250911, model_infos: list[tuple | dict],
             uplink_method: UplinkInference.SupportedUplinkMethods, 
             n_user, snr_range: Optional[list[int]] = None,
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
            th, sc, dataloader = cls._get_stuff(ue_cls, batch_size, model_info, shuffle=True, device=device)
            
            th_ls.append(th)

        partial_ui_cls = functools.partial(ui_cls, **ui_kwargs)

        # check if the result file already exists
        result_path = (
            cls.result_main_folder / 
            ui_cls.ui_folder_name(ui_kwargs) / 
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
             n_user, snr_range: Optional[list[int]] = None,
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
            th, sc, dataloader = cls._get_stuff(ue_cls, batch_size, model_info, shuffle=True, device=device)
            
            th_ls.append(th)

        partial_ui_cls = functools.partial(ui_cls, **ui_kwargs)

        # check if the result file already exists
        result_path = (
            cls.result_main_folder / 
            ui_cls.ui_folder_name(ui_kwargs) / 
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
    def _test_feat_val(cls, ui_cls: UplinkInference, ui_kwargs: dict,
             ue_cls: UplinkExperiment20250911, model_info: tuple | dict,
             uplink_method: UplinkInference.SupportedUplinkMethods, snr_range: Optional[list[int]] = None,
             batch_size: int = 50, n_batch: int = 20, device: Optional[str] = 'cuda:0'
             ):
        """
        Test the feature value of given UplinkInference class with the given model info and method.

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
        """
        model_info = ue_cls._normalize_model_info(model_info, tuple)
    

        # get the necessary stuff
        th, sc, dataloader = cls._get_stuff(ue_cls, 1, model_info, shuffle=True, device=device)

        partial_ui_cls = functools.partial(ui_cls, **ui_kwargs)

        # check if the result file already exists
        result_path = (
            cls.result_main_folder / 
            ui_cls.ui_folder_name(ui_kwargs) / 
            uplink_method / 
            ue_cls._get_model_name(*model_info) /  
            'feat_val.pkl'
        )

        if result_path.exists():
            print(f"[*] Result file {result_path} already exists, skipping test.")
            return
        
        result_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Result Path: {result_path}')

        
        data = next(iter(dataloader))
        data = to_device(data, device)
        awgn_channel = AWGNMultiUplinkChannel(n_user=3, snr_db=[0], interfere_mode='all')
        ui = partial_ui_cls(model=sc, power_constraint=[1] * 3, channel=awgn_channel, device=th.device)
        
        ui.setup_overhead(dataloader, method)
        # encoder features (Tuple)
        features = ui.model.make_features(data)
        if method == 'signals':
            features = ui.model.make_signals(features)
        
        # Masked features (Tuple)
        masked_inputs = ui.pre_process(features)
        
        # Features importances (Dict)
        feature_scores = ui.fsm.feature_scores
        feature_scores = [v for k, v in feature_scores.items()]
        
        def save_values(values, path):
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(metrics, f)
                
        val_tuples = [tp for tp in zip(features, feature_scores, masked_inputs)]
        save_values(val_tuples, result_path)
         
        
    @classmethod
    def test(cls):
        model_info = ('awgn', 12, 'udeepsc_msa', 48, 'cmu-mosei', 1)
        method = 'features'
        snr_range = np.linspace(-6, 12, 10).tolist()
        
        config = getattr(cls, 'config', None)
        if config is None:
            raise RuntimeError("OptimizeTestSuit.config was not set. Set it before calling test().")
        
        mi = config['model_info']
        task = config["task"]
        n_user = len(config["modalities"])
        
        def noFSM():
            method = 'no_FSM'
            model_infos = [(mi["channel_type"], mi["snr_db"], f'udeepsc_{task}', mi["encoder_out_dims"], mi["dataset_type"], i) for i in range(1, config['model_nums']["nosic"] + 1)]
            # model_infos = [('awgn', 12, 'udeepsc_ave', 48, 'ave', i) for i in range(1, 2)]

            return cls._test(
                UplinkInference, 
                {'power_constraint': [1] * n_user},
                cls.ue_cls, model_infos, method, n_user=n_user, divide_gain=True,batch_size=30, n_batch=None, device='cuda:0', snr_range=snr_range
            )
        def noFSM_oma():
            method = 'no_FSM_oma'
            model_infos = [(mi["channel_type"], mi["snr_db"], f'udeepscOMA_{task}', mi["encoder_out_dims_OMA"], mi["dataset_type"], i) for i in range(1, config['model_nums']["oma"] + 1)]
            # model_infos = [('awgn', 12, 'udeepscOMA_ave', 32, 'ave', i) for i in range(1, 3)]
            
            return cls._test_oma(
                UplinkInference, 
                {'power_constraint': [1] * n_user},
                cls.ue_cls, model_infos, method, batch_size=30, n_user=n_user, n_batch=None, device='cuda:0', snr_range=snr_range
            )
        def noFSM_sic():
            method = 'no_FSM_sic'
            model_infos = [(mi["channel_type"], mi["snr_db"], f'udeepscSIC_{task}', mi["encoder_out_dims"], mi["dataset_type"], i) for i in range(1, config['model_nums']["sic"] + 1)]
            power_constraint = [1]
            if task == 'msa':
                power_constraint = [0.8, 0.4, 1.8]
            elif task == 'ave':
                power_constraint = [1.5, 0.5]
            else:
                raise ValueError(f"Unknown task {task} for SIC power constraint setting.")

            return cls._test(
                UplinkInference, 
                {'power_constraint': power_constraint}, 
                cls.ue_cls, model_infos, method, n_user=n_user, divide_gain=False, batch_size=30, n_batch=None, device='cuda:0', snr_range=snr_range
            )
        
        def shap_select():
            "SHAP based selection"
            return cls._test(
                UplinkInference_SHAPbasedSelection, 
                {
                    'shap_val_path': cls.shap_file_path, 
                    'modalities': ['text', 'image', 'speech'], 
                    'm_ratios': [0.5, 0.5, 0.5], 
                    'fsm_class': FSM_Msa
                 },
                cls.ue_cls, model_info, method, batch_size=10, n_batch=20, device='cuda:0', snr_range=snr_range
            )
            
        def random_select(): 
            "Random selection"
            # return [dp_delayed(cls._test)(
            #     RotationMethod.HeteroUplinkInference_RandomRotation, 
            #     {'rotation_block_size': 16, 'rotation_seed_setting': 'in_setup', 'rotation_seed': 42},
            #     cls.ue_cls,
            #     (channel_type, snr_db, mt1, eoc1, dataset_type, model_id), (channel_type, snr_db, mt2, eoc2, dataset_type, model_id),
            #     'pn_repeat_splice',batch_size=50, n_batch=20, model_commutative=False
            # ) \
            # for (channel_type, snr_db), (mt1, eoc1), (mt2, eoc2), dataset_type, model_id in test_settings]
            return cls._test(
                UplinkInference_SHAPbasedSelection, 
                {
                    'shap_val_path': cls.shap_file_path, 
                    'modalities': ['text', 'image', 'speech'], 
                    'm_ratios': [0.5, 0.5, 0.5], 
                    'fsm_class': FSM_Msa
                 },
                cls.ue_cls, model_info, method, batch_size=10, n_batch=20, device='cuda:0', snr_range=snr_range
            )
        def test_feat_val():
            return cls._test_feat_val(
                UplinkInference_SHAPbasedSelection, 
                {
                    'shap_val_path': cls.shap_file_path, 
                    'modalities': ['text', 'image', 'speech'], 
                    'm_ratios': [0.2, 0.2, 0.2], 
                    'fsm_class': FSM_Msa
                 },
                cls.ue_cls, model_info, method, batch_size=10, n_batch=20, device='cuda:0', snr_range=snr_range
            )
        # shap_select()
        # random_select()
        noFSM()
        noFSM_oma()
        noFSM_sic()
    
    @classmethod
    def _plot_value():
        pass
    
    @classmethod
    def plot(cls):
        pass
            
if __name__ == '__main__':
    UITestSuite.test()
    method = 'no_FSM_oma'
    model_id = 2
    task = 'msa'
    dataset = 'cmu-mosei' if task == 'msa' else 'ave'
    
    model_name = f"awgn_12_udeepsc_{task}_symbols_24_{dataset}"
    oma_model_name = f"awgn_12_udeepscOMA_{task}_symbols_16_{dataset}"
    sic_model_name = f"awgn_12_udeepscSIC_{task}_symbols_24_{dataset}"
    result_path = (
            UITestSuite.result_main_folder / 
            UplinkInference.ui_folder_name({}) / 
            method / oma_model_name
        )
    
    def load_results(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    metrics = load_results(result_path / 'result.pkl')
    print(metrics)
    
    plot_result_path = (
            UITestSuite.result_main_folder / 
            UplinkInference.ui_folder_name({}) / 
            method / oma_model_name
        )
    print(plot_result_path)
    plot_general(
        plot_result_path, 'Accuracy (%)', 'acc',
        [result_path / 'result.pkl'],
        [
            {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:red', 'label': 'UDeepSC NO (w/o SD)'},
        ], task_type=task
    )
    # y_limited=[55, 85, 7]