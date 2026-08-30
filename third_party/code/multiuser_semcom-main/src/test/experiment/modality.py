import functools
import scipy.interpolate
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from pathlib import Path
import torchvision
import numpy as np
from tqdm import tqdm
import re
import pickle
import matplotlib.pyplot as plt
import itertools
import shutil
import pandas as pd
from functools import partial
import scipy
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, SequentialLR

from ...log import get_logger
from ...channel import *
from ...model.component import power_normalize, signal_power_per_complex_symbol, signal_power
from ..udeepsc.cp_load import *
from ...trainer.trainer_udeepsc import UDeepSCNoSICTrainer_Msa, UDeepSCNoSICATrainer_AVE
from ...trainer.trainer import BaseTrainer, get_best_checkpoint
from ...dataset.udeepsc import MSA_dataset, make_udeepsc_msa_testdataloader, make_udeepsc_msa_dataloader
from ..shap.shap_helper  import KernelAccuracyWrapper
from ...channel import *
from ...utils import (
    str_type, tensor_complex2real, tensor_real2complex, to_device, calc_metrics, InfiniteDataLoader
)

def get_train_logger(save_dir: Path):
    import datetime
    time_str = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(save_dir) / f'multimod_{time_str}.ansi'
    logger = get_logger(f'multimod_{time_str}', str(log_path), stdout=True)
    return logger

def make_training_settings(model: torch.nn.Module, optimizer):
    criterion = torch.nn.MSELoss()
    lr_scheduler = SequentialLR(
        optimizer, schedulers=[
            ConstantLR(optimizer, factor=1.0,  total_iters=1000),
            ConstantLR(optimizer, factor=0.2,  total_iters=1000),
            ConstantLR(optimizer, factor=0.05, total_iters=1000)],
        milestones=[100, 200]
    )
    return criterion, lr_scheduler

class MultiModalSC(nn.Module):
    SupportedModalCombin = Literal['textspe','textimg', 'imgspe'] # Method for uplink superposition
    supported_modal_combin = ['textspe','textimg', 'imgspe']
    
    @staticmethod
    def _check_supported_types(tp, tp_list):
        """ check if the types are in the supported list, tp_list should be a list above """
        if tp not in tp_list:
            raise ValueError(f"Unsupported type: {tp} not in {tp_list}")
    
    def __init__(self):
        super().__init__()
        
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[Tuple[torch.Tensor], Any]:
        """
        Make features from inputs
        Args:
            inputs: inputs to make features, a tuple of tensor with shape (batch_size, *)
        Returns:
            features: Image, text, speech features made from inputs, must be tensor of shape (batch_size, # of features, feature_length)
            other_data: some other data that you need to pass to the receiver (e.g., how to reshape the features). 
                        this will be directly passed to transmitt_from_features()
        """
        raise NotImplementedError()
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        """
        Make signal from inputs
        Args:
            inputs: inputs to make signal, a tuple of tensor with shape (batch_size, *)
        Returns:
            signal: the signal made from features,  tensor of shape (batch_size, # of features, signal_length)
            other_data: some other data that you need to pass to the receiver (e.g., how to reshape the features). 
                        this will be directly passed to transmitt_from_features()
        """
        raise NotImplementedError()

    def decode_from_signal(self, signal: torch.Tensor, other_data: Any) -> torch.Tensor:
        """
        Do inference from interfered signal
        Args:
            signal: signal made from features, tensor of shape (batch_size, # of features, signal_length)
                    the features is the same as that returned from make_features
                    this signal may be interfered by the channel or other users
            other_data: from make_features()
        Returns:
            outputs: task result from features, tensor of shape (batch_size, *)
                                  this will be used in performance()
        """
        raise NotImplementedError()

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        """
        Calculate accuracy of the model prediction
        Args:
            targets: inputs to make signal, tensor of shape (batch_size, *)
            preds: model prediction of the task from features, tensor of shape (batch_size, *)
        Returns:
            accuracy: the accuracy of the task
        NOTE:
            for targets and preds:
                inputs , targets = next(iter(dataloader))
                preds = decode_from_signal(*make_signal(inputs))     # just an example

            do not do this:
                performance(reconstruct_data(inputs, reconstructed_inputs))
            this is not how this function is designed. you are expected to do this
                performance(inputs, reconstructed_inputs)

        """
        raise NotImplementedError()
    
class UDeepSC_MSA_MultiModalSC(MultiModalSC):
    """
        Class for UDeepSC no signal detection SemCom system on MSA task
        
        MSA task includes three modalities: text, image, speech
        
        Definition:
            - model: trained UDeepSC model by trainer for testing
            - modalCombin: modalities combination for preliminary experiment, e.g. 'textspe','textimg', 'imgspe', default is None (meaning use all modalities)
    """
    def __init__(self, model: nn.Module, modalCombin: MultiModalSC.SupportedModalCombin=None):
        super().__init__()
        
        MultiModalSC._check_supported_types(modalCombin, MultiModalSC.supported_modal_combin + [None])
        
        self.model = model
        self.ta_perform = 'msa'
        self.modalCombin = modalCombin
        
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        text, image, speech = inputs
        
        x_text = self.model.text_encoder(self.ta_perform, text, return_dict=False)[0]
        x_text = x_text[:,-2:-1,:]

        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token

        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_text, x_image, x_speech
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        x_text, x_image, x_speech = inputs
        x_text = self.model.msa_text_encoder_to_channel(x_text)
        x_image = self.model.msa_img_encoder_to_channel(x_image)
        x_speech = self.model.msa_spe_encoder_to_channel(x_speech)
        
        if self.modalCombin == 'textspe':
            return x_text, x_speech
        elif self.modalCombin == 'textimg':
            return x_text, x_image
        elif self.modalCombin == 'imgspe':
            return x_image, x_speech
        else:
            return x_text, x_image, x_speech

    def decode_from_signal(self, signal: torch.Tensor, other_data: Any) -> torch.Tensor:
        # print(signal.shape)
        batch_size = signal.shape[0]
        
        # check signal dim then 
        # pad signals to decoder input dim if dim is not consistant
        if signal.shape[-1] != self.model.num_symbols // 2:
            print(f'pad signal from {signal.shape[-1]} to {self.model.num_symbols}')
            sig_pad = F.pad(signal, (0, self.model.num_symbols), value=0)
        else:
            sig_pad = signal
        
        signal = tensor_complex2real(sig_pad, 'concat')

        x = self.model.msa_channel_to_decoder(signal)
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out
        

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        from ...utils import calc_metrics
        return calc_metrics(preds, targets)
    
class UDeepSCOMA_MSA_MultiModalSC(MultiModalSC):
    def __init__(self, model: nn.Module, modalCombin: MultiModalSC.SupportedModalCombin=None):
        super().__init__()
        
        MultiModalSC._check_supported_types(modalCombin, MultiModalSC.supported_modal_combin + [None])
        
        self.model = model
        self.ta_perform = 'msa'
        self.modalCombin = modalCombin
        
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        text, image, speech = inputs
        
        x_text = self.model.text_encoder(self.ta_perform, text, return_dict=False)[0]
        x_text = x_text[:,-2:-1,:]

        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token

        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_text, x_image, x_speech
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        x_text, x_image, x_speech = inputs
        x_text = self.model.text_encoder_to_channel(x_text)
        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        return x_text, x_image, x_speech

    def decode_from_signal(self, signals: Tuple[torch.Tensor], other_data: Any) -> torch.Tensor:
        
        # check signal dim then 
        # pad signals to decoder input dim if dim is not consistant
        
        x_text, x_image, x_spe = signals
        
        x_text = self.model.text_channel_decoder(x_text)
        x_text = self.model.text_channel_to_decoder(x_text)

        x_image = self.model.img_channel_decoder(x_image)
        x_image = self.model.img_channel_to_decoder(x_image)

        x_spe = self.model.spe_channel_decoder(x_spe)
        x_spe = self.model.spe_channel_to_decoder(x_spe)
        
        if self.modalCombin == 'textspe':
            x = torch.cat([x_text, x_spe], dim=1)
        elif self.modalCombin == 'textimg':
            x = torch.cat([x_image, x_text], dim=1)
        elif self.modalCombin == 'imgspe':
            x = torch.cat([x_image, x_spe], dim=1)
        else:
            x = torch.cat([x_image, x_text, x_spe], dim=1)
        
        batch_size = x.shape[0]
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out
        

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        from ...utils import calc_metrics
        return calc_metrics(preds, targets)
    
class UDeepSCSIC_MSA_MultiModalSC(MultiModalSC):
    def __init__(self, model: nn.Module, channel_type, modalCombin: MultiModalSC.SupportedModalCombin=None):
        super().__init__()
        
        MultiModalSC._check_supported_types(modalCombin, MultiModalSC.supported_modal_combin + [None])
        
        self.model = model
        self.channel_type = channel_type
        self.ta_perform = 'msa'
        self.modalCombin = modalCombin
    
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        text, image, speech = inputs
        
        x_text = self.model.text_encoder(self.ta_perform, text, return_dict=False)[0]
        x_text = x_text[:,-2:-1,:]

        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token

        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_text, x_image, x_speech
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        x_text, x_image, x_speech = inputs
        x_text = self.model.text_encoder_to_channel(x_text)
        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        if self.modalCombin == 'textspe':
            return x_text, x_speech
        elif self.modalCombin == 'textimg':
            return x_text, x_image
        elif self.modalCombin == 'imgspe':
            return x_image, x_speech
        else:
            return x_text, x_image, x_speech
    
    def SIC(self, signal: torch.Tensor, user_dim_index: int, power_constraints: torch.FloatTensor, h=None):
            """            
                Args
                    signal: real tensor in (batch_size, 1, *dim, symbol_dim)
                    power_constraint: the power constraint for users, length n_user (The order is [text, img, speech])
                    h: channel gain (Rayleigh or Rician)
                Return
                    a list of decode signals. The order is [Text, Image, Speech]
                    
                Note: 
                    The channel_encoders and channel_decoders order are [Text, Image, Speech], which correspond to the order in the return list
            """
            if self.modalCombin is not None:
                channel_encoders = []
                channel_decoders = []
                
                if 'text' in self.modalCombin:
                    channel_encoders.append(self.model.text_encoder_to_channel)
                    channel_decoders.append(self.model.text_channel_decoder)
                if 'img' in self.modalCombin:
                    channel_encoders.append(self.model.img_encoder_to_channel)
                    channel_decoders.append(self.model.img_channel_decoder)
                if 'spe' in self.modalCombin:
                    channel_encoders.append(self.model.spe_encoder_to_channel)
                    channel_decoders.append(self.model.spe_channel_decoder)        
            else:
                channel_encoders = [self.model.text_encoder_to_channel, self.model.img_encoder_to_channel, self.model.spe_encoder_to_channel]
                channel_decoders = [self.model.text_channel_decoder, self.model.img_channel_decoder, self.model.spe_channel_decoder]

            batch_size = signal.size()[0]
            num_users = len(power_constraints)
            # print(f"{num_users= }")
            if(num_users == 1):
                estimated = signal
                estimated = channel_decoders[0](estimated)
                return [estimated]
            
            if self.channel_type == "awgn":
                # Sort users by transmit power (descending order)
                user_indices = torch.argsort(power_constraints, dim=-1).detach().to('cpu').numpy()
            else: # Rayleigh or Rician
                if h is None:
                    raise ValueError("Channel gains (h) must be provided for Rayleigh channel.")
                # Compute effective received power |h_i|^2 * P_i
                # effective_power = torch.abs(h).detach()**2 * power_constraints
                user_indices = torch.argsort(power_constraints, dim=-1).detach().to('cpu').numpy()
                signal = tensor_real2complex(signal, 'concat')
                
            # make power constraint
            power_constraints = power_constraints.clone().detach().to(signal.device)  # (user,)
            power_constraints = power_constraints.view(1, num_users, 1)
            power_constraints = power_constraints.expand(batch_size, num_users, 1)


            decoded_signals = [None] * num_users

            for i in user_indices:
                """
                    Decoding steps:
                        1. Use channel decoder to decode stronger signal
                        
                        2. Channel encoding (same as transmitter) the signal to simulate the signal state of the transmitter
                        
                        3. Subtract encoded estimated signal from Y
                        
                        4. Repeat until all signals be detected
                """
                estimated = signal

                if self.channel_type == "rayleigh":
                    estimated = estimated / (h[:, i, 0] + 1e-10)
                    estimated = tensor_complex2real(estimated, 'concat')

                estimated = channel_decoders[i](estimated)
                decoded_signals[i] = estimated
                
                with torch.no_grad():
                    estimated = channel_encoders[i](estimated)

                # n_f, n_len = estimated.size()
                # estimated = estimated.flatten(1) 
                estimated_norm = power_normalize(estimated, power_constraints[:, i])

                if self.channel_type == "awgn":
                    signal = signal - estimated
                else: # Fading channel (Rayleigh or Rician)
                    # print(f'{estimated_norm.shape=}')
                    estimated_norm = tensor_real2complex(estimated_norm, 'concat')
                    signal = signal - h[:, i, 0] * estimated_norm

            return decoded_signals

    def decode_from_signal(self, signals: Tuple[torch.Tensor], other_data: Any) -> torch.Tensor:        
        
        if self.modalCombin == 'textspe':
            x_text, x_spe = signals
            
            x_text = self.model.text_channel_to_decoder(x_text)
            x_spe = self.model.spe_channel_to_decoder(x_spe)
            x = torch.cat([x_text, x_spe], dim=1)
        elif self.modalCombin == 'textimg':
            x_text, x_image = signals
            
            x_text = self.model.text_channel_to_decoder(x_text)
            x_image = self.model.img_channel_to_decoder(x_image)
            x = torch.cat([x_image, x_text], dim=1)
        elif self.modalCombin == 'imgspe':
            x_image, x_spe = signals
            
            x_image = self.model.img_channel_to_decoder(x_image)
            x_spe = self.model.spe_channel_to_decoder(x_spe)
            x = torch.cat([x_image, x_spe], dim=1)
        else:
            # Transmit and decode all modalities
            x_text, x_image, x_spe = signals
            
            x_text = self.model.text_channel_to_decoder(x_text)
            x_image = self.model.img_channel_to_decoder(x_image)
            x_spe = self.model.spe_channel_to_decoder(x_spe)

            x = torch.cat([x_image, x_text, x_spe], dim=1)
        
        batch_size = x.shape[0]
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out
        

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        from ...utils import calc_metrics
        return calc_metrics(preds, targets)
    
class UDeepSC_AVE_MultiModalSC(MultiModalSC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.ta_perform = 'ave'
        
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        image, speech = inputs

        self.AVE_batch_size = image.shape[0]
        image = image.view(image.size(0) * image.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token
        
        speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)
        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_image, x_speech
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        x_image, x_speech = inputs
        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        return x_image, x_speech

    def decode_from_signal(self, signal: torch.Tensor, other_data: Any) -> torch.Tensor:        
        batch_size, _, _ = signal.shape
        
        # check signal dim then 
        # pad signals to decoder input dim if dim is not consistant
        if signal.shape[-1] != self.model.num_symbols // 2:
            print(f'pad signal from {signal.shape[-1]} to {self.model.num_symbols}')
            sig_pad = F.pad(signal, (0, self.model.num_symbols), value=0)
        else:
            sig_pad = signal
        
        signal = tensor_complex2real(sig_pad, 'concat')

        x = self.model.channel_to_decoder(signal)
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out
        

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        from ...utils import compute_acc_AVE
        return compute_acc_AVE(preds, targets)
    
class UDeepSCOMA_AVE_MultiModalSC(MultiModalSC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.ta_perform = 'ave'
        
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        image, speech = inputs

        self.AVE_batch_size = image.shape[0]
        image = image.view(image.size(0) * image.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token
        
        speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)
        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_image, x_speech
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        x_image, x_speech = inputs
        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        return x_image, x_speech

    def decode_from_signal(self, signal: Tuple[torch.Tensor], other_data: Any) -> torch.Tensor:
        
        x_image, x_spe = signal
        
        x_image = self.model.img_channel_decoder(x_image)
        x_image = self.model.img_channel_to_decoder(x_image)

        x_spe = self.model.spe_channel_decoder(x_spe)
        x_spe = self.model.spe_channel_to_decoder(x_spe)
        
        x = torch.cat([x_image, x_spe], dim=1)
        
        batch_size = x.shape[0]
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))
        out = F.softmax(out, dim=-1)
        out = out.view(self.AVE_batch_size, -1, out.size(-1))

        return out
        

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        from ...utils import compute_acc_AVE
        return compute_acc_AVE(preds, targets)
    
class UDeepSCSIC_AVE_MultiModalSC(MultiModalSC):
    def __init__(self, model: nn.Module, channel_type):
        super().__init__()
        self.model = model
        self.channel_type = channel_type
        self.ta_perform = 'ave'
    
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        image, speech = inputs

        self.AVE_batch_size = image.shape[0]
        image = image.view(image.size(0) * image.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token
        
        speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)
        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_image, x_speech
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        x_image, x_speech = inputs
        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        return x_image, x_speech
    
    def SIC(self, signal: torch.Tensor, user_dim_index: int, power_constraints: torch.FloatTensor, 
            h=None):
            """            
                Args
                    signal: real tensor in (batch_size, 1, *dim, symbol_dim)
                    power_constraint: the power constraint for users, length n_user (The order is [text, img, speech])
                    h: channel gain (Rayleigh or Rician)
                Return
                    a list of decode signals. The order is [Text, Image, Speech]
                    
                Note: 
                    The channel_encoders and channel_decoders order are [Text, Image, Speech], which correspond to the order in the return list
            """
            
            channel_encoders = [self.model.img_encoder_to_channel, self.model.spe_encoder_to_channel]
            channel_decoders = [self.model.img_channel_decoder, self.model.spe_channel_decoder]

            num_users = len(power_constraints)
            # print(f"{num_users= }")
            if(num_users == 1):
                estimated = signal
                estimated = channel_decoders[0](estimated)
                return [estimated]
            
            if self.channel_type == "awgn":
                # Sort users by transmit power (descending order)
                user_indices = torch.argsort(power_constraints, dim=-1).detach().to('cpu').numpy()
            else: # Rayleigh or Rician
                if h is None:
                    raise ValueError("Channel gains (h) must be provided for Rayleigh channel.")
                # Compute effective received power |h_i|^2 * P_i
                # effective_power = torch.abs(h).detach()**2 * power_constraints
                user_indices = torch.argsort(power_constraints, dim=-1).detach().to('cpu').numpy()
                signal = tensor_real2complex(signal, 'concat')


            decoded_signals = [None] * num_users

            for i in user_indices:
                """
                    Decoding steps:
                        1. Use channel decoder to decode stronger signal
                        
                        2. Channel encoding (same as transmitter) the signal to simulate the signal state of the transmitter
                        
                        3. Subtract encoded estimated signal from Y
                        
                        4. Repeat until all signals be detected
                """
                estimated = signal

                if self.channel_type == "rayleigh":
                    estimated = estimated / h[:, i, 0]
                    estimated = tensor_complex2real(estimated, 'concat')

                estimated = channel_decoders[i](estimated)
                decoded_signals[i] = estimated
                
                with torch.no_grad():
                    estimated = channel_encoders[i](estimated)

                # n_f, n_len = estimated.size()
                # estimated = estimated.flatten(1) 
                # estimated_norm = power_normalize(estimated, power_constraints[i])

                if self.channel_type == "awgn":
                    signal = signal - estimated
                else: # Fading channel (Rayleigh or Rician)
                    # print(f'{estimated_norm.shape=}')
                    estimated = tensor_real2complex(estimated, 'concat')
                    signal = signal - h[:, i, 0] * estimated

            return decoded_signals

    def decode_from_signal(self, signals: torch.Tensor, other_data: Any) -> torch.Tensor:
        
        x_image, x_spe = signals
        x_image = self.model.img_channel_to_decoder(x_image)
        x_spe = self.model.spe_channel_to_decoder(x_spe)
        
        x = torch.cat([x_image, x_spe], dim=1)
        
        batch_size = x.shape[0]
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))
        # out = F.softmax(out, dim=-1)
        out = out.view(self.AVE_batch_size, -1, out.size(-1))

        return out
        

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        from ...utils import compute_acc_AVE
        return compute_acc_AVE(preds, targets)
    
    
class CIF_MultiModalSC(MultiModalSC):
    """
    # TODO: to be modified
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def make_features(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        text, image, speech = inputs
        
        x_text = self.model.text_encoder(self.ta_perform, text, return_dict=False)[0]
        x_text = x_text[:,:-1,:]

        x_image = self.model.img_encoder(image, self.ta_perform)

        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        
        return x_text, x_image, x_speech
    
    def make_signals(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        text, image, speech = inputs
        x_text = self.model.msa_text_encoder_to_channel(x_text)
        x_image = self.model.msa_img_encoder_to_channel(x_image)
        x_speech = self.model.msa_spe_encoder_to_channel(x_speech)
        
        return x_text, x_image, x_speech

    def decode_from_signal(self, signal: torch.Tensor, other_data: Any) -> torch.Tensor:
        batch_size, n_frames, f_dim = signal.shape[0]
        
        # check signal dim then 
        # pad signals to decoder input dim if dim is not consistant
        if signal.shape[-1] != self.model.num_symbols:
            sig_pad = F.pad(signal, (0, self.model.num_symbols), value=0)
        else:
            sig_pad = signal
        
        signal = tensor_complex2real(sig_pad, 'concat')

        x = self.model.msa_channel_to_decoder(signal)
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out
        

    def performance(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        from ...utils import calc_metrics
        return calc_metrics(preds, targets)
    
    
class UplinkInference:
    """
        Does inference, simulates uplink transmission
        Each user has its own encoder, generating its own signal from its own data
        (note that the signal length may be different for each user)
        The channel is shared, so the signal from each user is added together
        The receiver receives the signal decodes it with a unified decoder,
        and calculates the task performance

        The channel, and how the signal is superimposed exactly is implemented in
        this class
    """

    SupportedUplinkMethods = Literal['no_FSM','no_FSM_oma', 'no_FSM_sic', 'features', 'signals'] # Method for uplink superposition

    def __init__(self, model: MultiModalSC, power_constraint: list[float], 
                 channel: MultiUplinkChannel, device: torch.device):
        """
        Args:
            model: MultiModalSC model
            power_constraint: list of power constraints, each power constraint is for each user
                              this specifies the average power per complex signal symbol, ref. power_normalize()
            channel: the channel used for uplink transmission
                     divide_gain=True
            device: the device to use for the EDPs

            Please make sure channel is a MultiUplinkChannel that supports this amount of EDPs
        """
        self.model = model
        self.power_constraint = power_constraint
        self.channel = channel
        self.channel_type = channel.get_channel_type()
        
        self.total_power = sum(power_constraint)
        
        self.channel_gain = None
        self.device = device
        
        self.model.to(device)
        

    def uplink(self, inputs: Tuple[torch.Tensor], targets: torch.Tensor, method: SupportedUplinkMethods):
        """
            A wrapper for the uplink methods

            Args:
                inputs: tuple of inputs to each user, each input is a tensor of shape (batch, *)
                        each input should have the same batch size
                targets: the ground truth tensor of shape (batch, *)
                method: the method to use for uplink, ref. SupportedUplinkMethods
            
            NOTE: uplink should implement the same interface as this, except the `method` argument
        """
        method_ls = {
            'no_FSM': self._uplink_from_start,
            'no_FSM_oma': self._uplink_oma,
            'no_FSM_sic': self._uplink_SIC,
            'features': self._uplink_from_features,
            'signals': self._uplink_from_signals,
            # 'repeat_splice': self._uplink_repeat_splice,
        }
        if method not in method_ls:
            raise ValueError(f"method {method} not in {method_ls.keys()}")
        return method_ls[method](inputs, targets)

    def test_uplink(self, dataloader: DataLoader, method: SupportedUplinkMethods, n_batch: int = None):
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

        self.model.to(self.device)
        self.model.eval()

        total_acc = 0.0
        n_batch = len(dataloader) if n_batch is None else n_batch
        with torch.no_grad():
            for i, (inputs, targets) in tqdm(zip(range(n_batch), dataloader), desc='test uplink', total=n_batch, leave=False):
                inputs = to_device(inputs, self.device)
                targets = targets.to(self.device)
                
                cur_acc = self.uplink(inputs, targets, method)
                # metrics_ls.append(cur_acc)
                total_acc += float(cur_acc)
              
        avg_acc = float(total_acc / n_batch) * 100
        return avg_acc

    def _uplink_from_start(self, inputs: list[torch.Tensor], targets: torch.Tensor):
        """
        Do uplink transmission
        The signals from each users will be superimposed together at the front.
        Or, equivalently, the signals from each users will be padded with zeros up until they reach the maximum length of all signals (as we won't do any pruning or truncation to the signals, this may not used).

        This implementation will treat each batch independently, and the signals for each batch will be superimposed together at the front.
        e.g., the inputs all have batch size 10, USER1 generates signal of size (10, 250), USER2 generates 
              signal of size (10, 300), 
              Note that the maximum length of the signals is max(250, 300) = 300, so the superimposed signal
              will be of size (10, 300)
              - superimposed_signal[:, :250] = USER1_signal[:, :250] + USER2_signal[:, :250]
              - superimposed_signal[:, 250:] = USER2_signal[:, 250:]

              USER1's signals are padded with zeros so that they reach 300 length
        
        """
        signal_ls = []
        other_data_ls = []
        features = self.model.make_features(tuple(inputs))
        signals = self.model.make_signals(features)
        
        for signal, power_constraint in zip(signals, self.power_constraint):
            signal = tensor_real2complex(signal, 'concat')
            signal = power_normalize(signal, power_constraint)
            signal_ls.append(signal)

        # superimpose the signals by just adding them together
        # we pad all signals to the same length
        # (actually can use torch.nn.functional.pad constant mode, but i didn't know that and don't wanna change this)
        # signal_lengths = [s.size(1) for s in signal_ls]
        # max_signal_length = max(signal_lengths)
        # batch_size = signal_ls[0].size(0)
        # for i in range(len(signal_ls)):
        #     padded_signal = torch.zeros(batch_size, max_signal_length, dtype=torch.complex64, device=signal_ls[i].device)
        #     padded_signal[:, :signal_lengths[i]] = signal_ls[i]
        #     signal_ls[i] = padded_signal
        
        # transmit the signal
        signals = torch.stack(signal_ls, dim=1)
        signals = self.channel.interfere(signals, 1)
        if self.channel_type == "rayleigh":
            signals = signals.squeeze(dim=1)
        
        # decode
        preds = self.model.decode_from_signal(signals, None)
        acc = self.model.performance(preds, targets)

        return acc
    
    def _uplink_from_features(self, features: Tuple[torch.Tensor], targets: torch.Tensor):
        """
            uplink from features
            Args:
                features: tuple of features from semantic encoder or FSM, each feature is a tensor of shape (batch, # features, feature_dim)
                targets: the target tensor of shape (batch, *)
                method: the method to use for uplink, ref. uplink()
            Return:
                metrics: result of task performance()
        """
        signal_ls = []
        other_data_ls = []
        signals = self.model.make_signals(features)
        
        for signal, power_constraint in zip(signals, self.power_constraint):
            signal = tensor_real2complex(signal, 'concat')
            signal = power_normalize(signal, power_constraint)
            signal_ls.append(signal)
        
        # transmit the signal
        signals = torch.stack(signal_ls, dim=1)
        signals = self.channel.interfere(signals, 1)
        if self.channel_type == "rayleigh":
            signals = signals.squeeze(dim=1)
        
        # decode
        preds = self.model.decode_from_signal(signals, None)
        acc = self.model.performance(preds, targets)

        return acc
    
    def _uplink_from_signals(self, signals: Tuple[torch.Tensor], targets: torch.Tensor):
        """
            uplink from signals
            Args:
                signals: tuple of signals from JSCC or FSM, each signal is a tensor of shape (batch, signal_length)
                targets: the target tensor of shape (batch, *)
                method: the method to use for uplink, ref. uplink()
            Return:
                metrics: result of task performance()
        """
        signal_ls = []
        other_data_ls = []
        
        for signal, power_constraint in zip(signals, self.power_constraint):
            signal = tensor_real2complex(signal, 'concat')
            signal = power_normalize(signal, power_constraint)
            signal_ls.append(signal)
        
        # transmit the signal
        signals = torch.stack(signal_ls, dim=1)
        signals = self.channel.interfere(signals, 1)
        
        # decode
        preds = self.model.decode_from_signal(signals, None)
        acc = calc_metrics(preds, targets)

        return acc

    def _uplink_oma(self, inputs: list[torch.Tensor], targets: torch.Tensor):
        """
        Do uplink transmission
        The signals from each users will be transmitted independently through different channel.
        
        """
        signal_ls = []
        other_data_ls = []
        features = self.model.make_features(tuple(inputs))
        signals = self.model.make_signals(features)
        
        for signal, power_constraint in zip(signals, self.power_constraint):
            signal = tensor_real2complex(signal, 'concat')
            signal = power_normalize(signal, power_constraint)
            signal = self.channel.interfere(signal)
            signal = tensor_complex2real(signal, 'concat')
            signal_ls.append(signal)
        
        # decode
        preds = self.model.decode_from_signal(tuple(signal_ls), None)
        acc = self.model.performance(preds, targets)

        return acc
    def _uplink_SIC(self, inputs: list[torch.Tensor], targets: torch.Tensor):
        """
        Do non-thogonal uplink transmission
        The signals from each users will be superimposed together.
        The receiver will seprate signal of each user by semantic SIC
        
        """
        signal_ls = []
        other_data_ls = []
        features = self.model.make_features(tuple(inputs))
        signals = self.model.make_signals(features)
        
        
        if self.channel_type == "rayleigh":
            signal_st = torch.stack(signals, dim=1)
            comp_sig = signal_st.clone().detach().to(self.device)
            comp_sig = tensor_real2complex(comp_sig, 'concat')
            power_constraint = self.channel.get_signal_power_constraint(comp_sig, self.total_power, 1).clone().detach().to('cpu')
            self.power_constraint = power_constraint.numpy().tolist()
        
        for signal, power_constraint in zip(signals, self.power_constraint):
            signal = tensor_real2complex(signal, 'concat')
            signal = power_normalize(signal, power_constraint)
            signal_ls.append(signal)
            
        # transmit the signal
        signals = torch.stack(signal_ls, dim=1)
        self.channel_gain = self.channel.get_channel_gain().to(self.device) if self.channel_type == "rayleigh" else None
        signals = self.channel.interfere(signals, 1, channel_gain_tensor=self.channel_gain)
        signals = tensor_complex2real(signals, 'concat')
        
        # Do signal detection
        decoded_sig = self.model.SIC(signals, 1, torch.FloatTensor(self.power_constraint), h=self.channel_gain)
        
        # decode
        preds = self.model.decode_from_signal(tuple(decoded_sig), None)
        acc = self.model.performance(preds, targets)

        return acc
    
    @classmethod
    def _get(cls, ui_spec, name):
        return ui_spec[name] if isinstance(ui_spec, dict) else getattr(ui_spec, name, None)
    
    @classmethod
    def _get_all(cls, ui_spec, *names):
        return [ui_spec[name] if isinstance(ui_spec, dict) else getattr(ui_spec, name, None)
                for name in names]

    @classmethod
    def ui_folder_name(cls, ui_spec: dict) -> str:
        """
        Return the name for this UplinkInference instance, which will be used as a folder name later
        to store related results of the feature selection (FS) method implemented by this UI
        This should somewhat specify the general settings of the FS method... but also not too long because
        this will act as a folder name
        e.g., (for rotation) what block size is used, etc
        """
        return f'ui'
    
    @classmethod
    def parse_ui_folder_name(self, folder_name: str) -> dict | None:
        """
        Parse the folder name
        if this matches with the folder name of this class, return a dict with the specific settings of this FS method
        if this folder name isn't generated by this class ('s ui_folder_name), return None
        """
        return {} if folder_name == 'ui' else None
    
class TestHelper:
    """
        help doing modality related experiment
        can be passed to the below functions to do and plot modality related experiments
    """
    def __init__(self, device: torch.device):
        """
            device: for all EDPs
        """
        self.device = device
    def get_dataloader(self, n_user: int, batch_size: int, is_multiuser: bool, *args, **kwargs) -> DataLoader:
        """
            get test dataloader, args and kwargs will be passed to the dataloader
        """
        pass
    def get_val_dataloader(self, n_user: int, batch_size: int, is_multiuser: bool, *args, **kwargs) -> DataLoader:
        """
            get validation dataloader, args and kwargs will be passed to the dataloader
        """
        pass
    def get_multimodal_sc(self) -> MultiModalSC:
        """
            get MultiModalSC model
        """
        pass
    
def test_general(
        th_ls: list[TestHelper],
        result_path: Path,
        uplink_method: UplinkInference.SupportedUplinkMethods = 'features',
        test_datatype: Literal['test', 'val'] = 'test',
        batch_size: int = 20, n_batch: int = None,
        snr_range: list[int] = np.linspace(-5, 25, 13).tolist(),
        ui_cls: Callable = UplinkInference,
        n_user: int = 3,
        channel_type: str = None,
        divide_gain: bool=True,
        save_results: bool = True
        ):
    """

        Args:
            th: TestHelper
            result_path: the path to save the metrics, in pickle format
            uplink_method: the uplink method for UplinkInference, default is 'features'
            batch_size: the batch size for dataloader
            n_batch: the number of batch for the tests
            snr_range: the channel SNR range to test, in dB, should be increasing
            ui_cls: the class to use for UplinkInference, default is UplinkInference
                (technically we only use this as a constructor, so you can pass a factory function in here or something
                 we will pass in arguments ['edps', 'power_constraint', 'channel', 'device'])
        Returns:
            None, but the metrics will be saved to result_path in pickle format
            the metrics will be indexed like this:
                metrics[snr][channel][user][metric_type] = list of metric_values for
    """
    
    n_th = len(th_ls)
    
    if test_datatype == 'test':
        th_dataloader = th_ls[0].get_dataloader(1, batch_size, True, shuffle=False)
    elif test_datatype == 'val':
        th_dataloader = th_ls[0].get_val_dataloader(1, batch_size, False, shuffle=True)
    else:
        raise ValueError(f"Unknown test_datatype {test_datatype}, should be 'test' or 'val'")
    
    # th_dataloader = InfiniteDataLoader(th_dataloader)
    
    def avg_metric(metrics: list[dict]) -> dict:
        """
        Average the results of multiple models with different seeds
            Args:
                metrics: list of metric, e.g., from test_all_snr
                and is indexed like this
                metrics[i][snr][channel] = accuracy
        """
        
        n = len(metrics)
        snr_range = list(sorted(metrics[0].keys()))
        
        stat = {
            snr:{
                channel: sum([mt[snr][channel] for mt in metrics]) / n
                for channel in metrics[0][snr].keys()
            }
            for snr in snr_range
        }
        
        return stat

    def test_snr(n_user, dataloader, n_batch, th, snr_db, ch_type):
        """
            test the snr for the given edp and dataloader
            n_user: number of users
            edp_ls: list of edp
            dataloader: a multiuser dataloader
            snr: snr in dB
        """

        ret = {}
        
        if ch_type == 'awgn':
            awgn_channel = AWGNMultiUplinkChannel(n_user=n_user, snr_db=[snr_db], interfere_mode='all')
            th.channel_type = 'awgn'
            sc = th.get_multimodal_sc()
            ui = ui_cls(model=sc, channel=awgn_channel, device=th.device)
            ret['awgn'] = ui.test_uplink(dataloader, uplink_method, n_batch=n_batch)
        elif ch_type == 'rayleigh':
            rayleigh_channel = RayleighFadingMultiUplinkChannel(
                n_user=n_user, snr_db=[snr_db], channel_gain_var=[1], 
                divide_gain=divide_gain,
                noise_power_density_dBm=-90,    # ref. ISSNOMATrainer's note
                reference_distance=1,
                reference_path_loss=pow(10, -30/10),
                path_loss_exponent=4,
                distance=torch.Tensor([33, 83, 133]).reshape(3, 1),
                fading_mode='slow'
                )
            th.channel_type = 'rayleigh'
            sc = th.get_multimodal_sc()
            ui = ui_cls(model=sc, channel=rayleigh_channel, device=th.device)
            ret['rayleigh'] = ui.test_uplink(dataloader, uplink_method, n_batch=n_batch)
        else:
            raise ValueError(f"Unknown channel type{ch_type}, should be 'awgn' or 'rayleigh")

        return ret

    def test_all_snr(*args, **kwargs):
        acc_ls = []
        metric_ls = []
        for th in th_ls:
            # th_sc = th.get_multimodal_sc()
            metric = {}
            for snr in tqdm(snr_range, desc="Testing SNRs", leave=False):
                acc = test_snr(*args, **kwargs, th=th, snr_db=snr)
                # acc_ls.append(acc)
                metric[snr] = acc
            
            metric_ls.append(metric)
        
        metrics = avg_metric(metric_ls)
        
        return metrics

    def save_metrics(metrics, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(metrics, f)

    metrics = test_all_snr(n_user=n_user, dataloader=th_dataloader, n_batch=n_batch, ch_type=channel_type)
    
    if save_results:
        save_metrics(metrics, result_path)
    
    return metrics

def test_OMA_case(
        th_ls: list[TestHelper],
        result_path: Path,
        uplink_method: UplinkInference.SupportedUplinkMethods = 'no_FSM_oma',
        test_datatype: Literal['test', 'val'] = 'test',
        batch_size: int = 20, n_batch: int = 50,
        snr_range: list[int] = np.linspace(-5, 25, 13).tolist(),
        ui_cls: Callable = UplinkInference,
        save_results: bool = True
        ):
    """

        Args:
            th: TestHelper
            result_path: the path to save the metrics, in pickle format
            uplink_method: the uplink method for UplinkInference, default is 'no_FSM_oma'
            batch_size: the batch size for dataloader
            n_batch: the number of batch for the tests
            snr_range: the channel SNR range to test, in dB, should be increasing
            ui_cls: the class to use for UplinkInference, default is UplinkInference
                (technically we only use this as a constructor, so you can pass a factory function in here or something
                 we will pass in arguments ['edps', 'power_constraint', 'channel', 'device'])
        Returns:
            None, but the metrics will be saved to result_path in pickle format
            the metrics will be indexed like this:
                metrics[snr][channel][user][metric_type] = list of metric_values for
    """
    # make the MultimodalSC
    n_th = len(th_ls)
    
    if test_datatype == 'test':
        th_dataloader = th_ls[0].get_dataloader(1, batch_size, True, shuffle=False)
    elif test_datatype == 'val':
        th_dataloader = th_ls[0].get_val_dataloader(1, batch_size, False, shuffle=True)
    else:
        raise ValueError(f"Unknown test_datatype {test_datatype}, should be 'test' or 'val'")
    
    def avg_metric(metrics: list[dict]) -> dict:
        """
        Average the results of multiple models with different seeds
            Args:
                metrics: list of metric, e.g., from test_all_snr
                and is indexed like this
                metrics[i][snr][channel] = accuracy
        """
        
        n = len(metrics)
        snr_range = list(sorted(metrics[0].keys()))
        
        stat = {
            snr:{
                channel: sum([mt[snr][channel] for mt in metrics]) / n
                for channel in metrics[0][snr].keys()
            }
            for snr in snr_range
        }
        
        return stat

    def test_snr(n_user, sc, dataloader, n_batch, th, snr_db):
        """
            test the snr for the given edp and dataloader
            n_user: number of users
            edp_ls: list of edp
            dataloader: a multiuser dataloader
            snr: snr in dB
        """

        ret = {}

        awgn_channel = AWGNSingleChannel(snr_db=snr_db)
        ui = ui_cls(model=sc, channel=awgn_channel, device=th.device)
        ret['awgn'] = ui.test_uplink(dataloader, uplink_method, n_batch=n_batch)
            
        rayleigh_channel = RayleighFadingSingleChannel(snr_db=snr_db, channel_gain_var=1, divide_gain=True, fading_mode='slow')
        ui = ui_cls(model=sc, channel=rayleigh_channel, device=th.device)
        ret['rayleigh'] = ui.test_uplink(dataloader, uplink_method, n_batch=n_batch)

        return ret

    def test_all_snr(*args, **kwargs):
        acc_ls = []
        metric_ls = []
        for th in th_ls:
            th_sc = th.get_multimodal_sc()
            metric = {}
            for snr in tqdm(snr_range, desc="Testing SNRs", leave=False):
                acc = test_snr(*args, **kwargs, sc=th_sc, th=th, snr_db=snr)
                # acc_ls.append(acc)
                metric[snr] = acc
            
            metric_ls.append(metric)
        
        metrics = avg_metric(metric_ls)
        
        return metrics

    def save_metrics(metrics, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(metrics, f)

    metrics = test_all_snr(n_user=3, dataloader=th_dataloader, n_batch=n_batch)
    
    if save_results:
        save_metrics(metrics, result_path)
    
    return metrics

def make_stat(metrics) -> tuple[dict, list[int]]:
    """
    Args:
        metrics: e.g., from test_all_cases_revised's pickle file
            and is indexed like this
                metrics[snr][channel] = accuracy

    Returns: 
        stat: indexed like this:
            stat[channel] = list of accuracy for snr in snr_range
            You can index the stats like this:
                stats['awgn'] = [value for snr in snr_range]
            i.e., the list would be ready for plotting
        snr_range (list[int]): the SNR range used for the metrics, in dB
            ref. test_call_case_revised
    """

    snr_range = list(sorted(metrics.keys()))

    # metrics[snr]['awgn']= value
    # stats['awgn'] = [value for snr in snr_range]
    
    stats = {
        channel: [metrics[snr][channel] for snr in snr_range] 
        for channel in metrics[snr_range[0]].keys()
    }
    
    # print(metrics[snr_range[0]].keys())
    
    return stats, snr_range

def plot_general(
    save_dir: Path | None, 
    metric_name,
    result_filename: str,
    metric_filepaths: list[Path],
    line_kwargs: list[dict],
    legend_kwargs: dict = {'loc': 'lower right'},
    task_type: str = None,
    upper_bound_info: dict = None,
    channel_type: Literal['awgn', 'rayleigh'] = 'awgn',
    additional_title: str | None = None,
    override_title: str | None = None,
    limit_snr_range: tuple[float, float] = None, # NOT SUPPORTED YET
    text_index: Literal['color', 'number'] = 'color',
    y_limited: list[int] = None
):
    """
    it will plot the following 
    save_dir/(awgn|rayleigh)/snr_{result_filename}?.png

    Args: 
        [General]
            save_dir: the directory to save the result
                if None, then the plot will be returned as matplotlib figures
                the returned value would be a dict mapping relative path (that would've been saved in save_dir)
                to the matplotlib figure, e.g., {'prelim_plot/awgn/snr_psnr.png': fig}
                if not None, we directly save the figures to that directory

        [Metric Specification]
            metric_filename: the filename of the metric, this is used in the filename of the plot image
                            e.g., 'psnr', 'bleu1', 'msssim', 'semsim', etc.

        [Line Specification]
            metric_name: the name of the metric, this is used in the plot title and labels
                        e.g., 'Accuracy','PSNR (dB)', 'BLEU-1', 'SSIM', 'MS-SSIM', etc.
            metric_filepaths: the list of filepaths for each metric, 
                            we assume there's L metrics (i.e., len(metric_filepaths) = L)
                            each metric file will be drawn into a line in a plot
                            these files is generated from test_*() functions, ref. test_general() for format
            line_kwargs: the list of kwargs for each line, length L
                        you must at least give the following:
                        {'marker', 'linestyle', 'color', 'label'}
                        e.g., 
                            {'marker': 'o', 'linestyle': '-', 'color': 'tab:blue', 'label': 'Text Model 1 (separate)'}
                            {'marker': '^', 'linestyle': '--', 'color': 'tab:orange', 'label': 'Text Model 1 (MA with Text Model 2)'}
                            {'marker': '*', 'linestyle': '--', 'color': 'tab:green', 'label': 'Text Model 1 (MA with Image Model)'}
            task_type: type of the inference task. For plotting upper bounds
            upper_bound_info: override the upper bounds info for the task, in format: {'label': (value, 'color'), ...}
                e.g., {'Upper bound': (82, 'k')}
            
        [Additional Plot Specific (for snr_.*_h.png)]
            > For below, we called `metric_filepaths[0]'s line` the BLUE line.
            > If all lines of this plot is for one specific model in different scenarios, then we assume the blue line is the model's
            > performance when there's no multi-user interference 
            > BLUE@SNR=snr is the blue line's value at SNR = snr
            
            additional_title: the title of the plot would be something like "Accuracy v.s. Channel SNR (awgn)"
                if you want additional title text added below this, specify it here
            override_title: completely override the title of the plot
                if this is given, additional_title will be ignored
    Returns:
        if save_dir is not None, return None, but the plot will be saved to save_dir/(awgn|rayleigh)/snr_{metric_filename}.png        
        if save_dir is None, return a dict mapping relative path (that would've been saved in save_dir)
            to the matplotlib figure, e.g., {'awgn/snr_psnr.png': fig}
    
    Note: the metric format is:
        metric[snr][channel] = metric_value     
    
    NOTE: some commonly used template is as follows:
        plot_general(save_dir, 'Accuracy (%)', ...)
        
    """

    upper_bound = {
        'msa': {
            'Upper bound': (83, 'k'),
        },
        'ave':{
            'Upper bound': (57, 'k'),
        }
    }
    
    def _get_upper_bound(task_type:str) -> dict[str, int]:
        """return all upper bounds of the task"""
        if task_type in upper_bound:
            return upper_bound[task_type]
    
    # plot related function
    def _plot_upper_bound(ax, val: int, label:str, color='black', ):
        """ draw the red line, red = _get_red_line(...) """
        ax.axhline(y=val, color=color, linestyle='--', label=label, alpha=0.7, linewidth=1.3)

    def _add_text(img_path: Path | Image.Image, red: int, cur_stats: list[list[int]]):
        """ add text to the bottom of image """
        perf = [np.mean(cur_stats[i][-3:]) for i in range(len(cur_stats))]
        perf_text = f'Performance: Red = {red:.4f}\n'
        if text_index == 'color':
            perf_text += ', '.join([f'{line_kwargs[i]["color"]} = {perf[i]:.4f}' for i in range(len(cur_stats))])
        elif text_index == 'number':
            perf_text += ', '.join([f'[{i}] = {perf[i]:.4f}' for i in range(len(cur_stats))])
        dist_text = f'gap(red, *): \n'
        if text_index == 'color':
            dist_text += ', '.join([f'{line_kwargs[i]["color"]} = {red - perf[i]:.4f}' for i in range(len(cur_stats))])
        elif text_index == 'number':
            dist_text += ', '.join([f'[{i}] = {red - perf[i]:.4f}' for i in range(len(cur_stats))])
        text = perf_text + '\n' + dist_text
        from ... import utils
        text_img = utils.text_to_image(text, font_size=24, text_color=(0, 0, 0), bg_color=(255, 255, 255))

        if isinstance(img_path, Path):
            img = Image.open(img_path)
            utils.concat_image_v(img, text_img, bg_color=(255, 255, 255)).save(img_path)
        else:
            img = img_path
            img = utils.concat_image_v(img, text_img, bg_color=(255, 255, 255))
            return img

    def make_all_metrics():
        """
            all_metrics[i][snr]['awgn'] = accuracy
            where i is the index of the metric in metric_filepaths
        """
        def load_metrics(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        all_metrics = {
            i: load_metrics(path)
            for i, path in enumerate(metric_filepaths)
        }
        return all_metrics

    # make sure all lists are the same length
    lens = {
        'metric_filepaths': len(metric_filepaths), 
        # 'metric_user_ids': len(metric_user_ids), 
        'line_kwargs': len(line_kwargs), 
    }
    if len(set(lens.values())) != 1:
        raise ValueError(f"All lists must have the same length, but the lengths are: {lens}")

    # make all stats
    all_metrics = make_all_metrics()
    snr_ranges = []     # note that each metrics may have different snr_range,
                        # but we are gonna plot them in the same graph anyways
    stats = []
    for i in range(len(metric_filepaths)):
        stat, snr_range = make_stat(all_metrics[i])
        snr_ranges.append(snr_range)
        stats.append(stat)

    # do plot
    def plot_channel(cur_save_dir: Path, channel: Literal['awgn', 'rayleigh']):
        cur_save_dir.mkdir(parents=True, exist_ok=True)

        cur_stats = [
            stats[i][channel]
            for i in range(len(metric_filepaths))
        ]

        # plot the lines
        # figsize=(8,6)
        fig, ax1 = plt.subplots(figsize=(6, 7), layout='constrained')
        ax1.set_xlabel('Channel SNR (dB)')
        ax1.set_ylabel(f'{metric_name}')
        
        for i in range(len(metric_filepaths)):
            # crop the range with limit_snr_range
            if limit_snr_range:
                plot_xy = [(snr, stat) for snr, stat in zip(snr_ranges[i], cur_stats[i]) if limit_snr_range[0] <= snr <= limit_snr_range[1]]
                ax1.plot(*zip(*plot_xy), **line_kwargs[i])
            else:
                ax1.plot(snr_ranges[i], cur_stats[i], **line_kwargs[i])
        ax1.tick_params(axis='y', labelcolor='k')
        
        if y_limited is not None:
            ax1.set_ylim(y_limited[0], y_limited[1])
            ax1.set_yticks(np.linspace(y_limited[0], y_limited[1], num=y_limited[2]))
            
        x_tick = [a for i, a in enumerate(snr_ranges[0]) if a % 2 == 0]
        ax1.set_xticks(x_tick, labels=x_tick)
        
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        fig.legend(lines_1, labels_1, **legend_kwargs)
        # ax1.legend(loc="lower left")
        # fig.legend(loc='outside upper center')
        plt.grid(True)
        
        if override_title is not None:
            title = override_title
        else:
            title = f'{metric_name} vs Channel SNR ({channel})'
            if additional_title is not None:
                title = title + '\n' + additional_title
                
        plt.title(title)
        plt.savefig(str(cur_save_dir / f'{result_filename}.png'))
        plt.savefig(str(cur_save_dir / f'{result_filename}.pdf'))
        
        ## draw upper bound
        if upper_bound_info is not None:
            upper_bounds = upper_bound_info
        else:
            upper_bounds = _get_upper_bound(task_type)
        for lb, (val, color) in upper_bounds.items():
            _plot_upper_bound(ax1, val, label=lb, color=color)
        
        
        if fig.legends:
            fig.legends.clear()
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        fig.legend(lines_1, labels_1, **legend_kwargs)
        
        # ax1.legend(loc="lower right")
        plt.savefig(str(cur_save_dir / f'{result_filename}_ub.png'))
        plt.savefig(str(cur_save_dir / f'{result_filename}_ub.pdf'))
        

        # if add_text:
        #     _add_text(cur_save_dir / f'{metric_filename}_h.png', red, cur_stats)
        plt.close()
        
    plot_channel(save_dir / channel_type, channel_type)
    # plot_channel(save_dir / 'rayleigh', 'rayleigh')

class UplinkExperiment20250911:
    """
        The de-facto main class for the uplink task experiments

        Definition:
        - channel: the channel type during training, e.g., 'awgn', 'rayleigh'
        - snr: the SNR during training, in dB, e.g., 0, 5, 10, 15, 20
        - model_type: the model type for the model, e.g., 'udeepsc_msa', 'udeepsc_ave', 'cif', 
        - dataset_type: the dataset type for the model, e.g., 'cmu-mosei', 'mf', 'ave'
        - encoder_out_dims: the number of the output channels for the JSCC, e.g., 128, 256, 512
                                this number should exists for most models (e.g., output channel for image models, output dimension for text models)
                                this is also called e.g. encoder_output_dim, etc. 
                                the definition of this number is model dependent, please check the model definition
                                this number should be somewhat twice of the number of signal symbols used when fed with the same input
        
        So (channel, snr, model_type, dataset_type, encoder_out_dims) would define a model setting, it is trained in train() and tested (separately and MA'ed) in test()
        The model would have a standard name, e.g.,
            UplinkExperiment20250911._get_model_name('awgn', 0, 'udeepsc_msa', 32, 'CMU-MOSEI')
            >>> 'awgn_0_udeepsc_msa_symbols_32_CMU-MOSEI'
        Most times there would be multiple models trained for one setting, they are guarenteed to have different seed, 
        but you need to check the logs to see exactly which seed is set 
            UplinkExperiment20250911._get_model_name('awgn', 0, 'udeepsc_msa', 32, 'CMU-MOSEI', model_id=1)
            >>> 'awgn_0_udeepsc_msa_symbols_32_CMU-MOSEI_model_1'
    """

    """
        defines supported types
        please, no '_' or '.' or space in type name
        when update this, please check the whole TestHelper20250911 and training related functions in UplinkExperiment20250911 
        
        the (dataset_type, batch_size, n_epoch) during training would be:
        - (CMU-MOSEI, 50, 200)
    """

    SupportedModelType = Literal['udeepsc_msa', 'udeepsc_ave', 'udeepscOMA_msa', 'udeepscSIC_msa', 'udeepscOMA_ave', 'udeepscSIC_ave', 'cif']
    SupportedDatasetType = Literal['cmu-mosei', 'mf', 'ave']
    
    supported_model_type = ['udeepsc_msa', 'udeepsc_ave','udeepscOMA_msa', 'udeepscSIC_msa', 'udeepscOMA_ave', 'udeepscSIC_ave', 'cif']
    supported_dataset_type = ['cmu-mosei', 'mf', 'ave']


    """
        result_main_folder, it will have subdirectories like result/, etc.
        cp_main_folder, the main folder for model checkpoints
    """
    result_main_folder = Path('./tmp/20250912')
    cp_main_folder = Path('./checkpoint/20250911')
    use_latest_checkpoint = False

    @staticmethod
    def _check_supported_types(tp, tp_list):
        """ check if the types are in the supported list, tp_list should be a list above """
        if tp not in tp_list:
            raise ValueError(f"Unsupported type: {tp} not in {tp_list}")

    @classmethod
    def _get_model_name(cls, channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id=None):
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

    @classmethod
    def _parse_model_name(cls, model_name):
        """
        parse the model name into a dictionary of model info

        Args:
            model_name: from _get_model_name, 
                e.g., 'awgn_0_issnoma_symbols_32_cityscape' or 'awgn_0_issnoma_symbols_32_cityscape_model_1'
        Returns:
            model_info: a dictionary of model info,
                it should be guarenteed that `model_name == cls._get_model_name(**model_info)`
        """
        match = re.match(r'^([^_]+)_([^_]+)_([^_]+)_symbols_([^_]+)_([^_]+)(_([^_]+))?$', model_name)
        if not match:
            raise ValueError(f"Invalid model name: {model_name}")
        return {
            'channel_type': match.group(1), 
            'snr_db': int(match.group(2)), 
            'model_type': match.group(3), 
            'encoder_out_dims': int(match.group(4)) * 2, 
            'dataset_type': match.group(5), 
            'model_id': int(match.group(7)) if match.group(7) is not None else None,
        }
    
    @classmethod
    def _normalize_model_info(cls, model_info: dict | tuple, return_type: Type[dict] | Type[tuple]) -> dict | tuple:
        """
        (channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id) would define a model,
        and sometimes in the code we call the structure that specifies these settings model_info,
        but model_info can be either a dict or a tuple,
        this function will normalize the model_info to the specified return_type (dict or tuple)

        Args:
            model_info: a dict or tuple representing the model info,
                if dict, it should have keys: 'channel_type', 'snr_db', 'model_type', 'encoder_out_dims', 'dataset_type', 'model_id'
                if tuple, it should have the same order as above, i.e., 
                (channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
            return_type: the type to return, should be either dict or tuple
                if dict, it will return a dict
                if tuple, it will return a tuple in the order of (channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
        Returns:
            a dict or tuple representing the model info, depending on return_type

        Note:
            You can use model_info to call the above functions like this:
            - if model_info is a dict: cls._get_model_name(**model_info)
            - if model_info is a tuple: cls._get_model_name(*model_info)
        """
        if return_type not in [dict, tuple]:
            raise ValueError(f"return_type should be dict or tuple, got {return_type}")
        
        if isinstance(model_info, return_type):
            return model_info
            
        if isinstance(model_info, dict):
            # turn dict to tuple
            return (
                model_info['channel_type'], 
                model_info['snr_db'], 
                model_info['model_type'], 
                model_info['encoder_out_dims'], 
                model_info['dataset_type'], 
                model_info.get('model_id', None)
            )
        elif isinstance(model_info, tuple):
            # turn tuple to dict
            return {
                'channel_type': model_info[0],
                'snr_db': model_info[1],
                'model_type': model_info[2],
                'encoder_out_dims': model_info[3],
                'dataset_type': model_info[4],
                'model_id': model_info[5] if len(model_info) > 5 else None,
            }
        else:
            raise ValueError(f"Invalid model_info type: {type(model_info)}")
    
    @classmethod
    def _get_model_path(cls, channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id, check_exist=True):
        """
        Given the model info, return the path to the model checkpoint folder
        We assume the main checkpoint folder is cls.cp_main_folder, we will get the checkpoint from there
        
        Args:
            should be the same as _get_model_name(), but model_id cannot be None
            check_exist: whether to check if the path exists, this is to automatically raise an error
                if you try to get a model that does not exist
                set it to False if you are training a new model and want to get the path for it to store checkpoints

        Returns:
            a Path object representing the model checkpoint folder
            If check_exist is True, it will raise ValueError if the path does not exist
            If check_exist is False, it will return the path regardless of whether it exists or not
        """
        path = cls.cp_main_folder / cls._get_model_name(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id=model_id)
        if check_exist and not path.exists():
            raise ValueError(f"Model path not exist: {channel_type = }, {snr_db = }, {model_type = }, {encoder_out_dims = }, {dataset_type = }, {model_id = } (search {path = })")
        print(f'Model path: {path}')
        return path
    
    @classmethod
    def _get_seed(cls, *model_info_args, **model_info_kwargs):
        """
        given model info (will be passed to _get_model_name()), return a seed for the model
        this ensures that each model have a unique seed

        Args: 
            model_info_args, model_info_kwargs: the arguments to pass to _get_model_name()
                these should be the same as the arguments in _get_model_name()

        Return:
            an integer seed for the model, should be in range [0, 2**31 - 1]

        Note:
            this function is quite a late addition, so most old models (before 20250609) won't use this function
            tbh we just have to make sure that under the same settings, the model with different model_id uses different seeds,
            this should still hold true even for old models that doesn't use this function

            to see the actual seed used for a model, check the training log
        """
        import hashlib

        # we hash the model name, because the model name should be unique for each model
        # only use 31 bits for seed for safety, in case some set seed function doesn't support large integers
        # (this should be enough anyways)
        model_name = cls._get_model_name(*model_info_args, **model_info_kwargs)
        return int(hashlib.sha256(model_name.encode()).hexdigest(), 16) % (2**31)

        
    class TestHelper20250911(TestHelper):
        """
            The class for all tests, this class should have the ability to make all related 
            models and dataloaders for testing (this don't provide training dataloaders)

            Note:
                When we first design this, we are doing experiments with 2 image models and 2 text models,
                (to be more specific, image+image & text+text & image+text doing multiple access)
                so i made it s.t. we have 4 models that we can make in one class
        """
        def __init__(self, device, 
                        path: Path = None,
                        model_type: 'UplinkExperiment20250911.SupportedModelType' = None,
                        encoder_out_dims: int = None,
                        dataset_type: 'UplinkExperiment20250911.SupportedDatasetType' = None,
                        # optional
                        snr_db: int = 0,
                        channel_type: Literal['awgn', 'rayleigh'] = 'awgn',
                        use_latest_checkpoint: bool = True,
                        modal_combin: MultiModalSC.SupportedModalCombin = None
            ):
            """
            Args:
                *path: the checkpoint folder (we call get_best_checkpoint(*_path / 'checkpoint') to get the actual checkpoint file)
                *model_type, *dataset_type, *encoder_out_channels: ref. UplinkExperiment20250911
                                                                        this setting should be aligned with the model in *_path  

                All these parameters can be None, just in case you don't actually need them
                (e.g., if you're using this class for image only, you can set all text related parameters to None)
            """

            super().__init__(device=device)

            # check if the model / dataset type is at least valid
            UplinkExperiment20250911._check_supported_types(model_type, UplinkExperiment20250911.supported_model_type + [None])
            UplinkExperiment20250911._check_supported_types(dataset_type, UplinkExperiment20250911.supported_dataset_type + [None])
            
            if path is not None:
                if model_type is None or encoder_out_dims is None or dataset_type is None:
                    raise ValueError("Model type, encoder out dims, and dataset type must be provided")
                
            self.path = path
            self.model_type = model_type
            self.encoder_out_dims = encoder_out_dims
            self.dataset_type = dataset_type
            self.snr_db = snr_db
            self.channel_type = channel_type
            self.use_latest_checkpoint = use_latest_checkpoint
            self.modal_combin = modal_combin    
            
        def _make_model(self):
            if self.model_type == 'udeepsc_msa':
                from ...model.udeepsc import UDeepSCNoSIC_msa 
                model = UDeepSCNoSIC_msa(
                    num_symbols=self.encoder_out_dims,
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
            elif self.model_type == 'udeepsc_ave':
                from ...model.udeepsc import UDeepSCNoSIC_ave
                model = UDeepSCNoSIC_ave(
                    num_symbols=self.encoder_out_dims,
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
            elif self.model_type == 'udeepscOMA_msa':
                from ...model.udeepsc import UDeepSCOMA_msa
                model = UDeepSCOMA_msa(
                    num_symbols=self.encoder_out_dims,
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
            elif self.model_type == 'udeepscSIC_msa':
                from ...model.udeepsc import UDeepSCSIC_msa
                model = UDeepSCSIC_msa(
                    num_symbols=self.encoder_out_dims,
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
            elif self.model_type == 'udeepscOMA_ave':
                from ...model.udeepsc import UDeepSCOMA_ave
                model = UDeepSCOMA_ave(
                    num_symbols=self.encoder_out_dims,
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
            elif self.model_type == 'udeepscSIC_ave':
                from ...model.udeepsc import UDeepSCSIC_ave
                model = UDeepSCSIC_ave(
                    num_symbols=self.encoder_out_dims,
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
            elif self.model_type == 'cif':
                from ...model.cif import ChannelFusion
                n_t = 2
                n_r = 4
                n_class = 9
                model = ChannelFusion(
                    in_h=480, in_w=640, n_t=n_t, n_r=n_r, channel_gain_var=1.0, n_class=n_class
                )
            else:
                raise ValueError(f"Unknown image model type: {self.image_model_type}")
            return model
        
        def get_dataloader(self, n_user, batch_size: int, is_multiuser: bool, *args, **kwargs) -> DataLoader:
            if self.dataset_type == 'cmu-mosei':
                test_dataloader = make_udeepsc_msa_testdataloader(batch_size=batch_size, root='data/msadata')
            elif self.dataset_type == 'mf':
                from ...dataset.cif import make_channelfusion_MF_testdataloader
                test_dataloader = make_channelfusion_MF_testdataloader(batch_size=batch_size, root='data/MF')
            elif self.dataset_type == 'ave':
                from ...dataset.ave import make_AVE_testdataloader
                test_dataloader = make_AVE_testdataloader(batch_size=batch_size, root='data/avedata')
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            return test_dataloader
        
        def get_val_dataloader(self, n_user, batch_size: int, is_multiuser: bool, *args, **kwargs) -> DataLoader:
            if self.dataset_type == 'cmu-mosei':
                _, val_dataloader = make_udeepsc_msa_dataloader(batch_size=10, root='data/msadata')
            elif self.dataset_type == 'mf':
                from ...dataset.cif import make_channelfusion_MF_dataloaders
                _, val_dataloader = make_channelfusion_MF_dataloaders(batch_size=10, root='data/MF')
            elif self.dataset_type == 'ave':
                from ...dataset.ave import make_AVE_dataloaders
                _, val_dataloader = make_AVE_dataloaders(batch_size=10, root='data/avedata')
            else:
                raise ValueError(f"Unknown text dataset type: {self.dataset_type}")
            return val_dataloader
        
        def _get_sc(self, path, snr_db, modalCombin):
            model = self._make_model()
            # print(f"Model Path: {path}")
            BaseTrainer.load_model(checkpoint_path=get_best_checkpoint(path / 'checkpoint', use_latest=self.use_latest_checkpoint), model=model, map_location='cpu')
            model.to(self.device)
            if self.model_type == 'udeepsc_msa':
                sc = UDeepSC_MSA_MultiModalSC(model, modalCombin)
            elif self.model_type == 'udeepscOMA_msa':
                sc = UDeepSCOMA_MSA_MultiModalSC(model, modalCombin)
            elif self.model_type == 'udeepscSIC_msa':
                sc = UDeepSCSIC_MSA_MultiModalSC(model, self.channel_type, modalCombin)
            elif self.model_type == 'udeepsc_ave':
                sc = UDeepSC_AVE_MultiModalSC(model)
            elif self.model_type == 'udeepscOMA_ave':
                sc = UDeepSCOMA_AVE_MultiModalSC(model)
            elif self.model_type == 'udeepscSIC_ave':
                sc = UDeepSCSIC_AVE_MultiModalSC(model, self.channel_type)
            elif self.model_type == 'cif':
                sc = CIF_MultiModalSC(model)
            return sc
        
        def get_multimodal_sc(self) -> MultiModalSC:
            return self._get_sc(self.path, self.snr_db, self.modal_combin)
    
    @classmethod
    def _make_uplink_channel(cls, n_user: int = 1, channel_type: Literal['awgn', 'rayleigh'] = 'awgn', snr_db: int = 0):
        from ...channel import AWGNMultiUplinkChannel, RayleighFadingMultiUplinkChannel
        if channel_type not in ['awgn', 'rayleigh']:
            raise ValueError(f"channel_type must be one of ['awgn', 'rayleigh'], but got {channel_type}")
        if channel_type == 'rayleigh':
            return RayleighFadingMultiUplinkChannel(n_user=n_user, snr_db=[snr_db], channel_gain_var=[1], divide_gain=True, fading_mode='slow')
        return AWGNMultiUplinkChannel(n_user=n_user, snr_db=[snr_db], divide_gain=True)
    
    @classmethod
    def _make_train_dataloaders(cls, dataset_type: SupportedDatasetType, batch_size: int):
        UplinkExperiment20250911._check_supported_types(dataset_type, UplinkExperiment20250911.supported_image_dataset_type)

        if dataset_type == 'cmu-mosei':
            from ...dataset.udeepsc import make_udeepsc_msa_dataloader
            train_dataloader, val_dataloader = make_udeepsc_msa_dataloader(
                batch_size=batch_size, root='data/msadata'
            )
            return train_dataloader, val_dataloader
        elif dataset_type == 'mf':
            from ...dataset.cif import make_channelfusion_MF_dataloaders
            train_dataloader, val_dataloader = make_channelfusion_MF_dataloaders(
                batch_size=batch_size, root='data/MF'
            )
            return train_dataloader, val_dataloader
        elif dataset_type == 'ave':
            from ...dataset.ave import make_AVE_dataloaders
            train_dataloader, val_dataloader = make_AVE_dataloaders(
                batch_size=batch_size, root='data/avedata'
            )
            return train_dataloader, val_dataloader
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    @classmethod
    def train_model(cls, save_dir: Path, seed: Optional[int],
                          model_type: SupportedModelType, dataset_type: SupportedDatasetType, encoder_out_dims: int, 
                          channel_type: Literal['awgn', 'rayleigh'], n_user: int, snr_db: int, 
                          batch_size: int, n_epoch: int,
                          device: Optional[int] = None):
        
        UplinkExperiment20250911._check_supported_types(model_type, UplinkExperiment20250911.supported_image_model_type)
        UplinkExperiment20250911._check_supported_types(dataset_type, UplinkExperiment20250911.supported_image_dataset_type)

        from ...trainer.trainer_udeepsc import UDeepSCNoSICTrainer_Msa,UDeepSCNoSICATrainer_AVE
        from ...trainer.trainer_CIF import ChannelFusionTrainer
        from ...train.optim_factory import create_optimizer

        th = cls.TestHelper20250911(
            device=None, 
            pathh=None, 
            model_type=model_type, encoder_out_dims=encoder_out_dims, dataset_type=dataset_type,
            use_latest_checkpoint=cls.use_latest_checkpoint
        )

        logger = get_train_logger(save_dir)
        
        model = th._make_model()
        channel = cls._make_uplink_channel(n_user=n_user, channel_type=channel_type, snr_db=snr_db)
        train_dataloader, val_dataloader = cls._make_train_dataloaders(dataset_type=dataset_type, batch_size=batch_size)
        
        optimizer = create_optimizer({}, model)
        criterion, lr_scheduler = make_training_settings(model)

        trainer_class = {
            'udeepsc_msa': UDeepSCNoSICTrainer_Msa,
            'cif': ChannelFusionTrainer,
            'udeepsc_ave': UDeepSC_AVE_MultiModalSC
        }[model_type]

        trainer = trainer_class(
            logger=logger,
            model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler,
            save_dir=save_dir, display_interval=10,
            n_epoch=n_epoch, gpus=[device], seed=seed,
            resume_checkpoint=None, weights_init=None,
            train_dataloader=train_dataloader, val_dataloader=val_dataloader, channel=channel,
            accumulate_batch=1,
            model_saving_policy='every_min_val_loss',
            power_constraint=[1],
        )

        trainer.train()

    @classmethod
    def _test_noSIC(cls, 
                    model_info,
                    task_type, y_limited=None, n_user=3,
                    batch_size: int = 20, n_batch: int = 50,  device: Optional[str]=None):
        model_info = UplinkExperiment20250911._normalize_model_info(model_info, tuple)
        channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id = model_info
        
        model_name = cls._get_model_name(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
        save_dir = cls.result_main_folder / f'noSIC'/model_name
        plot_save_dir = cls.result_main_folder/ 'test_plot'/ 'no_SIC' / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        partial_ui_cls = functools.partial(UplinkInference, power_constraint=[1.8, 0.8, 0.4])


        print(f'Do test noSIC: {model_name}')

        try:
            checkpoint_path = cls._get_model_path(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
            th = cls.TestHelper20250911(
                device=device,
                path=checkpoint_path,
                model_type=model_type,
                encoder_out_dims=encoder_out_dims,
                dataset_type=dataset_type,
                snr_db=0,
                use_latest_checkpoint=False
            )
            if not (save_dir / f'result.pkl').exists():
                test_general([th], save_dir / 'result.pkl', uplink_method='no_FSM', batch_size=batch_size, n_batch=n_batch, snr_range=np.linspace(-6, 12, 10).tolist(), ui_cls=partial_ui_cls, n_user=n_user, channel_type='awgn')     
            
            print(f"Plot save to {plot_save_dir}")
            plot_general(
                plot_save_dir, 'Accuracy (%)', 'acc',
                [save_dir / 'result.pkl'],
                [
                    {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:red', 'label': 'UDeepSC NO (w/o SD)'},
                ], task_type=task_type, y_limited=y_limited
            )
                
        except Exception as e:
            print(f"Error in _test_noSIC for {model_name}: {e}")
            return
    
    @classmethod
    def _test_OMA(cls, model_info, task_type, y_limited=None,batch_size: int = 20, n_batch: int = 50,  device: Optional[str]=None):
        
        model_info = UplinkExperiment20250911._normalize_model_info(model_info, tuple)
        channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id = model_info
        model_name = cls._get_model_name(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
        save_dir = cls.result_main_folder / f'OMA' /model_name
        # save_dir = cls.result_main_folder / "ui_test/ui/no_FSM_oma" / model_name
        plot_save_dir = cls.result_main_folder/ 'test_plot'/ 'OMA' / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f'Do test OMA: {model_name}')

        checkpoint_path = cls._get_model_path(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
        th = cls.TestHelper20250911(
            device=device,
            path=checkpoint_path,
            model_type=model_type,
            encoder_out_dims=encoder_out_dims,
            dataset_type=dataset_type,
            snr_db=0,
            use_latest_checkpoint=False
        )
        if not (save_dir / f'result.pkl').exists():
            test_OMA_case([th], save_dir / 'result.pkl', uplink_method='no_FSM_oma', batch_size=batch_size, n_batch=n_batch, snr_range=np.linspace(-6, 12, 10).tolist(),ui_cls=UplinkInference)     
        
        print(f"Plot save to {plot_save_dir}")
        plot_general(
            plot_save_dir, 'Accuracy (%)', 'acc',
            [save_dir / 'result.pkl'],
            [
                {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:blue', 'label': 'UDeepSC'},
            ], task_type=task_type, y_limited=y_limited
        )
        
    @classmethod
    def _test_SIC(cls, 
                    model_info,
                    task_type, y_limited=None, n_user=3,
                    batch_size: int = 20, n_batch: int = 50,  device: Optional[str]=None):
        model_info = cls._normalize_model_info(model_info, tuple)
        channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id = model_info
        
        model_name = cls._get_model_name(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
        save_dir = cls.result_main_folder / f'SIC'/ model_name
        plot_save_dir = cls.result_main_folder/ 'test_plot'/ 'SIC' / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        partial_ui_cls = functools.partial(UplinkInference, power_constraint=[1.8, 0.8, 0.4])


        print(f'Do test SIC: {model_name}')

        try:
            checkpoint_path = cls._get_model_path(channel_type, snr_db, model_type, encoder_out_dims, dataset_type, model_id)
            th = cls.TestHelper20250911(
                device=device,
                path=checkpoint_path,
                model_type=model_type,
                encoder_out_dims=encoder_out_dims,
                dataset_type=dataset_type,
                channel_type=channel_type,
                snr_db=0,
                use_latest_checkpoint=False
            )
            if not (save_dir / f'result.pkl').exists():
                test_general([th], save_dir / 'result.pkl', uplink_method='no_FSM_sic', batch_size=batch_size, n_batch=n_batch, snr_range=np.linspace(-6, 12, 10).tolist(), ui_cls=partial_ui_cls, divide_gain=False, n_user=n_user, channel_type=channel_type)     
            
            print(f"Plot save to {plot_save_dir}")
            plot_general(
                plot_save_dir, 'Accuracy (%)', 'acc',
                [save_dir / 'result.pkl'],
                [
                    {'marker': 'o', 'linestyle': '-', 'alpha': 0.8, 'color': 'tab:green', 'label': 'UDeepSC NO (w/ SD)'},
                ], task_type=task_type, channel_type=channel_type, y_limited=y_limited
            )
                
        except Exception as e:
            import traceback, logging
            logging.error(f"Error in _test_SIC for {model_name}: {e}", exc_info=True)
            # traceback.print_exc()
            return
                
    
    @classmethod
    def test(cls, *args):
        def noSIC(*args):
            encoder_out_dims = 48
            model_id = 3
            task = 'msa'
            dataset_type = 'cmu-mosei' if task == 'msa' else 'ave'
            cls._test_noSIC(
                ('rayleigh', 12, f'udeepsc_{task}', encoder_out_dims, dataset_type, model_id), task, y_limited=[55, 85, 7], n_user=3,
                batch_size=10, n_batch=None, device='cuda:0'
            )
        def OMA(*args):
            encoder_out_dims = 16
            model_id = None
            task = 'ave'
            dataset_type = 'cmu-mosei' if task == 'msa' else 'ave'
            cls._test_OMA(
                ('awgn', 12, f'udeepscOMA_{task}', encoder_out_dims, dataset_type, model_id), task, 
                batch_size=10, n_batch=None, device='cuda:1'
            )
        def SIC(*args):
            encoder_out_dims = 48
            model_id = 1
            task = 'msa'
            dataset_type = 'cmu-mosei' if task == 'msa' else 'ave'
            cls._test_SIC(
                ('rayleigh', 12, f'udeepscSIC_{task}', encoder_out_dims, dataset_type, model_id), task, 
                y_limited=[55, 85, 7], 
                n_user=3,
                batch_size=10, n_batch=None, device='cuda:0'
            )   
        
        noSIC()
        # OMA()
        # SIC()
class UplinkExperiment20250914(UplinkExperiment20250911):
    """
    these should run the same logic as ModalityExperiment20250426, 
    but it uses least validation loss checkpoints instead, and use ./tmp/20250608_least_val as the result folder

    please make sure the models you tested with this class actually has least validation loss checkpoints saved,

    basically, any model with model id >= 3 should have least validation loss checkpoints saved
    """
    result_main_folder = Path('./tmp/20250915')
    cp_main_folder = Path('./checkpoint/20250915')
    use_latest_checkpoint = False

class UplinkExperiment20251105(UplinkExperiment20250911):
    """
    these should run the same logic as ModalityExperiment20250426, 
    but it uses least validation loss checkpoints instead, and use ./tmp/20250608_least_val as the result folder

    please make sure the models you tested with this class actually has least validation loss checkpoints saved,

    basically, any model with model id >= 3 should have least validation loss checkpoints saved
    """
    result_main_folder = Path('./tmp/20251105')
    cp_main_folder = Path('./checkpoint/20251105')
    use_latest_checkpoint = False


if __name__ == '__main__':
    import sys
    print("Test test_general()")
    
    UplinkExperiment20251105.test()
    
    
      