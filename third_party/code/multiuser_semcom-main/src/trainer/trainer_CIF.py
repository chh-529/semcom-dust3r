import torch
from torch import nn
from pathlib import Path
from typing import *
import random

from tqdm import tqdm

from ..model.cif import ChannelFusion, MFNet
from ..model.component import signal_power, power_normalize
from ..channel import *
from ..utils import toColor, tensor_real2complex, tensor_complex2real, str_type
from .trainer import TaskOrientTrainer

# ------

from timeit import default_timer

class ChannelFusionTrainer(TaskOrientTrainer):
    """
        Expect self.model to be ChannelFusion model, and dataloaders to yield (imgs, labels)
        where imgs is of size (batch_size, n_user, channel, h, w)
    """
    def __init__(self,power_constraint: list[float], n_t:int, n_r:int, *args, **kwargs):
        """
            n_user: number of transmitter
            power constraint: the power constraint for users, length n_user
        """
        self.n_user = 2
        self.n_r = n_r
        self.n_t = n_t
        self.power_constraint = torch.tensor(power_constraint)
        super().__init__(*args, **kwargs)

    def transmit(self, inputs, only_get_signal: bool=False):
        """
            Assumes:
                inputs: real tensor of shape (batch, c, h, w)
                model: ChannelFusion
                channel: RayleighFadingD2DChannel
                
            NOTE:
                The paper proposed euqivalent channel layer with channel gain, 
                so the channel here is only for add AWGN noise
        """
        self.model: ChannelFusion
        # self.channel: AWGNSingleChannel
        self.channel: RayleighFadingMultiChannel

        if isinstance(self.channel, VariateMultiChannel):
            self.channel.resample_snr()

        batch_size, c, h, w = inputs.size()

        # make power constraint
        power_constraint = self.power_constraint.clone().detach().to(self.device)  # (user,)
        power_constraint = power_constraint.view(1, self.n_user, 1)
        power_constraint = power_constraint.expand(batch_size, self.n_user, 1)

        # Assume 2 users for this implementation (the original paper also assume 2 user)
        # split inputs to RGB and Infra
        rgb_inputs = inputs[:,:3]
        inf_inputs = inputs[:,3:]

        rgb_signal = self.model.rgbEncoder(rgb_inputs)
        inf_signal = self.model.infraEncoder(inf_inputs)
        
        signal = torch.stack((rgb_signal, inf_signal), dim=1) # (batch_size, 2, 1, 2400)
        channel_gain = self.channel._make_channel_gain(signal, 1).to(signal.device) # (batch_size, n_tx, 1, 1, 2400)
        
        # we should get channel gain h1(rgb) = h_{BS,1} and h2(inf) = h_{BS, 2}
        rgb_channel_gain = channel_gain[:, 0, :, :, 0] # (batch_size, 1, 1)
        rgb_channel_gain = rgb_channel_gain.expand(batch_size, self.n_t, self.n_r)
        inf_channel_gain = channel_gain[:, 1, :, :, 0]
        inf_channel_gain = inf_channel_gain.expand(batch_size, self.n_t, self.n_r)
        
        rgb_precoding = self.model.rgbPrecoder(rgb_channel_gain)
        rgb_signal = self.model.rgbEquivPrecLayer(rgb_signal, rgb_precoding) # (batch_size, 1, 2, 2400)

        inf_precoding = self.model.infraPrecoder(inf_channel_gain)
        inf_signal = self.model.infEquivPrecLayer(inf_signal, inf_precoding) # (batch_size, 1, 2, 2400)
        
        # make superimposed signal
        # (batch_size, 2, 1, 2, 2400)
        signal = torch.stack((rgb_signal, inf_signal), dim=1)
        # signal = torch.sum(signal, dim=1)
        signal = tensor_real2complex(signal, 'concat')
        
        # add noise
        signal = self.channel.interfere(signal, 1, channel_gain_tensor=channel_gain)
        signal = tensor_complex2real(signal, 'concat')

        seg_results = self.model.decoder(signal)

        return seg_results
    
class MFNetTrainer(TaskOrientTrainer):
    """
        Expect self.model to be MFNet model, and dataloaders to yield (imgs, labels)
        where imgs is of size (batch_size, n_user, channel, h, w)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transmit(self, inputs):
        """
            Assumes:
                inputs: real tensor of shape (batch, c, h, w)
                model: MFNet
                channel: AWGNMultiD2DChannel
        """
        self.model: MFNet
        self.channel: AWGNSingleChannel

        batch_size, c, h, w = inputs.size()

        seg_results = self.model(inputs)

        return seg_results
        