"""
Model for multimodal channel information fusion
"""

import numpy as np
import os
import math
import numpy as np
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data
from torch.autograd import Function
from ..utils import str_type

"""
Ref:
- https://github.com/haqishen/MFNet-pytorch

Modified to SemCom version introduced in `"Multimodal and Multiuser Semantic Communications 
for Channel-Level Information Fusion" <https://ieeexplore.ieee.org/document/9921202>` _, 
by Xuewen Luo, Ruobin Gao, Hsiao-Hwa Chen, Shuyi Chen, Qing Guo, 
Ponnuthurai Nagaratnam Suganthan, (2024).
"""

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

class ConvBnLeakyRelu2d(nn.Module):
    """
    A convolutional block consist of Conv2d, batch normalization and leaky relu
    """
    
    def __init__(self, in_channels, out_channels, kernel_size:int | tuple = 3, padding: int | tuple = 1, stride: int | tuple = 1, dilation=1, groups=1):
        """
            Args:
                in_channels: input channels
                out_channels: output channels
                kernel_size: kernel size for Conv2d, 
                             int: n for size = n * n or
                             tuple: (h, w) for size = h * w 
                padding: padding for Conv2d
                stride: stride for Conv2d
                dilation: dilation for dilated convolution
                groups: groups
        """
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    
    def forward(self, inputs):
        """
            Args:
                inputs: input tensor with shape (batch_size, in_channels, h, w)
            
            Returns:
                a tensor with shape (batch_size, out_channels, h / stride, w / stride)
        """
        return F.leaky_relu(self.bn(self.conv(inputs)), negative_slope=0.2)


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
            Args:
                in_channels: input channels
                out_channels: output channels
        """
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
    def forward(self,x):
        """
            Args:
                inputs: a tensor with shape (batch_size, in_channels, h, w)
            
            Returns:
                a tensor with shape (batch_size)
        """
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x
    
class MFNet(nn.Module):

    def __init__(self, n_class):
        super(MFNet, self).__init__()
        rgb_ch = [16,48,48,96,96]
        inf_ch = [16,16,16,36,36]

        self.conv1_rgb   = ConvBnLeakyRelu2d(3, rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb   = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb   = MiniInception(rgb_ch[3], rgb_ch[4])

        self.conv1_inf   = ConvBnLeakyRelu2d(1, inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf   = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf   = MiniInception(inf_ch[3], inf_ch[4])

        self.decode4     = ConvBnLeakyRelu2d(rgb_ch[3]+inf_ch[3], rgb_ch[2]+inf_ch[2])
        self.decode3     = ConvBnLeakyRelu2d(rgb_ch[2]+inf_ch[2], rgb_ch[1]+inf_ch[1])
        self.decode2     = ConvBnLeakyRelu2d(rgb_ch[1]+inf_ch[1], rgb_ch[0]+inf_ch[0])
        self.decode1     = ConvBnLeakyRelu2d(rgb_ch[0]+inf_ch[0], n_class)
        

    def forward(self, x):
        # split data into RGB and INF
        x_rgb = x[:,:3]
        x_inf = x[:,3:]

        # encode
        x_rgb    = self.conv1_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb, kernel_size=2, stride=2) # pool1
        x_rgb    = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2) # pool2
        x_rgb    = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2) # pool3
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2) # pool4
        x_rgb    = self.conv5_rgb(x_rgb)

        x_inf    = self.conv1_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf, kernel_size=2, stride=2) # pool1
        x_inf    = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2) # pool2
        x_inf    = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2) # pool3
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2) # pool4
        x_inf    = self.conv5_inf(x_inf)

        x = torch.cat((x_rgb, x_inf), dim=1) # fusion RGB and INF

        # decode
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool4
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool3
        x = self.decode3(x + torch.cat((x_rgb_p3, x_inf_p3), dim=1))
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool2
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool1
        x = self.decode1(x)

        return x

class RGBEncoder(nn.Module): 
    def __init__(self, n_t):
        super(RGBEncoder, self).__init__()
        self.n_t = n_t

        # input: (batch_size, in_channels, h, w)
        rgb_ch = [16,48,48,96,96]

        self.conv1_rgb   = ConvBnLeakyRelu2d(3, rgb_ch[0])         # -> (batch_size, 16, h, w)
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1]) # -> (batch_size, 48, h/2, w/2)
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1]) # -> (batch_size, 48, h/2, w/2)
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2]) # -> (batch_size, 48, h/4, w/4)
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2]) # -> (batch_size, 48, h/4, w/4)
        self.conv4_rgb   = MiniInception(rgb_ch[2], rgb_ch[3])     # -> (batch_size, 96, h/8, w/8)
        self.conv5_rgb   = MiniInception(rgb_ch[3], rgb_ch[4])     # -> (batch_size, 96, h/16, w/16)
        self.conv6_rgb = ConvBnLeakyRelu2d(rgb_ch[4], 48)          # -> (batch_size, 48, h/16, w/16)
        self.conv7_rgb = ConvBnLeakyRelu2d(48, 16)                 # -> (batch_size, 16, h/16, w/16)
        self.conv8_rgb = ConvBnLeakyRelu2d(16, n_t)                  # -> (batch_size, 2, h/16, w/16)
        
    
    def forward(self, x, power_constraint: torch.Tensor | None = None):
        """
            Args:
                x: a real tensor with shape (batch_size, in_channels, h, w)
                power_constraint: real tensor with shape (batch_size, 1), 
                                  average power of symbol: E[z^* z] / K = P,
                                  where K is signal length.
                                  if given None, skip the power normalization completely
            Returns:
                semantic feature data with flatten and power normalization
        """
        x    = self.conv1_rgb(x)
        x    = F.max_pool2d(x, kernel_size=2, stride=2) # pool1 -> (batch_size, 16, h/2, w/2)
        x    = self.conv2_1_rgb(x)
        x_p2 = self.conv2_2_rgb(x)
        x    = F.max_pool2d(x_p2, kernel_size=2, stride=2) # pool2 -> (batch_size, 48, h/4, w/4)
        x    = self.conv3_1_rgb(x)
        x_p3 = self.conv3_2_rgb(x)
        x    = F.max_pool2d(x_p3, kernel_size=2, stride=2) # pool3 -> (batch_size, 48, h/8, w/8)
        x_p4 = self.conv4_rgb(x)
        x    = F.max_pool2d(x_p4, kernel_size=2, stride=2) # pool4 -> (batch_size, 96, h/16, w/16)
        x    = self.conv5_rgb(x)
        x    = self.conv6_rgb(x)
        x    = self.conv7_rgb(x)
        x    = self.conv8_rgb(x)

        x = torch.flatten(x, start_dim=1) # (batch_size, 2 * h/16 * w/16)
        x = x.unsqueeze(1) # (batch_size, 1, 2 * h/16 * w/16)
        # print(x.shape)
        
        # do power normalization
        # return self.power_normalize(x, power_constraint)

        return x

class InfraEncoder(nn.Module):
    def __init__(self, n_t):
        super(InfraEncoder, self).__init__()

        self.n_t = n_t

        # input: (batch_size, in_channels, h, w)
        inf_ch = [16,48,48,96,96]

        self.conv1_inf   = ConvBnLeakyRelu2d(1, inf_ch[0])          # -> (batch_size, 16, h, w)
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])  # -> (batch_size, 48, h/2, w/2)
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])  # -> (batch_size, 48, h/2, w/2)
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])  # -> (batch_size, 48, h/4, w/4)
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])  # -> (batch_size, 48, h/4, w/4)
        self.conv4_inf   = MiniInception(inf_ch[2], inf_ch[3])      # -> (batch_size, 96, h/8, w/8)
        self.conv5_inf   = MiniInception(inf_ch[3], inf_ch[4])      # -> (batch_size, 96, h/16, w/16)
        self.conv6_inf = ConvBnLeakyRelu2d(inf_ch[4], 48)           # -> (batch_size, 48, h/16, w/16)
        self.conv7_inf = ConvBnLeakyRelu2d(48, 16)                  # -> (batch_size, 16, h/16, w/16)
        self.conv8_inf = ConvBnLeakyRelu2d(16, n_t)                   # -> (batch_size, 2, h/16, w/16)
        
    
    def forward(self, x, power_constraint: torch.Tensor | None = None):
        """
            Args:
                x: a real tensor with shape (batch_size, in_channels, h, w)
                power_constraint: real tensor with shape (batch_size, 1), 
                                  average power of symbol: E[z^* z] / K = P,
                                  where K is signal length.
                                  if given None, skip the power normalization completely
            Returns:
                semantic feature data with power normalization
        """
        x    = self.conv1_inf(x)
        x    = F.max_pool2d(x, kernel_size=2, stride=2) # pool1 -> (batch_size, 16, h/2, w/2)
        x    = self.conv2_1_inf(x)
        x_p2 = self.conv2_2_inf(x)
        x    = F.max_pool2d(x_p2, kernel_size=2, stride=2) # pool2 -> (batch_size, 48, h/4, w/4)
        x    = self.conv3_1_inf(x)
        x_p3 = self.conv3_2_inf(x)
        x    = F.max_pool2d(x_p3, kernel_size=2, stride=2) # pool3 -> (batch_size, 48, h/8, w/8)
        x_p4 = self.conv4_inf(x)
        x    = F.max_pool2d(x_p4, kernel_size=2, stride=2) # pool4 -> (batch_size, 96, h/16, w/16)
        x    = self.conv5_inf(x)
        x    = self.conv6_inf(x)
        x    = self.conv7_inf(x)
        x    = self.conv8_inf(x)

        # (batch_size, 2 * h/16 * w/16)
        x = torch.flatten(x, start_dim=1)
        x = x.unsqueeze(1) # (batch_size, 1, 2 * h/16 * w/16)
        # print(x.shape)
        
        # do power normalization
        # return self.power_normalize(x, power_constraint)
        return x
    
class SemanticPrecoder(nn.Module):
    """Semantic precoding network (per‑user)

    Given the (complex) channel gain matrix H of shape (batch, n_r, n_t, 2), the network outputs a
    complex precoding matrix F of shape (B, n_t, n_t, 2) (by default) that is
    subsequently mapped onto an Equivalent convolutional layer.

    Args:
    ----------
        n_t : int
            Number of transmit antennas at the user (``n_t`` in the paper).
        n_r : int
            Number of receive antennas at the base‑station (only used to size the input layer).  
            With default settings the output is (n_t, n_r=n_t); 
            set `square=False` if you prefer an (n_t, n_r) precoder.
        hidden_dim : int, default 128
            Width of the two hidden layers.
        dropout_p : float, default 0.1
            Dropout probability after each hidden layer.
        power_constraint : tensor or None, default None
            If not None, the Frobenius norm of F is scaled so that
            |F_k|_F^2 ≤ power_constraint for every user k.
        square : bool, default False
            True ➔ output shape (n_t, n_r=n_t).
            False ➔ output shape (n_t, n_r).
    """
    def __init__(
        self,
        n_t: int,
        n_r: int,
        hidden_dim: int = 128,
        dropout_p: float = 0.1,
        square: bool = True,
    ) -> None:
        super().__init__()
        self.n_t = n_t
        self.n_r = n_r
        self.square = square
        out_elems = n_t * (n_t if square else n_r)  # number of complex entries
        self.fc1 = nn.Linear(2 * n_t * n_r, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2 * out_elems)
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout_p)
        # self.power_constraint = power_constraint

    def forward(self, channel_gain: torch.Tensor) -> torch.Tensor:
        """
        Args:
        ----------
            channel_gain : torch.Tensor
                Complex channel matrix with shape (batch, n_t, n_r, symbol_dim).

        Returns
        -------
            torch.Tensor
                Precoding matrix F with shape (batch, n_t, n_r or n_t, 2).
        """
        if channel_gain.is_complex():
            channel_gain = torch.view_as_real(channel_gain)  # → (..., 2)
        batch_size, n_t, n_r, _ = channel_gain.shape

        assert n_t == self.n_t and n_r == self.n_r, "input dims mismatch"
        x = channel_gain.reshape(batch_size, -1)  # (batch_size, 2*n_r*n_t)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)  # (B, 2*out_elems)

        # reshape to complex matrix
        out_cols = self.n_t if self.square else self.n_r
        F_c = x.view(batch_size, self.n_t, out_cols, 2)  # (real/imag split)

        return F_c

class EquivalentPrecodingLayer(nn.Module):
    """
        Map complex precoding matrix F to an Equivalent grouped Conv2d.

        This layer reproduces the linear operation y = Fx (on real/imag
        stacked channels) using a 2‑D convolution with kernel size (2, n_t)
        and stride (2, n_t).
    """

    def __init__(self, n_t: int, n_r:int):
        super().__init__()
        self.n_t = n_t
        self.n_r = n_r

    @staticmethod
    def _precoder_to_conv_weight(precoding: torch.Tensor) -> torch.Tensor:
        """
        Convert a single complex precoder to a real 2‑D convolution kernel.

        Args:
        ----------
            precoding : torch.Tensor
                Shape (n_t, n_r, 2) (last dim real/imag).

        Returns:
        -------
                Weight tensor
        """
        n_in, n_out, _ = precoding.shape
        w = torch.zeros(2 * n_out, 1, 2, n_in, dtype=precoding.dtype, device=precoding.device)
        for r in range(n_out):
            real = precoding[:, r, 0]
            imag = precoding[:, r, 1]
            w[2 * r, 0, 0, :] = real
            w[2 * r, 0, 1, :] = -imag
            w[2 * r + 1, 0, 0, :] = imag
            w[2 * r + 1, 0, 1, :] = real
        
        return w

    def _batch_precoder_to_weight(self, precoding: torch.Tensor) -> torch.Tensor:
        """
        Vectorised version of `_precoder_to_conv_weight` for a batch.

        Parameters
        ----------
            precoding : torch.Tensor
                Shape (B, n_in, n_out, 2)

        Returns
        -------
            Weight tensor with shape (2 · n_out, 1, 2, n_in)
        """
        return torch.cat([self._precoder_to_conv_weight(f) for f in precoding], 0)

    def forward(self, x: torch.Tensor, precoding: torch.Tensor) -> torch.Tensor:
        """
        Apply precoding.

        Args:
        ----------
            x : Transmitted signal of shape (B, 1, 2 * n_t * L)
                    L: the number of time slots. It stores stacked real/imag rows for each slot exactly as in the paper.
            precoding : torch.Tensor
                Complex precoder from `SemanticPrecoder` with shape
                (B, n_out, n_in, 2).

        Returns
        -------
                Same shape as x but with the channel dimension grown to
                2 * n_in (as required by the subsequent channel layer).
        """
        # print(f'{x.shape=}')

        batch_size = x.shape[0]
        x = x.view(1, batch_size, 2, -1)

        ntL = x.shape[-1] # nt * L
        n_in, n_out = precoding.shape[1], precoding.shape[2] # n_t, n_r
        assert ntL % n_in == 0, "symbol tensor width not multiple of n_t"
        
        weight = self._batch_precoder_to_weight(precoding) # (B*2*n_out, 1, 2, n_in)
        
        y = F.conv2d(x, weight, stride=(2, n_in), padding=(0, 0), groups=batch_size) # (1, B·2 n_out, 1, L)

        L = y.shape[-1]
        y = y.view(batch_size, 1, 2, n_out * L) # (batch_size, 1, 2, n_t*L)

        return y
        
class ChannelFusDecoder(nn.Module):
    def __init__(self, in_h: int, in_w: int, n_t: int, n_r: int, n_class: int):
        """
            Args: 
                n_t: number of antennas at senders
                n_r: number of antennas at receiver
                n_class: number of class of the segmentation task
        """
        super(ChannelFusDecoder, self).__init__()

        rgb_ch = [16,48,48,96,96]
        inf_ch = [16,48,48,96,96]
        self.in_h = in_h
        self.in_w = in_w
        self.n_t = n_t
        self.n_r = n_r

        self.postprocess = ConvBnLeakyRelu2d(1, 2 * n_t, kernel_size=(2, n_t), stride=(2, n_t), padding=0) # post-process layer
        self.decode9     = ConvBnLeakyRelu2d(2 * n_t, 2)            # -> (batch_size, 2, h/16, w/16)
        self.decode8     = ConvBnLeakyRelu2d(2, 16)                 # -> (batch_size, 16, h/16, w/16)
        self.decode7     = ConvBnLeakyRelu2d(16, 48)                # -> (batch_size, 48, h/16, w/16)
        self.decode6     = ConvBnLeakyRelu2d(48, rgb_ch[4])         # -> (batch_size, 96, h/16, w/16)
        self.decode5     = ConvBnLeakyRelu2d(rgb_ch[4], rgb_ch[3])  # -> (batch_size, 96, h/8, w/8)
        self.decode4     = ConvBnLeakyRelu2d(rgb_ch[3], rgb_ch[2])  # -> (batch_size, 48, h/4, w/4)
        self.decode3     = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[1])  # -> (batch_size, 48, h/2, w/2)
        self.decode2     = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[0])  # -> (batch_size, 16, h, w)
        self.decode1     = ConvBnLeakyRelu2d(rgb_ch[0], n_class)    # -> (batch_size, 9, h, w)
        
    def forward(self, x):
        """
            Args:
                
        """
        
        x = self.postprocess(x)
        # print(f'[Decoder] postprocess(outputs) = {x.shape}')
        x = x.view(x.size(0), -1, self.in_h // 16, self.in_w // 16) # reshape to (batch_size, 2 * n_t, h/16, w/16)
        # upsampling
        x = self.decode9(x)
        x = self.decode8(x)
        x = self.decode7(x)
        x = self.decode6(x)

        # decode
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool4 -> (batch_size, 16, h/8, w/8)
        x = self.decode5(x)
        # print(f'[Decoder] unpool4(outputs) = {x.shape}')
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool3 -> (batch_size, 16, h/4, w/4)
        x = self.decode4(x)
        # print(f'[Decoder] unpool3(outputs) = {x.shape}')
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool2 -> (batch_size, 16, h/2, w/2)
        x = self.decode3(x)
        # print(f'[Decoder] unpool2(outputs) = {x.shape}')
        x = F.interpolate(x, scale_factor=2.0, mode='nearest') # unpool1 -> (batch_size, 16, h, w)
        x = self.decode2(x)
        # print(f'[Decoder] unpool1(outputs) = {x.shape}')
        x = self.decode1(x)

        return x

class ChannelFusion(nn.Module):
    def __init__(self, in_h: int, in_w: int, channel_gain_var: float = 1.0, n_t: int = 2, n_r: int = 4, n_class: int = 9):
        """
            EquivalentPrecoding:
                In the paper, there is an Equivalent precoding conv layer F^{n_t, n_t} and a Equivalent channel layer H^{n_r, n_t}. However, we assume Equivalent precoding conv layer only here and let its shape be (n_r, n_t)

            Args:
                n_t: number of antennas of users
                n_r: number of antennas of receiver
                n_class: number of classes of segmentation task
            Assumes:
                n_r can be divided by n_t.
        """
        super(ChannelFusion, self).__init__()

        self.n_t = n_t
        self.n_r = n_r
        
        self.rgbEncoder = RGBEncoder(n_t)
        self.infraEncoder = InfraEncoder(n_t)
        
        self.rgbPrecoder = SemanticPrecoder(n_t=n_t, n_r=n_r, hidden_dim=128, square=False)
        self.infraPrecoder = SemanticPrecoder(n_t=n_t, n_r=n_r, hidden_dim=128, square=False)
        self.rgbEquivPrecLayer = EquivalentPrecodingLayer(n_t, n_r)
        self.infEquivPrecLayer = EquivalentPrecodingLayer(n_t, n_r)
        
        self.decoder = ChannelFusDecoder(in_h, in_w, n_t, n_r, n_class)

    # def forward(self, x, power_constraint: torch.Tensor | None):
    #     # split data into RGB and INF
    #     x_rgb = x[:,:3]
    #     x_inf = x[:,3:]

    #     x_rgb = self.rgbEncoder(x_rgb, power_constraint)
    #     x_rgb = self.rgbChannelBlock(x_rgb)
        
    #     x_inf = self.infraEncoder(x_inf, power_constraint)
    #     x_inf = self.infraChannelBlock(x_inf)

def unit_test_ChFusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.__version__)
    print(torch.cuda.is_available())  
    print("Device:", device)
    torch.cuda.empty_cache()

    h = 480
    w = 640
    t_antennas = 2
    r_antennas = 4
    n_class = 9

    model = ChannelFusion(in_h=480, in_w=640, n_t=t_antennas, n_r=r_antennas, n_class=n_class)
    model.to(device)

    # Example input tensor
    torch.use_deterministic_algorithms(True)
    input_tensor = torch.randn(2, 4, h, w).to(device)  # [batch_size, channels, height, width]
    input_tensor = input_tensor.to(device)

    rgb_input = input_tensor[:,:3]
    inf_input = input_tensor[:,3:]
    
    print("RGB Input tensor:", str_type(rgb_input))
    print("Infra Input tensor:", str_type(inf_input))

    power_constraint = torch.ones(2, 1).to(device)
    H_var = torch.ones(2, t_antennas, r_antennas).to(device) # (batch_size, n_t, n_r)
   
    H = torch.complex(
            torch.randn_like(H_var.float()) * torch.sqrt(H_var / 2),
            torch.randn_like(H_var.float()) * torch.sqrt(H_var / 2)
        )
    
    encoded_rgb = model.rgbEncoder(rgb_input)
    F_rgb = model.rgbPrecoder(H)
    # print("RGB Precoding tensor:", str_type(F_rgb))
    signal_rgb = model.rgbEquivPrecLayer(encoded_rgb, F_rgb)
    
    encoded_inf = model.infraEncoder(inf_input)
    F_inf = model.infraPrecoder(H)
    signal_inf = model.infEquivPrecLayer(encoded_inf, F_inf)

    print("RGB encoded tensor:", str_type(encoded_rgb))
    print("Infra encoded tensor:", str_type(encoded_inf))
    
    print("RGB transmit tensor:", str_type(signal_rgb)) 
    print("Infra transmit tensor:", str_type(signal_inf))  

    encoded_tensor = torch.stack((signal_rgb, signal_inf), dim=1)
    # make superimposed signal
    encoded_tensor = torch.sum(encoded_tensor, dim=1) 

    decoded_tensor = model.decoder(encoded_tensor)
    print("Decoded tensor:", str_type(decoded_tensor)) 
    
if __name__ == '__main__':
    unit_test_ChFusion()