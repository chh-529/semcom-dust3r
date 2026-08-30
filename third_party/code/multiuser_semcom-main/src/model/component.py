"""
    shared components of models
"""

from typing import *
from timeit import default_timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.utils.data
from torch.autograd import Function

from ..utils import str_type

def print_debug(msg, *args, **kwargs):
    """
        in case you wanna log all the outputs from the model,
        for debugging purposes
        remember to change the logger
    """
    # import logging
    # logger = logging.getLogger('deepma_test_1')
    # import re, colorama
    # if 'inputs = ' not in msg:
    #     msg = re.sub(r'(\[.*\])([^\[\]]*) = (.*)', f'{colorama.Fore.YELLOW}\\1{colorama.Fore.BLUE}\\2 = \\3{colorama.Style.RESET_ALL}', msg)
    # logger.info(msg, *args, **kwargs)
    pass

"""
GDN implementation by Jorge Pessoa from tensorflow code, 
under MIT license

ref.
- This implementation: https://github.com/jorge-pessoa/pytorch-gdn
- GDN from tensorflow: https://www.tensorflow.org/api_docs/python/tfc/layers/GDN

NOTE: There are other implementations available, such as compressai's GDN module,
      just in case you wanna cross reference other people's pytorch code...

- CompressAI: https://interdigitalinc.github.io/CompressAI/_modules/compressai/layers/gdn.html#GDN
"""

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
  
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


def power_normalize(signal: torch.Tensor, power_constraint_per_complex_symbol: torch.Tensor, allow_less_power: bool = False):
    """
        This would scale each signal s.t. the magnitude of the signal (sum of square) is equal to 
        the number of COMPLEX symbols times power_constraint

        i.e., for one batch, E[z^* z] = K * P
                where:
                z is the complex signal (i.e., complex signal symbol stream) (i.e., torch_real2complex(signal)),
                K is the number of complex symbols (i.e., len(z)), 
                P is the power constraint (i.e., the average power of complex symbol)

        Args:
            signal: real or complex tensor of shape (*batch_size, signal_length)
                    just make sure the last dimension is the signal
            power_constraint_per_complex_symbol: the average power of complex symbol,
                                                 real tensor of shape (*batch_size, 1) OR a float
            allow_less_power: whether to allow the signal to be less than power_constraint_per_complex_symbol
                              (i.e., if the power of the input is already less than the constraint,
                               then keep it as is instead of amplifying it to fit the constraint exactly)
                              
        Returns:
            The same as signal, but power normalized.
        
        NOTE: 
            Note that no matter if the signal is real or complex, we can do power normalize s.t.
            the average power of the complex symbol is power_constraint_per_complex_symbol

            Also, the power constraint specifies COMPLEX signal's average power.
            We assume that you are using tensor_real2complex / tensor_complex2real to transform between
            real and complex signal, i.e., 1 complex signal is 2 real signals combined.

            Note that the following equation holds, and no matter which case, the average power per complex symbol is always 1:
               tensor_real2complex(power_normalize(signal, 1), 'concat')
            == power_normalize(tensor_real2complex(signal, 'concat'), 1)
    """
    
    K = signal.size()[-1] if signal.is_complex() else signal.size()[-1] / 2
    power = torch.sum(torch.abs(signal ** 2), dim=-1, keepdim=True)

    if allow_less_power:
        # if average power is lower than constraint, make `power` equal to constraint to do nothing later
        # if average power is higher than constraint, just use the original formula to scale it to equal constraint
        power = torch.max(power, torch.tensor(K * power_constraint_per_complex_symbol))
    
    return signal * torch.sqrt(
        K * power_constraint_per_complex_symbol / power
    )

def signal_power(signal: torch.Tensor):
    """
        Calculate the power of the signal, this works for both real and complex signal.
        i.e., the signal is of shape (*batch_size, signal_length) where signal_length is even
        and the last dimension is the complex signal

        Args:
            signal: real or complex tensor of shape (*batch_size, signal_length)
        Returns:
            real tensor of shape (*batch_size)
    """
    return torch.sum(torch.abs(signal ** 2), dim=-1)

def signal_power_per_complex_symbol(signal: torch.Tensor):
    K = signal.size()[-1] if signal.is_complex() else signal.size()[-1] / 2
    return torch.sum(torch.abs(signal ** 2), dim=-1) / K

if __name__ == '__main__':
    from ..utils import tensor_complex2real, tensor_real2complex
    signal = torch.range(1, 20).view(2, 10)
    print(signal_power(tensor_real2complex(signal,  'concat')))
    print(signal_power(signal))

    print(signal_power(power_normalize(signal, 100, True)))
    print(signal_power(power_normalize(signal, 100, False)))
    