"""
    semcom specific stuff
    
    NOTE: YOU SHOULD MAKE SURE CHANNEL GAIN AND NOISE DOES NOT HAVE GRADIENT
    (i.e., if you need the signal for e.g., SNR power computation, remember to detach() the signal
    because if you don't the gradient might affect the noise or vice versa... afaik)
"""
import torch
import torch.types
from typing import *
import math

from .utils import tensor_real2complex, tensor_complex2real, get_class_str, str_type

class SingleChannel:
    """
        only one transmitter and receiver
    """
    def __init__(self):
        raise NotImplementedError()
    
    def interfere(signal: torch.Tensor) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor with any shape,
                        will treat the last dimension as one signal (i.e., a list of complex numbers)
            Return:
                the signal with interference
        """
        raise NotImplementedError()

    def get_channel_type(self) -> str:
        raise NotImplementedError()

class AWGNSingleChannel(SingleChannel):
    """
        AWGN channel
    """
    def __init__(self, snr_db: float):
        """
            Args:
                snr_db: signal-to-noise ratio in decibel
        """
        self.snr_db = snr_db
    
    def __str__(self):
        return get_class_str(self, snr_db=self.snr_db)
    
    @staticmethod
    def make_awgn_noise(signal: torch.Tensor, snr_db: float | torch.Tensor):
        """
            make AWGN noise for a signal (without applying the signal on it)
            the noise for a signal will be n ~ CN(0, sigma^2 * I),
            where CN is circularly symmetric complex Gaussian distribution

            Args:
                signal: complex tensor with any shape
                snr_db: signal-to-noise ratio in decibel
                        can be float or a real tensor (same shape and device as signal)
            Return:
                the complex noise that can be applied onto it

            NOTE:
                ref. https://en.wikipedia.org/wiki/Complex_normal_distribution#Complex_normal_random_vector
                n ~ CN(0, sigma^2 ^ I) means apply CN(0, sigma^2) for each symbol.
                z ~ CN(0, 1) = R(z), I(z) sampled independently and R(z) ~ N(0, 1/2), I(z) ~ N(0, 1/2)
        """
        signal_power = torch.mean(signal.abs().detach()**2, dim=-1, keepdim=True)

        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear # basically sigma^2

        noise_real = torch.randn_like(signal.real) * torch.sqrt(noise_power / 2)
        noise_imag = torch.randn_like(signal.imag) * torch.sqrt(noise_power / 2)
        noise = torch.complex(noise_real, noise_imag)

        return noise
    
    def interfere(self, signal: torch.Tensor) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor with any shape,
                        will treat the last dimension as one signal (i.e., a list of complex numbers)
            Return:
                the signal with interference
        """
        return signal + AWGNSingleChannel.make_awgn_noise(signal, self.snr_db)
    
    def get_channel_type(self) -> str:
        return 'awgn'

# TODO: make it single channel, also make other fading single channel support fast and slow fading
class FadingSingleChannel(SingleChannel):
    def _make_channel_gain(self, signal: torch.Tensor) -> torch.Tensor:
        """
            Make channel gain for the given signal.
            
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*batch_dim, symbol_dim).
            Return:
                channel_gain: the channel gain for the transmitter to the receiver.
                a complex tensor with size (*batch_dim, symbol_dim)

                this is defined as:
                channel_gain[*batch_dim, k] := h^k, 

                do NOT do anything to the signal: it is here to give information about what we're
                transmitting (e.g., the signal symbol count, etc)
            
            NOTE: need to respect the settings of fading mode and other things (e.g., channel_gain_var)
        """
        raise NotImplementedError()
    
    def _make_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """
            Make noise for the given signal.
            The signal is already multiplied by the channel gain (h*x)
            
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*batch_dim, symbol_dim).
            Return:
                noise: the noise for the received signal for the transmitter.
                a complex tensor with size (*batch_dim, symbol_dim)

                this is defined as:
                noise[*batch_dim, k] := n^k

                do NOT do anything to the signal: it is here to give information about what we're
                transmitting (e.g., the signal symbol count, etc)
        """
        raise NotImplementedError()

    def interfere(self, signal: torch.Tensor,
                  noise_tensor: torch.Tensor = None, channel_gain_tensor: torch.Tensor = None) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*batch_dim, symbol_dim):
                noise_tensor: the noise tensor to be used in the interference,
                              for experiment reproducibility purposes: in case if you have previous
                              noise tensor that you wanna apply this time.
                              if None (default), will generate random noise tensor with the parameters
                              of this channel.
                channel_gain_tensor: same way as noise_tensor works

            Return:
                the signal with interference and superposition for the receiver
                will have dimension (*batch_dim, symbol_dim)

                if self.divide_gain, will divide by the channel gain first before returning
        """
        # make channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        if channel_gain_tensor is not None:
            channel_gain = channel_gain_tensor.clone().detach().to(signal.device)
        else:
            channel_gain = self._make_channel_gain(signal).to(signal.device)

        self.channel_gain = channel_gain.clone().detach().to('cpu')

        # make superimposed signal
        signal = signal * channel_gain

        # add noise        
        if noise_tensor is not None:
            noise = noise_tensor.clone().detach().to(signal.device)
        else:
            noise = self._make_noise(signal).to(signal.device)

        self.noise = noise.clone().detach().to('cpu')
        signal += noise

        if self.divide_gain:
            signal = signal / channel_gain
        
        return signal

    def get_channel_gain(self) -> torch.Tensor:
        """
            Return the random channel gain used in the last interfere()
            Note that if you give channel_gain_tensor parameter in the last interfere(), 
            then this function will return that tensor
            Do note that this tensor is on CPU due to not wanting to occupy GPU memory
            
            Return:
                complex tensor of size (*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)
                where dim1, dim2, symbol_dim is determined by the signal used in the last interfere()

                if we only focus on the dimension (self.n_tx, self.n_rx, symbol_dim),
                get_channel_gain()[i, j, k] := h^k_{i, j}
        """
        return self.channel_gain

    def get_noise(self) -> torch.Tensor:
        """
            Roughly the same as get_channel_gain()
        """
        return self.noise
    
    def get_channel_type(self) -> str:
        raise NotImplementedError()

class RayleighFadingSingleChannel(FadingSingleChannel):
    def __init__(self, snr_db: float, channel_gain_var: float, divide_gain: bool, fading_mode: str = 'fast'):
        """
            Args:
                snr_db: signal-to-noise ratio in decibel
                channel_gain_var: channel gain variance
                divide_gain: whether to divide the channel gain after AWGN channel
            
            The channel gain is sampled for each signal symbol and is of CN(0, channel_gain_var),
            where CN is circularly symmetric complex Gaussian distribution

            The noise is: signal_symbol * channel_gain + noise
        """
        self.snr_db = snr_db
        self.channel_gain_var = channel_gain_var
        self.divide_gain = divide_gain
        self.fading_mode = fading_mode
    
    def __str__(self):
        return get_class_str(self, snr_db=self.snr_db, channel_gain_var=self.channel_gain_var, divide_gain=self.divide_gain)
    
    @staticmethod
    def make_fixed_channel_gain(channel_gain_var: torch.Tensor):
        """
        Generate a single complex channel gain with the given variance,
        then expand it to match the shape of `signal`.
        """
        
        channel_gain = torch.complex(
            torch.ones_like(channel_gain_var.float()) * torch.sqrt(channel_gain_var / 2),
            torch.ones_like(channel_gain_var.float()) * torch.sqrt(channel_gain_var / 2)
        )
        return channel_gain
    
    @staticmethod
    def make_channel_gain_from_var(signal: torch.Tensor, channel_gain_var: float):
        channel_gain = torch.complex(
            torch.randn_like(signal.real) * math.sqrt(channel_gain_var / 2),
            torch.randn_like(signal.real) * math.sqrt(channel_gain_var / 2)
        )
        return channel_gain
    
    @staticmethod
    def make_channel_gain_from_var_tensor(channel_gain_var: torch.Tensor):
        channel_gain = torch.complex(
            torch.randn_like(channel_gain_var.float()) * torch.sqrt(channel_gain_var / 2),
            torch.randn_like(channel_gain_var.float()) * torch.sqrt(channel_gain_var / 2)
        )
        return channel_gain
    
    @staticmethod
    def make_channel_gain_no_phase_from_var_tensor(channel_gain_var: torch.Tensor):
        channel_gain = torch.randn_like(channel_gain_var.float()) * torch.sqrt(channel_gain_var / 2)
        return channel_gain
    
    def _make_channel_gain(self, signal: torch.Tensor) -> torch.Tensor:
        if self.fading_mode == 'fast':
            return RayleighFadingSingleChannel.make_channel_gain_from_var(signal, self.channel_gain_var)
        elif self.fading_mode == 'slow':
            channel_gain_var = self.channel_gain_var * torch.ones(signal.size()[:-1])
            channel_gain_var = channel_gain_var.unsqueeze(-1)
            return RayleighFadingSingleChannel.make_channel_gain_from_var_tensor(channel_gain_var).expand(signal.size())
    
    def _make_noise(self, signal: torch.Tensor) -> torch.Tensor:
        return AWGNSingleChannel.make_awgn_noise(signal, self.snr_db)
    
    def get_channel_type(self) -> str:
        return 'rayleigh'

class RicianFadingSingleChannel(FadingSingleChannel):
    """
        Partially deprecated. I only implement MultiChannel version.
        You can say this version is from U-DeepSC code.
    """
    def __init__(self, snr_db: float, channel_gain_var: float, divide_gain: bool, k_factor: float):
        """
            Args:
                snr_db: signal-to-noise ratio in decibel
                channel_gain_var: channel gain variance
                divide_gain: whether to divide the channel gain after AWGN channel
            
            The channel gain is sampled for each signal symbol and is of CN(0, channel_gain_var),
            where CN is circularly symmetric complex Gaussian distribution

            The noise is: signal_symbol * channel_gain + noise
        """
        self.snr_db = snr_db
        self.channel_gain_var = channel_gain_var
        self.divide_gain = divide_gain
        self.k_factor = k_factor
    
    def __str__(self):
        return get_class_str(self, snr_db=self.snr_db, channel_gain_var=self.channel_gain_var, divide_gain=self.divide_gain, k_factor=self.k_factor)
    
    @staticmethod
    def make_channel_gain_from_var(signal: torch.Tensor, channel_gain_var: float, k_factor: float):
        """
        Generate Rician fading channel gain based on K-factor and variance.
        """
        mean = torch.sqrt(k_factor / (k_factor + 1)) 
        std = math.sqrt(1 / (k_factor + 1))

        channel_gain = torch.complex(
            torch.randn_like(signal.real) * std * math.sqrt(channel_gain_var / 2) + mean,
            torch.randn_like(signal.real) * std * math.sqrt(channel_gain_var / 2) + mean
        )
        return channel_gain
    
    @staticmethod
    def make_channel_gain_from_var_tensor(channel_gain_var: torch.Tensor, k_factor: torch.Tensor):
        """
        Generate Rician fading channel gain based on K-factor tensor and variance tensor.
        """
        mean = torch.sqrt(k_factor / (k_factor + 1)) 
        std = torch.sqrt(1 / (k_factor + 1))
        
        channel_gain = torch.complex(
            torch.randn_like(channel_gain_var.float()) *std * torch.sqrt(channel_gain_var / 2) + mean,
            torch.randn_like(channel_gain_var.float()) *std * torch.sqrt(channel_gain_var / 2) + mean
        )
        return channel_gain

    @staticmethod
    def make_channel_gain_from_var_tensor(
        channel_gain_var: torch.Tensor, 
        k_factor: torch.Tensor):
        """
        Generate Rician fading channel gain based on K-factor tensor and variance tensor.
        """
        mean = torch.sqrt(k_factor / (k_factor + 1)) 
        std = torch.sqrt(1 / (k_factor + 1))
        
        channel_gain = torch.complex(
            torch.randn_like(channel_gain_var.float()) *std * torch.sqrt(channel_gain_var / 2) + mean,
            torch.randn_like(channel_gain_var.float()) *std * torch.sqrt(channel_gain_var / 2) + mean
        )
        return channel_gain

    def _make_channel_gain(self, signal: torch.Tensor) -> torch.Tensor:
        if self.fading_mode == 'fast':
            return RicianFadingSingleChannel.make_channel_gain_from_var(signal, self.channel_gain_var)
        elif self.fading_mode == 'slow':
            channel_gain_var = self.channel_gain_var * torch.ones(signal.size()[:-1])
            channel_gain_var = channel_gain_var.unsqueeze(-1)
            return RicianFadingSingleChannel.make_channel_gain_from_var_tensor(channel_gain_var).expand(signal.size())
    
    def _make_noise(self, signal: torch.Tensor) -> torch.Tensor:
        return AWGNSingleChannel.make_awgn_noise(signal, self.snr_db)
    
    def get_channel_type(self) -> str:
        return 'rician'

class VariateChannel:
    """
        Assumes the derived class has self.snr_db defined, and is a tensor
        of any shape
    """

    def resample_snr(self):
        """
            Resample SNR for random noise part.
            Will return the resampled self.snr_db
            
            the returned SNR will be used in the following call to interfere(),
            until resample_snr() is called again
        """
        raise NotImplementedError()

class UniformVariateSingleChannel(VariateChannel):
    """
        Still RayleighFadingMultiD2DChannel, but SNR can change
        
        Usage: call resample_snr() to resample the SNR before calling interfere()
        the resampled SNR can be accessed via self.snr_db
    """
    def __init__(self, snr_range: tuple[int, int], *args, **kwargs):
        """
            snr_range: the range of sampled SNR.
            
            The rest is for RayleighFadingMultiD2DChannel. The SNR for that
            can be some random value, we will resample it later anyways
        """
        self.snr_low = snr_range[0]
        self.snr_high = snr_range[1]
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return get_class_str(
            self, 
            snr_range=(self.snr_low, self.snr_high),
            cur_snr_db=self.snr_db, 
            underlying_channel=super().__str__()
        )

    def set_snr_range(self, snr_range: tuple[int, int]):
        """
            Set the SNR range for this channel.
            This will be used in resample_snr()
        """
        self.snr_low = snr_range[0]
        self.snr_high = snr_range[1]
        
    def resample_snr(self):
        # support snr_db is tensor or a number
        if isinstance(self.snr_db, torch.Tensor):
            self.snr_db = torch.rand(self.snr_db.size())
            self.snr_db = self.snr_db * (self.snr_high - self.snr_low) + self.snr_low
        else:
            self.snr_db = torch.rand(1).item() * (self.snr_high - self.snr_low) + self.snr_low

# -------

"""
Multi channel info: uplink, downlink, D2D
Those deal with superposition (and channel estimation)

Note that VariateMultiD2DChannel exists! Remember to deal with this as special case!
(I can't really integrate resample_snr() into interfere() because that would make
 DeepMA's DMANet untrivial to implement (?))
"""

class MultiUplinkChannel:
    def interfere(signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, user_dim, *dim2, symbol_dim):
                        - user_dim: for users, indexed by user_dim_index
                        - symbol_dim: for signal symbols
            Return:
                the signal with interference and superposition for the receiver
                will have dimension (*dim1, *dim2, symbol_dim)
        """
        raise NotImplementedError()

class MultiDownlinkChannel:
    def interfere(signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signal from one transmitter.
                        with shape (*dim1, *dim2, symbol_dim):
                        - symbol_dim: for signal symbols
            Return:
                the signal with interference for each receiver (user)
                will have dimension (*dim1, user_dim, *dim2, symbol_dim)
                - user_dim: for users, indexed by user_dim_index (will do unsqueeze() there)
        """
        raise NotImplementedError()

class MultiD2DChannel:
    def interfere(signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signal from multiple transmitters.
                        with shape (*dim1, user_dim, *dim2, symbol_dim):
                        - user_dim: for users (both transmitters and receivers), 
                                    indexed by user_dim_index
                        - symbol_dim: for signal symbols
            Return:
                the signal with interference for each receivers.
                will have dimension (*dim2, user_dim, *dim2, symbol_dim)
                - user_dim: for users (both transmitters and receivers), 
                            indexed by user_dim_index
        """
        raise NotImplementedError()

class VariateMultiChannel(VariateChannel):
    """
        DEPRECATED, shouldn't use this, use VariateChannel instead
        Assumes the derived class has self.snr_db defined, and is a tensor
        of any shape
    """
    def resample_snr(self):
        """
            Resample SNR for random noise part.
            Will return the resampled self.snr_db
            
            the returned SNR will be used in the following call to interfere(),
            until resample_snr() is called again
        """
        raise NotImplementedError()

class NopChannel:
    """
        Equivalent to no channel, just pass the original signal through
        Should handle most cases where the return signal should be the exact same as the input.
        
        Does not support e.g., MultiUplinkChannel and MultiDownlinkChannel
        Does support for MultiD2D channel (so consider use that instead of the above two)

        Can have a channel just in case you want some other attributes of the original channel (e.g., snr_db),
        you can treat this as only overriding interfere() method
    """
    def __init__(self, original_channel=None):
        self.original_channel = original_channel
    def interfere(self, signal: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return signal
    def get_original_channel(self):
        return self.original_channel
    def __getattr__(self, name):
        return getattr(self.original_channel, name)

class UniformVariateMultiChannel(VariateMultiChannel):
    """
        Still RayleighFadingMultiD2DChannel, but SNR can change
        
        Usage: call resample_snr() to resample the SNR before calling interfere()
        the resampled SNR can be accessed via self.snr_db
    """
    def __init__(self, snr_range: tuple[int, int], *args, **kwargs):
        """
            snr_range: the range of sampled SNR.
            
            The rest is for RayleighFadingMultiD2DChannel. The SNR for that
            can be some random value, we will resample it later anyways
        """
        self.snr_low = snr_range[0]
        self.snr_high = snr_range[1]
        super().__init__(*args, **kwargs)
        
    def __str__(self):
        return get_class_str(
            self, 
            snr_range=(self.snr_low, self.snr_high),
            cur_snr_db=self.snr_db, 
            underlying_channel=super().__str__()
        )
        
    def resample_snr(self):
        self.snr_db = torch.rand(self.snr_db.size())
        self.snr_db = self.snr_db * (self.snr_high - self.snr_low) + self.snr_low

class AWGNMultiChannel:
    """
        AWGN channel
    """
    def __init__(self, n_rx: int, snr_db: list[float]):
        if len(snr_db) != n_rx:
            raise Exception(f'{len(snr_db) = } != {n_rx = }')
        
        self.n_rx = n_rx
        self.snr_db = torch.tensor(snr_db)  # should be on cpu
    
    def __str__(self):
        return get_class_str(self, n_rx=self.n_rx, snr_db=self.snr_db.tolist())
    
    def interfere(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, n_tx, *dim2, symbol_dim):
                        - n_tx: number of transmitters, can be any number
                                indexed by user_dim_index
                        - symbol_dim: for signal symbols
            Return:
                the signal with interference and superposition for the receiver
                will have dimension (*dim1, n_rx, *dim2, symbol_dim)
        """
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # make superimposed signal
        signal = torch.sum(signal, dim=user_dim_index)  # (*dim1, *dim2, symbol_dim)
        signal = signal.view(*dim1, 1, *dim2, symbol_dim)
        signal = signal.expand(*dim1, self.n_rx, *dim2, symbol_dim)

        # make noise for each receiver: (*dim1, user_dim, *dim2, symbol_dim)
        snr_db = self.snr_db.clone().detach().to(signal.device)
        snr_db = snr_db.view(*dim1_1, self.n_rx, *dim2_1, 1)
        snr_db = snr_db.expand_as(signal)
        noise = AWGNSingleChannel.make_awgn_noise(signal, snr_db)

        # add noise (wrt receiver)
        signal = signal + noise

        return signal

class AWGNMultiUplinkChannel(AWGNMultiChannel, MultiUplinkChannel):
    def __init__(self, snr_db: float):
        super().__init__(n_rx=1, snr_db=[snr_db])
    def interfere(self, signal, user_dim_index):
        return super().interfere(signal, user_dim_index).squeeze(user_dim_index)

class AWGNMultiDownlinkChannel(AWGNMultiChannel, MultiDownlinkChannel):
    def __init__(self, n_rx: int, snr_db: list[float]):
        super().__init__(n_rx=n_rx, snr_db=snr_db)
    def interfere(self, signal, user_dim_index):
        return super().interfere(signal.unsqueeze(user_dim_index), user_dim_index)


class AWGNMultiD2DChannel(MultiD2DChannel):
    """
        AWGN channel
    """
    def __init__(self, n_user: int, snr_db: list[float]):
        """
            Args:
                n_user: number of transmitter-receiver pairs in this environment
                        (or almost equivalently, encoder-decoder pair (EDP))
                snr_db: signal-to-noise ratio in decibel for the receiver
                        length n_user
        """
        if len(snr_db) != n_user:
            raise Exception(f'{len(snr_db) = } != {n_user = }')
        
        self.n_user = n_user
        self.snr_db = torch.tensor(snr_db)  # should be on cpu
    
    def __str__(self):
        return get_class_str(self, n_user=self.n_user, snr_db=self.snr_db.tolist())
    
    def interfere(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, user_dim, *dim2, symbol_dim):
                        - user_dim: for users, indexed by user_dim_index
                        - symbol_dim: for signal symbols
            Return:
                the signal with interference and superposition for the receiver
                will have dimension (*dim1, user_dim, *dim2, symbol_dim)
        """
        if signal.size()[user_dim_index] != self.n_user:
            raise Exception(f'{signal.size()[user_dim_index] = } != {self.n_user = }')

        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # make superimposed signal
        signal = torch.sum(signal, dim=user_dim_index)  # (*dim1, *dim2, symbol_dim)
        signal = signal.view(*dim1, 1, *dim2, symbol_dim)
        signal = signal.expand(*dim1, self.n_user, *dim2, symbol_dim)

        # make noise for each receiver: (*dim1, user_dim, *dim2, symbol_dim)
        snr_db = self.snr_db.clone().detach().to(signal.device)
        snr_db = snr_db.view(*dim1_1, self.n_user, *dim2_1, 1)
        snr_db = snr_db.expand_as(signal)
        noise = AWGNSingleChannel.make_awgn_noise(signal, snr_db)

        # add noise (wrt receiver)
        signal = signal + noise

        return signal

class FadingMultiChannel:
    """
        This should handle any number of transmitter and receiver.

        We define notation:
        - `C^{NxM}` be the complex matrix of NxM.
        
        We assume:
        - there are N_Tx transmitters and N_Rx receivers.
        - this is a flat fading channel (i.e., the output y^k is only affected by the input x^k / only has a single filter tap)

        For integer k, t, r (all 1-indexing), we define:
        - x^k_{t} (C^{1x1}): the kth signal symbol sent by transmitter t
        - h^k_{t, r} (C^{1x1}): the channel gain for the kth signal symbol from transmitter t to receiver r
        - n^k_{r} (C^{1x1}): the noise for receiver r (for the kth signal symbol)
        - y^k_{r} (C^{1x1}): the received signal for receiver r (for the kth signal symbol)

        then for signal symbol k received by receiver r: y^k_{r}: (C^{1x1}, r in [1, ..., N_Rx]),
            y^k_{r} = \sum_{t=1}^{N_Tx} [h^k_{t, r} * x^k_{t}] + n^k_{r}

        NOTE:
            this is a little bit different to DeepMA's notation and formula.
            namely:
            - DeepMA's notation := our notation:
                - z_{t, k} := x^k_{t}
                - h^k_{t, r} := h^k_{t, r}
                - z^{rev}_{r, k} := y^k_{r}
            - DeepMA's formula is the same as ours for D2D and uplink,
              but for uplink, the input signal should already be the summation of all transmitted signals,
              i.e., DeepMA's formula (3), the summation `\sum_{n=1}^N z_{n, k}` should be done on the signal
              before inputting into FadingMultiDownlinkChannel
    """
    def __init__(self, 
                 n_tx: int, n_rx: int,
                 divide_gain: bool, interfere_mode: Literal['all', 'self', 'other'] = 'all', 
                 fading_mode: Literal['fast', 'slow'] = 'fast',
                 *args, **kwargs):
        """
            Args:
                n_tx: number of transmitter
                n_rx: number of receiver
                divide_gain: whether to divide the result y^k_{r} by the channel gain in interference() before returning
                             this can be True only when `n_tx == 1 or n_rx == 1 or n_tx == n_rx` (downlink, uplink, or D2D)
                             - downlink: y^k_{r} is divided by h^k_{1, r}
                             - uplink: y^k_{1} is divided by h^k_{t, 1}, 
                                       and we return N_Tx signals back instead of the originally intended 1 (= N_Rx)
                             - D2D: y^k_{r} is divided by h^k_{r, r}
                interfere_mode: for DeepMA, ref below
                fading_mode: 'fast' or 'slow', for fast fading or slow fading, ref below.
            
            For interference mode:
                'all': superimpose all signals, every receiver gets the same signal (before channel estimation)
                       this is the default for all cases (even including n_tx != n_rx)
                'self': only works when n_tx == n_rx
                        receiver r only gets signal from transmitter r (effectively orthogonal multiple access, e.g., TDMA)
                        i.e., y^k_{r} = h^k_{r, r} * x^k_{r} + n^k_{r} 
                'other': only works when n_tx == n_rx
                         receiver r only gets superimposed signal from transmitter OTHER THAN transmitter r (DeepMA need this)
                         i.e., y^k_{r} = \sum_{t=1}^{N_Tx} [h^k_{t, r} * x^k_{t}] - (h^k_{r, r} * x^k_{r}) + n^k_{r}
            
            For fading mode:
                'fast': for given i, j: h^k_{i, j} are sampled independently for each k
                'slow': for given i, j: h^k_{i, j} is the same for all k
            
            As for how h^k_{i, j} is sampled, refer to _make_channel_gain()
        """

        super().__init__(*args, **kwargs)

        self.n_tx = n_tx
        self.n_rx = n_rx
        self.divide_gain = divide_gain
        self.interfere_mode = interfere_mode
        self.fading_mode = fading_mode

        if interfere_mode not in ['all', 'self', 'other']:
            raise Exception(f'Interfere mode "{interfere_mode}" invalid')
        if interfere_mode in ['self', 'other'] and n_tx != n_rx:
            raise Exception(f'{interfere_mode = } invalid ({n_tx = } != {n_rx = })')
        
        if fading_mode not in ['fast', 'slow']:
            raise Exception(f'Fading mode "{fading_mode}" invalid')
        
        if divide_gain and not (n_tx == 1 or n_rx == 1 or n_tx == n_rx):
            raise Exception(f'{divide_gain = } invalid ({n_tx = }, {n_rx = })')
        
        self.noise = None
        self.channel_gain = None

    def __str__(self):
        # return get_class_str(
        #     self,
        #     n_tx=self.n_tx,
        #     n_rx=self.n_rx,
        #     divide_gain=self.divide_gain,
        #     interfere_mode=self.interfere_mode,
        #     fading_mode=self.fading_mode
        # )
        return f"FadingMultiChannel({str_type(vars(self), iter_limit_items=10, array_limit_items=20)})"

    @staticmethod
    def _get_diagonal(input: torch.Tensor, first_index: int):
        """
            Get diagonal while maintaining position
            
            Args:
                tensor: shape (*dim1, a_1, a_2, *dim2), a_1 is indexed by `first_index`, a_1 == a_2 := a
            Returns:
                tensor: shape (*dim1, a, *dim2), the diagonal of dimension a_1 and a_2
        """
        dim1 = tuple(input.size()[:first_index])
        # get channel gain: (*dim1, *dim2, a)
        input = input.diagonal(0, first_index, first_index+1)
        # change it back to (*dim1, a, *dim2)
        dim_len = len(input.size())
        dims = list(range(len(dim1))) + [dim_len - 1] + list(range(len(dim1), dim_len-1))
        input = torch.permute(input, dims)

        return input

    def _make_channel_gain(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Make channel gain for the given signal.
            
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, transmitter_dim = n_tx, *dim2, symbol_dim).
                        dim1 and dim2 is treated as batch dimensions.
                user_dim_index: the index of transmitter_dim in the signal tensor
            Return:
                channel_gain: the channel gain for each transmitter to the receiver.
                a complex tensor with size (*dim1, transmitter_dim = n_tx, receiver_dim = n_rx, *dim2, symbol_dim)

                this is defined as:
                channel_gain[*dim1, i, j, *dim2, k] := h^k_{i, j}

                do NOT do anything to the signal: it is here to give information about what we're
                transmitting (e.g., the signal symbol count, etc)
            
            NOTE: need to respect the settings of fading mode and other things (e.g., channel_gain_var)
        """
        raise NotImplementedError()
    
    def _make_noise(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Make noise for the given signal.
            The signal is already multiplied by the channel gain (h*x)
            
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, transmitter_dim = n_tx, *dim2, symbol_dim).
                        dim1 and dim2 is treated as batch dimensions.
                user_dim_index: the index of transmitter_dim in the signal tensor
            Return:
                noise: the noise for the received signal for each transmitter.
                a complex tensor with size (*dim1, receiver_dim = n_rx, *dim2, symbol_dim)

                this is defined as:
                noise[*dim1, r, *dim2, k] := n^k_{r}

                do NOT do anything to the signal: it is here to give information about what we're
                transmitting (e.g., the signal symbol count, etc)
            
            NOTE: you can get the channel gain used from _get_channel_gain()
        """
        raise NotImplementedError()

    def interfere(self, signal: torch.Tensor, user_dim_index: int, 
                  noise_tensor: torch.Tensor = None, channel_gain_tensor: torch.Tensor = None) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, transmitter_dim = n_tx, *dim2, symbol_dim):
                        - transmitter_dim: for transmitters, indexed by user_dim_index
                        - symbol_dim: for signal symbols
                user_dim_index: the index of transmitter_dim in the signal tensor
                noise_tensor: the noise tensor to be used in the interference,
                              for experiment reproducibility purposes: in case if you have previous
                              noise tensor that you wanna apply this time.
                              if None (default), will generate random noise tensor with the parameters
                              of this channel.
                channel_gain_tensor: same way as noise_tensor works

            Return:
                the signal with interference and superposition for the receiver
                will have dimension (*dim1, receiver_dim = n_rx, *dim2, symbol_dim)

                if self.divide, will divide by the channel gain first before returning
        """
        if signal.size()[user_dim_index] != self.n_tx:
            raise Exception(f'{signal.size()[user_dim_index] = } != {self.n_tx = }')

        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # make channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        if channel_gain_tensor is not None:
            channel_gain = channel_gain_tensor.clone().detach().to(signal.device)
        else:
            channel_gain = self._make_channel_gain(signal, user_dim_index).to(signal.device)

        self.channel_gain = channel_gain.clone().detach().to('cpu')

        # expand signal to (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        # for now, for user dimensions, signal[i, j] = x_{i}
        signal = signal.view(*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)
        signal = signal.expand_as(channel_gain)

        # make superimposed signal
        signal = signal * channel_gain

        if self.interfere_mode == 'all':
            signal = torch.sum(signal, dim=user_dim_index)
        elif self.interfere_mode == 'self':
            signal = self._get_diagonal(signal, user_dim_index)
        elif self.interfere_mode == 'other':
            signal = torch.sum(signal, dim=user_dim_index) - self._get_diagonal(signal, user_dim_index)
        else:
            raise Exception(f'Interfere mode "{self.interfere_mode}" invalid')

        # add noise        
        if noise_tensor is not None:
            noise = noise_tensor.clone().detach().to(signal.device)
        else:
            noise = self._make_noise(signal, user_dim_index).to(signal.device)

        self.noise = noise.clone().detach().to('cpu')
        signal += noise

        if self.divide_gain:
            if self.n_tx == 1:
                channel_gain = channel_gain.squeeze(dim=user_dim_index)
            elif self.n_rx == 1:
                channel_gain = channel_gain.squeeze(dim=user_dim_index+1)
            else: # n_tx == n_rx
                channel_gain = self._get_diagonal(channel_gain, user_dim_index)

            signal = signal / (channel_gain + 1e-8)
        
        return signal

    def get_channel_gain(self) -> torch.Tensor:
        """
            Return the random channel gain used in the last interfere()
            Note that if you give channel_gain_tensor parameter in the last interfere(), 
            then this function will return that tensor
            Do note that this tensor is on CPU due to not wanting to occupy GPU memory
            
            Return:
                complex tensor of size (*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)
                where dim1, dim2, symbol_dim is determined by the signal used in the last interfere()

                if we only focus on the dimension (self.n_tx, self.n_rx, symbol_dim),
                get_channel_gain()[i, j, k] := h^k_{i, j}
        """
        return self.channel_gain

    def get_noise(self) -> torch.Tensor:
        """
            Roughly the same as get_channel_gain()
        """
        return self.noise
    
    def get_channel_type(self) -> str:
        """
            Return the channel type string
        """
        return 'FadingMultiChannel'
 
class AWGNMultiChannel(FadingMultiChannel):
    """
        im cheating. AWGN is basically setting h = 1
    """
    def __init__(self, n_tx: int, n_rx: int, snr_db: list[float] | torch.Tensor,
                 divide_gain: bool = False, 
                 interfere_mode: Literal['all', 'self', 'other'] = 'all'):
        """
            Args:
                snr_db: the SNR in dB for each receiver, float tensor of size (n_rx,)
        """
        # divide_gain would affect the return size when it's turned into a uplink class
        # interfere_mode still works in AWGN so we keep that argument
        # fading mode is not important and we won't use that in implementation
        super().__init__(n_tx, n_rx, divide_gain=divide_gain, interfere_mode=interfere_mode, fading_mode='fast')
        
        self.snr_db = torch.tensor(snr_db)
        if self.snr_db.size() != (n_rx,):
            raise Exception(f'{self.snr_db.size() = } invalid ({n_rx = })')
    

    def _make_channel_gain(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        # just return 1 for every symbol
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]

        return torch.ones(*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim).to(signal.device)

    def _make_noise(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        snr_dim = self.snr_db.size()    # == (n_rx,)
        snr_db = self.snr_db.clone().detach().to(signal.device)
        snr_db = snr_db.view(*dim1_1, *snr_dim, *dim2_1, 1)
        snr_db = snr_db.expand(*dim1, *snr_dim, *dim2, symbol_dim)

        return AWGNSingleChannel.make_awgn_noise(signal, snr_db)
    
    def get_channel_type(self) -> str:
        return 'awgn'

class RayleighFadingMultiChannel(FadingMultiChannel):
    """
        Rayleigh channel implementation
        h^k_{t, r} (C^{1x1}) = path_loss_factor * L_{NLOS})

        where:
            path_loss_factor_{r, t}: the path loss factor between Transmitter t and Receiver r
                                     - in IS-SNOMA, this is sqrt(PL(d^s_i)
                                     - in U-DeepSC, this is 1
            K: the K factor
               - in IS-SNOMA, this is 2dB = 10^{0.5}
            L_{NLOS}: the non-line-of-sight component, ~ CN(0, channel_gain_var)
              - in most works, channel_gain_var = 1.0
    """
    def __init__(self, 
                 n_tx: int, n_rx: int, snr_db: list[float] | torch.Tensor, 
                 divide_gain: bool, 
                 noise_power_density_dBm: torch.Tensor | float, 
                 reference_distance: float,
                 reference_path_loss: float,
                 path_loss_exponent: float,
                 distance: torch.Tensor | float,
                 channel_gain_var: list[list[float]] | torch.Tensor | None = None,
                 interfere_mode: Literal['all', 'self', 'other'] = 'all', 
                 fading_mode: Literal['fast', 'slow'] = 'fast',
                ):
        """
            Args:
                Basically same as FadingMultiChannel
                snr_db: the SNR in dB for each receiver, float tensor of size (n_rx,)
                noise_power_density_dBm (torch.Tensor | float):
                    The power density N_0 of the noise, in dBm. a float tensor with shape (n_rx,)
                reference_distance (float): 
                    The reference distance of the reference path loss d_0
                reference_path_loss (float): 
                    The reference path loss \rho_0, equals PL(d_0) by definition
                path_loss_exponent (float): 
                    the exponent for the path loss distance ratio l.
                distance (torch.Tensor | float): 
                    the distance of each transmitter-receiver pair. a float tensor with shape (n_tx, n_rx)
                    unit same as reference_distance
                channel_gain_var: the variance of channel gain of NLOS component, should be of size (n_tx, n_rx). L_{NLOS} ~ CN(0, channel_gain_var).
                                  if not given or given None, will use 1.0 for all channel gains.
        """
        super().__init__(n_tx, n_rx, divide_gain, interfere_mode, fading_mode)
        
        def fill_default_value(name, value, size):
            if isinstance(value, torch.Tensor):
                if value.size() != size:
                    raise Exception(f'{name}: size {value.size() = } invalid (desired size = {size})')
                return value.clone().detach().to('cpu')
            return torch.full(size, value)
        
        self.snr_db = torch.tensor(snr_db)
        if self.snr_db.size() != (n_rx,):
            raise Exception(f'{self.snr_db.size() = } invalid ({n_rx = })')
        
        if isinstance(channel_gain_var, torch.Tensor):
            channel_gain_var = channel_gain_var.clone().detach()
        elif isinstance(channel_gain_var, torch.Tensor):
            channel_gain_var = torch.tensor(channel_gain_var)
        else:
            channel_gain_var = torch.full((n_tx, n_rx), 1.0)

        if channel_gain_var.size() != (n_tx, n_rx):
            raise Exception(f'{channel_gain_var.size() = } invalid ({n_tx = }, {n_rx = })')

        self.channel_gain_var = channel_gain_var
        
        self.noise_power_density_dBm = fill_default_value('noise_power_density_dBm', noise_power_density_dBm, (self.n_rx,))

        self.reference_distance = reference_distance
        self.reference_path_loss = reference_path_loss
        self.distance = distance
        self.path_loss_exponent = path_loss_exponent

    def get_signal_power_constraint(self, signal: torch.Tensor, total_power: float, user_dim_index: int)-> torch.Tensor:
        '''
            Get signal power constraint from total power constraint and channel gain
            
             
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                total_power: total transmitted power constraint
                user_dim_index: the index of transmitter_dim in the signal tensor
            
            Return:
                The power constraint for each transmitter will have dimension (n_tx, )
        '''
        self.n_tx = signal.size()[user_dim_index]
        total_power = torch.full((self.n_tx, 1), total_power).to(signal.device)

        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        dim1_1 = (0,) * len(dim1)
        dim2_1 = (0,) * len(dim2)

        channel_gain = self._make_channel_gain(signal, user_dim_index).to(signal.device)

        # save channel gain for later use
        self.channel_gain = channel_gain.clone().detach().to('cpu')

        # channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        # reduce channel gain dimension to (n_tx, n_rx)
        index = (
            *dim1_1,            # dim1 singleton indices
            slice(None),        # :
            slice(None),        # :
            *dim2_1,            # dim2 singleton indices
            0                   # symbol dim
        )
        reduced_channel_gain = channel_gain[index]
        
        # make signal power constraint from channel gain
        h2 = reduced_channel_gain.abs().detach() ** 2
        invert_h2 = 1 / h2
        
        power_constraint = torch.sqrt((invert_h2 / invert_h2.sum(dim=0, keepdim=True)) * total_power)
        power_constraint = power_constraint.view(self.n_tx,)

        return power_constraint
    
    def _make_channel_gain(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # make channel gain by expanding the given gain variance in user dimension
        # channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        cg_dim = self.channel_gain_var.size()
        channel_gain_var = self.channel_gain_var.clone().detach().to(signal.device)
        channel_gain_var = channel_gain_var.view(*dim1_1, *cg_dim, *dim2_1, 1)
        channel_gain_var = channel_gain_var.expand(*dim1, *cg_dim, *dim2, 1)
        
        # channel_gain = RayleighFadingSingleChannel.make_fixed_channel_gain(channel_gain_var)
        # channel_gain = channel_gain.expand(*dim1, *cg_dim, *dim2, symbol_dim)
        
        # return channel_gain
        
        if self.fading_mode == 'fast':
            # expand symbol_dim and then make channel gain to make each symbol have different channel gain
            channel_gain_var = channel_gain_var.expand(*dim1, *cg_dim, *dim2, symbol_dim)
            NLOS_channel_gain = RayleighFadingSingleChannel.make_channel_gain_from_var_tensor(channel_gain_var)
        elif self.fading_mode == 'slow':
            # make channel gain and then expand symbol_dim to make each symbol have the same channel gain
            # NLOS_channel_gain = RayleighFadingSingleChannel.make_channel_gain_from_var_tensor(channel_gain_var)
            NLOS_channel_gain = RayleighFadingSingleChannel.make_channel_gain_no_phase_from_var_tensor(channel_gain_var)
            NLOS_channel_gain = NLOS_channel_gain.expand(*dim1, *cg_dim, *dim2, symbol_dim)
        else:
            raise Exception(f'Fading mode "{self.fading_mode}" invalid')
        
        # make path loss factor: sqrt(PL(d_i))
        path_loss_factor = torch.sqrt(
            self.reference_path_loss * torch.pow(self.distance / self.reference_distance, -self.path_loss_exponent)
        ).to(signal.device)
        path_loss_factor = path_loss_factor.view(*dim1_1, self.n_tx, self.n_rx, *dim2_1, 1)
        path_loss_factor = path_loss_factor.expand(*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)
        
        # combine all together
        channel_gain = path_loss_factor * NLOS_channel_gain
        
        return channel_gain
    
    def _make_noise(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Make noise for the given signal.
            The signal is already multiplied by the channel gain (h*x)
            
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, transmitter_dim = n_tx, *dim2, symbol_dim).
                        dim1 and dim2 is treated as batch dimensions.
                user_dim_index: the index of transmitter_dim in the signal tensor
            Return:
                noise: the noise for the received signal for each transmitter.
                a complex tensor with size (*dim1, receiver_dim = n_rx, *dim2, symbol_dim)

                this is defined as:
                noise[*dim1, r, *dim2, k] := n^k_{r}

                do NOT do anything to the signal: it is here to give information about what we're
                transmitting (e.g., the signal symbol count, etc)
            
            NOTE: you can get the channel gain used from _get_channel_gain()
        """
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        snr_dim = self.snr_db.size()    # == (n_rx,)
        snr_db = self.snr_db.clone().detach().to(signal.device)
        snr_db = snr_db.view(*dim1_1, *snr_dim, *dim2_1, 1)
        snr_db = snr_db.expand(*dim1, *snr_dim, *dim2, symbol_dim)

        return AWGNSingleChannel.make_awgn_noise(signal, snr_db)
    
    def get_channel_type(self) -> str:
        return 'rayleigh'

class RicianFadingMultiChannel(FadingMultiChannel):
    """
        Rician channel implementation
        h^k_{t, r} (C^{1x1}) = path_loss_factor_{r, t} * (sqrt(K / K+1) * L_{LOS} + sqrt(1 / K+1) * L_{NLOS})

        where:
            path_loss_factor_{r, t}: the path loss factor between Transmitter t and Receiver r
                                     - in IS-SNOMA, this is sqrt(PL(d^s_i)
                                     - in U-DeepSC, this is 1
            K: the K factor
               - in IS-SNOMA, this is 2dB = 10^{0.5}
            L_{LOS}: the line-of-sight component, with a phase los_phase_{r, t}, this equals exp(j * los_phase_{r, t})
              - in IS-SNOMA and U-DeepSC, this is 1, (phase = 0)
            L_{NLOS}: the non-line-of-sight component, ~ CN(0, channel_gain_var)
              - in most works, channel_gain_var = 1.0
    """
    def __init__(self, 
                 n_tx: int, n_rx: int, 
                 divide_gain: bool, 
                 noise_power_density_dBm: torch.Tensor | float, 
                 k_factor: float,
                 los: torch.Tensor | torch.ComplexType,
                 nlos_channel_gain_var: torch.Tensor | float,
                 reference_distance: float,
                 reference_path_loss: float,
                 path_loss_exponent: float,
                 distance: torch.Tensor | float,
                 interfere_mode: Literal['all', 'self', 'other'] = 'all', 
                 fading_mode: Literal['fast', 'slow'] = 'fast',
                 ):
        """
            Args:
                For the below arguments, if specified as `torch.Tensor | type`,
                it means you can specify each element by giving a tensor or fill every element with the same number.

                noise_power_density_dBm (torch.Tensor | float):
                    The power density N_0 of the noise, in dBm. a float tensor with shape (n_rx,)
                k_factor (float): 
                    The K factor of Rician fading. In float (linear scale).
                los (torch.Tensor | torch.complex): 
                    The line of sight component (L_{LOS}). a complex tensor with shape (n_tx, n_rx)
                nlos_channel_gain_var (torch.Tensor | float): 
                    The variance of NLOS component. L_{NLOS} ~ CN(0, nlos_channel_gain_var).
                reference_distance (float): 
                    The reference distance of the reference path loss d_0
                reference_path_loss (float): 
                    The reference path loss \rho_0, equals PL(d_0) by definition
                path_loss_exponent (float): 
                    the exponent for the path loss distance ratio l.
                distance (torch.Tensor | float): 
                    the distance of each transmitter-receiver pair. a float tensor with shape (n_tx, n_rx)
                    unit same as reference_distance
        """
        super().__init__(n_tx, n_rx, divide_gain, interfere_mode, fading_mode)

        def fill_default_value(name, value, size):
            if isinstance(value, torch.Tensor):
                if value.size() != size:
                    raise Exception(f'{name}: size {value.size() = } invalid (desired size = {size})')
                return value.clone().detach().to('cpu')
            return torch.full(size, value)
        
        self.noise_power_density_dBm = fill_default_value('noise_power_density_dBm', noise_power_density_dBm, (n_rx,))
        self.k_factor = k_factor
        self.los = fill_default_value('los', los, (n_tx, n_rx))
        self.nlos_channel_gain_var = fill_default_value('nlos_channel_gain_var', nlos_channel_gain_var, (n_tx, n_rx))
        self.reference_distance = reference_distance
        self.reference_path_loss = reference_path_loss
        self.distance = fill_default_value('distance', distance, (n_tx, n_rx))
        self.path_loss_exponent = fill_default_value('path_loss_exponent', path_loss_exponent, (n_tx, n_rx))

    def _make_channel_gain(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # make NLOS channel gain
        # NLOS channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        cg_dim = self.nlos_channel_gain_var.size()
        nlos_channel_gain_var = self.nlos_channel_gain_var.clone().detach().to(signal.device)
        nlos_channel_gain_var = nlos_channel_gain_var.view(*dim1_1, *cg_dim, *dim2_1, 1)
        nlos_channel_gain_var = nlos_channel_gain_var.expand(*dim1, *cg_dim, *dim2, 1)

        if self.fading_mode == 'fast':
            # expand symbol_dim and then make channel gain to make each symbol have different channel gain
            nlos_channel_gain_var = nlos_channel_gain_var.expand(*dim1, *cg_dim, *dim2, symbol_dim)
            NLOS_channel_gain = RayleighFadingSingleChannel.make_channel_gain_from_var_tensor(nlos_channel_gain_var)
        elif self.fading_mode == 'slow':
            # make channel gain and then expand symbol_dim to make each symbol have the same channel gain
            NLOS_channel_gain = RayleighFadingSingleChannel.make_channel_gain_from_var_tensor(nlos_channel_gain_var)
            NLOS_channel_gain = NLOS_channel_gain.expand(*dim1, *cg_dim, *dim2, symbol_dim)
        else:
            raise Exception(f'Fading mode "{self.fading_mode}" invalid')

        # make LOS channel gain
        # LOS channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        LOS_channel_gain = self.los.clone().detach().to(signal.device)
        LOS_channel_gain = LOS_channel_gain.view(*dim1_1, self.n_tx, self.n_rx, *dim2_1, 1)
        LOS_channel_gain = LOS_channel_gain.expand(*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)

        # make path loss factor sqrt(PL(d_i))
        path_loss_factor = torch.sqrt(
            self.reference_path_loss * torch.pow(self.distance / self.reference_distance, -self.path_loss_exponent)
        ).to(signal.device)
        path_loss_factor = path_loss_factor.view(*dim1_1, self.n_tx, self.n_rx, *dim2_1, 1)
        path_loss_factor = path_loss_factor.expand(*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)

        # combine all together
        channel_gain = path_loss_factor * (
            math.sqrt(self.k_factor / (self.k_factor + 1)) * LOS_channel_gain + 
            math.sqrt(1 / (self.k_factor + 1)) * NLOS_channel_gain
        )

        return channel_gain

    def _make_noise(self, signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Given a signal x \in C^K, we add the noise with individual power of CN(0, N_0) = CN(0, noise_power_density)

            ref. IS_SNOMATrainer note.
        """
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # Convert signal power from dBm to watts
        noise_power_density = 10 ** (self.noise_power_density_dBm / 10) * 1e-3  

        # convert it to the same size as signal
        noise_power_density = noise_power_density.view(*dim1_1, self.n_rx, *dim2_1, 1)
        noise_power_density = noise_power_density.expand(*dim1, self.n_rx, *dim2, symbol_dim)

        # Generate complex Gaussian noise: CN(0, sigma^2 * I)
        noise_real = torch.randn_like(noise_power_density) * torch.sqrt(noise_power_density / 2)
        noise_imag = torch.randn_like(noise_power_density) * torch.sqrt(noise_power_density / 2)
        noise = torch.complex(noise_real, noise_imag)

        return noise
    
    def get_channel_type(self) -> str:
        return 'rician'

"""
    We need a bunch of channels for different link modes, e.g.,:
        RayleighFadingMultiD2DChannel
        VariateRicianFadingMultiDownlinkChannel
        (and a lot more)
    We use functions to produce them
    (Also suggests to use keyword arguments for everything)
"""

def _make_fading_d2d_channel_class(fading_multi_channel_class):
    class FadingMultiD2DChannel(fading_multi_channel_class, MultiD2DChannel):
        def __init__(self, n_user: int, *args, **kwargs):
            super().__init__(n_user, n_user, *args, **kwargs)
        # we don't need to do much else
        def __str__(self):
            return f"FadingMultiD2DChannel(channel_class={fading_multi_channel_class}, {super().__str__()})"
    return FadingMultiD2DChannel

def _make_fading_downlink_channel_class(fading_multi_channel_class):
    class FadingMultiDownlinkChannel(fading_multi_channel_class, MultiDownlinkChannel):
        def __init__(self, n_user: int, *args, **kwargs):
            super().__init__(1, n_user, *args, **kwargs)

        def interfere(self, signal, user_dim_index: int):
            # we want: (*dim1, *dim2, symbol_dim) -> (*dim1, receiver_dim, *dim2, symbol_dim)

            # we turn signal (*dim1, *dim2, symbol_dim) to (*dim1, 1, *dim2, symbol_dim)
            signal = signal.unsqueeze(user_dim_index)

            # no matter what self.divide_gain is, the return shape should be the same
            # i.e., (*dim1, receiver_dim, *dim2, symbol_dim)
            return super().interfere(signal, user_dim_index)
        
        def __str__(self):
            return f"FadingMultiDownlinkChannel(channel_class={fading_multi_channel_class}, {super().__str__()})"
        
    return FadingMultiDownlinkChannel

def _make_fading_uplink_channel_class(fading_multi_channel_class):
    class FadingMultiUplinkChannel(fading_multi_channel_class, MultiUplinkChannel):
        def __init__(self, n_user: int, *args, **kwargs):
            super().__init__(n_user, 1, *args, **kwargs)

        def interfere(self, signal: torch.Tensor, user_dim_index: int, 
                  noise_tensor: torch.Tensor = None, channel_gain_tensor: torch.Tensor = None):
            # we want: (*dim1, transmit_dim, *dim2, symbol_dim) -> (*dim1, *dim2, symbol_dim)
            
            signal = super().interfere(signal, user_dim_index, channel_gain_tensor=channel_gain_tensor)
            
            if self.divide_gain:
                # the interfere() outputs (*dim1, n_user, *dim2, symbol_dim)
                # and we'll let it be
                return signal.squeeze(user_dim_index)
            
            # not dividing gain, the interfere() outputs (*dim1, 1, *dim2, symbol_dim)
            return signal.squeeze(user_dim_index)
        
        def __str__(self):
            return f"FadingMultiUplinkChannel(channel_class={fading_multi_channel_class}, {super().__str__()})"
        
    return FadingMultiUplinkChannel

def _make_variate_channel_class(channel_class, variate_class):
    # we do variate_class first and then channel_class
    # because variate_class is easier and they can pass other arguments to channel_class
    class VariateChannel(variate_class, channel_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def __str__(self):
            return f"VariateChannel(channel_class={channel_class}, variate_class={variate_class}, {super().__str__()})"
    return VariateChannel

# now we make a bunch of channels

AWGNMultiD2DChannel = _make_fading_d2d_channel_class(AWGNMultiChannel)
AWGNMultiDownlinkChannel = _make_fading_downlink_channel_class(AWGNMultiChannel)
AWGNMultiUplinkChannel = _make_fading_uplink_channel_class(AWGNMultiChannel)
RayleighFadingMultiD2DChannel = _make_fading_d2d_channel_class(RayleighFadingMultiChannel)
RayleighFadingMultiDownlinkChannel = _make_fading_downlink_channel_class(RayleighFadingMultiChannel)
RayleighFadingMultiUplinkChannel = _make_fading_uplink_channel_class(RayleighFadingMultiChannel)
RicianFadingMultiD2DChannel = _make_fading_d2d_channel_class(RicianFadingMultiChannel)
RicianFadingMultiDownlinkChannel = _make_fading_downlink_channel_class(RicianFadingMultiChannel)
RicianFadingMultiUplinkChannel = _make_fading_uplink_channel_class(RicianFadingMultiChannel)
VariateRayleighFadingMultiD2DChannel = _make_variate_channel_class(RayleighFadingMultiD2DChannel, UniformVariateMultiChannel)
VariateRayleighFadingMultiDownlinkChannel = _make_variate_channel_class(RayleighFadingMultiDownlinkChannel, UniformVariateMultiChannel)
VariateRayleighFadingMultiUplinkChannel = _make_variate_channel_class(RayleighFadingMultiUplinkChannel, UniformVariateMultiChannel)

VariateRayleighFadingSingleChannel = _make_variate_channel_class(RayleighFadingSingleChannel, UniformVariateSingleChannel)
VariateRicianFadingSingleChannel = _make_variate_channel_class(RicianFadingSingleChannel, UniformVariateSingleChannel)
VariateAWGNSingleChannel = _make_variate_channel_class(AWGNSingleChannel, UniformVariateSingleChannel)

def _test_channel_fadingmultichannel():
    from .utils import str_type, toColor

    settings = []
    for n_tx, n_rx in [(3, 3), (3, 1), (1, 3)]:
        for divide_gain in [True, False]:
            for interference_mode in ['all', 'self', 'other']:
                for fading_mode in ['fast', 'slow']:
                    settings.append((n_tx, n_rx, divide_gain, interference_mode, fading_mode))
    
    for n_tx, n_rx, divide_gain, interference_mode, fading_mode in settings:
        print(toColor(f'{n_tx = }, {n_rx = }, {divide_gain = }, {interference_mode = }, {fading_mode = }', 'yellow'))
        try:
            channel = RicianFadingMultiChannel(
                n_tx, n_rx, [10] * n_rx, divide_gain, 
                2, torch.full((n_tx, n_rx), 0), torch.full((n_tx, n_rx), 1), 
                None, interference_mode, fading_mode)    
            # channel = RayleighFadingMultiChannel(
            #     n_tx, n_rx, [10] * n_rx, divide_gain, None, interference_mode, fading_mode
            # )    
            input = torch.randn(10, 20, n_tx, 30, 40, dtype=torch.complex64)
            print(f'    input = {str_type(input)}')
            output = channel.interfere(input, 2)
            print(f'    output = {str_type(output)}')
            print(f'    channel_gain = {str_type(channel.get_channel_gain())}')
        except Exception as e:
            print(toColor(f'    Exception: {e}', 'cyan'))

def _test_channel_function_class():
    from .utils import str_type, toColor

    settings = []
    for n_user in [3]:
        for divide_gain in [True, False]:
            for interference_mode in ['all', 'self', 'other']:
                for fading_mode in ['fast', 'slow']:
                    settings.append((n_user, divide_gain, interference_mode, fading_mode))
    
    # D2D
    def D2D():
        for n_user, divide_gain, interference_mode, fading_mode in settings:
            print(toColor(f'{n_user = }, {divide_gain = }, {interference_mode = }, {fading_mode = }', 'yellow'))
            try:
                channel = RayleighFadingMultiD2DChannel(
                    n_user, [10] * n_user, divide_gain, 
                    None, interference_mode, fading_mode)    
                input = torch.randn(10, 20, n_user, 30, 40, dtype=torch.complex64)
                print(f'    input = {str_type(input)}')
                output = channel.interfere(input, 2)
                print(f'    output = {str_type(output)}')
                print(f'    channel_gain = {str_type(channel.get_channel_gain())}')
            except Exception as e:
                print(toColor(f'    Exception: {e}', 'cyan'))
    def Uplink():
        for n_user, divide_gain, interference_mode, fading_mode in settings:
            print(toColor(f'{n_user = }, {divide_gain = }, {interference_mode = }, {fading_mode = }', 'yellow'))
            try:
                channel = RayleighFadingMultiUplinkChannel(
                    n_user, [10], divide_gain, 
                    None, interference_mode, fading_mode)    
                input = torch.randn(10, 20, n_user, 30, 40, dtype=torch.complex64)
                print(f'    input = {str_type(input)}')
                output = channel.interfere(input, 2)
                print(f'    output = {str_type(output)}')
                print(f'    channel_gain = {str_type(channel.get_channel_gain())}')
            except Exception as e:
                print(toColor(f'    Exception: {e}', 'cyan'))
    def Downlink():
        for n_user, divide_gain, interference_mode, fading_mode in settings:
            print(toColor(f'{n_user = }, {divide_gain = }, {interference_mode = }, {fading_mode = }', 'yellow'))
            try:
                channel = RayleighFadingMultiDownlinkChannel(
                    n_user, [10] * n_user, divide_gain, 
                    None, interference_mode, fading_mode)    
                input = torch.randn(10, 20, 30, 40, dtype=torch.complex64)
                print(f'    input = {str_type(input)}')
                output = channel.interfere(input, 2)
                print(f'    output = {str_type(output)}')
                print(f'    channel_gain = {str_type(channel.get_channel_gain())}')
            except Exception as e:
                print(toColor(f'    Exception: {e}', 'cyan'))
    def Variate():
        channel = VariateRayleighFadingMultiD2DChannel(
            [10, 40],
            3, [10] * 3, True, 
            None, 'all', 'fast')
        print(channel.snr_db)
        channel.resample_snr()
        print(channel.snr_db)
            
    # D2D()
    # Uplink()
    # Downlink()
    Variate()

if __name__ == '__main__':
    channel = VariateRayleighFadingSingleChannel((0, 10), 10, 1, False, 'slow')
    signal = torch.randn(10, 20, 30, dtype=torch.complex64)
    channel.interfere(signal)
    print(channel.get_channel_gain()[0, 0])
    print(channel.snr_db)