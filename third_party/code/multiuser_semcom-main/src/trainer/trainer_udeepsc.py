import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from typing import *
import random

from tqdm import tqdm

from ..model.udeepsc import UDeepSCNoSIC_msa, UDeepSCNoSIC_ave, UDeepSCOMA_msa, UDeepSCSIC_msa, UDeepSCOMA_ave
from ..model.component import signal_power, power_normalize
from ..channel import *
from ..utils import toColor, tensor_real2complex, tensor_complex2real, str_type
from .trainer import TaskOrientTrainer

class UDeepSCNoSICTrainer_Msa(TaskOrientTrainer):
    def __init__(self,power_constraint: list[float], ta_perform,  *args, **kwargs):
        """
            n_user: number of transmitter
            
        """
        self.power_constraint = torch.tensor(power_constraint)
        self.ta_perform = ta_perform
        super().__init__(*args, **kwargs)
        
    def _encode(self, inputs):
        self.model: UDeepSCNoSIC_msa
        
        text, image, speech = inputs  # (batch, # of frames, feature dim)
        
        # Transmitter side
        x_text = self.model.text_encoder(self.ta_perform, text, return_dict=False)[0]
        x_text = x_text[:,-2:-1,:] # only use the CLS token

        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token

        x_speech = self.model.spe_encoder(speech, self.ta_perform) if speech is not None else None
        x_speech = x_speech[:,0,:].unsqueeze(1) if speech is not None else None # only use the CLS token
        
        return x_text, x_image, x_speech
    
    def _channel_encode(self, inputs):
        x_text, x_image, x_speech = inputs
        x_text = self.model.msa_text_encoder_to_channel(x_text)
        x_image = self.model.msa_img_encoder_to_channel(x_image)
        x_speech = self.model.msa_spe_encoder_to_channel(x_speech) if x_speech is not None else None
        
        return x_text, x_image, x_speech
    
    def _transmit_decode(self, inputs):
        self.model: UDeepSCNoSIC_msa
        
        n_user = len(inputs)
        x_text, x_image, x_speech = inputs
        batch_size = x_image.shape[0]
        
        # make power constraint
        power_constraint = self.power_constraint.clone().detach().to(self.device)  # (user,)
        power_constraint = power_constraint.view(1, n_user, 1)
        power_constraint = power_constraint.expand(batch_size, n_user, 1)

        # signal = torch.stack([x_text, x_image], dim=1)
        signal = torch.stack([x_text, x_image], dim=1) if x_speech is None else torch.stack([x_text, x_image, x_speech], dim=1)  # (batch, n_user, c, l)
        
        _, _, n_f, n_len = signal.size()
        signal = signal.flatten(2)
        signal = power_normalize(signal, power_constraint)
        signal = signal.reshape(batch_size, n_user, n_f, n_len)
        signal = tensor_real2complex(signal, 'concat')  
        #(batch_size, 1, 8)

        # make channel info
        # (snr_db.size() may be (1,) or (n_user,))
        # channel_info = self.channel.snr_db.clone().detach().to(self.device)  # (user,)
        # channel_info_size = channel_info.size()[0]
        # channel_info = channel_info.view(1, channel_info_size, 1)
        # channel_info = channel_info.expand(batch_size, n_user, 1)


        # add noise
        signal = self.channel.interfere(signal, 1)
        signal = tensor_complex2real(signal, 'concat')

        x = self.model.msa_channel_to_decoder(signal)
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out

    def transmit(self, inputs, only_get_features: bool=False):
        """
            Assumes:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [text, image, speech]
                model: UDeepSCNoSIC
                channel: MultiUplinkChannel
        """

        self.model: UDeepSCNoSIC_msa
        self.channel: MultiUplinkChannel

        n_user = len(inputs)
        x_text, x_image, x_speech = self._encode(inputs)
        batch_size = x_image.shape[0]

        if only_get_features:
            return x_text, x_image, x_speech
        
        x_text, x_image, x_speech  = self._channel_encode((x_text, x_image, x_speech))
        out = self._transmit_decode((x_text, x_image, x_speech))

        return out
    
    def get_features(self, inputs, is_JSCC:bool=False):
        """
            just get the features, for testing purposes
            these features are outputed by encoders. 
            
            Args:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [image, speech]
            Returns:
                real tensor of shape (batch, user, )
        """
        if is_JSCC:
            feats = self._encode(inputs)
            return self._channel_encode(feats)
        
        return self._encode(inputs)
    
    def get_decode_result(self, inputs):
        return self._transmit_decode(inputs)

class UDeepSCOMATrainer_Msa(TaskOrientTrainer):
    def __init__(self,power_constraint: list[float], ta_perform,  *args, **kwargs):
        """
            n_user: number of transmitter
            
        """
        self.power_constraint = torch.tensor(power_constraint)
        self.ta_perform = ta_perform
        super().__init__(*args, **kwargs)
        
    def _encode(self, inputs):
        self.model: UDeepSCOMA_msa
        
        text, image, speech = inputs  # (batch, # of frames, feature dim)
        
        # Transmitter side
        x_text = self.model.text_encoder(self.ta_perform, text, return_dict=False)[0]
        x_text = x_text[:,-2:-1,:] # only use the CLS token
        # x_text = self.model.msa_text_encoder_to_channel(x_text)

        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token
        # x_image = self.model.msa_img_encoder_to_channel(x_image)

        x_speech = self.model.spe_encoder(speech, self.ta_perform) if speech is not None else None
        x_speech = x_speech[:,0,:].unsqueeze(1) if speech is not None else None # only use the CLS token
        # x_speech = self.model.msa_spe_encoder_to_channel(x_speech) if x_speech is not None else None
        
        return x_text, x_image, x_speech
    
    def _channel_encode(self, inputs):
        x_text, x_image, x_speech = inputs
        x_text = self.model.text_encoder_to_channel(x_text)
        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech) if x_speech is not None else None
        
        return x_text, x_image, x_speech
    
    def _transmit_decode(self, inputs):
        self.model: UDeepSCOMA_msa
        self.channel: SingleChannel
        
        n_user = len(inputs)
        
        signal_ls = []
        for signal, power_constraint in zip(inputs, self.power_constraint):
            signal = tensor_real2complex(signal, 'concat')
            signal = power_normalize(signal, power_constraint)
            signal = self.channel.interfere(signal)
            signal = tensor_complex2real(signal, 'concat')
            signal_ls.append(signal)

        x_text, x_image, x_spe = signal_ls

        x_text = self.model.text_channel_decoder(x_text)
        x_text = self.model.text_channel_to_decoder(x_text)

        x_image = self.model.img_channel_decoder(x_image)
        x_image = self.model.img_channel_to_decoder(x_image)

        x_spe = self.model.spe_channel_decoder(x_spe)
        x_spe = self.model.spe_channel_to_decoder(x_spe)
        
        x = torch.cat([x_image, x_text, x_spe], dim=1)
         
        batch_size = x.shape[0]
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out

    def transmit(self, inputs, only_get_features: bool=False):
        """
            Assumes:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [text, image, speech]
                model: UDeepSCNoSIC
                channel: MultiUplinkChannel
        """

        self.model: UDeepSCNoSIC_msa
        self.channel: SingleChannel
        
        if isinstance(self.channel, VariateChannel):
            self.channel.resample_snr()

        n_user = len(inputs)
        x_text, x_image, x_speech = self._encode(inputs)

        if only_get_features:
            return x_text, x_image, x_speech
        
        x_text, x_image, x_speech  = self._channel_encode((x_text, x_image, x_speech))
        out = self._transmit_decode((x_text, x_image, x_speech))

        return out
    
    def get_features(self, inputs, is_JSCC:bool=False):
        """
            just get the features, for testing purposes
            these features are outputed by encoders. 
            
            Args:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [image, speech]
            Returns:
                real tensor of shape (batch, user, )
        """
        if is_JSCC:
            feats = self._encode(inputs)
            return self._channel_encode(feats)
        
        return self._encode(inputs)
    
    def get_decode_result(self, inputs):
        return self._transmit_decode(inputs)
    
class UDeepSCSICTrainer_Msa(TaskOrientTrainer):
    def __init__(self, power_constraint: list[float], ta_perform, channel_type, *args, **kwargs):
        """
            n_user: number of transmitter
            
        """
        super().__init__(*args, **kwargs)
        self.power_constraint = torch.tensor(power_constraint)
        self.ta_perform = ta_perform
        self.channel_type = self.channel.get_channel_type()
        self.channel_gain = None
        self.total_power = 3
    
    def _SIC(self, signal: torch.Tensor, user_dim_index: int, 
            channel_encoders: list[nn.Module], channel_decoders: list[nn.Module], 
            h=None):
        """            
            Args
                signal: real tensor in (batch_size, 1, *dim, symbol_dim)
                power_constraint: the power constraint for users, length n_user (The order is [text, img, speech])
                h: channel gain (Rayleigh or Rician)
            Return
                a list of decode signals
        """

        device = signal.device
        batch_size = signal.size()[0]
        dim = tuple(signal.size()[user_dim_index + 1:])

        num_users = len(self.power_constraint)
        # print(f"{num_users= }")
        if(num_users == 1):
            estimated = signal
            estimated = channel_decoders[0](estimated)
            return [estimated]
        
        if self.channel_type == "awgn":
            # Sort users by transmit power (descending order)
            user_indices = torch.argsort(self.power_constraint, dim=-1).detach().to('cpu').numpy()
        else: # Rayleigh or Rician
            if h is None:
                raise ValueError("Channel gains (h) must be provided for Rayleigh channel.")
            h = h.squeeze(dim=user_dim_index + 1)
            # Compute effective received power |h_i|^2 * P_i
            # effective_power = torch.abs(h).detach()**2 * power_constraints
            user_indices = torch.argsort(self.power_constraint, dim=-1).detach().to('cpu').numpy()
            signal = tensor_real2complex(signal, 'concat')

        # make power constraint
        power_constraints = self.power_constraint.clone().detach().to(self.device)  # (user,)
        power_constraints = power_constraints.view(1, num_users, 1)
        power_constraints = power_constraints.expand(batch_size, num_users, 1)

        # decoded_signals = torch.zeros((batch_size, num_users, *dim)).to(device)
        decoded_signals = [None] * num_users
        # remaining_signal  = signal.clone().detach().to(device)

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
                estimated = estimated / (h[:, i] + 1e-10)
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
                signal = signal - h[:, i] * estimated_norm

        return decoded_signals
    
    def _encode(self, inputs):
        self.model: UDeepSCSIC_msa
        
        text, image, speech = inputs  # (batch, # of frames, feature dim)
        
        # Transmitter side
        x_text = self.model.text_encoder(self.ta_perform, text, return_dict=False)[0]
        x_text = x_text[:,-2:-1,:] # only use the CLS token

        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token

        x_speech = self.model.spe_encoder(speech, self.ta_perform) if speech is not None else None
        x_speech = x_speech[:,0,:].unsqueeze(1) if speech is not None else None # only use the CLS token
        
        return x_text, x_image, x_speech
    
    def _channel_encode(self, inputs):
        x_text, x_image, x_speech = inputs
        x_text = self.model.text_encoder_to_channel(x_text)
        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech) if x_speech is not None else None
        
        return x_text, x_image, x_speech
    
    def _transmit_decode(self, inputs):
        self.model: UDeepSCSIC_msa
        
        n_user = len(inputs)
        x_text, x_image, x_speech = inputs
        batch_size = x_image.shape[0]
        
        # signal = torch.stack([x_text, x_image], dim=1)
        signal = torch.stack([x_text, x_image], dim=1) if x_speech is None else torch.stack([x_text, x_image, x_speech], dim=1)  # (batch, n_user, c, l)
        
        if self.channel_type == "rayleigh":
            comp_sig = signal.clone().detach().to(self.device)
            comp_sig = tensor_real2complex(comp_sig, 'concat')
            self.power_constraint = self.channel.get_signal_power_constraint(comp_sig, self.total_power, 1) # (n_tx, )
            self.channel_gain = self.channel.get_channel_gain().to(self.device)
            
        
        # make power constraint
        power_constraint = self.power_constraint.clone().detach().to(self.device)  # (user,)
        power_constraint = power_constraint.view(1, n_user, 1)
        power_constraint = power_constraint.expand(batch_size, n_user, 1)
        
        _, _, n_f, n_len = signal.size()
        signal = signal.flatten(2)
        signal = power_normalize(signal, power_constraint)
        signal = signal.reshape(batch_size, n_user, n_f, n_len)
        signal = tensor_real2complex(signal, 'concat')  
        #(batch_size, 1, symbol_length)

        channel_encoders = [self.model.text_encoder_to_channel, self.model.img_encoder_to_channel, self.model.spe_encoder_to_channel]
        channel_decoders = [self.model.text_channel_decoder, self.model.img_channel_decoder, self.model.spe_channel_decoder]

        # add noise
        signal = self.channel.interfere(signal, 1, channel_gain_tensor=self.channel_gain)
        channel_gain = self.channel.get_channel_gain().to(self.device)
        signal = tensor_complex2real(signal, 'concat')
        signals = self._SIC(signal, 1, channel_encoders, channel_decoders, h=channel_gain)

        x_text, x_img, x_spe = signals
        x_text = self.model.text_channel_to_decoder(x_text)
        x_img = self.model.img_channel_to_decoder(x_img)
        x_spe = self.model.spe_channel_to_decoder(x_spe)
        
        # x = torch.cat([x_img, x_spe], dim=1)
        x = torch.cat([x_img, x_text, x_spe], dim=1)
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))

        return out

    def transmit(self, inputs, only_get_features: bool=False):
        """
            Assumes:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [text, image, speech]
                model: UDeepSCNoSIC
                channel: MultiUplinkChannel
        """

        self.model: UDeepSCNoSIC_msa
        self.channel: MultiUplinkChannel

        n_user = len(inputs)
        x_text, x_image, x_speech = self._encode(inputs)
        batch_size = x_image.shape[0]

        if only_get_features:
            return x_text, x_image, x_speech
        
        x_text, x_image, x_speech  = self._channel_encode((x_text, x_image, x_speech))
        out = self._transmit_decode((x_text, x_image, x_speech))

        return out
    
    def get_features(self, inputs, is_JSCC:bool=False):
        """
            just get the features, for testing purposes
            these features are outputed by encoders. 
            
            Args:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [image, speech]
            Returns:
                real tensor of shape (batch, user, )
        """
        if is_JSCC:
            feats = self._encode(inputs)
            return self._channel_encode(feats)
        
        return self._encode(inputs)
    
    def get_decode_result(self, inputs):
        return self._transmit_decode(inputs)

class UDeepSCNoSICATrainer_AVE(TaskOrientTrainer):
    def __init__(self,power_constraint: list[float], ta_perform,  *args, **kwargs):
        """
            n_user: number of transmitter
            
        """
        self.power_constraint = torch.tensor(power_constraint)
        self.ta_perform = ta_perform
        super().__init__(*args, **kwargs)
        
    def _encode(self, inputs):
        self.model: UDeepSCNoSIC_ave
        image, speech = inputs

        # Transmitter side
        image = image.view(image.size(0) * image.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token
        
        speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)
        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_image, x_speech
    
    def _channel_encode(self, inputs):
        x_image, x_speech = inputs

        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        return x_image, x_speech
    
    def _transmit_decode(self, inputs):
        self.model: UDeepSCNoSIC_ave
        self.channel: MultiUplinkChannel
        
        n_user = len(inputs)
        x_image, x_speech = inputs
        
        signal = torch.stack([x_image, x_speech], dim=1)# (batch, n_user, 1, symbol_length)
        
        batch_size = signal.shape[0]
        # make power constraint
        power_constraint = self.power_constraint.clone().detach().to(self.device)  # (user,)
        power_constraint = power_constraint.view(1, n_user, 1)
        power_constraint = power_constraint.expand(batch_size, n_user, 1)
        
        _, _, n_f, n_len = signal.size()
        signal = signal.flatten(2)
        
        signal = power_normalize(signal, power_constraint)
        signal = signal.reshape(batch_size, n_user, n_f, n_len)
        signal = tensor_real2complex(signal, 'concat')  #(batch_size, 1, 8)

        # make channel info
        # (snr_db.size() may be (1,) or (n_user,))
        # channel_info = self.channel.snr_db.clone().detach().to(self.device)  # (user,)
        # channel_info_size = channel_info.size()[0]
        # channel_info = channel_info.view(1, channel_info_size, 1)
        # channel_info = channel_info.expand(batch_size, n_user, 1)


        # add noise
        signal = self.channel.interfere(signal, 1)
        signal = tensor_complex2real(signal, 'concat')

        x = self.model.channel_to_decoder(signal)
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))
        # out = F.softmax(out, dim=-1)

        return out

    def transmit(self, inputs, only_get_features: bool=False):
        """
            Assumes:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [text, image, speech]
                model: UDeepSCNoSIC
                channel: MultiUplinkChannel
        """

        self.model: UDeepSCNoSIC_ave
        self.channel: MultiUplinkChannel

        image, speech = inputs
        AVE_batch_size = image.shape[0]

        x_image, x_speech = self._encode(inputs)

        if only_get_features:
            return x_image, x_speech
        
        x_image, x_speech  = self._channel_encode((x_image, x_speech))
        out = self._transmit_decode((x_image, x_speech))
        out = out.view(AVE_batch_size, -1, out.size(-1))

        return out
    
    def get_features(self, inputs, is_JSCC:bool=False):
        """
            just get the features, for testing purposes
            these features are outputed by encoders. 
            
            Args:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [image, speech]
            Returns:
                real tensor of shape (batch, user, )
        """
        if is_JSCC:
            feats = self._encode(inputs)
            return self._channel_encode(feats)
        
        return self._encode(inputs)
    
    def get_decode_result(self, inputs):
        return self._transmit_decode(inputs)

    
class UDeepSCOMATrainer_Ave(TaskOrientTrainer):
    def __init__(self,power_constraint: list[float], ta_perform,  *args, **kwargs):
        """
            n_user: number of transmitter
            
        """
        self.power_constraint = torch.tensor(power_constraint)
        self.ta_perform = ta_perform
        super().__init__(*args, **kwargs)
        
    def _encode(self, inputs):
        self.model: UDeepSCOMA_ave
        image, speech = inputs

        # Transmitter side
        image = image.view(image.size(0) * image.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token
        
        speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)
        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_image, x_speech
    
    def _channel_encode(self, inputs):
        x_image, x_speech = inputs

        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        return x_image, x_speech
    
    def _transmit_decode(self, inputs):
        self.model: UDeepSCOMA_ave
        self.channel: SingleChannel
        
        n_user = len(inputs)
        x_image, _ = inputs
        
        signal_ls = []
        for signal, power_constraint in zip(inputs, self.power_constraint):
            signal = tensor_real2complex(signal, 'concat')
            signal = power_normalize(signal, power_constraint)
            signal = self.channel.interfere(signal)
            signal = tensor_complex2real(signal, 'concat')
            signal_ls.append(signal)

        x_image, x_spe = signal_ls
        
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

        return out

    def transmit(self, inputs, only_get_features: bool=False):
        """
            Assumes:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [text, image, speech]
                model: UDeepSCNoSIC
                channel: MultiUplinkChannel
        """

        self.model: UDeepSCOMA_ave
        self.channel: SingleChannel
        
        if isinstance(self.channel, VariateChannel):
            self.channel.resample_snr()

        image, speech = inputs
        AVE_batch_size = image.shape[0]

        x_image, x_speech = self._encode(inputs)

        if only_get_features:
            return x_image, x_speech
        
        x_image, x_speech  = self._channel_encode((x_image, x_speech))
        out = self._transmit_decode((x_image, x_speech))
        out = out.view(AVE_batch_size, -1, out.size(-1))

        return out
    
    def get_features(self, inputs, is_JSCC:bool=False):
        """
            just get the features, for testing purposes
            these features are outputed by encoders. 
            
            Args:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [image, speech]
            Returns:
                real tensor of shape (batch, user, )
        """
        if is_JSCC:
            feats = self._encode(inputs)
            return self._channel_encode(feats)
        
        return self._encode(inputs)
    
    def get_decode_result(self, inputs):
        return self._transmit_decode(inputs)
    
class UDeepSCSICTrainer_Ave(TaskOrientTrainer):
    def __init__(self, power_constraint: list[float], ta_perform, channel_type, *args, **kwargs):
        """
            n_user: number of transmitter
            
        """
        self.power_constraint = torch.tensor(power_constraint)
        self.ta_perform = ta_perform
        self.channel_type = channel_type
        super().__init__(*args, **kwargs)
    
    def _SIC(self, signal: torch.Tensor, user_dim_index: int, power_constraints: torch.FloatTensor, 
            channel_encoders: list[nn.Module], channel_decoders: list[nn.Module], 
            h=None):
        """            
            Args
                signal: real tensor in (batch_size, 1, *dim, symbol_dim)
                power_constraint: the power constraint for users, length n_user (The order is [text, img, speech])
                h: channel gain (Rayleigh or Rician)
            Return
                a list of decode signals
        """

        device = signal.device
        batch_size = signal.size()[0]
        dim = tuple(signal.size()[user_dim_index + 1:])

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


        # decoded_signals = torch.zeros((batch_size, num_users, *dim)).to(device)
        decoded_signals = [None] * num_users
        # remaining_signal  = signal.clone().detach().to(device)

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
    
    def _encode(self, inputs):
        self.model: UDeepSCNoSIC_ave
        image, speech = inputs

        # Transmitter side
        image = image.view(image.size(0) * image.size(1), -1, 512) # (batch_size * time_steps, 49, 512)
        x_image = self.model.img_encoder(image, self.ta_perform)
        x_image = x_image[:,0,:].unsqueeze(1) # only use the CLS token
        
        speech = speech.view(-1, speech.size(-1)) # (batch_size * time_steps, 128)
        x_speech = self.model.spe_encoder(speech, self.ta_perform)
        x_speech = x_speech[:,0,:].unsqueeze(1) # only use the CLS token
        
        return x_image, x_speech
    
    def _channel_encode(self, inputs):
        x_image, x_speech = inputs

        x_image = self.model.img_encoder_to_channel(x_image)
        x_speech = self.model.spe_encoder_to_channel(x_speech)
        
        return x_image, x_speech
    
    def _transmit_decode(self, inputs):
        self.model: UDeepSCNoSIC_ave
        self.channel: MultiUplinkChannel
        
        n_user = len(inputs)
        x_image, x_speech = inputs
        
        signal = torch.stack([x_image, x_speech], dim=1)# (batch, n_user, 1, symbol_length)
        
        batch_size = signal.shape[0]
        # make power constraint
        power_constraint = self.power_constraint.clone().detach().to(self.device)  # (user,)
        power_constraint = power_constraint.view(1, n_user, 1)
        power_constraint = power_constraint.expand(batch_size, n_user, 1)
        
        _, _, n_f, n_len = signal.size()
        signal = signal.flatten(2)
        
        signal = power_normalize(signal, power_constraint)
        signal = signal.reshape(batch_size, n_user, n_f, n_len)
        signal = tensor_real2complex(signal, 'concat')  #(batch_size, 1, 8)

        # make channel info
        # (snr_db.size() may be (1,) or (n_user,))
        # channel_info = self.channel.snr_db.clone().detach().to(self.device)  # (user,)
        # channel_info_size = channel_info.size()[0]
        # channel_info = channel_info.view(1, channel_info_size, 1)
        # channel_info = channel_info.expand(batch_size, n_user, 1)

        channel_encoders = [self.model.img_encoder_to_channel, self.model.spe_encoder_to_channel]
        channel_decoders = [self.model.img_channel_decoder, self.model.spe_channel_decoder]

        # add noise
        signal = self.channel.interfere(signal, 1)
        signal = tensor_complex2real(signal, 'concat')
        signals = self._SIC(signal, 1, self.power_constraint, channel_encoders, channel_decoders)

        x_img, x_spe = signals
        x_img = self.model.img_channel_to_decoder(x_img)
        x_spe = self.model.spe_channel_to_decoder(x_spe)
        
        # x = torch.cat([x_img, x_spe], dim=1)
        x = torch.cat([x_img, x_spe], dim=1)
        
        query_embed = self.model.task_dict[self.ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
        out = self.model.decoder(query_embed, x, None, None, None) 

        out = self.model.head[self.ta_perform](out.mean(1))
        # out = F.softmax(out, dim=-1)

        return out

    def transmit(self, inputs, only_get_features: bool=False):
        """
            Assumes:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [text, image, speech]
                model: UDeepSCNoSIC
                channel: MultiUplinkChannel
        """

        self.model: UDeepSCNoSIC_ave
        self.channel: SingleChannel
        
        if isinstance(self.channel, VariateChannel):
            self.channel.resample_snr()

        image, speech = inputs
        AVE_batch_size = image.shape[0]

        x_image, x_speech = self._encode(inputs)

        if only_get_features:
            return x_image, x_speech
        
        x_image, x_speech  = self._channel_encode((x_image, x_speech))
        out = self._transmit_decode((x_image, x_speech))
        out = out.view(AVE_batch_size, -1, out.size(-1))

        return out
    
    def get_features(self, inputs, is_JSCC:bool=False):
        """
            just get the features, for testing purposes
            these features are outputed by encoders. 
            
            Args:
                inputs: A list of tensors,
                        each is a real tensor of shape (batch, c, l)
                        assume modality order is [image, speech]
            Returns:
                real tensor of shape (batch, user, )
        """
        if is_JSCC:
            feats = self._encode(inputs)
            return self._channel_encode(feats)
        
        return self._encode(inputs)
    
    def get_decode_result(self, inputs):
        return self._transmit_decode(inputs)

        