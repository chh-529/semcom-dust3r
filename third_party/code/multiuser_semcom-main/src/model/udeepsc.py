"""
Model for unified DeepSC (U-DeepSC)
"""

import math
import numpy as np
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from typing import *

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from transformers import BertModel

import timm
net = timm.create_model("vit_base_patch16_384", pretrained=True)

"""
Ref:
- https://github.com/zhang-guangyi/t-udeepsc

Modified to non-orthogonal version introduced in `"A Unified Multi-Task Semantic Communication  System for Multimodal Data" 
<https://ieeexplore.ieee.org/document/10431795>` _, 
by Guangyi Zhang, Qiyu Hu, Zhijin Qin, Yunlong Cai, Guanding Yu, Xiaoming Tao, (2024).
"""

VQA_NUMCLASS = 3129  # number of VQA class
MSA_NUMCLASS = 1     # number of MSA class
AVE_NUMCLASS = 29    # number of AVE class

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, policy=None, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        # print(key.shape)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        x, self.attn = self.attention(query, key, value, policy=policy, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, policy=None, mask=None, eps=1e-6):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            scores += (mask * -1e9)
            # attention weights
        if policy is None:
            p_attn = F.softmax(scores, dim = -1)
            return torch.matmul(p_attn, value), p_attn
        else:
            B, N1, _ = policy.size()
            B, H, N1, N2 = scores.size()
            attn_policy = policy.reshape(B, 1, 1, N2)
            temp = torch.zeros((B, 1, N1, N2), dtype=attn_policy.dtype, device=attn_policy.device)
            attn_policy = attn_policy + temp
            max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
            scores = scores - max_scores
            scores = scores.to(torch.float32).exp_() * attn_policy.to(torch.float32)
            p_attn = (scores + eps/N1) / (scores.sum(dim=-1, keepdim=True) + eps)
            return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, policy, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        attn_output = self.self_mha(x, x, x, None, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
     
        src_output = self.src_mha(x, memory, memory, policy, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate) 
                                            for _ in range(depth)])
        
    def forward(self, x, memory, policy= None, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, policy, look_ahead_mask, trg_padding_mask)  
        return x  

class ChannelDecoder(nn.Module):
    def __init__(self, 
                 num_symbols: int,
                 output_dim: int, 
                 ):
        super(ChannelDecoder, self).__init__()
        
        self.num_symbols = num_symbols
        self.output_dim = output_dim

        self.linear1 = nn.Linear(num_symbols, output_dim)
        self.linear2 = nn.Linear(output_dim, 256)
        self.linear3 = nn.Linear(256, output_dim)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.LN = nn.LayerNorm(output_dim, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        # output = self.layernorm(x1 + x5)
        output = self.LN(x5)

        return output

class ViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed_imgr = PatchEmbed(
            img_size=32, patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_imgc = PatchEmbed(
            img_size=224, patch_size=32, in_chans=in_chans, embed_dim=embed_dim)
        self.linear_embed_vqa = nn.Linear(2048, self.embed_dim)
        self.linear_embed_msa = nn.Linear(35, self.embed_dim)
        self.linear_embed_ave = nn.Linear(512, self.embed_dim)
        num_patches_imgr = self.patch_embed_imgr.num_patches
        num_patches_imgc = self.patch_embed_imgc.num_patches
        # TODO: Add the cls token
        self.cls_token = {}
        self.cls_token['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['ave'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.task_embedd = {}
        self.task_embedd['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['ave'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed_imgc = nn.Parameter(torch.zeros(1, num_patches_imgc + 1, embed_dim))
            self.pos_embed_imgr = nn.Parameter(torch.zeros(1, num_patches_imgr + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed_imgc = get_sinusoid_encoding_table(num_patches_imgc + 1, embed_dim)
            self.pos_embed_imgr = get_sinusoid_encoding_table(num_patches_imgr + 1, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        for key in self.cls_token.keys():
            trunc_normal_(self.cls_token[key], std=.02)
            trunc_normal_(self.task_embedd[key], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, ta_perform):
        '''
            input: (batch, channel, img_w, img_h)
        '''
        if ta_perform.startswith('vqa'):
            x = self.linear_embed_vqa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('msa'):
            x = self.linear_embed_msa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device)
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('ave'):
            # x.shape = (batch_size * time_steps, 7 * 7, 512)
            x = self.linear_embed_ave(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device)
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('img'):
            x = self.patch_embed_imgr(x) if ta_perform.startswith('imgr') else self.patch_embed_imgc(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x), dim=1)
            if ta_perform.startswith('imgr'):
                x = x + self.pos_embed_imgr.type_as(x).to(x.device).clone().detach()
            elif ta_perform.startswith('imgc'):
                x = x + self.pos_embed_imgc.type_as(x).to(x.device).clone().detach()
            x = torch.cat((x, task_embedd), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform):
        x = self.forward_features(x, ta_perform)
        return x



class SPTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.linear_embed = nn.Linear(74, self.embed_dim)
        self.linear_embed_ave = nn.Linear(128, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.task_embedd, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('msa'):
            x = self.linear_embed(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd.expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('ave'):
            x = self.linear_embed_ave(x).unsqueeze(1)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd.expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform=None):
        x = self.forward_features(x, ta_perform)
        return x
    

class UDeepSCNoSIC(nn.Module):
    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4, 
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):

        super().__init__()

        if mode=='tiny':
            text_embed_dim = 128
        elif mode=='small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512

        self.img_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        # bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/UDeepSC_Base/pretrained_models/bert-{mode}"
        bert_ckpt = f"prajjwal1/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt) 
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=speech_embed_dim,
                                depth=speech_encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.img_embed_dim = img_embed_dim
        self.text_embed_dim = text_embed_dim
        self.speech_embed_dim = speech_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.decoder = Decoder(depth=decoder_depth, embed_dim=decoder_embed_dim, 
                                num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, 
                                drop_rate=drop_rate)
        
        self.sigmoid_layer = nn.Sigmoid()

class UDeepSCOMA_msa(UDeepSCNoSIC):
    def __init__(self, num_symbols=16,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)

        
        self.num_symbols = num_symbols

        self.img_encoder_to_channel =  nn.Linear(self.img_embed_dim, self.num_symbols)
        self.text_encoder_to_channel = nn.Linear(self.text_embed_dim, self.num_symbols)
        self.spe_encoder_to_channel =  nn.Linear(self.speech_embed_dim, self.num_symbols)

        self.img_channel_decoder  = ChannelDecoder(self.num_symbols, self.img_embed_dim)
        self.text_channel_decoder = ChannelDecoder(self.num_symbols, self.text_embed_dim)
        self.spe_channel_decoder  = ChannelDecoder(self.num_symbols, self.speech_embed_dim)
        
        self.img_channel_to_decoder = nn.Linear(self.img_embed_dim, self.decoder_embed_dim)
        self.text_channel_to_decoder = nn.Linear(self.text_embed_dim, self.decoder_embed_dim)
        self.spe_channel_to_decoder = nn.Linear(self.speech_embed_dim, self.decoder_embed_dim)

        self.task_dict = nn.ModuleDict()
        self.task_dict['msa']  = nn.Embedding(25, self.decoder_embed_dim)

        self.head = nn.ModuleDict()
        self.head['msa'] = nn.Linear(self.decoder_embed_dim, MSA_NUMCLASS)
class UDeepSCSIC_msa(UDeepSCOMA_msa):
    def __init__(self, num_symbols=16,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)

class UDeepSCNoSIC_msa(UDeepSCNoSIC):
    def __init__(self, num_symbols=16,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)

        
        self.num_symbols = num_symbols

        self.msa_img_encoder_to_channel =  nn.Linear(self.img_embed_dim, self.num_symbols)
        self.msa_text_encoder_to_channel =  nn.Linear(self.text_embed_dim, self.num_symbols)
        self.msa_spe_encoder_to_channel =   nn.Linear(self.speech_embed_dim, self.num_symbols)

        self.msa_channel_to_decoder = nn.Linear(self.num_symbols, self.decoder_embed_dim)

        self.task_dict = nn.ModuleDict()
        self.task_dict['msa']  = nn.Embedding(25, self.decoder_embed_dim)

        self.head = nn.ModuleDict()
        self.head['msa'] = nn.Linear(self.decoder_embed_dim, MSA_NUMCLASS)
        
class UDeepSCNoSIC_ave(UDeepSCNoSIC):
    def __init__(self, num_symbols=16,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)

        
        self.num_symbols = num_symbols

        self.img_encoder_to_channel = nn.Linear(self.img_embed_dim, self.num_symbols)
        self.spe_encoder_to_channel = nn.Linear(self.speech_embed_dim, self.num_symbols)

        self.channel_to_decoder = nn.Linear(self.num_symbols, self.decoder_embed_dim)

        self.task_dict = nn.ModuleDict()
        self.task_dict['ave'] = nn.Embedding(25, self.decoder_embed_dim)

        self.head = nn.ModuleDict()
        self.head['ave'] = nn.Linear(self.decoder_embed_dim, AVE_NUMCLASS)
        
class UDeepSCOMA_ave(UDeepSCNoSIC):
    def __init__(self, num_symbols=16,
                *args, **kwargs
                ):

        super().__init__(*args, **kwargs)

        
        self.num_symbols = num_symbols

        self.img_encoder_to_channel = nn.Linear(self.img_embed_dim, self.num_symbols)
        self.spe_encoder_to_channel = nn.Linear(self.speech_embed_dim, self.num_symbols)

        self.img_channel_decoder  = ChannelDecoder(self.num_symbols, self.img_embed_dim)
        self.spe_channel_decoder  = ChannelDecoder(self.num_symbols, self.speech_embed_dim)
        
        self.img_channel_to_decoder = nn.Linear(self.img_embed_dim, self.decoder_embed_dim)
        self.spe_channel_to_decoder = nn.Linear(self.speech_embed_dim, self.decoder_embed_dim)

        self.task_dict = nn.ModuleDict()
        self.task_dict['ave'] = nn.Embedding(25, self.decoder_embed_dim)

        self.head = nn.ModuleDict()
        self.head['ave'] = nn.Linear(self.decoder_embed_dim, AVE_NUMCLASS)
        
class UDeepSCSIC_ave(UDeepSCOMA_ave):
    def __init__(self, num_symbols=16,
                 *args, **kwargs
                 ):

        super().__init__(*args, **kwargs)