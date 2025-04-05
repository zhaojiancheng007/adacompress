import math
from click import prompt
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from layers import RSTB
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from numpy import ceil

from compressai.models.utils import conv, deconv, update_registered_buffers
from einops import rearrange
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
from torch import Tensor

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

SCALES_LEVELS = 64



def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class Alignment(torch.nn.Module):
    """Image Alignment for model downsample requirement"""

    def __init__(self, divisor=64., mode='pad', padding_mode='replicate'):
        super().__init__()
        self.divisor = float(divisor)
        self.mode = mode
        self.padding_mode = padding_mode
        self._tmp_shape = None

    def extra_repr(self):
        s = 'divisor={divisor}, mode={mode}'
        if self.mode == 'pad':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    @staticmethod
    def _resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def _align(self, input):
        H, W = input.size()[-2:]
        H_ = int(ceil(H / self.divisor) * self.divisor)
        W_ = int(ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
            self._tmp_shape = None
            return input

        self._tmp_shape = input.size()
        if self.mode == 'pad':
            return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(H_, W_))

    def _resume(self, input, shape=None):
        if shape is not None:
            self._tmp_shape = shape
        if self._tmp_shape is None:
            return input

        if self.mode == 'pad':
            output = input[..., :self._tmp_shape[-2], :self._tmp_shape[-1]]
        elif self.mode == 'resize':
            output = self._resize(input, size=self._tmp_shape[-2:])

        return output

    def align(self, input):
        """align"""
        if input.dim() == 4:
            return self._align(input)

    def resume(self, input, shape=None):
        """resume"""
        if input.dim() == 4:
            return self._resume(input, shape)

    def forward(self, func, *args, **kwargs):
        pass
    
class SFMA(nn.Module):
    def __init__(self, in_dim=128, middle_dim=64,adapt_factor=1):
        super().__init__()
        self.factor = adapt_factor
        self.s_down1 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_down2 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_dw = nn.Conv2d(middle_dim, middle_dim, 5, 1, 2, groups=middle_dim)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
       
        self.f_down = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
        self.f_dw = nn.Conv2d(middle_dim, middle_dim, 3, 1, 1, groups=middle_dim)
        self.f_inter = nn.Conv2d(middle_dim, middle_dim, 1, 1, 0)
        self.sg = nn.Sigmoid()
    
    def forward(self, x):
        '''
        input: 
        x: intermediate feature 
        output:
        x_tilde: adapted feature
        '''
        _, _, H, W = x.shape

        y = torch.fft.rfft2(self.f_down(x), dim=(2, 3), norm='backward')
        y_amp = torch.abs(y)
        y_phs = torch.angle(y)
        # we only modulate the amplitude component for better training stability
        y_amp_modulation = self.f_inter(self.f_relu1(self.f_dw(y_amp)))
        y_amp = y_amp * self.sg(y_amp_modulation)
        y_real = y_amp * torch.cos(y_phs)
        y_img = y_amp * torch.sin(y_phs)
        y = torch.complex(y_real, y_img)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        
        f_modulate = self.f_up(self.f_relu2(y))
        s_modulate = self.s_up(self.s_relu(self.s_dw(self.s_down1(x)) * self.s_down2(x)))
        x_tilde = x + (s_modulate + f_modulate)*self.factor
        return x_tilde 

class SFMAUpsample(nn.Module):
    def __init__(self, in_dim=128, middle_dim=64, adapt_factor=1, scale_factor=1):
        """
        :param scale_factor: 1 表示不变，2 表示2倍上采样
        """
        super().__init__()
        self.factor = adapt_factor
        self.scale = scale_factor

        # 空间路径
        self.s_down1 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_down2 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_dw = nn.Conv2d(middle_dim, middle_dim, 5, 1, 2, groups=middle_dim)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)

        # 频率路径
        self.f_down = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
        self.f_dw = nn.Conv2d(middle_dim, middle_dim, 3, 1, 1, groups=middle_dim)
        self.f_inter = nn.Conv2d(middle_dim, middle_dim, 1, 1, 0)
        self.sg = nn.Sigmoid()

        # 上采样模块
        if self.scale > 1:
            self.upsample = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        """
        :param x: [B, C, H, W]
        :return: x_tilde: [B, C, H*scale, W*scale]
        """
        B, C, H, W = x.shape

        # === 频率路径 ===
        y = self.f_down(x)
        y_fft = torch.fft.rfft2(y, dim=(2, 3), norm='backward')
        y_amp = torch.abs(y_fft)
        y_phs = torch.angle(y_fft)

        y_amp_mod = self.f_inter(self.f_relu1(self.f_dw(y_amp)))
        y_amp = y_amp * self.sg(y_amp_mod)

        y_real = y_amp * torch.cos(y_phs)
        y_imag = y_amp * torch.sin(y_phs)
        y_mod = torch.complex(y_real, y_imag)
        y_ifft = torch.fft.irfft2(y_mod, s=(H, W), norm='backward')

        f_modulate = self.f_up(self.f_relu2(y_ifft))
        f_modulate = self.upsample(f_modulate)

        # === 空间路径 ===
        s_feat = self.s_down1(x) * self.s_down2(x)
        s_feat = self.s_dw(s_feat)
        s_feat = self.s_relu(s_feat)
        s_modulate = self.s_up(s_feat)
        s_modulate = self.upsample(s_modulate)

        # === 输出融合 ===
        x_up = self.upsample(x)  # 原始x上采样后相加
        x_tilde = x_up + (s_modulate + f_modulate) * self.factor
        return x_tilde

# class TIC_SFMA(nn.Module):
#     """
#     Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
#     """
#     def __init__(self, N=128, M=192,  input_resolution=(256,256), in_channel=3):
#         super().__init__()

#         depths = [2, 4, 6, 2, 2, 2]
#         num_heads = [8, 8, 8, 16, 16, 16]
#         window_size = 8
#         mlp_ratio = 2.
#         qkv_bias = True
#         qk_scale = None
#         drop_rate = 0.
#         attn_drop_rate = 0.
#         drop_path_rate = 0.1
#         norm_layer = nn.LayerNorm
#         use_checkpoint= False

#         self.weight_seg = nn.Parameter(torch.ones(3)) 
#         self.weight_human = nn.Parameter(torch.ones(3)) 
#         self.weight_sal = nn.Parameter(torch.ones(3)) 
#         self.weight_normal = nn.Parameter(torch.ones(3)) 



#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
#         self.encoder_sfmas = nn.Sequential(
#             SFMA(N),
#             SFMA(N),
#             SFMA(N),
#         )
#         self.decoder_sfmas  = nn.Sequential(
#             SFMA(N),
#             SFMA(N),
#             SFMA(N)
#         )
#         self.task_sfmas =  nn.Sequential(
#             SFMAUpsample(N,scale_factor=4),
#             SFMAUpsample(N,scale_factor=4),
#             SFMAUpsample(N,scale_factor=4),
#             SFMAUpsample(N,scale_factor=4),
#             SFMAUpsample(N,scale_factor=2),
#             SFMAUpsample(N,scale_factor=2),
#             SFMAUpsample(N,scale_factor=2),
#             SFMAUpsample(N,scale_factor=2),
#             SFMAUpsample(N,scale_factor=1),
#             SFMAUpsample(N,scale_factor=1),
#             SFMAUpsample(N,scale_factor=1),
#             SFMAUpsample(N,scale_factor=1),
#         )

#         self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
#         self.g_a1 = RSTB(dim=N,
#                         input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
#                         depth=depths[0],
#                         num_heads=num_heads[0],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint
#         )
#         self.g_a2 = conv(N, N, kernel_size=3, stride=2)
#         self.g_a3 = RSTB(dim=N,
#                         input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
#                         depth=depths[1],
#                         num_heads=num_heads[1],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint        )
#         self.g_a4 = conv(N, N, kernel_size=3, stride=2)
#         self.g_a5 = RSTB(dim=N,
#                         input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
#                         depth=depths[2],
#                         num_heads=num_heads[2],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint      )
#         self.g_a6 = conv(N, M, kernel_size=3, stride=2)
#         self.g_a7 = RSTB(dim=M,
#                         input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
#                         depth=depths[3],
#                         num_heads=num_heads[3],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint        )

#         self.h_a0 = conv(M, N, kernel_size=3, stride=2)
#         self.h_a1 = RSTB(dim=N,
#                          input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
#                          depth=depths[4],
#                          num_heads=num_heads[4],
#                          window_size=window_size//2,
#                          mlp_ratio=mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
#                          norm_layer=norm_layer,
#                          use_checkpoint=use_checkpoint     )
#         self.h_a2 = conv(N, N, kernel_size=3, stride=2)
#         self.h_a3 = RSTB(dim=N,
#                          input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
#                          depth=depths[5],
#                          num_heads=num_heads[5],
#                          window_size=window_size//2,
#                          mlp_ratio=mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
#                          norm_layer=norm_layer,
#                          use_checkpoint=use_checkpoint        )

#         depths = depths[::-1]
#         num_heads = num_heads[::-1]
#         self.h_s0 = RSTB(dim=N,
#                          input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
#                          depth=depths[0],
#                          num_heads=num_heads[0],
#                          window_size=window_size//2,
#                          mlp_ratio=mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
#                          norm_layer=norm_layer,
#                          use_checkpoint=use_checkpoint        )
#         self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
#         self.h_s2 = RSTB(dim=N,
#                          input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
#                          depth=depths[1],
#                          num_heads=num_heads[1],
#                          window_size=window_size//2,
#                          mlp_ratio=mlp_ratio,
#                          qkv_bias=qkv_bias, qk_scale=qk_scale,
#                          drop=drop_rate, attn_drop=attn_drop_rate,
#                          drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
#                          norm_layer=norm_layer,
#                          use_checkpoint=use_checkpoint        )
#         self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

#         self.entropy_bottleneck = EntropyBottleneck(N)
#         self.gaussian_conditional = GaussianConditional(None)
        
#         self.g_s0 = RSTB(dim=M,
#                         input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
#                         depth=depths[2],
#                         num_heads=num_heads[2],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint        )
#         self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
#         self.g_s2 = RSTB(dim=N,
#                         input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
#                         depth=depths[3],
#                         num_heads=num_heads[3],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint        )
#         self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
#         self.g_s4 = RSTB(dim=N,
#                         input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
#                         depth=depths[4],
#                         num_heads=num_heads[4],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint        )
#         self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
#         self.g_s6 = RSTB(dim=N,
#                         input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
#                         depth=depths[5],
#                         num_heads=num_heads[5],
#                         window_size=window_size,
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias, qk_scale=qk_scale,
#                         drop=drop_rate, attn_drop=attn_drop_rate,
#                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
#                         norm_layer=norm_layer,
#                         use_checkpoint=use_checkpoint        )
#         self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
#         self.init_std=0.02
      
#         self.apply(self._init_weights)  
#     def g_a(self, x, x_size=None):
#         attns = []
#         if x_size is None:
#             x_size = x.shape[2:4]
#         x = self.g_a0(x)

#         x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
#         x =self.encoder_sfmas[0](x)
#         attns.append(attn)
#         x = self.g_a2(x)

#         x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
#         x = self.encoder_sfmas[1](x)
#         attns.append(attn)
#         x = self.g_a4(x)

#         x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
#         x = self.encoder_sfmas[2](x)
#         attns.append(attn)
#         x = self.g_a6(x)

#         x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
#         attns.append(attn)
#         return x, attns

#     def g_s(self, x, x_size=None):
#         attns = []
#         if x_size is None:
#             x_size = (x.shape[2]*16, x.shape[3]*16)
#         x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
#         attns.append(attn)

#         x = self.g_s1(x)
#         x = self.decoder_sfmas[2](x)

#         seg_stage4 = self.task_sfmas[0](x)
#         human_stage4 = self.task_sfmas[1](x)
#         sal_stage4 = self.task_sfmas[2](x)
#         normals_stage4 = self.task_sfmas[3](x)

#         x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
#         attns.append(attn)


#         x = self.g_s3(x)
#         x = self.decoder_sfmas[1](x)

#         seg_stage3 = self.task_sfmas[4](x)
#         human_stage3 = self.task_sfmas[5](x)
#         sal_stage3 = self.task_sfmas[6](x)
#         normals_stage3 = self.task_sfmas[7](x)

#         x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
#         attns.append(attn)

#         x = self.g_s5(x)
#         x = self.decoder_sfmas[0](x)

#         seg_stage2 = self.task_sfmas[8](x)
#         human_stage2 = self.task_sfmas[9](x)
#         sal_stage2 = self.task_sfmas[10](x)
#         normals_stage2 = self.task_sfmas[11](x)

#         x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
#         attns.append(attn)

#         # combine seg
#         w_seg = F.softmax(self.weight_seg, dim=0)
#         seg_fused = w_seg[0]*seg_stage4 + w_seg[1]*seg_stage3 + w_seg[2]*seg_stage2
#         seg_x, seg_attn = self.g_s6(seg_fused, (x_size[0]//2, x_size[1]//2))
#         seg_x = self.g_s7(seg_x)

#         # combine human
#         w_human = F.softmax(self.weight_human, dim=0)
#         human_fused = w_human[0]*human_stage4 + w_human[1]*human_stage3 + w_human[2]*human_stage2
#         human_x, seg_attn = self.g_s6(human_fused, (x_size[0]//2, x_size[1]//2))
#         human_x = self.g_s7(human_x)

#         # combine sal
#         w_sal = F.softmax(self.weight_sal, dim=0)
#         sal_fused = w_sal[0]*sal_stage4 + w_sal[1]*sal_stage3 + w_sal[2]*sal_stage2
#         sal_x, seg_attn = self.g_s6(sal_fused, (x_size[0]//2, x_size[1]//2))
#         sal_x = self.g_s7(sal_x)

#         # combine normal
#         w_normal = F.softmax(self.weight_normal, dim=0)
#         normals_fused = w_normal[0]*normals_stage4 + w_normal[1]*normals_stage3 + w_normal[2]*normals_stage2
#         normals_x, seg_attn = self.g_s6(normals_fused, (x_size[0]//2, x_size[1]//2))
#         normals_x = self.g_s7(normals_x)

#         x = self.g_s7(x)

#         # collect features. for evaluation
#         features = {
#             "seg": {
#                 "stage2": seg_stage2.detach(),
#                 "stage3": seg_stage3.detach(),
#                 "stage4": seg_stage4.detach(),
#                 "fused": seg_fused.detach()
#             },
#             "human_parts": {
#                 "stage2": human_stage2.detach(),
#                 "stage3": human_stage3.detach(),
#                 "stage4": human_stage4.detach(),
#                 "fused": human_fused.detach()
#             },
#             "sal": {
#                 "stage2": sal_stage2.detach(),
#                 "stage3": sal_stage3.detach(),
#                 "stage4": sal_stage4.detach(),
#                 "fused": sal_fused.detach()
#             },
#             "normals": {
#                 "stage2": normals_stage2.detach(),
#                 "stage3": normals_stage3.detach(),
#                 "stage4": normals_stage4.detach(),
#                 "fused": normals_fused.detach()
#             }
#         }

#         return x, seg_x, human_x, sal_x, normals_x, features

#     def h_a(self, x, x_size=None):
#         if x_size is None:
#             x_size = (x.shape[2]*16, x.shape[3]*16)
#         x = self.h_a0(x)
#         x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
#         x = self.h_a2(x)
#         x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
#         return x

#     def h_s(self, x, x_size=None):
#         if x_size is None:
#             x_size = (x.shape[2]*64, x.shape[3]*64)
#         x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
#         x = self.h_s1(x)
#         x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
#         x = self.h_s3(x)
#         return x

#     def aux_loss(self):
#         """Return the aggregated loss over the auxiliary entropy bottleneck
#         module(s).
#         """
#         aux_loss = sum(
#             m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
#         )
#         return aux_loss

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}

#     def forward(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         y, attns_a = self.g_a(x)
#         z = self.h_a(y)
#         _, z_likelihoods = self.entropy_bottleneck(z)
#         z_offset = self.entropy_bottleneck._get_medians()
#         z_tmp = z - z_offset
#         z_hat = ste_round(z_tmp) + z_offset        
#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
#         y_hat = ste_round(y-means_hat)+means_hat  
#         x_hat, x_seg, x_human, x_sal, x_normal, features = self.g_s(y_hat)
     
#         return {
#             "x_hat": x_hat,
#             "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
#             "semseg": x_seg,
#             "human_parts": x_human,
#             'sal': x_sal,
#             'normals': x_normal,
#             "features": features
#         }

#     def update(self, scale_table=None, force=False):
#         """Updates the entropy bottleneck(s) CDF values.

#         Needs to be called once after training to be able to later perform the
#         evaluation with an actual entropy coder.

#         Args:
#             scale_table (bool): (default: None)  
#             force (bool): overwrite previous values (default: False)

#         Returns:
#             updated (bool): True if one of the EntropyBottlenecks was updated.

#         """
#         if scale_table is None:
#             scale_table = get_scale_table()
#         self.gaussian_conditional.update_scale_table(scale_table, force=force)

#         updated = False
#         for m in self.children():
#             if not isinstance(m, EntropyBottleneck):
#                 continue
#             rv = m.update(force=force)
#             updated |= rv
#         return updated

#     def load_state_dict(self, state_dict, strict=True):
#         # Dynamically update the entropy bottleneck buffers related to the CDFs
#         update_registered_buffers(
#             self.entropy_bottleneck,
#             "entropy_bottleneck",
#             ["_quantized_cdf", "_offset", "_cdf_length"],
#             state_dict,
#         )
#         update_registered_buffers(
#             self.gaussian_conditional,
#             "gaussian_conditional",
#             ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
#             state_dict,
#         )
#         super().load_state_dict(state_dict, strict=strict)

#     @classmethod
#     def from_state_dict(cls, state_dict):
#         """Return a new model instance from `state_dict`."""
#         N = state_dict["g_a0.weight"].size(0)
#         M = state_dict["g_a6.weight"].size(0)
#         net = cls(N, M)
#         net.load_state_dict(state_dict)
#         return net

#     def compress(self, x):
#         x_size = (x.shape[2], x.shape[3])
#         y, attns_a = self.g_a(x, x_size)
#         z = self.h_a(y, x_size)

#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

#         gaussian_params = self.h_s(z_hat, x_size)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

#     def decompress(self, strings, shape):
#         assert isinstance(strings, list) and len(strings) == 2
#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_hat = self.gaussian_conditional.decompress(
#             strings[0], indexes, means=means_hat
#         )
#         x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat}

class TIC_SFMA(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192,  input_resolution=(256,256), in_channel=3):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False



        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.encoder_sfmas = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N)
            
        )
        self.decoder_sfmas  = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N)
            
        )

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint      )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint     )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        self.init_std=0.02
      
        self.apply(self._init_weights)  
    def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)

        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x =self.encoder_sfmas[0](x)
        attns.append(attn)
        x = self.g_a2(x)

        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.encoder_sfmas[1](x)
        attns.append(attn)
        x = self.g_a4(x)

        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.encoder_sfmas[2](x)
        attns.append(attn)
        x = self.g_a6(x)

        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)

        x = self.g_s1(x)
        x = self.decoder_sfmas[2](x)
        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)


        x = self.g_s3(x)
        x = self.decoder_sfmas[1](x)
        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)

        x = self.g_s5(x)
        x = self.decoder_sfmas[0](x)
        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)

        x = self.g_s7(x)
        return x, attns

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y-means_hat)+means_hat  
        x_hat, attns_s = self.g_s(y_hat)
     
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "attn_a": attns_a,
            "attn_s": attns_s
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class DFFN(nn.Module):
    def __init__(self, dim=128, ffn_expansion_factor=0.5, bias=True, scale=1):

        super(DFFN, self).__init__()

        self.scale = scale

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        if self.scale > 1:
            self.upsample = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        x_residul = x
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_up = self.upsample(x+x_residul)

        return x_up

class GatedFusion(nn.Module):
    def __init__(self, in_channels, num_scales):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, num_scales, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features (List[Tensor]): list of feature maps, each with shape [B, C, H, W]
        Returns:
            fused feature map: [B, C, H, W]
        """
        B, C, H, W = features[0].shape
        concat = torch.cat(features, dim=1)  # [B, C*num_scales, H, W]
        gate = self.gate(concat)            # [B, num_scales, H, W]
        gate = gate.unsqueeze(2)            # [B, num_scales, 1, H, W]
        stacked = torch.stack(features, dim=1)  # [B, num_scales, C, H, W]
        fused = (gate * stacked).sum(dim=1)     # [B, C, H, W]
        return fused

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            scaled feature: [B, C, H, W]
        """
        attention = self.fc(x)  # [B, C, 1, 1]
        return x * attention

class CrossScaleAttention(nn.Module):
    def __init__(self, channels, num_scales):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels * num_scales, channels, kernel_size=1)
        self.value = nn.Conv2d(channels * num_scales, channels, kernel_size=1)
        self.final = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features (List[Tensor]): [F1, F2, F3], each [B, C, H, W]
        Returns:
            fused feature: [B, C, H, W]
        """
        B, C, H, W = features[0].shape
        upsampled = [F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False) for f in features]
        q = self.query(upsampled[0])  # Use first as query
        kv = torch.cat(upsampled, dim=1)  # [B, C*num_scales, H, W]
        k = self.key(kv)
        v = self.value(kv)

        attn = F.softmax((q * k).sum(1, keepdim=True), dim=2)  # [B, 1, H, W]
        out = attn * v
        return self.final(out)

class TIC_Fusion(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192,  input_resolution=(256,256), in_channel=3):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.encoder_sfmas = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N),
        )
        self.decoder_sfmas  = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N)
        )
        self.task_sfmas  = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
        )
        self.task_heads_sfmas = nn.ModuleDict({
            "seg": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, 3, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
            "human": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, 3, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
            "sal": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, 3, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
            "normal": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, 3, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
        })
        self.task_gate_sfmas = nn.ParameterDict({
            "seg": nn.Sequential(
            nn.Conv2d(N * 3, N, kernel_size=1),
            nn.Sigmoid()),
            "human":nn.Sequential(
            nn.Conv2d(N * 3, N, kernel_size=1),
            nn.Sigmoid()),
            "sal": nn.Sequential(
            nn.Conv2d(N * 3, N, kernel_size=1),
            nn.Sigmoid()),
            "normal": nn.Sequential(
            nn.Conv2d(N * 3, N, kernel_size=1),
            nn.Sigmoid()),
        })

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint      )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint     )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        self.init_std=0.02
      
        self.apply(self._init_weights)  
    def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)

        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x =self.encoder_sfmas[0](x)
        attns.append(attn)
        x = self.g_a2(x)

        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.encoder_sfmas[1](x)
        attns.append(attn)
        x = self.g_a4(x)

        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.encoder_sfmas[2](x)
        attns.append(attn)
        x = self.g_a6(x)

        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)

        x = self.g_s1(x)
        x = self.decoder_sfmas[2](x)

        seg_stage4 = self.task_sfmas[0](x) # H/16
        human_stage4 = self.task_sfmas[1](x)
        sal_stage4 = self.task_sfmas[2](x)
        normals_stage4 = self.task_sfmas[3](x)

        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)


        x = self.g_s3(x)
        x = self.decoder_sfmas[1](x)

        seg_stage3 = self.task_sfmas[4](x) #H/8
        human_stage3 = self.task_sfmas[5](x)
        sal_stage3 = self.task_sfmas[6](x)
        normals_stage3 = self.task_sfmas[7](x)

        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)

        x = self.g_s5(x)
        x = self.decoder_sfmas[0](x)

        seg_stage2 = self.task_sfmas[8](x) # H/4
        human_stage2 = self.task_sfmas[9](x)
        sal_stage2 = self.task_sfmas[10](x)
        normals_stage2 = self.task_sfmas[11](x)

        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)

        # combine seg
        seg_fused = self.fuse(seg_stage4, seg_stage3, seg_stage2,'seg')
        seg_residual = self.task_heads_sfmas["seg"](seg_fused) 
        

        # combine human
        human_fused = self.fuse(human_stage4, human_stage3, human_stage2,'human')
        human_residual = self.task_heads_sfmas["human"](human_fused) 

        # combine sal
        sal_fused = self.fuse(sal_stage4, sal_stage3, sal_stage2,'sal')
        sal_residual = self.task_heads_sfmas["sal"](sal_fused) 


        # combine normal
        normal_fused = self.fuse(normals_stage4, normals_stage3, normals_stage2,'normal')
        normal_residual = self.task_heads_sfmas["normal"](normal_fused) 

        x = self.g_s7(x)

        # collect features. for evaluation
        features = {
            "seg": {
                "stage2": seg_stage2.detach(),
                "stage3": seg_stage3.detach(),
                "stage4": seg_stage4.detach(),
                "fused": seg_fused.detach()
            },
            "human_parts": {
                "stage2": human_stage2.detach(),
                "stage3": human_stage3.detach(),
                "stage4": human_stage4.detach(),
                "fused": human_fused.detach()
            },
            "sal": {
                "stage2": sal_stage2.detach(),
                "stage3": sal_stage3.detach(),
                "stage4": sal_stage4.detach(),
                "fused": sal_fused.detach()
            },
            "normals": {
                "stage2": normals_stage2.detach(),
                "stage3": normals_stage3.detach(),
                "stage4": normals_stage4.detach(),
                "fused": normal_fused.detach()
            }
        }

        return x, seg_residual, human_residual, sal_residual, normal_residual, features

    def fuse(self, f_s4, f_s3, f_s2, task):
        # f_s4 --> shape of f_s2
        features = [f_s4, f_s3, f_s2]

        target_size = f_s2.shape[2:]
        upsampled_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                     for f in features]
        concat = torch.cat(upsampled_features, dim=1)  # [B, C*num_scales, H, W]
        gate = self.task_gate_sfmas[task](concat)            # [B, C, H, W]
        gate = gate.unsqueeze(1)            # [B, num_scales, 1, H, W]
        stacked = torch.stack(upsampled_features, dim=1)  # [B, num_scales, C, H, W]
        fused = (gate * stacked).sum(dim=1)     # [B, C, H, W]
        return fused
    
    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y-means_hat)+means_hat  
        x_hat, seg_residual, human_residual, sal_residual, normal_residual, features = self.g_s(y_hat)
     
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "semseg": x_hat + seg_residual,
            "human_parts": x_hat + human_residual,
            'sal': x_hat + sal_residual,
            'normals': x_hat + normal_residual,
            "features": features
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class TIC_SUM(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192,  input_resolution=(256,256), in_channel=3):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.encoder_sfmas = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N),
        )
        self.decoder_sfmas  = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N)
        )
        self.task_sfmas  = nn.Sequential(
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
            SFMA(N),
        )
        self.task_heads_sfmas = nn.ModuleDict({
            "seg": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, N//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
            "human": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, N//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
            "sal": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, N//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
            "normal": nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(N, N//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(N//2, 3, kernel_size=3, padding=1)
            ),
        })
        # self.task_gate_sfmas = nn.ParameterDict({
        #     "seg": nn.Sequential(
        #     nn.Conv2d(N * 3, N, kernel_size=1),
        #     nn.Sigmoid()),
        #     "human":nn.Sequential(
        #     nn.Conv2d(N * 3, N, kernel_size=1),
        #     nn.Sigmoid()),
        #     "sal": nn.Sequential(
        #     nn.Conv2d(N * 3, N, kernel_size=1),
        #     nn.Sigmoid()),
        #     "normal": nn.Sequential(
        #     nn.Conv2d(N * 3, N, kernel_size=1),
        #     nn.Sigmoid()),
        # })

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint      )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint     )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)
        self.init_std=0.02
      
        self.apply(self._init_weights)  
    def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)

        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x =self.encoder_sfmas[0](x)
        attns.append(attn)
        x = self.g_a2(x)

        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.encoder_sfmas[1](x)
        attns.append(attn)
        x = self.g_a4(x)

        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.encoder_sfmas[2](x)
        attns.append(attn)
        x = self.g_a6(x)

        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)

        x = self.g_s1(x)
        x = self.decoder_sfmas[2](x)

        seg_stage4 = self.task_sfmas[0](x) # H/16
        human_stage4 = self.task_sfmas[1](x)
        sal_stage4 = self.task_sfmas[2](x)
        normals_stage4 = self.task_sfmas[3](x)

        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)


        x = self.g_s3(x)
        x = self.decoder_sfmas[1](x)

        seg_stage3 = self.task_sfmas[4](x) #H/8
        human_stage3 = self.task_sfmas[5](x)
        sal_stage3 = self.task_sfmas[6](x)
        normals_stage3 = self.task_sfmas[7](x)

        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)

        x = self.g_s5(x)
        x = self.decoder_sfmas[0](x)

        seg_stage2 = self.task_sfmas[8](x) # H/4
        human_stage2 = self.task_sfmas[9](x)
        sal_stage2 = self.task_sfmas[10](x)
        normals_stage2 = self.task_sfmas[11](x)

        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)

        # combine seg
        seg_fused = self.sum(seg_stage4, seg_stage3, seg_stage2,'seg')
        seg_residual = self.task_heads_sfmas["seg"](seg_fused) 
        

        # combine human
        human_fused = self.sum(human_stage4, human_stage3, human_stage2,'human')
        human_residual = self.task_heads_sfmas["human"](human_fused) 

        # combine sal
        sal_fused = self.sum(sal_stage4, sal_stage3, sal_stage2,'sal')
        sal_residual = self.task_heads_sfmas["sal"](sal_fused) 


        # combine normal
        normal_fused = self.sum(normals_stage4, normals_stage3, normals_stage2,'normal')
        normal_residual = self.task_heads_sfmas["normal"](normal_fused) 

        x = self.g_s7(x)

        # collect features. for evaluation
        features = {
            "seg": {
                "stage2": seg_stage2.detach(),
                "stage3": seg_stage3.detach(),
                "stage4": seg_stage4.detach(),
                "fused": seg_fused.detach()
            },
            "human_parts": {
                "stage2": human_stage2.detach(),
                "stage3": human_stage3.detach(),
                "stage4": human_stage4.detach(),
                "fused": human_fused.detach()
            },
            "sal": {
                "stage2": sal_stage2.detach(),
                "stage3": sal_stage3.detach(),
                "stage4": sal_stage4.detach(),
                "fused": sal_fused.detach()
            },
            "normals": {
                "stage2": normals_stage2.detach(),
                "stage3": normals_stage3.detach(),
                "stage4": normals_stage4.detach(),
                "fused": normal_fused.detach()
            }
        }

        return x, seg_residual, human_residual, sal_residual, normal_residual, features

    def fuse(self, f_s4, f_s3, f_s2, task):
        # f_s4 --> shape of f_s2
        features = [f_s4, f_s3, f_s2]

        target_size = f_s2.shape[2:]
        upsampled_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                     for f in features]
        concat = torch.cat(upsampled_features, dim=1)  # [B, C*num_scales, H, W]
        gate = self.task_gate_sfmas[task](concat)            # [B, C, H, W]
        gate = gate.unsqueeze(1)            # [B, num_scales, 1, H, W]
        stacked = torch.stack(upsampled_features, dim=1)  # [B, num_scales, C, H, W]
        fused = (gate * stacked).sum(dim=1)     # [B, C, H, W]
        return fused
    
    def sum(self, f_s4, f_s3, f_s2, task):
        # f_s4 --> shape of f_s2
        features = [f_s4, f_s3, f_s2]

        target_size = f_s2.shape[2:]
        upsampled_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                     for f in features]
        sumed_feature = upsampled_features[0] + upsampled_features[1] + upsampled_features[2]
        # concat = torch.cat(upsampled_features, dim=1)  # [B, C*num_scales, H, W]
        # gate = self.task_gate_sfmas[task](concat)            # [B, C, H, W]
        # gate = gate.unsqueeze(1)            # [B, num_scales, 1, H, W]
        # stacked = torch.stack(upsampled_features, dim=1)  # [B, num_scales, C, H, W]
        # fused = (gate * stacked).sum(dim=1)     # [B, C, H, W]
        return sumed_feature
    
    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y-means_hat)+means_hat  
        x_hat, seg_residual, human_residual, sal_residual, normal_residual, features = self.g_s(y_hat)
     
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "semseg": x_hat + seg_residual,
            "human_parts": x_hat + human_residual,
            'sal': x_hat + sal_residual,
            'normals': x_hat + normal_residual,
            "features": features,
            "residuals": [seg_residual, human_residual,sal_residual,normal_residual]
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}






def plot_tsne(features_dict, save_path):
    """
    Args:
        features_dict: dict of {task_name: list of [C, H, W] torch.Tensor}
        save_path: output folder
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    os.makedirs(save_path, exist_ok=True)
    
    all_feat = []
    all_label = []
    task_names = list(features_dict.keys())
    for i, task in enumerate(task_names):
        for feat in features_dict[task]:
            pooled = F.adaptive_avg_pool2d(feat.unsqueeze(0), 1).squeeze()  # [C]
            all_feat.append(pooled.cpu().numpy())
            all_label.append(i)
    all_feat = np.stack(all_feat)
    all_feat = StandardScaler().fit_transform(all_feat)
    print('#'*10, all_feat.shape, '#'*10)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, init="pca", learning_rate="auto")
    feat_2d = tsne.fit_transform(all_feat)

    plt.figure(figsize=(8, 6))
    for i, task in enumerate(task_names):
        idx = np.array(all_label) == i
        plt.scatter(feat_2d[idx, 0], feat_2d[idx, 1], label=task, alpha=0.6)
    plt.legend()
    plt.title("t-SNE of Multi-Task Features")
    plt.savefig(os.path.join(save_path, "tsne.png"))
    plt.close()

def featuremap_to_heatmap(feat_tensor):
    """
    Args:
        feat_tensor: torch.Tensor of shape [C, H, W]
    Returns:
        heatmap: np.ndarray of shape [H, W], dtype uint8
    """
    heatmap = feat_tensor.mean(0).cpu().detach()  # [H, W]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    return (heatmap.numpy() * 255).astype(np.uint8)

def overlay_heatmap_on_image(img_tensor, feat_tensor, save_path, name='overlay'):
    """
    Args:
        img_tensor: [3, H, W] torch.Tensor
        feat_tensor: [C, H, W] torch.Tensor
        save_path: directory
        name: output filename
    """
    os.makedirs(save_path, exist_ok=True)
    img = img_tensor.cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)

    heatmap = featuremap_to_heatmap(feat_tensor)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(os.path.join(save_path, f"{name}.png"), overlay)

def analyze_features(model_output, input_image, save_path):
    """
    Args:
        model_output: dict, include model_output["features"]
        input_image: [B, 3, H, W] image tensor
    """
    features = model_output["features"]

    # 保存热图
    for task in features:
        fused_feat = features[task]["fused"][1]  # [C, H, W]
        overlay_heatmap_on_image(input_image, fused_feat, os.path.join(save_path, "heatmap"), name=task)

    # 生成 t-SNE 分布图
    # task_feat_dict = {k: [features[k]["fused"][0]] for k in features}
    # plot_tsne(task_feat_dict, os.path.join(save_path, "tsne"))