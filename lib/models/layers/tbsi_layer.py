from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock


class TBSILayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.ca_s2t_v2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2s_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_s2t_i2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2s_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2t_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2t_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

    def forward(self, x_v, x_i, lens_z):
        # x_v: [B, N, C], N = 320
        # x_i: [B, N, C], N = 320
        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2) #NOTE: [B,lens_z,2c] 将两个模态的 z （template） 最后一维链接
        fused_t = self.t_fusion(fused_t)  # [B, 64, C] # NOTE: [B,lens_z,c] paper 中的 Z_m
        
        # NOTE: q: fused_t  k and v : TIR 的 search region . 使用 TIR search region 来增强 fused_t 
        fused_t = self.ca_s2t_i2f(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :] 
        
        # NOTE: q: RGB 的 search region， k and v ： fused_t . 增强 RGB search region 中的 目标区域 X_mtr
        temp_x_v = self.ca_t2s_f2v(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1))[:, lens_z:, :]
        # NOTE: q: fused_t k and v : RGB 的 searh region. 使用 RGB search region 来增强 fused_t
        fused_t = self.ca_s2t_v2f(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        # NOTE: q: TIR 的 search region， k and v ： fused_t . 增强 TIR search region 中的 目标区域 X_rtm
        temp_x_i = self.ca_t2s_f2i(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1))[:, lens_z:, :]

        x_v[:, lens_z:, :] = temp_x_v
        x_i[:, lens_z:, :] = temp_x_i
        # NOTE: 更新rgb 的 template  使用 q： rgb template  k and v： fused_t
        x_v[:, :lens_z, :] = self.ca_t2t_f2v(torch.cat([x_v[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]

        # NOTE: 同理更新 TIR 的template
        x_i[:, :lens_z, :] = self.ca_t2t_f2i(torch.cat([x_i[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]

        return x_v, x_i
