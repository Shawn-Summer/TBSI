# TBSI 解读

## 模型结构

`base_backbone.py` 和 OStrack 是完全一致

### `lib/models/tbsi_track/vit_tbsi_care.py` 多模态的backbone


```python
def forward_features(self, z, x):
    # NOTE: z 和 x [2,B,C,H,W]
    B, H, W = x[0].shape[0], x[0].shape[2], x[0].shape[3]
    
    # NOTE: search region 和 template 的补丁会被映射到 同一个特征空间，此时的向量距离或相似度才能真实反映原始补丁的视觉相似性
    x_v = self.patch_embed(x[0])
    z_v = self.patch_embed(z[0])
    x_i = self.patch_embed(x[1])
    z_i = self.patch_embed(z[1]) 

    if self.add_cls_token:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.cls_pos_embed

    # Visible and infrared data share the positional encoding and other parameters in ViT
    z_v += self.pos_embed_z
    x_v += self.pos_embed_x
    z_i += self.pos_embed_z
    x_i += self.pos_embed_x

    if self.add_sep_seg:
        x += self.search_segment_pos_embed
        z += self.template_segment_pos_embed

    x_v = combine_tokens(z_v, x_v, mode=self.cat_mode)
    x_i = combine_tokens(z_i, x_i, mode=self.cat_mode)
    if self.add_cls_token:
        x = torch.cat([cls_tokens, x], dim=1)

    x_v = self.pos_drop(x_v)
    x_i = self.pos_drop(x_i)

    lens_z = self.pos_embed_z.shape[1]
    lens_x = self.pos_embed_x.shape[1]
    
    tbsi_index = 0
    for i, blk in enumerate(self.blocks): # NOTE: 这里是先进入 transformer blocks 然后 进入tbsi_layers 进行融合
        x_v = blk(x_v)
        x_i = blk(x_i) # NOTE: 这里不同模态还是享用同一参数
        if self.tbsi_loc is not None and i in self.tbsi_loc:
            x_v, x_i = self.tbsi_layers[tbsi_index](x_v, x_i, lens_z)
            tbsi_index += 1

    x_v = recover_tokens(x_v, lens_z, lens_x, mode=self.cat_mode)
    x_i = recover_tokens(x_i, lens_z, lens_x, mode=self.cat_mode)
    x = torch.cat([x_v, x_i], dim=1)
    # NOTE: x_v 是[B,L_vz+L_vx,C']
    # NOTE: x_i 是[B,L_iz+L_ix,C']
    # NOTE: x 是 x_v 和 x_i 在 dim=1 concat 所以shape是 [B,L_vz+L_vx+L_iz+L_ix,C']
    
    aux_dict = {"attn": None}
    return self.norm(x), aux_dict
```

可以看到:
1. z和x的输入都是两个模态的，shape是 [2,B,C,H,W]
2. 他的两个模态v和i都使用完全一样的模型，其中blk使用的是基础的attention模块，唯一的区别在于特征融合，即新增了`self.tbsi_layers[a]()`这个融合模块

### 特征融合模块

```python
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
```

注意到：这个融合模块穿插在blks中,用以更新blks之间的 TIR and RGB 的 search region 和 template

### 检测头，位于文件`lib/models/tbsi_track.py`

```python
    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        num_template_token = 64
        num_search_token = 256
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :] # NOTE: RGB的 search region
        enc_opt2 = cat_feature[:, -num_search_token:, :] # NOTE: TIR的 search region
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2) # NOTE: [B,L_x,2c]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous() # NOTE: [B,1,2c,L_x]
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s) # NOTE: [B,2c,H,W]
        opt_feat = self.tbsi_fuse_search(opt_feat) # NOTE: [B,c,H,W] 实际上是个 卷积层 将 channel 从2c 变为 c 且 H 和 W 不变

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

```

具体来说进入 head 之前需要将 两个模态的 search region 拿出来，然后fuse一下，feed 到head中
