# TBSI 解读

## 环境

```bash
conda create -n tbsi python=3.9
conda activate tbsi
bash install.sh
```

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

## 配置文件

在`experiments/tbsi_track`中有2份配置文件

- `vitb_256_tbsi_32x1_1e4_lasher_15ep_sot.yaml`：这里32表示batch_size，1 and 1e4 表示学习率，15ep 表示epoch=15，sot 表示使用sot预训练模型
- `vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k.yam`：这里32表示batch_size，4 and 4e4 表示学习率，15ep 表示epoch=15，in1k 表示使用imageNet-1k预训练模型

这里做一个压力较小的配置：
- `vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k.yaml`：即把batch_size 改为4


### 单卡训练

```bash
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k --mode single
```

则返回如图：

```bash
/data1/zhh/xiaxulong/TBSI/lib/train/../../lib/train/trainers/ltr_trainer.py:92: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
[train: 1, 50 / 15000] FPS: 9.6 (24.4)  ,  DataTime: 0.013 (0.014)  ,  ForwardTime: 0.390  ,  TotalTime: 0.417  ,  Loss/total: 18.64630  ,  Loss/giou: 1.27215  ,  Loss/l1: 0.32796  ,  Loss/location: 14.46217  ,  IoU: 0.06257
[train: 1, 100 / 15000] FPS: 13.9 (25.0)  ,  DataTime: 0.009 (0.014)  ,  ForwardTime: 0.265  ,  TotalTime: 0.288  ,  Loss/total: 13.33946  ,  Loss/giou: 1.23385  ,  Loss/l1: 0.31899  ,  Loss/location: 9.27680  ,  IoU: 0.06321
[train: 1, 150 / 15000] FPS: 16.3 (25.5)  ,  DataTime: 0.008 (0.014)  ,  ForwardTime: 0.224  ,  TotalTime: 0.245  ,  Loss/total: 10.90748  ,  Loss/giou: 1.17087  ,  Loss/l1: 0.29255  ,  Loss/location: 7.10299  ,  IoU: 0.08298
[train: 1, 200 / 15000] FPS: 17.8 (24.8)  ,  DataTime: 0.007 (0.014)  ,  ForwardTime: 0.203  ,  TotalTime: 0.224  ,  Loss/total: 9.45336  ,  Loss/giou: 1.12395  ,  Loss/l1: 0.26661  ,  Loss/location: 5.87242  ,  IoU: 0.10698

```

由于每个 epoch 加载 60000 个sample，每个batch_size 为 4,则一个epoch 需要 15000 个batch。

### 多卡训练

如果使用多卡训练则改用`multiple`模式，并且节点数设置为2，如下所示：

```bash
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k --mode multiple --nproc_per_node 2
```

```bash
  with autocast():
[train: 1, 50 / 7500] FPS: 6.8 (13.3)  ,  DataTime: 0.042 (0.013)  ,  ForwardTime: 0.532  ,  TotalTime: 0.587  ,  Loss/total: 18.02869  ,  Loss/giou: 1.21229  ,  Loss/l1: 0.30613  ,  Loss/location: 14.07343  ,  IoU: 0.07183
[train: 1, 50 / 7500] FPS: 6.8 (13.1)  ,  DataTime: 0.013 (0.012)  ,  ForwardTime: 0.562  ,  TotalTime: 0.586  ,  Loss/total: 17.41292  ,  Loss/giou: 1.26302  ,  Loss/l1: 0.31521  ,  Loss/location: 13.31080  ,  IoU: 0.05621
[train: 1, 100 / 7500] FPS: 9.0 (13.3)  ,  DataTime: 0.024 (0.013)  ,  ForwardTime: 0.407  ,  TotalTime: 0.444  ,  Loss/total: 12.44758  ,  Loss/giou: 1.15054  ,  Loss/l1: 0.29104  ,  Loss/location: 8.69131  ,  IoU: 0.07693
[train: 1, 100 / 7500] FPS: 9.0 (13.2)  ,  DataTime: 0.009 (0.012)  ,  ForwardTime: 0.422  ,  TotalTime: 0.443  ,  Loss/total: 12.08502  ,  Loss/giou: 1.16829  ,  Loss/l1: 0.28592  ,  Loss/location: 8.31886  ,  IoU: 0.07768
[train: 1, 150 / 7500] FPS: 10.1 (13.3)  ,  DataTime: 0.007 (0.013)  ,  ForwardTime: 0.376  ,  TotalTime: 0.396  ,  Loss/total: 9.79286  ,  Loss/giou: 1.10193  ,  Loss/l1: 0.25359  ,  Loss/location: 6.32103  ,  IoU: 0.11215
[train: 1, 150 / 7500] FPS: 10.1 (13.3)  ,  DataTime: 0.017 (0.014)  ,  ForwardTime: 0.365  ,  TotalTime: 0.396  ,  Loss/total: 10.00085  ,  Loss/giou: 1.08524  ,  Loss/l1: 0.25449  ,  Loss/location: 6.55794  ,  IoU: 0.11129
[train: 1, 200 / 7500] FPS: 10.7 (13.2)  ,  DataTime: 0.007 (0.013)  ,  ForwardTime: 0.353  ,  TotalTime: 0.372  ,  Loss/total: 8.45294  ,  Loss/giou: 1.04953  ,  Loss/l1: 0.22545  ,  Loss/location: 5.22664  ,  IoU: 0.14804
[train: 1, 200 / 7500] FPS: 10.7 (13.0)  ,  DataTime: 0.014 (0.014)  ,  ForwardTime: 0.344  ,  TotalTime: 0.372  ,  Loss/total: 8.61675  ,  Loss/giou: 1.04251  ,  Loss/l1: 0.22715  ,  Loss/location: 5.39601  ,  IoU: 0.14491
[train: 1, 250 / 7500] FPS: 11.2 (13.5)  ,  DataTime: 0.013 (0.014)  ,  ForwardTime: 0.332  ,  TotalTime: 0.358  ,  Loss/total: 7.74029  ,  Loss/giou: 1.01666  ,  Loss/l1: 0.20987  ,  Loss/location: 4.65761  ,  IoU: 0.16931
```

由于每个 epoch 加载 60000 个sample，每个batch_size 为 4,则一个epoch 需要 15000 个batch，但是这里使用了两张gpu，所以每张gpu只需要处理7500个batch

### 训练全流程

修改 `vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k.yaml` 将其，训练集大小改为 100,验证集为20,然后batch_size 为4,训练epoch=5，测试epoch=2

```bash
[train: 1, 25 / 25] FPS: 6.0 (26.3)  ,  DataTime: 0.018 (0.013)  ,  ForwardTime: 0.635  ,  TotalTime: 0.666  ,  Loss/total: 26.42655  ,  Loss/giou: 1.39104  ,  Loss/l1: 0.34422  ,  Loss/location: 21.92337  ,  IoU: 0.04523
Epoch Time: 0:00:16.647864
Avg Data Time: 0.01828
Avg GPU Trans Time: 0.01254
Avg Forward Time: 0.63510
/data1/zhh/xiaxulong/TBSI/output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k/checkpoints/train/tbsi_track/vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k
directory doesn't exist. creating...
[train: 2, 25 / 25] FPS: 25.3 (30.6)  ,  DataTime: 0.022 (0.014)  ,  ForwardTime: 0.121  ,  TotalTime: 0.158  ,  Loss/total: 10.96970  ,  Loss/giou: 1.21749  ,  Loss/l1: 0.32555  ,  Loss/location: 6.90697  ,  IoU: 0.06132
Epoch Time: 0:00:03.945361
Avg Data Time: 0.02197
Avg GPU Trans Time: 0.01449
Avg Forward Time: 0.12135
/data1/zhh/xiaxulong/TBSI/output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k/checkpoints/train/tbsi_track/vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k
[train: 3, 25 / 25] FPS: 21.5 (29.8)  ,  DataTime: 0.045 (0.015)  ,  ForwardTime: 0.127  ,  TotalTime: 0.186  ,  Loss/total: 8.51263  ,  Loss/giou: 1.13843  ,  Loss/l1: 0.30017  ,  Loss/location: 4.73492  ,  IoU: 0.08470
Epoch Time: 0:00:04.661120
Avg Data Time: 0.04459
Avg GPU Trans Time: 0.01468
Avg Forward Time: 0.12717
/data1/zhh/xiaxulong/TBSI/output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k/checkpoints/train/tbsi_track/vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k
[train: 4, 25 / 25] FPS: 14.8 (28.5)  ,  DataTime: 0.129 (0.016)  ,  ForwardTime: 0.125  ,  TotalTime: 0.271  ,  Loss/total: 7.36806  ,  Loss/giou: 1.15214  ,  Loss/l1: 0.29215  ,  Loss/location: 3.60304  ,  IoU: 0.07176
Epoch Time: 0:00:06.765074
Avg Data Time: 0.12914
Avg GPU Trans Time: 0.01647
Avg Forward Time: 0.12500
/data1/zhh/xiaxulong/TBSI/output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k/checkpoints/train/tbsi_track/vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k
[train: 5, 25 / 25] FPS: 15.2 (3.0)  ,  DataTime: 0.121 (0.016)  ,  ForwardTime: 0.126  ,  TotalTime: 0.263  ,  Loss/total: 6.55375  ,  Loss/giou: 1.10242  ,  Loss/l1: 0.26880  ,  Loss/location: 3.00491  ,  IoU: 0.09060
Epoch Time: 0:00:06.568892
Avg Data Time: 0.12082
Avg GPU Trans Time: 0.01552
Avg Forward Time: 0.12641
/data1/zhh/xiaxulong/TBSI/output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k/checkpoints/train/tbsi_track/vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k
Finished training!
```


使用paper中的setting进行实验： `vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k.yaml`进行训练，使用7个gpu训练

```bash
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --mode multiple --nproc_per_node 7
```

```bash
[train: 1, 50 / 267] FPS: 5.3 (35.5)  ,  DataTime: 4.020 (0.078)  ,  ForwardTime: 1.919  ,  TotalTime: 6.017  ,  Loss/total: 14.19029  ,  Loss/giou: 1.12714  ,  Loss/l1: 0.26963  ,  Loss/location: 10.58786  ,  IoU: 0.08875
[train: 1, 50 / 267] FPS: 5.3 (35.6)  ,  DataTime: 3.072 (0.056)  ,  ForwardTime: 2.889  ,  TotalTime: 6.017  ,  Loss/total: 14.36190  ,  Loss/giou: 1.13130  ,  Loss/l1: 0.27082  ,  Loss/location: 10.74518  ,  IoU: 0.08779
[train: 1, 50 / 267] FPS: 5.3 (35.6)  ,  DataTime: 0.050 (0.053)  ,  ForwardTime: 5.914  ,  TotalTime: 6.017  ,  Loss/total: 14.16640  ,  Loss/giou: 1.12512  ,  Loss/l1: 0.26890  ,  Loss/location: 10.57166  ,  IoU: 0.08855
[train: 1, 50 / 267] FPS: 5.3 (35.6)  ,  DataTime: 1.677 (0.074)  ,  ForwardTime: 4.266  ,  TotalTime: 6.017  ,  Loss/total: 14.35724  ,  Loss/giou: 1.12270  ,  Loss/l1: 0.26739  ,  Loss/location: 10.77491  ,  IoU: 0.08905
[train: 1, 50 / 267] FPS: 5.3 (35.7)  ,  DataTime: 4.071 (0.059)  ,  ForwardTime: 1.887  ,  TotalTime: 6.017  ,  Loss/total: 14.24263  ,  Loss/giou: 1.12417  ,  Loss/l1: 0.26655  ,  Loss/location: 10.66152  ,  IoU: 0.08819
[train: 1, 50 / 267] FPS: 5.3 (35.7)  ,  DataTime: 0.076 (0.074)  ,  ForwardTime: 5.867  ,  TotalTime: 6.017  ,  Loss/total: 14.23415  ,  Loss/giou: 1.12240  ,  Loss/l1: 0.26527  ,  Loss/location: 10.66300  ,  IoU: 0.08983
[train: 1, 50 / 267] FPS: 5.3 (35.5)  ,  DataTime: 1.888 (0.068)  ,  ForwardTime: 4.061  ,  TotalTime: 6.017  ,  Loss/total: 14.36478  ,  Loss/giou: 1.12769  ,  Loss/l1: 0.26808  ,  Loss/location: 10.76898  ,  IoU: 0.09180
```

60000/32/7=267 没问题，即每个epoch训练267个batch,大概训练一晚上即可。


### gpu 占用分析

当batch_size=4时，gpu 内存 5636 MB 大概为5个多g
当batch_size=8时，gpu 内存 8202 Mb 大概为8个g
当batch_size=16时，gpu 内存 12274 Mb 大概为12个g
当batch_size=24时，gpu 内存 16392 Mb 大概为16个g
当batch_size=32时，gpu 内存 20368 Mb 大概为20个g

## 评估

### 生成测试结果

#### lasher 数据集的结果输出

首先在`local.py`中添加评估数据集的路径，例如使用lasher数据集

```python
    settings.lasher_path = '/data1/zhh/xiaxulong/TBSI/data/lasher'
```

如果在`output`中有训练好的checkpoint，则执行如下代码

```bash
python tracking/test.py tbsi_track vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k --dataset_name lasher_test --threads 6 --num_gpus 1
```

然后就会在`output/test`中生成一些测试集的结果:

```python
(tbsi) ➜  test git:(main) ✗ tree
.
└── tracking_results
    └── tbsi_track
        └── vitb_256_tbsi_4x4_4e4_lasher_15ep_in1k
            ├── 10runone.txt
            ├── 10runone_time.txt
            ├── 11leftboy.txt
            ├── 11leftboy_time.txt
            ├── 11runtwo.txt
            ├── 11runtwo_time.txt
            ├── 2runseven.txt
            ├── 2runseven_time.txt
            ├── 3bike1.txt
            ├── 3bike1_time.txt
            ├── 3men.txt
            ├── 3men_time.txt
            ├── 3pinkleft.txt
            ├── 3pinkleft_time.txt
            ├── AQtruck2north.txt
```

#### rgbt210 数据集的结果输出

```bash
python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name rgbt210 --threads 6 --num_gpus 1
```

#### rgbt234 数据集的结果输出

```bash
python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name rgbt234 --threads 6 --num_gpus 1
```

### 绘制结果

```python
python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name rgbt210
```

可以得到所需要的metric：

Precision NormPrec Success

