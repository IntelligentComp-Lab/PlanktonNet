# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines.
Implementations modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
from functools import partial
import torch
import torch.nn as nn

from timm.layers import trunc_normal_, DropPath, SqueezeExcite
from timm.models import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'twoSS2D': _cfg(url=''),
}


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, ):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    """
    MobileMamba's inter-stage 2x downsampler.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, )
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, )

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class MobileMambaPatchEmbed(nn.Module):
    """
    MobileMamba's stem (patch embed) layer.
    Replicates the 4x stride=2 convolutions.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.patch_embed = torch.nn.Sequential(
            Conv2d_BN(in_channels, out_channels // 4, 3, 2, 1),
            torch.nn.ReLU(),
            Conv2d_BN(out_channels // 4, out_channels // 2, 3, 2, 1),
            torch.nn.ReLU(),
            Conv2d_BN(out_channels // 2, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.patch_embed(x)


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpSwiGLU(nn.Module):
    """
    MLP variant using SwiGLU (Gated Linear Unit with SiLU).
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.SiLU, drop=0., **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features

        hidden_features = int(mlp_ratio * in_features)
        up_features = int(hidden_features * 2 / 3)

        if up_features <= 0:
            up_features = hidden_features // 2

        self.fc1 = nn.Conv2d(in_features, up_features, kernel_size=1)
        self.act = act_layer()
        self.fc_gate = nn.Conv2d(in_features, up_features, kernel_size=1)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(up_features, out_features, kernel_size=1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x_value = self.act(self.fc1(x))
        x_gate = self.fc_gate(x)
        x_gated = x_value * x_gate
        x = self.drop1(x_gated)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SimpleFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.conv1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


# ==================================================================
# [方案三] LeSRA_Adapter: 局部增强 SRA
# ==================================================================
class LeSRA_Adapter(nn.Module):
    """
    Local-Enhanced SRA: 在 Attention 之后并行一个 Depthwise Conv 来恢复局部细节
    """

    def __init__(self, dim, drop=0., num_heads=8, reduction_ratio=4, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.r = reduction_ratio

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1)

        # SRA 的下采样部分
        if self.r > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.r, stride=self.r, groups=dim)
            self.sr_norm = nn.GroupNorm(1, dim)
        else:
            self.sr = nn.Identity()
            self.sr_norm = nn.Identity()

        # [NEW] 局部增强模块: 3x3 DWConv
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        # 1. Q
        q = self.q_proj(x).flatten(2).transpose(1, 2)
        q = q.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. K, V
        if self.r > 1:
            x_down = self.sr_norm(self.sr(x))
        else:
            x_down = self.sr_norm(x)

        kv = self.kv_proj(x_down).flatten(2).transpose(1, 2)
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 3. Attention (Global Context)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_attn = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out_attn = out_attn.permute(0, 2, 1).reshape(B, C, H, W)

        # 4. [NEW] Local Enhancement (Local Context)
        # 将 Attention 的结果加上一个由 DWConv 提取的局部特征
        # 注意：这里我们对 out_attn 再做一次局部卷积，或者直接对 x 做局部卷积并联
        # 论文中常见做法是: Output = Proj( Attention(x) + LocalConv(x) ) 或者在 V 上做。
        # 这里采用最稳健的后处理增强：
        out_local = self.local_conv(out_attn)

        # 融合：全局 + 局部
        out = out_attn + out_local

        return self.out_proj(out)


class MetaFormerBlock(nn.Module):
    """
    HybridFormerBlock:
    op1: DWConv7x7 (residual)
    op2: FFN_1 (residual)
    op3: Token Mixer (residual)
    op4: MLP (residual)
    """

    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=1e-5, res_scale_init_value=None,
                 ffn1_mlp_ratio=2.0
                 ):
        super().__init__()

        self.norm_op1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.drop_path_op1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_op1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale_op1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm_op2 = nn.GroupNorm(num_groups=1, num_channels=dim)
        ffn1_hidden_dim = int(dim * ffn1_mlp_ratio)
        self.ffn1 = SimpleFFN(dim, hidden_features=ffn1_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.drop_path_op2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_op2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale_op2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale_op1(x) + \
            self.layer_scale_op1(
                self.drop_path_op1(
                    self.dwconv(self.norm_op1(x))
                )
            )
        x = self.res_scale_op2(x) + \
            self.layer_scale_op2(
                self.drop_path_op2(
                    self.ffn1(self.norm_op2(x))
                )
            )
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class MetaFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=None,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 drop_path_rate=0.,
                 head_dropout=0.0,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6),
                 head_fn=nn.Linear,
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths]
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage

        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                                  token_mixer=token_mixers[i],
                                  mlp=mlps[i],
                                  drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_values[i],
                                  res_scale_init_value=res_scale_init_values[i],
                                  ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        # [修改]：保留 head 定义以防报错，但 Forward 不会用到它
        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([2, 3]))

    def forward(self, x):
        # [修改]：只返回特征向量 (Batch, 448)，专为 ArcFace 设计
        # 不要通过 self.head
        x = self.forward_features(x)
        return x


@register_model
def PlanktonNet(pretrained=False, **kwargs):
    DOWNSAMPLE_LAYERS_THREE_STAGES = [
        partial(MobileMambaPatchEmbed),
        partial(PatchMerging),
        partial(PatchMerging),
    ]

    model = MetaFormer(
        depths=[2, 3, 2],
        dims=[200, 376, 448],  # 最后输出特征维度 448
        downsample_layers=DOWNSAMPLE_LAYERS_THREE_STAGES,
        token_mixers=[LeSRA_Adapter, LeSRA_Adapter, LeSRA_Adapter],
        mlps=MlpSwiGLU,
        head_fn=MlpHead,
        **kwargs)

    model.default_cfg = default_cfgs['twoSS2D']
    return model


