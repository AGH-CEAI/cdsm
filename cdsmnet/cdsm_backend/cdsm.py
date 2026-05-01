# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import List

import torch
import numpy as np

from cdsm_backend.rotate3D import Rotate3D


class CDSM(torch.nn.Module):
    def __init__(self, num_features_in: int, num_features_out: int,
                 cdsm_feats_x: List[int], cdsm_feats_y: int,
                 cdsm_feats_z_in: List[int], cdsm_feats_z_out: int,
                 num_output_blocks: int, norm: str = 'batch') -> None:
        super(CDSM, self).__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.cdsm_feats_x = cdsm_feats_x
        self.cdsm_feats_y = cdsm_feats_y
        self.cdsm_feats_z_in = cdsm_feats_z_in
        self.cdsm_feats_z_out = cdsm_feats_z_out
        self.num_reduce_blocks = int(np.log2(sum(self.cdsm_feats_x) // self.cdsm_feats_y))
        self.num_output_blocks = num_output_blocks
        self.norm = norm

        assert len(self.cdsm_feats_x) == len(self.cdsm_feats_z_in), \
            'number of cdsm_feats_x must be equal to cdsm_feats_z_in!'

        for i in range(len(self.cdsm_feats_x)):
            setattr(self, 'cdsm_block_{}'.format(i), self._make_cdsm_block(i))

        for i in range(self.num_reduce_blocks):
            setattr(self, 'reduce_block_{}'.format(i), self._make_reduce_block(i))

        for i in range(self.num_output_blocks):
            setattr(self, 'output_block_{}'.format(i), self._make_output_block(i))

    def _get_norm_layer(self, out_channels: int) -> torch.nn.Module:
        if self.norm == 'batch':
            return torch.nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        elif self.norm == 'layer':
            return torch.nn.GroupNorm(1, out_channels, eps=4e-5)
        elif self.norm == 'instance':
            return torch.nn.GroupNorm(out_channels, out_channels, eps=4e-5)
        elif 'group' in self.norm:
            div = int(self.norm.split('_')[1]) if len(self.norm.split('_')) > 1 else 2
            groups = max(1, out_channels // div)
            return torch.nn.GroupNorm(groups, out_channels, eps=4e-5)
        else:
            raise ValueError('normalization should be one of \'batch\', \'instance\', \'layer\' or \'group_X\'')

    def _make_cdsm_block(self, i: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(*[
            torch.nn.Conv2d(self.num_features_in, self.cdsm_feats_x[i], kernel_size=1),
            self._get_norm_layer(self.cdsm_feats_x[i]),
            torch.nn.Mish(),
            Rotate3D([np.pi, -np.pi / 2], [[0, 1, 0], [0, 0, 1]]),
            torch.nn.Conv2d(self.cdsm_feats_z_in[i], self.cdsm_feats_z_out, kernel_size=3, padding=1),
            self._get_norm_layer(self.cdsm_feats_z_out),
            torch.nn.Mish()
        ])

    def _make_reduce_block(self, i: int) -> torch.nn.Sequential:
        feats_in = self.cdsm_feats_z_out * (i+1)
        feats_out = min(feats_in*2, self.num_features_out)
        return torch.nn.Sequential(*[
            torch.nn.Conv2d(feats_in, feats_out, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            self._get_norm_layer(feats_out),
            torch.nn.Mish(),
        ])

    def _make_output_block(self, i: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(*[
            torch.nn.Conv2d(self.num_features_out, self.num_features_out,
                            kernel_size=3, padding=1, stride=2 if i else 1),
            self._get_norm_layer(self.num_features_out),
            torch.nn.Mish(),
            torch.nn.Conv2d(self.num_features_out, self.num_features_out, kernel_size=3, padding=1),
            self._get_norm_layer(self.num_features_out),
            torch.nn.Mish()
        ])

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        levels = []
        for i, l in enumerate(x):
            levels.append(getattr(self, 'cdsm_block_{}'.format(i))(l))
        grid = torch.cat([self._pad_last(l, self.cdsm_feats_y) for l in levels[::-1]], dim=-2)

        for i in range(self.num_reduce_blocks):
            grid = getattr(self, 'reduce_block_{}'.format(i))(grid)

        outs = []
        for i in range(self.num_output_blocks):
            grid = getattr(self, 'output_block_{}'.format(i))(grid)
            outs.append(grid)
        return outs

    @staticmethod
    def _pad_last(x: torch.Tensor, shape: int = 80) -> torch.Tensor:
        pad_shape = list(x.shape)
        if pad_shape[-1] == shape:
            return x
        else:
            diff = shape - pad_shape[-1]
            pre = diff // 2
            post = diff // 2
            if diff % 2:
                pre += 1
            return torch.nn.ConstantPad1d((pre, post), 0)(x)
