# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import Dict, List

import torch


class Fusion(torch.nn.Module):
    def __init__(self, fusion_type: str, features_in: int, features_out: int) -> None:
        super(Fusion, self).__init__()
        self.fusion_type = fusion_type
        self.features_in = features_in
        self.features_out = features_out

        if self.fusion_type == 'concat':
            self._fusion = self._concat_fusion
        else:
            raise ValueError('Unknown fusion type: \'{}\''.format(self.fusion_type))

    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        return self._fusion(x)

    @staticmethod
    def _concat_fusion(x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        fus = []
        for i, (vis, pc) in enumerate(list(zip(x['vis'], x['pc']))):
            fus.append(torch.concat([vis, pc], dim=1))
        return fus
