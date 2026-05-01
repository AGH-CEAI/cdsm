# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import List

import torch
from pyquaternion import quaternion


class Rotate3D(torch.nn.Module):
    """
    Implements tensor rotation by given angle alone given axis.
    Angles and axis are in a list format to create more complex transforms.
    Takes into account torch format NCHW.
    """

    def __init__(self, angle: List, axis: List):
        super(Rotate3D, self).__init__()
        self.angle = angle
        self.axis = axis
        if isinstance(angle, list):
            assert len(angle) == len(axis), 'number of angles must equal number of axis!'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rot_x = []
        for b_x in x:
            rot_x.append(self._rotate(b_x))
        return torch.concat([b_x[None] for b_x in rot_x])

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape

        cords = torch.reshape(
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, shape[-3]),
                    torch.arange(0, shape[-2]),
                    torch.arange(0, shape[-1])),
                dim=-1),
            shape=(-1, 3)
        ).type(torch.int64)

        q_transform = quaternion.Quaternion([1, 0, 0, 0])
        for q_angle, q_axis in zip(self.angle, self.axis):
            new_q = quaternion.Quaternion(axis=q_axis, angle=q_angle)
            q_transform = new_q * q_transform

        q_matrix = q_transform.transformation_matrix[:3, :3]
        q_matrix = torch.tensor(q_matrix, device=cords.device, dtype=torch.float64)

        new_cords = torch.round(cords.type(torch.float64) @ q_matrix).type(torch.int64)
        new_cords = new_cords - torch.min(new_cords, dim=0, keepdims=True)[0]

        new_shape = torch.max(new_cords, dim=0)[0] + 1
        new_vals = x[cords[:, 0], cords[:, 1], cords[:, 2]]

        new_tensor = torch.zeros((new_shape[0], new_shape[1], new_shape[2]), device=x.device, dtype=x.dtype)
        new_tensor[new_cords[:, 0], new_cords[:, 1], new_cords[:, 2]] = new_vals
        return new_tensor


if __name__ == '__main__':
    import numpy as np

    org_cam = np.zeros((80, 32, 80))
    org_cam[:, 26, 45] = 1

    rot_cam = Rotate3D([np.pi, -np.pi / 2], [[0, 1, 0], [0, 0, 1]])(torch.tensor(org_cam)[None])[0]
    coords = np.where(rot_cam == 1)

    print(coords[0])
    print(coords[1])
    print(coords[2])
