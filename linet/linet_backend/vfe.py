# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import Dict, Tuple

import torch


class VFE(torch.nn.Module):
    def __init__(self, device: str, grid_size: list, grid_center: list, pc_feats: int,
                 voxel_size: list, voxel_features: int, voxel_max_points: int,
                 convert_pc2voxel: bool) -> None:
        super(VFE, self).__init__()
        self.grid_size = torch.tensor(grid_size, device=device)
        self.grid_center = torch.tensor(grid_center, device=device)
        self.voxel_size = torch.tensor(voxel_size, device=device)
        self.voxel_max_points = torch.tensor(voxel_max_points, device=device)

        self.voxel_grid_size = (self.grid_size / self.voxel_size).type(torch.int64).to(device=device)

        self.convert_pc2voxel = convert_pc2voxel
        self.vfe_layer1 = VFELayer(pc_feats, voxel_features // 2)
        self.vfe_layer2 = VFELayer(voxel_features // 2, voxel_features)
        self.vfe_scatter = VFEScatter(self.voxel_grid_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.convert_pc2voxel:
            feature_buffer, coordinate_buffer = self.pc2voxel(data['x_pc'])
        else:
            feature_buffer = data['x_feature_buffer']
            coordinate_buffer = data['x_coordinate_buffer']
        features = self.vfe_layer1(feature_buffer)
        features = self.vfe_layer2(features)
        grids = self.vfe_scatter(features, coordinate_buffer)
        return grids

    def pc2voxel(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        voxel_index = torch.floor((point_cloud[..., :3] + self.grid_center) / self.voxel_size).type(torch.int64)
        bound_x = torch.logical_and(torch.ge(voxel_index[..., 0], 0),
                                    torch.less(voxel_index[..., 0], self.voxel_grid_size[0]))
        bound_y = torch.logical_and(torch.ge(voxel_index[..., 1], 0),
                                    torch.less(voxel_index[..., 1], self.voxel_grid_size[1]))
        bound_z = torch.logical_and(torch.ge(voxel_index[..., 2], 0),
                                    torch.less(voxel_index[..., 2], self.voxel_grid_size[2]))
        bound_box = torch.logical_and(torch.logical_and(bound_x, bound_y), bound_z)

        feature_buffer = []
        coordinate_buffer = []
        batch_size = point_cloud.shape[0]
        for i in range(batch_size):
            b_point_cloud = point_cloud[i, bound_box[i]]
            b_voxel_index = voxel_index[i, bound_box[i]]

            # [K, 3] coordinate buffer as described in the paper
            # [K, 1] number_buffer store number of points in each voxel grid
            b_coordinate_buffer, b_number_buffer = torch.unique(b_voxel_index, dim=0, return_counts=True)
            K = len(b_number_buffer)
            # [K, T, point_size+3] feature buffer as described in the paper + 1 (count)
            b_feature_buffer = torch.zeros(
                (K, int(self.voxel_max_points), b_point_cloud.shape[-1] + 4),
                device=b_point_cloud.device
            )

            self.fill_feature_buffer(K, b_point_cloud, b_voxel_index,
                b_number_buffer, b_coordinate_buffer, b_feature_buffer, int(self.voxel_max_points))

            feature_buffer.append(b_feature_buffer)
            coordinate_buffer.append(b_coordinate_buffer)
        return (
            torch.nn.utils.rnn.pad_sequence(feature_buffer, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(coordinate_buffer, batch_first=True)
        )

    @staticmethod
    def fill_feature_buffer(buffers_len: int, pc: torch.Tensor, voxel_index: torch.Tensor,
                            number_buffer: torch.Tensor, coordinate_buffer: torch.Tensor,
                            feature_buffer: torch.Tensor, max_points: int) -> None:
        voxel_count = torch.clamp(number_buffer, 1, max_points)
        for k in range(buffers_len):
            voxel_points = pc[torch.all(torch.eq(voxel_index, coordinate_buffer[k]), dim=1)]
            # fill feature buffer with #voxel_count point data, point distance to the center and voxel_count value
            feature_buffer[k, :voxel_count[k]] = torch.concat([
                voxel_points[:voxel_count[k]],
                voxel_points[:voxel_count[k], :3] - voxel_points[:voxel_count[k], :3].mean(dim=0, keepdim=True),
                torch.tile(voxel_count[k], [int(voxel_count[k]), 1]),
            ], dim=-1).float()


class VFELayer(torch.nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super(VFELayer, self).__init__()
        self.units = int(cout / 2)
        self.fc = torch.nn.Linear(cin, self.units)
        self.bn = torch.nn.BatchNorm1d(self.units)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # point-wise feauture
        b, k, t, _ = x.shape
        pointwise = self.fc(x.view(b * k * t, -1))
        pointwise = self.act(self.bn(pointwise).view(b, k, t, -1))
        # [B, K, 1, units]
        aggregated = torch.max(pointwise, 2, keepdim=True)[0]
        # [B, K, T, units]
        repeated = aggregated.repeat(1, 1, pointwise.shape[-2], 1)
        # [B, K, T, 2 * units]
        pointwise_concat = torch.cat((pointwise, repeated), dim=3)
        mask = torch.not_equal(torch.max(x, dim=-1, keepdim=True)[0], 0)
        return pointwise_concat * mask.float()


class VFEScatter(torch.nn.Module):
    def __init__(self, voxel_grid_size: torch.Tensor):
        super(VFEScatter, self).__init__()
        self.voxel_grid_size = voxel_grid_size.tolist()

    def forward(self, features: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        # [B, K, FEATURES]
        voxelwise = torch.max(features, dim=2)[0]
        # [B, GRID_X, GRID_Y, GRID_Z, FEATURES]
        grid = torch.zeros((voxelwise.shape[0], self.voxel_grid_size[0], self.voxel_grid_size[1],
            self.voxel_grid_size[2], voxelwise.shape[-1]), dtype=voxelwise.dtype, device=voxelwise.device)

        # write voxel features to correct grid cells
        b, k, c = coordinates.shape
        coordinates = coordinates.view(b*k, c)
        batch_idx = [torch.repeat_interleave(torch.arange(0, b), k).type(torch.int64).to(coordinates.device)]
        slices = [coordinates[:, i] for i in range(coordinates.shape[-1])]
        grid[batch_idx + slices] = voxelwise.view(b*k, -1)
        # [B, GRID_X, GRID_Y, GRID_Z, FEATURES]
        return grid.view(b, self.voxel_grid_size[0], self.voxel_grid_size[1], -1)
