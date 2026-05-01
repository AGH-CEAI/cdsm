# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from overrides import overrides

from mlgears.core.objects import BaseDatasetLoader, BaseModelLoader
from mlgears.utils.pointcloud import filter_visible_labels


class ModelLoader(BaseModelLoader):
    NAME = 'cdsm_loader'
    VERSION = 'v1.0'
    PARTS = 10

    def __init__(self, processed_path: str,
                 visibility: float = 0.0,
                 visibility_points: int = 0,
                 visibility_scale: float = 1.0,
                 angle_limit: List = (-90, 90),
                 preprocess_pc: bool = False,
                 preprocess_labels: bool = False,
                 tag: Optional[str] = None) -> None:
        super().__init__(processed_path=processed_path, tag=tag)
        self.sub_loaders = dict(zip(['vis', 'pc'], self.LINKED_OBJECTS.values()))
        self.visibility = visibility
        self.visibility_points = visibility_points
        self.visibility_scale = visibility_scale
        self.angle_limit = np.deg2rad(angle_limit)
        self.preprocess_pc = preprocess_pc
        self.preprocess_labels = preprocess_labels
        self.anchors_num = self.sub_loaders['pc'].anchors_num
        self.classes_num = self.sub_loaders['pc'].classes_num
        self.features = self.sub_loaders['pc'].features

        if not self.preprocess_labels:
            self.set_pre_batch_map_function(
                lambda x, y: (x,  self.sub_loaders['pc'].preprocess_labels_fn(y['y_labels']))
            )

    @overrides
    def _define_shapes(self, preprocessed: bool = False) -> [Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        vis_input_shapes, _ = self.sub_loaders['vis']._define_shapes(preprocessed)
        pc_input_shapes, pc_output_shapes = self.sub_loaders['pc']._define_shapes(preprocessed)
        return {**vis_input_shapes, **pc_input_shapes}, pc_output_shapes

    @overrides
    def _define_types(self) -> [Dict[str, str], Dict[str, str]]:
        vis_input_types, _ = self.sub_loaders['vis']._define_types()
        pc_input_types, pc_output_types = self.sub_loaders['pc']._define_types()
        return {**vis_input_types, **pc_input_types}, pc_output_types

    @overrides
    def default_collate_fn(self) -> Callable:
        if not self.preprocess_pc:
            def collate(batch):
                b_size, p_size = np.max([b[0]['x_pc'].shape for b in batch], axis=0)
                padded_batch = [(self.pad(b[0], {'x_pc': np.array([b_size, p_size])}, 1e6), b[1]) for b in batch]
                return torch.utils.data._utils.collate.default_collate(padded_batch)
        else:
            def collate(batch):
                b_size = np.max([b[0]['x_feature_buffer'].shape[0] for b in batch], axis=0)
                pc_loader = self.sub_loaders['pc']
                pad_dict = {
                    'x_img': np.array(self.input_shapes['x_img'], dtype=np.int64),
                    'x_feature_buffer': np.array(
                        [b_size, pc_loader.voxel_max_points, pc_loader.pc_features + 4], dtype=np.int64
                    ),
                    'x_coordinate_buffer': np.array([b_size, 3], dtype=np.int64)
                }
                padded_batch = [(self.pad(b[0], pad_dict, 0), b[1]) for b in batch]
                return torch.utils.data._utils.collate.default_collate(padded_batch)
        return collate

    @overrides
    def load_x(self, sample_id: str, dataset_loader: BaseDatasetLoader) -> Optional[Dict]:
        vis_x = self.sub_loaders['vis'].load_x(sample_id, dataset_loader)
        pc_x = self.sub_loaders['pc'].load_x(sample_id, dataset_loader)
        if vis_x is None or pc_x is None:
            return None
        else:
            return {**vis_x, **pc_x}

    @overrides
    def load_y(self, sample_id: str, dataset_loader: BaseDatasetLoader,
               load_x_output: Optional[Dict] = None) -> Optional[Dict]:
        labels = []
        label_data = dataset_loader.get_labels(sample_id)
        if not self.preprocess_pc:
            pc_data = load_x_output['x_pc']
        else:
            if self.sub_loaders['pc'].pc_type == 'lidar':
                pc_data = dataset_loader.get_lidar(sample_id)
            else:  # pc_type == radar
                pc_data = dataset_loader.get_radar(sample_id)
            pc_data = self.sub_loaders['pc'].filter_pc(pc_data)

        if self.visibility_points > 0:
            visible_labels = filter_visible_labels(label_data.list_bbs_3d, pc_data,
                                                   scaling=self.visibility_scale,
                                                   req_points=self.visibility_points)
        else:
            visible_labels = label_data.list_bbs_3d

        for bbox in visible_labels:
            if bbox.class_id >= 6 or bbox.visibility < self.visibility or \
                    bbox.x < -self.sub_loaders['pc'].grid_center[0] or \
                    bbox.x > self.sub_loaders['pc'].grid_size[0] - self.sub_loaders['pc'].grid_center[0] or \
                    bbox.y < -self.sub_loaders['pc'].grid_center[1] or \
                    bbox.y > self.sub_loaders['pc'].grid_size[1] - self.sub_loaders['pc'].grid_center[1] or \
                    bbox.azimuth_angle < self.angle_limit[0] or bbox.azimuth_angle > self.angle_limit[1]:
                continue
            labels.append([bbox.x, bbox.y, bbox.z,
                           bbox.dx, bbox.dy, bbox.dz,
                           bbox.yaw, bbox.pitch, bbox.roll,
                           bbox.class_id])

        # import matplotlib
        # matplotlib.use('Qt5Agg')
        # import matplotlib.pyplot as plt
        # plt.plot(0, 0, 'x', color='red')
        # plt.scatter(pc_data[:, 0], pc_data[:, 1], s=0.1, c=pc_data[:, 3])
        # for bbox in visible_labels:
        #     if bbox.class_id >= 6 or bbox.visibility < self.visibility or \
        #             bbox.x < -self.sub_loaders['pc'].grid_center[0] or \
        #             bbox.x > self.sub_loaders['pc'].grid_size[0] - self.sub_loaders['pc'].grid_center[0] or \
        #             bbox.y < -self.sub_loaders['pc'].grid_center[1] or \
        #             bbox.y > self.sub_loaders['pc'].grid_size[1] - self.sub_loaders['pc'].grid_center[1] or \
        #             bbox.azimuth_angle < self.angle_limit[0] or bbox.azimuth_angle > self.angle_limit[1]:
        #         continue
        #     c = bbox.get_3d_vertices()
        #     plt.plot(c[:, 0], c[:, 1], color='r')
        # plt.xlim(
        #     -self.sub_loaders['pc'].grid_center[0],
        #     self.sub_loaders['pc'].grid_size[0]-self.sub_loaders['pc'].grid_center[0]
        # )
        # plt.ylim(
        #     -self.sub_loaders['pc'].grid_center[1],
        #     self.sub_loaders['pc'].grid_size[1]-self.sub_loaders['pc'].grid_center[1]
        # )
        # plt.show(block=True)

        if labels:
            if not self.preprocess_labels:
                return {'y_labels': np.array(labels)}
            else:
                processed_labels = self.sub_loaders['pc'].preprocess_labels_fn(np.array(labels))
                processed_labels = {k: v.numpy().astype(self.output_types[k]) for k, v in processed_labels.items()}
                return processed_labels

    def get_anchors(self) -> Dict[str, np.ndarray]:
        return self.sub_loaders['pc'].get_anchors()
