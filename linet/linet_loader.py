# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import Callable, Dict, Optional, List

import torch
import numpy as np
from overrides import overrides

from mlgears.base import BaseDatasetLoader, BaseModelLoader
from mlgears.utils.pointcloud import filter_visible_labels

from linet_backend.vfe import VFE


class ModelLoader(BaseModelLoader):
    NAME = 'linet_loader'
    VERSION = 'v1.0'
    PARTS = 10

    def __init__(self, processed_path: str, pc_type: str, pc_features: int,
                 grid_size: List, grid_center: List,
                 voxel_size: List, voxel_features: int, voxel_max_points: int,
                 output_scales: List, classes: List, features: int,
                 anchors_scales: List, anchors_ratios: List, anchors_height: float,
                 visibility_points: int = 1, visibility_scale: float = 1.0,
                 preprocess_pc: bool = False, preprocess_labels: bool = False,
                 non_target_iou: float = 0.5, ignore_target_iou: float = 0.5,
                 tag: Optional[str] = None) -> None:
        super().__init__(processed_path=processed_path, tag=tag)
        assert pc_type in ['lidar', 'radar']
        self.pc_type = pc_type
        self.pc_features = pc_features
        self.grid_size = np.array(grid_size)
        self.grid_center = np.array(grid_center)
        self.voxel_size = np.array(voxel_size)
        self.voxel_features = voxel_features
        self.voxel_max_points = voxel_max_points
        self.output_scales = np.array(output_scales)
        self.classes = dict(zip(classes, range(len(classes))))
        self.classes_num = len(self.classes)
        self.features = features
        self.anchors_scales = np.array(anchors_scales)
        self.anchors_ratios = np.array(anchors_ratios)
        self.anchors_height = anchors_height
        self.anchors_num = len(self.anchors_scales) * len(self.anchors_ratios)
        self.visibility_points = visibility_points
        self.visibility_scale = visibility_scale
        self.preprocess_pc = preprocess_pc
        self.preprocess_labels = preprocess_labels
        self.non_target_iou = non_target_iou
        self.ignore_target_iou = ignore_target_iou

        if self.preprocess_pc:
            self.vfe = VFE(
                'cpu', list(self.grid_size), list(self.grid_center), self.pc_features + 4,
                list(self.voxel_size), self.voxel_features, self.voxel_max_points, True
            )

        if not self.preprocess_labels:
            self.set_pre_batch_map_function(
                lambda x, y: (x, self.preprocess_labels_fn(y['y_labels']))
            )

    @overrides
    def _define_shapes(self, preprocessed: bool = False) -> [Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if not self.preprocess_pc:
            if self.pc_type == 'lidar':
                input_shapes = {'x_pc': np.array([-1, self.pc_features], dtype=np.int64)}
            else:  # pc_type == radar
                input_shapes = {'x_pc': np.array([-1, self.pc_features], dtype=np.int64)}
        else:
            input_shapes = {
                'x_feature_buffer': np.array([-1, self.voxel_max_points, self.pc_features + 4], dtype=np.int64),
                'x_coordinate_buffer': np.array([-1, 3], dtype=np.int64)
            }
        output_shapes = {}
        if not preprocessed and not self.preprocess_labels:
            output_shapes['y_labels'] = np.array([-1, 10])
        else:
            for s in self.output_scales:
                scaled_grid = (self.grid_size[0:2] // s).astype('int64')
                output_shapes['y_bbox_{}x{}'.format(scaled_grid[0], scaled_grid[1])] = np.array(
                    [scaled_grid[0], scaled_grid[1], self.anchors_num, self.features], dtype=np.int64)
                output_shapes['y_cls_{}x{}'.format(scaled_grid[0], scaled_grid[1])] = np.array(
                    [scaled_grid[0], scaled_grid[1], self.anchors_num, self.classes_num], dtype=np.int64)
        return input_shapes, output_shapes

    @overrides
    def _define_types(self) -> [Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if not self.preprocess_pc:
            input_types = {'x_pc': 'float32'}
        else:
            input_types = {'x_feature_buffer': 'float32',
                           'x_coordinate_buffer': 'int64'}
        output_types = {}
        if not self.preprocess_labels:
            output_types['y_labels'] = 'float32'
        else:
            for s in self.output_scales:
                scaled_grid = (self.grid_size[0:2] // s).astype('int64')
                output_types['y_bbox_{}x{}'.format(scaled_grid[0], scaled_grid[1])] = 'float32'
                output_types['y_cls_{}x{}'.format(scaled_grid[0], scaled_grid[1])] = 'float32'
        return input_types, output_types

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
                pad_dict = {
                    'x_feature_buffer': np.array([b_size, self.voxel_max_points, self.pc_features + 4], dtype=np.int64),
                    'x_coordinate_buffer': np.array([b_size, 3], dtype=np.int64)
                }
                padded_batch = [(self.pad(b[0], pad_dict, 0), b[1]) for b in batch]
                return torch.utils.data._utils.collate.default_collate(padded_batch)
        return collate

    @overrides
    def load_x(self, sample_id: str, dataset_loader: BaseDatasetLoader) -> Optional[Dict]:
        if self.pc_type == 'lidar':
            pc_data = dataset_loader.get_lidar(sample_id)
        else:  # pc_type == radar
            pc_data = dataset_loader.get_radar(sample_id)
        filtered_pc_data = self.filter_pc(pc_data)
        if len(filtered_pc_data):
            if not self.preprocess_pc:
                return {'x_pc': filtered_pc_data}
            else:
                feature_buffer, x_coordinate_buffer = self.vfe.pc2voxel(torch.tensor(filtered_pc_data[None, ...]))
                return {'x_feature_buffer': feature_buffer[0].numpy(),
                        'x_coordinate_buffer': x_coordinate_buffer[0].numpy()}

    def filter_pc(self, pc: np.ndarray) -> np.ndarray:
        x_min = pc[:, 0] + self.grid_center[0] > 0
        x_max = pc[:, 0] + self.grid_center[0] < self.grid_size[0]
        y_min = pc[:, 1] + self.grid_center[1] > 0
        y_max = pc[:, 1] + self.grid_center[1] < self.grid_size[1]
        z_min = pc[:, 2] + self.grid_center[2] > 0
        z_max = pc[:, 2] + self.grid_center[2] < self.grid_size[2]
        return pc[x_min * x_max * y_min * y_max * z_min * z_max]

    @overrides
    def load_y(self, sample_id: str,
               dataset_loader: BaseDatasetLoader,
               load_x_output: Optional[Dict] = None) -> Optional[Dict]:
        labels = []
        label_data = dataset_loader.get_labels(sample_id)
        if not self.preprocess_pc:
            pc_data = load_x_output['x_pc']
        else:
            if self.pc_type == 'lidar':
                pc_data = dataset_loader.get_lidar(sample_id)
            else:  # pc_type == radar
                pc_data = dataset_loader.get_radar(sample_id)
            pc_data = self.filter_pc(pc_data)

        visible_labels = filter_visible_labels(label_data.list_bbs_3d, pc_data,
                                               scaling=self.visibility_scale,
                                               req_points=self.visibility_points)

        for bbox in visible_labels:
            if bbox.class_id >= 6:
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
        # for l in visible_labels:
        #     c = l.get_3d_vertices()
        #     plt.plot(c[:, 0], c[:, 1], color='r')
        # plt.xlim(-self.grid_center[0], self.grid_size[0]-self.grid_center[0])
        # plt.ylim(-self.grid_center[1], self.grid_size[1]-self.grid_center[1])
        # plt.show(block=True)

        if labels:
            if not self.preprocess_labels:
                return {'y_labels': np.array(labels)}
            else:
                processed_labels = self.preprocess_labels_fn(np.array(labels))
                processed_labels = {k: v.numpy().astype(self.output_types[k]) for k, v in processed_labels.items()}
                return processed_labels

    def get_anchors(self) -> Dict[str, np.ndarray]:
        generated_anchors = {}
        for s in self.output_scales:
            anchors = torch.zeros((self.anchors_num, self.features))

            # scale base_size
            anchors_scales = torch.tensor(self.anchors_scales).float()
            anchors_ratios = torch.tensor(self.anchors_ratios).float()
            anchors[:, 3:5] = s * torch.tile(anchors_scales, (2, len(self.anchors_ratios))).T

            # compute areas of anchors
            areas = anchors[:, 3] * anchors[:, 4]

            # correct for ratios
            anchors[:, 3] = torch.sqrt(areas / anchors_ratios.repeat_interleave(len(self.anchors_scales)))
            anchors[:, 4] = anchors[:, 3] * anchors_ratios.repeat_interleave(len(self.anchors_scales))

            grid_size = (self.grid_size // s).astype('int64')
            K = grid_size[0] * grid_size[1]

            shift_x = (torch.arange(0, grid_size[0]) + 0.5) * s - self.grid_center[0]
            shift_y = (torch.arange(0, grid_size[1]) + 0.5) * s - self.grid_center[1]
            shift_x, shift_y = torch.meshgrid(shift_x, shift_y)

            shifts = torch.zeros((K, self.features))
            shifts[:, 0] = shift_x.ravel()
            shifts[:, 1] = shift_y.ravel()
            shifts[:, 2] = self.anchors_height / 2
            shifts[:, 5] = self.anchors_height

            A = anchors.shape[0]
            shifted_anchors = anchors.reshape((1, A, self.features)) + shifts.reshape((K, 1, self.features))
            shifted_anchors = shifted_anchors.reshape((grid_size[0], grid_size[1], A, self.features))

            generated_anchors['{}x{}'.format(grid_size[0], grid_size[1])] = shifted_anchors
        return generated_anchors

    def calculate_iou(self, anchors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        labels_area = labels[:, 3] * labels[:, 4]
        anchors_area = anchors[..., 3] * anchors[..., 4]

        labels_x1 = labels[:, 0] - labels[:, 3] / 2
        labels_x2 = labels[:, 0] + labels[:, 3] / 2
        labels_y1 = labels[:, 1] - labels[:, 4] / 2
        labels_y2 = labels[:, 1] + labels[:, 4] / 2

        anchors_x1 = anchors[..., 0] - anchors[..., 3] / 2
        anchors_x2 = anchors[..., 0] + anchors[..., 3] / 2
        anchors_y1 = anchors[..., 1] - anchors[..., 4] / 2
        anchors_y2 = anchors[..., 1] + anchors[..., 4] / 2

        iw = torch.minimum(anchors_x2[..., None], labels_x2) - \
             torch.maximum(anchors_x1[..., None], labels_x1)
        ih = torch.minimum(anchors_y2[..., None], labels_y2) - \
             torch.maximum(anchors_y1[..., None], labels_y1)

        iw = torch.clip(iw, min=0, max=np.maximum(0, iw.max()))
        ih = torch.clip(ih, min=0, max=np.maximum(0, ih.max()))

        intersection = iw * ih

        union = anchors_area[..., None] + labels_area - intersection
        union = torch.clip(union, min=1, max=np.maximum(1, union.max()))

        return intersection / union

    def preprocess_labels_fn(self, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess labels list to training input grid format

        Parameters
        ----------
        labels: array, shape=(m, 10) (..., (x, y, z, dx, dy, dz, yaw, pitch, roll, cls))

        Returns
        -------
        y: dict of arrays, shaped like model output grids at scales
        """

        assert (labels[..., -1] < len(self.classes)).all(), 'class id must be less than num_classes'

        labels = torch.tensor(labels).float()
        all_anchors = self.get_anchors()
        processed_labels = {}
        for scale, anchors in all_anchors.items():
            y_box = torch.zeros_like(anchors)
            y_cls = torch.ones((*anchors.shape[:-1], self.classes_num)) * -1

            iou = self.calculate_iou(anchors, labels[:, :6])
            iou_max, iou_idx = iou.max(axis=-1)

            no_targets_mask = iou_max < self.non_target_iou
            targets_mask = iou_max > self.ignore_target_iou

            y_cls[torch.where(no_targets_mask)] = 0

            for l in torch.unique(iou_idx[torch.where(targets_mask)]):
                onehot = torch.zeros(self.classes_num)
                onehot[int(labels[l, -1])] = 1.
                coords = torch.where(torch.logical_and(iou_idx == l, targets_mask))
                y_cls[coords] = onehot
                y_box[coords] = labels[l, :-1]

            processed_labels['y_bbox_{}'.format(scale)] = y_box.float()
            processed_labels['y_cls_{}'.format(scale)] = y_cls.float()

        return processed_labels
