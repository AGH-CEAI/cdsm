# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import Dict, Optional, List

import torch
import numpy as np
from overrides import overrides

from mlgears.base import BaseDatasetLoader, BaseModelLoader
from mlgears.utils.image.letter_box import resize_image_letterbox


class ModelLoader(BaseModelLoader):
    NAME = 'vinet_loader'
    VERSION = 'v1.0'
    PARTS = 10

    def __init__(self, processed_path: str, input_size: List,
                 grid_size: List, grid_center: List,
                 output_scales: List, classes: List, features: int,
                 anchors_scales: List, anchors_ratios: List, anchors_height: float,
                 visibility: float = 0.0, angle_limit: List = (-90, 90),
                 preprocess_labels: bool = False,
                 non_target_iou: float = 0.5, ignore_target_iou: float = 0.5,
                 tag: Optional[str] = None) -> None:
        super().__init__(processed_path=processed_path, tag=tag)
        self.input_size = np.array(input_size)
        self.grid_size = np.array(grid_size)
        self.grid_center = np.array(grid_center)
        self.output_scales = np.array(output_scales)
        self.classes = dict(zip(classes, range(len(classes))))
        self.classes_num = len(self.classes)
        self.features = features
        self.anchors_scales = np.array(anchors_scales)
        self.anchors_ratios = np.array(anchors_ratios)
        self.anchors_height = anchors_height
        self.anchors_num = len(self.anchors_scales) * len(self.anchors_ratios)
        self.visibility = visibility
        self.angle_limit = np.deg2rad(angle_limit)
        self.preprocess_labels = preprocess_labels
        self.non_target_iou = non_target_iou
        self.ignore_target_iou = ignore_target_iou

        if not self.preprocess_labels:
            self.set_pre_batch_map_function(
                lambda x, y: (x, self.preprocess_labels_fn(y['y_labels']))
            )

    @overrides
    def _define_shapes(self, preprocessed: bool = False) -> [Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        input_shapes = {
            'x_img': np.array([self.input_size[1], self.input_size[0], self.input_size[2]], dtype=np.int64)
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
    def load_x(self, sample_id: str, dataset_loader: BaseDatasetLoader) -> Optional[Dict]:
        cam_image = dataset_loader.get_camera(sample_id)
        _, image_data = resize_image_letterbox(cam_image, self.input_shapes['x_img'][:2][::-1])
        return self.pad({'x_img': image_data}, self.input_shapes)

    @overrides
    def load_y(self, sample_id: str,
               dataset_loader: BaseDatasetLoader,
               load_x_output: Optional[Dict] = None) -> Optional[Dict]:
        labels = []
        label_data = dataset_loader.get_labels(sample_id)

        for bbox in label_data.list_bbs_3d:
            if bbox.class_id >= 6 or bbox.visibility < self.visibility or \
                    bbox.x < -self.grid_center[0] or bbox.x > self.grid_size[0] - self.grid_center[0] or \
                    bbox.y < -self.grid_center[1] or bbox.y > self.grid_size[1] - self.grid_center[1] or \
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
        # for bbox in label_data.list_bbs_3d:
        #     if bbox.class_id >= 6 or bbox.visibility < self.visibility or \
        #             bbox.x < -self.grid_center[0] or bbox.x > self.grid_size[0] - self.grid_center[0] or \
        #             bbox.y < -self.grid_center[1] or bbox.y > self.grid_size[1] - self.grid_center[1] or \
        #             bbox.azimuth_angle < self.angle_limit[0] or bbox.azimuth_angle > self.angle_limit[1]:
        #         continue
        #     c = bbox.get_3d_vertices()
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
