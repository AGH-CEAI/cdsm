# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

from typing import Dict, Optional, List

import torch
import numpy as np
from overrides import overrides

from mlgears.data_classes import BoundingBox
from mlgears.base import BaseDatasetLoader, BaseModelLoader
from mlgears.utils.image.letter_box import resize_image_letterbox


class ModelLoader(BaseModelLoader):
    NAME = 'vinet_loader'
    VERSION = 'v1.0'
    PARTS = 10

    def __init__(self, processed_path: str,
                 input_size: List, output_scales: List, classes: List, features: int,
                 anchors_scales: List, anchors_ratios: List,
                 visibility: float = 0.0, preprocess_labels: bool = False,
                 non_target_iou: float = 0.5, ignore_target_iou: float = 0.5,
                 tag: Optional[str] = None) -> None:
        super().__init__(processed_path=processed_path, tag=tag)
        self.input_size = np.array(input_size)
        self.output_scales = np.array(output_scales)
        self.classes = dict(zip(classes, range(len(classes))))
        self.classes_num = len(self.classes)
        self.features = features
        self.anchors_scales = np.array(anchors_scales)
        self.anchors_ratios = np.array(anchors_ratios)
        self.anchors_num = len(self.anchors_scales) * len(self.anchors_ratios)
        self.visibility = visibility
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
            output_shapes['y_labels'] = np.array([-1, 5])
        else:
            for s in self.output_scales:
                scaled_grid = self.input_size[0:2] // s
                output_shapes['y_bbox_{}x{}'.format(scaled_grid[1], scaled_grid[0])] = np.array(
                    [scaled_grid[1], scaled_grid[0], self.anchors_num, self.features], dtype=np.int64)
                output_shapes['y_cls_{}x{}'.format(scaled_grid[1], scaled_grid[0])] = np.array(
                    [scaled_grid[1], scaled_grid[0], self.anchors_num, self.classes_num], dtype=np.int64)
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
        org_cam_size = dataset_loader.get_camera(sample_id).shape

        for bbox in label_data.list_bbs_cam:
            label = self.parse_label(bbox, np.array(org_cam_size))
            if label:
                labels.append(label)

        # import matplotlib.pyplot as plt
        # import cv2
        # disp_image = np.ascontiguousarray(load_x_output['x_img'] * 255, 'uint8')
        # class_dict = {v: k for k, v in self.classes.items()}
        # for label in labels:
        #     x1, y1, x2, y2 = label[:4]
        #     cv2.rectangle(disp_image, (x1, y1), (x2,y2), 255)
        #     cv2.putText(disp_image, class_dict[label[4]], (x1, y1), 1, 1, 255)
        # plt.imshow(disp_image)
        # plt.show(block=True)

        if labels:
            if not self.preprocess_labels:
                return {'y_labels': np.array(labels)}
            else:
                processed_labels = self.preprocess_labels_fn(np.array(labels))
                processed_labels = {k: v.numpy().astype(self.output_types[k]) for k, v in processed_labels.items()}
                return self.pad(processed_labels, self.output_shapes)

    def parse_label(self, bbox: BoundingBox, image_size: np.ndarray) -> Optional[tuple]:
        if bbox.class_id >= 6 or bbox.visibility < self.visibility:
            return

        _input_shape = np.array(self.input_shapes['x_img'][0:2], dtype=np.float32)
        _image_shape = np.array(image_size[:2], dtype=np.float32)
        _new_shape = np.round(_image_shape * np.min(_input_shape / _image_shape))
        offset = (_input_shape - _new_shape) / 2.
        scale = _new_shape / _image_shape

        y, x = (np.array([bbox.y, bbox.x]) * scale + offset).astype(int)
        if y <= 0 or y >= _new_shape[0] or x <= 0 or x >= _new_shape[1]:
            return
        h, w = (np.array([bbox.dy, bbox.dx]) * scale).astype(int)

        x1, y1 = x-w//2, y-h//2
        x2, y2 = x+w//2, y+h//2
        return x1, y1, x2, y2, bbox.class_id

    def get_anchors(self) -> Dict[str, np.ndarray]:
        generated_anchors = {}
        for s in self.output_scales:
            num_anchors = len(self.anchors_ratios) * len(self.anchors_scales)
            anchors = torch.zeros((num_anchors, 4))

            # scale base_size
            anchors_scales = torch.tensor(self.anchors_scales).float()
            anchors_ratios = torch.tensor(self.anchors_ratios).float()
            anchors[:, 2:] = s * torch.tile(anchors_scales, (2, len(self.anchors_ratios))).T

            # compute areas of anchors
            areas = anchors[:, 2] * anchors[:, 3]

            # correct for ratios
            anchors[:, 2] = torch.sqrt(areas / anchors_ratios.repeat_interleave(len(self.anchors_scales)))
            anchors[:, 3] = anchors[:, 2] * anchors_ratios.repeat_interleave(len(self.anchors_scales))

            # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
            anchors[:, 0::2] -= torch.tile(anchors[:, 2] * 0.5, (2, 1)).T
            anchors[:, 1::2] -= torch.tile(anchors[:, 3] * 0.5, (2, 1)).T

            grid_size = self.input_size // s

            shift_x = (torch.arange(0, grid_size[0]) + 0.5) * s
            shift_y = (torch.arange(0, grid_size[1]) + 0.5) * s
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shifts = torch.vstack((
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel(),
            )).T

            A = anchors.shape[0]
            K = shifts.shape[0]
            shifted_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).permute((1, 0, 2)))
            shifted_anchors = shifted_anchors.reshape((grid_size[1], grid_size[0], A, 4))

            generated_anchors['{}x{}'.format(grid_size[1], grid_size[0])] = shifted_anchors

        return generated_anchors

    def calculate_iou(self, anchors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        labels_area = (labels[:, 2] - labels[:, 0]) * (labels[:, 3] - labels[:, 1])
        anchors_area = (anchors[..., 2] - anchors[..., 0]) * (anchors[..., 3] - anchors[..., 1])

        iw = torch.minimum(anchors[..., 2][..., None], labels[:, 2]) - \
             torch.maximum(anchors[..., 0][..., None], labels[:, 0])
        ih = torch.minimum(anchors[..., 3][..., None], labels[:, 3]) - \
             torch.maximum(anchors[..., 1][..., None], labels[:, 1])

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
        labels: array, shape=(m, 5) (..., (x1, y1, x2, y2, class_id))

        Returns
        -------
        y: dict of arrays, shaped like model output grids at scales
        """

        assert (labels[..., 4] < len(self.classes)).all(), 'class id must be less than num_classes'

        labels = torch.tensor(labels).float()
        all_anchors = self.get_anchors()
        processed_labels = {}
        for scale, anchors in all_anchors.items():
            y_box = torch.zeros_like(anchors)
            y_cls = torch.ones((*anchors.shape[:-1], self.classes_num)) * -1

            iou = self.calculate_iou(anchors, labels[:, :4])
            iou_max, iou_idx = iou.max(axis=-1)

            no_targets_mask = iou_max < self.non_target_iou
            targets_mask = iou_max > self.ignore_target_iou

            y_cls[torch.where(no_targets_mask)] = 0

            for l in torch.unique(iou_idx[torch.where(targets_mask)]):
                onehot = torch.zeros(self.classes_num)
                onehot[int(labels[l, 4])] = 1.
                coords = torch.where(torch.logical_and(iou_idx == l, targets_mask))
                y_cls[coords] = onehot
                y_box[coords] = labels[l, :4]

            processed_labels['y_bbox_{}'.format(scale)] = y_box.float()
            processed_labels['y_cls_{}'.format(scale)] = y_cls.float()

        return processed_labels
