# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

import sys

import numpy as np

sys.setrecursionlimit(5000)

import torch
import torchvision

from typing import Callable, Dict, Any, Optional, Union, List, TypeVar
from overrides import overrides

from mlgears.data_classes import SampleLabels, BoundingBox
from mlgears.base import BaseDatasetLoader, BaseModel
from vinet_loader import ModelLoader

from vinet_backend.bifpn import BiFPN
from vinet_backend.cdsm import CDSM
from vinet_backend.classification_head import ClassificationHead
from vinet_backend.regression_head import RegressionHead
from vinet_backend.utils import BBoxTransform

T = TypeVar('T')


class Model(BaseModel):
    NAME = 'vinet_model'
    VERSION = 'v1.0'
    BACKBONES = {
        'efficientnet_v2_s': {
            'model': torchvision.models.efficientnet_v2_s,
            'outputs': [3, 5, 7],
            'weights': torchvision.models.EfficientNet_V2_S_Weights.DEFAULT,
        },
        'efficientnet_v2_m': {
            'model': torchvision.models.efficientnet_v2_m,
            'outputs': [3, 5, 8],
            'weights': torchvision.models.EfficientNet_V2_M_Weights.DEFAULT,
        },
        'efficientnet_v2_l': {
            'model': torchvision.models.efficientnet_v2_l,
            'outputs': [3, 5, 8],
            'weights': torchvision.models.EfficientNet_V2_L_Weights.DEFAULT,
        }
    }

    def __init__(self, backbone: str,
                 pretrained: bool = False,
                 freeze_backbone: bool = False,
                 bifpn_features: int = 128,
                 bifpn_repeats: int = 3,
                 bifpn_output: bool = False,
                 bifpn_norm: str = 'batch',
                 cdsm_features_x: List = [160, 96, 40, 16, 8],
                 cdsm_features_z: int = 64,
                 cdsm_features_out: int = 256,
                 cdsm_norm: str = 'batch',
                 prediction_features: int = 64,
                 prediction_depth: int = 3,
                 score_threshold: float = 0.05,
                 iou_threshold: float = 0.5,
                 predict_from_labels: bool = False,
                 use_jit: bool = False) -> None:
        super(Model, self).__init__(use_jit)
        self._model_loader = None

        self.backbone = backbone
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        self.bifpn_features = bifpn_features
        self.bifpn_repeats = bifpn_repeats
        self.bifpn_output = bifpn_output
        self.bifpn_norm = bifpn_norm

        self.cdsm_features_x = cdsm_features_x
        self.cdsm_features_z = cdsm_features_z
        self.cdsm_features_out = cdsm_features_out
        self.cdsm_norm = cdsm_norm

        self.prediction_features = prediction_features
        self.prediction_depth = prediction_depth

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.predict_from_labels = predict_from_labels

    def at_high_threshold(self, fn: Callable[..., T]) -> Callable[..., T]:
        def wrapped_fn(*args, **kwargs) -> T:
            threshold = self.score_threshold
            self.score_threshold = 0.99
            results = fn(*args, **kwargs)
            self.score_threshold = threshold
            return results
        return wrapped_fn

    @overrides
    def get_memory_usage(self) -> Optional[float]:
        return self.at_high_threshold(super().get_memory_usage)()

    @overrides
    def get_flops(self, verbose: bool = False) -> Optional[int]:
        return self.at_high_threshold(super().get_flops)(verbose)

    @overrides
    def plot(self, file_path: str) -> None:
        return self.at_high_threshold(super().plot)(file_path)

    @overrides
    def network_structure(self, model_loader: ModelLoader) \
            -> Union[torch.nn.Module, torch.jit.ScriptModule]:
        self._model_loader = model_loader

        assert self.backbone in self.BACKBONES.keys(), \
            'Backbone \'{}\' not supported. Try one of: {}'.format(
                self.backbone, list(self.BACKBONES.keys()))

        backbone_dict = self.BACKBONES[self.backbone]
        if self.pretrained:
            backbone_model = backbone_dict['model'](backbone_dict['weights'])
        else:
            backbone_model = backbone_dict['model']()
        if self.freeze_backbone:
            for p in backbone_model.parameters():
                p.requires_grad = False

        backbone_output = backbone_dict['outputs']
        backbone_output_size = []
        for i in backbone_output:
            m = backbone_model.features[i]
            if hasattr(m, 'out_channels'):
                backbone_output_size.append(m.out_channels)
            else:
                backbone_output_size.append(list(m._modules.values())[-1].out_channels)

        class Net(self.get_model_class()):
            def __init__(net):
                super(Net, net).__init__()
                net.input_shapes, net.output_shapes = model_loader._define_shapes(preprocessed=True)

                net.anchors = model_loader.get_anchors()

                net.add_module('backbone', backbone_model.features)
                net.bifpn = BiFPN(
                    input_size=backbone_output_size,
                    feature_size=self.bifpn_features,
                    num_layers=self.bifpn_repeats,
                    norm=self.bifpn_norm
                )
                cdsm_feats_y = int(model_loader.grid_size[1] * model_loader.output_scales[0])
                input_ratio = model_loader.input_size[1] / model_loader.input_size[0]
                net.cdsm = CDSM(
                    num_features_in=self.bifpn_features,
                    num_features_out=self.cdsm_features_out,
                    cdsm_feats_x=self.cdsm_features_x,
                    cdsm_feats_y=cdsm_feats_y,
                    cdsm_feats_z_in=[int(cdsm_feats_y * input_ratio / 2**i) for i in range(5)],
                    cdsm_feats_z_out=self.cdsm_features_z,
                    num_output_blocks=len(model_loader.output_scales),
                    norm=self.cdsm_norm
                )
                net.regression_head = RegressionHead(
                    self.cdsm_features_out,
                    self.prediction_features,
                    self.prediction_depth,
                    model_loader.anchors_num,
                    model_loader.features
                )
                net.classification_head = ClassificationHead(
                    self.cdsm_features_out,
                    self.prediction_features,
                    self.prediction_depth,
                    model_loader.anchors_num,
                    model_loader.classes_num
                )

                net.bbox_transform = BBoxTransform()

                net.init_weights()

            def init_weights(net):
                modules = [net.bifpn, net.classification_head, net.regression_head]
                if not self.pretrained:
                    modules.insert(0, net.backbone)
                for module in modules:
                    for m in module.modules():
                        if isinstance(m, torch.nn.Conv2d):
                            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                            m.weight.data.normal_(0, np.sqrt(2. / n))
                            if m.bias is not None:
                                m.bias.data.fill_(0)
                        elif isinstance(m, torch.nn.BatchNorm2d):
                            m.weight.data.fill_(1)
                            m.bias.data.fill_(0)
                            m.eval()

                net.classification_head.output.weight.data.normal_(0, 4e-4)
                net.regression_head.output.weight.data.normal_(0, 4e-4)

            def forward(net, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                image = input_dict['x_img']
                x = image.permute(0, 3, 1, 2)
                backbone_feature_maps = []
                for i, submodule in enumerate(net.backbone):
                    x = submodule(x)
                    if i in backbone_output:
                        backbone_feature_maps.append(x)

                features = net.bifpn(backbone_feature_maps)
                cdsm_features = net.cdsm(features)

                reg = [net.regression_head(feature) for feature in cdsm_features]
                cls = [net.classification_head(feature) for feature in cdsm_features]

                if net.training:
                    output = {}
                    for i, k in enumerate(net.anchors.keys()):
                        output['y_bbox_{}'.format(k)] = reg[i]
                        output['y_cls_{}'.format(k)] = cls[i]
                    if self.bifpn_output:
                        output['bifpn'] = features
                    return output
                else:
                    return net.postprocess_preds(reg, cls)

            def postprocess_preds(net, reg: List[torch.Tensor], cls: List[torch.Tensor]) -> Dict:
                bboxes = net.bbox_transform(net.anchors, reg)
                cls = torch.concat([c.reshape(c.shape[0], -1, c.shape[-1]) for c in cls], axis=1)
                scores = torch.max(cls, dim=2, keepdim=True)[0]
                scores_over_thresh = (scores > self.score_threshold)[..., 0]
                results = {'scores': [], 'classes': [], 'boxes': []}

                for b in range(scores.shape[0]):
                    if scores_over_thresh[b].sum() == 0:
                        results['scores'].append(scores[b, scores_over_thresh[b], 0])
                        results['classes'].append(cls[b, scores_over_thresh[b], :])
                        results['boxes'].append(bboxes[b, scores_over_thresh[b], :])
                        continue

                    b_cls = cls[b, scores_over_thresh[b], :]
                    b_bboxes = bboxes[b, scores_over_thresh[b], :]
                    b_scores = scores[b, scores_over_thresh[b], 0]

                    box_corners = (
                        torch.tile(b_bboxes[..., 3:5], (1, 4)) *
                        torch.tensor([.5, .5, .5, -.5, -.5, .5, -.5, -.5],
                                     device=b_bboxes.device, dtype=b_bboxes.dtype)
                    ).reshape(-1, 4, 2)

                    rot_matrix = torch.vstack(
                        [torch.cos(b_bboxes[..., 6]),
                         torch.sin(b_bboxes[..., 6]),
                         -torch.sin(b_bboxes[..., 6]),
                         torch.cos(b_bboxes[..., 6])]
                    ).T.reshape(-1, 2, 2)

                    rot_corners = box_corners @ rot_matrix
                    tr_rot_corners = rot_corners + b_bboxes[..., 0:2][:, None, :]
                    final_corners = torch.cat([
                        tr_rot_corners.min(dim=1)[0],
                        tr_rot_corners.max(dim=1)[0]
                    ], dim=-1)

                    final_corners = (final_corners - final_corners.min() + 1) * 10

                    nms_idx = torchvision.ops.nms(final_corners, b_scores, self.iou_threshold)
                    nms_scores, nms_class = b_cls[nms_idx, :].max(dim=1)

                    results['scores'].append(nms_scores)
                    results['classes'].append(nms_class)
                    results['boxes'].append(b_bboxes[nms_idx, :])

                return {k: (torch.concat([v[None, ...] for v in b_v])) for k, b_v in results.items()}

        return Net()

    @overrides
    def predict_labels(self, sample_id: str, model_loader_load_x: Any,
                       dataset_loader: Optional[BaseDatasetLoader]) -> SampleLabels:
        for k, v in model_loader_load_x.items():
            model_loader_load_x[k] = v[None, ...]

        if self.predict_from_labels:
            predictions = self.get_preds_from_label(sample_id, model_loader_load_x, dataset_loader)
        else:
            predictions = self.predict(model_loader_load_x)

        if not predictions:
            return SampleLabels(
                sample_id=sample_id,
                datasetloader=dataset_loader,
                bounding_boxes_3d=[],
                is_ground_truth=False
        )

        scores = predictions['scores'][0].cpu().detach().numpy()
        classes = predictions['classes'][0].cpu().detach().numpy()
        boxes = predictions['boxes'][0].cpu().detach().numpy()

        bbs = []
        for i, box in enumerate(boxes):
            bbs.append(BoundingBox(
                coords=box,
                class_name=dataset_loader.get_class_name(classes[i]),
                class_id=classes[i],
                objectiveness=scores[i],
                is_ground_truth=False
            ))

        return SampleLabels(
            sample_id=sample_id,
            datasetloader=dataset_loader,
            bounding_boxes_3d=bbs,
            is_ground_truth=False
        )

    def get_preds_from_label(self, sample_id: str, model_loader_load_x: Any,
                       dataset_loader: Optional[BaseDatasetLoader]) -> Optional[Dict]:
        labels = self._model_loader.load_y(sample_id, dataset_loader, model_loader_load_x)
        if not labels:
            return
        cls = [torch.tensor(v, device=self.device())[None, ...] for k, v in labels.items() if 'cls' in k]
        boxes = [torch.tensor(v, device=self.device())[None, ...] for k, v in labels.items() if 'bbox' in k]
        all_anchors = [v.to(self.device()) for v in self.internal_model.anchors.values()]

        from vinet_loss import Loss
        get_targets = Loss.get_targets

        reg = []
        for b, anchors in zip(boxes, all_anchors):
            targets = get_targets(b, anchors[None, ...])
            reg.append(targets)

        return self.internal_model.postprocess_preds(reg, cls)
