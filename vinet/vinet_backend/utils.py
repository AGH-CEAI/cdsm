# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

import torch
import numpy as np


class BBoxTransform(torch.nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, anchors, deltas):
        batch_size = deltas[0].shape[0]
        deltas = torch.concat([d.reshape(batch_size, -1, 4) for d in deltas], axis=1)
        anchors = torch.tile(
            torch.concat([a.reshape(1, -1, 4).to(deltas.device) for a in anchors.values()], axis=1),
            dims=(batch_size, 1, 1)
        )

        self.std.to(deltas.device)
        self.mean.to(deltas.device)

        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * widths
        ctr_y = anchors[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class BBoxClip(torch.nn.Module):
    def __init__(self):
        super(BBoxClip, self).__init__()

    def forward(self, boxes, width, height):
        b_x1 = torch.clamp(boxes[:, :, 0], min=0)
        b_y1 = torch.clamp(boxes[:, :, 1], min=0)
        b_x2 = torch.clamp(boxes[:, :, 2], max=width)
        b_y2 = torch.clamp(boxes[:, :, 3], max=height)
        return torch.stack([b_x1, b_y1, b_x2, b_y2], dim=-1)