# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

import torch
import numpy as np


class BBoxTransform(torch.nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25]).astype(np.float32))
        else:
            self.std = std

    def forward(self, anchors, deltas):
        batch_size = deltas[0].shape[0]
        deltas = torch.concat([d.reshape(batch_size, -1, 9) for d in deltas], axis=1)
        anchors = torch.tile(
            torch.concat([a.reshape(1, -1, 9).to(deltas.device) for a in anchors.values()], axis=1),
            dims=(batch_size, 1, 1)
        )

        self.std.to(deltas.device)
        self.mean.to(deltas.device)

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dz = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dl = deltas[:, :, 3] * self.std[3] + self.mean[3]
        dw = deltas[:, :, 4] * self.std[4] + self.mean[4]
        dh = deltas[:, :, 5] * self.std[5] + self.mean[5]
        yaw = deltas[:, :, 6] * self.std[6] + self.mean[6]
        pitch = deltas[:, :, 7] * self.std[7] + self.mean[7]
        roll = deltas[:, :, 8] * self.std[8] + self.mean[8]

        pred_x = anchors[:, :, 0] + dx * anchors[:, :, 3]
        pred_y = anchors[:, :, 1] + dy * anchors[:, :, 4]
        pred_z = anchors[:, :, 2] + dz * anchors[:, :, 5]
        pred_l = torch.exp(dl) * anchors[:, :, 3]
        pred_w = torch.exp(dw) * anchors[:, :, 4]
        pred_h = torch.exp(dh) * anchors[:, :, 5]

        yaw = torch.atan2(yaw, pitch)
        pitch = torch.zeros_like(pitch)
        roll = torch.zeros_like(roll)

        pred_boxes = torch.stack(
            [pred_x, pred_y, pred_z, pred_l, pred_w, pred_h, yaw, pitch, roll],
            dim=2
        )

        return pred_boxes
