# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

import torch


class RegressionHead(torch.nn.Module):
    def __init__(self, num_features_in, feature_size, depth, num_anchors, box_features):
        super(RegressionHead, self).__init__()
        self.box_features = box_features
        prediction_net = []
        for _ in range(depth):
            prediction_net.append(torch.nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1))
            prediction_net.append(torch.nn.Mish())
            num_features_in = feature_size
        self.prediction_net = torch.nn.Sequential(*prediction_net)

        self.output = torch.nn.Conv2d(feature_size, num_anchors * box_features, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.prediction_net(x)
        out = self.output(out)

        out = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, -1, self.box_features)
        return out.contiguous()
