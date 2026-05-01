# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

import torch


class ClassificationHead(torch.nn.Module):
    def __init__(self, num_features_in, feature_size, depth, num_anchors, num_classes):
        super(ClassificationHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        classification_net = []
        for _ in range(depth):
            classification_net.append(torch.nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1))
            classification_net.append(torch.nn.Mish())
            num_features_in = feature_size
        self.classification_net = torch.nn.Sequential(*classification_net)

        self.output = torch.nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.classification_net(x)
        out = self.output(out)
        out = self.activation(out)

        out = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out.contiguous()
