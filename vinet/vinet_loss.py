# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

import torch


class Loss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25,
                 gamma: float = 2.0,
                 epsilon: float = 1e-6) -> None:
        super(Loss, self).__init__()
        try:
            model_loader_list = [v for v in self._INTERNAL_PARAMS['linked'].values()
                                 if v.CONFIG_TYPE == 'ModelLoader']
            model_loader = model_loader_list[0]
            self.anchors = model_loader.get_anchors()
            self.classes = model_loader.classes
        except Exception as e:
            raise Exception('Model loader with get_anchors() method '
                            'needs to be provided as LinkedObject!: {}'.format(e))

        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon

    def get_target_anchors(self, target: torch.Tensor) -> torch.Tensor:
        s1, s2 = target.shape[0:2]
        return self.anchors['{}x{}'.format(s1, s2)].to(target.device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.shape[-1] == len(self.classes) and torch.max(y_true) <= 1:
            loss_fn = self.classification_loss
        else:
            loss_fn = self.regresion_loss
        total_loss = []
        for b in range(y_true.shape[0]):
            b_y_pred = y_pred[b]
            b_y_true = y_true[b]
            if torch.max(b_y_true) == 0:
                total_loss.append(
                    torch.tensor(0.0, device=b_y_true.device, dtype=b_y_true.dtype)
                )
            else:
                total_loss.append(loss_fn(b_y_pred, b_y_true))

        return torch.stack(total_loss).mean(dim=0, keepdim=True)

    def regresion_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        positives = torch.where(y_true.max(axis=-1).values > 0)
        anchors = self.get_target_anchors(y_true)

        anchors = anchors[positives]
        y_pred = y_pred[positives]
        y_true = y_true[positives]

        targets = self.get_targets(y_true, anchors)

        regression_diff = torch.abs(targets - y_pred)
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0
        )

        return regression_loss.mean()

    @staticmethod
    def get_targets(y_true: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        anchors_width = anchors[..., 2] - anchors[..., 0]
        anchors_height = anchors[..., 3] - anchors[..., 1]
        anchors_ctr_x = anchors[..., 0] + 0.5 * anchors_width
        anchors_ctr_y = anchors[..., 1] + 0.5 * anchors_height

        true_width = torch.clamp(y_true[..., 2] - y_true[..., 0], min=1)
        true_height = torch.clamp(y_true[..., 3] - y_true[..., 1], min=1)
        true_ctr_x = y_true[..., 0] + 0.5 * true_width
        true_ctr_y = y_true[..., 1] + 0.5 * true_height

        targets_dx = (true_ctr_x - anchors_ctr_x) / anchors_width
        targets_dy = (true_ctr_y - anchors_ctr_y) / anchors_height
        targets_dw = torch.log(true_width / anchors_width)
        targets_dh = torch.log(true_height / anchors_height)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=-1)
        targets = targets / torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=targets.device)
        return targets

    def classification_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        alpha_factor = torch.ones(y_true.shape).to(y_true.device) * self.alpha
        alpha_factor = torch.where(torch.eq(y_true, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(y_true, 1.), 1. - y_pred, y_pred)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        bce = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
        cls_loss = focal_weight * bce
        cls_loss = torch.where(torch.ne(y_true, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=cls_loss.device))
        num_positive_anchors = torch.eq(y_true, 1.).sum()
        norm_cls_loss = cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
        return norm_cls_loss
