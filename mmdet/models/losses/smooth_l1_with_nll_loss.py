# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .smooth_l1_loss import smooth_l1_loss
from .utils import weight_reduce_loss


@LOSSES.register_module()
class SmoothL1WithNLL(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', covariance_type='diagonal', annealing_step=174640, loss_weight=1.0):
        super(SmoothL1WithNLL, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.covariance_type = covariance_type
        self.annealing_step = annealing_step

    def forward(self,
                pred,
                pred_cov,
                target,
                current_step,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            prev_cov (torch.Tensor): The covariance of the prediction
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        #? log of the covariance; need to clamp it, else nll goes to infinity
        #* eq 8 of What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision
        pred_cov = torch.clamp(pred_cov, -7.0, 7.0)
        loss_bbox = 0.5 * torch.exp(-pred_cov) * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction='none',
            avg_factor=None,
            **kwargs)
        loss_cov_reg = 0.5 * pred_cov
        loss_bbox += loss_cov_reg

        loss_bbox = weight_reduce_loss(loss_bbox, weight, reduction, avg_factor)
        
        #? Perform loss annealing
        standard_loss_bbox = smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        probabilistic_weight = min(1.0, current_step/self.annealing_step)
        probabilistic_weight = (100**probabilistic_weight-1.0)/(100.0-1.0)
        loss_bbox = (1.0 - probabilistic_weight) * \
            standard_loss_bbox + probabilistic_weight * loss_bbox
        loss_bbox = self.loss_weight * loss_bbox
        return loss_bbox
