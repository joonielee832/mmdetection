import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .focal_loss import sigmoid_focal_loss, py_sigmoid_focal_loss, py_focal_loss_with_prob

@LOSSES.register_module()
class FocalAttenuatedLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 num_samples=10,
                 activated=False):
        """

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalAttenuatedLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        self.num_samples = num_samples

    def forward(self,
                pred,
                pred_var,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            pred (torch.Tensor): The predicted variance
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    #* The following is for debugging; not implemented for Probabilistic Retinanet
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            #? Produce normal samples using logits as the mean and std computed above
            pred_var = torch.sqrt(torch.exp(pred_var))
            univ_norm_dists = torch.distributions.normal.Normal(pred, scale=pred_var)
            pred_stochastic = univ_norm_dists.rsample((self.num_samples,))
            pred_stochastic = pred_stochastic.reshape(pred_stochastic.shape[1]*self.num_samples,pred_stochastic.shape[2])
            
            #? Produce copies of the target classes to match the number of samples
            target = torch.unsqueeze(target, 0)
            target = torch.repeat_interleave(target, self.num_samples, dim=0).reshape(target.shape[1]*self.num_samples)
            weight = torch.unsqueeze(weight, 0)
            weight = torch.repeat_interleave(weight, self.num_samples, dim=0).reshape(weight.shape[1]*self.num_samples)
            
            avg_factor *= self.num_samples
            loss_cls = self.loss_weight * calculate_loss_func(
                pred_stochastic,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls