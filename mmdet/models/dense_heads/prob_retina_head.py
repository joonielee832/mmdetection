import warnings

import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule

from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
# from .anchor_head import AnchorHead
from .retina_head import RetinaHead
from .dense_test_mixins import BBoxTestMixin

@HEADS.register_module()
class ProbabilisticRetinaHead(RetinaHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 dropout_rate=0.0,
                 bbox_covariance_type="diagonal",
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dropout_rate = dropout_rate
        # Diagonal covariance matrix has N elements
        # Number of elements required to describe an NxN covariance matrix is
        # computed as: (N*(N+1))/2
        self.bbox_cov_dims = 4 if bbox_covariance_type == "diagonal" else 10

        super(ProbabilisticRetinaHead, self).__init__(
            num_classes,
            in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            if self.dropout_rate > 0.0:
                self.cls_convs.append(nn.Dropout(p=self.dropout_rate))
                self.reg_convs.append(nn.Dropout(p=self.dropout_rate))
        self.cls_net = nn.Sequential(*self.cls_convs)
        self.reg_net = nn.Sequential(*self.reg_convs)
        
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        
        for modules in [self.retina_cls, self.retina_reg]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        
        #? Create module for classification variance estimation
        self.retina_cls_var = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        
        for layer in self.retina_cls_var.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, -10.0)
        
        #? Create module for bounding box covariance estimation
        self.retina_reg_cov = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.bbox_cov_dims,
            3,
            padding=1)
        
        for layer in self.retina_reg_cov.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.0001)
                nn.init.constant_(layer.bias, 0.0)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                cls_var (Tensor): Cls variance for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_cov (Tensor): Box covariances for a single scale level,
                    the channels number is num_anchors * bbox_cov_dims
        """
        cls_feat = self.cls_net(x)
        reg_feat = self.reg_net(x)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        
        #? cls variance and bbox covariance estimation
        cls_var = self.retina_cls_var(cls_feat)
        bbox_cov = self.retina_reg_cov(reg_feat)

        return cls_score, bbox_pred, cls_var, bbox_cov

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
                - cls_vars (list[Tensor]): Classification variances for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is self.num_base_priors * num_classes.
                - bbox_covs (list[Tensor]): Box covariances for all scale \
                    levels, each is a 4-D tensor, the channels number is \
                    num_base_priors * bbox_cov_dims (4)
        """
        return multi_apply(self.forward_single, feats)
    
    #TODO: loss_single function overwrite from anchor_head
    def loss_single(self, cls_score, bbox_pred, cls_var, bbox_cov, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            cls_var (Tensor): Box class variance for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_cov (Tensor): Box corner covariance for each scale
                level with shape (N, num_anchors * self.bbox_cov_dims, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        if cls_var is not None:
            cls_var = cls_var.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, cls_var, labels, label_weights, avg_factor=num_total_samples)
        
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if bbox_cov is not None:
            bbox_cov = bbox_cov.permute(0, 2, 3, 1).reshape(-1, self.bbox_cov_dims)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            #! Decoding not implemented for bbox covariance for IouLoss
            raise NotImplementedError("Decoding not implemented for bbox covariance for other reg losses")
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_cov,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, None, None
    
    #TODO: loss function overwrite from anchor_head
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'cls_vars', 'bbox_covs'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_vars,
             bbox_covs,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            cls_vars (list[Tensor]): Box score variances for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_covs (list[Tensor]): Box covariances for each scale
                level with shape (N, num_anchors * bbox_cov_dims, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        breakpoint()
        # first: [4, 90, 92, 160]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        breakpoint()
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_cls_vars, losses_bbox_covs = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            cls_vars,
            bbox_covs,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        
        #! Remove
        losses_cls_vars = [torch.ones_like(l_cls) for l_cls in losses_cls]
        losses_bbox_covs = [torch.ones_like(l_bbox) for l_bbox in losses_bbox]
        
        return dict(loss_cls_score=losses_cls, 
                    loss_bbox_reg=losses_bbox, 
                    loss_cls_var=losses_cls_vars, 
                    loss_bbox_cov=losses_bbox_covs)