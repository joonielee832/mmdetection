import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class ProbabilisticRetinaHead(AnchorHead):
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
                 loss_prob=dict(cls_var_loss=None, bbox_cov_loss=None),
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
        self.compute_cls_var = loss_prob['cls_var_loss'] != None
        self.compute_bbox_cov = loss_prob['bbox_cov_loss'] != None
        
        if self.compute_cls_var:
            self.cls_var_loss = loss_prob['cls_var_loss']['name']
            self.cls_var_num_samples = loss_prob['cls_var_loss']['num_samples']
        if self.compute_bbox_cov:
            self.bbox_cov_loss = loss_prob['bbox_cov_loss']['name']
            self.bbox_cov_type = loss_prob['bbox_cov_loss']['covariance_type']
            if self.bbox_cov_type == 'diagonal':
                # Diagonal covariance matrix has N elements
                self.bbox_cov_dims = 4
            else:
                # Number of elements required to describe an NxN covariance matrix is
                # computed as: (N*(N+1))/2
                self.bbox_cov_dims = 10
        super(ProbabilisticRetinaHead, self).__init__(
            num_classes,
            in_channels,
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
        if self.compute_cls_var:
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
        if self.compute_bbox_cov:
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
        """
        # cls_feat = x
        # reg_feat = x
        # for cls_conv in self.cls_convs:
        #     cls_feat = cls_conv(cls_feat)
        # for reg_conv in self.reg_convs:
        #     reg_feat = reg_conv(reg_feat)
        cls_feat = self.cls_net(x)
        reg_feat = self.reg_net(x)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        
        #? cls variance and bbox covariance estimation
        if self.compute_cls_var:
            cls_var = self.retina_cls_var(cls_feat)
        else:
            cls_var = None
        if self.compute_bbox_cov:
            bbox_cov = self.retina_reg_cov(reg_feat)
        else:
            bbox_cov = None
        return cls_score, bbox_pred, cls_var, bbox_cov
        # return cls_score, bbox_pred
