
import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models import HEADS


@HEADS.register_module()
class OccFuser(BaseModule):
    def __init__(
        self, bev_h, bev_w, bev_z, nbr_classes=20, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=True
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Conv3d(in_dims, hidden_dims, kernel_size=3, dilation=2,padding=2),
            nn.BatchNorm3d(hidden_dims),
            nn.Conv3d(hidden_dims, out_dims, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_dims),
        )
        self.z_embeding = nn.Embedding(bev_z, in_dims)
        self.classifier = nn.Conv3d(out_dims, nbr_classes, 1)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint
    
    def forward(self, bev_feature):

        bs, _, c = bev_feature.shape
        bev_feature = bev_feature.permute(0, 2, 1).reshape(bs, c, self.bev_h, self.bev_w)
        if self.scale_h != 1 or self.scale_w != 1:
            bev_hw = F.interpolate(
                bev_hw, 
                size=(self.bev_h*self.scale_h, self.bev_w*self.scale_w),
                mode='bilinear'
            )
        bev_feature = bev_feature[..., None] + self.z_embeding.weight.permute(1, 0)[None, :, None, None, :]
        bev_feature = self.decoder(bev_feature)
        logits = self.classifier(bev_feature)
        return logits
