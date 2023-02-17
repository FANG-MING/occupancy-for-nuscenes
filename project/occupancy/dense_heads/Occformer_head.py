
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmseg.models import builder, HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16


@HEADS.register_module()
class OccFormerHead(BaseModule):

    def __init__(self,
                 *args,
                 transformer=None,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 pc_range,
                 **kwargs):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_z = self.pc_range[5] - self.pc_range[2]
        bev_mask = torch.zeros(1, bev_h, bev_w)
        self.register_buffer('bev_mask', bev_mask)

        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = builder.build_head(transformer)
        self.embed_dims = self.transformer.embed_dims
        self._init_layers()

    def _init_layers(self):
        self.bev_embedding_hw = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        bev_queries_hw = self.bev_embedding_hw.weight.to(dtype)

        bev_mask = self.bev_mask.expand(bs, -1, -1)
        bev_pos_hw = self.positional_encoding(bev_mask).to(dtype)
        bev_pos_hw = bev_pos_hw.reshape(bs, self.bev_h*self.bev_w, -1).permute(1, 0, 2)
        return self.transformer(
            mlvl_feats,
            bev_queries_hw,
            self.bev_h,
            self.bev_w,
            query_pos=bev_pos_hw,
            img_metas=img_metas,
        )
