import torch
import timm
import torch.nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class TIMMResNet(nn.Module):
    timm_archs = {
        18: "resnet18",
        26: "resnet26",
        34: "resnet34",
        50: "resnet50",
        101: "resnet101d",
        152: "resnet152d",
        200: "resnet200d",
    }

    def __init__(self, depth, out_indices, output_stride=8, pretrained=True):
        super(TIMMResNet, self).__init__()
        if depth not in self.timm_archs.keys():
            raise KeyError(f"invalid depth {depth} for resnet")

        self.model = timm.create_model(
            self.timm_archs[depth],
            features_only=True,
            output_stride=output_stride,
            out_indices=out_indices,
            pretrained=pretrained,
        )

    def forward(self, x):
        out = self.model(x)
        return tuple(out)

    def init_weights(self, pretrained=None):
        pass
