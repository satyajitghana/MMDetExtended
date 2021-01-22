# MMDetExtended

![logo](logo.png)

This is a fork of [mmdetection](https://github.com/open-mmlab/mmdetection/issues/4273) with support for integrating [TIMM](https://github.com/rwightman/pytorch-image-models) models

## Run with docker

```
docker/build
docker/run --gpu
```

Now modify configs/retinanet/retinanet_timm_r50_fpn_1x_catdog.py with your own dataset

And start training with

```
python tools/train.py /opt/src/configs/retinanet/retinanet_timm_r50_fpn_1x_catdog.py
```

## Models

Here is an example of Resnet18 from TIMM integrated with MMDetection

```python
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

```

And it can be used as a backbone in config file like

```python
    backbone=dict(type="TIMMResNet", depth=50, out_indices=(1, 2, 3, 4)),
```

## TODO

- [ ] Find a better way instead of cloning and using mmdetection, rather we would want to just install and use, without modifying source and building it
- [ ] Find a way to integrate all of TIMM models, easy and quickly
