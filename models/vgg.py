from .models import Model
from typing import List
from models.layers import Layer
from blockchain.block import Block
import torch.nn as nn
import random
from collections import OrderedDict
import torchvision


class VGG11_Linear(Layer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            output: bool,
            index: int,
            parameters: OrderedDict,
    ):
        ll: List[nn.Module] = [
            nn.Linear(in_features, out_features)
        ]

        if not output:
            ll.extend([
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
            ])

        self.layer = nn.Sequential(*ll)
        new_state = self.create_dict(parameters, index)
        self.layer.load_state_dict(new_state)

    def create_dict(self, params: OrderedDict, index: int) -> OrderedDict:
        new_state = OrderedDict()
        key = f"classifier.{index}."
        for t in ["weight", "bias"]:
            k = key+t
            new_state["0." + t] = params[k]

        return new_state


class VGG11_Conv(Layer):
    """
    The layer consists of
    Conv2d, ReLU and MaxPool2d
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            pooling: bool,
            feature_output: bool,
            index: int,
            parameters: OrderedDict,
    ):
        ll = [
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True)
        ]
        if pooling:
            ll.append(nn.MaxPool2d(2, 2))

        if feature_output:
            ll.append(nn.AdaptiveAvgPool2d((7, 7)))

        self.layer = nn.Sequential(*ll)

        new_state = self.create_dict(parameters, index)
        self.layer.load_state_dict(new_state)

    def create_dict(self, params: OrderedDict, index: int) -> OrderedDict:
        new_state = OrderedDict()
        key = f"features.{index}."
        for t in ["weight", "bias"]:
            k = key+t
            new_state["0." + t] = params[k]

        return new_state


class VGG11(Model):
    MODEL = [
        ["VGG11_Conv", 3, 64, True, False, 0],
        ["VGG11_Conv", 64, 128, True, False, 3],
        ["VGG11_Conv", 128, 256, False, False, 6],
        ["VGG11_Conv", 256, 256, True, False, 8],
        ["VGG11_Conv", 256, 512, False, False, 11],
        ["VGG11_Conv", 512, 512, True, False, 13],
        ["VGG11_Conv", 512, 512, False, False, 16],
        ["VGG11_Conv", 512, 512, True, True, 18],
        ["VGG11_Linear", 25088, 4096, False, 0],
        ["VGG11_Linear", 4096, 4096, False, 3],
        ["VGG11_Linear", 4096, 1000, False, 6],
    ]

    LAYER = {"VGG11_Conv": VGG11_Conv, "VGG11_Linear": VGG11_Linear}

    def __init__(self, blocks: List[Block]):
        super().__init__(blocks)

    @classmethod
    def load_model(cls) -> nn.Module:
        # temp will move to using fetch_weights
        # return torchvision.models.vgg11(torchvision.models.VGG11_Weights.IMAGENET1K_V1)
        return torchvision.models.vgg11()

    @classmethod
    def new(cls) -> Model:
        model = cls.load_model()
        state = model.state_dict()
        blocks = cls.build(state, cls.MODEL, cls.LAYER)
        return cls(blocks)


# class VGG16(Model):
#     def __init__(self):
#         pass

#     def blocks(self) -> List[Block]:
#         pass

#     @classmethod
#     def build(cls) -> Model:
#         pass
