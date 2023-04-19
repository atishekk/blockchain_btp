from .models import Model
from typing import List
from models.layers import VGG11_Conv, VGG11_Linear
from blockchain.block import Block
import torch.nn as nn
import torchvision
from collections import OrderedDict
from .layers import Layer


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

    LAYER = {"VGG11_Conv": VGG11_Conv, "VGG11_Linear": VGG11_Linear, "": Layer}

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

    @classmethod
    def build_layer(cls, name: str, index: int, state: OrderedDict) -> Layer | None:
        p = [index] + cls.MODEL[index-1][1:] + [state, True]
        return cls.LAYER[name](*p)
