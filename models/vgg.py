from .models import Model
from typing import List
from models.layers import VGG11_Conv, VGG11_Linear
from blockchain.block import Block
import torch.nn as nn
import torchvision
from collections import OrderedDict
from .layers import Layer
from pathlib import Path
import torch


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

    def __init__(self, blocks: List[Block], order: List[int], verify_hash: bytes):
        super().__init__(blocks, order, verify_hash)

    @classmethod
    def load_model(cls, state_file: Path) -> nn.Module:
        model = torchvision.models.vgg11()
        model.load_state_dict(torch.load(str(state_file)))
        return model

    @classmethod
    def new(cls, state_file: Path) -> Model:
        model = cls.load_model(state_file)
        state = model.state_dict()
        blocks, order, verify_hash = cls.build(state, cls.MODEL, cls.LAYER)
        return cls(blocks, order, verify_hash)

    @classmethod
    def build_layer(cls, name: str, index: int, state: OrderedDict) -> Layer | None:
        p = [index] + cls.MODEL[index-1][1:] + [state, True]
        return cls.LAYER[name](*p)
