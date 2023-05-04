from typing import List
from pathlib import Path
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision

from .models import Model
from models.layers import VGG11_Conv, VGG11_Linear
from blockchain.block import Block
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

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224

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

    def load_and_preprocess(self, img_path: Path) -> torch.Tensor:
        img = Image.open(str(img_path))
        img = img.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        img_array = np.asarray(img, np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        tensor = torch.from_numpy(img_array)
        norm = torchvision.transforms.Normalize(self.MEAN, self.STD)
        return norm(tensor)
