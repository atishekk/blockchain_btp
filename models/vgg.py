from .models import Model
from typing import List
from models.layers import Layer
from blockchain.block import Block, Sentinel, BlockMetadata, Neighbours
import torch.nn as nn
import random
from collections import OrderedDict
import torchvision
from cryptography.hazmat.primitives import serialization


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
        self._blocks = blocks
        self.sentinel = blocks[0]

    def blocks(self) -> List[Block]:
        return self._blocks

    def get_sentinel(self) -> Block:
        return self.sentinel

    @classmethod
    def load_model(cls) -> nn.Module:
        # temp will move to using fetch_weights
        # return torchvision.models.vgg11(torchvision.models.VGG11_Weights.IMAGENET1K_V1)
        return torchvision.models.vgg11()

    @ classmethod
    def build(cls) -> Model:
        """
        Build the VGG11 model
        """

        model = cls.load_model()
        state = model.state_dict()
        sentinel = Sentinel(
            BlockMetadata(),
            Neighbours(bytes(0)),
            Layer()
        )
        blocks: List[Block] = [sentinel]
        prev_block = sentinel
        order = list(range(1, len(cls.MODEL) + 1))
        random.shuffle(order)
        current_layer = Layer()
        for l in order:
            layer = cls.MODEL[l - 1]
            layer.append(state)
            match layer[0]:
                case "VGG11_Conv":
                    current_layer = VGG11_Conv(*layer[1:])
                case "VGG11_Linear":
                    current_layer = VGG11_Linear(*layer[1:])
            new_block = Block(BlockMetadata(), Neighbours(
                prev_block.hash), current_layer)
            blocks.append(new_block)
            prev_block = new_block

        blocks = cls.update_keys(blocks, order)
        blocks[0].neighbours.hash = blocks[-1].hash
        return VGG11(blocks)

    @classmethod
    def update_keys(cls, blocks: List[Block], order: List[int]) -> List[Block]:
        order += [0]
        m = len(cls.MODEL) + 1
        for layer_num in range(0, m):
            curr_layer = blocks[order.index(layer_num)]
            next_layer = blocks[order.index((layer_num + 1) % m)]
            prev_layer = blocks[order.index((layer_num - 1) % m)]

            curr_layer.neighbours.set_next_pub_key(
                next_layer.pub.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            )
            curr_layer.neighbours.set_prev_pub_key(
                prev_layer.pub.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            )

        return blocks

# class VGG16(Model):
#     def __init__(self):
#         pass

#     def blocks(self) -> List[Block]:
#         pass

#     @classmethod
#     def build(cls) -> Model:
#         pass
