from abc import ABC, abstractmethod
from collections import OrderedDict
import torch.nn as nn
from typing import List


class Layer(ABC):
    def __init__(self):
        self.layer = nn.Sequential()
        self.layer_name = ""
        self.layer_num = 0

    def create_state(self, params: OrderedDict, index: int) -> OrderedDict:
        return OrderedDict()


class VGG11_Linear(Layer):
    def __init__(
            self,
            layer_num: int,
            in_features: int,
            out_features: int,
            output: bool,
            index: int,
            parameters: OrderedDict,
            loading: bool = False
    ):
        self.layer_name = "VGG11_Linear"
        self.layer_num = layer_num
        ll: List[nn.Module] = [
            nn.Linear(in_features, out_features)
        ]

        if not output:
            ll.extend([
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
            ])

        self.layer = nn.Sequential(*ll)
        if not loading:
            new_state = self.create_state(parameters, index)
        else:
            new_state = parameters
        self.layer.load_state_dict(new_state)

    def create_state(self, params: OrderedDict, index: int) -> OrderedDict:
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
            layer_num: int,
            in_channels: int,
            out_channels: int,
            pooling: bool,
            feature_output: bool,
            index: int,
            parameters: OrderedDict,
            loading: bool = False
    ):
        self.layer_name = "VGG11_Conv"
        self.layer_num = layer_num
        ll = [
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True)
        ]
        if pooling:
            ll.append(nn.MaxPool2d(2, 2))

        if feature_output:
            ll.append(nn.AdaptiveAvgPool2d((7, 7)))

        self.layer = nn.Sequential(*ll)
        if not loading:
            new_state = self.create_state(parameters, index)
        else:
            new_state = parameters
        self.layer.load_state_dict(new_state)

    def create_state(self, params: OrderedDict, index: int) -> OrderedDict:
        new_state = OrderedDict()
        key = f"features.{index}."
        for t in ["weight", "bias"]:
            k = key+t
            new_state["0." + t] = params[k]

        return new_state
