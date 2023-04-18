from abc import ABC, abstractmethod
from typing import List
from blockchain.block import Block
import torch.nn as nn


class Model(ABC):
    @abstractmethod
    def blocks(self) -> List[Block]:
        pass

    @classmethod
    @abstractmethod
    def build(cls) -> "Model":
        pass

    @classmethod
    @abstractmethod
    def load_model(cls) -> nn.Module:
        pass

    @classmethod
    def fetch_weights(cls):
        pass
