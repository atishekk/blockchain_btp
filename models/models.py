from abc import ABC, abstractmethod
from typing import List, Dict, Any
from blockchain.block import Block
import torch.nn as nn
from models.layers import Layer
from blockchain.block import Sentinel, Block, BlockMetadata, Neighbours
import random
from cryptography.hazmat.primitives import serialization


class Model(ABC):
    def __init__(self, blocks: List[Block]) -> None:
        self._blocks = blocks
        self.sentinel = blocks[0]

    def blocks(self) -> List[Block]:
        return self._blocks

    def get_sentinel(self) -> Block:
        return self.sentinel

    @classmethod
    def build(
            cls,
            state: Dict[str, Any],
            MODEL: List[List[Any]],
            LAYER: Dict[str, Any],
    ) -> List[Block]:
        sentinel = Sentinel(
            BlockMetadata(),
            Neighbours(bytes(0)),
            Layer()
        )
        blocks: List[Block] = [sentinel]
        prev_block = sentinel
        order = list(range(1, len(MODEL) + 1))
        # random.shuffle(order)
        current_layer = Layer()
        for l in order:
            layer = MODEL[l - 1]
            layer.append(state)
            if layer[0] in LAYER:
                current_layer = LAYER[layer[0]](*layer[1:])
            new_block = Block(BlockMetadata(), Neighbours(
                prev_block.hash), current_layer)
            blocks.append(new_block)
            prev_block = new_block
        blocks = cls.update_keys(blocks, order, MODEL)
        blocks[0].neighbours.hash = blocks[-1].hash
        return blocks

    @classmethod
    def update_keys(
        cls,
        blocks: List[Block],
        order: List[int],
        MODEL: List[List[Any]]
    ) -> List[Block]:
        order = [0] + order
        m = len(MODEL) + 1
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

    @classmethod
    @abstractmethod
    def load_model(cls) -> nn.Module:
        pass

    @classmethod
    def fetch_weights(cls):
        pass
