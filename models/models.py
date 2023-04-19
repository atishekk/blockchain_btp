from abc import ABC, abstractmethod
from typing import List, Dict, Any, cast
from blockchain.block import Block
import torch.nn as nn
from models.layers import Layer
from blockchain.block import Sentinel, Block, BlockMetadata, Neighbours
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlitedict import SqliteDict
import pickle
from collections import OrderedDict


class Model(ABC):
    def __init__(self, blocks: List[Block]) -> None:
        self._blocks = blocks
        self.sentinel = blocks[0]

    def blocks(self) -> List[Block]:
        return self._blocks

    def get_sentinel(self) -> Block:
        return self.sentinel

    @classmethod
    def encode(cls, model: "Model", file: str):
        keys = [b.hash for b in model.blocks()]
        db = SqliteDict(file)
        db["keys"] = pickle.dumps(keys)

        for block in model.blocks():
            b = (
                block.hash,
                block.AES,
                block.metadata,
                block.neighbours.encode(),
                block.nonce,
                block.priv.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ),
                block.params.layer.state_dict(),
                block.params.layer_name,
                block.params.layer_num,
            )
            db[block.hash] = pickle.dumps(b)

        db.commit()
        db.close()

    @ classmethod
    def decode(cls, file: str) -> "Model":
        db = SqliteDict(file)
        keys = pickle.loads(db["keys"])
        blocks = []

        for i, k in enumerate(keys):
            (hash,
             AES,
             metadata,
             neighbours,
             nonce,
             priv,
             state,
             layer_name,
             layer_num,
             ) = pickle.loads(db[k])

            if i == 0:
                block = Sentinel.load(
                    metadata,
                    Neighbours.decode(neighbours),
                    Layer(),
                    hash,
                    nonce,
                    AES,
                    cast(rsa.RSAPrivateKey,
                         serialization.load_pem_private_key(priv, None))
                )
            else:
                layer = cls.build_layer(layer_name, layer_num, state)
                l_priv = cast(
                    rsa.RSAPrivateKey,
                    serialization.load_pem_private_key(priv, None)
                )
                l_neighbours = Neighbours.decode(neighbours)
                block = Block.load(
                    metadata,
                    l_neighbours,
                    layer,
                    hash,
                    nonce,
                    AES,
                    l_priv)
            blocks.append(block)
        db.close()
        return cls(blocks)

    @ classmethod
    @ abstractmethod
    def build_layer(cls, name: str, index: int, state: OrderedDict) -> Layer:
        pass

    @ classmethod
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

            if layer[0] in LAYER:
                current_layer = LAYER[layer[0]](*([l] + layer[1:] + [state]))
            new_block = Block.build(BlockMetadata(), Neighbours(
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
