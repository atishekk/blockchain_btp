import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, cast, Tuple
import random
import pickle
from collections import OrderedDict
from pathlib import Path

import torch.nn as nn
import torch
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlitedict import SqliteDict

from blockchain.block import Block
from models.layers import Layer
from blockchain.block import Sentinel, Block, BlockMetadata, Neighbours
from vm.request import QueryInput


class Model(ABC):
    """
        The class represents the blockchain
        based CNN model representation
    """

    def __init__(self, blocks: List[Block], order: List[int], verify_hash: bytes) -> None:
        self._blocks = blocks
        self.sentinel = cast(Sentinel, blocks[0])
        self.order = order
        self.verify_hash = verify_hash

    def blocks(self) -> List[Block]:
        return self._blocks

    def get_sentinel(self) -> Block:
        return self.sentinel

    @abstractmethod
    def load_and_preprocess(self, img_path: Path) -> torch.Tensor:
        pass

    @classmethod
    def encode(cls, model: "Model", file: Path) -> bytes:
        """

        """
        keys = [b.hash for b in model.blocks()]
        db = SqliteDict(str(file))
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

        db["order"] = pickle.dumps(model.order)
        db["verify"] = pickle.dumps(model.verify_hash)

        db.commit()
        db.close()

        # encrypt the model file
        fernet_key = Fernet.generate_key()
        encryptor = Fernet(fernet_key)
        with open(file, "rb") as dict_model:
            data = dict_model.read()

        en_data = encryptor.encrypt(data)

        with open(file, "wb") as model_file:
            model_file.write(en_data)
        return fernet_key

    @ classmethod
    def decode(cls, file: Path, key: str) -> "Model":
        decryptor = Fernet(key)
        with open(file, "rb") as model_file:
            data = model_file.read()

        de_data = decryptor.decrypt(data)
        de_file = Path("de_"+file.stem + ".sqlite")
        with open(de_file, "wb") as model_file:
            model_file.write(de_data)

        db = SqliteDict(str(de_file))
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
        order = pickle.loads(db["order"])
        verify_hash = pickle.loads(db["verify"])
        db.close()
        os.remove(de_file)
        return cls(blocks, order, verify_hash)

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
    ) -> Tuple[List[Block], List[int], bytes]:

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

        verify_hash = cls.build_hash(blocks, order)
        return blocks, order, verify_hash

    @classmethod
    def build_hash(cls, blocks: List[Block], order: List[int]) -> bytes:
        hasher = hashes.Hash(hashes.SHA3_256())
        order = [0] + order
        for layer_num in range(len(order)):
            block = blocks[order.index(layer_num)]
            hasher.update(block.hash)
        return hasher.finalize()

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
    def load_model(cls, state_file: Path) -> nn.Module:
        pass

    @classmethod
    @abstractmethod
    def new(cls, state_file: Path) -> "Model":
        pass

    @classmethod
    def fetch_model_file(cls, model_name: str) -> str:
        # fetch the model file here from IPFS
        return model_name + ".sqlite"
