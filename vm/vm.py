import json
from typing import cast, List, Dict, Type
from enum import Enum
from pathlib import Path

import torch
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes

from models.models import Model
from models.vgg import VGG11
from blockchain.block import Sentinel
from .request import RequestQueue, Request, QueryInput, SetupInput, RequestType, Result


class State(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    WORKING = "WORKING"
    RECOVERY = "RECOVERY"


class VM:
    MODELS: Dict[str, Type[Model]] = {"VGG11": VGG11}

    def __init__(self, queue: RequestQueue) -> None:
        self.queue = queue
        self.trusted = self._init_trusted_list()
        self.priv = self._init_RSA_key()
        self.state = State.NOT_READY
        self.device = torch.device("cpu")

    def _init_trusted_list(self) -> Dict[bytes, List[RequestType]]:
        """
        Reads the trusted sources json file named .trusted
        in the current directory
        """
        trusted = dict()
        with open(".trusted") as f:
            for key, perm in json.load(f).items():
                k = bytes(key, "utf-8")
                v = [RequestType(x) for x in perm]
                trusted[k] = v
        return trusted

    def _init_RSA_key(self) -> rsa.RSAPrivateKey:
        """
        Reads the private key from the PEM formated file
        name key.pem
        """
        with open("key.pem", "rb") as f:
            return cast(
                rsa.RSAPrivateKey,
                serialization.load_pem_private_key(f.read(), None)
            )

    def setup(self, input: SetupInput):
        self.state = State.WORKING

        if input.model in self.MODELS:
            model_cls = self.MODELS[input.model]
            self.model = model_cls.decode(input.model_file, input.key)
            self.state = State.READY
            self.setup_input = input

        else:
            raise Exception("")

    def query(self, input: QueryInput) -> Result:
        """
        Perform inference on the model
        """
        self.state = State.WORKING

        iters = len(self.model.blocks()) - 1
        # sentinel processing
        sen_out = self.model.sentinel.compute(input, self.device)
        self.queue.add(Request().add_results(sen_out))

        for _ in range(iters):
            req = self.queue.peek()
            inter = req.get_recent_res()
            for b in self.model.blocks():
                if b.check_prev(inter) and type(b) is not Sentinel:
                    res = b.compute(inter, self.device)
                    req.add_results(res)
                    break
                else:
                    continue
        if self.verify():
            self.state = State.READY
            res = self.queue.peek().get_recent_res()
            return self.model.sentinel.finalise(res, self.device)
        else:
            self.recovery_mode()
            raise Exception("")

    def verify(self) -> bool:
        hasher = hashes.Hash(hashes.SHA3_256())
        for res in self.queue.peek().set:
            hasher.update(res.block_hash)

        h = hasher.finalize()
        return h == self.model.verify_hash

    def recovery_mode(self):
        self.state = State.RECOVERY
        # perform revovery here
        self.state = State.READY
        # raise an exception here
        raise Exception("")
