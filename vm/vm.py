import json
from typing import cast, List, Dict

from models.models import Model
from request import RequestQueue, RequestType
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from enum import Enum


class State(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class VM:
    def __init__(self, queue: RequestQueue) -> None:
        self.queue = queue
        self.trusted = self._init_trusted_list()
        self.priv = self._init_RSA_key()
        self.state = State.NOT_READY

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

    def get_vm_status(self) -> State:
        return self.state

    def run(self, model: Model):
        """
        Perform inference on the model
        """
        pass


def infer(vm: VM, model: Model, data: bytes):
    pass
