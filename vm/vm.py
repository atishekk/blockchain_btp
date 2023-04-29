import json
from typing import cast, List, Dict, Type

from models.models import Model
from models.vgg import VGG11
from .request import RequestQueue, Request, QueryInput, SetupInput, Input, RequestType
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from enum import Enum


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

    def setup(self, input: SetupInput) -> Model:
        self.state = State.WORKING
        input.validate()

        if input.model in self.MODELS:
            model_cls = self.MODELS[input.model]
            file = model_cls.fetch_model_file(input.model)
            self.state = State.READY
            return model_cls.decode(file)
        else:
            raise Exception("")

    def query(self, model: Model, input: QueryInput):
        """
        Perform inference on the model
        """
        self.state = State.WORKING
        input.validate()

        sen_out = model.sentinel.compute(input)
        iters = len(model.blocks())
        self.queue.add(Request().add_results(sen_out))

        for _ in range(iters):
            req = self.queue.peek()
            inter = req.get_recent_res()
            for b in model.blocks():
                if b.check_prev(inter):
                    res = b.compute(inter)
                    req.add_results(res)
                    break
                else:
                    continue
        if self.verify():
            self.state = State.READY
            return self.queue.peek().get_recent_res().result()
        else:
            self.recovery_mode()

    def verify(self) -> bool:
        pass

    def recovery_mode(self):
        self.state = State.RECOVERY
        # perform revovery here
        self.state = State.READY
        # raise an exception here
