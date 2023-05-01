from ordered_set import OrderedSet
import uuid
from collections import deque
from abc import ABC, abstractmethod
import torch
from enum import Enum
import pickle
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vm.vm import VM


class Input(ABC):
    @abstractmethod
    def validate(self, vm: "VM"):
        pass


class SetupInput(Input):
    def __init__(self, model: str) -> None:
        self.model = model

    def validate(self, vm: "VM"):
        if self.model not in vm.MODELS:
            raise Exception(f"Invalid query input - model: {self.model}")


class QueryInput(Input):
    def __init__(
            self,
            data: torch.Tensor,
            public_key: bytes,
            signature: bytes
    ) -> None:
        self.public_key = public_key
        self.data = data
        self.signature = signature

    def validate(self, vm: "VM"):
        pass


class Result:
    def __init__(self) -> None:
        pass


class Intermediate:
    def __init__(
            self,
            block_hash: bytes,
            en_AES: bytes,
            comp: bytes,
            digital_sig: bytes,
            iv: bytes
    ) -> None:
        self.block_hash = block_hash
        self.en_AES = en_AES
        self.comp = comp
        self.digital_sig = digital_sig
        self.iv = iv

    def serialise(self) -> bytes:
        return pickle.dumps((self.block_hash, self.en_AES, self.comp, self.iv))

    def result(self) -> Result:
        pass


class RequestType(str, Enum):
    SETUP = "SETUP"
    QUERY = "QUERY"


class Request:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.set: OrderedSet[Intermediate] = OrderedSet([])

    def add_results(self, res: Intermediate) -> "Request":
        self.set.append(res)
        return self

    def get_recent_res(self) -> Intermediate:
        return self.set[-1]


class RequestQueue:
    def __init__(self, ledger_file: Path) -> None:
        self.queue: deque[Request] = deque()
        self.ledger = ledger_file

    def add(self, req: Request):
        self.queue.append(req)

    def remove(self) -> Request:
        return self.queue.popleft()

    def peek(self) -> Request:
        return self.queue[0]
