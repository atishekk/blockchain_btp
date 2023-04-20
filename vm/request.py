from ordered_set import OrderedSet
import uuid
from collections import deque
from abc import ABC, abstractmethod
import torch
from enum import Enum


class Input(ABC):

    @abstractmethod
    def validate(self) -> bool:
        pass


class SetupInput(Input):
    def __init__(self, model: str) -> None:
        self.model = model

    def validate(self) -> bool:
        pass


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

    def validate(self) -> bool:
        pass


class Intermediate:
    def __init__(self) -> None:
        pass

    def result(self):
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
    def __init__(self, ledger_file: str) -> None:
        self.queue: deque[Request] = deque()
        self.ledger = ledger_file

    def add(self, req: Request):
        self.queue.append(req)

    def remove(self) -> Request:
        return self.queue.popleft()

    def peek(self) -> Request:
        return self.queue[0]
