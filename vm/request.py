from collections import deque
import uuid
from enum import Enum


class RequestType(str, Enum):
    SETUP = "SETUP"
    QUERY = "QUERY"


class Request:
    def __init__(self, req_type: RequestType) -> None:
        self.id = uuid.uuid4()
        self.set = set()
        self.req_type = req_type


class RequestQueue:
    def __init__(self) -> None:
        self.queue = deque()
