from collections import deque
import uuid
from ..utils.ordered_set import OrderedSet


class Request:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.set = OrderedSet()


class RequestQueue:
    def __init__(self) -> None:
        self.queue = deque()
