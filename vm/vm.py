from .request import RequestQueue


class VM:
    def __init__(self, queue: RequestQueue) -> None:
        self.queue = queue
