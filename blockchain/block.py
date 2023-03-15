import time

from cryptography.hazmat.primitives import hashes

from ..models.layers import LayerMetadata, LayerParams


class BlockMetadata:
    def __init__(self, nonce: int, layer_meta: LayerMetadata) -> None:
        self.timestamp = time.time_ns()
        self.nonce = nonce
        self.layer_metadata = layer_meta


class BlockNeighbours:
    def __init__(self) -> None:
        pass


class Block:
    def __init__(self, metadata: BlockMetadata, neighbours: BlockNeighbours, params: LayerParams) -> None:
        self.metadata = metadata
        self.neighbours = neighbours
        self.params = params
        self.hash = self.compute_hash()

    def compute_hash():
        digest = hashes.Hash(hashes.SHA3_256())
