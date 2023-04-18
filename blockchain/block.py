import time
import os
from typing import Tuple
from cryptography.hazmat.primitives import serialization
import base64

from cryptography.hazmat.primitives.asymmetric import rsa

from utils.proof_of_work import ProofOfWork

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.layers import Layer


class BlockMetadata:
    def __init__(self) -> None:
        self.timestamp = time.time_ns()


class Neighbours:
    def __init__(
            self,
            hash: bytes,
    ) -> None:
        self.hash = hash

    def set_prev_pub_key(self, prev_pub_key: bytes):
        self.prev_pub_key = serialization.load_pem_public_key(prev_pub_key)

    def set_next_pub_key(self, next_pub_key: bytes):
        self.next_pub_key = serialization.load_pem_public_key(next_pub_key)


class Block:

    def __init__(
            self,
            metadata: BlockMetadata,
            neighbours: Neighbours,
            params: "Layer"
    ) -> None:
        """
        Build a new block using the blockmetadata, 
        blockneighbour's data, layer's parameters (state_dict)
        also compute a priv-pub key pair and the AES key
        """
        self.metadata = metadata
        self.neighbours = neighbours
        self.params = params
        self.nonce, self.hash = self.compute_hash()
        self.AES = self.generate_key()
        self.priv = self.generate_key_pair()
        self.pub = self.priv.public_key()

    def __repr__(self) -> str:
        return f"""
        Block(
            {base64.encodebytes(self.hash)}, 
            {self.neighbours.prev_pub_key}, 
            {self.neighbours.next_pub_key},
            {base64.encodebytes(self.neighbours.hash)}
        )"""

    def compute_hash(self) -> Tuple[int, bytes]:
        """
        Computes the hash of the current block
        using a Proof of Work algorithm
        """
        pow = ProofOfWork(ProofOfWork.TARGET_BITS, self)
        return pow.run()

    def generate_key(self) -> bytes:
        """
        Generate a 256 bit AES key
        """
        return os.urandom(32)

    def generate_key_pair(self) -> rsa.RSAPrivateKey:
        """ 
        generates the public-private key pair for the 
        current block
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048)
        return private_key


class Sentinel(Block):
    def __init__(
            self,
            metadata: BlockMetadata,
            neighbours: Neighbours,
            params: "Layer"
    ) -> None:
        super().__init__(metadata, neighbours, params)

    def __repr__(self) -> str:
        return f"""
        Sentinel(
            {base64.encodebytes(self.hash)}, 
            {self.neighbours.prev_pub_key}, 
            {self.neighbours.next_pub_key},
            {base64.encodebytes(self.neighbours.hash)}
        )"""
