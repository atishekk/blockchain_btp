from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from blockchain.block import Block

import pickle
from cryptography.hazmat.primitives import hashes


class ProofOfWork:

    TARGET_BITS = 4
    MAX_NONCE = 2 ** 64

    def __init__(self, target_bits: int, block: "Block"):
        self.block = block
        self.target_bits = target_bits
        self._target = 1 << (256 - target_bits)

    def _prepare_data(self, nonce) -> bytes:
        """
        serialise the data to be hashed
        """
        b = self.block
        params = (b.neighbours.hash, b.metadata, self.target_bits, nonce)
        return pickle.dumps(params)

    def run(self) -> Tuple[int, bytes]:
        """
        computes the POW hash and the nonce value 
        for the block
        """
        nonce = 0
        h = bytes()
        while nonce < self.MAX_NONCE:
            # compute hash
            data = self._prepare_data(nonce)
            digest = hashes.Hash(hashes.SHA3_256())
            digest.update(data)
            h = digest.finalize()

            # if less than the target number then hash is valid
            if int.from_bytes(h, "big") < self._target:
                break
            else:
                nonce += 1
        return (nonce, h)

    def validate(self) -> bool:
        """
        validate that the block has a valid hash
        """
        if self.block.hash == bytes(bytearray(32)):
            return True

        # compute the hash using the block's nonce value
        data = self._prepare_data(self.block.nonce)
        digest = hashes.Hash(hashes.SHA3_256())
        digest.update(data)
        h = digest.finalize()

        # if the hash is greater than the target then hash is invalid
        if int.from_bytes(h, "big") > self._target:
            return False
        return True

    def compute(self) -> bytes:
        data = self._prepare_data(self.block.nonce)
        digest = hashes.Hash(hashes.SHA3_256())
        digest.update(data)
        h = digest.finalize()
        return h
