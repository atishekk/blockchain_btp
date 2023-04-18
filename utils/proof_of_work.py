from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blockchain.block import Block

import pickle
import sys
from cryptography.hazmat.primitives import hashes


class ProofOfWork:

    MAX_NONCE = sys.maxsize
    TARGET_BITS = 1

    def __init__(self, target_bits: int, block: "Block"):
        self.block = block
        self.target_bits = target_bits
        self._target = 1 << (256 - target_bits)

    def _prepare_data(self, nonce) -> bytes:
        """
        serialise the data to be hashed
        """
        b = self.block
        params = (b.params, b.neighbours.hash,
                  b.metadata, self.target_bits, nonce)
        return pickle.dumps(params)

    def run(self) -> tuple[int, bytes]:
        """
        computes the POW hash and the nonce value 
        for the block
        """
        nonce = 0
        h = bytes()
        while nonce < ProofOfWork.MAX_NONCE:
            data = self._prepare_data(nonce)
            digest = hashes.Hash(hashes.SHA3_256())
            digest.update(data)
            h = digest.finalize()

            if int.from_bytes(h, "big") < self._target:
                break
            else:
                nonce += 1
        return (nonce, h)

    def validate(self) -> bool:
        """
        validate that the block has a valid hash
        """
        data = self._prepare_data(self.block.nonce)
        digest = hashes.Hash(hashes.SHA3_256())
        digest.update(data)
        h = digest.finalize()

        if int.from_bytes(h, "big") > self._target:
            return False
        return True
