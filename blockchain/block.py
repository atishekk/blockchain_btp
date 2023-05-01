import time
import os
import pickle
from io import BytesIO
from typing import Tuple
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from utils.proof_of_work import ProofOfWork
from vm.request import Intermediate, QueryInput

import numpy as np
import torch


from typing import TYPE_CHECKING, cast
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
        self.prev_pub_key = cast(rsa.RSAPublicKey,
                                 serialization.load_pem_public_key(prev_pub_key))

    def set_next_pub_key(self, next_pub_key: bytes):
        self.next_pub_key = cast(
            rsa.RSAPublicKey, serialization.load_pem_public_key(next_pub_key))

    def __repr__(self) -> str:
        return f"""
            Neigh(
                prev_hash:  {self.hash.hex()}
            )
        """

    @classmethod
    def decode(cls, data: Tuple[bytes, bytes, bytes]) -> "Neighbours":
        n = Neighbours(data[0])
        n.set_prev_pub_key(data[1])
        n.set_next_pub_key(data[2])
        return n

    def encode(self) -> Tuple[bytes, bytes, bytes]:
        return (
            self.hash,
            self.prev_pub_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ),
            self.next_pub_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ),

        )


class Block:

    def __init__(
            self,
            metadata: BlockMetadata,
            neighbours: Neighbours,
            params: "Layer",
            AES: bytes,
            priv: rsa.RSAPrivateKey
    ) -> None:
        """
        Build a new block using the blockmetadata,
        blockneighbour's data, layer's parameters (state_dict)
        also compute a priv-pub key pair and the AES key
        """
        self.metadata = metadata
        self.neighbours = neighbours
        self.params = params
        self.AES = AES
        self.priv = priv
        self.pub = self.priv.public_key()

    def set_hash(self, nonce: int, hash: bytes):
        self.hash = hash
        self.nonce = nonce

    @classmethod
    def load(
        cls,
        metadata: BlockMetadata,
        neighbours: Neighbours,
        params: "Layer",
        hash: bytes,
        nonce: int,
        AES: bytes,
        priv: rsa.RSAPrivateKey
    ) -> "Block":
        b = Block(metadata, neighbours, params, AES, priv)
        b.set_hash(nonce, hash)
        pow = ProofOfWork(ProofOfWork.TARGET_BITS, b)
        if pow.validate():
            return b
        else:
            raise Exception("Invalid Hash")

    @classmethod
    def build(
        cls,
        metadata: BlockMetadata,
        neighbours: Neighbours,
        params: "Layer",
    ) -> "Block":
        AES = cls.generate_key()
        priv = cls.generate_key_pair()
        b = Block(metadata, neighbours, params, AES, priv)
        b.compute_hash()
        return b

    def __repr__(self) -> str:
        return f"""
        Block(
            hash:       {self.hash.hex()},
            prev_pub:   {self.neighbours.prev_pub_key},
            next_pub:   {self.neighbours.next_pub_key},
            prev_hash:  {self.neighbours.hash.hex()}
            nonce:      {self.nonce}
            metadata:   {self.metadata.timestamp}
            AES:        {self.AES.hex()}
            neigh:      {self.neighbours}
        )"""

    def compute_hash(self):
        """
        Computes the hash of the current block
        using a Proof of Work algorithm
        """
        pow = ProofOfWork(ProofOfWork.TARGET_BITS, self)
        self.nonce, self.hash = pow.run()

    @classmethod
    def generate_key(cls) -> bytes:
        """
        Generate a 256 bit AES key
        """
        return os.urandom(32)

    @classmethod
    def generate_key_pair(cls) -> rsa.RSAPrivateKey:
        """
        generates the public-private key pair for the
        current block
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048)
        return private_key

    def compute(self, data: Intermediate) -> Intermediate:
        # decrypt AES key
        AES = self.priv.decrypt(
            data.en_AES,
            padding.OAEP(
                padding.MGF1(hashes.SHA256()),
                hashes.SHA256(),
                None
            )
        )

        # decrypt the intermediate data
        cipher = Cipher(algorithms.AES(AES), modes.CBC(data.iv))
        decryptor = cipher.decryptor()
        de_comp = decryptor.update(data.comp) + decryptor.finalize()

        # convert the data to a torch tensor
        b = BytesIO(de_comp)
        array: np.ndarray = np.load(b)
        tensor = torch.from_numpy(array)

        # forward inference
        out_tensor: torch.Tensor = self.params.layer(tensor)

        # serialise the output
        out_array = out_tensor.detach().cpu().numpy()
        out_b = BytesIO()
        np.save(out_b, out_array)
        out_comp_bytes = out_b.getvalue()

        # encrypt the intermediate computation using the block's AES
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.AES), modes.CBC(iv))
        encryptor = cipher.encryptor()
        out_comp = encryptor.update(out_comp_bytes) + encryptor.finalize()

        # encrypt the block's AES
        out_AES = self.neighbours.next_pub_key.encrypt(
            self.AES,
            padding.OAEP(
                padding.MGF1(hashes.SHA256()),
                hashes.SHA256(),
                None
            )
        )

        # compute the blocks hash
        pow = ProofOfWork(ProofOfWork.TARGET_BITS, self)
        block_hash = pow.compute()

        # compute the digital signature
        res = Intermediate(block_hash, out_AES, out_comp, bytes(), iv)
        serialised = res.serialise()
        digital_sig = self.priv.sign(
            serialised,
            padding.PSS(
                padding.MGF1(hashes.SHA256()),
                padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        res.digital_sig = digital_sig
        return res

    def check_prev(self, data: Intermediate) -> bool:
        try:
            prev_pub = self.neighbours.prev_pub_key
            serialised = data.serialise()
            prev_pub.verify(
                data.digital_sig,
                serialised,
                padding.PSS(
                    padding.MGF1(hashes.SHA256()),
                    padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False


class Sentinel(Block):
    def __init__(
            self,
            metadata: BlockMetadata,
            neighbours: Neighbours,
            params: "Layer",
    ) -> None:

        self.metadata = metadata
        self.neighbours = neighbours
        self.params = params
        self.AES = Sentinel.generate_key()
        self.priv = Sentinel.generate_key_pair()
        self.pub = self.priv.public_key()
        self.set_hash(0, bytes(bytearray(32)))

    @ classmethod
    def load(cls, metadata: BlockMetadata, neighbours: Neighbours, params: "Layer", hash: bytes, nonce: int, AES: bytes, priv: rsa.RSAPrivateKey) -> "Sentinel":
        return cast(Sentinel, super().load(metadata, neighbours, params, hash, nonce, AES, priv))

    def __repr__(self) -> str:
        return f"""
        Sentinel(
            hash:       {self.hash.hex()},
            prev_pub:   {self.neighbours.prev_pub_key},
            next_pub:   {self.neighbours.next_pub_key},
            prev_hash:  {self.neighbours.hash.hex()}
            nonce:      {self.nonce}
            metadata:   {self.metadata.timestamp}
            AES:        {self.AES.hex()}
            neigh:      {self.neighbours}
        )"""

    def compute(self, data: QueryInput) -> Intermediate:
        # serialise the tensor
        out_array = data.data.detach().cpu().numpy()
        out_b = BytesIO()
        np.save(out_b, out_array)
        out_comp_bytes = out_b.getvalue()

        # encrypt the input data
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.AES), modes.CBC(iv))
        encryptor = cipher.encryptor()
        out_comp = encryptor.update(out_comp_bytes) + encryptor.finalize()

        out_AES = self.neighbours.next_pub_key.encrypt(
            self.AES,
            padding.OAEP(
                padding.MGF1(hashes.SHA256()),
                hashes.SHA256(),
                None
            )
        )

        # compute the blocks hash
        pow = ProofOfWork(ProofOfWork.TARGET_BITS, self)
        block_hash = pow.compute()

        # compute the digital signature
        res = Intermediate(block_hash, out_AES, out_comp, bytes(), iv)
        serialised = res.serialise()
        digital_sig = self.priv.sign(
            serialised,
            padding.PSS(
                padding.MGF1(hashes.SHA256()),
                padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        res.digital_sig = digital_sig
        return res
