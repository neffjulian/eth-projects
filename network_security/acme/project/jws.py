import json
import math
import helper

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization

class RSAKey:
    def __init__(self) -> None:
        self.alg = "RS256" # Used for the header
        self.private_key = rsa.generate_private_key(
            public_exponent = 65537,
            key_size = 2048,
            backend = default_backend()
        )
        self.kid = None # Is used if the key has already been used

    def sign(self, bts: bytes):
        signed =  self.private_key.sign(
            data = bts,
            padding = padding.PKCS1v15(),
            algorithm = hashes.SHA256()
        )
        return signed

    def encode_pn(self, x: int):
        pn = helper.decode_utf_8(
                helper.encode_base_64(
                    x.to_bytes(
                        length = math.ceil(x.bit_length() / 8),
                        byteorder = "big"
                    )
                )
            )
        return pn

    def jwk(self) -> dict:
        public_key = self.private_key.public_key()
        n = public_key.public_numbers().n
        e = public_key.public_numbers().e
        jwk = {
            'kty': 'RSA',
            'n': self.encode_pn(n),
            'e': self.encode_pn(e)
        }
        return jwk

    def save_pem(self):
        pem = self.private_key.private_bytes(
            encoding = serialization.Encoding.PEM,
            format = serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm = serialization.NoEncryption()
        )
        return pem

    def load_pem(self, pem):
        cert = x509.load_pem_x509_certificate(
            data = pem,
            backend = default_backend()
        ).public_bytes(
            encoding = serialization.Encoding.DER
        )
        return cert

class JWS:
    def __init__(self, key: RSAKey) -> None:
        self.key = key

    def create(self, header: dict = {}, payload: dict = {}) -> str:
        if self.key.kid is None:
            header['jwk'] = self.key.jwk()
        else:
            header['kid'] = self.key.kid
        header = helper.encode(header)
        payload = helper.encode(payload)
        signature = helper.encode(self.key.sign(header + b'.' + payload))
        jws = json.dumps({
            'protected': helper.decode_utf_8(header),
            'payload': helper.decode_utf_8(payload),
            'signature': helper.decode_utf_8(signature)
        })
        return jws