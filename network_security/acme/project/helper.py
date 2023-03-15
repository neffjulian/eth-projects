import base64
import json

def encode_utf_8(msg: str) -> bytes:
    return msg.encode()

def decode_utf_8(bts: bytes) -> str:
    return bts.decode()

def encode_base_64(msg: str) -> bytes:
    return base64.urlsafe_b64encode(msg).rstrip(b"=")

def decode_base_64(bts:bytes) -> str:
    padding = 4 - (len(bts) % 4)
    return base64.urlsafe_b64encode(bts + b'=' * padding)

def encode(msg: dict | str) -> bytes:
    if isinstance(msg, dict):
        msg = json.dumps(msg)
    if isinstance(msg, str):
        msg = encode_utf_8(msg)
    return encode_base_64(msg)

def decode(bts: bytes) -> str:
    return decode_utf_8(bts)
