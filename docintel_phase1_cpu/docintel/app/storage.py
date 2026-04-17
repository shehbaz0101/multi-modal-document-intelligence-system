"""
Storage abstraction. Local filesystem for dev, S3/GCS/Azure for prod.

Everything that produces or consumes a file (pages, figures, originals)
goes through this interface. Never touch open()/boto3 directly elsewhere.
"""
from __future__ import annotations

import hashlib
import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> str:
        """Store bytes under a key; return a URI addressing them."""

    @abstractmethod
    def get(self, uri: str) -> bytes: ...

    @abstractmethod
    def exists(self, uri: str) -> bool: ...

    @staticmethod
    def content_hash(data: bytes) -> str:
        """Canonical doc_id: sha256 of the bytes."""
        return hashlib.sha256(data).hexdigest()


class LocalStorage(Storage):
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Key is a relative path under root; no traversal allowed.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"Path traversal blocked: {key}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put(self, key: str, data: bytes) -> str:
        p = self._path(key)
        p.write_bytes(data)
        return f"file://{p}"

    def get(self, uri: str) -> bytes:
        path = uri.removeprefix("file://")
        return Path(path).read_bytes()

    def exists(self, uri: str) -> bool:
        path = uri.removeprefix("file://")
        return Path(path).exists()


class S3Storage(Storage):
    """Production backend. Left as a sketch — trivial to complete with boto3."""

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        # import boto3; self.s3 = boto3.client("s3")

    def put(self, key: str, data: bytes) -> str:
        raise NotImplementedError("Complete with boto3 put_object; return s3://...")

    def get(self, uri: str) -> bytes:
        raise NotImplementedError

    def exists(self, uri: str) -> bool:
        raise NotImplementedError


def build_storage(backend: str, root: str) -> Storage:
    if backend == "local":
        return LocalStorage(root)
    if backend == "s3":
        # Parse s3://bucket/prefix from root
        assert root.startswith("s3://")
        path = root.removeprefix("s3://")
        bucket, _, prefix = path.partition("/")
        return S3Storage(bucket=bucket, prefix=prefix)
    raise ValueError(f"Unknown storage backend: {backend}")
