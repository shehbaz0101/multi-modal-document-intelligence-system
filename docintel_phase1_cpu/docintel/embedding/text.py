"""
CPU-optimised text embedder.

On a GPU machine: FlagEmbedding + BAAI/bge-m3 (1024-dim) is the default.
On a CPU machine (Dell 5310, i5-10th gen, 16GB RAM): we swap to
sentence-transformers + ONNX runtime, which gives ~5-10x the throughput
of native PyTorch on CPU.

Recommended models by speed/quality on CPU:
┌──────────────────────────────┬───────┬──────────────┬───────────────────┐
│ Model                        │  Dim  │ Size on disk │ Speed (i5 CPU)    │
├──────────────────────────────┼───────┼──────────────┼───────────────────┤
│ all-MiniLM-L6-v2             │  384  │ ~22 MB       │ ~1500 sent/sec    │
│ BAAI/bge-small-en-v1.5       │  384  │ ~33 MB       │ ~1200 sent/sec    │
│ BAAI/bge-base-en-v1.5        │  768  │ ~110 MB      │ ~350 sent/sec     │
│ BAAI/bge-m3  (multilingual)  │ 1024  │ ~570 MB      │ ~40 sent/sec  ⚠   │
└──────────────────────────────┴───────┴──────────────┴───────────────────┘

Default here is all-MiniLM-L6-v2 via ONNX. Swap to bge-small for better
quality with an almost identical CPU footprint.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from app.schema import Block, BlockType

log = logging.getLogger(__name__)


class TextEmbedder(ABC):
    model_name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (len(texts), dim) L2-normalised float32 array."""


def block_to_embedding_text(block: Block) -> str:
    """
    Canonical rule for what text we embed per block.
    Centralised here so ingest-time and query-time always agree.
    """
    if block.type == BlockType.FIGURE:
        return (block.figure_caption or "").strip()
    if block.type == BlockType.TABLE:
        return (block.table_markdown or block.text or "").strip()
    if block.type == BlockType.HEADING:
        return f"[heading] {(block.text or '').strip()}"
    return (block.text or "").strip()


class SentenceTransformerEmbedder(TextEmbedder):
    """
    sentence-transformers + ONNX runtime.
    Works well on CPU — all-MiniLM-L6-v2 does ~1500 sentences/sec on an i5.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        use_onnx: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size

        if use_onnx:
            try:
                self._model = SentenceTransformer(
                    model_name,
                    backend="onnx",
                    model_kwargs={"file_name": "model_optimized.onnx"},
                )
                log.info("embedder loaded with ONNX backend: %s", model_name)
            except Exception:
                self._model = SentenceTransformer(model_name)
                log.info("embedder: ONNX unavailable, using PyTorch: %s", model_name)
        else:
            self._model = SentenceTransformer(model_name)

        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


def build_text_embedder(model_name: str) -> TextEmbedder:
    return SentenceTransformerEmbedder(model_name=model_name)
