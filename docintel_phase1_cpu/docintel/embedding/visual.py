"""
Track B — Visual page embeddings via the ColPali family.

**This module is a Phase 2 placeholder.** Phase 1 runs text-only and this
file is not imported on the hot path. We define the interface now so the
retrieval router and index schemas can already carve out their place for
visual hits — which means the Phase 2 upgrade is additive, not a rewrite.

Recommended models (ViDoRe leaderboard, current as of 2026):
  * vidore/colqwen2-v1.0  — best accuracy, ~2B params, needs a GPU
  * vidore/colpali-v1.2   — original, still competitive
  * vidore/colflor-v1.0   — ~174M params, ~5x faster, 96% of ColQwen2 acc.
                            Default this above ~100k pages.

Each ColPali-style model produces a multi-vector embedding per page
(~1030 vectors at 448x448 patch size). Late-interaction MaxSim at query
time. Your vector DB must support multi-vector indexing (Qdrant 1.10+,
Vespa, Milvus 2.4+, Weaviate 1.26+).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class VisualPageEmbedder(ABC):
    model_name: str
    dim: int                          # per-patch dim (e.g. 128 for ColQwen2)
    patches_per_page: int             # approx; ColQwen2 ~= 1030

    @abstractmethod
    def embed_pages(self, image_uris: list[str]) -> list[np.ndarray]:
        """Return one (n_patches, dim) matrix per page."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Return a (n_query_tokens, dim) matrix."""


def build_visual_embedder(model_name: str) -> VisualPageEmbedder:
    raise NotImplementedError(
        "Visual embedding is Phase 2. See docs/phase2_colpali.md for the "
        "integration plan. Until then, set EMBEDDING_VISUAL_ENABLED=false."
    )
