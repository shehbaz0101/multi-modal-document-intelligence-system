"""Tests for RRF fusion — the IR-math piece most likely to regress silently."""
from __future__ import annotations

from indexing.base import IndexHit
from retrieval.fusion import reciprocal_rank_fusion


def _hit(doc: str, block: str, score: float, source: str) -> IndexHit:
    return IndexHit(
        doc_id=doc,
        page_number=1,
        block_id=block,
        score=score,
        source_index=source,
        payload={"doc_id": doc, "block_id": block, "modality": "text"},
    )


def test_single_list_preserves_order():
    """With one list, RRF is monotonic in the input rank."""
    hits = [
        _hit("d1", "b0", 0.9, "dense"),
        _hit("d1", "b1", 0.8, "dense"),
        _hit("d1", "b2", 0.7, "dense"),
    ]
    fused = reciprocal_rank_fusion({"dense": hits})
    assert [h.block_id for h in fused] == ["b0", "b1", "b2"]


def test_two_lists_agree_boosts_shared_result():
    """A result ranked highly by both retrievers should beat one ranked only by one."""
    dense = [_hit("d1", "shared", 0.9, "dense"), _hit("d1", "only_dense", 0.8, "dense")]
    bm25  = [_hit("d1", "shared", 10.0, "bm25"),  _hit("d1", "only_bm25", 9.0, "bm25")]

    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25})
    # 'shared' is rank 0 in both lists → highest fused score
    assert fused[0].block_id == "shared"


def test_weights_bias_fusion():
    """When the visual retriever is weighted up, its top hit wins ties."""
    dense  = [_hit("d1", "text_hit", 0.9, "dense")]
    visual = [_hit("d1", "figure_hit", 0.85, "visual")]

    # Equal weights → text wins by tiebreak on insertion order
    equal = reciprocal_rank_fusion({"dense": dense, "visual": visual})
    # With heavy visual weight, figure_hit wins
    biased = reciprocal_rank_fusion(
        {"dense": dense, "visual": visual},
        weights={"dense": 0.2, "visual": 0.8},
    )
    assert biased[0].block_id == "figure_hit"
    # Sanity: the equal-weight case has both results present
    assert {h.block_id for h in equal} == {"text_hit", "figure_hit"}


def test_raw_scores_preserved_for_observability():
    """Failure-mode UIs depend on per-retriever scores; they must survive fusion."""
    dense = [_hit("d1", "b0", 0.9, "dense")]
    bm25  = [_hit("d1", "b0", 5.5, "bm25")]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25})
    assert fused[0].payload["_raw_scores"] == {"dense": 0.9, "bm25": 5.5}
    assert set(fused[0].payload["_fusion_sources"]) == {"dense", "bm25"}


def test_empty_input_is_safe():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": []}) == []
