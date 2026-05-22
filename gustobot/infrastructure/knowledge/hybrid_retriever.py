"""Hybrid retrieval: RRF fusion of dense (ANN) and sparse (BM25) results."""

from typing import Any, Dict, List, Optional


def rrf_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fuse dense ANN and sparse BM25 results using Reciprocal Rank Fusion.

    Args:
        dense_results: Results from dense vector search (Milvus), each with 'id' and 'score'.
        sparse_results: Results from sparse BM25 search, each with 'id' and 'bm25_score'.
        k: RRF constant (default 60, per literature).
        dense_weight: Weight multiplier for dense ranks.
        sparse_weight: Weight multiplier for sparse ranks.
        top_k: Max results to return. If None, returns all fused results.

    Returns:
        Fused and sorted list of results with 'hybrid_score' attached.
    """
    if not dense_results and not sparse_results:
        return []

    if not sparse_results:
        return dense_results[:top_k] if top_k else dense_results

    if not dense_results:
        return sparse_results[:top_k] if top_k else sparse_results

    # Assign ranks (1-based) to each result set, sorted by score descending
    dense_ranked = sorted(
        dense_results, key=lambda r: r.get("score", 0.0), reverse=True
    )
    sparse_ranked = sorted(
        sparse_results, key=lambda r: r.get("bm25_score", 0.0), reverse=True
    )

    # RRF: for each document, accumulate weight / (k + rank)
    rrf_scores: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_ranked, start=1):
        doc_id = doc.get("id")
        if doc_id is None:
            continue
        rrf_scores[doc_id] = {
            "id": doc_id,
            "content": doc.get("content", ""),
            "metadata": doc.get("metadata", {}),
            "dense_score": doc.get("score", 0.0),
            "sparse_score": 0.0,
            "rrf_score": dense_weight / (k + rank),
        }

    for rank, doc in enumerate(sparse_ranked, start=1):
        doc_id = doc.get("id")
        if doc_id is None:
            continue
        if doc_id in rrf_scores:
            rrf_scores[doc_id]["rrf_score"] += sparse_weight / (k + rank)
            rrf_scores[doc_id]["sparse_score"] = doc.get("bm25_score", 0.0)
        else:
            rrf_scores[doc_id] = {
                "id": doc_id,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "dense_score": 0.0,
                "sparse_score": doc.get("bm25_score", 0.0),
                "rrf_score": sparse_weight / (k + rank),
            }

    # Sort by RRF score descending
    fused = sorted(rrf_scores.values(), key=lambda r: r["rrf_score"], reverse=True)

    # Attach hybrid_score and clean up
    for item in fused:
        item["hybrid_score"] = item.pop("rrf_score")

    return fused[:top_k] if top_k else fused
