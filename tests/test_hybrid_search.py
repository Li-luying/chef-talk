"""Tests for BM25 index and RRF fusion modules."""

import pytest

from gustobot.infrastructure.knowledge.bm25_index import BM25Index
from gustobot.infrastructure.knowledge.hybrid_retriever import rrf_fusion


class TestBM25Index:
    def test_build_and_search(self, tmp_path):
        idx = BM25Index(index_path=str(tmp_path / "test_bm25.pkl"))
        idx.build(
            documents=[
                "鱼香肉丝的做法是猪肉切丝",
                "麻婆豆腐需要豆腐和肉末",
                "红烧肉要用五花肉和冰糖",
            ],
            doc_ids=["d1", "d2", "d3"],
        )
        results = idx.search("鱼香肉丝", top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "d1"
        assert results[0]["bm25_score"] > 0

    def test_empty_corpus(self, tmp_path):
        idx = BM25Index(index_path=str(tmp_path / "empty.pkl"))
        idx.build([], [])
        assert idx.search("测试", top_k=5) == []

    def test_add_and_persist(self, tmp_path):
        path = str(tmp_path / "persist.pkl")
        idx = BM25Index(index_path=path)
        idx.build(["鱼香肉丝的做法"], ["d1"])
        idx.add("麻婆豆腐的做法", "d2")
        assert idx.get_stats()["document_count"] == 2

        idx.save()
        idx2 = BM25Index(index_path=path)
        assert idx2.load() is True
        assert idx2.get_stats()["document_count"] == 2
        results = idx2.search("麻婆豆腐", top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "d2"

    def test_delete(self, tmp_path):
        idx = BM25Index(index_path=str(tmp_path / "delete.pkl"))
        idx.build(["文档甲", "文档乙", "文档丙"], ["a", "b", "c"])
        idx.delete(["b"])
        assert idx.get_stats()["document_count"] == 2
        results = idx.search("文档", top_k=5)
        ids = {r["id"] for r in results}
        assert "b" not in ids

    def test_clear(self, tmp_path):
        idx = BM25Index(index_path=str(tmp_path / "clear.pkl"))
        idx.build(["文档甲"], ["a"])
        idx.clear()
        assert idx.get_stats()["document_count"] == 0

    def test_rebuild_from_corpus(self, tmp_path):
        idx = BM25Index(index_path=str(tmp_path / "rebuild.pkl"))
        idx.build(["鱼香肉丝做法"], ["d1"])
        idx.add("麻婆豆腐做法", "d2")
        idx.add("红烧肉做法", "d3")
        idx.rebuild_from_corpus()
        assert idx.get_stats()["document_count"] == 3
        results = idx.search("麻婆豆腐", top_k=2)
        assert any(r["id"] == "d2" for r in results)

    def test_missing_pickle(self, tmp_path):
        idx = BM25Index(index_path=str(tmp_path / "nonexistent.pkl"))
        assert idx.load() is False

    def test_chinese_multi_word_query(self, tmp_path):
        idx = BM25Index(index_path=str(tmp_path / "chinese.pkl"))
        idx.build(
            ["宫保鸡丁需要鸡肉花生", "水煮鱼需要草鱼豆芽", "回锅肉需要五花肉蒜苗"],
            ["a", "b", "c"],
        )
        results = idx.search("水煮鱼的做法", top_k=3)
        assert len(results) > 0
        assert results[0]["id"] == "b"


class TestRRFFusion:
    def test_basic_fusion(self):
        dense = [
            {"id": "a", "content": "doc a", "score": 0.9},
            {"id": "b", "content": "doc b", "score": 0.8},
        ]
        sparse = [
            {"id": "b", "content": "doc b", "bm25_score": 10.0},
            {"id": "c", "content": "doc c", "bm25_score": 5.0},
        ]
        fused = rrf_fusion(dense, sparse, k=60, top_k=3)
        assert len(fused) == 3
        assert fused[0]["id"] == "b"
        assert "hybrid_score" in fused[0]

    def test_only_dense(self):
        dense = [{"id": "a", "score": 0.9}]
        fused = rrf_fusion(dense, [], top_k=5)
        assert len(fused) == 1

    def test_only_sparse(self):
        sparse = [{"id": "a", "bm25_score": 5.0}]
        fused = rrf_fusion([], sparse, top_k=5)
        assert len(fused) == 1

    def test_both_empty(self):
        assert rrf_fusion([], []) == []

    def test_top_k_limit(self):
        dense = [{"id": f"d{i}", "score": 1.0 - i * 0.1} for i in range(10)]
        sparse = [{"id": f"s{i}", "bm25_score": 10.0 - i} for i in range(10)]
        fused = rrf_fusion(dense, sparse, k=60, top_k=3)
        assert len(fused) == 3

    def test_with_weights(self):
        dense = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
        sparse = [{"id": "c", "bm25_score": 10.0}]
        fused = rrf_fusion(dense, sparse, k=60, dense_weight=2.0, sparse_weight=1.0, top_k=3)
        assert len(fused) == 3
        assert fused[0]["id"] == "a"
