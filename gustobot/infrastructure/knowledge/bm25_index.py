"""BM25 sparse index with jieba tokenization for Chinese text."""

import math
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger


def _default_tokenizer(text: str) -> List[str]:
    """Tokenize text using jieba for Chinese. Falls back to whitespace split for other."""
    import jieba
    tokens = []
    for word in jieba.lcut(text):
        word = word.strip()
        if word:
            tokens.append(word)
    return tokens


class _BM25Scorer:
    """Lightweight BM25 scorer with standard IDF formula.

    Uses the standard BM25Okapi formula:
      score(q, d) = Σ idf(q_i) * tf(q_i, d) * (k1+1) / (tf(q_i, d) + k1 * (1 - b + b * |d|/avgdl))

    With the standard IDF:
      idf(q_i) = log(1 + (N - n(q_i) + 0.5) / (n(q_i) + 0.5))
    """

    def __init__(
        self,
        tokenized_corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.avgdl = 0.0
        self.doc_len: List[int] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        if self.corpus_size == 0:
            return

        nd: Dict[str, int] = {}
        total_terms = 0
        for document in tokenized_corpus:
            self.doc_len.append(len(document))
            total_terms += len(document)

            freq = {}
            for word in document:
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)

            for word in freq:
                nd[word] = nd.get(word, 0) + 1

            self.corpus_size += 1

        self.corpus_size = len(tokenized_corpus)
        self.avgdl = total_terms / self.corpus_size if self.corpus_size > 0 else 0.0

        # Standard BM25 IDF: log(1 + (N - n + 0.5) / (n + 0.5))
        for word, n in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5))

    def get_scores(self, query: List[str]) -> np.ndarray:
        """Compute BM25 scores for a query against all documents."""
        if self.corpus_size == 0:
            return np.array([])

        score = np.zeros(self.corpus_size)
        doc_len_arr = np.array(self.doc_len)
        for q in query:
            idf_val = self.idf.get(q, 0.0)
            if idf_val == 0.0:
                continue
            q_freq = np.array([doc.get(q, 0) for doc in self.doc_freqs])
            score += idf_val * (
                q_freq * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len_arr / self.avgdl))
            )
        return score


class BM25Index:
    """BM25 sparse retrieval index with pickle persistence.

    Builds a BM25 index from a corpus of documents for keyword-based retrieval.
    Supports incremental add/delete and full persistence to disk.
    """

    VERSION = 1

    def __init__(
        self,
        index_path: str = "./data/kb/bm25_index.pkl",
        tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.tokenizer = tokenizer_fn or _default_tokenizer

        self._corpus: List[str] = []
        self._doc_ids: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._bm25: Optional[_BM25Scorer] = None

        self._dirty_count = 0
        self._rebuild_threshold = 50

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------
    def build(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Build BM25 index from a list of document texts."""
        if not documents:
            self._corpus, self._doc_ids, self._metadatas = [], [], []
            self._bm25 = None
            self.save()
            return

        tokenized_corpus = [self.tokenizer(doc) for doc in documents]
        self._corpus = list(documents)
        self._doc_ids = list(doc_ids)
        self._metadatas = list(metadatas) if metadatas else [{} for _ in documents]
        self._bm25 = _BM25Scorer(tokenized_corpus)
        self._dirty_count = 0
        self.save()

    def rebuild_from_corpus(self) -> None:
        """Rebuild BM25 from current in-memory corpus."""
        if not self._corpus:
            return
        tokenized_corpus = [self.tokenizer(doc) for doc in self._corpus]
        self._bm25 = _BM25Scorer(tokenized_corpus)
        self._dirty_count = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist BM25 index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "corpus": self._corpus,
            "doc_ids": self._doc_ids,
            "metadatas": self._metadatas,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self) -> bool:
        """Load BM25 index from disk. Returns False if file not found."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("version") != self.VERSION:
                logger.warning("BM25 index version mismatch, rebuilding")
                return False

            self._corpus = payload["corpus"]
            self._doc_ids = payload["doc_ids"]
            self._metadatas = payload.get("metadatas", [{} for _ in self._corpus])

            tokenized_corpus = [self.tokenizer(doc) for doc in self._corpus]
            self._bm25 = _BM25Scorer(tokenized_corpus)
            self._dirty_count = 0
            logger.info("Loaded BM25 index with {} documents", len(self._corpus))
            return True
        except Exception as exc:
            logger.warning("Failed to load BM25 index: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add(
        self,
        document: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single document to the index."""
        self._corpus.append(document)
        self._doc_ids.append(doc_id)
        self._metadatas.append(metadata or {})
        self._dirty_count += 1
        self._rebuild_if_needed()

    def add_batch(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add multiple documents."""
        if not documents:
            return
        self._corpus.extend(documents)
        self._doc_ids.extend(doc_ids)
        self._metadatas.extend(metadatas or [{} for _ in documents])
        self._dirty_count += len(documents)
        self._rebuild_if_needed()

    def delete(self, doc_ids: List[str]) -> None:
        """Remove documents by doc_id."""
        id_set = set(doc_ids)
        before = len(self._corpus)
        remaining = [
            (d, i, m)
            for d, i, m in zip(self._corpus, self._doc_ids, self._metadatas)
            if i not in id_set
        ]
        self._corpus = [r[0] for r in remaining]
        self._doc_ids = [r[1] for r in remaining]
        self._metadatas = [r[2] for r in remaining]
        self._dirty_count += before - len(self._corpus)
        if self._dirty_count > 0:
            self.rebuild_from_corpus()
        self.save()

    def clear(self) -> None:
        """Clear all documents."""
        self._corpus.clear()
        self._doc_ids.clear()
        self._metadatas.clear()
        self._bm25 = None
        self._dirty_count = 0
        self.save()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search BM25 index. Returns list of dicts with keys: id, content, bm25_score, metadata."""
        if not self._bm25 or not self._corpus:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "id": self._doc_ids[idx],
                "content": self._corpus[idx],
                "bm25_score": float(scores[idx]),
                "metadata": dict(self._metadatas[idx]) if idx < len(self._metadatas) else {},
            })
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": len(self._corpus),
            "index_path": str(self.index_path),
            "index_exists": self.index_path.exists(),
            "bm25_loaded": self._bm25 is not None,
            "dirty_count": self._dirty_count,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _rebuild_if_needed(self) -> None:
        if self._dirty_count >= self._rebuild_threshold:
            self.rebuild_from_corpus()
            self.save()
