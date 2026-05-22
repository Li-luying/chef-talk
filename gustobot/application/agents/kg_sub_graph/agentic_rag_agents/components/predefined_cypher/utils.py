import re
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── 正则参数提取（纯规则，不调 LLM，稳）──

_FLAVOR_NAMES = [
    "麻辣", "香辣", "酸辣", "酸甜", "甜酸", "咸鲜", "酱香", "清淡",
    "孜然", "咖喱", "蒜香", "葱香", "椒盐", "黑椒", "鱼香", "怪味",
    "红油", "麻酱", "姜汁", "芥末", "陈皮", "五香", "烟熏", "糟香",
]


def _extract_params_by_rules(
    user_question: str, param_names: List[str]
) -> Dict[str, str]:
    """从用户问题中用正则提取参数，不依赖 LLM。"""
    params: Dict[str, str] = {}
    for name in param_names:
        value = None
        if name == "dish_name":
            # 匹配中文菜名：2-6个汉字的连续序列，优先匹配"XX菜"或"XX怎么做"前面的部分
            m = re.search(r"([一-鿿]{2,6})(?:的|这道|那道|菜|怎么做|如何做|做法|食材|配料|口味|工艺|步骤|是什么|有什么|介绍|推荐)", user_question)
            if m:
                value = m.group(1)
            else:
                # 兜底：取前2-5个连续汉字
                m = re.search(r"([一-鿿]{2,5})", user_question)
                if m:
                    value = m.group(1)
        elif name == "ingredient_name":
            m = re.search(r"([一-鿿]{2,4})(?:食材|材料|用料|原料)", user_question)
            if m:
                value = m.group(1)
            elif "食材" in user_question or "材料" in user_question or "用什么" in user_question:
                m = re.search(r"用\s*([一-鿿]{2,4})", user_question)
                if m:
                    value = m.group(1)
        elif name == "flavor_name":
            for fname in _FLAVOR_NAMES:
                if fname in user_question:
                    value = fname
                    break
        elif name == "method_name":
            m = re.search(r"([一-鿿]{1,3})(?:工艺|做法|烹饪|技法)", user_question)
            if m:
                value = m.group(1)
            elif "炒" in user_question: value = "炒"
            elif "蒸" in user_question: value = "蒸"
            elif "炖" in user_question: value = "炖"
            elif "煮" in user_question: value = "煮"
            elif "烤" in user_question: value = "烤"
            elif "炸" in user_question: value = "炸"
        elif name == "step_order":
            m = re.search(r"第\s*(\d+)\s*步", user_question)
            if m:
                value = m.group(1)
        elif name == "type_name":
            m = re.search(r"([一-鿿]{2,3})(?:类型|分类|大类)", user_question)
            if m:
                value = m.group(1)
        if value:
            params[name] = value.strip()
    return params


class VectorQueryMatcher:
    """使用 TF-IDF 向量化实现的查询匹配器。"""

    def __init__(
        self,
        predefined_cypher_dict: Dict[str, str],
        query_descriptions: Dict[str, str],
        similarity_threshold: float = 0.5,
    ) -> None:
        self.predefined_cypher_dict = predefined_cypher_dict
        self.query_descriptions = query_descriptions
        self.similarity_threshold = similarity_threshold

        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._query_vectors = self._compute_query_vectors()

    def _compute_query_vectors(self) -> Dict[str, np.ndarray]:
        keys: List[str] = []
        corpus: List[str] = []
        for query_name in self.predefined_cypher_dict:
            description = self.query_descriptions.get(query_name, "")
            keys.append(query_name)
            corpus.append(f"{query_name} {description}".strip())

        if not corpus:
            return {}

        matrix = self._vectorizer.fit_transform(corpus).toarray()
        return {
            key: np.asarray(vector, dtype=np.float32) for key, vector in zip(keys, matrix)
        }

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self._vectorizer.get_feature_names_out())))
        return self._vectorizer.transform(texts).toarray()

    def match_query(self, user_question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not user_question or not self._query_vectors:
            return []

        question_vector = self._embed([user_question])
        if question_vector.size == 0:
            return []
        question_vector = question_vector[0]

        similarities: List[tuple[str, float]] = []
        for query_name, vector in self._query_vectors.items():
            score = cosine_similarity([question_vector], [vector])[0][0]
            similarities.append((query_name, float(score)))

        similarities.sort(key=lambda item: item[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for query_name, score in similarities[:top_k]:
            if score >= self.similarity_threshold:
                results.append(
                    {
                        "query_name": query_name,
                        "similarity": score,
                        "cypher": self.predefined_cypher_dict[query_name],
                    }
                )
        return results

    def extract_parameters(
        self, user_question: str, query_name: str, llm: Any | None = None
    ) -> Dict[str, str]:
        """纯正则提取参数，不依赖 LLM，避免 JSON 解析失败。"""
        if query_name not in self.predefined_cypher_dict:
            return {}

        cypher_template = self.predefined_cypher_dict[query_name]
        param_names = re.findall(r"\$(\w+)", cypher_template)
        if not param_names:
            return {}

        return _extract_params_by_rules(user_question, param_names)


def create_vector_query_matcher(
    predefined_cypher_dict: Dict[str, str],
    query_descriptions: Optional[Dict[str, str]] = None,
) -> VectorQueryMatcher:
    descriptions = query_descriptions or {
        key: key.replace("_", " ") for key in predefined_cypher_dict.keys()
    }
    return VectorQueryMatcher(predefined_cypher_dict, descriptions)
