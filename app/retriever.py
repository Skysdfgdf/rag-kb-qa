"""检索与重排。

链路：
  1. 向量召回 top-N 候选（语义匹配）。
  2. 在候选集内做 BM25 词面匹配，再用 Reciprocal Rank Fusion (RRF)
     把“语义排名”和“词面排名”融合，得到兼顾两者的排序。
  3. 截取前 top_k 作为喂给大模型的上下文。

RRF 公式：score = Σ  1 / (k + rank)，k 通常取 60。
相比直接用向量距离，融合词面信号能显著提升包含专有名词 /
数字 / 罕见词的查询效果——这是纯向量检索容易翻车的地方。
"""
import re
from typing import Dict, List, Optional

from .config import settings
from .vectorstore import VectorStore

_K_RRF = 60


def _tokenize(text: str) -> List[str]:
    """英文/数字按词，中文按字（BM25 在候选集上做轻量词面匹配，足以补足信号）。"""
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


def _reciprocal_rank_fusion(rankings: List[List[str]], k: int = _K_RRF) -> List[str]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class Retriever:
    def __init__(self, vs: Optional[VectorStore] = None):
        self.vs = vs or VectorStore()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        doc_filter: Optional[str] = None,
    ) -> List[Dict]:
        top_k = top_k or settings.top_k
        where = {"doc_id": doc_filter} if doc_filter else None

        res = self.vs.query(question, n_results=settings.vector_top_k, where=where)
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        if not ids:
            return []

        # 向量召回本身已按距离升序排好
        vector_rank = list(ids)
        rankings = [vector_rank]

        # 词面重排（可选，且依赖 rank_bm25；缺失则跳过，不影响主流程）
        if settings.use_rerank and len(ids) > 1:
            try:
                from rank_bm25 import BM25Okapi

                tokenized = [_tokenize(d) for d in docs]
                bm25 = BM25Okapi(tokenized)
                scores = bm25.get_scores(_tokenize(question))
                bm25_order = [ids[i] for i in sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)]
                rankings.append(bm25_order)
            except Exception:
                pass

        fused_ids = _reciprocal_rank_fusion(rankings)
        by_id = {i: (d, m, dist) for i, d, m, dist in zip(ids, docs, metas, dists)}

        results: List[Dict] = []
        for fid in fused_ids[:top_k]:
            d, m, dist = by_id[fid]
            results.append({"id": fid, "text": d, "metadata": m, "distance": dist})
        return results
