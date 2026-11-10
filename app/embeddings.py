"""Embedding 抽象与三种实现。

- FastEmbedEmbedder：本地 onnx 推理，无需 torch，默认使用 BGE 中文小模型。
- OpenAIEmbedder：走 OpenAI 兼容接口（也可用于任意兼容服务）。
- OfflineEmbedder：本地哈希向量（字符 n-gram 词袋 + L2 归一），无需联网下载，
  作为离线/演示模式使用，能正确处理字面命中类检索。

设计要点：检索时 query 与 passage 使用同一模型，但 BGE 系列要求 query 端
拼接检索指令前缀，否则召回质量明显下降——这正是手写链路比“黑盒调用”更能
体现理解的地方。
"""
import hashlib
import math
from functools import lru_cache
from typing import List

from .config import settings

# BGE 官方推荐的 query 前缀（中英文 BGE v1.5 均使用该英文指令）
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.is_bge = "bge" in model_name.lower()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # fastembed 的 embed 返回 numpy 数组的迭代器
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        q = (_BGE_QUERY_PREFIX + text) if self.is_bge else text
        return list(self.model.embed([q]))[0].tolist()


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class OfflineEmbedder(Embedder):
    """本地哈希向量：中文 jieba 分词 + 字符 bigram 词袋，L2 归一。

    无需联网、无需下载，适合离线演示与测试整条 RAG 链路。
    中文部分走 jieba 分词（本地词典，无网络依赖），英文/数字走正则切词，
    再叠加汉字 bigram 作为兜底，使“关键词命中”类检索明显优于纯字符级。
    它仍不具备语义泛化能力（同义改写可能召不回），但足以跑通并体现检索效果。
    """

    def __init__(self, dim: int = 512):
        import re

        self.dim = dim
        self._re = re
        try:
            import jieba

            jieba.setLogLevel(20)
            self._jieba = jieba
        except Exception:  # noqa: BLE001
            self._jieba = None

    def _tokens(self, text: str) -> List[str]:
        text = (text or "").lower()
        re_ = self._re
        toks: List[str] = []
        # 英文 / 数字词
        toks += re_.findall(r"[a-z0-9_]+", text)
        # 中文分词
        if self._jieba is not None:
            for w in self._jieba.cut(text):
                w = w.strip()
                if not w:
                    continue
                if re_.match(r"[\u4e00-\u9fff]", w):
                    toks.append(w)
                elif len(w) > 1 and re_.match(r"[a-z0-9]", w):
                    toks.append(w)
        else:
            toks += list(text)
        # 汉字片段叠加 bigram 兜底，增强细微匹配
        for seg in re_.findall(r"[\u4e00-\u9fff]+", text):
            toks += [seg[i : i + 2] for i in range(len(seg) - 1)]
        return toks

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for t in self._tokens(text):
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """进程内单例：避免重复加载模型。

    默认使用 fastembed（BGE 中文模型）。若模型加载失败（例如远端沙箱无法
    访问 HuggingFace 下载权重），自动降级为离线哈希向量，保证服务始终可用——
    检索仍能命中字面相同的内容，只是不具备语义泛化能力。
    """
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            settings.openai_api_key, settings.openai_base_url, settings.openai_embed_model
        )
    if settings.embedding_provider == "offline":
        return OfflineEmbedder(dim=settings.offline_embed_dim)
    try:
        return FastEmbedEmbedder(settings.embedding_model)
    except Exception as ex:  # noqa: BLE001
        import sys

        print(
            f"[embeddings] 警告：fastembed 模型加载失败，已降级为离线哈希向量：{ex}",
            file=sys.stderr,
        )
        return OfflineEmbedder(dim=settings.offline_embed_dim)
