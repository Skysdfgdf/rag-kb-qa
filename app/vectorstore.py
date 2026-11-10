"""Chroma 向量库的轻封装（按租户隔离）。

每个租户一个独立 collection：
  - 主设备（默认租户）沿用 settings.collection_name（默认 kb_default），
    这样升级到多租户后老数据无需重新 embedding；
  - 其他租户使用 kb_<tenant_id>，超过 Chroma 的 63 字符上限时改用哈希名。

多个 collection 共享同一个 data/chroma 持久化目录，但彼此不可见。
"""
import hashlib
import uuid
from typing import Dict, List, Optional

import chromadb

from . import tenants
from .config import settings
from .embeddings import get_embedder

_client = None


def get_client():
    """全局共享一个 PersistentClient（重复创建会互相抢锁并告警）。"""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client


def collection_name(tenant: str) -> str:
    tenant = tenants.normalize(tenant)
    if tenants.is_default(tenant):
        return settings.collection_name
    name = "kb_" + tenant
    if len(name) > 63:  # Chroma 限制：3~63 字符
        name = "kb_" + hashlib.sha1(tenant.encode("utf-8")).hexdigest()[:32]
    return name


def drop_collection(tenant: str) -> bool:
    """按租户名直接删 collection。

    注意：不能在“缓存里没这个租户”时就跳过——那样只删了目录、向量数据
    会永久留在磁盘上（服务重启后管理员清理游客库就会踩到）。
    """
    try:
        get_client().delete_collection(collection_name(tenant))
        return True
    except Exception:  # noqa: BLE001 集合本就不存在
        return False


class VectorStore:
    def __init__(self, tenant: Optional[str] = None):
        self.tenant = tenants.normalize(tenant)
        self.name = collection_name(self.tenant)
        self.client = get_client()
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": "cosine"},  # 余弦距离，值越小越相似
        )
        self.embedder = get_embedder()

    def add(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not texts:
            return []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedder.embed_documents(texts)
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return ids

    def query(
        self,
        text: str,
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> Dict:
        q_emb = self.embedder.embed_query(text)
        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def delete_by_doc(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.collection.count()

    def drop(self) -> None:
        """删除整个 collection（管理员清理游客库时使用）。"""
        try:
            self.client.delete_collection(self.name)
        except Exception:  # noqa: BLE001 集合本就不存在时忽略
            pass
