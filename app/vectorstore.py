"""向量库统一入口（按租户隔离）。

通过 settings.vector_backend 选择底层实现：
  - numpy  （默认）纯 NumPy 余弦检索，零原生依赖，PyInstaller 打包安全
  - chroma         原 chromadb 实现（个人知识库场景下可回退使用）

两个后端对外接口完全一致：VectorStore(tenant).add/query/delete_by_doc/count/drop
以及模块级 drop_collection(tenant)。

每个租户一个独立 collection：
  - 主设备（默认租户）沿用 settings.collection_name（默认 kb_default）；
  - 其他租户使用 kb_<tenant_id>（chroma 后端；numpy 后端按租户目录隔离）。
"""
import hashlib
from typing import Dict, Optional

from . import tenants
from .config import settings


def collection_name(tenant: str) -> str:
    tenant = tenants.normalize(tenant)
    if tenants.is_default(tenant):
        return settings.collection_name
    name = "kb_" + tenant
    if len(name) > 63:  # Chroma 限制：3~63 字符
        name = "kb_" + hashlib.sha1(tenant.encode("utf-8")).hexdigest()[:32]
    return name


def drop_collection(tenant: str) -> bool:
    if settings.vector_backend == "numpy":
        from .np_backend import drop_collection as np_drop
        return np_drop(tenant)
    from .chroma_backend import drop_collection as chroma_drop
    return chroma_drop(tenant)


def _make(tenant: Optional[str]):
    if settings.vector_backend == "numpy":
        from .np_backend import get_or_create
        return get_or_create(tenants.normalize(tenant or settings.default_tenant))
    from .chroma_backend import ChromaVectorStore
    return ChromaVectorStore(tenant)


class VectorStore:
    """后端代理：按配置转发到 numpy / chroma 实现。"""

    def __init__(self, tenant: Optional[str] = None):
        self._impl = _make(tenant)

    def add(self, texts, metadatas, ids=None):
        return self._impl.add(texts, metadatas, ids)

    def query(self, text, n_results: int = 20, where: Optional[Dict] = None):
        return self._impl.query(text, n_results=n_results, where=where)

    def delete_by_doc(self, doc_id: str) -> None:
        self._impl.delete_by_doc(doc_id)

    def count(self) -> int:
        return self._impl.count()

    def drop(self) -> None:
        self._impl.drop()
