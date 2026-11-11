"""纯 NumPy 向量存储（按租户隔离）——替代 chromadb 的轻量实现。

背景：chromadb 1.x 的 Rust 原生绑定在 PyInstaller 打包后必然段错误
（多个版本实测均崩溃），因此自实现一个兼容接口的本地向量库：

- 每个租户一个目录：data/vecstore/<tenant>/ 下
    embeddings.npy  —— float32 向量矩阵（按行对齐 rows.jsonl）
    rows.jsonl      —— 每行一个 {"id", "text", "metadata"}
- 检索 = L2 归一化后矩阵点积（即余弦相似度），个人知识库规模
  （数千块）下单次检索毫秒级，足够快。
- distance 语义与 Chroma 一致：1 - cosine_similarity，越小越相似。
- 写入采用"临时文件 + 原子替换"，避免写一半崩溃留下坏数据。

对外接口与 chroma 版 VectorStore 完全一致（query 返回同构的
ids/documents/metadatas/distances 四层列表），上层代码零改动。
"""
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import tenants
from .config import settings
from .embeddings import get_embedder

_cache: Dict[str, "NumpyVectorStore"] = {}


def _store_dir(tenant: str) -> Path:
    return Path(settings.data_dir) / "vecstore" / tenants.normalize(tenant)


def drop_collection(tenant: str) -> bool:
    """按租户名删除整个向量目录。目录本就不存在时返回 False。"""
    d = _store_dir(tenant)
    existed = d.exists()
    if existed:
        for f in d.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass
    _cache.pop(tenants.normalize(tenant), None)
    return existed


class NumpyVectorStore:
    def __init__(self, tenant: Optional[str] = None):
        self.tenant = tenants.normalize(tenant)
        self.dir = _store_dir(self.tenant)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = get_embedder()
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) 已 L2 归一化
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        rows_path = self.dir / "rows.jsonl"
        emb_path = self.dir / "embeddings.npy"
        if not rows_path.exists() or not emb_path.exists():
            return
        self._ids, self._texts, self._metas = [], [], []
        with open(rows_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self._ids.append(r["id"])
                self._texts.append(r["text"])
                self._metas.append(r.get("metadata") or {})
        self._matrix = np.load(emb_path).astype(np.float32)
        if self._matrix.shape[0] != len(self._ids):  # 数据不一致时以行文件为准
            n = min(self._matrix.shape[0], len(self._ids))
            self._matrix = self._matrix[:n]
            self._ids, self._texts, self._metas = (
                self._ids[:n], self._texts[:n], self._metas[:n],
            )

    def _save(self) -> None:
        # np.save 会给不以 .npy 结尾的文件名自动补后缀，故临时文件须以 .npy 结尾
        tmp_emb = self.dir / "embeddings.tmp.npy"
        tmp_rows = self.dir / "rows.jsonl.tmp"
        np.save(tmp_emb, self._matrix.astype(np.float32))
        with open(tmp_rows, "w", encoding="utf-8") as f:
            for i, t, m in zip(self._ids, self._texts, self._metas):
                f.write(json.dumps({"id": i, "text": t, "metadata": m}, ensure_ascii=False) + "\n")
        os.replace(tmp_emb, self.dir / "embeddings.npy")
        os.replace(tmp_rows, self.dir / "rows.jsonl")

    # ---------- 对外接口（与 chroma 版一致） ----------
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
        embeddings = np.asarray(
            self.embedder.embed_documents(texts), dtype=np.float32
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        if self._matrix is None or self._matrix.size == 0:
            self._matrix = embeddings
        else:
            self._matrix = np.vstack([self._matrix, embeddings])
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metas.extend(metadatas)
        self._save()
        return ids

    def query(
        self,
        text: str,
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> Dict:
        n = len(self._ids)
        if n == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.asarray([self.embedder.embed_query(text)], dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        sims = (self._matrix @ q.T).ravel()  # (n,)

        mask = None
        if where and "doc_id" in where:
            mask = np.array(
                [m.get("doc_id") == where["doc_id"] for m in self._metas], dtype=bool
            )
            if not mask.any():
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            sims = np.where(mask, sims, -np.inf)

        k = min(n_results, n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        top = [int(i) for i in top if sims[i] > -np.inf]

        return {
            "ids": [[self._ids[i] for i in top]],
            "documents": [[self._texts[i] for i in top]],
            "metadatas": [[self._metas[i] for i in top]],
            "distances": [[float(1.0 - sims[i]) for i in top]],
        }

    def delete_by_doc(self, doc_id: str) -> None:
        keep = [i for i, m in enumerate(self._metas) if m.get("doc_id") != doc_id]
        if len(keep) == len(self._ids):
            return
        removed = len(self._ids) - len(keep)
        self._ids = [self._ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._metas = [self._metas[i] for i in keep]
        self._matrix = self._matrix[keep] if removed and self._matrix is not None else self._matrix
        if not self._ids:
            self._matrix = np.zeros((0, self._matrix.shape[1] if self._matrix is not None else 1), dtype=np.float32)
        self._save()

    def count(self) -> int:
        return len(self._ids)

    def drop(self) -> None:
        drop_collection(self.tenant)


def get_or_create(tenant: str) -> NumpyVectorStore:
    t = tenants.normalize(tenant)
    if t not in _cache:
        _cache[t] = NumpyVectorStore(t)
    return _cache[t]
