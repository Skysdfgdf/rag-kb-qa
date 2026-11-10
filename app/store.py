"""文档清单（注册表），按租户隔离。

向量库本身不擅长“按文档列出/统计”，因此每个租户维护一个轻量 JSON：
data/tenants/<tenant_id>/documents.json，记录 doc_id、文件名、切片数、上传时间。
删除文档时同时清理注册表、向量库以及 uploads/ 里的原始文件。
"""
import json
from datetime import datetime
from typing import Dict, List, Optional

from . import tenants


def _load(tenant: str) -> Dict:
    p = tenants.registry_file(tenant)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 清单损坏按空库处理，不让整个服务挂掉
        return {}


def _save(tenant: str, data: Dict) -> None:
    p = tenants.registry_file(tenant)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)  # 原子替换


def add_document(tenant: str, doc_id: str, name: str, chunk_count: int) -> Dict:
    data = _load(tenant)
    data[doc_id] = {
        "doc_id": doc_id,
        "name": name,
        "chunk_count": chunk_count,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save(tenant, data)
    return data[doc_id]


def list_documents(tenant: str) -> List[Dict]:
    docs = list(_load(tenant).values())
    docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return docs


def get_document(tenant: str, doc_id: str) -> Optional[Dict]:
    return _load(tenant).get(doc_id)


def remove_document(tenant: str, doc_id: str) -> bool:
    data = _load(tenant)
    if doc_id in data:
        del data[doc_id]
        _save(tenant, data)
        return True
    return False
