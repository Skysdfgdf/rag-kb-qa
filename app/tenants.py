"""多租户：每台设备 / 每位用户一套互相隔离的资料库。

为什么需要
    早期版本全实例共用一个库，任何人上传都进同一个资料库。
    现在按 tenant_id 物理隔离：各自独立的 documents.json、uploads/ 目录，
    以及各自独立的 Chroma collection（collection 共享同一个 data/chroma
    持久化目录，只是名字不同，互不可见）。

目录布局
    data/tenants/index.json                租户索引（列表 / 首次创建 / 最后活跃）
    data/tenants/<tenant_id>/documents.json
    data/tenants/<tenant_id>/uploads/

tenant_id 规则
    只允许 [A-Za-z0-9_-]{4,64}，拒绝路径分隔符与 ".."，杜绝目录穿越。
    主设备（管理员）固定为 settings.default_tenant（默认 "admin"）。
    游客设备由前端生成 g_<32 位随机串>，猜不到，等同于该设备的私有凭证。
"""
import json
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings

# 只允许安全字符；长度 ≥4 避免 ".."、"a/" 这类畸形值
_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

_LOCK = threading.Lock()
_index_cache: Optional[Dict] = None
_last_touch: Dict[str, float] = {}
_TOUCH_INTERVAL = 60  # 秒：同一个租户 60s 内只更新一次活跃时间，避免每个请求都写盘


# ---------- 基础判定 ----------
def is_valid(tenant: str) -> bool:
    return bool(tenant) and bool(_TENANT_RE.match(tenant))


def normalize(tenant: Optional[str]) -> str:
    """空值回落到主设备租户。"""
    t = (tenant or "").strip()
    return t or settings.default_tenant


def is_default(tenant: str) -> bool:
    return tenant == settings.default_tenant


# ---------- 路径 ----------
def tenants_root() -> Path:
    return Path(settings.tenants_dir)


def tenant_dir(tenant: str) -> Path:
    return tenants_root() / tenant


def uploads_dir(tenant: str) -> Path:
    return tenant_dir(tenant) / "uploads"


def registry_file(tenant: str) -> Path:
    return tenant_dir(tenant) / "documents.json"


def exists(tenant: str) -> bool:
    return tenant_dir(tenant).is_dir()


# ---------- 索引 ----------
def _index_path() -> Path:
    return tenants_root() / "index.json"


def _load_index() -> Dict:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    p = _index_path()
    if not p.exists():
        _index_cache = {}
        return _index_cache
    try:
        _index_cache = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(_index_cache, dict):
            _index_cache = {}
    except Exception:  # noqa: BLE001 索引损坏不该拖垮服务
        _index_cache = {}
    return _index_cache


def _save_index(data: Dict) -> None:
    global _index_cache
    _index_cache = data
    root = tenants_root()
    root.mkdir(parents=True, exist_ok=True)
    tmp = _index_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_index_path())  # 原子替换，避免读到写了一半的文件


def touch(tenant: str) -> None:
    """登记租户并更新最后活跃时间（节流）。"""
    now = time.time()
    if now - _last_touch.get(tenant, 0) < _TOUCH_INTERVAL:
        return
    _last_touch[tenant] = now
    with _LOCK:
        idx = dict(_load_index())
        item = dict(idx.get(tenant) or {})
        stamp = datetime.now().isoformat(timespec="seconds")
        item.setdefault("created_at", stamp)
        item["last_seen"] = stamp
        item["kind"] = "admin" if is_default(tenant) else "guest"
        idx[tenant] = item
        _save_index(idx)


def ensure(tenant: str) -> None:
    """创建租户目录（只在上传等写操作时调用，避免被扫描器刷出大量空目录）。"""
    d = tenant_dir(tenant)
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    reg = d / "documents.json"
    if not reg.exists():
        reg.write_text("{}", encoding="utf-8")
    touch(tenant)


def doc_count(tenant: str) -> int:
    p = registry_file(tenant)
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, dict) else 0
    except Exception:  # noqa: BLE001
        return 0


def list_tenants() -> List[Dict]:
    """所有仍在磁盘上的租户（供管理员总览）。"""
    idx = _load_index()
    out: List[Dict] = []
    for t in sorted(idx.keys()):
        if not is_valid(t) or not exists(t):
            continue
        meta = idx.get(t) or {}
        out.append(
            {
                "tenant": t,
                "kind": "admin" if is_default(t) else meta.get("kind", "guest"),
                "created_at": meta.get("created_at", ""),
                "last_seen": meta.get("last_seen", ""),
                "doc_count": doc_count(t),
            }
        )
    # 索引里没登记但磁盘上存在的目录也补上，避免“有库却看不到”
    for d in sorted(tenants_root().glob("*")) if tenants_root().exists() else []:
        if d.is_dir() and is_valid(d.name) and d.name not in idx:
            out.append(
                {
                    "tenant": d.name,
                    "kind": "admin" if is_default(d.name) else "guest",
                    "created_at": "",
                    "last_seen": "",
                    "doc_count": doc_count(d.name),
                }
            )
    out.sort(key=lambda x: (x["kind"] != "admin", x["last_seen"] or ""), reverse=False)
    return out


def register_default() -> None:
    """确保主设备租户存在并登记（启动时调用）。"""
    d = tenant_dir(settings.default_tenant)
    (d / "uploads").mkdir(parents=True, exist_ok=True)
    reg = d / "documents.json"
    if not reg.exists():
        reg.write_text("{}", encoding="utf-8")
    _last_touch.pop(settings.default_tenant, None)
    touch(settings.default_tenant)


# ---------- 删除整个租户 ----------
def remove(tenant: str) -> bool:
    """删除租户目录与索引记录（向量集合由调用方另行清理）。"""
    if is_default(tenant):
        raise ValueError("主设备资料库不允许删除")
    d = tenant_dir(tenant)
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    with _LOCK:
        idx = dict(_load_index())
        idx.pop(tenant, None)
        _save_index(idx)
    _last_touch.pop(tenant, None)
    return True


# ---------- 老版本数据迁移 ----------
def migrate_legacy() -> Optional[str]:
    """把老版本的单库数据搬到 data/tenants/<default>/，幂等。

    老版本把清单放在 data/documents.json、原件放在 data/uploads/，
    现在要变成 data/tenants/admin/ 下的私有目录。
    注意：只有在目标清单「不存在或为空」时才搬，重复启动不会二次迁移。
    """
    src_reg = Path(settings.documents_registry)
    src_up = Path(settings.uploads_dir)
    dst = tenant_dir(settings.default_tenant)
    dst_reg = dst / "documents.json"
    dst_up = dst / "uploads"

    msg: Optional[str] = None
    existing: Dict = {}
    if dst_reg.exists():
        try:
            existing = json.loads(dst_reg.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            existing = {}

    if src_reg.exists() and not existing:
        dst.mkdir(parents=True, exist_ok=True)
        dst_up.mkdir(parents=True, exist_ok=True)
        moved = 0
        if src_up.is_dir():
            for f in src_up.iterdir():
                if f.is_file():
                    target = dst_up / f.name
                    if not target.exists():
                        shutil.move(str(f), str(target))
                    moved += 1
        shutil.move(str(src_reg), str(dst_reg))
        try:
            src_up.rmdir()  # 只在已空时成功
        except OSError:
            pass
        msg = f"已将旧版单库迁移为租户 {settings.default_tenant}：原文 {moved} 个文件"

    register_default()
    return msg
