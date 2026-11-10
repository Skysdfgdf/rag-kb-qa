"""进程内限流：按资料库 / IP 计数。

为什么自己做
    本项目是单进程部署（本地与「发布为应用」都是一实例一进程），
    用 Redis 属于过度设计；标准库的 deque + 一把锁就够了，
    也不给部署平台多加一个依赖。将来真要多实例，把 _HITS 换成 Redis 即可，
    对外接口函数不用动。

算法
    滑动窗口：每个 key 保存最近若干次命中的时间戳，判断时先弹出窗口外的记录，
    再看剩余条数是否已到上限。相比「固定窗口计数器」，它不会在两个窗口的
    交界处一次性放过双倍流量。

内存
    每个 key 最多保留 limit 条时间戳；另外每 5 分钟清理一次整桶过期的 key，
    即使被扫描器刷出大量 key 也不会无限增长。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple

_LOCK = threading.Lock()
_HITS: Dict[str, Deque[float]] = {}

_GC_INTERVAL = 300.0   # 每 5 分钟清理一次
_MAX_KEEP = 3600.0     # 比最长窗口（1 小时）更旧的 key 视为可回收
_last_gc = 0.0


def check(key: str, limit: int, window: int) -> Tuple[bool, int]:
    """记一次命中并判断是否放行。

    返回 (allowed, retry_after)。被拒时 retry_after 是建议等待的秒数。
    limit <= 0 表示不限制。
    """
    if limit <= 0 or window <= 0:
        return True, 0

    now = time.time()
    cutoff = now - window
    with _LOCK:
        dq = _HITS.get(key)
        if dq is None:
            dq = _HITS[key] = deque()
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            # 等最早那一次滑出窗口
            return False, max(1, int(dq[0] + window - now) + 1)
        dq.append(now)
        _gc(now)
    return True, 0


def _gc(now: float) -> None:
    """调用方已持锁：清掉整桶都过期的 key。"""
    global _last_gc
    if now - _last_gc < _GC_INTERVAL:
        return
    _last_gc = now
    horizon = now - _MAX_KEEP
    for k, dq in list(_HITS.items()):
        if not dq or dq[-1] < horizon:
            _HITS.pop(k, None)


def reset() -> None:
    """清空全部计数（测试用）。"""
    global _last_gc
    with _LOCK:
        _HITS.clear()
        _last_gc = 0.0


def stats() -> Dict[str, int]:
    """当前被跟踪的 key 数量（诊断用）。"""
    with _LOCK:
        return {"keys": len(_HITS), "hits": sum(len(d) for d in _HITS.values())}
