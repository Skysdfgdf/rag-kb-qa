"""FastAPI 应用：多租户文档管理 + 带溯源的问答。

多租户要点
    每台设备（或每位用户）对应一个 tenant，各自拥有独立的
    documents.json / uploads/ / Chroma collection，上传只进自己的库。
    主设备（管理员）使用 settings.default_tenant，需要口令才能进入；
    其他设备默认以游客身份进入，前端为自己生成一个随机 tenant。

身份判定
    admin  —— 主设备管理员，可增删主库与任何游客库
    user   —— 主设备只读口令（可选）
    guest  —— 游客，只能读写自己那个租户的库
    shared —— 未设口令时的主库，只读
"""
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import generator, ingest, ratelimit, retriever, store, tenants
from .config import PROJECT_ROOT, settings
from .models import (
    DeleteDocRequest,
    DeleteTenantRequest,
    DocumentInfo,
    LoginRequest,
    QueryRequest,
    QueryResponse,
    SessionInfo,
    TenantCheckRequest,
    TenantRequest,
)
from .vectorstore import VectorStore

ALLOWED_EXT = {".txt", ".md", ".pdf", ".docx"}

# 构建标记：用于确认线上实例是否已加载最新代码
BUILD = "2026-11-10T17:20"

app = FastAPI(title="RAG 知识库问答", version="2.0.0")


@app.on_event("startup")
def _boot() -> None:
    """启动时把老版本单库数据迁移成主设备租户，并确保目录就绪。"""
    msg = tenants.migrate_legacy()
    if msg:
        print(f"[tenants] {msg}")


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    """页面 HTML 一律不缓存。

    否则前端改动后浏览器仍用旧副本，会出现“接口有数据但页面显示暂无文档”
    这类莫名其妙的症状（还要用户手动强刷才恢复）。
    """
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---------- 向量库实例缓存（每个租户一个）----------
_stores: Dict[str, VectorStore] = {}


def get_vs(tenant: str, create: bool = True) -> Optional[VectorStore]:
    """取得某租户的向量库；create=False 时不做任何创建（读请求用）。"""
    tenant = tenants.normalize(tenant)
    if not tenants.is_valid(tenant):
        return None
    if not create and not tenants.exists(tenant):
        return None
    vs = _stores.get(tenant)
    if vs is None:
        vs = VectorStore(tenant)
        _stores[tenant] = vs
    return vs


def drop_vs(tenant: str) -> None:
    """删除某租户的向量集合（无论它是否已被缓存过）。"""
    _stores.pop(tenant, None)
    from .vectorstore import drop_collection

    drop_collection(tenant)


# ---------- 身份与租户解析 ----------
class Access:
    """一次请求的身份 + 所属租户。"""

    __slots__ = ("tenant", "role")

    def __init__(self, tenant: str, role: str):
        self.tenant = tenant
        self.role = role

    @property
    def is_main(self) -> bool:
        return tenants.is_default(self.tenant)

    @property
    def can_write(self) -> bool:
        if self.role == "admin":
            return True
        # 游客只能写自己的库；主库对游客永远只读
        return self.role == "guest" and not self.is_main


def _token_from_request(request: Request, explicit: Optional[str] = None) -> str:
    """令牌通道：body 显式字段 > Authorization 头 > ?token > Cookie。

    平台网关会剥离 Authorization / 查询参数 / Cookie，只放行 POST body，
    所以 body 里的 token 是第一优先，前端所有请求都带它。
    """
    if explicit:
        return explicit
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token") or request.cookies.get("rag_token") or ""


def _sharecode_ok(request: Request) -> bool:
    expected = settings.remote_sharecode
    return bool(expected) and request.query_params.get("sharecode") == expected


def resolve(request: Request, tenant_raw: Optional[str], token_raw: Optional[str] = None) -> Access:
    """解析本次请求的租户与角色；主库缺少有效口令时抛 401。"""
    tenant = tenants.normalize(tenant_raw)
    if not tenants.is_valid(tenant):
        raise HTTPException(400, "资料库标识不合法（仅允许字母、数字、下划线、短横线）")

    if not tenants.is_default(tenant):
        return Access(tenant, "guest")  # 游客自有库：持有租户 ID 即视为拥有者

    admin_pwd = settings.effective_admin_password
    if not admin_pwd:
        # 未设口令（线上常态）：带正确 sharecode 视为管理员（供平台控制台 / sync.py），
        # 否则主库降级为只读共享库。
        return Access(tenant, "admin" if _sharecode_ok(request) else "shared")

    tok = _token_from_request(request, token_raw)
    if tok and tok == admin_pwd:
        return Access(tenant, "admin")
    if settings.user_password and tok == settings.user_password:
        return Access(tenant, "user")
    raise HTTPException(401, "主资料库需要管理员口令")


def _require_write(acc: Access) -> None:
    if acc.can_write:
        return
    if acc.is_main:
        raise HTTPException(403, "共享主库为只读：请用管理员口令登录，或上传到你自己的资料库")
    raise HTTPException(403, "当前身份无写入权限")


# ---------- 限流 ----------
def _client_ip(request: Request) -> str:
    """取真实客户端 IP。

    部署平台在前面挂了网关，request.client.host 会变成网关地址，
    真正来源在 X-Forwarded-For 的第一段。两者都拿不到时归为 unknown，
    这样所有匿名流量共用一个桶（宁可严一点，也不要完全不限）。

    已知限制：WorkBuddy 部署平台的网关会剥离 XFF，且每次请求用不同的
    出口 IP 连接应用（实测轮换），所以「按 IP」这层在平台上不生效——
    真正起作用的是按资料库/全局的配额；自托管（nginx 直连）时按 IP 精确。
    """
    xff = request.headers.get("x-forwarded-for") or ""
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real = request.headers.get("x-real-ip") or ""
    if real.strip():
        return real.strip()
    return request.client.host if request.client else "unknown"


def _enforce(bucket: str, key: str, limit: int, window: int, label: str) -> None:
    """超限则抛 429，并给出建议等待秒数。"""
    if not settings.rate_limit_enabled:
        return
    allowed, retry = ratelimit.check(f"{bucket}:{key}", limit, window)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"{label}过于频繁，请 {retry} 秒后再试",
            headers={"Retry-After": str(retry)},
        )


def _guard_query(acc: Access, request: Request) -> None:
    """问答限流：先按 IP 兜一层，再按资料库算配额。

    大模型每次调用都真金白银，且网关超时限制在 60s，
    所以问答是全站最需要保护的口子。管理员是主设备本人，跳过配额。
    """
    _enforce("ip", _client_ip(request), settings.rate_limit_ip_per_min, 60, "请求")
    if acc.role == "admin":
        return
    _enforce("q1m", acc.tenant, settings.rate_limit_query_per_min, 60, "提问")
    _enforce("q1h", acc.tenant, settings.rate_limit_query_per_hour, 3600, "提问")


def _read_store(acc: Access):
    """读书用：返回 (向量库, 是否用主库兜底)。"""
    vs = get_vs(acc.tenant, create=False)
    if vs is not None and vs.count() > 0:
        return vs, False
    if not acc.is_main and settings.guest_fallback_shared:
        shared = get_vs(settings.default_tenant, create=False)
        if shared is not None and shared.count() > 0:
            return shared, True
    return vs, False


# ---------- 会话 / 登录 ----------
@app.post("/api/session", response_model=SessionInfo)
def session(req: TenantRequest, request: Request):
    """前端启动时调用：确认这台设备用哪个资料库、什么身份。"""
    acc = resolve(request, req.tenant, req.token)
    _, fallback = _read_store(acc)
    return SessionInfo(
        tenant=acc.tenant,
        role=acc.role,
        is_main=acc.is_main,
        can_write=acc.can_write,
        doc_count=len(store.list_documents(acc.tenant)),
        shared_fallback=fallback,
        tenant_exists=tenants.exists(acc.tenant),
    )


@app.post("/api/tenant/check")
def tenant_check(req: TenantCheckRequest, request: Request):
    """校验某个资料库编号是否存在（换设备恢复时先验一下，避免进到空库）。

    编号本身就是这台设备的凭证（96 位随机，猜不到），所以这里不需要口令；
    但仍然限流，防止被当成枚举接口来扫。
    """
    _enforce("tcheck", _client_ip(request), settings.rate_limit_check_per_min, 60, "编号查询")

    raw = (req.tenant or "").strip()
    if not raw:
        raise HTTPException(400, "请填写资料库编号")
    t = tenants.normalize(raw)
    if not tenants.is_valid(t):
        raise HTTPException(400, "资料库编号格式不正确（只允许字母、数字、下划线、短横线）")
    return {
        "tenant": t,
        "exists": tenants.exists(t),
        "doc_count": tenants.doc_count(t),
        "is_main": tenants.is_default(t),
    }


@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    # 口令接口必须限流：不限的话可以无限次试密码。
    # 两层：按 IP（自托管时精确）+ 全局（部署平台网关轮换出口 IP，按 IP 失效，
    # 全局桶是防爆破的最后一道闸，正常人一分钟不会登录 30 次）。
    _enforce("login_g", "__all__", settings.rate_limit_login_global_per_min, 60, "登录尝试")
    _enforce("login", _client_ip(request), settings.rate_limit_login_per_min, 60, "登录尝试")

    admin_pwd = settings.effective_admin_password
    main = settings.default_tenant
    if not admin_pwd:
        return {"ok": True, "auth_required": False, "role": "shared", "tenant": main}
    if req.password == admin_pwd:
        return {"ok": True, "auth_required": True, "role": "admin", "tenant": main}
    if settings.user_password and req.password == settings.user_password:
        return {"ok": True, "auth_required": True, "role": "user", "tenant": main}
    raise HTTPException(status_code=401, detail="访问口令不正确")


@app.post("/api/me")
def me(request: Request, req: Optional[TenantRequest] = None):
    tenant = (req.tenant if req else None) or settings.default_tenant
    token = req.token if req else None
    acc = resolve(request, tenant, token)
    return {"role": acc.role, "tenant": acc.tenant, "auth_required": settings.auth_enabled}


@app.get("/api/health")
def health():
    vs = get_vs(settings.default_tenant, create=False)
    return {
        "status": "ok",
        "build": BUILD,
        "collection": settings.collection_name,
        "vectors": vs.count() if vs else 0,
        "auth_required": settings.auth_enabled,
        "has_user_login": bool(settings.user_password),
        "max_upload_mb": settings.max_upload_size // (1024 * 1024),
        "multi_tenant": True,
        "default_tenant": settings.default_tenant,
        "tenant_count": len(tenants.list_tenants()),
        "rate_limit": {
            "enabled": settings.rate_limit_enabled,
            "query_per_min": settings.rate_limit_query_per_min,
            "query_per_hour": settings.rate_limit_query_per_hour,
        },
        "rl_stats": ratelimit.stats(),
    }


# ---------- 文档上传 ----------
@app.post("/api/documents", response_model=DocumentInfo)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    tenant: Optional[str] = Form(None),
    token: Optional[str] = Form(None),
):
    acc = resolve(request, tenant or request.query_params.get("tenant"), token)
    _require_write(acc)
    _enforce("upload", acc.tenant, settings.rate_limit_upload_per_min, 60, "上传")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型: {ext}（仅支持 txt/md/pdf/docx）")

    # 先按客户端声明大小快速拦截，再分块读取并硬上限，防止超大文件占用内存
    declared = getattr(file, "size", None)
    if declared is not None and declared > settings.max_upload_size:
        mb = settings.max_upload_size // (1024 * 1024)
        raise HTTPException(413, f"文件过大：最大允许 {mb}MB，当前约 {declared // (1024 * 1024)}MB")

    if not tenants.exists(acc.tenant):
        # 新游客库：限制总数，避免被扫描器刷出大量目录
        if len(tenants.list_tenants()) >= settings.max_tenants:
            raise HTTPException(429, "本实例的资料库数量已达上限，请联系管理员清理")
        tenants.ensure(acc.tenant)
    elif not acc.is_main and len(store.list_documents(acc.tenant)) >= settings.guest_max_docs:
        raise HTTPException(
            429, f"单个资料库最多 {settings.guest_max_docs} 篇文档，请先删除不需要的再上传"
        )

    up_dir = tenants.uploads_dir(acc.tenant)
    up_dir.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4().hex
    save_path = up_dir / f"{doc_id}{ext}"

    try:
        # 分块读取并限制总大小（流式，避免整文件载入内存）
        data = bytearray()
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > settings.max_upload_size:
                mb = settings.max_upload_size // (1024 * 1024)
                raise HTTPException(413, f"文件过大：最大允许 {mb}MB")
        save_path.write_bytes(bytes(data))

        # 1) 解析 -> 2) 切分 -> 3) 写入本租户的向量库
        pages = await run_in_threadpool(ingest.load_text_from_file, str(save_path))
        chunks = await run_in_threadpool(ingest.chunk_pages, pages)
        if not chunks:
            raise ValueError("文档解析后为空，未产生任何文本块")

        metadatas = [
            {
                "doc_id": doc_id,
                "doc_name": file.filename,
                "tenant": acc.tenant,
                "page": m.get("page", 0),
                "chunk_index": i,
            }
            for i, (_, m) in enumerate(chunks)
        ]
        texts = [t for t, _ in chunks]
        vs = get_vs(acc.tenant)
        if vs is None:
            raise ValueError("资料库初始化失败")
        await run_in_threadpool(vs.add, texts, metadatas)
        store.add_document(acc.tenant, doc_id, file.filename, len(chunks))
        tenants.touch(acc.tenant)
    except HTTPException:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001
        # 出错时清理已落盘的文件，避免脏数据
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"文档入库失败: {e}")

    return store.get_document(acc.tenant, doc_id)


# ---------- 文档列表 ----------
@app.post("/api/documents/list", response_model=list[DocumentInfo])
def list_documents_post(req: TenantRequest, request: Request):
    acc = resolve(request, req.tenant, req.token)
    return store.list_documents(acc.tenant)


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents_get(request: Request, tenant: Optional[str] = None, token: Optional[str] = None):
    """GET 版本：本地 / 自托管 / sync.py 使用（平台网关会剥离查询参数）。"""
    acc = resolve(request, tenant, token)
    return store.list_documents(acc.tenant)


# ---------- 删除文档 ----------
def _delete_doc(acc: Access, doc_id: str) -> Dict:
    _require_write(acc)
    if not store.get_document(acc.tenant, doc_id):
        raise HTTPException(404, "文档不存在")
    vs = get_vs(acc.tenant, create=False)
    if vs is not None:
        vs.delete_by_doc(doc_id)
    store.remove_document(acc.tenant, doc_id)
    # 一并清理 uploads 下的原始文件（文件名以 doc_id 打头），
    # 否则删掉文档后会留下无主文件，越积越多。
    for p in Path(tenants.uploads_dir(acc.tenant)).glob(f"{doc_id}.*"):
        try:
            p.unlink()
        except OSError:
            pass
    return {"deleted": doc_id, "tenant": acc.tenant, "ok": True}


@app.post("/api/documents/delete")
def delete_document_post(req: DeleteDocRequest, request: Request):
    acc = resolve(request, req.tenant, req.token)
    return _delete_doc(acc, req.doc_id)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, request: Request, tenant: Optional[str] = None):
    acc = resolve(request, tenant, None)
    return _delete_doc(acc, doc_id)


# ---------- 管理员：租户总览 ----------
@app.post("/api/admin/tenants")
def admin_tenants(req: TenantRequest, request: Request):
    acc = resolve(request, req.tenant or settings.default_tenant, req.token)
    if acc.role != "admin":
        raise HTTPException(403, "权限不足：仅管理员可查看所有资料库")
    return tenants.list_tenants()


@app.post("/api/admin/tenants/delete")
def admin_delete_tenant(req: DeleteTenantRequest, request: Request):
    acc = resolve(request, settings.default_tenant, req.token)
    if acc.role != "admin":
        raise HTTPException(403, "权限不足：仅管理员可删除资料库")
    target = tenants.normalize(req.tenant)
    if tenants.is_default(target):
        raise HTTPException(400, "主设备资料库不允许删除")
    if not tenants.is_valid(target) or not tenants.exists(target):
        raise HTTPException(404, "资料库不存在")
    drop_vs(target)
    tenants.remove(target)
    return {"deleted": target, "ok": True}


@app.get("/api/netcheck")
def netcheck():
    """诊断：测试应用所在环境到关键外部服务的连通性。"""
    import socket
    import time as _time

    out = {}
    for name, host in [("deepseek", "api.deepseek.com"), ("huggingface", "huggingface.co")]:
        t0 = _time.time()
        try:
            s = socket.create_connection((host, 443), timeout=8)
            s.close()
            out[name] = f"ok {int((_time.time() - t0) * 1000)}ms"
        except Exception as e:  # noqa: BLE001
            out[name] = f"fail {type(e).__name__}"
    return out


# ---------- 问答 ----------
_EMPTY_MAIN = "知识库中还没有任何文档，请先在左侧上传资料。"
_EMPTY_GUEST = "你的资料库里还没有文档。上传后，回答只会基于你自己的资料。"


def _empty_answer(acc: Access) -> str:
    return _EMPTY_MAIN if acc.is_main else _EMPTY_GUEST


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    acc = resolve(request, req.tenant, req.token)
    if not req.question or not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    _guard_query(acc, request)

    vs, fallback = _read_store(acc)
    if vs is None:
        return QueryResponse(answer=_empty_answer(acc), citations=[], contexts=[], shared_fallback=False)

    contexts = await run_in_threadpool(
        retriever.Retriever(vs).retrieve, req.question, req.top_k, req.doc_id
    )
    if not contexts:
        return QueryResponse(answer=_empty_answer(acc), citations=[], contexts=[], shared_fallback=fallback)

    g = get_generator()
    try:
        result = g.generate(req.question, contexts)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"大模型调用失败：{type(e).__name__}: {e}") from e
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        contexts=contexts,
        shared_fallback=fallback,
    )


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """流式问答（SSE）。事件格式：data: {json}\n\n"""
    acc = resolve(request, req.tenant, req.token)
    if not req.question or not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    _guard_query(acc, request)

    vs, fallback = _read_store(acc)
    if vs is None:
        async def _empty():
            yield "data: " + json.dumps(
                {
                    "type": "done",
                    "answer": _empty_answer(acc),
                    "citations": [],
                    "shared_fallback": False,
                },
                ensure_ascii=False,
            ) + "\n\n"

        return StreamingResponse(_empty(), media_type="text/event-stream")

    contexts = await run_in_threadpool(
        retriever.Retriever(vs).retrieve, req.question, req.top_k, req.doc_id
    )
    if not contexts:
        async def _empty2():
            yield "data: " + json.dumps(
                {
                    "type": "done",
                    "answer": _empty_answer(acc),
                    "citations": [],
                    "shared_fallback": fallback,
                },
                ensure_ascii=False,
            ) + "\n\n"

        return StreamingResponse(_empty2(), media_type="text/event-stream")

    g = get_generator()

    def event_stream():
        try:
            for ev in g.stream(req.question, contexts):
                if ev.get("type") == "done":
                    ev["shared_fallback"] = fallback
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
        except Exception as e:  # noqa: BLE001
            yield "data: " + json.dumps(
                {"type": "error", "message": f"大模型调用失败：{type(e).__name__}: {e}"},
                ensure_ascii=False,
            ) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def get_generator():
    """根据配置选择真实大模型或离线演示生成器。"""
    if settings.llm_provider == "mock":
        return generator.MockGenerator()
    return generator.Generator()


# 前端静态资源（在 API 路由之后挂载，确保 /api/* 优先匹配）
_FRONTEND_DIR = PROJECT_ROOT / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
