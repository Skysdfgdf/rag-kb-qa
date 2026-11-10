"""请求/响应数据模型。"""
from typing import Dict, List, Optional

from pydantic import BaseModel


class Citation(BaseModel):
    index: int
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None
    page: Optional[int] = None
    text: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    doc_id: Optional[str] = None  # 限定在某篇文档内检索；为空则本租户全库
    stream: bool = False
    # 多租户：租户标识与口令都放 body，平台网关只放行 POST body
    tenant: Optional[str] = None
    token: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    contexts: List[Dict]  # 检索到的原始上下文（含来源元数据），便于前端/调试
    shared_fallback: bool = False  # 是否使用了主库兜底检索


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    chunk_count: int
    created_at: str


class TenantRequest(BaseModel):
    """列表/会话类请求：只带租户与令牌。"""

    tenant: Optional[str] = None
    token: Optional[str] = None


class DeleteDocRequest(BaseModel):
    doc_id: str
    tenant: Optional[str] = None
    token: Optional[str] = None


class DeleteTenantRequest(BaseModel):
    tenant: str
    token: Optional[str] = None


class SessionInfo(BaseModel):
    tenant: str
    role: str  # admin | user | guest | shared
    is_main: bool
    can_write: bool
    doc_count: int
    shared_fallback: bool
    tenant_exists: bool = True  # 该资料库是否已在服务端存在（区分「新库」与「编号写错」）


class TenantCheckRequest(BaseModel):
    """换设备恢复：只校验编号是否存在。"""

    tenant: Optional[str] = None


class LoginRequest(BaseModel):
    password: str = ""
    tenant: Optional[str] = None
