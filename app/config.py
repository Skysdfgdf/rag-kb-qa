"""全局配置（基于 pydantic-settings，可从 .env 或环境变量读取）。"""
from pathlib import Path

import sys

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/ 目录；项目根 = app/ 的上一级
APP_DIR = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    # PyInstaller 打包后：.env / data/ / frontend/ 都放在 exe 同级目录
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = APP_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Embedding ----------
    embedding_provider: str = "fastembed"      # fastembed | openai | offline(本地哈希,免下载)
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    openai_embed_model: str = "text-embedding-3-small"
    offline_embed_dim: int = 512               # 仅 offline 模式使用

    # ---------- 向量库 ----------
    chroma_path: str = str(PROJECT_ROOT / "data" / "chroma")
    collection_name: str = "kb_default"
    # 向量后端：numpy（默认，纯 Python 实现，exe 打包安全）| chroma（chromadb）
    # chromadb 1.x 的 Rust 绑定在 PyInstaller 下会段错误，故 exe 必须用 numpy
    vector_backend: str = "numpy"

    # ---------- 数据目录 ----------
    data_dir: str = str(PROJECT_ROOT / "data")
    # 以下两项是「老版本单库」的路径，只在启动迁移时使用；
    # 迁移后每个租户的数据在 data/tenants/<tenant_id>/ 下。
    uploads_dir: str = str(PROJECT_ROOT / "data" / "uploads")
    documents_registry: str = str(PROJECT_ROOT / "data" / "documents.json")

    # ---------- 多租户（每台设备一套独立资料库）----------
    # 主设备（管理员）固定使用该租户，游客设备各自生成 g_<随机> 租户
    default_tenant: str = "admin"
    tenants_dir: str = str(PROJECT_ROOT / "data" / "tenants")
    # 游客库还没有任何文档时，问答兜底检索主库（只读），避免新设备看到空库
    guest_fallback_shared: bool = True
    # 最多允许创建的租户数量，防止被扫描器刷爆磁盘
    max_tenants: int = 200
    # 单个游客库最多文档数（主库不受限），防止单台设备把磁盘刷满
    guest_max_docs: int = 50

    # ---------- 切分 ----------
    chunk_size: int = 480
    chunk_overlap: int = 80

    # ---------- 检索 ----------
    top_k: int = 6
    vector_top_k: int = 20
    use_rerank: bool = True

    # ---------- 大模型（OpenAI 兼容）----------
    llm_provider: str = "openai"               # openai | mock(离线演示,无需 Key)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ---------- OpenAI（embedding）----------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # ---------- 访问控制 / 角色权限（远端部署安全）----------
    # 启用条件：设置 ADMIN_PASSWORD 即开启鉴权。
    #   - 管理员（ADMIN_PASSWORD）：拥有全部权限（上传 / 删除 / 列表 / 问答）。
    #   - 普通用户（USER_PASSWORD，可选）：只读身份，仅可列表 / 问答，不能上传或删除。
    #   - 两者都为空 = 不鉴权（本地开发，等同管理员全权）。
    # 注：旧版单一口令 APP_PASSWORD 仍可用，等价于管理员口令（留空即可）。
    admin_password: str = ""
    user_password: str = ""
    app_password: str = ""          # 向后兼容：等价于管理员口令
    # 单文件最大字节数（默认 10MB）
    max_upload_size: int = 10 * 1024 * 1024

    # ---------- 限流（防刷大模型额度 / 防口令爆破）----------
    # 滑动窗口计数，进程内实现（见 app/ratelimit.py）。设为 0 表示不限。
    rate_limit_enabled: bool = True
    rate_limit_query_per_min: int = 20     # 单个资料库每分钟提问次数（管理员不受此限）
    rate_limit_query_per_hour: int = 300   # 单个资料库每小时提问次数
    rate_limit_ip_per_min: int = 120       # 单个 IP 每分钟请求总数（防换租户绕过）
    rate_limit_upload_per_min: int = 10    # 单个资料库每分钟上传次数
    rate_limit_login_per_min: int = 10     # 单个 IP 每分钟登录尝试次数（防爆破；平台网关下不生效）
    rate_limit_login_global_per_min: int = 30  # 全站每分钟登录尝试总数（不依赖 IP，防爆破兜底）
    rate_limit_check_per_min: int = 30     # 单个 IP 每分钟资料库编号查询次数

    # ---------- 线上同步目标（sync.py 使用）----------
    # 本地库 -> 线上库的单向增量同步，填线上域名与 sharecode 即可。
    remote_base_url: str = ""
    remote_sharecode: str = ""

    @property
    def auth_enabled(self) -> bool:
        return bool(self.admin_password or self.app_password)

    @property
    def effective_admin_password(self) -> str:
        return self.admin_password or self.app_password


settings = Settings()
