# 📚 RAG 知识库问答（带引用溯源）

一个**不依赖 LangChain 全家桶**、手写核心链路的检索增强生成（RAG）问答系统。
上传文档后提问，系统返回答案并**标明每条结论的出处**（文件名 + 页码 + 原文片段），
前端点击引用徽标即可跳转到对应来源。

## ✨ 特性

- **手写 RAG 链路**：解析 → 切分 → Embedding → 向量召回 → 重排 → 生成 → 引用解析，每一步都可见、可改。
- **引用溯源**：答案中的 `[1][2]` 由大模型标注、系统校验映射回原文，避免“幻觉式引用”。
- **混合检索 + RRF 重排**：向量语义召回 + BM25 词面召回，用 Reciprocal Rank Fusion 融合，提升专有名词/数字类问题的命中。
- **多格式文档**：`txt / md / pdf / docx`，PDF 自动按页标注来源。
- **可插拔组件**：Embedding 支持本地 FastEmbed（BGE 中文小模型，零 API Key）或 OpenAI；大模型支持任意 OpenAI 兼容接口（OpenAI / DeepSeek / 通义 / 智谱 / Ollama 等）。
- **流式输出**：SSE 流式问答，逐字呈现。

## 🧱 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 后端 | FastAPI | 文档上传 / 问答 / 管理 API |
| 向量库 | Chroma | 本地持久化，零运维 |
| Embedding | FastEmbed (BAAI/bge-small-zh-v1.5) | 本地 onnx，无需 torch |
| 重排 | rank_bm25 + RRF | 手写融合 |
| 大模型 | OpenAI 兼容 Chat API | 可接任意兼容服务 |

## 🚀 快速开始

```bash
# 1. 安装依赖（建议新建虚拟环境）
pip install -r requirements.txt

# 2. 配置（复制示例并填入你的 LLM API Key）
cp .env.example .env
#   编辑 .env：设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
#   Embedding 默认用本地 FastEmbed，无需 Key；如需换 OpenAI embedding 见下方说明

# 3. 启动
python run.py
#   或： uvicorn app.main:app --host 0.0.0.0 --port 8000
#   或： Windows 下直接双击 start.bat（自动开浏览器）

# 4. 打开浏览器
#   http://localhost:8000
```

Windows 用户可以直接双击根目录的 **`start.bat`**：它会检查 Python 环境、检测 8000 端口是否被占用，启动服务后自动打开浏览器；关闭该黑窗口即停止服务。

### 📦 打包为 Windows exe（免 Python 环境）

```bash
python build_exe.py
```

产物在 `dist/rag-kb-qa/`（onedir 模式）。双击 `rag-kb-qa.exe` 后**不弹黑色控制台窗口**：服务在后台启动、自动打开浏览器，右下角任务栏出现托盘图标；**右键托盘图标 → 退出**才会停止服务。关闭浏览器标签页不影响服务，重复双击也只会打开一次页面（不会起第二个实例）。

`.env`、`data/`、`frontend/` 都放在 exe 同级目录，**整个文件夹拷贝到其他 Windows 电脑即可运行**（无需安装 Python）。运行日志写在同级的 `logs/app.log`，排查问题时看这里；`停止服务.bat` 用于托盘图标找不到时强制结束。

注意事项：

- 向量后端默认使用纯 NumPy 实现（`app/np_backend.py`）。chromadb 1.x 的 Rust 原生绑定在 PyInstaller 打包后必然段错误，因此 exe 版本不支持 `VECTOR_BACKEND=chroma`。
- 打包环境要求 `onnxruntime` 为 1.20.x（新版在打包后 import 即崩溃，`requirements.txt` 已锁定）。
- 托盘图标依赖 `pystray` + `Pillow`（打包脚本已声明 `pystray._win32` 隐藏导入）。
- 想把现有资料库带进 exe：把 `data/` 和 `.env` 拷到 exe 同级目录即可（向量数据在 `data/vecstore/<租户>/`）。
- 重新打包前若提示 dist 目录被占用/残留，先手动删除（或改名）`dist/rag-kb-qa`。

> 连不上 HuggingFace 时（国内网络常见），把 `.env` 里的 `EMBEDDING_PROVIDER` 改成 `offline`，启动立刻完成，无需等待模型下载。切换 embedding 实现后必须清空向量库重建（chroma 后端为 `data/chroma`，numpy 后端为 `data/vecstore`），否则向量维度不一致会报错。
启动后左侧上传文档，右侧输入问题即可。

## 🔌 切换大模型 / Embedding

`.env` 中：

- **大模型**：`LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL`，只要兼容 `/v1/chat/completions` 即可。
  例如本地 Ollama：`LLM_BASE_URL=http://localhost:11434/v1`、`LLM_MODEL=llama3`、`LLM_API_KEY=ollama`。
- **Embedding 改用 OpenAI**：`EMBEDDING_PROVIDER=openai`、`OPENAI_API_KEY=...`、`OPENAI_EMBED_MODEL=text-embedding-3-small`。

### 🧪 离线 / 演示模式（无需联网、无需任何 Key）

如果你的环境暂时无法下载模型或没有 LLM Key，可用内置的离线模式把整条链路跑通：

```bash
# 临时覆盖配置启动（不改 .env）
EMBEDDING_PROVIDER=offline LLM_PROVIDER=mock python run.py
```

- `EMBEDDING_PROVIDER=offline`：使用本地哈希向量（字符 n-gram 词袋），不下载任何模型，
  能正确处理“字面命中”类检索。适合演示与自测，不具备语义泛化能力。
- `LLM_PROVIDER=mock`：不调用大模型，直接基于检索到的上下文拼出带 `[n]` 引用的答案，
  用于在无 Key/无网环境下验证“上传 → 检索 → 引用溯源”的完整流程。

> 生产环境请改回 `EMBEDDING_PROVIDER=fastembed`（或 openai）、`LLM_PROVIDER=openai`。
> 注意：更换 Embedding 模型后，旧的向量数据维度可能不兼容，建议清空 `data/chroma` 再入库。

## 📡 API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 + 向量数量 |
| POST | `/api/documents` | 上传文档并入库（multipart, 字段 `file`） |
| GET | `/api/documents` | 列出已入库文档 |
| DELETE | `/api/documents/{doc_id}` | 删除文档及其切片 |
| POST | `/api/query` | 问答（非流式），返回 `answer` + `citations` + `contexts` |
| POST | `/api/query/stream` | 流式问答（SSE） |

`/api/query` 请求体示例：
```json
{
  "question": "文档的核心结论是？",
  "top_k": 6,
  "doc_id": "",
  "stream": false
}
```

## 🗂️ 目录结构

```
rag-kb-qa/
├── app/
│   ├── config.py        # 配置（pydantic-settings）
│   ├── embeddings.py    # Embedding 抽象 + FastEmbed/OpenAI 实现
│   ├── vectorstore.py   # 向量库统一入口（numpy / chroma 双后端）
│   ├── np_backend.py    # 纯 NumPy 向量后端（默认，exe 打包安全）
│   ├── chroma_backend.py# chromadb 后端（可选，仅源码运行）
│   ├── ingest.py        # 文档解析 + 递归切分
│   ├── retriever.py     # 向量召回 + BM25 RRF 重排
│   ├── generator.py     # Prompt 构建 + LLM 调用 + 引用解析
│   ├── store.py         # 文档清单（JSON 注册表）
│   ├── models.py        # 请求/响应模型
│   └── main.py          # FastAPI 路由
├── frontend/index.html  # 单页前端（上传/问答/溯源）
├── launcher.py          # exe 入口：后台启动 + 托盘图标 + 自动开浏览器
├── app_icon.py          # 用代码绘制程序图标/托盘图标
├── build_exe.py         # 一键打包（PyInstaller onedir）
├── data/                # uploads + vecstore + documents.json（运行时生成）
├── requirements.txt
├── .env.example
├── sync.py              # 本地库 -> 线上库 增量同步
├── start.bat            # Windows 双击启动
└── run.py
```

## 🔍 引用溯源是怎么做到的（核心思路）

1. 检索到的上下文块按 `[1][2]...` 顺序编号，并在每块前标注 `来源文件名 + 页码`。
2. 系统提示要求模型**仅在资料内作答**，并用方括号编号内联引用；资料不足时明确说“无法回答”，不编造。
3. 拿到回答后用正则抽取 `[n]`，按顺序去重并映射回原始块，输出**答案 + 被引用的出处清单**。

引用是“模型标注、系统校验”，而非模型自由生成链接，因此能最大限度抑制幻觉引用。

## ⚙️ 可调参数（.env）

- `CHUNK_SIZE` / `CHUNK_OVERLAP`：切分粒度与重叠。
- `TOP_K`：喂给大模型的上下文块数；`VECTOR_TOP_K`：向量召回候选数。
- `USE_RERANK`：是否启用 BM25 + RRF 重排。
- `LLM_TEMPERATURE` / `LLM_MAX_TOKENS`：生成控制。

## 👥 多设备与资料库隔离（多租户）

每台设备（或每位用户）拥有**完全独立**的资料库：`documents.json`、`uploads/`、
Chroma collection 三层全部按租户隔离，上传的文档只进自己的库，互相看不到。

- **主设备（管理员）**：使用固定租户（默认 `admin`），需要 `ADMIN_PASSWORD` 登录。
  管理员可在「资料库总览」里查看所有库并清理不再使用的游客库。
- **游客设备**：无需任何口令。前端首次访问会生成一个随机租户 ID
  （`g_` + 96 位随机，保存在浏览器 localStorage），等同于这台设备的私有凭证；
  同一浏览器再次打开自动回到自己的库，换浏览器/清缓存则是新库。
- **换设备不丢库**：游客模式点右上角身份徽章（或点顶部的「库 xxxx」标签）可打开
  「我的资料库」面板，里面有**资料库编号**与**恢复链接**。在新设备上粘贴编号
  （或直接打开恢复链接）即可回到原来的库；也可以主动「换一个新的空资料库」重新开始。
  编号是 96 位随机串，猜不到，本身就是这台设备的凭证——所以不要把编号发给不信任的人。
- **共享主库兜底**：游客库还没有文档时，问答自动检索主库（只读，界面上有提示）；
  上传第一篇文档后，回答只基于自己的资料。

存储布局：

```
data/
├── chroma/                 # 所有租户的 collection 共享目录（名字不同）
└── tenants/
    ├── index.json          # 租户索引（总览用）
    ├── admin/              # 主设备
    │   ├── documents.json
    │   └── uploads/
    └── g_xxxxxxxx/         # 每个游客设备一个目录
```

## 🔐 访问控制（角色权限）

| 身份 | 进入方式 | 权限 |
|---|---|---|
| admin | `ADMIN_PASSWORD` 口令登录 | 全部：主库增删、游客库清理、总览 |
| user | `USER_PASSWORD` 口令登录（可选） | 主库只读 |
| guest | 免口令，自动分配独立库 | 自己的库可增删 |
| shared | 未设口令时的主库 | 只读（带平台 sharecode 时视为 admin） |

**行为说明：**
- 口令与租户都通过 **POST body** 传输（`token` / `tenant` 字段），同时兼容
  `Authorization: Bearer` 头与 `?token=`——部署平台网关会剥离头和查询参数，
  body 是唯一在本地和线上都稳定的通道。
- 受保护接口：上传/删除仅限有写入权限的身份。`/api/health`、`/` 保持公开。
- 登录接口：`POST /api/login` `{"password": "..."}` → `{"ok": true, "role": "admin"|"user", "tenant": "admin"}`。
- 上传文件除类型限制（txt/md/pdf/docx）外，还会在读取前按声明大小快拦、读取时分块硬上限，防止超大文件撑爆内存。
- 游客库数量上限 `MAX_TENANTS`（默认 200），防止被扫描器刷爆磁盘。

> 提示：共享口令适合小团队/个人。若要更严格的权限（多用户、配额、审计），建议在 nginx 层做鉴权或接入 OAuth。

## 🚦 限流（防刷大模型额度 / 防口令爆破）

问答每次都真实调用大模型，是全站最值钱的口子；登录接口则可能被暴力猜口令。
两者都有滑动窗口限流（进程内实现，见 `app/ratelimit.py`，无外部依赖）：

| 维度 | 默认 | 说明 |
|---|---|---|
| 单资料库提问 | 20 次/分钟，300 次/小时 | 管理员不受此限 |
| 单 IP 总请求 | 120 次/分钟 | 兜底：防止换租户绕过配额 |
| 单资料库上传 | 10 次/分钟 | |
| 单 IP 登录尝试 | 10 次/分钟 | 防暴力猜口令 |
| 全站登录尝试 | 30 次/分钟 | 不依赖 IP 的兜底（见下方说明） |
| 单 IP 编号查询 | 30 次/分钟 | 「我的资料库」恢复校验接口 |

超限返回 `429` + `Retry-After` 秒数，前端会把「提问过于频繁，请 N 秒后再试」直接显示在回答区。
所有阈值都在 `.env` 里可调（`RATE_LIMIT_*`），设为 `0` 表示不限。

> **平台部署的重要说明**：实测 WorkBuddy「发布为应用」的网关会剥离 X-Forwarded-For，
> 且每次请求用不同的出口 IP 连接应用（轮换的云内网段），因此**「按 IP」这几层在线上不生效**。
> 线上真正起作用的是：按资料库的提问/上传配额 + 全站登录总数兜底（都不依赖 IP）。
> 自托管（nginx 直连、真实客户端 IP）时所有层级都精确生效。

## 🌐 远端运行

本服务绑定 `0.0.0.0` 并监听 `PORT` 环境变量（部署平台注入，默认 8000），**不只是本地**。常见做法：

1. **云服务器 / VPS**：`pip install -r requirements.txt && python run.py`，开放端口（或 nginx 反代到 80/443 + HTTPS + 域名）。
2. **WorkBuddy「发布为应用」**：本项目是单端口 HTTP 服务，可直接发布为在线链接（已内置管理员/访客权限）。
3. **Docker**：把 Python 环境 + 代码打包，丢到任意容器平台。

远端部署 checklist：①开启 `ADMIN_PASSWORD`（并建议设置 `USER_PASSWORD` 供只读访客）；②确认 `data/chroma`、`data/uploads` 所在磁盘**持久化**（否则重启丢库）；③密钥用环境变量注入而非明文 `.env`；④前置 HTTPS。

> **WorkBuddy「发布为应用」部署注意（实测结论，务必看）**
>
> 1. 部署会把本地 `.env` **一起打包上传**，所以 `ADMIN_PASSWORD` 非空时，线上实例会变成 `auth_required: true`。
> 2. 但该平台网关会剥离 `Authorization` 头、`?token=` 参数和 Cookie（实测连 `?sharecode=` 也不会透传给应用），应用内登录在线上**无法持续生效**，任何接口都会 401。
> 3. 结论：发布前把 `ADMIN_PASSWORD` / `USER_PASSWORD` **留空**，由平台 sharecode 控权；本地想保留登录体验，用 `start.bat` 启动（内部 `set` 环境变量注入口令，优先级高于 `.env`）。
> 4. 角色权限模型适用于云服务器 / Docker 等自托管场景。

## 🔄 本地与线上同步

本地和线上是**两套独立实例**，各自持有一份 `data/`，不会自动同步——部署只是把当时的 `data/` 打了个快照带走，之后各写各的。

`sync.py` 提供「本地 → 线上」的单向增量同步：

```bash
python sync.py --dry-run   # 先看差异（本地有、线上没有的文档）
python sync.py             # 真正上传
python sync.py --force     # 同名文档也重传（会产生线上副本）
```

规则：按**文件名**比对两边清单，只推「本地有、线上没有」的；线上已有同名文档自动跳过；**不会删除线上任何文档**。

配置写在 `.env`：

```ini
REMOTE_BASE_URL=https://your-app.app.workbuddy.host
REMOTE_SHARECODE=xxxxxxxx
```

注意事项：
- 线上沙箱休眠后首次请求需要冷启动，可能返回 504；脚本会自动退避重试 3 次，仍失败就隔一两分钟重跑（已同步的会跳过）。
- 单文件建议 5MB 以内：网关单请求超时约 60s，超大文件容易中断，可拆成多份传。
- 反向（线上 → 本地）没有实现：线上文档在网页上可随时下载，重新入库即可。
- 重新部署会用**本地 `data/` 覆盖线上**，线上独有的文档会丢，部署前先确认。

## 📘 内置示例文档：Python 中文官方教程

`samples/` 目录附带一份可直接入库的示例文档：

- `samples/python_zh_tutorial.md` —— **Python 3 官方教程中文版**（抓取自 docs.python.org/zh-cn/3/tutorial，16 章全，约 240KB，保留章节标题层级与 676 个代码块）。
- `samples/fetch_python_zh_tutorial.py` —— 抓取/清洗脚本，可随时重跑更新（`python samples/fetch_python_zh_tutorial.py`）。

**用法**：管理员登录后在网页上传该文件，或调用 `POST /api/documents` 上传。入库后即可提问，例如：

- 「Python 中如何用 open 函数读写文件？」
- 「如何定义类？类与实例、方法是什么关系？」
- 「列表和元组有什么区别？」

答案会带 `[n]` 引用，点击可溯源到教程原文片段。


