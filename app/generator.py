"""生成与引用溯源。

核心思路（手写，不依赖框架的“引用插件”）：
  1. 把检索到的上下文块按顺序编号 [1]、[2]...，并附上“来源文件名 + 页码”。
  2. 系统提示要求模型仅在资料范围内作答，并用方括号编号内联引用。
  3. 拿到回答后，用正则把 [n] 抽出来，映射回对应的上下文块，
     输出“答案 + 被引用出处清单”，实现可点击溯源。

这样引用是“模型标注、系统校验”的，而不是模型凭空写的一串链接，
能最大程度避免幻觉式引用。
"""
import re
from typing import Dict, List

from .config import settings

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请仅依据下面提供的【参考资料】回答问题。
要求：
1. 回答要准确、简洁，使用与用户问题相同的语言。
2. 必须在正文中用方括号编号引用资料，例如 [1]、[3]，编号对应下方资料的序号，可在句末或句中标注。
3. 如果提供的资料中不包含答案，请明确说明“根据现有资料无法回答该问题”，不要编造任何内容。
4. 不要自行罗列参考资料列表，引用信息会由系统自动附加。"""


def _page_label(meta: Dict) -> str:
    page = meta.get("page")
    return f"第{page}页" if page else "全文"


def build_messages(question: str, contexts: List[Dict]) -> List[Dict[str, str]]:
    ref_lines = []
    for i, c in enumerate(contexts, start=1):
        src = c["metadata"].get("doc_name", "未知")
        ref_lines.append(f"[{i}] (来源: {src}, {_page_label(c['metadata'])})")
        ref_lines.append(c["text"])
        ref_lines.append("")  # 空行分隔
    ref_text = "\n".join(ref_lines)
    user = f"【参考资料】\n{ref_text}\n【用户问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_citations(answer: str, contexts: List[Dict]) -> List[Dict]:
    """从回答中提取 [n]，按顺序去重并映射回上下文块，生成出处清单。"""
    nums = [int(n) for n in _CITATION_RE.findall(answer)]
    seen = set()
    citations: List[Dict] = []
    for n in nums:
        if 1 <= n <= len(contexts) and n not in seen:
            seen.add(n)
            c = contexts[n - 1]
            citations.append(
                {
                    "index": n,
                    "doc_id": c["metadata"].get("doc_id"),
                    "doc_name": c["metadata"].get("doc_name"),
                    "page": c["metadata"].get("page"),
                    "text": c["text"],
                }
            )
    return citations


class Generator:
    def __init__(self):
        from openai import OpenAI

        # 显式超时：避免远端网络挂起导致请求无限等待（托管平台网关一般 60s 就断）
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=45.0,
            max_retries=1,
        )

    def _common_kwargs(self):
        return dict(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    def generate(self, question: str, contexts: List[Dict]) -> Dict:
        """非流式：返回 {answer, citations}。"""
        messages = build_messages(question, contexts)
        resp = self.client.chat.completions.create(
            messages=messages, **self._common_kwargs()
        )
        answer = resp.choices[0].message.content or ""
        return {"answer": answer, "citations": parse_citations(answer, contexts)}

    def stream(self, question: str, contexts: List[Dict]):
        """流式：逐块 yield 事件字典。
        {"type": "token", "text": "..."}  -> 增量文本
        {"type": "done",  "answer": ..., "citations": [...]} -> 结束与溯源
        """
        messages = build_messages(question, contexts)
        resp = self.client.chat.completions.create(
            messages=messages, stream=True, **self._common_kwargs()
        )
        acc = ""
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                acc += delta
                yield {"type": "token", "text": delta}
        yield {
            "type": "done",
            "answer": acc,
            "citations": parse_citations(acc, contexts),
        }


class MockGenerator:
    """离线演示生成器：不调用任何 LLM，直接基于检索到的上下文拼出带 [n] 引用的答案。

    用途：
      1. 在无 API Key / 无网络的環境下，仍能跑通“上传→检索→引用溯源”的完整 UI 流程；
      2. 用于测试 RAG 链路本身（检索与引用映射是否正确），而不消耗模型额度。

    生产环境请使用 Generator（真实大模型）。
    """

    def generate(self, question: str, contexts: List[Dict]) -> Dict:
        if not contexts:
            return {"answer": "（演示模式）知识库为空，请先上传文档。", "citations": []}
        lines = [f"（演示模式 · 离线生成）针对问题「{question}」，从资料中检索到以下要点：", ""]
        for i, c in enumerate(contexts, start=1):
            snippet = " ".join(c["text"].split())
            lines.append(f"[{i}] {snippet}")
        answer = "\n".join(lines)
        # 复用同一套引用解析，保证与真实模式行为一致
        return {"answer": answer, "citations": parse_citations(answer, contexts)}

    def stream(self, question: str, contexts: List[Dict]):
        result = self.generate(question, contexts)
        answer = result["answer"]
        # 模拟逐段流式输出
        step = 16
        for i in range(0, len(answer), step):
            yield {"type": "token", "text": answer[i : i + step]}
        yield {
            "type": "done",
            "answer": answer,
            "citations": result["citations"],
        }
