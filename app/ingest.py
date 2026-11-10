"""文档解析与切分。

支持：txt / md / pdf / docx。
- 解析阶段：把文档拆成“带页码的页面片段”（PDF 按页，其余为整篇）。
- 切分阶段：递归字符切分（优先按段落/句子断句，再硬切），保留重叠，
  以兼顾语义完整与检索粒度。
"""
import os
import re
from typing import Dict, List, Tuple

from .config import settings

# 递归切分使用的分隔符优先级（先按大粒度断，再逐级细化）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", ""]


def _load_txt(path: str) -> List[Tuple[str, Dict]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [(text, {"page": 0})]


def _load_pdf(path: str) -> List[Tuple[str, Dict]]:
    from PyPDF2 import PdfReader

    reader = PdfReader(path)
    out: List[Tuple[str, Dict]] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        if txt.strip():
            out.append((txt, {"page": i}))
    return out


def _load_docx(path: str) -> List[Tuple[str, Dict]]:
    from docx import Document

    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    return [(text, {"page": 0})]


def load_text_from_file(path: str) -> List[Tuple[str, Dict]]:
    """返回 [(文本片段, 元数据), ...]，元数据含 page（0 表示整篇无页码）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md", ".text"):
        return _load_txt(path)
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    raise ValueError(f"不支持的文件类型: {ext}")


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int, seps) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sep = seps[0]
    if sep:
        # 按当前分隔符切，再贪心拼接到 chunk_size
        pieces = [p for p in text.split(sep) if p.strip()]
        chunks: List[str] = []
        current = ""
        for p in pieces:
            candidate = current + (sep if current else "") + p
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                current = p
            else:
                current = candidate
        if current:
            chunks.append(current)
        # 对仍然超长的块用更细的分隔符递归
        result: List[str] = []
        for c in chunks:
            if len(c) > chunk_size and len(seps) > 1:
                result.extend(_split_recursive(c, chunk_size, chunk_overlap, seps[1:]))
            else:
                result.append(c)
        return result
    else:
        # 最后兜底：按字符硬切，并保留重叠
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def chunk_pages(
    pages: List[Tuple[str, Dict]],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Tuple[str, Dict]]:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    out: List[Tuple[str, Dict]] = []
    for text, meta in pages:
        for piece in _split_recursive(text, chunk_size, chunk_overlap, _SEPARATORS):
            out.append((piece, dict(meta)))
    return out


def split_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """便捷方法：直接对一段文本切分（不带页码）。"""
    return _split_recursive(
        text,
        chunk_size or settings.chunk_size,
        chunk_overlap or settings.chunk_overlap,
        _SEPARATORS,
    )
