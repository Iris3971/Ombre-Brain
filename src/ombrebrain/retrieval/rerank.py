"""重排 —— 检索命中再过一遍交叉编码器。

LoCoMo 1536 题的读数（2026-09-13）：OB 的七维融合 hit@5 0.56，前 30 条过一遍
bge-reranker-v2-m3 之后 0.77，MRR 0.43 → 0.65。一次 API 调用换来的，是检索这层最大的杠杆。

边界：
  · 只改**顺序**，不加不减：候选还是 search() 过门的那些；重排失败（网络、限流）原序返回
  · 不碰分数、不碰 touch、不碰任何元数据；rule.md 15 里"情感算分"的设计没动
  · key 复用 embedding 那把（同一家）；config.retrieval.rerank.enabled 默认关
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import httpx

logger = logging.getLogger("ombre_brain.rerank")

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_POOL = 30
_DOC_CHARS = 800
_TIMEOUT_S = 20.0


def rerank_config(config: Mapping[str, Any] | None) -> dict:
    cfg = ((config or {}).get("retrieval") or {}).get("rerank") or {}
    if not isinstance(cfg, Mapping):
        cfg = {}
    try:
        pool = int(cfg.get("pool", _DEFAULT_POOL))
    except (TypeError, ValueError):
        pool = _DEFAULT_POOL
    emb = (config or {}).get("embedding") or {}
    base = str(cfg.get("base_url") or emb.get("base_url") or "https://api.siliconflow.cn/v1").rstrip("/")
    key = str(cfg.get("api_key") or os.getenv("OMBRE_RERANK_API_KEY") or emb.get("api_key") or os.getenv("OMBRE_EMBED_API_KEY") or "")
    return {
        "enabled": str(cfg.get("enabled", False)).strip().lower() in ("1", "true", "yes", "on"),
        "pool": max(2, min(100, pool)),
        "model": str(cfg.get("model") or _DEFAULT_MODEL),
        "base_url": base,
        "api_key": key,
    }


async def _call(query: str, docs: list[str], top_n: int, cfg: Mapping[str, Any]) -> list[int]:
    """返回按相关度排好的下标。测试里把这个换掉。"""
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        r = await client.post(
            cfg["base_url"] + "/rerank",
            headers={"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"},
            json={"model": cfg["model"], "query": query[:500], "documents": docs, "top_n": min(top_n, len(docs))},
        )
        r.raise_for_status()
        data = r.json()
    return [int(x["index"]) for x in data.get("results") or []]


def _doc_text(bucket: Mapping[str, Any]) -> str:
    meta = bucket.get("metadata") or {}
    title = str(meta.get("title") or "")
    body = str(bucket.get("content") or "")
    return ((title + "\n") if title else "") + body[:_DOC_CHARS]


async def rerank_buckets(query: str, buckets: list[dict], top_n: int, config: Mapping[str, Any] | None) -> list[dict]:
    """前 pool 条重排，取 top_n；其余（pool 之外的）按原序接在后面。失败原样返回。"""
    cfg = rerank_config(config)
    if not cfg["enabled"] or not cfg["api_key"] or len(buckets) < 2 or not (query or "").strip():
        return buckets
    head, tail = buckets[:cfg["pool"]], buckets[cfg["pool"]:]
    try:
        order = await _call(query, [_doc_text(b) for b in head], max(top_n, 1), cfg)
    except Exception as exc:
        logger.warning("rerank failed, keeping original order / 重排失败，原序返回: %s: %s", type(exc).__name__, exc)
        return buckets
    seen = set()
    ranked = []
    for i in order:
        if 0 <= i < len(head) and i not in seen:
            seen.add(i)
            ranked.append(head[i])
    ranked.extend(b for i, b in enumerate(head) if i not in seen)
    return ranked + tail
