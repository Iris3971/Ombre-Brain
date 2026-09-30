"""一跳扩展 —— 命中一个点，把它旁边的点也带上，点就成了线。

三种边，全部派生、不用 LLM、不写盘：
  · 前后：同一场里紧挨着的（metadata.seq 相邻；没有 seq 的桶按 created 相隔 ≤ 2 小时）
  · 同实体：共享 ≥ 2 个稀有词（IDF 高的 token，取自正文 + 标题）
  · 已有链接：frontmatter 的 relation_links / links.sources / links.recollection_of

用在检索的候选生成之后、重排之前：候选每条最多带 per_hit 个邻居，总共最多 max_extra 条，
邻居排在候选后面，由重排决定去留。config.retrieval.hop.enabled 默认关。

LoCoMo 上排序到头了（多跳 0.73、常识 0.39）——证据常在命中那句的上一句/下一句，或另一场里说到同一个人的那句。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}|[一-鿿]{2,4}")
_STOP = set("the and for with that this from have has had was were are you your our they them their she her his him its into about would could should there here what when where which while been being than then also just like some more most very much many over under after before only other same such into onto each every".split())
_ADJ_HOURS = 2.0
_MIN_SHARED = 2
_RARE_DF_MAX = 0.05     # 出现在 ≤5% 桶里的词才算"稀有"


def hop_config(config: Mapping[str, Any] | None) -> dict:
    cfg = ((config or {}).get("retrieval") or {}).get("hop") or {}
    if not isinstance(cfg, Mapping):
        cfg = {}

    def _int(k, d):
        try:
            return max(0, int(cfg.get(k, d)))
        except (TypeError, ValueError):
            return d
    return {
        "enabled": str(cfg.get("enabled", False)).strip().lower() in ("1", "true", "yes", "on"),
        "per_hit": _int("per_hit", 2),
        "max_extra": _int("max_extra", 20),
        "seed": _int("seed", 15),          # 只给前 seed 条命中找邻居
        "edges": tuple(str(x) for x in (cfg.get("edges") or ("adjacent", "links", "entity"))),
    }


def _tokens(bucket: Mapping[str, Any]) -> set[str]:
    meta = bucket.get("metadata") or {}
    text = str(meta.get("title") or "") + " " + str(bucket.get("content") or "")
    out = set()
    for t in _TOKEN_RE.findall(text):
        t = t.lower()
        if t in _STOP:
            continue
        out.add(t)
    return out


def _created(bucket: Mapping[str, Any]) -> datetime | None:
    raw = str((bucket.get("metadata") or {}).get("created") or "")[:19]
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


class HopIndex:
    """对一批桶建一次（每次检索的候选全集），然后 neighbors(id)。几百条毫秒级；几千条也够用。"""

    def __init__(self, buckets: list[dict]):
        self.by_id = {b["id"]: b for b in buckets}
        self.order: dict[str, list[str]] = defaultdict(list)      # session/day → ids by seq/created
        self.pos: dict[str, tuple[str, int]] = {}
        self.links: dict[str, set[str]] = defaultdict(set)
        toks: dict[str, set[str]] = {}
        df: Counter = Counter()
        for b in buckets:
            meta = b.get("metadata") or {}
            # 前后
            key = str(meta.get("session") or meta.get("grow_batch_id") or "")
            c = _created(b)
            if not key and c is not None:
                key = c.strftime("%Y-%m-%d")
            seq = meta.get("seq")
            try:
                seq_v = int(seq) if seq is not None else (int(c.timestamp()) if c else 0)
            except (TypeError, ValueError):
                seq_v = 0
            self.order[key].append((seq_v, b["id"]))
            # 已有链接
            links = meta.get("links") or {}
            if isinstance(links, dict):
                for sid in links.get("sources") or []:
                    self.links[b["id"]].add(str(sid))
                    self.links[str(sid)].add(b["id"])
                if links.get("recollection_of"):
                    self.links[b["id"]].add(str(links["recollection_of"]))
                    self.links[str(links["recollection_of"])].add(b["id"])
            for rl in meta.get("relation_links") or []:
                if isinstance(rl, dict) and rl.get("target_bucket_id") and str(rl.get("status") or "active") == "active":
                    self.links[b["id"]].add(str(rl["target_bucket_id"]))
                    self.links[str(rl["target_bucket_id"])].add(b["id"])
            toks[b["id"]] = _tokens(b)
            df.update(toks[b["id"]])
        n = max(1, len(buckets))
        # 稀有 = 出现在 ≤5% 的桶里（小库放宽到 3 条），且至少两条——只出现一次的词连不上任何人
        cap = max(3, math.ceil(_RARE_DF_MAX * n))
        rare = {t for t, k in df.items() if 2 <= k <= cap}
        self.rare_toks = {bid: (ts & rare) for bid, ts in toks.items()}
        self.inv: dict[str, set[str]] = defaultdict(set)
        for bid, ts in self.rare_toks.items():
            for t in ts:
                self.inv[t].add(bid)
        for key, lst in self.order.items():
            lst.sort()
            for i, (_, bid) in enumerate(lst):
                self.pos[bid] = (key, i)
        self.has_seq = any((b.get("metadata") or {}).get("seq") is not None for b in buckets)

    def neighbors(self, bid: str, per_hit: int, edges=("adjacent", "links", "entity")) -> list[str]:
        out: list[str] = []
        # 1. 前后
        if "adjacent" in edges and bid in self.pos:
            key, i = self.pos[bid]
            lst = self.order[key]
            me = _created(self.by_id[bid])
            for j in (i - 1, i + 1):
                if 0 <= j < len(lst):
                    other = lst[j][1]
                    if self.has_seq:
                        out.append(other)
                    else:
                        oc = _created(self.by_id[other])
                        if me and oc and abs((oc - me).total_seconds()) <= _ADJ_HOURS * 3600:
                            out.append(other)
        # 2. 已有链接
        if "links" in edges:
            out.extend(x for x in self.links.get(bid, ()) if x in self.by_id)
        # 3. 同实体：共享稀有词最多的
        if "entity" in edges:
            shared: Counter = Counter()
            for t in self.rare_toks.get(bid, ()):
                for other in self.inv.get(t, ()):
                    if other != bid:
                        shared[other] += 1
            out.extend(o for o, k in shared.most_common(per_hit) if k >= _MIN_SHARED)
        seen, uniq = set(), []
        for x in out:
            if x != bid and x not in seen and x in self.by_id:
                seen.add(x)
                uniq.append(x)
        return uniq[: max(per_hit, 2)]


def expand(hits: list[dict], pool: list[dict], config: Mapping[str, Any] | None) -> list[dict]:
    """hits 后面接上邻居（不在 hits 里的），保持 hits 原序。关着时原样返回。"""
    cfg = hop_config(config)
    if not cfg["enabled"] or not hits or not pool:
        return hits
    idx = HopIndex(pool)
    have = {h["id"] for h in hits}
    extra: list[dict] = []
    for h in hits[: cfg["seed"]]:
        for nb in idx.neighbors(h["id"], cfg["per_hit"], cfg["edges"]):
            if nb not in have:
                have.add(nb)
                b = dict(idx.by_id[nb])
                b["hop_from"] = h["id"]
                extra.append(b)
                if len(extra) >= cfg["max_extra"]:
                    return hits + extra
    return hits + extra
