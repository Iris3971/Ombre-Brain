"""
========================================
tools/breath/search.py — 有 query 的检索模式
========================================

走 breath(query=...) 时进入这里。一次向量查询与 bucket_manager 的
关键词/BM25 检索融合，命中后逐字返回桶正文并套 token 预算。

关键行为：
- domain/valence/arousal 作为过滤参数传给 bucket_mgr.search
- embedding 未配置/未启用/调用失败时明确提示并继续关键词/BM25 检索
- 向量通道阈值 sim>=0.65；domain/tags/type 过滤与关键词通道完全一致
- 命中正文不经过 LLM 摘要、改写或压缩，直接返回当前存储的 content
- **3.6.0 起本分支完全只读**：命中不再 touch()。检索是「我去找它」，强化是
  「找到之后，这条确实要紧」——绑在一起的话，查得勤就等于重要。要强化某条，
  读完针对那一条 trace(bucket_id, reinforce=True)
- 检索结果不足时，从低权重旧桶里随机漂出 3-5 条「忽然想起来」
- 命中 0 条时回 webhook 报空，并给出可操作的引导文案

不做什么（边界）：
- 不返回 feel/plan/letter（专用通道有自己的入口）
- pinned/permanent 仍可检索并标为核心准则；protected 只在显式
  检索命中时返回，并标为「受保护记忆」，不进入随机漂浮
- dont_surface/digested 在真实检索命中中保留；只限制无参浮现和非命中随机漂浮

对外暴露：surface_search(query, max_results, max_tokens, domain, valence,
                          arousal, tag_filter) → str
========================================
"""

from errors import ToolInputError
import hashlib
import json
import random
from datetime import datetime

from ombrebrain.policy.surfacing import SurfacePolicyVM
from ombrebrain.retrieval.rerank import rerank_buckets, rerank_config
from ombrebrain.retrieval.hop import expand as hop_expand, hop_config
from .. import _runtime as rt
from ..plan.core import is_letter_bucket
from ombrebrain.storage.attribution import names_from_config
from ombrebrain.storage.quote_store import quotes_from_metadata, render_quotes
from ._date_range import bucket_in_created_range, parse_created_range
from ._shared import bucket_has_tags, footprint_reader
from ._verbatim import render_stored_bucket
from utils import count_tokens_approx, parse_bool

_SURFACE_POLICY = SurfacePolicyVM.default()

_VECTOR_QUERY_TOPK = 50

_SEMANTIC_DISABLED_NOTE = "[检索降级：语义索引暂不可用，本次仅使用关键词/BM25。]"
_BUDGET_NOTICE = "[token 预算不足：命中的下一条记忆未被截断或摘要，请提高 max_tokens 后重试。]"


def _can_surface_search(bucket: dict, mode: str = "search") -> bool:
    return _SURFACE_POLICY.evaluate_bucket(bucket, mode=mode).allowed


# 给机器读的那一段的定界符。**这是一个契约，不是渲染的一部分。**
#
# 调用方原先只能去解析 `[bucket_id:...]` 这类人类渲染里的标记，渲染一改就静默
# 失效，而失效方向是「该藏的漏出来」。这个块换掉那条路：标记稳定、带 schema
# 版本号、由用例钉住；改它必须先让测试变红。
#
# 默认不出现（with_ids=False），所以既有调用方的输出一个字都不变。
_ID_BLOCK_MARK = "=== ombre:result-ids ==="
_ID_BLOCK_SCHEMA = 1


def _append_id_block(
    text: str,
    bucket_ids: list[str],
    call_mode: str,
    omitted_by_policy: int,
    with_ids: bool,
) -> str:
    """把机器可读的结果清单追加到渲染文本之后。"""
    if not with_ids:
        return text
    payload = json.dumps(
        {
            "schema": _ID_BLOCK_SCHEMA,
            "mode": "automatic" if call_mode == "automatic" else "manual",
            "bucket_ids": [bid for bid in bucket_ids if bid],
            "count": len([bid for bid in bucket_ids if bid]),
            # 被 dont_surface / digested 挡掉的条数。给它是为了让「过滤有没有
            # 真的生效」可观测——静默为 0 和静默漏出来在调用方眼里长得一样。
            "omitted_by_policy": omitted_by_policy,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{text}\n\n{_ID_BLOCK_MARK}\n```json\n{payload}\n```"


def normalize_call_mode(mode: object) -> str:
    """把调用方给的意图归一成 policy 认得的模式名。

    只认 manual / automatic 两个值，其余（含空、拼错）一律当 manual——
    默认必须是「今天的行为」，一个拼错的意图不该悄悄放宽或收紧过滤。
    """
    value = str(mode or "").strip().lower()
    return "automatic" if value == "automatic" else "search"


def _is_archived(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    return (
        str(meta.get("type") or "").strip().lower() == "archived"
        or bool(meta.get("deleted_at"))
        or bool(meta.get("tombstone"))
    )


def _when(meta: dict) -> str:
    """命中头部的日期标签 `[2026-08-27 周四] `。
    读答实验：答题模型看得到每条的日期、被要求按日期推，LME 正确率 0.64→0.80。
    config retrieval.surface_created 开关，默认关；created 缺失或写坏返回空串。"""
    cfg = getattr(rt, "config", None) or {}
    flag = ((cfg.get("retrieval") or {}) if isinstance(cfg, dict) else {}).get("surface_created", False)
    if not parse_bool(flag, default=False):
        return ""
    raw = str((meta or {}).get("created") or "").strip()[:19]
    try:
        t = datetime.fromisoformat(raw)
    except Exception:
        return ""
    return "[%s 周%s] " % (t.strftime("%Y-%m-%d"), "一二三四五六日"[t.weekday()])


def _collapse_config(config) -> dict:
    cfg = ((config or {}).get("retrieval") or {}).get("collapse_covered") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        keep = max(1, int(cfg.get("keep", 2) or 2))
    except (TypeError, ValueError):
        keep = 2
    try:
        min_group = max(2, int(cfg.get("min_group", 3) or 3))
    except (TypeError, ValueError):
        min_group = 3
    return {"enabled": parse_bool(cfg.get("enabled", False), default=False), "keep": keep, "min_group": min_group}


def collapse_covered(matches: list, pool: list, config, call_mode: str = "search") -> list:
    """同款折叠——被同一条要义盖住的命中只留 keep 条，要义顶在第一条前面。

    「上次读到第几回」六条同款并排，读的人挑不出哪条是哪次；人记住的是"雨天读红楼梦总会停在某句上"
    这一句加上一两次具体的。要义 = type gist 的桶（她自己写的），links.sources 是它盖住的事件。
    只改顺序和取舍，不动分数、不动任何桶；在截断之前做，掉出去的位置由后面的候选补上。
    """
    cfg = _collapse_config(config)
    if not cfg["enabled"] or len(matches) < cfg["min_group"]:
        return matches
    gist_of: dict = {}
    for b in pool or []:
        meta = (b or {}).get("metadata") or {}
        if str(meta.get("type") or "") != "gist":
            continue
        links = meta.get("links") or {}
        srcs = links.get("sources") if isinstance(links, dict) else None
        for sid in srcs or []:
            gist_of.setdefault(str(sid), b)
    if not gist_of:
        return matches
    groups: dict = {}
    for b in matches:
        g = gist_of.get(str(b.get("id") or ""))
        if g is not None:
            groups.setdefault(str(g.get("id")), []).append(str(b.get("id")))
    big = {gid for gid, members in groups.items() if len(members) >= cfg["min_group"]}
    if not big:
        return matches
    out, seen, inserted = [], {}, set()
    for b in matches:
        bid = str(b.get("id") or "")
        if bid in inserted:
            continue  # 要义已经被顶到它盖住的第一条前面了，原位置不再重复
        g = gist_of.get(bid)
        gid = str(g.get("id")) if g is not None else ""
        if gid in big:
            n = seen.get(gid, 0)
            if n == 0 and gid not in inserted and _can_surface_search(g, call_mode):
                out.append(g)
                inserted.add(gid)
            if n >= cfg["keep"]:
                continue
            seen[gid] = n + 1
        out.append(b)
    return out


def _gist_tag(meta: dict) -> str:
    """要义头部：[要义·15 件 08-14~09-10] 。links 里没数就只标 [要义]。"""
    links = (meta or {}).get("links") or {}
    if not isinstance(links, dict):
        return "[要义] "
    n = links.get("n")
    span = links.get("span") or []
    try:
        rng = "%s~%s" % (str(span[0])[5:10], str(span[1])[5:10]) if len(span) >= 2 else ""
    except Exception:
        rng = ""
    inner = "要义" + ("·%s 件" % n if n else "") + ((" " + rng) if rng else "")
    return "[%s] " % inner


def _render_archived_hit(bucket: dict, footprint: str) -> tuple[str, int]:
    bucket_id = str(bucket.get("id") or "")
    protected_mark = (
        "🛡️ [受保护记忆] "
        if parse_bool(
            (bucket.get("metadata", {}) or {}).get("protected"),
            default=False,
        )
        else ""
    )
    header = (
        f"{protected_mark}[query 命中·已删除到档案] [bucket_id:{bucket_id}] "
        "[状态:已退出日常记忆，原文仍保留]"
    )
    rendered, _ = render_stored_bucket(bucket, header, footprint)
    rendered += (
        "\n[反思：这条记忆对当下的我有帮助吗？它值得被再次回忆吗？]"
        f'\n[若决定恢复：trace(bucket_id="{bucket_id}", restore=True)]'
    )
    from utils import count_tokens_approx
    return rendered, count_tokens_approx(rendered)


async def _semantic_scores(query: str, top_k: int) -> tuple[dict[str, float], str]:
    """Run the vector query once and return scores plus an optional notice."""
    engine = rt.embedding_engine
    if not engine or not getattr(engine, "enabled", False):
        rt.logger.warning("breath semantic search unavailable; using keyword/BM25 only")
        return {}, _SEMANTIC_DISABLED_NOTE

    try:
        strict_search = getattr(engine, "search_similar_strict", None)
        if callable(strict_search):
            pairs = await strict_search(query, top_k=top_k)
        else:
            pairs = await engine.search_similar(query, top_k=top_k)
        return {bucket_id: float(score) for bucket_id, score in pairs}, ""
    except Exception as exc:
        rt.logger.warning(
            f"breath semantic search failed; using keyword/BM25 only: "
            f"{type(exc).__name__}: {exc}"
        )
        return {}, _SEMANTIC_DISABLED_NOTE


def _semantic_diagnostics(
    query: str,
    vector_scores: dict[str, float],
    semantic_notice: str,
) -> dict:
    """收集本次检索的可重建索引状态；不记录查询原文。"""
    engine_status: dict = {}
    status_reader = getattr(rt.embedding_engine, "status", None)
    if callable(status_reader):
        try:
            raw_status = status_reader()
            if isinstance(raw_status, dict):
                engine_status = dict(raw_status)
        except Exception as exc:
            engine_status = {"status_error": f"{type(exc).__name__}: {exc}"}

    outbox_status: dict = {}
    status_reader = getattr(getattr(rt, "embedding_outbox", None), "status", None)
    if callable(status_reader):
        try:
            raw_status = status_reader()
            if isinstance(raw_status, dict):
                outbox_status = dict(raw_status)
        except Exception as exc:
            outbox_status = {"status_error": f"{type(exc).__name__}: {exc}"}

    ranked = sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "query_hash": hashlib.sha256(
            query.encode("utf-8", errors="replace")
        ).hexdigest()[:12],
        "semantic_available": not bool(semantic_notice),
        "vector_candidates": len(vector_scores),
        "vector_top": [
            {"bucket_id": bucket_id, "score": round(score, 6)}
            for bucket_id, score in ranked[:5]
        ],
        "engine": {
            key: engine_status.get(key)
            for key in (
                "enabled", "backend", "model", "vector_dim",
                "embedding_count", "status_error",
            )
            if key in engine_status
        },
        "outbox": {
            key: outbox_status.get(key)
            for key in (
                "running", "provider_ready", "pending", "retrying",
                "abandoned", "last_success", "last_error", "status_error",
            )
            if key in outbox_status
        },
    }


async def surface_search(
    query: str,
    max_results: int,
    max_tokens: int,
    domain: str,
    valence: float,
    arousal: float,
    tag_filter: list,
    date_from: str = "",
    date_to: str = "",
    with_quotes: bool = False,
    created_from: "datetime | None" = None,
    created_to: "datetime | None" = None,
    mode: str = "manual",
    with_ids: bool = False,
) -> str:
    call_mode = normalize_call_mode(mode)
    omitted_by_policy = 0
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None
    # dispatch() 已经解析并校验过一次，直接用；单独调用本函数（测试、旧调用方）
    # 时才现场解析。两条路解析的是同一个 _date_range。
    if created_from is None and created_to is None:
        created_from, created_to = parse_created_range(date_from, date_to)

    _footprint = footprint_reader()

    # A full bucket id is an address, not a semantic query.  Resolve it before
    # embedding/BM25 work so callers can reliably read the on-disk source text
    # immediately before trace(content=...) without an LLM or derived index in
    # the path.  Archived/deleted and dedicated bucket types keep the same
    # visibility boundary as ordinary search.
    exact_id = query.strip()
    try:
        exact_reader = getattr(rt.bucket_mgr, "get_including_archive", None)
        exact_bucket = (
            await exact_reader(exact_id)
            if callable(exact_reader)
            else await rt.bucket_mgr.get(exact_id)
        )
    except Exception as exc:
        rt.logger.warning(
            f"breath exact bucket lookup failed; continuing with search: "
            f"{type(exc).__name__}: {exc}"
        )
        exact_bucket = None
    if exact_bucket:
        if is_letter_bucket(exact_bucket):
            raise ToolInputError("Letter 不通过普通 breath 检索返回；请使用 letter_read。")
        meta = exact_bucket.get("metadata", {}) or {}
        is_archived = _is_archived(exact_bucket)
        archived_original_kind = (
            _footprint.original_kind(exact_id, meta) if is_archived else "dynamic"
        )
        if (
            is_archived
            and archived_original_kind not in ("feel", "plan", "letter")
            and bucket_has_tags(meta, tag_filter)
            and bucket_in_created_range(exact_bucket, created_from, created_to)
        ):
            rendered, entry_tokens = _render_archived_hit(
                exact_bucket, _footprint(exact_bucket)
            )
            return rendered if entry_tokens <= max_tokens else _BUDGET_NOTICE
        if (
            not is_archived
            and meta.get("type") not in ("feel", "plan", "letter")
            and _can_surface_search(exact_bucket)
            and bucket_has_tags(meta, tag_filter)
            and bucket_in_created_range(exact_bucket, created_from, created_to)
        ):
            protected_mark = (
                "🛡️ [受保护记忆] "
                if parse_bool(meta.get("protected"), default=False)
                else ""
            )
            rendered, entry_tokens = render_stored_bucket(
                exact_bucket,
                f"{protected_mark}[exact_bucket_id:true] "
                f"[bucket_id:{exact_bucket['id']}]",
                _footprint(exact_bucket),
            )
            if entry_tokens > max_tokens:
                return _BUDGET_NOTICE
            # 3.6.0：按完整 ID 取桶同样只读。这条路径存在的理由就是「改之前先读一眼
            # 磁盘上的原文」（见上方注释），那是最不该被算作强化的一次读取——
            # 越是要改它，越会先读它，读一次涨一次权重是纯粹的自我实现。
            if rt.fire_webhook:
                await rt.fire_webhook(
                    "breath",
                    {"mode": "exact_id", "matches": 1, "chars": len(rendered)},
                )
            return rendered

    vector_scores, semantic_notice = await _semantic_scores(
        query, top_k=max(max_results, _VECTOR_QUERY_TOPK)
    )
    semantic_diag = _semantic_diagnostics(query, vector_scores, semantic_notice)
    rt.logger.info("op=breath_search phase=semantic diagnostics=%s", semantic_diag)

    # 重排开着时候选池要比 max_results 大，重排完再截
    _rr = rerank_config(getattr(rt, "config", None))
    all_for_hop: list = []
    if hop_config(getattr(rt, "config", None))["enabled"] or _collapse_config(getattr(rt, "config", None))["enabled"]:
        try:
            all_for_hop = await rt.bucket_mgr.list_all(include_archive=False)
        except Exception as exc:
            rt.logger.warning(f"hop list_all failed: {exc}")
    search_kwargs = {
        "limit": max(max_results, 20, _rr["pool"] if _rr["enabled"] else 0),
        "domain_filter": domain_filter,
        "query_valence": q_valence,
        "query_arousal": q_arousal,
        "vector_scores": vector_scores,
    }
    try:
        try:
            matches = await rt.bucket_mgr.search(
                query, include_archive=True, **search_kwargs
            )
        except TypeError as exc:
            # Lightweight third-party/test managers may predate the archive
            # option.  Preserve active search there; production supports it.
            if "include_archive" not in str(exc):
                raise
            matches = await rt.bucket_mgr.search(query, **search_kwargs)
    except Exception as e:
        rt.logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    # 自动召回（agent 每轮注入）时，准则桶不当检索命中——它们每轮睁眼 breath() 已经在场，
    # 这里再出现一次是重复，还会顶掉真正相关的事。显式 query（manual）照旧能搜到。
    _hide_core = call_mode == "automatic" and parse_bool(
        ((getattr(rt, "config", None) or {}).get("matching") or {}).get("automatic_hides_core"), default=False)
    eligible_matches = []
    for bucket in matches:
        meta = bucket.get("metadata", {}) or {}
        if is_letter_bucket(bucket):
            continue
        if _hide_core and (parse_bool(meta.get("pinned"), default=False) or str(meta.get("type") or "") == "permanent"):
            continue
        if _is_archived(bucket):
            original_kind = _footprint.original_kind(
                str(bucket.get("id") or ""), meta
            )
            if original_kind in ("feel", "plan", "letter"):
                continue
        elif meta.get("type") in ("feel", "plan", "letter"):
            continue
        elif not _can_surface_search(bucket, call_mode):
            # 分开数：被「别主动拿给我」这类标记挡掉的条数要报给调用方。
            # 调用方现在只能靠解析渲染文本自己剔，而那条路失效的方向是
            # 「该藏的漏出来」——一个明确的计数能让静默失效变成可观测的。
            decision = _SURFACE_POLICY.evaluate_bucket(bucket, mode=call_mode)
            if {"dont_surface", "digested"} & set(decision.reasons):
                omitted_by_policy += 1
            continue
        eligible_matches.append(bucket)
    matches = eligible_matches
    matches = [b for b in matches if bucket_has_tags(b["metadata"], tag_filter)]
    matches = [
        b for b in matches
        if bucket_in_created_range(b, created_from, created_to)
    ]
    # 一跳扩展——命中旁边的（前后 / 同实体 / 已有链接）也进池子，交给重排决定去留
    if hop_config(getattr(rt, "config", None))["enabled"] and matches:
        try:
            _pool = [b for b in all_for_hop if _can_surface_search(b, call_mode) and not is_letter_bucket(b)
                     and (b.get("metadata") or {}).get("type") not in ("feel", "plan", "letter")]
            matches = hop_expand(matches, _pool, getattr(rt, "config", None))
        except Exception as exc:
            rt.logger.warning(f"hop expand failed / 一跳扩展失败: {exc}")
    if _rr["enabled"] and len(matches) > 1:
        matches = await rerank_buckets(query, matches, max_results, getattr(rt, "config", None))
    # 同款折叠（要义盖住的只留 keep 条，要义顶前面），截断之前做
    try:
        _before = len(matches)
        matches = collapse_covered(matches, all_for_hop, getattr(rt, "config", None), call_mode)
        rt.logger.info("op=breath_search phase=collapse pool=%d before=%d after=%d", len(all_for_hop), _before, len(matches))
    except Exception as exc:
        rt.logger.warning(f"collapse covered failed / 同款折叠失败: {exc}")
    matches = matches[:max_results]
    rt.logger.info(
        "op=breath_search phase=ranking query_hash=%s matches=%s ids=%s",
        semantic_diag["query_hash"],
        len(matches),
        [bucket.get("id") for bucket in matches],
    )

    results = []
    token_used = 0
    budget_blocked = False
    for bucket in matches:
        meta = bucket["metadata"]
        bucket_id = bucket["id"]
        if _is_archived(bucket):
            rendered, entry_tokens = _render_archived_hit(bucket, _footprint(bucket))
        elif parse_bool(meta.get("protected"), default=False):
            header = _when(meta) + f"🛡️ [受保护记忆] [bucket_id:{bucket_id}]"
            rendered, entry_tokens = render_stored_bucket(
                bucket, header, _footprint(bucket)
            )
        elif meta.get("pinned") or meta.get("type") == "permanent":
            header = _when(meta) + f"📌 [核心准则] [bucket_id:{bucket_id}]"
            rendered, entry_tokens = render_stored_bucket(
                bucket, header, _footprint(bucket)
            )
        elif str(meta.get("type") or "") == "gist":
            header = _when(meta) + _gist_tag(meta) + f"[bucket_id:{bucket_id}]"
            rendered, entry_tokens = render_stored_bucket(
                bucket, header, _footprint(bucket)
            )
        elif bucket.get("vector_match"):
            header = _when(meta) + f"[语义关联] [bucket_id:{bucket_id}]"
            rendered, entry_tokens = render_stored_bucket(
                bucket, header, _footprint(bucket)
            )
        else:
            header = _when(meta) + f"[bucket_id:{bucket_id}]"
            rendered, entry_tokens = render_stored_bucket(
                bucket, header, _footprint(bucket)
            )
        # 引语：唯一的出口就在这里。默认不附加——每一条浮现路径
        # （breath / dream / catalog / feel）走的都是白名单渲染，读不到这个字段。
        # 只有我在这次调用里明确说了「我想知道当时是怎么说的」，它才出现。
        # 附加之后必须重算 token：预算是按实际返回的字数算的，不是按正文。
        if with_quotes:
            quote_block = render_quotes(
                quotes_from_metadata(meta),
                **names_from_config(getattr(rt, "config", None)),
            )
            if quote_block:
                rendered = f"{rendered}\n{quote_block}"
                entry_tokens = count_tokens_approx(rendered)
        if token_used + entry_tokens > max_tokens:
            budget_blocked = True
            break
        results.append(rendered)
        token_used += entry_tokens

    # --- 3.6.0：检索不再强化。retrieval ≠ reinforcement ---
    #
    # 原来这里对**每一条命中**都 touch()：刷新 last_active、activation_count +1。
    # 于是「被频繁查询」等价于「更重要」——我为了核对事实、debug、反复找同一件事
    # 而读一条记忆，读着读着它的权重就爬到了最高。实测最高积到 51，新桶再也排不进
    # 浮现区。这不是记忆在变重要，是我查得勤。
    #
    # 检索是「我去找它」，强化是「找到之后，这条确实要紧」。前者是我的动作，
    # 后者是关于这条记忆的判断，只有读完才知道。把两者绑在一起，等于让读取行为
    # 自己给自己投票。
    #
    # 所以这条路径现在是**只读的**，一条都不 touch。要强化某条，读完之后针对
    # 那一条显式说：trace(bucket_id, reinforce=True)。是「那一条」而不是「这批
    # 候选」——检索命中里绝大多数只是路过。

    # 检索命中不足时保留设计上的自由联想；用独立分区明确标记，
    # 避免调用方把随机旧桶误当成查询命中。
    if not budget_blocked and len(matches) < min(3, max_results):
        try:
            all_buckets = await rt.bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and _SURFACE_POLICY.evaluate_bucket(
                    b, mode="spontaneous"
                ).allowed
                and not is_letter_bucket(b)
                and b["metadata"].get("type") not in ("feel", "plan", "letter")
                and not parse_bool(
                    b["metadata"].get("protected"), default=False
                )
                # fork-only：技术备忘不参与「忽然想起来」的随机浮现。
                # 正常检索照旧命中 —— 这是我们相对上游唯一的改动。
                and not parse_bool(
                    b["metadata"].get("no_drift"), default=False
                )
                and rt.decay_engine.calculate_score(b["metadata"]) < 2.0
                and bucket_in_created_range(b, created_from, created_to)
            ]
            remaining_slots = max(0, max_results - len(matches))
            if low_weight and remaining_slots:
                drifted = random.sample(
                    low_weight,
                    min(random.randint(3, 5), len(low_weight), remaining_slots),
                )
                drift_results = []
                for b in drifted:
                    rendered, entry_tokens = render_stored_bucket(
                        b,
                        _when(b.get("metadata") or {}) + f"[联想浮现·非检索命中] [bucket_id:{b['id']}]",
                        _footprint(b),
                    )
                    if token_used + entry_tokens > max_tokens:
                        budget_blocked = True
                        break
                    drift_results.append(rendered)
                    token_used += entry_tokens
                if drift_results:
                    results.append("=== 忽然想起来（非检索命中） ===\n" + "\n---\n".join(drift_results))
        except Exception as e:
            rt.logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        if budget_blocked:
            text = f"{semantic_notice}\n{_BUDGET_NOTICE}" if semantic_notice else _BUDGET_NOTICE
            return _append_id_block(text, [], call_mode, omitted_by_policy, with_ids)
        if rt.fire_webhook:
            await rt.fire_webhook("breath", {"mode": "empty", "matches": 0})
        empty_text = (
            f"没有匹配到「{query}」相关的记忆。\n"
            "可以换个关键词试试，或用 breath() 看当下权重池；feel 用 breath_advanced(domain=\"feel\")，信件用 letter_read。"
        )
        if semantic_notice:
            empty_text = f"{semantic_notice}\n{empty_text}"
        return _append_id_block(empty_text, [], call_mode, omitted_by_policy, with_ids)

    final_text = "\n---\n".join(results)
    notices = []
    if semantic_notice:
        notices.append(semantic_notice)
    if budget_blocked:
        notices.append(_BUDGET_NOTICE)
    if notices:
        final_text = "\n".join(notices + [final_text])
    if rt.fire_webhook:
        await rt.fire_webhook("breath", {"mode": "ok", "matches": len(matches), "chars": len(final_text)})
    return _append_id_block(
        final_text,
        [str(b.get("id") or "") for b in matches],
        call_mode,
        omitted_by_policy,
        with_ids,
    )
