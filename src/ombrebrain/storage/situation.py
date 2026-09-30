"""情境指纹 —— 一条记忆写下时的场。

人记一件事，连着记的是当时在哪、什么天、几点、谁在、手上在干什么。之后勾起它的
也正是这些（情境依存记忆）。OB 的桶只有 created / valence / arousal，事后从正文猜
「窗边」「雨」是粗的。这里把场在**写下的那一刻**存进 frontmatter 的 `situation:`。

边界（一句话）：这块**永不进向量、永不进检索算分**——它只喂线索场的激活。
README「元数据不喂进算分」在这里仍然成立。

两种来源：
  · 调用方显式给 `context`（hold / plan 的可选参数）
  · 没给时按 config `situation.auto` 从本机的屋子状态读一份（rin 部署才有这些文件）
不管哪种，`date / hour / weekday` 缺了都按现在补。
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Mapping

KEYS = ("date", "hour", "weekday", "place", "weather", "present", "activity", "book", "mood")
_MAX_TEXT = 40
_MAX_NAME = 25
_MAX_PRESENT = 5
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f‪-‮⁦-⁩]")


def _clean(text: object) -> str:
    return _CTRL_RE.sub("", str(text or "")).replace("\n", " ").replace("\r", " ").strip()


def normalize_situation(value: object) -> dict | None:
    """白名单 + 限长。不认识的键丢掉，空的丢掉，全空返回 None。"""
    if not isinstance(value, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in KEYS:
        if key not in value:
            continue
        raw = value[key]
        if raw in (None, "", [], {}):
            continue
        if key in ("hour", "weekday"):
            try:
                iv = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if key == "hour" and 0 <= iv <= 23:
                out[key] = iv
            elif key == "weekday" and 0 <= iv <= 6:
                out[key] = iv
        elif key == "date":
            s = _clean(raw)[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                out[key] = s
        elif key == "present":
            items = raw if isinstance(raw, (list, tuple)) else [raw]
            names: list[str] = []
            for item in items:
                s = _clean(item)[:_MAX_NAME]
                if s and s not in names:
                    names.append(s)
                if len(names) >= _MAX_PRESENT:
                    break
            if names:
                out[key] = names
        elif key == "mood":
            try:
                v, a = float(raw[0]), float(raw[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if 0.0 <= v <= 1.0 and 0.0 <= a <= 1.0:
                out[key] = [round(v, 2), round(a, 2)]
        else:
            s = _clean(raw)[:_MAX_TEXT]
            if s:
                out[key] = s
    return out or None


def _fill_time(sit: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    sit.setdefault("date", now.strftime("%Y-%m-%d"))
    sit.setdefault("hour", now.hour)
    sit.setdefault("weekday", now.weekday())
    return sit


def _weather_word(text: str) -> str:
    if not text:
        return ""
    if "在下" in text or "雨" in text:
        return "雨"
    if re.search(r"3[0-9]度", text):
        return "热"
    if re.search(r"(?<!\d)(1[0-9]|[0-9])度", text):
        return "凉"
    if "晴" in text:
        return "晴"
    if "阴" in text:
        return "阴"
    return ""


def auto_situation(config: Mapping[str, Any] | None, now: datetime | None = None) -> dict | None:
    """rin 部署：从屋子状态、天气缓存、心情、最近对话里读此刻的场。

    config.situation:
      auto: true
      world_state: /path/to/world/state.json
      weather_cache: /path/to/world/.weather.json
      affect: /path/to/agent/logs/affect_state.json
      chat_history: /path/to/agent/logs/chat_history.jsonl
      owner_name: 阿明
    任何一个文件读不到都只是少一个维度，不报错。
    """
    cfg = (config or {}).get("situation") if isinstance(config, Mapping) else None
    if not isinstance(cfg, Mapping) or not cfg.get("auto"):
        return None
    now = now or datetime.now()
    sit: dict[str, Any] = {}
    try:
        st = json.loads(io.open(str(cfg.get("world_state") or ""), encoding="utf-8").read())
        if st.get("spot"):
            sit["place"] = str(st["spot"])
        if st.get("doing"):
            sit["activity"] = str(st["doing"])
        if st.get("visitor"):
            sit["present"] = [str(st["visitor"])]
        reading = st.get("reading") or {}
        if isinstance(reading, dict) and reading.get("slug"):
            sit["book"] = str(reading["slug"])
    except Exception:
        pass
    try:
        w = json.loads(io.open(str(cfg.get("weather_cache") or ""), encoding="utf-8").read()).get("text", "")
        word = _weather_word(str(w))
        if word:
            sit["weather"] = word
    except Exception:
        pass
    try:
        af = json.loads(io.open(str(cfg.get("affect") or ""), encoding="utf-8").read())
        sit["mood"] = [float(af.get("v", 0.5)), float(af.get("a", 0.3))]
    except Exception:
        pass
    owner = _clean(cfg.get("owner_name") or "")
    hist = str(cfg.get("chat_history") or "")
    if owner and hist and os.path.exists(hist) and "present" not in sit:
        try:
            cutoff = now - timedelta(minutes=90)
            with io.open(hist, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 65536))
                tail = fh.read().decode("utf-8", errors="replace")
            for line in tail.splitlines()[-80:]:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("role") != "user":
                    continue
                try:
                    t = datetime.fromisoformat(str(d.get("iso"))[:19])
                except Exception:
                    continue
                if t >= cutoff:
                    sit["present"] = [owner]
                    break
        except Exception:
            pass
    return normalize_situation(_fill_time(sit, now))


def situation_for_write(context: object, config: Mapping[str, Any] | None, now: datetime | None = None) -> dict | None:
    """hold / plan 写入时用：显式 context 优先，没有就 auto；都没有只留时间。"""
    explicit = normalize_situation(context)
    if explicit:
        return normalize_situation(_fill_time(dict(explicit), now))
    auto = auto_situation(config, now)
    if auto:
        return auto
    return normalize_situation(_fill_time({}, now))


def situation_of(metadata: Mapping[str, Any] | None) -> dict | None:
    """读回：桶元数据里的 situation（没有就 None）。"""
    if not isinstance(metadata, Mapping):
        return None
    return normalize_situation(metadata.get("situation"))


# ---------------------------------------------------------------- 前瞻记忆的线索（plan.cue）
CUE_KEYS = ("weather", "hour_from", "hour_to", "weekday", "date", "present", "text", "after_days", "place")


def normalize_cue(value: object) -> dict | None:
    """plan 的触发线索：和 situation 同构，多 `text`（他/她说到这个词）和 `after_days`。"""
    if not isinstance(value, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in CUE_KEYS:
        if key not in value or value[key] in (None, "", [], {}):
            continue
        raw = value[key]
        if key in ("hour_from", "hour_to"):
            try:
                iv = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= iv <= 23:
                out[key] = iv
        elif key == "weekday":
            try:
                iv = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= iv <= 6:
                out[key] = iv
        elif key == "after_days":
            try:
                iv = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= iv <= 365:
                out[key] = iv
        elif key == "date":
            s = _clean(raw)[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                out[key] = s
        elif key == "present":
            items = raw if isinstance(raw, (list, tuple)) else [raw]
            names = []
            for item in items:
                s = _clean(item)[:_MAX_NAME]
                if s and s not in names:
                    names.append(s)
                if len(names) >= _MAX_PRESENT:
                    break
            if names:
                out[key] = names
        else:
            s = _clean(raw)[:_MAX_TEXT]
            if s:
                out[key] = s
    return out or None


# ---------------------------------------------------------------- 要义 / 回忆版本的链接（hold.links）
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{6,40}$")
_KINDS = ("gist", "recollection")


def normalize_kind(value: object) -> str:
    s = _clean(value).lower()
    return s if s in _KINDS else ""


def normalize_links(value: object) -> dict | None:
    """gist: sources / n / span / confidence / witness；recollection: recollection_of。"""
    if not isinstance(value, Mapping):
        return None
    out: dict[str, Any] = {}
    src = value.get("sources")
    if isinstance(src, (list, tuple)):
        ids = []
        for item in src:
            s = _clean(item)
            if _ID_RE.match(s) and s not in ids:
                ids.append(s)
            if len(ids) >= 64:
                break
        if ids:
            out["sources"] = ids
    ro = _clean(value.get("recollection_of") or "")
    if _ID_RE.match(ro):
        out["recollection_of"] = ro
    for key in ("n", "witness"):
        try:
            iv = int(value.get(key))
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= iv <= 100000:
            out[key] = iv
    try:
        conf = float(value.get("confidence"))
        if 0.0 <= conf <= 1.0:
            out["confidence"] = round(conf, 2)
    except (TypeError, ValueError, OverflowError):
        pass
    span = value.get("span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        a, b = _clean(span[0])[:10], _clean(span[1])[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", b):
            out["span"] = [a, b]
    return out or None
