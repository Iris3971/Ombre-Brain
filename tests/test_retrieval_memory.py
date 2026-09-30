from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

import tools._runtime as rt
from decay_engine import DecayEngine
from ombrebrain.retrieval.bucket_scoring import passes_relevance_gate
from ombrebrain.storage.situation import (
    normalize_cue,
    normalize_kind,
    normalize_links,
    normalize_situation,
    situation_for_write,
)
from tools.dream.hints import build_feel_prompt, build_replay_hint
from tools.hold import attach_extras


# ---------------------------------------------------------------- situation / cue / links

def test_situation_whitelist_and_bounds():
    sit = normalize_situation({
        "place": "窗边", "weather": "雨", "hour": "22", "weekday": 6, "present": ["阿明", "阿明", "", "x" * 60],
        "mood": ["0.3", 0.9], "book": "红楼梦", "junk": "no", "date": "2026-09-13",
    })
    assert sit == {
        "date": "2026-09-13", "hour": 22, "weekday": 6, "place": "窗边", "weather": "雨",
        "present": ["阿明", "x" * 25], "book": "红楼梦", "mood": [0.3, 0.9],
    }
    assert normalize_situation({"hour": 99, "weekday": -1, "mood": [2, 0]}) is None
    assert normalize_situation("not a mapping") is None
    assert normalize_situation({"place": "a\x00b\nc"}) == {"place": "ab c"}


def test_situation_for_write_fills_time_when_nothing_given():
    now = datetime(2026, 9, 13, 21, 5)
    sit = situation_for_write(None, {}, now)
    assert sit == {"date": "2026-09-13", "hour": 21, "weekday": 6}
    sit = situation_for_write({"place": "厨房角落"}, {}, now)
    assert sit["place"] == "厨房角落" and sit["hour"] == 21


def test_cue_kind_links():
    assert normalize_cue({"weather": "晴", "hour_from": 8, "hour_to": 30, "after_days": 3, "text": "感冒"}) == {
        "weather": "晴", "hour_from": 8, "after_days": 3, "text": "感冒",
    }
    assert normalize_kind("GIST") == "gist" and normalize_kind("dynamic") == "" and normalize_kind(None) == ""
    links = normalize_links({"sources": ["abc123abc123", "abc123abc123", "bad id"], "confidence": 0.85,
                             "n": 15, "span": ["2026-08-14", "2026-09-10"], "recollection_of": "deadbeef0000"})
    assert links == {"sources": ["abc123abc123"], "recollection_of": "deadbeef0000", "n": 15,
                     "confidence": 0.85, "span": ["2026-08-14", "2026-09-10"]}
    assert normalize_links({"sources": []}) is None


# ---------------------------------------------------------------- hold 写完补元数据

class RecordingManager:
    def __init__(self):
        self.calls = []

    async def update(self, bucket_id, **kwargs):
        self.calls.append((bucket_id, kwargs))
        return True


@pytest.mark.asyncio
async def test_attach_extras_only_touches_new_buckets(monkeypatch):
    mgr = RecordingManager()
    monkeypatch.setattr(rt, "bucket_mgr", mgr)
    monkeypatch.setattr(rt, "config", {})
    monkeypatch.setattr(rt, "logger", MagicMock())

    out = await attach_extras("合并→abcdefabcdef 生活", context={"place": "窗边"})
    assert out == "合并→abcdefabcdef 生活" and mgr.calls == []

    out = await attach_extras("新建→abcdefabcdef 生活", context={"place": "窗边"}, kind="gist",
                              links={"sources": ["111111111111", "222222222222"], "confidence": 0.6})
    assert out == "新建→abcdefabcdef 生活"
    assert len(mgr.calls) == 1
    bid, kwargs = mgr.calls[0]
    assert bid == "abcdefabcdef"
    assert kwargs["situation"]["place"] == "窗边" and "hour" in kwargs["situation"]
    assert kwargs["type"] == "gist"
    assert kwargs["links"] == {"sources": ["111111111111", "222222222222"], "confidence": 0.6}


@pytest.mark.asyncio
async def test_attach_extras_reports_failure_without_hiding_body(monkeypatch):
    class Failing:
        async def update(self, bucket_id, **kwargs):
            return False

    monkeypatch.setattr(rt, "bucket_mgr", Failing())
    monkeypatch.setattr(rt, "config", {})
    monkeypatch.setattr(rt, "logger", MagicMock())
    out = await attach_extras("🫧feel→feel_202609132105_V030", context={"weather": "雨"})
    assert out.startswith("🫧feel→feel_202609132105_V030")
    assert "情境/链接没写上" in out


# ---------------------------------------------------------------- 门

def test_relevance_gate_only_opens_on_relevance():
    kw = dict(topic_threshold=0.6, bm25_threshold=0.5, semantic_threshold=0.55)
    assert passes_relevance_gate(literal_hit=True, topic=0, bm25=0, semantic=None, **kw)
    assert passes_relevance_gate(literal_hit=False, topic=0.61, bm25=0, semantic=None, **kw)
    assert passes_relevance_gate(literal_hit=False, topic=0.31, bm25=0.9, semantic=None, **kw)
    assert not passes_relevance_gate(literal_hit=False, topic=0.1, bm25=1.0, semantic=None, **kw)
    assert passes_relevance_gate(literal_hit=False, topic=0.0, bm25=0.0, semantic=0.56, **kw)
    # 新、重要、被摸过很多次——都不是相关性，门不认
    assert not passes_relevance_gate(literal_hit=False, topic=0.2, bm25=0.2, semantic=0.3, **kw)


# ---------------------------------------------------------------- ACT-R

def _meta(created_days, **extra):
    now = datetime.now()
    c = (now - timedelta(days=created_days)).strftime("%Y-%m-%dT%H:%M:%S")
    m = {"id": extra.pop("id", "m1"), "type": "dynamic", "importance": 5, "arousal": 0.3,
         "created": c, "last_active": c, "activation_count": 0}
    m.update(extra)
    return m


def test_default_model_is_untouched():
    eng = DecayEngine({"decay": {}}, bucket_mgr=None)
    assert eng.model == "ebbinghaus"
    m = _meta(10)
    expected = eng.calculate_score(m)
    eng2 = DecayEngine({"decay": {"model": "ebbinghaus"}}, bucket_mgr=None)
    assert eng2.calculate_score(m) == expected


def test_actr_power_law_and_recall_history(tmp_path):
    log = tmp_path / "recalls.jsonl"
    log.write_text("", encoding="utf-8")
    eng = DecayEngine({"decay": {"model": "actr", "recall_log": str(log)}}, bucket_mgr=None)
    fresh, month, year = eng.calculate_score(_meta(0.5)), eng.calculate_score(_meta(30)), eng.calculate_score(_meta(365))
    assert fresh > month > year > 0.3      # 幂律：一年后独一无二的事仍在阈值之上
    # 昨天想起过一次 → 30 天的那条明显抬起来
    import json
    at = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    log.write_text(json.dumps({"at": at, "recalled": [{"id": "m1"}]}) + "\n", encoding="utf-8")
    recalled = eng.calculate_score(_meta(30, id="m1"))
    assert recalled > month * 2


def test_actr_coverage_lowers_covered_events(tmp_path):
    import json
    ledger = tmp_path / "gists.json"
    ledger.write_text(json.dumps({"gists": [{"sources": ["c%d" % i for i in range(16)]}]}), encoding="utf-8")
    eng = DecayEngine({"decay": {"model": "actr", "gist_ledger": str(ledger)}}, bucket_mgr=None)
    covered = eng.calculate_score(_meta(20, id="c3"))
    unique = eng.calculate_score(_meta(20, id="u1"))
    assert unique > covered * 3     # 1/√16 = 0.25
    # type=gist 桶里的 links.sources 也算覆盖
    eng.note_gist_buckets([{"id": "g", "metadata": {"type": "gist", "links": {"sources": ["z1", "z2", "z3", "z4"]}}}])
    assert eng.calculate_score(_meta(20, id="z1")) < unique


def test_actr_keeps_short_circuits():
    eng = DecayEngine({"decay": {"model": "actr"}}, bucket_mgr=None)
    assert eng.calculate_score(_meta(5, pinned=True)) == 999.0
    assert eng.calculate_score(_meta(5, type="feel")) == 50.0


# ---------------------------------------------------------------- dream：重放 / 当时的感觉

class FakeEmbedding:
    enabled = True

    def __init__(self, vecs):
        self.vecs = vecs

    async def get_embedding(self, bid):
        return self.vecs.get(bid)

    @staticmethod
    def _cosine_similarity(a, b):
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)


def _b(bid, days_ago, vec, **meta):
    created = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S")
    m = {"name": bid, "type": "dynamic", "created": created, "last_active": created, "arousal": 0.3}
    m.update(meta)
    return {"id": bid, "content": f"{bid} 的正文", "metadata": m}


@pytest.mark.asyncio
async def test_replay_pairs_recent_with_old_not_with_recent(monkeypatch):
    recent = [_b("r1", 1, [1, 0, 0])]
    twin_recent = _b("r2", 2, [1, 0.01, 0])          # 很像但太新：不算重放
    old = _b("o1", 40, [0.9, 0.1, 0])
    far = _b("o2", 60, [0, 0, 1])
    pinned = _b("p1", 90, [1, 0, 0], pinned=True)
    monkeypatch.setattr(rt, "config", {"dream": {"replay_slots": 3}})
    monkeypatch.setattr(rt, "embedding_engine", FakeEmbedding({
        "r1": [1, 0, 0], "r2": [1, 0.01, 0], "o1": [0.9, 0.1, 0], "o2": [0, 0, 1], "p1": [1, 0, 0]}))
    monkeypatch.setattr(rt, "logger", MagicMock())
    out = await build_replay_hint(recent, recent + [twin_recent, old, far, pinned])
    assert "=== 重放" in out and "o1" in out
    assert "r2" not in out and "o2" not in out and "p1" not in out


@pytest.mark.asyncio
async def test_replay_off_by_default(monkeypatch):
    monkeypatch.setattr(rt, "config", {})
    monkeypatch.setattr(rt, "embedding_engine", FakeEmbedding({}))
    assert await build_replay_hint([_b("r1", 1, [1, 0, 0])], []) == ""


def test_feel_prompt_only_when_enabled_and_aroused(monkeypatch):
    recent = [_b("hot", 1, None, arousal=0.8), _b("cool", 1, None, arousal=0.2), _b("done", 1, None, arousal=0.9, digested=True)]
    monkeypatch.setattr(rt, "config", {})
    assert build_feel_prompt(recent) == ""
    monkeypatch.setattr(rt, "config", {"dream": {"feel_prompt": True}})
    out = build_feel_prompt(recent)
    assert "hot" in out and "cool" not in out and "done" not in out
