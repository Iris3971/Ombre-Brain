from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

import tools._runtime as rt
from tools.breath.surface import surface_default


class DisabledEmbedding:
    enabled = False


class WeightedDecay:
    is_running = True

    async def ensure_started(self):
        return None

    def calculate_score(self, metadata):
        return float(metadata.get("_score", metadata.get("importance") or 5))


class PlainBucketManager:
    def __init__(self, buckets):
        self.buckets = list(buckets)

    async def list_all(self, include_archive=False):
        return list(self.buckets)

    async def get_stats(self):
        return {"permanent_count": 0, "dynamic_count": len(self.buckets)}

    def footprint_snapshot(self):
        raise RuntimeError("no footprint in tests")


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _bucket(bucket_id, *, created, score=1.0, valence=0.5, arousal=0.3, importance=5):
    return {
        "id": bucket_id,
        "content": f"{bucket_id} 的正文。",
        "metadata": {
            "name": bucket_id, "type": "dynamic", "importance": importance,
            "domain": ["回归测试"], "created": _iso(created), "last_active": _iso(created),
            "activation_count": 3, "_score": score, "valence": valence, "arousal": arousal,
        },
    }


def _install(monkeypatch, manager, surfacing=None):
    monkeypatch.setattr(rt, "config", {"surfacing": surfacing or {}})
    monkeypatch.setattr(rt, "bucket_mgr", manager)
    monkeypatch.setattr(rt, "decay_engine", WeightedDecay())
    monkeypatch.setattr(rt, "embedding_engine", DisabledEmbedding())
    monkeypatch.setattr(rt, "logger", MagicMock())
    monkeypatch.setattr(rt, "mark_op", None)
    monkeypatch.setattr("tools.breath.surface.random.shuffle", lambda _seq: None)
    monkeypatch.setattr("tools.breath.surface.random.random", lambda: 1.0)


def _same_day_last_month():
    now = datetime.now()
    d = now - timedelta(days=30)
    while d.day != now.day:
        d -= timedelta(days=1)
    return d


@pytest.mark.asyncio
async def test_anniversary_bucket_gets_its_own_section(monkeypatch):
    now = datetime.now()
    strong = [_bucket(f"top-{i}", created=now - timedelta(days=3), score=50.0 - i) for i in range(4)]
    anniv = _bucket("anniv", created=_same_day_last_month(), score=0.2)
    _install(monkeypatch, PlainBucketManager(strong + [anniv]), surfacing={"recent_slots": 0})

    out = await surface_default(max_results=3, max_tokens=100_000, tag_filter=[])

    assert "=== 那天的今天 ===" in out
    assert "[bucket_id:anniv]" in out.split("=== 那天的今天 ===")[1]
    assert "[bucket_id:anniv]" not in out.split("=== 那天的今天 ===")[0]


@pytest.mark.asyncio
async def test_anniversary_needs_a_month_of_distance(monkeypatch):
    now = datetime.now()
    strong = [_bucket(f"top-{i}", created=now - timedelta(days=3), score=50.0 - i) for i in range(4)]
    too_young = _bucket("young", created=now - timedelta(days=7), score=0.2)
    _install(monkeypatch, PlainBucketManager(strong + [too_young]), surfacing={"recent_slots": 0})

    out = await surface_default(max_results=3, max_tokens=100_000, tag_filter=[])

    assert "=== 那天的今天 ===" not in out


@pytest.mark.asyncio
async def test_anniversary_slots_zero_disables(monkeypatch):
    now = datetime.now()
    strong = [_bucket(f"top-{i}", created=now - timedelta(days=3), score=50.0 - i) for i in range(4)]
    anniv = _bucket("anniv", created=_same_day_last_month(), score=0.2)
    _install(monkeypatch, PlainBucketManager(strong + [anniv]),
             surfacing={"recent_slots": 0, "anniversary_slots": 0})

    out = await surface_default(max_results=3, max_tokens=100_000, tag_filter=[])

    assert "=== 那天的今天 ===" not in out


@pytest.mark.asyncio
async def test_mood_reorders_but_does_not_add_or_drop(monkeypatch):
    now = datetime.now() - timedelta(days=3)
    sad = _bucket("sad", created=now, score=10.0, valence=0.1, arousal=0.6)
    glad = _bucket("glad", created=now, score=11.0, valence=0.9, arousal=0.6)
    _install(monkeypatch, PlainBucketManager([sad, glad]),
             surfacing={"recent_slots": 0, "sampling": {"enabled": False}})

    neutral = await surface_default(max_results=2, max_tokens=100_000, tag_filter=[])
    low = await surface_default(max_results=2, max_tokens=100_000, tag_filter=[],
                                mood_valence=0.1, mood_arousal=0.6)

    assert neutral.index("[bucket_id:glad]") < neutral.index("[bucket_id:sad]")
    assert low.index("[bucket_id:sad]") < low.index("[bucket_id:glad]")
    for bid in ("sad", "glad"):
        assert f"[bucket_id:{bid}]" in low


@pytest.mark.asyncio
async def test_mood_weight_zero_is_the_old_order(monkeypatch):
    now = datetime.now() - timedelta(days=3)
    sad = _bucket("sad", created=now, score=10.0, valence=0.1, arousal=0.6)
    glad = _bucket("glad", created=now, score=11.0, valence=0.9, arousal=0.6)
    _install(monkeypatch, PlainBucketManager([sad, glad]),
             surfacing={"recent_slots": 0, "mood_weight": 0})

    low = await surface_default(max_results=2, max_tokens=100_000, tag_filter=[],
                                mood_valence=0.1, mood_arousal=0.6)

    assert low.index("[bucket_id:glad]") < low.index("[bucket_id:sad]")
