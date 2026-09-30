import pytest

from ombrebrain.retrieval import rerank as R


def _b(i):
    return {"id": "b%d" % i, "content": "正文 %d" % i, "metadata": {"title": "t%d" % i}}


@pytest.mark.asyncio
async def test_rerank_reorders_pool_and_keeps_tail(monkeypatch):
    async def fake_call(query, docs, top_n, cfg):
        assert len(docs) == 3 and docs[0].startswith("t0")
        return [2, 0, 1]
    monkeypatch.setattr(R, "_call", fake_call)
    cfg = {"retrieval": {"rerank": {"enabled": True, "pool": 3, "api_key": "k"}}}
    out = await R.rerank_buckets("q", [_b(i) for i in range(5)], 10, cfg)
    assert [b["id"] for b in out] == ["b2", "b0", "b1", "b3", "b4"]


@pytest.mark.asyncio
async def test_rerank_off_or_failing_keeps_order(monkeypatch):
    buckets = [_b(i) for i in range(4)]
    assert await R.rerank_buckets("q", buckets, 10, {}) == buckets

    async def boom(query, docs, top_n, cfg):
        raise RuntimeError("网断了")
    monkeypatch.setattr(R, "_call", boom)
    cfg = {"retrieval": {"rerank": {"enabled": True, "api_key": "k"}}}
    assert [b["id"] for b in await R.rerank_buckets("q", buckets, 10, cfg)] == ["b0", "b1", "b2", "b3"]


def test_rerank_config_defaults_and_key_fallback(monkeypatch):
    monkeypatch.delenv("OMBRE_RERANK_API_KEY", raising=False)
    monkeypatch.setenv("OMBRE_EMBED_API_KEY", "emb-key")
    cfg = R.rerank_config({"embedding": {"base_url": "https://x/v1/"}})
    assert cfg["enabled"] is False and cfg["pool"] == 30 and cfg["base_url"] == "https://x/v1" and cfg["api_key"] == "emb-key"
    assert R.rerank_config({"retrieval": {"rerank": {"enabled": "true", "pool": 500}}})["pool"] == 100
