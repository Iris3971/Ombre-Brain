"""create(created=) / update(created=, last_active=) 要真的落到 frontmatter；不合法的时间串要报错而不是静默丢。"""
import frontmatter
import pytest


def _read(bucket_mgr, bucket_id):
    from pathlib import Path
    for p in Path(bucket_mgr.base_dir).rglob("*.md"):
        post = frontmatter.load(str(p))
        if post.metadata.get("id") == bucket_id:
            return post.metadata
    raise AssertionError("bucket file not found: %s" % bucket_id)


@pytest.mark.asyncio
async def test_create_accepts_created(bucket_mgr):
    bid = await bucket_mgr.create(content="回填的一条：六月十号做界面", created="2026-06-10T12:00:00+08:00")
    meta = _read(bucket_mgr, bid)
    assert str(meta["created"]).startswith("2026-06-10T12:00:00")
    assert str(meta["last_active"]).startswith("2026-06-10T12:00:00")


@pytest.mark.asyncio
async def test_update_sets_created_and_last_active(bucket_mgr):
    bid = await bucket_mgr.create(content="先按现在写，再改日期")
    ok = await bucket_mgr.update(bid, created="2026-07-18T12:00:00+08:00", last_active="2026-07-18T12:00:00+08:00")
    assert ok
    meta = _read(bucket_mgr, bid)
    assert str(meta["created"]).startswith("2026-07-18T12:00:00")
    assert str(meta["last_active"]).startswith("2026-07-18T12:00:00")


@pytest.mark.asyncio
async def test_bad_created_raises(bucket_mgr):
    with pytest.raises(ValueError):
        await bucket_mgr.create(content="坏日期", created="六月十号")
