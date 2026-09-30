"""I 工具/衰减引擎传给 update 的几个键要真的落到 frontmatter（以前静默丢）。"""
from pathlib import Path

import frontmatter
import pytest


def _meta(bucket_mgr, bucket_id):
    for p in Path(bucket_mgr.base_dir).rglob("*.md"):
        post = frontmatter.load(str(p))
        if post.metadata.get("id") == bucket_id:
            return post.metadata
    raise AssertionError("bucket file not found: %s" % bucket_id)


@pytest.mark.asyncio
async def test_update_keeps_tool_fields(bucket_mgr):
    bid = await bucket_mgr.create(content="一条会被顶掉的想法")
    assert await bucket_mgr.update(bid, i_superseded_by="abc123def456", anchor_candidate=True, breath_touch_count=3)
    meta = _meta(bucket_mgr, bid)
    assert meta["i_superseded_by"] == "abc123def456"
    assert meta["anchor_candidate"] is True
    assert meta["breath_touch_count"] == 3
    assert await bucket_mgr.update(bid, i_superseded_by="")
    assert "i_superseded_by" not in _meta(bucket_mgr, bid)
