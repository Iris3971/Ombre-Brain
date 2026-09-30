import pytest


@pytest.mark.asyncio
async def test_update_persists_situation_links_cue_and_gist_type(bucket_mgr):
    bid = await bucket_mgr.create(content="雨夜在厨房角落写下的一条。", bucket_type="dynamic", importance=4)
    ok = await bucket_mgr.update(
        bid,
        situation={"place": "厨房角落", "weather": "雨", "present": ["阿明"], "hour": 21, "junk": "x"},
        links={"sources": ["abcdefabcdef"], "confidence": 0.7},
        cue={"weather": "晴", "hour_from": 9},
    )
    assert ok
    b = await bucket_mgr.get(bid)
    meta = b["metadata"]
    assert meta["situation"] == {"place": "厨房角落", "weather": "雨", "present": ["阿明"], "hour": 21}
    assert meta["links"] == {"sources": ["abcdefabcdef"], "confidence": 0.7}
    assert meta["cue"] == {"weather": "晴", "hour_from": 9}

    # 空值 = 清掉；type 可以改成 gist / recollection（fork 加进 _EDITABLE_BUCKET_TYPES）
    assert await bucket_mgr.update(bid, cue={}, type="gist")
    b = await bucket_mgr.get(bid)
    assert "cue" not in b["metadata"] and b["metadata"]["type"] == "gist"
    assert await bucket_mgr.update(bid, type="recollection")
    assert (await bucket_mgr.get(bid))["metadata"]["type"] == "recollection"
