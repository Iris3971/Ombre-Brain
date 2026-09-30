import pytest


@pytest.mark.asyncio
async def test_rrf_fusion_unions_channels_and_ignores_importance(bucket_mgr):
    a = await bucket_mgr.create(content="我们在西湖边喝了龙井茶。", bucket_type="dynamic", importance=3)
    b = await bucket_mgr.create(content="今天很重要的一天，新的开始。", bucket_type="dynamic", importance=10)
    c = await bucket_mgr.create(content="茶叶要放在阴凉处。", bucket_type="dynamic", importance=5)
    bucket_mgr.fusion = "rrf"
    bucket_mgr.rrf_channel_k = 30
    # 向量通道只给 c 高分；BM25/字面通道会命中 a（龙井茶）；b 跟查询无关，再重要也进不来
    hits = await bucket_mgr.search("龙井茶", limit=10, vector_scores={c: 0.9, b: 0.2})
    ids = [h["id"] for h in hits]
    assert a in ids and c in ids and b not in ids
    assert ids[0] == a               # 字面 + BM25 两条通道都在第一，压过只有向量的 c
    assert all("score" in h for h in hits)


@pytest.mark.asyncio
async def test_weighted_fusion_is_default(bucket_mgr):
    assert bucket_mgr.fusion == "weighted"
