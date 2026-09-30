from ombrebrain.retrieval.hop import HopIndex, expand


def _b(i, text, session="s1", seq=None, created=None, **meta):
    m = {"session": session, "created": created or "2026-09-01T10:00:00"}
    if seq is not None:
        m["seq"] = seq
    m.update(meta)
    return {"id": "b%d" % i, "content": text, "metadata": m}


def test_neighbors_prev_next_entity_and_links():
    pool = [
        _b(0, "Caroline went to the support group on Tuesday.", seq=0),
        _b(1, "It was at the community center downtown.", seq=1),
        _b(2, "Mel bought a bike.", seq=2),
        _b(3, "Caroline said the support group helped her a lot.", session="s2", seq=0),
        _b(4, "Weather is nice.", session="s2", seq=1),
        _b(5, "unrelated", session="s3", seq=0, links={"recollection_of": "b0"}),
    ]
    idx = HopIndex(pool)
    nb = idx.neighbors("b0", per_hit=2)
    assert "b1" in nb                 # 下一句
    assert "b5" in nb                 # 回忆版本链接
    assert "b3" in nb or "b3" in idx.neighbors("b0", 4)   # 同实体：caroline / support / group
    assert "b4" not in nb


def test_expand_appends_neighbors_after_hits_and_respects_flags():
    pool = [_b(0, "Caroline went to the support group.", seq=0), _b(1, "It was downtown.", seq=1), _b(2, "Mel bought a bike.", seq=2)]
    hits = [dict(pool[0])]
    assert expand(hits, pool, {}) == hits
    out = expand(hits, pool, {"retrieval": {"hop": {"enabled": True, "per_hit": 2, "max_extra": 20}}})
    assert [b["id"] for b in out][0] == "b0" and "b1" in [b["id"] for b in out]
    assert out[1].get("hop_from") == "b0"


def test_time_adjacency_without_seq():
    pool = [
        _b(0, "早上煮了面。", session="", created="2026-09-01T08:00:00"),
        _b(1, "面糊了，倒掉了。", session="", created="2026-09-01T09:00:00"),
        _b(2, "晚上读红楼梦。", session="", created="2026-09-01T21:00:00"),
    ]
    idx = HopIndex(pool)
    assert idx.neighbors("b0", 2) == ["b1"]      # 一小时内相邻；晚上那条隔太远
