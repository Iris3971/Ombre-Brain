# -*- coding: utf-8 -*-
from tools.breath import search as S


def _b(i, t="dynamic", links=None):
    m = {"type": t, "created": "2026-08-%02dT10:00:00" % (10 + i % 20)}
    if links is not None:
        m["links"] = links
    return {"id": "b%d" % i, "content": "x%d" % i, "metadata": m}


def _pool_and_matches():
    gist = _b(99, "gist", {"sources": ["b1", "b2", "b3", "b4", "b5"], "n": 5, "span": ["2026-08-14", "2026-09-10"]})
    others = [_b(10), _b(11)]
    members = [_b(i) for i in range(1, 6)]
    pool = members + others + [gist]
    matches = [members[0], others[0], members[1], members[2], members[3], others[1], members[4]]
    return gist, matches, pool


def test_disabled_is_identity():
    _, matches, pool = _pool_and_matches()
    assert S.collapse_covered(list(matches), pool, {"retrieval": {}}) == matches


def test_collapse_keeps_two_and_inserts_gist_before_first(monkeypatch):
    monkeypatch.setattr(S, "_can_surface_search", lambda b, mode="search": True)
    gist, matches, pool = _pool_and_matches()
    out = S.collapse_covered(list(matches), pool, {"retrieval": {"collapse_covered": {"enabled": True, "keep": 2, "min_group": 3}}})
    assert [b["id"] for b in out] == ["b99", "b1", "b10", "b2", "b11"]


def test_small_group_untouched(monkeypatch):
    monkeypatch.setattr(S, "_can_surface_search", lambda b, mode="search": True)
    gist, matches, pool = _pool_and_matches()
    two = [matches[0], matches[1], matches[2]]  # b1, b10, b2 —— 同款只有两条
    out = S.collapse_covered(list(two), pool, {"retrieval": {"collapse_covered": {"enabled": True, "keep": 2, "min_group": 3}}})
    assert [b["id"] for b in out] == ["b1", "b10", "b2"]


def test_gist_already_present_not_duplicated(monkeypatch):
    monkeypatch.setattr(S, "_can_surface_search", lambda b, mode="search": True)
    gist, matches, pool = _pool_and_matches()
    with_gist = [matches[0], gist] + matches[1:]
    out = S.collapse_covered(list(with_gist), pool, {"retrieval": {"collapse_covered": {"enabled": True}}})
    ids = [b["id"] for b in out]
    assert ids.count("b99") == 1 and ids == ["b99", "b1", "b10", "b2", "b11"]  # 要义被顶到第一条同款前面，原位置不重复


def test_gist_tag():
    assert S._gist_tag({"links": {"n": 15, "span": ["2026-08-14", "2026-09-10"]}}) == "[要义·15 件 08-14~09-10] "
    assert S._gist_tag({"links": {}}) == "[要义] "
    assert S._gist_tag({}) == "[要义] "
