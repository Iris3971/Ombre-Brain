# -*- coding: utf-8 -*-
import inspect
import re

from tools import breath as B


def _resolve(surfacing, query, max_results):
    """跑 dispatch 里那一小段默认条数解析（不起整个运行时）。"""
    src = inspect.getsource(B.dispatch)
    m = re.search(r"    default_results = .*?    max_results = min\(max_results, 50\)\n", src, re.S)
    assert m, "找不到条数解析段"
    body = "\n".join(l[4:] for l in m.group(0).splitlines())
    ns = {"surfacing_cfg": surfacing, "query": query, "max_results": max_results, "max_tokens": 0}
    exec(body, ns)
    return ns["max_results"]


def test_search_default_when_configured():
    cfg = {"breath_max_results": 20, "search_max_results": 10}
    assert _resolve(cfg, "八月底下雨那天", 0) == 10


def test_wake_keeps_breath_max_results():
    cfg = {"breath_max_results": 20, "search_max_results": 10}
    assert _resolve(cfg, "", 0) == 20
    assert _resolve(cfg, "   ", 0) == 20


def test_explicit_max_results_wins():
    cfg = {"breath_max_results": 20, "search_max_results": 10}
    assert _resolve(cfg, "八月底", 15) == 15


def test_unset_falls_back():
    assert _resolve({"breath_max_results": 20}, "八月底", 0) == 20
    assert _resolve({"breath_max_results": 20, "search_max_results": "bad"}, "八月底", 0) == 20
