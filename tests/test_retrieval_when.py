# -*- coding: utf-8 -*-
import pytest

from tools import _runtime as rt
from tools.breath import search as S


def test_when_off_by_default(monkeypatch):
    monkeypatch.setattr(rt, "config", {"retrieval": {}}, raising=False)
    assert S._when({"created": "2026-08-27T14:00:00"}) == ""


def test_when_on(monkeypatch):
    monkeypatch.setattr(rt, "config", {"retrieval": {"surface_created": True}}, raising=False)
    assert S._when({"created": "2026-08-27T14:00:00"}) == "[2026-08-27 周四] "
    assert S._when({"created": "2026-09-13"}) == "[2026-09-13 周日] "


@pytest.mark.parametrize("bad", ["", "garbage", None, "2026-13-40T00:00:00"])
def test_when_bad_created(monkeypatch, bad):
    monkeypatch.setattr(rt, "config", {"retrieval": {"surface_created": True}}, raising=False)
    assert S._when({"created": bad}) == ""
