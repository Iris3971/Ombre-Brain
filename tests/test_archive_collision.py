"""归档不丢记忆回归测试（谨慎加固 D1）。

decay 只归档不删（已审计确认）。archive() 早先直接用原文件名做目标，万一 archive/
里已有同名文件，shutil.move 会覆盖 → 悄悄盖掉一条早先归档的记忆。现补了防撞名后缀。
本测试确保「归档多条相似记忆，谁也不会把谁盖掉」。
"""
import logging
import os

import frontmatter
import pytest


@pytest.mark.asyncio
async def test_archiving_similar_buckets_keeps_all(bucket_mgr):
    id1 = await bucket_mgr.create(content="第一条内容 AAA", name="重名记忆",
                                  domain=["测试"], bucket_type="dynamic", importance=3)
    id2 = await bucket_mgr.create(content="第二条内容 BBB", name="重名记忆",
                                  domain=["测试"], bucket_type="dynamic", importance=3)

    assert await bucket_mgr.archive(id1) is True
    assert await bucket_mgr.archive(id2) is True

    archived = await bucket_mgr.list_all(include_archive=True)
    blob = "\n".join(b.get("content", "") for b in archived)
    assert "第一条内容 AAA" in blob
    assert "第二条内容 BBB" in blob


@pytest.mark.asyncio
async def test_archive_collision_guard_appends_suffix(bucket_mgr, tmp_path, monkeypatch):
    """强制 dest 撞名：让 archive 目标基名固定，验证第二次归档不覆盖第一次。"""
    id1 = await bucket_mgr.create(content="原始归档内容 X", name="固定名",
                                  domain=["测试"], bucket_type="dynamic", importance=3)
    id2 = await bucket_mgr.create(content="后来归档内容 Y", name="固定名",
                                  domain=["测试"], bucket_type="dynamic", importance=3)

    # 把两个桶的文件基名强制成同一个，制造真正的 archive 撞名
    import os
    real_basename = os.path.basename
    monkeypatch.setattr(os.path, "basename",
                        lambda p: "collide.md" if str(p).endswith(".md") else real_basename(p))

    assert await bucket_mgr.archive(id1) is True
    assert await bucket_mgr.archive(id2) is True
    monkeypatch.undo()

    archived = await bucket_mgr.list_all(include_archive=True)
    blob = "\n".join(b.get("content", "") for b in archived)
    assert "原始归档内容 X" in blob   # 没被第二次归档覆盖
    assert "后来归档内容 Y" in blob


# ---- #118：撞名兜底不再二次拼 id，兜底名也被占时继续找下一个空名 ----


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _plant_copy(src, dest, body, bucket_id=None):
    post = frontmatter.load(src)
    post["type"] = "archived"
    if bucket_id is not None:
        post["id"] = bucket_id
    post.content = body
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))
    return _read_bytes(dest)


def _md_files(root):
    out = []
    for r, _, fs in os.walk(root):
        out.extend(os.path.join(r, f) for f in fs if f.endswith(".md"))
    return out


@pytest.mark.asyncio
async def test_archive_same_id_collision_no_double_id(bucket_mgr, caplog):
    bid = await bucket_mgr.create(content="活跃正文", name="名字",
                                  domain=["测试"], bucket_type="dynamic", importance=3)
    src = bucket_mgr._find_bucket_file(bid)
    adir = os.path.join(bucket_mgr.archive_dir, "测试")
    old = os.path.join(adir, os.path.basename(src))
    old_bytes = _plant_copy(src, old, "旧归档")

    with caplog.at_level(logging.WARNING):
        assert await bucket_mgr.archive(bid) is True

    new_files = [p for p in _md_files(adir) if os.path.abspath(p) != os.path.abspath(old)]
    assert len(new_files) == 1
    name = os.path.basename(new_files[0])
    assert f"_{bid}_{bid}" not in name
    assert name.endswith(f"_{bid}.md")
    assert _read_bytes(old) == old_bytes
    assert not os.path.exists(src)
    assert any(bid in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.asyncio
async def test_decay_converges_when_fallback_name_taken(bucket_mgr, decay_eng, monkeypatch, caplog):
    bid = await bucket_mgr.create(content="活跃正文", name="名字",
                                  domain=["测试"], bucket_type="dynamic", importance=3)
    src = bucket_mgr._find_bucket_file(bid)
    stem = os.path.splitext(os.path.basename(src))[0]
    adir = os.path.join(bucket_mgr.archive_dir, "测试")
    old1 = os.path.join(adir, f"{stem}.md")
    old2 = os.path.join(adir, f"{stem}_{bid}.md")
    b1 = _plant_copy(src, old1, "旧归档一")
    b2 = _plant_copy(src, old2, "旧归档二")
    monkeypatch.setattr(decay_eng, "calculate_score", lambda meta: 0.0)

    with caplog.at_level(logging.WARNING):
        r1 = await decay_eng.run_decay_cycle()
        r2 = await decay_eng.run_decay_cycle()

    assert r1["archived"] == 1
    assert r2["checked"] == 0
    assert _read_bytes(old1) == b1
    assert _read_bytes(old2) == b2
    assert not any(bid in os.path.basename(p) for p in _md_files(bucket_mgr.dynamic_dir))
    assert not any("already exists" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_archive_collision_with_other_id_keeps_both(bucket_mgr):
    bid = await bucket_mgr.create(content="活跃正文", name="名字",
                                  domain=["测试"], bucket_type="dynamic", importance=3)
    src = bucket_mgr._find_bucket_file(bid)
    adir = os.path.join(bucket_mgr.archive_dir, "测试")
    old = os.path.join(adir, os.path.basename(src))
    old_bytes = _plant_copy(src, old, "别的桶的旧归档", bucket_id="otherid00001")

    assert await bucket_mgr.archive(bid) is True

    assert _read_bytes(old) == old_bytes
    new_files = [p for p in _md_files(adir) if os.path.abspath(p) != os.path.abspath(old)]
    assert len(new_files) == 1
    assert f"_{bid}_{bid}" not in os.path.basename(new_files[0])
    assert frontmatter.load(new_files[0]).content.strip() == "活跃正文"


@pytest.mark.asyncio
async def test_archive_collision_bare_id_filename(bucket_mgr):
    bid = await bucket_mgr.create(content="活跃正文", name="名字",
                                  domain=["测试"], bucket_type="dynamic", importance=3)
    src = bucket_mgr._find_bucket_file(bid)
    bare = os.path.join(os.path.dirname(src), f"{bid}.md")
    os.replace(src, bare)
    adir = os.path.join(bucket_mgr.archive_dir, "测试")
    _plant_copy(bare, os.path.join(adir, f"{bid}.md"), "旧归档")

    assert await bucket_mgr.archive(bid) is True

    dest = os.path.join(adir, f"2_{bid}.md")
    assert os.path.isfile(dest)
    assert not os.path.exists(bare)
    found = bucket_mgr._find_bucket_file(bid)
    assert found and frontmatter.load(found).get("id") == bid


@pytest.mark.asyncio
async def test_soft_delete_collision_no_double_id(bucket_mgr):
    bid = await bucket_mgr.create(content="活跃正文", name="名字",
                                  domain=["测试"], bucket_type="dynamic", importance=3)
    src = bucket_mgr._find_bucket_file(bid)
    old = os.path.join(bucket_mgr.archive_dir, os.path.basename(src))
    old_bytes = _plant_copy(src, old, "旧的软删除")

    assert await bucket_mgr.delete(bid) is True

    names = os.listdir(bucket_mgr.archive_dir)
    assert not any(f"_{bid}_{bid}" in n for n in names)
    assert any(n.endswith(f"_{bid}.md") and n != os.path.basename(old) for n in names)
    assert _read_bytes(old) == old_bytes
