"""导入前备份必须覆盖导入会写的每一类文件。

上层在备份失败时会中止导入，理由写着「为避免覆盖后无法找回记忆」——
也就是说它把这个 zip 当**完整回滚点**用。而原先它只打包 `*.md`：
`_sources/` 下的原文证据、`.you` / `.them` 两个模块库都会被导入覆盖，
却一个都没备份。

一个不完整的回滚点比没有回滚点更危险：人会依着它去做不可逆的操作。
"""

import os
import zipfile

import pytest

import web.github as github_web
from web.github import (
    _pre_import_backup,
    _rollback_from_backup,
    _safe_rollback_member,
    _should_back_up_before_import,
)


# 导入实际会写的四类路径，来源见 github_sync 的安装循环与 _MODULE_SNAPSHOT_PATHS
被导入覆盖的 = [
    "2026/08/bucket_abc.md",
    "_sources/deadbeef.source",
    ".you/you.sqlite3",
    ".them/them.sqlite3",
    ".you/you.sqlite3-wal",
    ".them/them.sqlite3-shm",
]


@pytest.mark.parametrize("相对路径", 被导入覆盖的)
def test_导入会覆盖的都在备份范围里(相对路径):
    assert _should_back_up_before_import(相对路径), (
        f"{相对路径} 会被导入覆盖，却不在备份范围里——"
        "回滚点缺了它，导入失败后这份数据就找不回来了"
    )


@pytest.mark.parametrize("相对路径", [
    ".import_backups/pre_import_x.zip",   # 备份自己，不能套娃
    "embeddings.db",                      # 导入不写它，靠「重算所有向量」恢复
    "notes.txt",
])
def test_导入不碰的不必备份(相对路径):
    assert not _should_back_up_before_import(相对路径)


def test_真打出来的zip里四类文件都在(tmp_path):
    """不只测判定函数，测真跑一遍 zip 里到底有什么。"""
    for 相对路径 in 被导入覆盖的:
        目标 = tmp_path / 相对路径
        目标.parent.mkdir(parents=True, exist_ok=True)
        目标.write_bytes(b"x")
    (tmp_path / "embeddings.db").write_bytes(b"x")

    zip路径 = _pre_import_backup(str(tmp_path))
    assert zip路径, "备份没生成"
    with zipfile.ZipFile(zip路径) as z:
        打包了 = {n.replace(os.sep, "/") for n in z.namelist()}

    for 相对路径 in 被导入覆盖的:
        assert 相对路径 in 打包了, f"{相对路径} 没进备份"
    assert "embeddings.db" not in 打包了


# --- 回滚 ---


def test_失败后能把被覆盖的文件还原回去(tmp_path):
    """导入是一个文件一个文件装的，装到一半失败，前面的已经落盘。

    没有回滚时，本地就停在「一半远端、一半本地」的混合状态，
    而调用方只看到一句「失败」，很容易以为什么都没发生。
    """
    原始 = {
        "2026/08/a.md": b"local-a",
        "_sources/ref1.source": b"local-source",
        ".you/you.sqlite3": b"local-you-db",
    }
    for 相对路径, 内容 in 原始.items():
        目标 = tmp_path / 相对路径
        目标.parent.mkdir(parents=True, exist_ok=True)
        目标.write_bytes(内容)

    备份 = _pre_import_backup(str(tmp_path))
    assert 备份

    # 模拟导入装到一半：前两个被远端内容覆盖，第三个还没轮到
    (tmp_path / "2026/08/a.md").write_bytes(b"REMOTE-a")
    (tmp_path / "_sources/ref1.source").write_bytes(b"REMOTE-source")
    # 导入还新增了一个本地原本没有的文件
    新增 = tmp_path / "2026/08/from_remote.md"
    新增.write_bytes(b"REMOTE-new")

    结果 = _rollback_from_backup(str(tmp_path), 备份)

    assert 结果["ok"], 结果
    for 相对路径, 内容 in 原始.items():
        assert (tmp_path / 相对路径).read_bytes() == 内容, f"{相对路径} 没还原"
    # 新增的不删——删错一条记忆不可逆，宁可留下多余的
    assert 新增.exists()


def test_备份读不开时如实报错而不是假装还原了(tmp_path):
    结果 = _rollback_from_backup(str(tmp_path), str(tmp_path / "不存在.zip"))
    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert "备份读不开" in 结果["error"]


def test_备份里的越界路径不会被写出去(tmp_path):
    """备份是本地生成的，但它也可能被人动过手脚。"""
    坏包 = tmp_path / "bad.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("../../escaped.md", "x")
    结果 = _rollback_from_backup(str(tmp_path), str(坏包))
    assert 结果["ok"] is False
    assert not (tmp_path.parent.parent / "escaped.md").exists()


@pytest.fixture
def windows_语义(monkeypatch):
    monkeypatch.setattr(github_web, "_windows_name_rules", lambda: True)


@pytest.fixture
def posix_语义(monkeypatch):
    monkeypatch.setattr(github_web, "_windows_name_rules", lambda: False)


@pytest.mark.parametrize("name", ["safe.md:stream", "CON.txt", "trail. "])
def test_备份拒绝_windows_文件系统别名(tmp_path, name, windows_语义):
    坏包 = tmp_path / "alias.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr(name, "x")

    结果 = _rollback_from_backup(str(tmp_path), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0


def test_备份拒绝仅大小写不同的重复路径且不部分落盘(tmp_path, windows_语义):
    坏包 = tmp_path / "collision.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("safe/a.md", "first")
        z.writestr("safe/A.md", "second")

    结果 = _rollback_from_backup(str(tmp_path), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert not (tmp_path / "safe" / "a.md").exists()
    assert not (tmp_path / "safe" / "A.md").exists()


def test_备份拒绝文件与目录前缀冲突(tmp_path):
    坏包 = tmp_path / "prefix-collision.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("safe", "file")
        z.writestr("safe/child.md", "child")

    结果 = _rollback_from_backup(str(tmp_path), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert not (tmp_path / "safe").exists()


def test_备份不会跟随目标符号链接写出_vault(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_bytes(b"outside-original")
    target = tmp_path / "linked.md"
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("当前宿主不允许创建文件符号链接")
    坏包 = tmp_path / "symlink.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("linked.md", "backup-content")

    结果 = _rollback_from_backup(str(tmp_path), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert outside.read_bytes() == b"outside-original"


def test_备份不会跟随中间目录符号链接(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside.mkdir()
    linked = tmp_path / "linked-dir"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前宿主不允许创建目录符号链接")
    坏包 = tmp_path / "dir-symlink.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("linked-dir/escaped.md", "backup-content")

    结果 = _rollback_from_backup(str(tmp_path), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert not (outside / "escaped.md").exists()


def test_后缀不安全时前缀安全成员也不先落盘(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-late-outside.md"
    outside.write_bytes(b"outside-original")
    target = tmp_path / "late-linked.md"
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("当前宿主不允许创建文件符号链接")
    坏包 = tmp_path / "late-symlink.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("safe.md", "must-not-land")
        z.writestr("late-linked.md", "backup-content")

    结果 = _rollback_from_backup(str(tmp_path), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert not (tmp_path / "safe.md").exists()
    assert outside.read_bytes() == b"outside-original"


def _目录链接(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name != "nt":
        pytest.skip("当前宿主不允许创建目录符号链接")
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def test_恢复根目录本身是链接时照常还原(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.md").write_text("old", encoding="utf-8")
    root = tmp_path / "linked-root"
    _目录链接(root, real)
    备份 = tmp_path / "backup.zip"
    with zipfile.ZipFile(备份, "w") as z:
        z.writestr("a.md", "backup")

    结果 = _rollback_from_backup(str(root), str(备份))

    assert 结果["ok"] is True
    assert 结果["restored"] == 1
    assert (real / "a.md").read_text(encoding="utf-8") == "backup"


def test_备份不会穿过根目录以下的目录链接(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _目录链接(root / "jd", outside)
    坏包 = tmp_path / "junction.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("jd/escaped.md", "backup-content")

    结果 = _rollback_from_backup(str(root), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert not (outside / "escaped.md").exists()


@pytest.mark.parametrize(
    "name",
    [
        "dynamic/aux/2026-09-29 12-00-00_abc.md",
        "dynamic/Con/x.md",
        "dynamic/COM1.md",
        "dynamic/trail /x.md",
        "dynamic/dot./x.md",
        "dynamic/a:b.md",
    ],
)
def test_posix_语义下_windows_专属名字是合法成员(name, posix_语义):
    assert _safe_rollback_member(
        name, windows_rules=github_web._windows_name_rules()
    ) == tuple(name.split("/"))


@pytest.mark.parametrize("name", ["../x.md", "/abs.md", "a/../../x.md", "a\x00b.md", ""])
def test_posix_语义下越界照样拒绝(name, posix_语义):
    with pytest.raises(ValueError):
        _safe_rollback_member(name, windows_rules=False)


def test_posix_语义下设备名和结尾空格不再整包拒绝(tmp_path, posix_语义, monkeypatch):
    # 本机可能是 Windows，aux 这种名字真写会出错；只替换落盘两步，量的是预检放不放行。
    落盘: list[str] = []

    def _假目标(root, parts):
        return "/".join(parts)

    def _假写入(zf, info, target):
        落盘.append(target)

    monkeypatch.setattr(github_web, "_prepare_rollback_target", _假目标)
    monkeypatch.setattr(github_web, "_restore_zip_member", _假写入)
    成员 = [
        "a.md",
        "dynamic/aux/2026-09-29 12-00-00_abc.md",
        "dynamic/很长的域名截断后结尾是空格 /x.md",
        "dynamic/NUL.md",
    ]
    备份 = tmp_path / "backup.zip"
    with zipfile.ZipFile(备份, "w") as z:
        for name in 成员:
            z.writestr(name, "x")

    结果 = _rollback_from_backup(str(tmp_path / "root"), str(备份))

    assert 结果["ok"] is True
    assert 结果["restored"] == len(成员)
    assert 落盘 == 成员


def test_posix_语义下越界成员仍然整包拒绝(tmp_path, posix_语义):
    (tmp_path / "a.md").write_text("new", encoding="utf-8")
    坏包 = tmp_path / "bad.zip"
    with zipfile.ZipFile(坏包, "w") as z:
        z.writestr("a.md", "old")
        z.writestr("dynamic/aux/x.md", "x")
        z.writestr("../escaped.md", "x")

    结果 = _rollback_from_backup(str(tmp_path), str(坏包))

    assert 结果["ok"] is False
    assert 结果["restored"] == 0
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == "new"


@pytest.mark.skipif(os.name == "nt", reason="Windows 上 aux 是设备名，写不出来")
def test_linux_上域名叫_aux_的备份照常还原(tmp_path):
    root = tmp_path / "buckets"
    (root / "dynamic" / "aux").mkdir(parents=True)
    (root / "a.md").write_text("imported", encoding="utf-8")
    (root / "dynamic" / "aux" / "2026-09-29 12-00-00_abc.md").write_text(
        "imported", encoding="utf-8"
    )
    备份 = tmp_path / "backup.zip"
    with zipfile.ZipFile(备份, "w") as z:
        z.writestr("a.md", "old")
        z.writestr("dynamic/aux/2026-09-29 12-00-00_abc.md", "old-aux")
        z.writestr("dynamic/trail /x.md", "old-trail")

    结果 = _rollback_from_backup(str(root), str(备份))

    assert 结果["ok"] is True
    assert 结果["restored"] == 3
    assert (root / "a.md").read_text(encoding="utf-8") == "old"
    assert (
        root / "dynamic" / "aux" / "2026-09-29 12-00-00_abc.md"
    ).read_text(encoding="utf-8") == "old-aux"
    assert (root / "dynamic" / "trail " / "x.md").read_text(encoding="utf-8") == "old-trail"


@pytest.mark.skipif(os.name == "nt", reason="Windows 文件系统不分大小写")
def test_分大小写的文件系统上只差大小写的两份都还原(tmp_path):
    root = tmp_path / "buckets"
    root.mkdir()
    if github_web._case_insensitive_root(str(root)):
        pytest.skip("当前文件系统不分大小写")
    备份 = tmp_path / "backup.zip"
    with zipfile.ZipFile(备份, "w") as z:
        z.writestr("safe/a.md", "lower")
        z.writestr("safe/A.md", "upper")

    结果 = _rollback_from_backup(str(root), str(备份))

    assert 结果["ok"] is True
    assert 结果["restored"] == 2
    assert (root / "safe" / "a.md").read_text(encoding="utf-8") == "lower"
    assert (root / "safe" / "A.md").read_text(encoding="utf-8") == "upper"


def test_windows_上按不分大小写判重():
    if os.name != "nt":
        pytest.skip("只在 Windows 上断言")
    assert github_web._case_insensitive_root(".") is True
