# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU only (no device, no tt-metal context): the weight cache's one on-disk layout (bank_major.py, 2026-11-06).

A hit = every file of the layer is a complete <stem>.bm.bin (its header holds everything the loader needs); anything
else is a miss for the whole layer; an old-layout <stem>.tensorbin is never read, only named in the miss line (the
converter is the migration); the stager says once when tt-metal's pinned-memory limit is below its staging; the pinned
upload window (pinned_upload) raises tt-metal's limit only inside the outermost window, and on leaving it (also on an
error) unpins and frees the staging, then restores the limit; an explicit TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES is
kept."""

import os

import pytest
import torch

from ttnn.bringup.flat_routed_expert_ttnn import bank_major as bm
from ttnn.bringup.flat_routed_expert_ttnn import flat_expert as fx

INFO = {"rows": 64, "banks": 4, "dtype": "BFLOAT4_B", "shape": [1, 64, 128], "tile_bytes": 576}
MESH = (2, 1)


def _shard(seed):
    g = torch.Generator().manual_seed(seed)
    n = (INFO["rows"] // 32) * INFO["banks"] * INFO["tile_bytes"]
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)


def _write(path, seed=0):
    bm.write_bm(str(path), INFO, MESH, [_shard(seed + k) for k in range(MESH[0] * MESH[1])])
    return str(path)


def _files(tmp_path):
    stem = tmp_path / "L00.bfp4_mse.abc.flat.bf4.layk"
    return {n: f"{stem}.{n}{bm.EXT}" for n in ("w_gu", "w_d")}


class _FakeStager:
    def __init__(self):
        self.uploaded, self.prefetched = [], []

    def upload(self, path):
        self.uploaded.append(path)
        return ("device", os.path.basename(path))

    def prefetch(self, paths):
        self.prefetched.append(list(paths))


@pytest.fixture
def stager(monkeypatch):
    st = _FakeStager()
    monkeypatch.setattr(bm, "stager", lambda device, depth=None: st)
    return st


@pytest.fixture
def no_tensorbin_reads(monkeypatch):
    """Any read of a .tensorbin fails the test (the loader must never open one)."""
    import ttnn

    def boom(*a, **k):
        raise AssertionError(f"a .tensorbin was read: {a}")

    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", boom, raising=False)


def test_file_round_trip_and_completeness(tmp_path):
    p = _write(tmp_path / "x.bm.bin")
    hdr = bm.read_header(p)
    assert {k: hdr[k] for k in INFO} == INFO and hdr["mesh"] == list(MESH) and hdr["n_shards"] == 2
    assert bm.is_complete(p) and os.path.getsize(p) == bm.HDR + 2 * hdr["stride"] and hdr["stride"] % bm.ALIGN == 0
    back = [torch.empty(hdr["shard_bytes"], dtype=torch.uint8) for _ in range(2)]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as pool:
        bm.read_bm_into(p, back, pool, hdr)
    assert all(torch.equal(b, _shard(k)) for k, b in enumerate(back))
    with open(p, "r+b") as f:  # a damaged (short) file is not complete
        f.truncate(os.path.getsize(p) - 1)
    assert not bm.is_complete(p)
    (tmp_path / "foreign.bm.bin").write_bytes(b"\0" * 8192)
    assert not bm.is_complete(tmp_path / "foreign.bm.bin") and not bm.is_complete(tmp_path / "missing.bm.bin")
    assert not list(tmp_path.glob("*.tmp*"))  # written to a temp name, renamed into place


def test_permutation_is_inverted_exactly():
    banks, tb = 4, 8
    src = torch.arange(3 * banks * tb, dtype=torch.int64).to(torch.uint8)
    out, back = torch.empty_like(src), torch.empty_like(src)
    bm.width_to_bank_major(src, banks, tb, out)
    # bank b holds its tiles k = 0.. contiguously: host tile (k, b) at k * banks + b -> b * T + k
    assert torch.equal(out.view(banks, 3, tb)[1, 2], src.view(3, banks, tb)[2, 1])
    assert torch.equal(bm.bank_to_width_major(out, banks, tb, back), src)


def test_names(expect_error):
    assert bm.bm_path("/c/a.w_gu.tensorbin") == "/c/a.w_gu.bm.bin"
    assert bm.old_layout_path("/c/a.w_gu.bm.bin") == "/c/a.w_gu.tensorbin"
    with expect_error(AssertionError, "tensorbin"):
        bm.write_from_host("/c/a.w_gu.tensorbin", None)  # the miss path writes only .bm.bin


def test_hit_is_the_layers_bm_files(tmp_path, stager, no_tensorbin_reads):
    files = _files(tmp_path)
    for f in files.values():
        _write(f)
    got = fx._load_cached_set(files, device=None)
    assert got == {n: ("device", os.path.basename(f)) for n, f in files.items()}
    assert stager.uploaded == list(files.values())
    assert fx.prefetch_cached(None, list(files.values())) and stager.prefetched == [list(files.values())]


def test_a_missing_or_damaged_bm_file_misses_the_whole_layer(tmp_path, stager, no_tensorbin_reads):
    files = _files(tmp_path)
    _write(files["w_gu"])
    assert fx._load_cached_set(files, None) == {"w_gu": None, "w_d": None} and not stager.uploaded
    assert not fx.prefetch_cached(None, list(files.values())) and not stager.prefetched
    (tmp_path / os.path.basename(files["w_d"])).write_bytes(b"TTBMAJ01{}")  # foreign / damaged
    assert fx._load_cached_set(files, None) == {"w_gu": None, "w_d": None}
    assert fx._load_cached_set(dict.fromkeys(files), None) == {"w_gu": None, "w_d": None}  # no cache


def test_an_old_layout_cache_is_never_read_and_the_miss_names_the_converter(tmp_path, stager, no_tensorbin_reads):
    from loguru import logger

    files = _files(tmp_path)
    for f in files.values():
        open(bm.old_layout_path(f), "wb").write(b"old flatbuffer")  # .tensorbin only
    sink = []
    hid = logger.add(sink.append, level="WARNING")
    try:
        assert fx._load_cached_set(files, None) == {"w_gu": None, "w_d": None}
    finally:
        logger.remove(hid)
    msg = "".join(map(str, sink))
    assert "convert_expert_cache.py" in msg and "2 .tensorbin, never read" in msg
    assert not stager.uploaded


def test_the_stager_says_once_when_the_pinned_limit_is_below_its_staging(monkeypatch):
    from loguru import logger

    st = bm.PinnedStager.__new__(bm.PinnedStager)
    st.mesh, st.depth, st._warned_pin = None, 2, False
    st.views = {"k": [[torch.empty(3 << 20, dtype=torch.uint8)] * 4] * 2}  # 2 slots x 4 chips x 3 MiB = 24 MiB
    assert st.staged_bytes() == 24 << 20
    limit = [32 << 20]
    monkeypatch.setattr(bm, "pinned_limit", lambda mesh: limit[0])  # (tt-metal's limit, read from the mesh's env)
    sink = []
    hid = logger.add(sink.append, level="WARNING")
    try:
        st._check_pin_limit()
        assert not sink
        limit[0] = 0
        st._check_pin_limit()
        st._check_pin_limit()
    finally:
        logger.remove(hid)
    assert len(sink) == 1 and "issue-queue copy" in str(sink[0]) and bm.PINNED_LIMIT_ENV in str(sink[0])


class _Mesh:
    shape = (1, 2)


class _FakeOps:
    """tt-metal's side of the pinned window (the fork's bindings): the limit, the cache's entries per host tensor."""

    def __init__(self, limit):
        self.limit, self.pinned, self.calls = limit, set(), []

    def pinned_memory_limit(self, mesh):
        return self.limit

    def set_pinned_memory_limit(self, mesh, b):
        self.calls.append(("set", b))
        prev, self.limit = self.limit, b
        return prev

    def release_pinned(self, t):
        self.calls.append(("release", id(t)))
        self.pinned.discard(id(t))
        return 1

    def pinned_cache_entries(self):
        return len(self.pinned)


@pytest.fixture
def window(monkeypatch):
    from loguru import logger

    ops = _FakeOps(4 << 30)
    monkeypatch.setattr(bm, "_ops", lambda: ops)
    monkeypatch.delenv(bm.PINNED_LIMIT_ENV, raising=False)
    sink = []
    hid = logger.add(lambda m: sink.append(str(m)), level="INFO", filter=lambda r: "FLAT_CACHE" in r["message"])
    yield ops, sink
    logger.remove(hid)
    bm._STAGERS.clear()
    bm._WINDOWS.clear()


def _fake_stager(mesh, ops):
    """A stager of ``mesh`` with two staging slots pinned (as two uploads would leave them)."""
    st = bm.PinnedStager(mesh, depth=2, threads=1)
    slots = [object(), object()]
    st.sets = {"k": slots}
    st.views = {"k": [[torch.empty(1 << 20, dtype=torch.uint8)] * 2] * 2}
    ops.pinned |= {id(t) for t in slots}
    bm._STAGERS[id(mesh)] = st
    return st, slots


def test_the_pinned_limit_is_raised_only_inside_the_outermost_window_and_the_staging_released(window):
    ops, sink = window
    mesh = _Mesh()
    with bm.pinned_upload(mesh):  # (the model's layer loop: holds, raises nothing)
        assert ops.limit == 4 << 30 and not ops.calls
        for layer in range(3):
            with bm.pinned_upload(mesh, 16 << 30):  # (each layer's load)
                assert ops.limit == 16 << 30
                if layer == 0:
                    st, slots = _fake_stager(mesh, ops)
            assert bm._STAGERS.get(id(mesh)) is st and ops.pinned  # (kept across layers: the window is held)
    # after the last layer: the staging unpinned (before it is dropped), the limit restored, once
    assert ops.calls == [("set", 16 << 30), ("release", id(slots[0])), ("release", id(slots[1])), ("set", 4 << 30)]
    assert ops.limit == 4 << 30 and not ops.pinned and not st.sets and not st.views and id(mesh) not in bm._STAGERS
    assert len(sink) == 2 and "raised 4.0 GiB -> 16.0 GiB" in sink[0]
    assert f"({4 << 20} B) staged" in sink[1] and "2 pinned-memory cache entries dropped" in sink[1]
    assert "restored 16.0 -> 4.0 GiB" in sink[1]


def test_the_window_restores_the_limit_on_an_error_and_alone_releases_per_layer(window, expect_error):
    ops, sink = window
    mesh = _Mesh()
    with expect_error(RuntimeError, "upload failed"):
        with bm.pinned_upload(mesh, 16 << 30):
            _fake_stager(mesh, ops)
            raise RuntimeError("upload failed")
    assert ops.limit == 4 << 30 and not ops.pinned and not bm._WINDOWS and id(mesh) not in bm._STAGERS
    with bm.pinned_upload(mesh, 16 << 30):  # (a layer built alone: its own window)
        assert ops.limit == 16 << 30
    assert ops.limit == 4 << 30


def test_an_explicit_limit_is_kept_and_a_higher_one_not_lowered(window, monkeypatch):
    ops, sink = window
    mesh = _Mesh()
    monkeypatch.setenv(bm.PINNED_LIMIT_ENV, "0")  # (the KMD 2.9 workaround: the upload goes unpinned, slower)
    ops.limit = 0
    with bm.pinned_upload(mesh, 16 << 30):
        assert ops.limit == 0
    monkeypatch.delenv(bm.PINNED_LIMIT_ENV)
    ops.limit = 64 << 30
    with bm.pinned_upload(mesh, 16 << 30):
        assert ops.limit == 64 << 30
    assert not [c for c in ops.calls if c[0] == "set"] and ops.limit == 64 << 30
    assert all("unchanged" in m for m in sink if "released" in m)


MOCK_WINDOW = r"""
import os, ttnn
from ttnn.bringup.flat_routed_expert_ttnn import bank_major as bm
o = ttnn._ttnn.operations.bringup
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    base = o.pinned_memory_limit(mesh)
    with bm.pinned_upload(mesh):
        with bm.pinned_upload(mesh, base + (12 << 30)):
            assert o.pinned_memory_limit(mesh) == base + (12 << 30)
            st = bm.stager(mesh)
            hdr = {"rows": 64, "banks": 8, "dtype": "BFLOAT4_B", "shape": [1, 64, 256]}
            host, views = st.staging(hdr, 0)
            assert int(views[0].numel()) == 2 * 8 * 576
        assert o.pinned_memory_limit(mesh) == base + (12 << 30)
    assert o.pinned_memory_limit(mesh) == base and not st.sets and o.pinned_cache_entries() == 0
    assert o.release_pinned(host) == 2 * 8 * 576  # (a host tensor's shards; unpinned ones are a no-op)
    print("WINDOW_OK", base)
finally:
    ttnn.close_mesh_device(mesh)
"""


def test_the_window_on_tt_metal_mock_cluster(tmp_path):
    """The real bindings (set_pinned_memory_limit, release_pinned, pinned_cache_entries) on tt-metal's mock cluster:
    no card is touched (a mock device has no command queue, so nothing is uploaded or pinned)."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(bm.__file__).resolve().parents[4]
    mock = repo / "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
    if not mock.is_file():
        pytest.skip(f"no mock cluster descriptor {mock}")
    env = dict(os.environ, TT_METAL_MOCK_CLUSTER_DESC_PATH=str(mock), PYTHONPATH=str(repo), TT_METAL_HOME=str(repo))
    env.pop(bm.PINNED_LIMIT_ENV, None)
    r = subprocess.run(
        [sys.executable, "-c", MOCK_WINDOW], env=env, cwd=repo, capture_output=True, text=True, timeout=600
    )
    assert f"WINDOW_OK {4 << 30}" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    assert "FLAT_CACHE pinned limit raised 4.0 GiB -> 16.0 GiB" in r.stdout + r.stderr
    assert "FLAT_CACHE pinned staging released" in r.stdout + r.stderr
