# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU only (no device, no tt-metal context): the weight cache's one on-disk layout (bank_major.py, 2026-11-06).

A hit = every file of the layer is a complete <stem>.bm.bin (its header holds everything the loader needs); anything
else is a miss for the whole layer; an old-layout <stem>.tensorbin is never read, only named in the miss line (the
converter is the migration); the stager says once when tt-metal's pinned-memory limit is below its staging."""

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
    st.depth, st._warned_pin = 2, False
    st.views = {"k": [[torch.empty(3 << 20, dtype=torch.uint8)] * 4] * 2}  # 2 slots x 4 chips x 3 MiB = 24 MiB
    assert st.staged_bytes() == 24 << 20
    sink = []
    hid = logger.add(sink.append, level="WARNING")
    try:
        monkeypatch.setenv(bm.PINNED_LIMIT_ENV, str(32 << 20))
        st._check_pin_limit()
        assert not sink
        monkeypatch.setenv(bm.PINNED_LIMIT_ENV, "0")
        st._check_pin_limit()
        st._check_pin_limit()
    finally:
        logger.remove(hid)
    assert len(sink) == 1 and "issue-queue copy" in str(sink[0]) and bm.PINNED_LIMIT_ENV in str(sink[0])
