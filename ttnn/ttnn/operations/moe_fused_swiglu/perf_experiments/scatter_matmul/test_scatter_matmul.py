# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + device-ns bake-off for the `scatter_matmul` mini-op.

Correctness is PCC of the column root's two reduced blocks against an fp32 torch reference built
from the ACTUAL quantised (bfp8 x, bfp4_b w) per-core operands — which is what the device multiplies
and sums. Perf is `DEVICE KERNEL DURATION [ns]`, median of 3 post-JIT runs (this op family has a
2-4 % run-to-run band, so a single sample is not trusted).

    scripts/run_safe_pytest.sh --run-all <this file>
"""

import os
import pathlib

# PRIVATE PROFILER ARTIFACTS. $TT_METAL_HOME/generated/profiler is SHARED across every run on this
# box; the device flock serialises execution but not the artifacts. Whoever reaches teardown first
# consumes profile_log_device.csv and the loser silently gets NO data while still reporting PASS.
# This override is honoured on both sides (profiler_paths.hpp::get_profiler_artifacts_dir and
# tools/tracy/common.py::PROFILER_ARTIFACTS_DIR) and must be set before ttnn is imported.
os.environ.setdefault("TT_METAL_PROFILER_DIR", str((pathlib.Path(__file__).parent / "profiler").resolve()))
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

# NOTE: `torch` is imported LAZILY everywhere. scripts/validate_no_global_torch_imports.py forbids a
# module-level torch import under ttnn/ttnn/, and these benches live under the op directory.
import pytest
import ttnn
from loguru import logger

from ttnn.operations.moe_fused_swiglu.perf_experiments.scatter_matmul.bench import (
    CHIP_FREQ_MHZ,
    LOFI_CYCLES_PER_TILE_MAC,
    TILE,
    Geo,
    create_descriptor,
    feasible,
    hillis_steele_tree,
    l1_bytes,
    plan,
    roofline_ns,
    tile_macs,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# The op's 88-core geometry: 11 grid columns x KGROUPS 8 rows, per-core K = KR_PAD 28 tiles,
# per-column hidden = HN_PAD 6 tiles, runtime m_eff = 8 token tile-rows.
NCOLS = 11
FOCUS_K, FOCUS_M, FOCUS_N, FOCUS_KR = 8, 8, 6, 28
KR_FOR_K = {4: 28, 8: 28, 10: 23}  # the op's per-core K at each column depth


def geo_of(k=FOCUS_K, m=FOCUS_M, n=FOCUS_N, kr=None, ncols=NCOLS):
    return Geo(k=k, m=m, n=n, kr=kr if kr is not None else KR_FOR_K.get(k, 28), ncols=ncols)


FOCUS = geo_of()


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _pcc(a, b):
    import torch

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / denom)


# ---------------------------------------------------------------------------
# Tensors. Every operand is an L1 height shard consumed through a zero-copy
# tensor-backed CB: one shard per core, laid out as ONE 32-row strip of
# `n_tiles` side-by-side tiles, so flat tile index == CB page index.
# ---------------------------------------------------------------------------


def _sharded_config(cores, n_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(TILE, n_tiles * TILE),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _pack(mats, rows_t, cols_t):
    """Per-shard [rows_t*TILE, cols_t*TILE] matrices -> the flat (n_shards*TILE, rows_t*cols_t*TILE)
    strip layout, tile (i, j) at flat tile index i*cols_t + j (row-major, what matmul_block reads).

    Vectorised, not a tile loop: the loop version costs ~10 s at the largest cell and the sweeps
    rebuild these per test."""
    import torch

    t = torch.stack(mats)  # [S, R*32, C*32]
    s_, r_, c_ = len(mats), rows_t, cols_t
    return t.view(s_, r_, TILE, c_, TILE).permute(0, 2, 1, 3, 4).reshape(s_ * TILE, r_ * c_ * TILE).contiguous()


def _unpack_all(flat, n_shards, rows_t, cols_t, tile_base=0, n_tiles_total=None):
    """Inverse of `_pack` for every shard at once -> [S, rows_t*TILE, cols_t*TILE]."""
    import torch

    total = n_tiles_total if n_tiles_total is not None else rows_t * cols_t
    v = flat.view(n_shards, TILE, total, TILE)[:, :, tile_base : tile_base + rows_t * cols_t, :]
    return (
        v.view(n_shards, TILE, rows_t, cols_t, TILE)
        .permute(0, 2, 1, 3, 4)
        .reshape(n_shards, rows_t * TILE, cols_t * TILE)
    )


# Keyed on the LIVE device as well as the geometry: the `device` fixture is per-test, so a cache
# entry made under a previous device holds buffers on a closed one (which surfaces as
# `TT_FATAL: cq_id 0 is out of range` at the next generic_op, not as a use-after-free).
_TENSOR_CACHE = {}
_CACHE_DEVICE = [None]


def make_tensors(device, geo):
    """Resident x / wg / wu over the whole grid + the root-row output shard, plus the per-column
    fp32 reference built from the QUANTISED operands the device actually multiplies."""
    import torch

    if _CACHE_DEVICE[0] is not id(device):
        _TENSOR_CACHE.clear()
        _CACHE_DEVICE[0] = id(device)
    key = (geo.k, geo.m, geo.n, geo.kr, geo.ncols)
    if key in _TENSOR_CACHE:
        return _TENSOR_CACHE[key]

    torch.manual_seed(0xC0FFEE)
    n_shards = geo.k * geo.ncols
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(geo.ncols - 1, geo.k - 1))])
    roots = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(geo.ncols - 1, 0))])

    # Per-shard-distinct operands so a wrong-core / wrong-offset / wrong-column bug breaks PCC
    # rather than matching by luck. Zero-mean and small so no DC term swamps the block-float
    # mantissa (a DC-heavy pattern made a sibling bench score 0.776 on a CORRECT kernel).
    xs = [(torch.rand(geo.m * TILE, geo.kr * TILE) - 0.5) * 0.8 for _ in range(n_shards)]
    wgs = [(torch.rand(geo.kr * TILE, geo.n * TILE) - 0.5) * 0.8 for _ in range(n_shards)]
    wus = [(torch.rand(geo.kr * TILE, geo.n * TILE) - 0.5) * 0.8 for _ in range(n_shards)]

    x_t = ttnn.from_torch(
        _pack(xs, geo.m, geo.kr),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_config(grid, geo.x_tiles),
    )
    wg_t = ttnn.from_torch(
        _pack(wgs, geo.kr, geo.n),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_config(grid, geo.w_tiles),
    )
    wu_t = ttnn.from_torch(
        _pack(wus, geo.kr, geo.n),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_config(grid, geo.w_tiles),
    )
    out_t = ttnn.from_torch(
        torch.zeros((geo.ncols * TILE, 2 * geo.t * TILE), dtype=torch.float32),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded_config(roots, 2 * geo.t),
    )

    xq = _unpack_all(ttnn.to_torch(x_t).to(torch.float32), n_shards, geo.m, geo.kr)
    wgq = _unpack_all(ttnn.to_torch(wg_t).to(torch.float32), n_shards, geo.kr, geo.n)
    wuq = _unpack_all(ttnn.to_torch(wu_t).to(torch.float32), n_shards, geo.kr, geo.n)
    pg = torch.bmm(xq, wgq)  # [S, m*32, n*32] — every core's own partial, in fp32
    pu = torch.bmm(xq, wuq)
    refs = []
    for col in range(geo.ncols):
        idx = [geo.shard_index(col, r) for r in range(geo.k)]
        refs.append((pg[idx].sum(0), pu[idx].sum(0)))
    _TENSOR_CACHE[key] = (x_t, wg_t, wu_t, out_t, refs)
    return _TENSOR_CACHE[key]


def _check(out_t, geo, refs):
    """min PCC over every column root, gate and up."""
    import torch

    got = ttnn.to_torch(out_t).to(torch.float32)
    g = _unpack_all(got, geo.ncols, geo.m, geo.n, tile_base=0, n_tiles_total=2 * geo.t)
    u = _unpack_all(got, geo.ncols, geo.m, geo.n, tile_base=geo.t, n_tiles_total=2 * geo.t)
    return min(min(_pcc(g[col], refs[col][0]), _pcc(u[col], refs[col][1])) for col in range(geo.ncols))


def run_cell(device, shape, geo, mech="addchain", slots=1, measure=True, samples=2, fidelity=None):
    import torch

    x_t, wg_t, wu_t, out_t, refs = make_tensors(device, geo)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.zeros((geo.ncols * TILE, 2 * geo.t * TILE), dtype=torch.float32),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        ),
        out_t,
    )

    # A FRESH ProgramDescriptor per dispatch. Re-enqueuing one descriptor is not the same thing:
    # the collectives here are built on counting semaphores whose initial values belong to program
    # setup, so a repeat dispatch is not a clean repeat of the collective.
    def once():
        desc = create_descriptor(device, x_t, wg_t, wu_t, out_t, shape, geo, mech, slots, fidelity=fidelity)
        ttnn.generic_op([x_t, wg_t, wu_t, out_t], desc)

    once()
    pcc = None if shape == "mm_only" else _check(out_t, geo, refs)
    ns, got = None, []
    if measure:
        ttnn.synchronize_device(device)
        _read_kernel_ns(device)  # discard: the first run also paid JIT compile
        got = []
        for _ in range(samples):
            once()
            v = _read_kernel_ns(device)
            if v is not None:
                got.append(v)
        got.sort()
        ns = got[len(got) // 2] if got else None
    return pcc, ns, got


def _fmt(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(v).ljust(w[i]) for i, v in enumerate(r))
    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


# ---------------------------------------------------------------------------
# Host-only sanity
# ---------------------------------------------------------------------------


def test_geometry():
    """The plans really partition the block, and the L1 accounting is what bounds the surface."""
    g = FOCUS
    assert g.t == 48 and g.workers == 8 and g.a == 6, (g.t, g.workers, g.a)
    tree = hillis_steele_tree(8)
    assert len(tree[0]["children"]) == 3  # ceil(log2(8))
    assert sum(len(n["children"]) for n in tree.values()) == 7

    for shape in ("scatter", "ring"):
        p = plan(shape, g)
        assert sum(a for a in p["assigned"]) == g.t * (g.workers / g.workers)
        assert sorted(p["offsets"]) == sorted(i * g.a for i in range(g.k))

    rows = []
    for m in (1, 2, 4, 8):
        for n in (2, 4, 6, 8, 12):
            gg = geo_of(m=m, n=n)
            cells = []
            for shape in ("scatter", "tree", "direct", "ring"):
                ok, why = feasible(shape, gg, "addchain", 1)
                cells.append(f"{l1_bytes(shape, gg)[1] // 1024}K" if ok else "X")
            rows.append([m, n, gg.t, gg.workers, gg.a] + cells)
    logger.info(
        "\n=== scatter_matmul L1 (root core, KiB) over the m_eff x N surface at K=8, KR=28 ===\n"
        + _fmt(rows, ["m_eff", "N", "T", "W", "a", "scatter", "tree", "direct", "ring"])
    )


# ---------------------------------------------------------------------------
# Device: correctness first, smallest cell first
# ---------------------------------------------------------------------------

CORRECTNESS = [
    ("scatter", geo_of(k=4, m=2, n=2, kr=4), "addchain", 1),
    ("scatter", FOCUS, "addchain", 1),
    ("scatter", FOCUS, "pack_l1_acc", 1),
    ("scatter", FOCUS, "dest_acc", 1),
    ("scatter", FOCUS, "pack_l1_pair", 1),
    ("scatter_dual", FOCUS, "addchain", 1),
    ("tree", FOCUS, "addchain", 1),
    ("tree", FOCUS, "addchain", 2),
    ("direct", FOCUS, "addchain", 1),
    ("direct", FOCUS, "addchain", 2),
    ("ring", FOCUS, "addchain", 1),
]


@pytest.mark.parametrize("case", CORRECTNESS, ids=lambda c: f"{c[0]}_{c[2]}_s{c[3]}_K{c[1].k}m{c[1].m}n{c[1].n}")
def test_correctness(device, case):
    shape, geo, mech, slots = case
    ok, why = feasible(shape, geo, mech, slots)
    if not ok:
        pytest.skip(why)
    pcc, _, _ = run_cell(device, shape, geo, mech, slots, measure=False)
    logger.info(f"[correct] {shape:13s} {mech:13s} slots={slots} K={geo.k} m={geo.m} N={geo.n} pcc={pcc:.6f}")
    assert pcc > 0.99, f"{shape}/{mech}/slots={slots}: min PCC over {geo.ncols} roots = {pcc}"


# ---------------------------------------------------------------------------
# Sweeps. ONE CELL PER TEST, results appended to results.jsonl and tabulated by
# `test_zz_report`.
#
# WHY per-test and not one big loop: a single process that dispatches ~45 of
# these 88-core collective programs back to back hangs on dispatch (reproduced
# twice, at the same point in the menu; the shape it hung on — `direct` — runs
# clean for 4 consecutive dispatches in isolation, PCC 0.9998, 71.1 us). The
# accumulation, not the kernel, is the fault, so every cell gets a fresh device
# and stays well under that ceiling.
# ---------------------------------------------------------------------------

import json
import pathlib

RESULTS = pathlib.Path(__file__).parent / "results.jsonl"


def record(**row):
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


FIDELITY_NAMES = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


def measure(device, tag, shape, geo, mech="addchain", slots=1, fid="LoFi"):
    ok, why = feasible(shape, geo, mech, slots)
    if not ok:
        record(
            tag=tag,
            shape=shape,
            mech=mech,
            slots=slots,
            k=geo.k,
            m=geo.m,
            n=geo.n,
            kr=geo.kr,
            t=geo.t,
            a=geo.a,
            ns=None,
            pcc=None,
            l1=None,
            skip=why,
            fid=fid,
        )
        logger.info(f"[{tag}] {shape:13s} {mech:13s} s={slots} K={geo.k} m={geo.m} N={geo.n} SKIP: {why}")
        return
    pcc, ns, samples = run_cell(device, shape, geo, mech, slots, fidelity=FIDELITY_NAMES[fid])
    l1 = l1_bytes(shape, geo, mech, slots)[1]
    rl = roofline_ns(geo)
    record(
        tag=tag,
        shape=shape,
        mech=mech,
        slots=slots,
        k=geo.k,
        m=geo.m,
        n=geo.n,
        kr=geo.kr,
        t=geo.t,
        a=geo.a,
        ns=ns,
        pcc=pcc,
        l1=l1,
        skip=None,
        samples=samples,
        fid=fid,
        roofline=rl,
        tile_macs=tile_macs(geo),
        math_util=(None if ns in (None, 0) else rl / ns),
    )
    logger.info(
        f"[{tag}] {shape:13s} {mech:13s} s={slots} {fid} K={geo.k} m={geo.m} N={geo.n} "
        f"ns={'?' if ns is None else format(ns, '9.0f')} "
        f"util={'?' if ns is None else format(rl / ns * 100, '.1f')}% pcc={pcc} L1={l1 // 1024}K"
    )
    if pcc is not None:
        assert pcc > 0.99, f"{tag} {shape}/{mech}: pcc {pcc}"


MENU = (
    [("mm_only", "addchain", 1)]
    + [("scatter", m, 1) for m in ("addchain", "pack_l1_acc", "dest_acc", "pack_l1_pair")]
    + [("scatter_dual", m, 1) for m in ("addchain", "dest_acc", "pack_l1_pair")]
    + [("tree", m, s) for m in ("addchain", "pack_l1_acc") for s in (1, 2)]
    + [("direct", m, s) for m in ("addchain", "pack_l1_acc") for s in (1, 2)]
    + [("ring", "addchain", 1)]
)


@pytest.mark.parametrize("entry", MENU, ids=lambda e: f"{e[0]}_{e[1]}_s{e[2]}")
def test_focus_menu(device, entry):
    """Every shape x every legal mechanism at the op's own 88-core operating point."""
    shape, mech, slots = entry
    measure(device, "focus", shape, FOCUS, mech, slots)


M_AXIS = (1, 2, 4, 8)
N_AXIS = (2, 4, 6, 8, 12)
SURFACE_SHAPES = ("mm_only", "scatter", "scatter_dual", "tree", "direct", "ring")
SURFACE = [(m, n, s) for m in M_AXIS for n in N_AXIS for s in SURFACE_SHAPES]


@pytest.mark.parametrize("cell", SURFACE, ids=lambda c: f"m{c[0]}_N{c[1]}_{c[2]}")
def test_mn_surface(device, cell):
    """The m_eff x N surface at the op's KGROUPS. The SHAPE of this surface is the deliverable."""
    m, n, shape = cell
    measure(device, "surface", shape, geo_of(m=m, n=n), "addchain", 1)


KSWEEP = [(k, m, n, s) for k in (4, 8, 10) for m, n in ((8, 6), (4, 6), (1, 6)) for s in SURFACE_SHAPES]


@pytest.mark.parametrize("cell", KSWEEP, ids=lambda c: f"K{c[0]}_m{c[1]}_N{c[2]}_{c[3]}")
def test_kgroups_sweep(device, cell):
    k, m, n, shape = cell
    measure(device, "kgroups", shape, geo_of(k=k, m=m, n=n), "addchain", 1)


WINNER_SHAPES = (("scatter", "addchain"), ("scatter_dual", "pack_l1_pair"), ("ring", "addchain"))
WINNER_GRID = [(m, n, sh, mm) for m in M_AXIS for n in N_AXIS for sh, mm in WINNER_SHAPES]


@pytest.mark.parametrize("cell", WINNER_GRID, ids=lambda c: f"m{c[0]}_N{c[1]}_{c[2]}_{c[3]}")
def test_winner_surface(device, cell):
    """The top combination vs the shipped one across the whole m_eff x N surface — this is what
    the predicate is read off."""
    m, n, shape, mech = cell
    measure(device, "winner", shape, geo_of(m=m, n=n), mech, 1)


WINNER_K = [(k, m, n, sh, mm) for k in (4, 8, 10) for m, n in ((8, 6), (4, 6), (1, 6)) for sh, mm in WINNER_SHAPES]


@pytest.mark.parametrize("cell", WINNER_K, ids=lambda c: f"K{c[0]}_m{c[1]}_N{c[2]}_{c[3]}_{c[4]}")
def test_winner_kgroups(device, cell):
    k, m, n, shape, mech = cell
    measure(device, "winnerk", shape, geo_of(k=k, m=m, n=n), mech, 1)


MECHS4 = ("addchain", "pack_l1_acc", "dest_acc", "pack_l1_pair")
MECH_CELLS = [(8, 6), (4, 6), (8, 12), (1, 6), (8, 8)]
MECH_SHAPES = ("scatter", "scatter_dual")
MECH_GRID = [(m, n, sh, mm) for m, n in MECH_CELLS for sh in MECH_SHAPES for mm in MECHS4]


@pytest.mark.parametrize("cell", MECH_GRID, ids=lambda c: f"m{c[0]}_N{c[1]}_{c[2]}_{c[3]}")
def test_mech_cross(device, cell):
    """The FULL 2 (single-NoC vs dual-NoC transport) x 4 (accumulate mechanism) factorial, so the
    two axes can be separated instead of inferred from one diagonal."""
    m, n, shape, mech = cell
    measure(device, "mech", shape, geo_of(m=m, n=n), mech, 1)


# --- Fidelity probe: an INSTRUMENT, not a variant. bfp8 x bfp4 operands mean LoFi is one FPU pass,
# HiFi2 two, HiFi4 four, so T(HiFi_n) - T(LoFi) is the wall-clock cost of (n-1) EXTRA passes. Since
# one pass of FPU work is exactly `roofline_ns`, the ratio is the fraction of added FPU work that is
# EXPOSED (lands 1:1 on the wall) rather than hidden under transport. It is a LOWER BOUND on the FPU
# time, not the FPU time itself.
FID_CELLS = [(8, 6), (2, 6), (8, 12)]
FID_SHAPES = (("mm_only", "addchain"), ("scatter", "addchain"), ("scatter_dual", "pack_l1_pair"))
FID_GRID = [(m, n, sh, mm, f) for m, n in FID_CELLS for sh, mm in FID_SHAPES for f in ("LoFi", "HiFi2", "HiFi4")]


@pytest.mark.parametrize("cell", FID_GRID, ids=lambda c: f"m{c[0]}_N{c[1]}_{c[2]}_{c[4]}")
def test_fidelity_probe(device, cell):
    m, n, shape, mech, fid = cell
    measure(device, "fid", shape, geo_of(m=m, n=n), mech, 1, fid=fid)


# ---------------------------------------------------------------------------


def _fmt(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(v).ljust(w[i]) for i, v in enumerate(r))
    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def test_zz_report():
    """Tabulate everything recorded so far. Host-only; run last."""
    if not RESULTS.exists():
        pytest.skip("no results yet")
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        key = (r["tag"], r["shape"], r["mech"], r["slots"], r["k"], r["m"], r["n"], r.get("fid", "LoFi"))
        prev = by.get(key)
        # run_safe_pytest's precompile warm pass records a row with no profiler sample for every
        # cell; a real measurement always wins over one of those.
        if prev is None or (prev["ns"] is None and r["ns"] is not None):
            by[key] = r

    def get(tag, shape, k, m, n, mech="addchain", slots=1, fid="LoFi"):
        return by.get((tag, shape, mech, slots, k, m, n, fid))

    def ns_of(*a, **kw):
        r = get(*a, **kw)
        return None if r is None or r["ns"] is None else r["ns"]

    def cell(tag, shape, k, m, n, mech="addchain", slots=1, what="ns"):
        r = get(tag, shape, k, m, n, mech, slots)
        if r is None:
            return "."
        if r["ns"] is None:
            return "X"
        if what == "ns":
            return f"{r['ns']:.0f}"
        if what == "pcc":
            return f"{r['pcc']:.5f}"
        if what == "util":
            return f"{r['roofline'] / r['ns'] * 100:.0f}%"
        if what == "red":  # everything that is NOT the bare matmul
            base = ns_of(tag, "mm_only", k, m, n)
            return "?" if base is None else f"{r['ns'] - base:.0f}"
        raise ValueError(what)

    out = []
    menu = [v for kk, v in by.items() if kk[0] == "focus"]
    if menu:
        base = next((r["ns"] for r in menu if r["shape"] == "mm_only" and r["ns"] is not None), None)
        rl = roofline_ns(FOCUS)
        mrows = []
        for r in sorted(menu, key=lambda r: (r["ns"] is None, r["ns"] or 0)):
            if r["ns"] is None:
                continue
            mrows.append(
                [
                    r["shape"],
                    r["mech"],
                    r["slots"],
                    f"{r['ns']:.0f}",
                    f"{r['ns'] - base:.0f}" if base else "-",
                    f"{rl / r['ns'] * 100:.1f}%",
                    f"{r['pcc']:.6f}" if r["pcc"] is not None else "-",
                    f"{r['l1'] // 1024}K",
                ]
            )
        out.append(
            f"=== FOCUS MENU  K={FOCUS.k} m_eff={FOCUS.m} N={FOCUS.n} KR={FOCUS.kr} T={FOCUS.t}, "
            f"{FOCUS.ncols} concurrent columns = {FOCUS.k * FOCUS.ncols} cores.\n"
            f"    FPU roofline = {tile_macs(FOCUS)} tile-MACs x {LOFI_CYCLES_PER_TILE_MAC} cyc "
            f"/ {CHIP_FREQ_MHZ:.0f} MHz = {rl:.0f} ns/core.  math_util = roofline / ns ===\n"
            + _fmt(mrows, ["shape", "mech", "slots", "ns", "reduce_cost", "math_util", "pcc", "L1/core"])
        )

    for tag, title, cells in (
        (
            "surface",
            f"m_eff x N SURFACE (K={FOCUS_K} KR={FOCUS_KR}, {NCOLS} columns, addchain)",
            [(FOCUS_K, m, n) for m in M_AXIS for n in N_AXIS],
        ),
        ("kgroups", f"KGROUPS SWEEP ({NCOLS} columns, addchain)", sorted({(k, m, n) for k, m, n, _ in KSWEEP})),
    ):
        for what, lbl in (
            ("ns", "ns"),
            ("util", "MATH UTILISATION = roofline / ns"),
            ("red", "REDUCE COST = ns - mm_only(same cell)"),
        ):
            srows = []
            for k, m, n in cells:
                g = geo_of(k=k, m=m, n=n)
                mm = ns_of(tag, "mm_only", k, m, n)
                head = [
                    k,
                    m,
                    n,
                    g.t,
                    g.a,
                    f"{roofline_ns(g):.0f}",
                    "?" if mm is None else f"{roofline_ns(g) / mm * 100:.0f}%",
                ]
                srows.append(head + [cell(tag, sh, k, m, n, what=what) for sh in SURFACE_SHAPES])
            out.append(
                f"=== {title} — {lbl};  X = infeasible, . = not run ===\n"
                + _fmt(srows, ["K", "m_eff", "N", "T", "a", "roofline", "mm_eff"] + list(SURFACE_SHAPES))
            )

    for tag, title, cells in (
        (
            "winner",
            f"WINNER SURFACE (K={FOCUS_K} KR={FOCUS_KR}, {NCOLS} columns)",
            [(FOCUS_K, m, n) for m in M_AXIS for n in N_AXIS],
        ),
        ("winnerk", f"WINNER KGROUPS ({NCOLS} columns)", sorted({(k, m, n) for k, m, n, _, _ in WINNER_K})),
    ):
        wrows = []
        for k, m, n in cells:
            g = geo_of(k=k, m=m, n=n)
            base = cell(tag, "scatter", k, m, n, mech="addchain")
            vals = [cell(tag, sh, k, m, n, mech=mm) for sh, mm in WINNER_SHAPES]
            utils = [cell(tag, sh, k, m, n, mech=mm, what="util") for sh, mm in WINNER_SHAPES]
            rel = []
            for v in vals[1:]:
                rel.append(
                    f"{(float(v) / float(base) - 1) * 100:+.1f}%"
                    if v not in ("X", ".") and base not in ("X", ".")
                    else "-"
                )
            wrows.append([k, m, n, g.t, g.a] + vals + utils + rel)
        out.append(
            f"=== {title};  vs `scatter/addchain` (what the op ships) ===\n"
            + _fmt(
                wrows,
                [
                    "K",
                    "m_eff",
                    "N",
                    "T",
                    "a",
                    "scat/add",
                    "dual/pair",
                    "ring/add",
                    "u_scat",
                    "u_dual",
                    "u_ring",
                    "dual_vs",
                    "ring_vs",
                ],
            )
        )

    mech_rows = []
    for m, n in MECH_CELLS:
        g = geo_of(m=m, n=n)
        for sh in MECH_SHAPES:
            mech_rows.append(
                [sh, m, n, g.t, g.a]
                + [cell("mech", sh, FOCUS_K, m, n, mech=mm) for mm in MECHS4]
                + [cell("mech", sh, FOCUS_K, m, n, mech=mm, what="util") for mm in MECHS4]
                + [cell("mech", sh, FOCUS_K, m, n, mech=mm, what="pcc") for mm in ("addchain", "pack_l1_pair")]
            )
    out.append(
        "=== TRANSPORT x ACCUMULATE FACTORIAL (K=8): ns, then math_util, then pcc ===\n"
        + _fmt(
            mech_rows,
            ["shape", "m_eff", "N", "T", "a"]
            + [f"{x}" for x in MECHS4]
            + [f"u_{x[:4]}" for x in MECHS4]
            + ["pcc_add", "pcc_pair"],
        )
    )

    # Fidelity probe -> fraction of added FPU work that is EXPOSED.
    frows = []
    for m, n in FID_CELLS:
        g = geo_of(m=m, n=n)
        rl = roofline_ns(g)
        for sh, mm in FID_SHAPES:
            lo = ns_of("fid", sh, FOCUS_K, m, n, mech=mm, fid="LoFi")
            h2 = ns_of("fid", sh, FOCUS_K, m, n, mech=mm, fid="HiFi2")
            h4 = ns_of("fid", sh, FOCUS_K, m, n, mech=mm, fid="HiFi4")
            if lo is None:
                continue
            e2 = None if h2 is None else (h2 - lo) / rl
            e4 = None if h4 is None else (h4 - lo) / (3 * rl)
            agree = "-" if (e2 is None or e4 is None or e2 == 0) else f"{abs(e4 - e2) / e2 * 100:.0f}%"
            frows.append(
                [
                    sh,
                    mm,
                    m,
                    n,
                    f"{rl:.0f}",
                    f"{lo:.0f}",
                    "?" if h2 is None else f"{h2:.0f}",
                    "?" if h4 is None else f"{h4:.0f}",
                    "-" if e2 is None else f"{e2 * 100:.0f}%",
                    "-" if e4 is None else f"{e4 * 100:.0f}%",
                    agree,
                    f"{rl / lo * 100:.0f}%",
                ]
            )
    out.append(
        "=== FIDELITY PROBE — FRACTION OF ADDED FPU WORK THAT IS EXPOSED (a LOWER BOUND on FPU\n"
        "    time, not the FPU time). exp2 = (HiFi2-LoFi)/roofline, exp4 = (HiFi4-LoFi)/(3*roofline);\n"
        "    if the two disagree by >~10% something other than FPU passes is scaling with fidelity ===\n"
        + _fmt(
            frows,
            [
                "shape",
                "mech",
                "m_eff",
                "N",
                "roofline",
                "LoFi",
                "HiFi2",
                "HiFi4",
                "exp2",
                "exp4",
                "disagree",
                "math_util",
            ],
        )
    )
    logger.info("\n\n" + "\n\n".join(out) + "\n")
