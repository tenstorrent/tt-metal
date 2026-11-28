# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bake-off: bf16 ROW_MAJOR -> TILE reorder on the DM RISCs (the op's R5) vs on the Tensix (candidate.py).

Run from the tt-metal root (TT_METAL_HOME / PYTHONPATH = this checkout):

  correctness (the gate, before any timing counts):
    scripts/run_safe_pytest.sh --run-all <this file> -k exact
  timing, one process per case (launch 1 of each variant is a warm-up, launches 2..N measured):
    BENCH_CASES=c49 BENCH_ORDER=op,cand-fast-raw-w2-b16-d2,stock \
      scripts/run_safe_pytest.sh --profile --no-py-zones <this file> -k bench
    python3 <dir>/summarize.py            # newest profiler CSV + newest bench log
"""

import json
import os
import sys
import time
from pathlib import Path

# In-process device profiler (set before the device opens): every launch is read back on its own, so
# no host-side op mapping (tracy) is needed. setdefault keeps an outer tracy run's settings.
if os.environ.get("BENCH_INPROC", "1") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest

# Test-only dependency; the ttnn package forbids a module-level torch import.
torch = pytest.importorskip("torch")
import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate  # noqa: E402

from eval.golden_tests.g6_to_layout import feature_spec as spec  # noqa: E402
from eval.golden_tests.g6_to_layout import helpers  # noqa: E402

g6_to_layout = candidate.op_mod.g6_to_layout
LOG_DIR = HERE / "runs"

SPECIAL_BITS = [
    0x0000,
    0x8000,  # +-0
    0x0001,
    0x0040,
    0x007F,
    0x8001,
    0x8040,
    0x807F,  # subnormals
    0x0080,
    0x8080,
    0x7F7F,
    0xFF7F,  # min normal, max finite
    0x7F80,
    0xFF80,  # +-inf
    0x7FC1,
    0xFFFF,
    0x7F81,
    0x7FC0,
    0xFF81,
    0xFFC0,  # NaN payloads
]


def _as_bf16(bits_i64):
    return bits_i64.to(torch.int32).to(torch.int16).view(torch.bfloat16)


def all_patterns_input(shape, seed):
    """Every bf16 bit pattern at least once (shuffled over tiles and faces), then the specials, then
    random patterns, to fill `shape`."""
    n = 1
    for d in shape:
        n *= d
    g = torch.Generator().manual_seed(seed)
    bits = torch.randperm(65536, generator=g)
    specials = torch.tensor(SPECIAL_BITS)
    pieces = [bits, specials]
    have = bits.numel() + specials.numel()
    if have < n:
        pieces.append(torch.randint(0, 65536, (n - have,), generator=g))
    flat = torch.cat(pieces)
    if flat.numel() > n:
        # Smaller than one full set: keep the specials, then as many shuffled patterns as fit.
        flat = torch.cat([specials, bits])[:n]
    else:
        flat = flat[torch.randperm(n, generator=g)]
    return _as_bf16(flat).reshape(shape)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _run_and_check(device, x, buffer, fn):
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer)
    tt_x = helpers.create_ttnn_input_tensor(
        x, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem
    )
    out = fn(tt_x)
    got = ttnn.to_torch(out).reshape(x.shape)
    ttnn.deallocate(out)
    ttnn.deallocate(tt_x)
    mismatch = (_bits(got) != _bits(x)).sum().item()
    return mismatch, got


EXACT_SHAPES = [
    (1, 4, 128, 128),  # 65536 values, tile aligned, block_cols 4 (fast tilize, one 4-tile chunk)
    (3, 97, 233),  # all patterns + specials, H and W non-aligned, block_cols 8 (chunks 4+4)
    (1, 64, 128, 32),  # 4 x 65536, block_cols 1 (standard tilize path)
    (16, 1, 1536),  # H = 1: one live row per tile-row, column-split blocks
    (1, 2, 70, 208),  # 7 tile columns (fast tilize chunks 4 + 3), H non-aligned
    (1, 1, 1, 185),  # tiny W non-aligned
    (2, 40, 1000),  # 32 tile columns per row (chunks of 4)
]

EXACT_VARIANTS = [
    "op",
    "cand-fast-none-w2-b16-d2-u16",  # helper: UInt16 fails its fast gate -> standard tilize
    "cand-fastapi-none-w2-b16-d2-u16",  # fast tilize compute API, UInt16 CBs, no flag call
    "cand-fastapi-none-w0-b16-d2-u16",  # R6 reader schedule (barrier per block)
    "cand-fastapi-none-w2-b4-d2-u16",  # taller blocks
    "cand-fastapi-none-w4-b16-d3-u16",  # deeper stage, 4 tile-rows in flight
    "cand-fast-raw-w2-b16-d2",  # control: Float16_b CBs with the preserve flag (expected to fail)
    "cand-fast-none-w0-b16-d2-u16-g1",  # granular writer
    "cand-live-none-w0-b16-d2-u16-g1",  # live columns only (compute API)
    "cand-tile-none-w0-b16-d2-u16-g1",  # per-tile push (raw LLK)
    "cand-tile-none-w0-b4-d2-u16-g1",  # per-tile push, taller blocks (ragged padding tile-rows)
    "cand-opk-none-w0-b16-d2-u16",  # the op's own R6 compute kernel, verbatim, on UInt16 CBs
    "cand-opk-none-w0-b4-d2-u16",
]


@pytest.mark.parametrize("buffer", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["dram", "l1"])
@pytest.mark.parametrize("shape", EXACT_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", EXACT_VARIANTS)
def test_exact(device, monkeypatch, variant, shape, buffer):
    x = all_patterns_input(shape, seed=hash(shape) & 0xFFFF)
    with monkeypatch.context() as m:
        if variant != "op":
            candidate.install(m, candidate.Cfg.parse(variant))
        else:
            candidate.clear_caches()
        mismatch, got = _run_and_check(device, x, buffer, lambda t: g6_to_layout(t, ttnn.TILE_LAYOUT))
    candidate.clear_caches()
    total = x.numel()
    detail = ""
    if mismatch:
        bad = (_bits(got) != _bits(x)).flatten()
        xi = _bits(x).flatten()[bad][:8].tolist()
        gi = _bits(got).flatten()[bad][:8].tolist()
        detail = " first: " + ", ".join(f"{a & 0xFFFF:04x}->{b & 0xFFFF:04x}" for a, b in zip(xi, gi))
        # which input classes failed
        xb = _bits(x).flatten()[bad].to(torch.int32) & 0xFFFF
        exp0 = ((xb & 0x7F80) == 0).sum().item()
        exp_ff = ((xb & 0x7F80) == 0x7F80).sum().item()
        detail += f" | exp0={exp0} expFF={exp_ff} other={xb.numel() - exp0 - exp_ff}"
    print(f"\nEXACT variant={variant} shape={shape} buffer={buffer} mismatches={mismatch}/{total}{detail}")
    if variant.startswith("cand") and not variant.endswith("-u16"):
        return  # control: Float16_b CBs, reported, not gated
    assert mismatch == 0


BIG_SHAPES = [(1, 128, 256, 128), (1, 128, 256, 256), (1, 64, 128, 256), (1, 128, 256, 32)]


@pytest.mark.parametrize("buffer", [ttnn.BufferType.DRAM, ttnn.BufferType.L1], ids=["dram", "l1"])
@pytest.mark.parametrize("shape", BIG_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", ["cand-fast-none-w0-b16-d2-u16", "cand-opk-none-w0-b16-d2-u16"])
def test_exact_big(device, monkeypatch, variant, shape, buffer):
    """Production shapes (multi-tile full-row and ragged blocks) holding every bf16 pattern at least once
    plus ~1% special classes throughout (random 16-bit fill)."""
    x = all_patterns_input(shape, seed=11)
    with monkeypatch.context() as m:
        candidate.install(m, candidate.Cfg.parse(variant))
        mismatch, got = _run_and_check(device, x, buffer, lambda t: g6_to_layout(t, ttnn.TILE_LAYOUT))
    candidate.clear_caches()
    c = {k: v for k, v in _census(_bits(x)).items()}
    print(
        f"\nEXACTBIG variant={variant} shape={shape} buffer={buffer} mismatches={mismatch}/{x.numel()} specials_in={c}"
    )
    assert mismatch == 0


DIAG_VARIANTS = [
    "cand-copydiag-raw-w2-b16-d2",  # unpack A -> MOVA2D (preserve) -> pack, no tilize
    "cand-copydiag-none-w2-b16-d2",  # same, flag forced to flush after copy_init
    "cand-slow-late-w2-b16-d2",  # flag pinned after tilize_init
    "cand-fastapi-raw-w2-b16-d2",  # bf16, fast tilize compute API
    "cand-slow-raw-w2-b16-d2-u16",  # UInt16 CBs, standard tilize
    "cand-fastapi-raw-w2-b16-d2-u16",  # UInt16 CBs, fast tilize compute API
    "cand-fast-raw-w2-b16-d2-u16",  # UInt16 CBs, helper (falls back to standard tilize)
]


def _census(bits):
    b = bits.flatten().to(torch.int32) & 0xFFFF
    exp = b & 0x7F80
    man = b & 0x007F
    return {
        "neg0": int((b == 0x8000).sum()),
        "subnorm": int(((exp == 0) & (man != 0)).sum()),
        "inf": int(((exp == 0x7F80) & (man == 0)).sum()),
        "nan": int(((exp == 0x7F80) & (man != 0)).sum()),
    }


@pytest.mark.parametrize("variant", DIAG_VARIANTS)
def test_diag(device, monkeypatch, variant):
    shape = (1, 4, 128, 128)
    x = all_patterns_input(shape, seed=7)
    with monkeypatch.context() as m:
        candidate.install(m, candidate.Cfg.parse(variant))
        mismatch, got = _run_and_check(device, x, ttnn.BufferType.DRAM, lambda t: g6_to_layout(t, ttnn.TILE_LAYOUT))
    candidate.clear_caches()
    if "copydiag" in variant:
        # Pages copied verbatim: compare the value multisets.
        a = torch.sort(_bits(x).flatten().to(torch.int32) & 0xFFFF).values
        b = torch.sort(_bits(got).flatten().to(torch.int32) & 0xFFFF).values
        mismatch = int((a != b).sum())
    print(f"\nDIAG variant={variant} mismatches={mismatch} in={_census(_bits(x))} out={_census(_bits(got))}")


# --------------------------------------------------------------------------- bench


def _scenario(case_id):
    want = int(case_id.lstrip("c"))
    for entry in spec.INPUTS:
        if entry[0]["case"] == want:
            return entry[0]
    raise KeyError(case_id)


def _op_id():
    try:
        return int(ttnn._ttnn.get_device_operation_id())
    except Exception:
        return None


_KEYS = {
    "kernel": "DEVICE KERNEL DURATION [ns]",
    "ncrisc": "DEVICE NCRISC KERNEL DURATION [ns]",
    "trisc1": "DEVICE TRISC1 KERNEL DURATION [ns]",
    "brisc": "DEVICE BRISC KERNEL DURATION [ns]",
}


def _read_programs(device):
    """Per program dispatched since the last read: {kernel, ncrisc, trisc1, brisc} ns."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    out = []
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            rec = {}
            for name, key in _KEYS.items():
                entry = results.get(key)
                if entry is not None:
                    rec[name] = float(entry.duration)
            if "kernel" in rec:
                out.append(rec)
    return out


def _bench_cases():
    names = [c.strip() for c in os.environ.get("BENCH_CASES", "c49").split(",") if c.strip()]
    return names


@pytest.mark.parametrize("case_id", _bench_cases())
def test_bench(device, monkeypatch, case_id):
    scenario = _scenario(case_id)
    shape = tuple(scenario["shape"])
    assert scenario["layout"] == ttnn.ROW_MAJOR_LAYOUT and scenario["dtype"] == ttnn.bfloat16
    order = [v.strip() for v in os.environ.get("BENCH_ORDER", "op,cand-fast-raw-w2-b16-d2,stock").split(",")]
    launches = int(os.environ.get("BENCH_LAUNCHES", "4"))
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = helpers.create_ttnn_input_tensor(
        x,
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=helpers.input_memory_config(scenario, device),
    )
    kwargs = helpers.op_kwargs(scenario)
    LOG_DIR.mkdir(exist_ok=True)
    log = LOG_DIR / os.environ.get("BENCH_LOG", "bench.jsonl")
    for variant in order:
        with monkeypatch.context() as m:
            if variant.startswith("cand"):
                candidate.install(m, candidate.Cfg.parse(variant))
                fn = lambda t: g6_to_layout(t, ttnn.TILE_LAYOUT, **kwargs)  # noqa: E731
            elif variant == "op":
                candidate.clear_caches()
                fn = lambda t: g6_to_layout(t, ttnn.TILE_LAYOUT, **kwargs)  # noqa: E731
            elif variant == "stock":
                fn = lambda t: ttnn.to_layout(t, ttnn.TILE_LAYOUT, **kwargs)  # noqa: E731
            else:
                raise ValueError(variant)
            plan = None
            if variant != "stock":
                plan = candidate.op_mod.select_plan(tt_x, ttnn.TILE_LAYOUT, **kwargs)
                plan = {
                    k: plan[k]
                    for k in ("regime", "num_cores", "block_rows_tiles", "block_cols_tiles", "l1_bytes_per_core")
                }
            ids = []
            ok = None
            programs = []
            ttnn.synchronize_device(device)
            _read_programs(device)  # flush whatever ran before
            for k in range(launches):
                before = _op_id()
                out = fn(tt_x)
                ttnn.synchronize_device(device)
                ids.append((before, _op_id()))
                programs.append(_read_programs(device))
                if k == launches - 1:
                    got = ttnn.to_torch(out).reshape(shape)
                    ok = bool(torch.equal(_bits(got), _bits(x)))
                ttnn.deallocate(out)
            candidate.clear_caches()
        rec = {
            "t": time.time(),
            "case": case_id,
            "shape": shape,
            "variant": variant,
            "ids": ids,
            "programs": programs,
            "exact": ok,
            "plan": plan,
        }
        print(f"\nBENCH {json.dumps(rec)}", flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if variant != "stock" and "-a" not in variant:
            assert ok, f"{variant} not bit-exact on {case_id}"
    ttnn.deallocate(tt_x)


@pytest.mark.parametrize("variant", ["cand-fastapi-none-w0-b16-d2-u16", "cand-fast-raw-w0-b16-d2"])
def test_diag_map(device, monkeypatch, variant):
    """Where does each datum land? value = (row % 32) * 256 + col, unique inside a tile-row."""
    shape = (1, 1, 32 * 28, 256)
    r = torch.arange(shape[2]).remainder(32).unsqueeze(1)
    c = torch.arange(256).unsqueeze(0)
    code = (r * 256 + c).to(torch.int32)
    x = code.to(torch.int16).view(torch.bfloat16).reshape(shape)
    with monkeypatch.context() as m:
        candidate.install(m, candidate.Cfg.parse(variant))
        plan = candidate.op_mod.select_plan(
            helpers.create_ttnn_input_tensor(
                x, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            ttnn.TILE_LAYOUT,
        )
        print(f"\nDIAGMAP plan block={plan['block_rows_tiles']}x{plan['block_cols_tiles']} cores={plan['num_cores']}")
        for core in plan["cores"][:3]:
            print(f"DIAGMAP core {core['core']} tiles [{core['tile_begin']}, +{core['num_tiles']})")
        mismatch, got = _run_and_check(device, x, ttnn.BufferType.DRAM, lambda t: g6_to_layout(t, ttnn.TILE_LAYOUT))
    candidate.clear_caches()
    g = _bits(got).reshape(shape[2], 256).to(torch.int32) & 0xFFFF
    want = code
    bad = g != want
    print(f"DIAGMAP variant={variant} mismatches={int(bad.sum())}/{g.numel()}")
    t0 = bad[0:32, 0:32]
    print(f"DIAGMAP tile(0,0) bad per row {t0.sum(1).tolist()}")
    print(f"DIAGMAP tile(0,0) bad per col {t0.sum(0).tolist()}")
    for rr in (0, 1, 8, 9, 16, 24):
        print(f"DIAGMAP tile(0,0) row {rr} cols 0..31 from {[(int(v) // 256, int(v) % 256) for v in g[rr, 0:32:4]]}")
    t1 = g[0:32, 32:64]
    for rr in (0, 1, 2, 8, 16):
        print(
            f"DIAGMAP tile(0,1) row {rr} cols 0..31 step4 from {[(int(v) // 256, int(v) % 256) for v in t1[rr, 0:32:4]]}"
        )
    # Tile-level census over the first 2 tile-rows: (tile-row, tile col) -> mismatches, and a sample.
    for tr in range(2):
        for tc in range(8):
            blk = bad[tr * 32 : (tr + 1) * 32, tc * 32 : (tc + 1) * 32]
            n = int(blk.sum())
            if n:
                gg = g[tr * 32 : (tr + 1) * 32, tc * 32 : (tc + 1) * 32]
                src = [(int(v) // 256, int(v) % 256) for v in gg[[0, 1, 8, 16], 0]]
                print(f"DIAGMAP tile ({tr},{tc}) bad={n} rows0,1,8,16 col0 came from (row,col) {src}")
