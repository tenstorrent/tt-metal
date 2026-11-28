# small_tensor_split bench: device kernel ns of g6_to_layout under candidate work splits, and of
# stock ttnn.to_layout, on the production cells of feature_spec.INPUTS. In-process device profiler
# (perf-lab method): per launch, ReadDeviceProfiler -> DEVICE KERNEL DURATION of that program.
# Launch 1 of every (cell, variant) compiles and is checked bit-exact; launches 2..N are timed and
# the median is reported. Passes alternate the variant order (forward, reverse, ...).
#
#   SPLIT_CELLS     comma list of production case ids (c2025,c2006,...) or extra ids (x1, ...)
#   SPLIT_VARIANTS  comma list of variants of splits.VARIANTS plus "stock"
#   SPLIT_LAUNCHES  launches per (cell, variant, pass), default 12
#   SPLIT_PASSES    passes, default 2 (forward, then reverse order)
#   SPLIT_OUT       JSON-lines output file (appended), default results.jsonl next to this file
#   SPLIT_TAG       tag stored with every record
# Variants: splits.VARIANTS, rect<b>x<m>, rec (the recommended rule, splits.recommended_split), stock.
# Extra geometries: x* / d* (bf16 TILE -> RM, DRAM; d8 in L1), r* (bfloat8_b TILE -> RM, DRAM).
# Summary: python3 summarize.py [--all]
#
# Run (from the slot root, TT_METAL_HOME / PYTHONPATH exported to it):
#   SPLIT_CELLS=c2025 SPLIT_VARIANTS=s1,rowflat,stock scripts/run_safe_pytest.sh \
#       ttnn/ttnn/operations/g6_to_layout/perf_experiments/small_tensor_split/test_split_bench.py::test_split_bench
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
# Own device-log dir: siblings in this checkout write generated/profiler/.logs concurrently.
os.environ.setdefault("TT_METAL_PROFILER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiler"))

import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import pytest

# Test-only dependency; the ttnn package forbids a module-level torch import.
torch = pytest.importorskip("torch")
import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import splits  # noqa: E402
import subprocess  # noqa: E402

# kernels_m<mode>[_nz]/ are generated from kernels/ (one directory per compile-time split mode).
if not (HERE / "kernels_m3").is_dir():
    subprocess.run([str(HERE / "sync_kernels.sh")], check=True)

importlib.import_module("ttnn.operations.g6_to_layout")
op_mod = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd  # noqa: E402
from ttnn.operations.g6_to_layout import g6_to_layout  # noqa: E402

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
BF16, BF8B = ttnn.bfloat16, ttnn.bfloat8_b
TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


def _production_cells():
    from eval.golden_tests.g6_to_layout import feature_spec as spec

    cells = {}
    for (entry,) in spec.INPUTS:
        cells[f"c{entry['case']}"] = entry
    return cells


# Extra correctness geometries for the experiment kernels (H % 32 != 0 multi-image folds, W % 32 != 0,
# H < 32). TILE bf16 DRAM -> RM.
_EXTRA = {
    "x1": (3, 45, 100),
    "x2": (2, 7, 300),
    "x3": (5, 33, 70),
    "x4": (1, 2, 96, 1000),
    "x5": (4, 20, 40),
    # domain sweep of the band rule (fewer tiles than cores)
    "d1": (1, 32, 32),
    "d2": (1, 64, 32),
    "d3": (2, 32, 64),
    "d4": (1, 3, 64),
    "d5": (1, 32, 1696),
    "d6": (3, 17, 260),
    "d7": (1, 1, 32, 1000),
    "d8": (1, 32, 512),  # L1 in and out (see _L1_EXTRA)
}
_L1_EXTRA = {"d8"}
# bfloat8_b (packer_untilize) extras, DRAM in / out
_BF8B_EXTRA = {"r1": (1, 1024, 256), "r2": (1, 2048, 256), "r3": (1, 4096, 256), "r4": (1, 256, 1024)}
_EXTRA.update(_BF8B_EXTRA)


def _cell(cell_id, cells):
    if cell_id in _EXTRA:
        return {
            "shape": _EXTRA[cell_id],
            "dtype": BF8B if cell_id in _BF8B_EXTRA else BF16,
            "layout": TILE,
            "buffer": ttnn.BufferType.L1 if cell_id in _L1_EXTRA else ttnn.BufferType.DRAM,
            "target_layout": RM,
            "out_dtype": None,
            "memory_config": None,
        }
    return cells[cell_id]


_ANALYSES = []  # every analysis of the last read: [{name: ns}] per program


def _read_programs(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    out = []
    _ANALYSES.clear()
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is not None:
                out.append(float(entry.duration))
                _ANALYSES.append({k: float(v.duration) for k, v in results.items()})
    return out


def _make_input(scn, device):
    shape = tuple(scn["shape"])
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    mem = ttnn.L1_MEMORY_CONFIG if scn["buffer"] == ttnn.BufferType.L1 else ttnn.DRAM_MEMORY_CONFIG
    tt_x = ttnn.from_torch(x, dtype=scn["dtype"], layout=scn["layout"], device=device, memory_config=mem)
    if scn["dtype"] == BF8B:
        ref = ttnn.to_torch(tt_x).reshape(shape)  # the host's bfp8 decode
    else:
        ref = x
    return tt_x, ref


def _kwargs(scn):
    kw = {}
    if scn.get("out_dtype") is not None:
        kw["dtype"] = scn["out_dtype"]
    if scn.get("memory_config") is not None:
        kw["memory_config"] = scn["memory_config"]
    return kw


def _exact(out, ref):
    got = ttnn.to_torch(out).reshape(ref.shape)
    if got.dtype != ref.dtype:
        got = got.to(ref.dtype)
    return bool(torch.equal(got.view(torch.int16), ref.view(torch.int16)))


class _Patch:
    """Applies one split variant to the program descriptor for the duration of a with-block."""

    def __init__(self, variant):
        self.variant = variant
        self.saved = None
        self.last = None

    def __enter__(self):
        fn = splits.lookup(self.variant)
        self.saved = (pd._tile_split, pd._choose_blocking, pd.KERNEL_DIR)
        if fn is None:
            return self
        orig_choose = pd._choose_blocking
        patch = self

        def tile_split(grid_x, grid_y, geometry):
            s = fn((grid_x, grid_y), geometry)
            patch.last = s
            # every variant uses the row-wise prefix of the grid (splits._cores)
            all_cores = ttnn.num_cores_to_corerangeset(s.num_cores, ttnn.CoreCoord(grid_x, grid_y), True)
            assert len(s.core_args) == s.num_cores
            return s.num_cores, all_cores, s.core_args, s.max_tiles

        def choose_blocking(geometry, core_tiles_max, max_block_tiles, blocks_per_core):
            s = patch.last
            if s is not None and s.blocking is not None:
                rows, cols = s.blocking
                assert rows * cols <= max_block_tiles, (s.blocking, max_block_tiles)
                return s.blocking
            return orig_choose(geometry, core_tiles_max, max_block_tiles, blocks_per_core)

        pd._tile_split = tile_split
        pd._choose_blocking = choose_blocking
        # Probe the split to learn whether it needs the experiment kernels.
        return self

    def set_kernels(self):
        if self.last is not None and self.last.kernel_mode is not None:
            pd.KERNEL_DIR = HERE / f"kernels_m{self.last.kernel_mode}"

    def __exit__(self, *exc):
        pd._tile_split, pd._choose_blocking, pd.KERNEL_DIR = self.saved
        op_mod._PLAN_CACHE.clear()
        op_mod._PROGRAM_CACHE.clear()


def _run_variant(device, cell_id, scn, variant, tt_x, ref, launches):
    target = scn["target_layout"]
    kw = _kwargs(scn)
    rec = {"cell": cell_id, "variant": variant, "shape": list(scn["shape"])}
    if variant == "stock":
        run = lambda: ttnn.to_layout(tt_x, target, **kw)  # noqa: E731
    else:
        op_mod._PLAN_CACHE.clear()
        op_mod._PROGRAM_CACHE.clear()
        run = lambda: g6_to_layout(tt_x, target, **kw)  # noqa: E731
    if variant == "rec":
        # the regime the op selects for this call (unpatched plan), for the recommended rule
        op_mod._PLAN_CACHE.clear()
        out_dtype0 = op_mod._output_dtype(tt_x, target, kw.get("dtype"))
        out_mc0 = op_mod._output_memory_config(tt_x, kw.get("memory_config"))
        splits.CURRENT["regime"] = op_mod._plan(tt_x, target, out_dtype0, out_mc0)[1].regime
        op_mod._PLAN_CACHE.clear()
        op_mod._PROGRAM_CACHE.clear()
    patch = _Patch(variant if variant != "stock" else "s1")
    with patch:
        if variant != "stock":
            # Plan once to learn the kernel dir, then build with it.
            out_dtype = op_mod._output_dtype(tt_x, target, kw.get("dtype"))
            out_mc = op_mod._output_memory_config(tt_x, kw.get("memory_config"))
            _, plan = op_mod._plan(tt_x, target, out_dtype, out_mc)
            patch.set_kernels()
            rec["regime"] = plan.regime
            rec["num_cores"] = plan.num_cores
            rec["block"] = [plan.block_rows_tiles, plan.block_cols_tiles]
            rec["kernel_mode"] = None if patch.last is None else patch.last.kernel_mode
            if patch.last is not None and patch.last.kernel_mode is not None and plan.regime != pd.STAGED_REORDER:
                rec["skipped"] = f"experiment kernels need staged_reorder, plan is {plan.regime}"
                return rec
            rec["args_head"] = [list(a) for _, a in plan.core_args[:4]]
        ttnn.synchronize_device(device)
        _read_programs(device)  # flush
        out = run()
        ttnn.synchronize_device(device)
        first = _read_programs(device)
        rec["exact"] = _exact(out, ref)
        ttnn.deallocate(out)
        times = []
        per_key = {}
        for _ in range(launches - 1):
            out = run()
            ttnn.synchronize_device(device)
            progs = _read_programs(device)
            ttnn.deallocate(out)
            times.append(sum(progs) if progs else None)
            if len(_ANALYSES) == 1:
                for k, v in _ANALYSES[0].items():
                    per_key.setdefault(k, []).append(v)
        rec["analyses"] = {k: statistics.median(v) for k, v in per_key.items()}
        rec["programs_per_launch"] = len(first)
        valid = [t for t in times if t is not None]
        rec["ns"] = statistics.median(valid) if valid else None
        rec["ns_min"] = min(valid) if valid else None
        rec["ns_all"] = valid
    return rec


def test_split_bench(device):
    cells = _production_cells()
    cell_ids = [c.strip() for c in os.environ.get("SPLIT_CELLS", "c2025").split(",") if c.strip()]
    variants = [v.strip() for v in os.environ.get("SPLIT_VARIANTS", "s1,stock").split(",") if v.strip()]
    launches = int(os.environ.get("SPLIT_LAUNCHES", "12"))
    passes = int(os.environ.get("SPLIT_PASSES", "2"))
    out_path = Path(os.environ.get("SPLIT_OUT", str(HERE / "results.jsonl")))
    tag = os.environ.get("SPLIT_TAG", time.strftime("%Y%m%d-%H%M%S"))
    grid = device.compute_with_storage_grid_size()
    print(f"\nSPLIT_DEVICE arch={device.arch()} grid={grid.x}x{grid.y}")
    failures = []
    for cell_id in cell_ids:
        scn = _cell(cell_id, cells)
        tt_x, ref = _make_input(scn, device)
        for p in range(passes):
            order = variants if p % 2 == 0 else list(reversed(variants))
            for variant in order:
                try:
                    rec = _run_variant(device, cell_id, scn, variant, tt_x, ref, launches)
                except Exception as err:  # noqa: BLE001
                    rec = {"cell": cell_id, "variant": variant, "error": f"{type(err).__name__}: {err}"[:500]}
                rec["pass"] = p
                rec["tag"] = tag
                with out_path.open("a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                ns = rec.get("ns")
                print(
                    f"SPLIT {cell_id:6s} {variant:12s} pass={p} ns={ns if ns is None else round(ns)} "
                    f"exact={rec.get('exact')} cores={rec.get('num_cores')} block={rec.get('block')} "
                    f"regime={rec.get('regime')} err={rec.get('error')}",
                    flush=True,
                )
                if "error" in rec or (variant != "stock" and rec.get("exact") is False):
                    failures.append((cell_id, variant, rec.get("error"), rec.get("exact")))
        ttnn.deallocate(tt_x)
    assert not failures, failures
