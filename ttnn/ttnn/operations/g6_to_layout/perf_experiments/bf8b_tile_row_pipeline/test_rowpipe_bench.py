# Isolated bake-off bench of the bf8b TILE -> RM pipeline (R3 packer_untilize, R4 sharded_in_place).
#
# One process = one kernel-dir variant. Each selected production case runs through the golden suite's
# own runner (eval/golden_tests/g6_to_layout/helpers.run_g6_to_layout: production placement, production
# keywords, exact bit gate against the host's bf16 rendering of the bf8b input), with the op dispatched
# G6_LAUNCHES times on the same input; launch 1's result is the one checked, launches 2..N are timings
# (ops_csv.py summary --skip-first 1).
#
# Environment:
#   G6_KDIR       kernel dir (relative to this folder or absolute). Default: kernels_baseline.
#   G6_MARKS_ROW  1: cb_nan_marks paged per tile-row (16-B summary header + one NanMarks record per
#                 tile column); the CB keeps its byte budget (pages = depth x block_rows_tiles).
#   G6_CASES      comma list of production case ids (feature_spec INPUTS), e.g. 37,20,29.
#   G6_EXTRAS     JSON extras for the runner, e.g. {"input_gen":"bf16_specials","seed":1234,"gate":"exact"}.
#   G6_LAUNCHES   dispatches per case (default 6).
#   G6_IMPL       new (default) | stock (ttnn.to_layout black box).
#   G6_SHAPE      JSON shape overriding the case's (correctness probes of ragged shapes), e.g. [1,2,50,96].
#   G6_SHARD      JSON shard placement overriding the case's: {"layout": "HEIGHT"|"BLOCK",
#                 "grid": [[[x0, y0], [x1, y1]], ...], "shard_shape": [rows, cols]}.
#
# Run (from the tt-metal root, TT_METAL_HOME / PYTHONPATH exported to it):
#   G6_KDIR=kernels_rowpipe G6_MARKS_ROW=1 G6_CASES=37 scripts/run_safe_pytest.sh --profile --no-py-zones \
#       ttnn/ttnn/operations/g6_to_layout/perf_experiments/bf8b_tile_row_pipeline/test_rowpipe_bench.py
import importlib
import json
import os
import random
import sys
import zlib
from pathlib import Path

import pytest

# Test-only dependency; the ttnn package forbids a module-level torch import.
torch = pytest.importorskip("torch")
import ttnn

importlib.import_module("ttnn.operations.g6_to_layout")
op_mod = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd

SPEC = importlib.import_module("eval.golden_tests.g6_to_layout.feature_spec")
HELPERS = importlib.import_module("eval.golden_tests.g6_to_layout.helpers")

HERE = Path(__file__).resolve().parent
KDIR = Path(os.environ.get("G6_KDIR", "kernels_baseline"))
KDIR = KDIR if KDIR.is_absolute() else HERE / KDIR
MARKS_ROW = os.environ.get("G6_MARKS_ROW", "0") == "1"
CASES = [c.strip() for c in os.environ.get("G6_CASES", "37").split(",") if c.strip()]
EXTRAS = json.loads(os.environ.get("G6_EXTRAS", "null"))
LAUNCHES = int(os.environ.get("G6_LAUNCHES", "6"))
IMPL = os.environ.get("G6_IMPL", "new")
SHAPE = json.loads(os.environ.get("G6_SHAPE", "null"))
SHARD = json.loads(os.environ.get("G6_SHARD", "null"))

# Per-tile-row marks page: a 16-B header (word 0 = OR of the tile-row's NaN group bitmaps) + one
# NanMarks record per tile column (mirrors kernels_rowpipe/g6_to_layout_bfp8_nan_marks.hpp).
MARKS_ROW_HEADER_BYTES = 16


def _row_marks_page_bytes(block_cols_tiles):
    return pd._round_up(MARKS_ROW_HEADER_BYTES + block_cols_tiles * pd.NAN_MARK_RECORD_BYTES, pd.NAN_MARKS_PAGE_ALIGN)


def _install_row_marks(monkeypatch):
    real_compute_plan = op_mod.compute_plan

    def compute_plan(*args, **kwargs):
        plan = real_compute_plan(*args, **kwargs)
        if plan.regime in pd._UNTILIZE_REGIMES:
            whole_page = pd._nan_marks_page_bytes(plan.block_rows_tiles * plan.block_cols_tiles)
            depth_blocks = plan.cb_bytes[pd.CB_NAN_MARKS] // whole_page
            plan.cb_bytes[pd.CB_NAN_MARKS] = (
                depth_blocks * plan.block_rows_tiles * _row_marks_page_bytes(plan.block_cols_tiles)
            )
        return plan

    monkeypatch.setattr(op_mod, "compute_plan", compute_plan)
    monkeypatch.setattr(
        pd.Plan, "nan_marks_page_bytes", property(lambda self: _row_marks_page_bytes(self.block_cols_tiles))
    )


def _stock(input_tensor, layout, **kwargs):
    return ttnn.to_layout(input_tensor, layout, **kwargs)


@pytest.mark.use_module_device
@pytest.mark.parametrize("case", CASES)
def test_bench(case, device, monkeypatch):
    monkeypatch.setattr(pd, "KERNEL_DIR", KDIR)
    if MARKS_ROW:
        _install_row_marks(monkeypatch)
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
    seed = zlib.crc32(f"c{case}".encode()) & 0x7FFFFFFF
    torch.manual_seed(seed)
    random.seed(seed)

    scenario = SPEC.scenario_of(int(case))
    if SHAPE is not None:
        scenario["shape"] = tuple(SHAPE)
    if SHARD is not None:
        scenario["memory_layout"] = getattr(ttnn.TensorMemoryLayout, SHARD["layout"] + "_SHARDED")
        scenario["shard_spec"] = {
            "grid": [tuple(tuple(c) for c in r) for r in SHARD["grid"]],
            "shard_shape": tuple(SHARD["shard_shape"]),
            "orientation": ttnn.ShardOrientation.ROW_MAJOR,
        }
    real_op = _stock if IMPL == "stock" else HELPERS.g6_to_layout
    calls = []

    def repeated(*args, **kwargs):
        kept = real_op(*args, **kwargs)
        ttnn.synchronize_device(device)
        for _ in range(LAUNCHES - 1):
            out = real_op(*args, **kwargs)
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)
        calls.append(1)
        return kept

    monkeypatch.setattr(HELPERS, "g6_to_layout", repeated)
    print(
        f"\nRB case=c{case} impl={IMPL} kdir={KDIR.name} marks_row={int(MARKS_ROW)} extras={EXTRAS} "
        f"shape={scenario['shape']} layout={scenario['memory_layout']} shard={scenario['shard_spec']}",
        flush=True,
    )
    HELPERS.run_g6_to_layout(
        (scenario,),
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        output_dtype=HELPERS.expected_output_dtype(scenario),
        memory_layout=scenario["memory_layout"],
        device=device,
        extras=EXTRAS,
    )
    assert calls, "runner never called the op"
    print(f"RB_OK case=c{case} impl={IMPL} kdir={KDIR.name} exact check passed", flush=True)
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
