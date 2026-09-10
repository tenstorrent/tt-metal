"""passb_op_count -- correctness gate for the pass-B reorder, on the SUITE's own runner.

Perf is measured in bench.py; this file only answers "is the reorder correct where
the op is correct?".  It drives `eval.golden_tests.rms_norm_ttnn.helpers.run_rms_norm_ttnn`
(the golden suite's own harness, including its per-case pcc threshold) over a
STRUCTURAL subset of LOOSE_CASES chosen to hit every axis the reorder touches:
ROW_MAJOR activations (the untilize consumer downstream of pass B), non-tile-aligned
W and poisoned padding, a ROW_MAJOR / (Wt,32)-blocked weight, gamma+bias (the
in-place cb_normalized stage the reorder moves), all four placements (so both the
local-stat and the cross-core-stat gates are exercised), degenerate ranks, and the
caller-supplied program_config (subblock_w == PASS_B_BLK).

  RMS_VARIANTS=swapx scripts/tt-probe.sh rms_norm_ttnn <<'PY'
  import sys; sys.path.insert(0, "<this dir>"); import correctness; correctness.main()
  PY
"""

import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

from pathlib import Path

import ttnn

import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.golden_harness import parametrize_loose_cases
from eval.golden_tests.rms_norm_ttnn.feature_spec import LOOSE_CASES, INVALID
from eval.golden_tests.rms_norm_ttnn.helpers import run_rms_norm_ttnn
from ttnn.operations.rms_norm_ttnn import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED

HERE = Path(__file__).resolve().parent

# indices into LOOSE_CASES (1:1 with parametrize_loose_cases' order)
CASES = [
    1,  # 32x32768 interleaved f32       -- widest W / chunked pass B
    3,  # 256x512 HEIGHT_SHARDED f32     -- local stat, sharded
    5,  # 256x512 BLOCK_SHARDED f32      -- cross-core stat
    29,
    30,
    31,
    32,  # 224x3072 ROW_MAJOR activations, all 4 placements (untilize)
    61,
    64,  # 928x1696 ROW_MAJOR, interleaved + BLOCK
    100,  # 32x2848 BLOCK
    163,  # 352x1023 WIDTH  -- non-aligned W, cross-core
    204,  # 32x4095 BLOCK   -- non-aligned W
    361,
    363,  # 99991x64 interleaved + WIDTH -- extreme aspect
    369,
    371,
    372,  # pad_poison 32x40
    385,
    388,  # pad_poison 224x72
    397,
    399,
    400,  # weight_form gamma_bias, RM weight, 3 placements
    413,
    416,
    424,  # weight_form gamma_bias, RM weight, more shapes
    429,
    430,
    431,  # degenerate ranks / non-aligned / RM
    432,  # zero rows
    435,
    436,
    437,
    438,  # program_config (subblock_w == PASS_B_BLK)
    441,
    444,  # program_config BLOCK + WIDTH
]


def main():
    params = parametrize_loose_cases(LOOSE_CASES, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID)
    variants = [v for v in os.environ.get("RMS_VARIANTS", "swapx").split(",") if v]
    idxs = [int(i) for i in os.environ.get("RMS_CASES", ",".join(str(i) for i in CASES)).split(",")]
    saved = PD.KERNEL_DIR
    device = ttnn.open_device(device_id=0)
    npass = nfail = 0
    try:
        for v in variants:
            PD.KERNEL_DIR = HERE / f"k_{v}"
            for i in idxs:
                inputs, axes, extras = params[i].values
                tag = (
                    f"{v} #{i} {inputs[0]} {str(axes['layout']).split('.')[-1]} "
                    f"{str(axes['memory_layout']).split('.')[-1]} {axes['gamma_mode']}"
                )
                try:
                    run_rms_norm_ttnn(inputs, device=device, extras=extras, **axes)
                    npass += 1
                    print(f"RESULT PASS {tag}", flush=True)
                except Exception as e:
                    nfail += 1
                    msg = str(e).replace("\n", " ")[:200]
                    print(f"RESULT FAIL {tag} :: {type(e).__name__}: {msg}", flush=True)
            PD.KERNEL_DIR = saved
    finally:
        PD.KERNEL_DIR = saved
        ttnn.close_device(device)
    print(f"RESULT ==== {npass} passed, {nfail} failed ====")
