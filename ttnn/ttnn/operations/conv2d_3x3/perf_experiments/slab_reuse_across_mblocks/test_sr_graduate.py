# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Integration candidate check: graduate/ (the op with the core-wide TILE slab, no knobs) against the real op on the
same inputs, bit-exact (torch.equal). Not profiled: the copy and the real op share zone names.

  SR_COPY=graduate scripts/run_safe_pytest.sh --run-all <this file> --import-mode=prepend
"""

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SR_COPY", "graduate")
sys.path.insert(0, str(Path(__file__).parent))
import sr_load  # noqa: E402
import test_slab_reuse as t  # noqa: E402

assert sr_load.COPY == "graduate"

GRADUATE_CASES = [
    "focus",
    "sem_res2_fuse_conv1",
    "stem_conv3",
    "sem_res2_fuse_conv0",
    "res2_conv2",
    "center_head1",
    "stem_conv1",
    "dram_H128_W256_Ci64_Co64_tile_bf8b",
    "dram_H64_W128_Ci128_Co128_rm_bf16",
    "block_res4_conv2_lofi",
]


@pytest.mark.parametrize("case_id", GRADUATE_CASES)
def test_sr_graduate_vs_op(case_id, device, monkeypatch):
    import torch

    from ttnn.operations.conv2d_3x3 import conv2d_3x3 as real_conv, prepare_conv2d_bias, prepare_conv2d_weights

    case = t.CASES[case_id]
    spy = t._Spy()
    monkeypatch.setattr(t.op_mod, "select_blocking", spy)
    out_grad, ref = t.run_case(case, device, seed=2)
    core = spy.blocking.core_slab
    # The same inputs through the real op: swap the copy's entry points for the real ones.
    monkeypatch.setattr(t.opc, "conv2d_3x3", real_conv)
    monkeypatch.setattr(t.opc, "prepare_conv2d_weights", prepare_conv2d_weights)
    monkeypatch.setattr(t.opc, "prepare_conv2d_bias", prepare_conv2d_bias)
    out_real, _ = t.run_case(case, device, seed=2)
    same = bool(torch.equal(out_grad, out_real))
    pcc = t._pcc(out_grad, ref)
    with open(t.RESULTS, "a") as f:
        f.write(json.dumps(dict(kind="graduate", case=case_id, core_slab=core, identical=same, pcc=pcc)) + "\n")
    assert same, f"{case_id}: graduate differs from the real op"
    assert pcc >= case["pcc"]
