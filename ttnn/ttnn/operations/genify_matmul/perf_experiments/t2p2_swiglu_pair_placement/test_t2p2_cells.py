# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t2p2 extra domain cells: swiglu shapes outside the perf files that take the bank split under the graduation
(unit-test shapes, bf16 B). Same body as perf/test_genify_matmul_perf_r15.py; GENIFY_MATMUL_ONLY selects by id;
GENIFY_MATMUL_CHECK=1 prints PCC."""
import math
import os

import pytest
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

# (id, m, k, n, A dtype, B dtype, fidelity, A in L1)
CELLS = [
    ("u128x4096x4096", 128, 4096, 4096, "bfloat16", "bfloat16", "HiFi2", True),
    ("u256x2048x3072", 256, 2048, 3072, "bfloat16", "bfloat16", "HiFi2", False),
    ("u128x2048x4096", 128, 2048, 4096, "bfloat16", "bfloat16", "HiFi4", False),
]


@pytest.mark.parametrize("case", CELLS, ids=[c[0] for c in CELLS])
def test_t2p2_cell(case, device, request):
    import torch  # local: ttnn's global-torch-import check

    only = [t for t in os.environ.get("GENIFY_MATMUL_ONLY", "").split(",") if t]
    if only and request.node.callspec.id not in only:
        pytest.skip("not selected")
    _, m, k, n, a_dt, w_dt, fid, a_l1 = case
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    a_mc = ttnn.L1_MEMORY_CONFIG if a_l1 else ttnn.DRAM_MEMORY_CONFIG
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(
        ta,
        tb,
        dtype=ttnn.bfloat16,
        compute_kernel_config=cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        activation="swiglu",
    )
    ttnn.synchronize_device(device)
    if os.environ.get("GENIFY_MATMUL_CHECK") == "1":
        ref = ttnn.to_torch(ta).float() @ ttnn.to_torch(tb).float()
        t = ref.reshape(*ref.shape[:-1], n // 64, 2, 32)
        ref = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(*ref.shape[:-1], n // 2)
        got = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        print(f"CHECK {request.node.callspec.id} pcc={pcc:.6f}")
        assert pcc > 0.999, pcc
