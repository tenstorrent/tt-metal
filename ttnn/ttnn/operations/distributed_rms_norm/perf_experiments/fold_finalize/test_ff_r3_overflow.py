# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf_experiments/fold_finalize copy of test_distributed_rms_norm_r3.py, run on the candidate wire.
Refinement 3 pin: the compact inv all-gather uses a zeroed record slot as its arrival flag (inv = rsqrt(ss/W + eps)
is never +0). A row whose sum of squares overflows to +inf has inv = 0 — finalize writes it as -0, so the slot still
reads as "arrived". Without that, this case hangs (the members wait on the slot forever).

Rows 0..31 of every device are huge (|x| ~ 1e30: x^2 overflows fp32), the rest are randn: the op must complete and
every other row must hold the acceptance PCC. (Measured: the Tensix FPU / SFPU saturate rather than produce inf, so
this inv is a large finite value, not 0 — the -0 guard is defensive.) Run with --no-precompile (module-level
from_torch inputs read back as zeros under the up-front precompile collector, as for the debug drivers).
"""

from __future__ import annotations

import os

import pytest
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.topology import system_mesh_shape
from ttnn.operations.distributed_rms_norm.perf_experiments.fold_finalize.drms_ff import distributed_rms_norm

os.environ.setdefault("DRMS_FF_VARIANT", "compact")  # the candidate fold (perf_experiments copy)

_SYSTEM = system_mesh_shape()
pytestmark = [
    pytest.mark.skipif(_SYSTEM[0] * _SYSTEM[1] < 2, reason="distributed_rms_norm needs a multi-device system"),
    pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile"),
]


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "bf16acc"])
@pytest.mark.parametrize("M, W_local", [(96, 512), (640, 1792)])
def test_overflowed_row_sum_does_not_hang(mesh_cache, M, W_local, fp32_dest_acc_en):
    import torch  # lazy: no global torch import under ttnn/ (pre-commit)

    cluster_axis = 1 if _SYSTEM[1] >= 2 else 0
    mesh = mesh_cache.get(_SYSTEM)
    rows, cols = _SYSTEM
    x = torch.randn((rows, cols, 1, 1, M, W_local), dtype=torch.float32)
    x[..., :32, :] = 1e30
    x = x.to(torch.bfloat16)
    glob = torch.cat([torch.cat(list(x[r]), dim=1) for r in range(rows)], dim=0)
    tx = ttnn.from_torch(
        glob,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False
    )
    out = distributed_rms_norm(
        tx, cluster_axis=cluster_axis, epsilon=1e-6, topology=ttnn.Topology.Linear, compute_kernel_config=cfg
    )
    devs = ttnn.get_device_tensors(out)
    for r in range(rows):
        for c in range(cols):
            # the group of device (r, c): its mesh row (cluster_axis 1) or mesh column (cluster_axis 0)
            grp = [x[r, j] for j in range(cols)] if cluster_axis == 1 else [x[i, c] for i in range(rows)]
            full = torch.cat([g.float() for g in grp], dim=-1)
            ref = torch.nn.functional.rms_norm(full, (full.shape[-1],), eps=1e-6)
            p = c if cluster_axis == 1 else r
            ref = ref[..., p * W_local : (p + 1) * W_local]
            dev = ttnn.to_torch(devs[r * cols + c]).float().reshape(ref.shape)
            # rows < 32: the FPU / SFPU saturate instead of producing inf, so the device returns finite (garbage)
            # values there where torch returns 0 — pre-existing arithmetic, not pinned. What is pinned: no hang, and
            # every other row (other tile-rows of the same cores and waves) is right.
            a, b = dev[..., 32:, :].flatten(), ref[..., 32:, :].flatten()
            assert torch.isfinite(b).all(), f"reference not finite: {(~torch.isfinite(b)).sum()}"
            bad = (~torch.isfinite(dev[..., 32:, :])).nonzero()
            assert torch.isfinite(a).all(), f"device ({r},{c}) non-finite at {bad[:4].tolist()} n={len(bad)}"
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            assert pcc > 0.995, f"device ({r},{c}) pcc {pcc}"
