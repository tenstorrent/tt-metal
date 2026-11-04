# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pv_rescale_fusion — whole-op A/B on the 1x4 mesh: baseline = the real op, candidate = op_copy/ (the op with the
fused_l compute kernel; PVR_OP_VARIANT=0 runs the copy with the op's current P.V + rescale as a sanity baseline).

  PVR_IMPL=base|cand (default cand)
  Perf (one test per session; analyze.py reads the LAST run of each chip):
    TT_METAL_DEVICE_PROFILER=1 PVR_PERF=1 scripts/run_safe_pytest.sh <this file> -k focus -s
    python <dir>/analyze.py out.json
  Correctness: scripts/run_safe_pytest.sh --run-all <this file> -k domain
Every call is checked against the golden reference (eval helpers.check_output, TOLERANCES).
"""
import os

import pytest
import ttnn

import eval.golden_tests.ring_joint_scaled_dot_product_attention.helpers as H
from ttnn.operations.ring_joint_scaled_dot_product_attention import ring_joint_scaled_dot_product_attention as base_op
from ttnn.operations.ring_joint_scaled_dot_product_attention.perf_experiments.pv_rescale_fusion.op_copy import (
    ring_joint_scaled_dot_product_attention as cand_op,
)

IMPL = os.environ.get("PVR_IMPL", "cand")
PERF = os.environ.get("PVR_PERF", "0") == "1"
RING14 = {"mesh_shape": (1, 4), "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}


def _axes(**kw):
    a = dict(
        dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        fp32_dest_acc_en=False,
        layout=ttnn.TILE_LAYOUT,
        joint_mode="none",
        mask_mode="none",
        is_balanced=False,
        seq_padding="exact",
        cluster_axis=1,
        topology=ttnn.Topology.Ring,
    )
    a.update(kw)
    return a


def _qkv(b, h, n, d, jl=None):
    t = ((b, h, n, d),) * 3
    return t + (((b, h, jl, d),) if jl else ())


FOCUS = (
    _qkv(1, 40, 8192, 128),
    _axes(seq_padding="padded"),
    {"num_links": 2, "math_fidelity": ttnn.MathFidelity.HiFi2, "logical_n": 32760, "sample_rows": 256},
)

DOMAIN = [
    ("min_1x1x64x64", _qkv(1, 1, 64, 64), _axes(), {}),
    ("small_1x2x128x64_padded", _qkv(1, 2, 128, 64), _axes(seq_padding="padded"), {}),
    ("wan_1x10x512x128", _qkv(1, 10, 512, 128), _axes(), {}),
    ("wan_1x10x512x128_linear", _qkv(1, 10, 512, 128), _axes(topology=ttnn.Topology.Linear), {}),
    ("ltx_1x8x1024x128_fp32", _qkv(1, 8, 1024, 128), _axes(fp32_dest_acc_en=True), {}),
    ("bf8kv_1x10x512x128", _qkv(1, 10, 512, 128), _axes(kv_dtype=ttnn.bfloat8_b), {}),
    ("joint_1x6x1024x128_L128", _qkv(1, 6, 1024, 128, 128), _axes(joint_mode="replicated"), {}),
    ("flux_joint_1x6x512x128_L512", _qkv(1, 6, 512, 128, 512), _axes(joint_mode="replicated"), {}),
    ("U_just_above_C_1x27x256x64", _qkv(1, 27, 256, 64), _axes(), {}),
    ("ragged_q_1x24x288x64_padded", _qkv(1, 24, 288, 64), _axes(seq_padding="padded"), {}),
    ("padded_1x27x256x64", _qkv(1, 27, 256, 64), _axes(seq_padding="padded"), {}),
    ("dh256_1x5x384x256", _qkv(1, 5, 384, 256), _axes(), {}),
    ("dh96_1x10x512x96", _qkv(1, 10, 512, 96), _axes(), {}),
]


def _run(inputs, axes, extras, mesh_device, calls):
    H.ring_joint_scaled_dot_product_attention = cand_op if IMPL == "cand" else base_op
    for _ in range(calls):
        H.run_ring_joint_scaled_dot_product_attention(inputs, mesh_device=mesh_device, extras=extras, **axes)
        ttnn.synchronize_device(mesh_device)
        if PERF:
            ttnn.ReadDeviceProfiler(mesh_device)


@pytest.mark.parametrize("mesh_shape,fabric_config", [(RING14["mesh_shape"], RING14["fabric_config"])])
def test_focus(mesh_shape, fabric_config, mesh_device):
    inputs, axes, extras = FOCUS
    _run(inputs, axes, extras, mesh_device, 2 if PERF else 1)


@pytest.mark.parametrize("mesh_shape,fabric_config", [(RING14["mesh_shape"], RING14["fabric_config"])])
@pytest.mark.parametrize("case", DOMAIN, ids=[c[0] for c in DOMAIN])
def test_domain(case, mesh_shape, fabric_config, mesh_device):
    _, inputs, axes, extras = case
    _run(inputs, axes, extras, mesh_device, 2 if PERF else 1)
