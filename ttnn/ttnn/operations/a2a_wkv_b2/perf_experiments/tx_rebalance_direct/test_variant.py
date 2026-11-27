# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for the tx_rebalance_direct variant: the acceptance test's seeds, explicit config
and back-to-back bit-identity, against the same fp64 reference and tolerance."""
import os
import sys

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variant_op import a2a_wkv_b2_variant, config_tag  # noqa: E402
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import (  # noqa: E402
    MESH_SHAPE,
    _make_inputs,
    _expected,
    _check,
    _fabric,
    _mesh,
)


@_fabric
@_mesh
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_variant_default(mesh_device, seed):
    print("\nCONFIG", config_tag())
    chips, attn, w = _make_inputs(mesh_device, seed)
    out = a2a_wkv_b2_variant(attn, w)
    _check(out, _expected(mesh_device, chips, w), mesh_device)


@_fabric
@_mesh
@pytest.mark.parametrize("scale", [0.01, 8.0], ids=["small_values", "large_values"])
def test_variant_explicit(mesh_device, scale):
    chips, attn, w = _make_inputs(mesh_device, 3, scale=scale)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    out = a2a_wkv_b2_variant(attn, w, compute_kernel_config=cfg)
    _check(out, _expected(mesh_device, chips, w), mesh_device)


@_fabric
@_mesh
def test_variant_back_to_back(mesh_device):
    import torch

    sets = [_make_inputs(mesh_device, s) for s in (4, 5)]
    first = []
    for chips, attn, w in sets:
        out = a2a_wkv_b2_variant(attn, w)
        first.append(_check(out, _expected(mesh_device, chips, w), mesh_device))
        ttnn.deallocate(out)
    for i in range(8):
        k = i % 2
        _, attn, w = sets[k]
        out = a2a_wkv_b2_variant(attn, w)
        got = [ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
        for chip, (a, b) in enumerate(zip(got, first[k])):
            assert torch.equal(a, b), f"call {i + 1}: chip {chip} differs from the first call"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X, "trace_region_size": 1 << 20}],
    indirect=True,
)
@_mesh
def test_variant_trace_correct(mesh_device):
    """Trace replays back-to-back, then the last replay's output checked against the reference."""
    chips, attn, w = _make_inputs(mesh_device, 0)
    ttnn.deallocate(a2a_wkv_b2_variant(attn, w))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = a2a_wkv_b2_variant(attn, w)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(20):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    _check(out, _expected(mesh_device, chips, w), mesh_device)
    ttnn.release_trace(mesh_device, tid)
