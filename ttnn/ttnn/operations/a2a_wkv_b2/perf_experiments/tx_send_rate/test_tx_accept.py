# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for a2a_wkv_b2 (immutable spec for the implementer).

Every chip of the 2x4 LoudBox gets its own attn [1, 96, 160, 512]; W [1, 96, 512, 128]
is replicated. Chip r of each ring along cluster_axis must end with

    out_r[0, h, 160*j + t, :] = attn_j[0, 24*r + h, t, :] @ W[0, 24*r + h]

The op has exactly one shape; the parametrization covers distinct input sets (seeds,
including one with a large dynamic range), both call patterns, back-to-back calls and
the shape contract. Gate: PCC and relative RMS from the golden suite's limits
(helpers.TOLERANCES / eval.metrics.DEFAULT_TOLERANCES), over all chips together.
"""

import pytest
import ttnn

from eval import metrics
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tx_op import a2a_wkv_b2  # tx_send_rate variant (env TXV)

MESH_SHAPE = (2, 4)
CLUSTER_AXIS = 1
NUM_LINKS = 2
ATTN_SHAPE = (1, 96, 160, 512)
W_SHAPE = (1, 96, 512, 128)
WEIGHT_SCALE = 0.05

_mesh = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
_fabric = pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True
)


def _tolerance():
    try:
        from eval.golden_tests.a2a_wkv_b2.helpers import TOLERANCES

        pcc = TOLERANCES[(ttnn.bfloat8_b, False)]
        return metrics.Tolerance(pcc=pcc)
    except (ImportError, KeyError):
        pcc, rms = metrics.default_tolerance(ttnn.bfloat8_b, False)
        return metrics.Tolerance(pcc=pcc, rms=rms)


def _ring_members(mesh_device, line):
    rows, cols = mesh_device.shape
    if CLUSTER_AXIS == 1:
        return [line * cols + c for c in range(cols)]
    return [r * cols + line for r in range(rows)]


def torch_reference(ring_inputs, w):
    """fp64 reference for one ring, in ring order."""
    import torch

    n = len(ring_inputs)
    hpc = ring_inputs[0].shape[1] // n
    w64 = w.to(torch.float64)
    outs = []
    for r in range(n):
        heads = slice(r * hpc, (r + 1) * hpc)
        attn = torch.cat([x[:, heads].to(torch.float64) for x in ring_inputs], dim=2)
        outs.append(torch.matmul(attn, w64[:, heads]).to(torch.float32))
    return outs


def _make_inputs(mesh_device, seed, attn_shape=ATTN_SHAPE, w_shape=W_SHAPE, scale=1.0):
    import torch

    torch.manual_seed(42 + seed)
    chips = [
        (torch.randn(attn_shape, dtype=torch.float32) * scale).to(torch.bfloat16)
        for _ in range(mesh_device.get_num_devices())
    ]
    w = (torch.randn(w_shape, dtype=torch.float32) * WEIGHT_SCALE).to(torch.bfloat16)
    shards = [ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for c in chips]
    attn_tt = ttnn.from_host_shards(shards, mesh_device.shape).to(mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    w_tt = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return chips, attn_tt, w_tt


def _expected(mesh_device, chips, w_tt):
    import torch

    w_dev = ttnn.to_torch(ttnn.get_device_tensors(w_tt)[0])  # the bf8_b values the op multiplies by
    rows, cols = mesh_device.shape
    lines = rows if CLUSTER_AXIS == 1 else cols
    out = [None] * len(chips)
    for line in range(lines):
        members = _ring_members(mesh_device, line)
        for m, y in zip(members, torch_reference([chips[m] for m in members], w_dev)):
            out[m] = y
    return out


def _check(out_tt, expected, mesh_device):
    import torch

    n = mesh_device.shape[CLUSTER_AXIS]
    out_shape = [1, ATTN_SHAPE[1] // n, ATTN_SHAPE[2] * n, W_SHAPE[-1]]
    assert list(out_tt.shape) == out_shape, f"shape {list(out_tt.shape)} != {out_shape}"
    assert out_tt.dtype == ttnn.bfloat8_b
    assert out_tt.layout == ttnn.TILE_LAYOUT
    mc = out_tt.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    actual = [ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(out_tt)]
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert torch.isfinite(a).all(), f"chip {i}: non-finite output"
    tol = _tolerance()
    m = metrics.compute_metrics_torch(torch.stack(actual), torch.stack(expected), readback_dtype=torch.float32)
    misses = metrics._gate_metrics(m, tol, torch.stack(actual), torch.stack(expected))
    assert not misses, f"{misses}; metrics={m}"
    return actual


@_fabric
@_mesh
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a2a_wkv_b2_default_config(mesh_device, seed):
    chips, attn, w = _make_inputs(mesh_device, seed)
    out = a2a_wkv_b2(attn, w)
    _check(out, _expected(mesh_device, chips, w), mesh_device)


@_fabric
@_mesh
@pytest.mark.parametrize("scale", [0.01, 8.0], ids=["small_values", "large_values"])
def test_a2a_wkv_b2_explicit_config(mesh_device, scale):
    chips, attn, w = _make_inputs(mesh_device, 3, scale=scale)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    out = a2a_wkv_b2(attn, w, cluster_axis=CLUSTER_AXIS, num_links=NUM_LINKS, compute_kernel_config=cfg)
    _check(out, _expected(mesh_device, chips, w), mesh_device)


@_fabric
@_mesh
def test_a2a_wkv_b2_back_to_back(mesh_device):
    """Alternating inputs: every call correct and bit-identical to the first call on the same inputs."""
    import torch

    sets = [_make_inputs(mesh_device, s) for s in (4, 5)]
    first = []
    for chips, attn, w in sets:
        out = a2a_wkv_b2(attn, w)
        first.append(_check(out, _expected(mesh_device, chips, w), mesh_device))
        ttnn.deallocate(out)
    for i in range(4):
        k = i % 2
        _, attn, w = sets[k]
        out = a2a_wkv_b2(attn, w)
        got = [ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
        for chip, (a, b) in enumerate(zip(got, first[k])):
            assert torch.equal(a, b), f"call {i + 1}: chip {chip} differs from the first call"


@_fabric
@_mesh
@pytest.mark.parametrize(
    "attn_shape, w_shape",
    [
        ((1, 96, 160, 256), W_SHAPE),
        ((1, 90, 160, 512), (1, 90, 512, 128)),
        (ATTN_SHAPE, (1, 24, 512, 128)),
    ],
    ids=["latent_mismatch", "heads_not_divisible", "weight_not_all_heads"],
)
def test_a2a_wkv_b2_shape_contract(mesh_device, attn_shape, w_shape, expect_error):
    _, attn, w = _make_inputs(mesh_device, 6, attn_shape=attn_shape, w_shape=w_shape)
    with expect_error((ValueError, RuntimeError), ""):  # any message: the op owns the wording
        a2a_wkv_b2(attn, w)


@_fabric
@_mesh
def test_a2a_wkv_b2_input_in_l1_refused(mesh_device, expect_error):
    _, attn, w = _make_inputs(mesh_device, 7)
    attn_l1 = ttnn.to_memory_config(attn, ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), ""):  # any message: the op owns the wording
        a2a_wkv_b2(attn_l1, w)
