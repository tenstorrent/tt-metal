# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""a2a_wkv_b2 — fused all-to-all (sequence-sharded -> head-sharded) + per-head wkv_b2 projection.

For the chip at position r of its ring along ``cluster_axis`` (n chips), with attn_j the input
of the chip at position j of the same ring:

    out_r[0, h, T*j + t, :] = attn_j[0, H/n*r + h, t, :] @ W[0, H/n*r + h, :, :]

One ``ttnn.generic_op`` over a ``ttnn.MeshProgramDescriptor``: every chip applies W to its own
tokens of all H heads first (matmul-first, regime R1 of op_design.md), then sends each
destination chip its H/n heads' bfloat8_b result over the fabric, straight into the destination's
L1-interleaved output tiles.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue


# ---------------------------------------------------------------------------
# Compute config factory (the golden axis-tagger reads the same factory)
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — one shape, so no shape-derived axis.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = {}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0 is the whole TARGET (one value per axis).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],  # attn
    "layout": [ttnn.ROW_MAJOR_LAYOUT],  # attn
    "weight_dtype": [ttnn.bfloat8_b],  # W
    "fp32_dest_acc_en": [False],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Shape contract (the one problem this op solves)
# ---------------------------------------------------------------------------

_ATTN_SHAPE = (1, 96, 160, 512)  # per chip: [1, H, T, L]
_W_SHAPE = (1, 96, 512, 128)  # per chip: [1, H, L, V] (all heads, replicated)
_RING_SIZE = 4
_TILE = 32


def _is_dram_interleaved(t) -> bool:
    mc = t.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _check_shape_contract(input_tensor, weight, cluster_axis):
    attn_shape = tuple(input_tensor.shape)
    w_shape = tuple(weight.shape)
    if attn_shape != _ATTN_SHAPE:
        raise ValueError(f"a2a_wkv_b2: attn must be {list(_ATTN_SHAPE)} per chip, got {list(attn_shape)}")
    if w_shape != _W_SHAPE:
        raise ValueError(f"a2a_wkv_b2: W must be {list(_W_SHAPE)} per chip (all heads), got {list(w_shape)}")
    _, H, T, L = attn_shape
    if w_shape[1] != H or w_shape[2] != L:
        raise ValueError(f"a2a_wkv_b2: W {list(w_shape)} does not match attn {list(attn_shape)}")
    if T % _TILE != 0:
        # Two sources would share an output tile row (data race).
        raise ValueError(f"a2a_wkv_b2: tokens per chip ({T}) must be a multiple of {_TILE}")
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    if not (0 <= cluster_axis < len(mesh_shape)):
        raise ValueError(f"a2a_wkv_b2: cluster_axis={cluster_axis} out of range for mesh {mesh_shape}")
    n = mesh_shape[cluster_axis]
    if n != _RING_SIZE:
        raise ValueError(f"a2a_wkv_b2: mesh axis {cluster_axis} must be {_RING_SIZE} chips long, got {n}")
    if H % n != 0:
        raise ValueError(f"a2a_wkv_b2: heads ({H}) must divide by the ring size ({n})")
    if not _is_dram_interleaved(input_tensor):
        raise ValueError("a2a_wkv_b2: attn must be DRAM interleaved")
    if not _is_dram_interleaved(weight):
        raise ValueError("a2a_wkv_b2: W must be DRAM interleaved")
    if weight.layout != ttnn.TILE_LAYOUT:
        raise ValueError("a2a_wkv_b2: W must be TILE_LAYOUT")


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, weight, *, cluster_axis=1, num_links=2, compute_kernel_config=None):
    cfg = compute_kernel_config or default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "weight_dtype": weight.dtype,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", False)),
    }
    tagger_inputs = (list(input_tensor.shape), list(weight.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(tagger_inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"a2a_wkv_b2: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"a2a_wkv_b2: unsupported combination (refinement candidate): {exc}")

    _check_shape_contract(input_tensor, weight, cluster_axis)

    n = tuple(input_tensor.device().shape)[cluster_axis]
    # Streams per direction = distinct hop counts in one direction (antipode split adds one per side).
    streams_per_direction = (n - 1) // 2 + (1 if n % 2 == 0 else 0)
    if num_links != streams_per_direction:
        # R6 (several streams on one router via a mux) is deferred.
        raise NotImplementedError(
            f"a2a_wkv_b2: num_links={num_links} unsupported; Phase 0 needs num_links == {streams_per_direction}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def a2a_wkv_b2(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    cluster_axis: int = 1,
    num_links: int = 2,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
    )
    from ttnn.operations.a2a_wkv_b2.a2a_wkv_b2_program_descriptor import (
        create_mesh_program_descriptor,
        get_cross_chip_semaphores,
    )

    cfg = compute_kernel_config or default_compute_kernel_config()
    mesh_device = input_tensor.device()
    n = tuple(mesh_device.shape)[cluster_axis]
    _, H, T, _ = tuple(input_tensor.shape)
    V = tuple(weight.shape)[-1]

    # Created on the first (eager) call only and cached per mesh device: never inside a trace.
    sem_ready, sem_done = get_cross_chip_semaphores(mesh_device)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H // n, T * n, V]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.L1_MEMORY_CONFIG,
    )
    mesh_pd = create_mesh_program_descriptor(
        input_tensor,
        weight,
        output_tensor,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_kernel_config=cfg,
        sem_ready=sem_ready,
        sem_done=sem_done,
    )
    return ttnn.generic_op([input_tensor, weight, output_tensor], mesh_pd)
