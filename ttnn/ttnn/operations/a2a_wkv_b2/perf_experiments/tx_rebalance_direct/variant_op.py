# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tx_rebalance_direct: the op's entry point, built from variant_descriptor.py (knobs via env:
TXR_OVERFLOW=x, TXR_WRITER=base|mr, TXR_PLACEMENT, TXR_OVERFLOW_POS=last|first)."""
import os
import sys

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variant_descriptor as vd  # noqa: E402

from ttnn.operations.a2a_wkv_b2.a2a_wkv_b2 import default_compute_kernel_config, validate  # noqa: E402


def a2a_wkv_b2_variant(input_tensor, weight, *, cluster_axis=1, num_links=2, compute_kernel_config=None):
    validate(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
    )
    cfg = compute_kernel_config or default_compute_kernel_config()
    mesh_device = input_tensor.device()
    n = tuple(mesh_device.shape)[cluster_axis]
    _, H, T, _ = tuple(input_tensor.shape)
    V = tuple(weight.shape)[-1]
    sem_ready, sem_done = vd.get_cross_chip_semaphores(mesh_device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H // n, T * n, V]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
    )
    mesh_pd = vd.create_mesh_program_descriptor(
        input_tensor,
        weight,
        out,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_kernel_config=cfg,
        sem_ready=sem_ready,
        sem_done=sem_done,
    )
    return ttnn.generic_op([input_tensor, weight, out], mesh_pd)


def config_tag():
    return (
        f"writer={vd.TX_WRITER} x={vd.DIRECT_OVERFLOW_HEADS} placement={vd.HEAD_PLACEMENT} "
        f"pos={vd.OVERFLOW_POS} early_done={int(vd.EARLY_DONE)}"
    )
