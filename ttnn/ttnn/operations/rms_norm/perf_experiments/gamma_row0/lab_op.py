# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the gamma_row0 isolated bake-off.

Same call shape as `ttnn.operations.rms_norm.rms_norm`, but wired to this
experiment's `lab_descriptor.create_program_descriptor` so the real op is never
touched.  The precision contract is a pass-through: whatever
`compute_kernel_config` the caller supplies reaches the compute kernel verbatim,
identically for every arm.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from .lab_descriptor import create_program_descriptor


def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def loose_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    """The exact corner every perf-gated feature_spec LOOSE_CASE runs at."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def lab_rms_norm(
    input_tensor: "ttnn.Tensor",
    *,
    gamma: Optional["ttnn.Tensor"] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor" = None,
    levers: dict = None,
) -> "ttnn.Tensor":
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )
    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=cfg,
        levers=levers,
    )
    tensors = [input_tensor] if gamma is None else [input_tensor, gamma]
    tensors.append(output_tensor)
    return ttnn.generic_op(tensors, program_descriptor)
