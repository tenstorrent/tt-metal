# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — main entry point.

Phase-0 envelope (per op_design.md):
- input dtype: float32
- input layout: ROW_MAJOR_LAYOUT (kernel handles tilize/untilize internally)
- rank >= 2; final two dims tile-aligned (H % 32 == 0, W % 32 == 0)
- optional gamma / beta: ROW_MAJOR_LAYOUT, shape (1, 1, 1, W), float32
- epsilon: keyword-only float, default 1e-5
- compute_kernel_config: None (entry installs Phase-0 default) or explicit
    ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True,
                                  math_approx_mode=False)

validate() rejects everything outside the table above with NotImplementedError.
"""

import ttnn

from .layer_norm_rm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Phase-0 default compute kernel config
# ---------------------------------------------------------------------------


def _default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def _validate_compute_kernel_config(cfg) -> None:
    """The Phase-0 contract: HiFi4 + fp32_dest_acc_en=True + math_approx_mode=False."""
    if cfg.math_fidelity != ttnn.MathFidelity.HiFi4:
        raise NotImplementedError(f"layer_norm_rm: math_fidelity={cfg.math_fidelity!r} not in SUPPORTED [HiFi4]")
    if not bool(cfg.fp32_dest_acc_en):
        raise NotImplementedError("layer_norm_rm: fp32_dest_acc_en=False not in SUPPORTED [True]")
    if bool(cfg.math_approx_mode):
        raise NotImplementedError("layer_norm_rm: math_approx_mode=True not in SUPPORTED [False]")


def _validate_affine(name: str, tensor, W: int) -> None:
    """Validate gamma / beta: ROW_MAJOR_LAYOUT, fp32, shape (1, 1, 1, W)."""
    if tensor.dtype != ttnn.float32:
        raise NotImplementedError(f"layer_norm_rm: {name}.dtype={tensor.dtype!r} not in SUPPORTED [float32]")
    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise NotImplementedError(f"layer_norm_rm: {name}.layout={tensor.layout!r} not in SUPPORTED [ROW_MAJOR_LAYOUT]")
    shape = list(tensor.shape)
    if shape != [1, 1, 1, W]:
        raise NotImplementedError(f"layer_norm_rm: {name}.shape={tuple(shape)} not in SUPPORTED [(1, 1, 1, {W})]")


def validate(
    input_tensor,
    gamma=None,
    beta=None,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> None:
    """Runtime gate. Raises NotImplementedError for anything outside SUPPORTED."""
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()
    _validate_compute_kernel_config(compute_kernel_config)

    if input_tensor.dtype != ttnn.float32:
        raise NotImplementedError(
            f"layer_norm_rm: input_tensor.dtype={input_tensor.dtype!r} not in SUPPORTED [float32]"
        )
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise NotImplementedError(
            f"layer_norm_rm: input_tensor.layout={input_tensor.layout!r} not in SUPPORTED [ROW_MAJOR_LAYOUT]"
        )

    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise NotImplementedError(f"layer_norm_rm: input rank={len(shape)} not in SUPPORTED [>=2]")

    H = shape[-2]
    W = shape[-1]
    if H % 32 != 0:
        raise NotImplementedError(f"layer_norm_rm: H={H} not tile-aligned (H % 32 != 0)")
    if W % 32 != 0:
        raise NotImplementedError(f"layer_norm_rm: W={W} not tile-aligned (W % 32 != 0)")

    if gamma is not None:
        _validate_affine("gamma", gamma, W)
    if beta is not None:
        _validate_affine("beta", beta, W)

    if not (epsilon > 0):
        raise NotImplementedError(f"layer_norm_rm: epsilon={epsilon} must be finite, positive")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Per-row layer normalization over the last dim of a ROW_MAJOR_LAYOUT
    float32 tensor.

    y[..., h, w] = ((x[..., h, w] - mean(x[..., h, :])) /
                    sqrt(var(x[..., h, :]) + epsilon))
                   * (gamma[w] if gamma else 1)
                   + (beta[w]  if beta  else 0)

    Args:
        input_tensor: rank >= 2, float32, ROW_MAJOR_LAYOUT, on-device.
            The final two dims must be tile-aligned (H % 32 == 0, W % 32 == 0).
        gamma: optional scale tensor of shape (1, 1, 1, W), float32, RM.
        beta:  optional shift tensor of shape (1, 1, 1, W), float32, RM.
        epsilon: positive float; added to the variance before rsqrt.
        compute_kernel_config: None (default Phase-0 config installed) or
            explicit ttnn.ComputeConfigDescriptor matching the Phase-0
            contract (math_fidelity=HiFi4, fp32_dest_acc_en=True,
            math_approx_mode=False).
        memory_config: output memory config (default DRAM interleaved).

    Returns:
        Output tensor with the same shape, dtype, and layout as input_tensor.
    """
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()

    validate(
        input_tensor,
        gamma,
        beta,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape = input shape; same dtype and (RM) layout.
    output_shape = list(input_tensor.shape)

    # allocate_tensor_on_device requires POSITIONAL args, not keyword args.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    # Build the IO tensor list — output tensor must be LAST.
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
