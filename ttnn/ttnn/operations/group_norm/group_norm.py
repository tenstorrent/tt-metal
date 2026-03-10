# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Group Norm - Main Entry Point

This file provides the user-facing function that:
1. Validates input tensor and parameters
2. Prepares gamma/beta tensors (host-side replication + tilize)
3. Creates per-group scaler mask tiles
4. Allocates output tensor on device
5. Creates the program descriptor
6. Launches via ttnn.generic_op

Usage:
    from ttnn.operations.group_norm import group_norm
    output = group_norm(input_tensor, num_groups=G, gamma=gamma, beta=beta, eps=1e-5)
"""

import struct

import ttnn

from .group_norm_program_descriptor import create_program_descriptor


def group_norm(
    input_tensor: ttnn.Tensor,
    num_groups: int = 1,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Group Normalization operation.

    Args:
        input_tensor: Input tensor of shape (N, 1, H*W, C) in ROW_MAJOR_LAYOUT on device.
        num_groups: Number of groups G. Must divide C; C must be divisible by 32.
        gamma: Per-channel scale tensor of shape (1, 1, 1, C) in bfloat16.
                If None, a tensor of ones is created on the host.
        beta: Per-channel bias tensor of shape (1, 1, 1, C) in bfloat16.
              If None, a tensor of zeros is created on the host.
        eps: Small constant for numerical stability (default: 1e-5).
        memory_config: Memory configuration for output tensor (default: DRAM interleaved).

    Returns:
        Output tensor of shape (N, 1, H*W, C) in ROW_MAJOR_LAYOUT.
    """
    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Extract shape
    shape = input_tensor.shape
    N = shape[0]
    C = shape[3]
    HW = shape[2]

    # Validate
    _validate_input(input_tensor, num_groups)

    # Prepare gamma/beta on host if not provided, then move to device as TILE_LAYOUT
    gamma_device, beta_device = _prepare_gamma_beta(gamma, beta, C, device)

    # Create per-group scaler mask tiles: (1, 1, 32, G*C) TILE_LAYOUT
    # Each group g has Ct tiles. Tile (g, ct) has 1/K at columns belonging to group g, 0 elsewhere.
    group_scaler_device = _prepare_group_scaler(N, HW, C, num_groups, device)

    # Allocate output tensor: same shape, same layout (ROW_MAJOR), same dtype
    output_shape = [shape[i] for i in range(len(shape))]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Pack eps as uint32 for runtime args
    eps_packed = _float_to_uint32(eps)

    # Create program descriptor
    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma_device,
        beta_device,
        group_scaler_device,
        output_tensor,
        num_groups=num_groups,
        eps_packed=eps_packed,
    )

    # Execute - output tensor MUST be last in the list
    return ttnn.generic_op(
        [input_tensor, gamma_device, beta_device, group_scaler_device, output_tensor],
        program_descriptor,
    )


def _validate_input(input_tensor: ttnn.Tensor, num_groups: int) -> None:
    """Validate input tensor and parameters."""
    shape = input_tensor.shape
    if len(shape) != 4:
        raise ValueError(f"group_norm: input must be 4D (N, 1, H*W, C), got {len(shape)}D")

    N, one, HW, C = shape[0], shape[1], shape[2], shape[3]

    if one != 1:
        raise ValueError(f"group_norm: dimension 1 must be 1, got {one}")

    if C % num_groups != 0:
        raise ValueError(f"group_norm: C={C} must be divisible by num_groups={num_groups}")

    if HW % 32 != 0:
        raise ValueError(f"group_norm: H*W={HW} must be divisible by 32")

    if C % 32 != 0:
        raise ValueError(f"group_norm: C={C} must be divisible by 32")


def _prepare_gamma_beta(gamma, beta, C, device):
    """
    Prepare gamma and beta tensors for the device.

    The design requires gamma/beta as (1, 1, 32, C) TILE_LAYOUT bfloat16 tensors,
    where the single row is replicated 32 times to fill a tile height.

    Args:
        gamma: Optional gamma tensor (1, 1, 1, C) or None
        beta: Optional beta tensor (1, 1, 1, C) or None
        C: Number of channels
        device: Target device

    Returns:
        (gamma_device, beta_device) as TILE_LAYOUT tensors on device
    """
    import torch  # Local import: global torch imports are forbidden in ttnn package

    if gamma is None:
        gamma_torch = torch.ones(1, 1, 32, C, dtype=torch.bfloat16)
    else:
        # gamma is (1, 1, 1, C) - replicate to (1, 1, 32, C)
        if isinstance(gamma, ttnn.Tensor):
            gamma_torch = ttnn.to_torch(gamma)
        else:
            gamma_torch = gamma
        gamma_torch = gamma_torch.reshape(1, 1, 1, C).expand(1, 1, 32, C).contiguous()

    if beta is None:
        beta_torch = torch.zeros(1, 1, 32, C, dtype=torch.bfloat16)
    else:
        if isinstance(beta, ttnn.Tensor):
            beta_torch = ttnn.to_torch(beta)
        else:
            beta_torch = beta
        beta_torch = beta_torch.reshape(1, 1, 1, C).expand(1, 1, 32, C).contiguous()

    gamma_device = ttnn.from_torch(
        gamma_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_device = ttnn.from_torch(
        beta_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return gamma_device, beta_device


def _prepare_group_scaler(N, HW, C, num_groups, device):
    """
    Create per-group scaler mask tiles for the mean computation.

    For each group g and tile column ct, creates a 32x32 tile where:
    - Element (r, j) = 1.0 if global column (ct*32 + j) belongs to group g
    - Element (r, j) = 0 otherwise
    This is a binary membership mask (1/0). The 1/K scaling is applied
    by the reduce scaler CB on the device side.

    The tensor shape is (1, 1, 32, G*C) which tilizes to G*Ct tiles.
    Tile index for group g, tile column ct = g * Ct + ct.

    Args:
        N: batch size
        HW: spatial dimension (H*W)
        C: channels
        num_groups: G
        device: target device

    Returns:
        TILE_LAYOUT tensor on device with G*Ct tiles
    """
    import torch  # Local import

    G = num_groups
    Ct = C // 32
    channels_per_group = C // G

    # Create (1, 1, 32, G*C) tensor - binary mask (1.0 / 0.0)
    scaler = torch.zeros(1, 1, 32, G * C, dtype=torch.bfloat16)

    for g in range(G):
        group_start = g * channels_per_group  # global channel start for group g
        group_end = (g + 1) * channels_per_group  # global channel end

        # In the tensor, group g's tiles start at column offset g * C
        # Tile column ct corresponds to global columns [ct*32, (ct+1)*32)
        for ct in range(Ct):
            col_start_global = ct * 32
            col_end_global = (ct + 1) * 32

            # Determine overlap of this tile column with group g
            overlap_start = max(col_start_global, group_start)
            overlap_end = min(col_end_global, group_end)

            if overlap_start < overlap_end:
                # Local column indices within this tile
                local_start = overlap_start - col_start_global
                local_end = overlap_end - col_start_global

                # Position in the flattened tensor: group g starts at column g*C, tile ct at col ct*32
                tensor_col_start = g * C + ct * 32 + local_start
                tensor_col_end = g * C + ct * 32 + local_end

                scaler[:, :, :, tensor_col_start:tensor_col_end] = 1.0

    group_scaler_device = ttnn.from_torch(
        scaler,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return group_scaler_device


def _float_to_uint32(value: float) -> int:
    """Convert a float to its uint32 bit representation."""
    return int.from_bytes(struct.pack("f", value), byteorder="little")
