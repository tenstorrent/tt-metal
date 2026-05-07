# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Python helper utilities for ttnn.experimental.moe_compute.

This module provides reference implementations for preparing weight tensors
for the fused MoE compute operation. The functions here are "executable
specifications" that produce the exact byte layout the kernels expect.

**Weight tensor layout overview**

``ttnn.experimental.moe_compute`` takes **two packed weight tensor arguments**
that contain **three logical expert weight matrices**:

- ``matmul_w0_w1_tensor``: W0 (gate) and W1 (up) weights interleaved/packed.
- ``matmul_w2_tensor``: W2 (down projection) weights.

The layout is highly specific to the MoE ring-all-to-all kernel implementation.
Constants defined here must stay in sync with
``moe_ring_common.h`` (TILE_SIZE=32, W0_W1_TILES_PER_TXN=14, etc.).

**Constants (must match moe_ring_common.h)**

- ``NUM_W0_W1_TILES_H = 224`` (tiles) -> 7168 elements for the reference hidden size
- ``NUM_W2_TILES_H = 64`` (tiles) -> 2048 elements
- ``W0_W1_TILES_PER_TXN = 14``
- ``W2_TILES_PER_TXN = 14``
- ``W2_TXNS_PER_BLOCK = 2``
- ``W2_BLOCKS_PER_EXPERT = 50``

**Bias support (``has_bias=True``)**

When bias is enabled, callers must pack bias values into the weight tensors
in a kernel-specific format and set ``has_bias=True`` on the ``moe_compute`` call:

- **W0/W1 bias (b0, b1)**: PyTorch format is ``(L, E, N)`` where L=layers, E=experts,
  N=intermediate dim. This is expanded to tile format ``(L, E, 32, N)`` with only
  row 0 populated, concatenated **after** W0/W1 along the K (input) dimension, then
  K is padded to a multiple of 14 tiles (W0_W1_TILES_PER_TXN). For the reference
  config (hidden=7168), K grows from 224 to 225 tiles, padded to 238 tiles (7616
  elements).

- **W2 bias (b2)**: PyTorch format is ``(L, E, K)``. Expanded to tile format
  ``(L, E, 32, K)`` with row 0 populated, K-column-sharded like W2, appended along
  the N (intermediate) axis **without** ring-rotation (matching GPT-OSS behavior).
  N+32 is then padded to 70 tiles (2240 elements) to align DRAM reads.

**Output shapes (reference config)**

- ``prepare_w0_w1_tensor_for_moe_compute``: ``(num_cores, L, E, 3, K, 4*TILE_SIZE)``
- ``prepare_w0_w1_tensor_with_bias``: K becomes ``K_padded`` (e.g., 7616)
- ``prepare_w2_tensor_for_moe_compute``: ``(num_cores, L, E, 5, N+192, 4*TILE_SIZE)``
- ``prepare_w2_tensor_with_bias``: N+192 becomes N_target (e.g., 2240)

The leading ``num_cores`` dimension (typically 12) corresponds to DRAM bank layout.

**DRAM sharding**

Callers must create DRAM-sharded memory configs with heights derived from the
padded dimensions above. See ``test_moe_compute_6U.py`` for the exact shard
spec calculations using ``K_for_shard`` and ``w2_N_total``.

**Available functions**

- Non-bias: ``prepare_w0_w1_tensor_for_moe_compute``, ``prepare_w2_tensor_for_moe_compute``
- With bias: ``prepare_w0_w1_tensor_with_bias``, ``prepare_w2_tensor_with_bias``
- Helpers: ``cluster_distance``, ``map_shared_experts``, ``add_shared_expert_weights``

See individual function docstrings for argument details and layout invariants.
"""

from __future__ import annotations

import math
from typing import Sequence

import ttnn


def cluster_distance(d0: int, d1: int, mesh_shape: tuple[int, int], cluster_axis: int) -> int | None:
    """Calculate Manhattan distance between two devices along the cluster axis.

    Returns None if devices are not on the same cluster line, otherwise returns
    the distance along the cluster axis.
    """
    c0 = (d0 // mesh_shape[1], d0 % mesh_shape[1])
    c1 = (d1 // mesh_shape[1], d1 % mesh_shape[1])

    return None if c0[1 - cluster_axis] != c1[1 - cluster_axis] else abs(c0[cluster_axis] - c1[cluster_axis])


def map_shared_experts(
    expert_mapping_tensor: "torch.Tensor",
    shared_expert_ids_to_devices: dict[int, list[int]],
    mesh_shape: Sequence[int],
    cluster_axis: int,
) -> "torch.Tensor":
    """
    Map shared experts to their nearest on-axis device for dispatch operations.

    This function extends the expert mapping tensor to include shared experts by determining
    the optimal device assignment for each shared expert based on cluster topology. For each
    dispatching device, it selects the nearest receiving device on the same cluster axis
    that has the shared expert.

    Args:
        expert_mapping_tensor: 2D tensor of shape [devices, routed_experts] containing
            linearized mesh coordinates of the device owning each expert.
        shared_expert_ids_to_devices: Dictionary mapping shared expert IDs to lists of
            device IDs where they are replicated. Expert IDs must be contiguous
            continuations of routed expert IDs.
        mesh_shape: Tuple/list representing the dimensions of the device mesh (e.g., (4, 4)).
        cluster_axis: Axis along which devices are clustered (0 or 1). Determines the
            direction of nearest-neighbor search for shared experts.

    Returns:
        torch.Tensor: Extended mapping tensor of shape [devices, routed_experts + shared_experts]
            where each entry [d, e] contains the device ID that device d should dispatch
            expert e to. For shared experts, this is the nearest device on the same
            cluster axis that has the expert.

    Raises:
        RuntimeError: If shared experts are not distributed evenly across devices.
        RuntimeError: If shared expert IDs are not contiguous with routed expert IDs.

    Notes:
        - The function uses Manhattan distance along the cluster axis to find nearest devices.
        - If no device with the shared expert is on the same cluster axis, a default
          device is selected (the first in the list).
        - This mapping is critical for efficient MoE dispatch operations in distributed systems.
    """
    import torch

    # assuming [devices, experts] -> linearized mesh coordinate of owning device
    if len(expert_mapping_tensor.shape) != 2:
        raise RuntimeError(f"Invalid shape of mapping tensor. Expected: 2. Got: {len(expert_mapping_tensor.shape)}")

    devices = expert_mapping_tensor.shape[0]
    routed_experts = expert_mapping_tensor.shape[1]

    shared_experts = len(shared_expert_ids_to_devices)

    shared_experts_per_device = get_shared_experts_per_device(shared_expert_ids_to_devices, devices)

    if not len(set(shared_experts_per_device)) == 1:
        raise RuntimeError("Shared Experts must be distributed such that all devices have an equal number of experts")

    if list(range(routed_experts)) + sorted([se for se in shared_expert_ids_to_devices]) != list(
        range(routed_experts + shared_experts)
    ):
        raise RuntimeError("Shared expert IDs should be a contiguous continuation of routed expert IDs ")

    routed_and_shared_expert_mapping = torch.cat(
        [expert_mapping_tensor, torch.zeros((devices, shared_experts), dtype=expert_mapping_tensor.dtype)], dim=1
    )
    for disp_d in range(devices):
        for se, rec_ds in shared_expert_ids_to_devices.items():
            min_distance = mesh_shape[cluster_axis] + 1

            # just pick one as the default case. If none of the device assignments are on the same cluster axis as
            # disp_d then this expert will also get skipped by dispatch on disp_d
            routed_and_shared_expert_mapping[disp_d, se] = rec_ds[0]
            for rec_d in rec_ds:
                distance = cluster_distance(disp_d, rec_d, mesh_shape, cluster_axis)
                if distance is not None and distance < min_distance:
                    routed_and_shared_expert_mapping[disp_d, se] = rec_d
                    min_distance = distance

    return routed_and_shared_expert_mapping


def get_shared_experts_per_device(shared_expert_ids_to_devices: dict[int, list[int]], devices: int) -> list[int]:
    """
    Calculate the number of shared experts assigned to each device.

    This function counts how many shared experts are assigned to each device based on the
    provided mapping. It's used to verify even distribution of shared experts and to
    determine memory requirements for each device.

    Args:
        shared_expert_ids_to_devices: Dictionary mapping shared expert IDs to lists of
            device IDs where they are replicated.
        devices: Total number of devices in the system.

    Returns:
        List[int]: A list of length 'devices' where each element represents the number
            of shared experts assigned to that device index.

    Example:
        >>> mapping = {0: [0, 2], 1: [1, 3]}  # Expert 0 on devices 0,2; Expert 1 on devices 1,3
        >>> get_shared_experts_per_device(mapping, 4)
        [1, 1, 1, 1]  # Each device has 1 shared expert
    """
    shared_experts_per_device = [0] * devices
    for ds in shared_expert_ids_to_devices.values():
        for d in ds:
            shared_experts_per_device[d] += 1
    return shared_experts_per_device


def add_shared_expert_weights(
    routed_w0: "torch.Tensor",  # (layers, routed experts, hidden, matmul N)
    routed_w1: "torch.Tensor",  # (layers, routed experts, hidden, matmul N)
    routed_w2: "torch.Tensor",  # (layers, routed experts, matmul N, hidden)
    shared_w0: dict[int, "torch.Tensor"],  # id: (layers, 1, hidden, matmul N)
    shared_w1: dict[int, "torch.Tensor"],  # id: (layers, 1, hidden, matmul N)
    shared_w2: dict[int, "torch.Tensor"],  # id: (layers, 1, matmul N, hidden)
    shared_expert_ids_to_device: dict[int, list[int]],
    num_devices: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Combine routed and shared expert weights into a unified tensor format for MoE computation.

    This function reorganizes MoE expert weights by combining routed experts (unique to each
    device) with shared experts (replicated across multiple devices) into a single tensor
    format. It ensures proper weight distribution across devices for efficient MoE dispatch.

    Args:
        routed_w0: First layer weights for routed experts.
            Shape: [layers, routed_experts, hidden_dim, matmul_n]
        routed_w1: Second layer weights for routed experts.
            Shape: [layers, routed_experts, hidden_dim, matmul_n]
        routed_w2: Third layer weights for routed experts.
            Shape: [layers, routed_experts, matmul_n, hidden_dim]
        shared_w0: Dictionary mapping expert IDs to first layer weights for shared experts.
            Values have shape: [layers, 1, hidden_dim, matmul_n]
        shared_w1: Dictionary mapping expert IDs to second layer weights for shared experts.
            Values have shape: [layers, 1, hidden_dim, matmul_n]
        shared_w2: Dictionary mapping expert IDs to third layer weights for shared experts.
            Values have shape: [layers, 1, matmul_n, hidden_dim]
        shared_expert_ids_to_device: Dictionary mapping shared expert IDs to lists of
            device IDs where they should be replicated.
        num_devices: Total number of devices in the system.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Three tensors containing the
            combined weights (w0, w1, w2) with both routed and shared experts arranged
            for each device. Each output tensor has shape:
            [layers, num_devices * total_experts_per_device, ...]

    Raises:
        RuntimeError: If shared experts are not evenly distributed across devices
            (validated in get_shared_experts_per_device).

    Notes:
        - Routed experts are distributed sequentially across devices.
        - Shared experts are appended after routed experts for each device.
        - The function assumes contiguous expert IDs and even distribution.
        - Better validation of the shared_expert_ids_to_device mapping occurs in
          map_shared_experts function (also generally required for shared expert usage).

    Example:
        >>> # 4 devices, 8 routed experts (2 per device), 2 shared experts
        >>> routed_w0 = torch.randn(1, 8, 256, 512)
        >>> shared_w0 = {8: torch.randn(1, 1, 256, 512), 9: torch.randn(1, 1, 256, 512)}
        >>> mapping = {8: [0, 2], 9: [1, 3]}  # Each shared expert on 2 devices
        >>> result_w0, _, _ = add_shared_expert_weights(
        ...     routed_w0, routed_w1, routed_w2,
        ...     shared_w0, shared_w1, shared_w2,
        ...     mapping, 4)
        >>> result_w0.shape
        torch.Size([1, 12, 256, 512])  # 4 devices * 3 experts/device
    """
    import torch

    num_routed_experts = routed_w0.shape[1]
    num_routed_experts_per_device = num_routed_experts // num_devices
    num_shared_experts_per_device = get_shared_experts_per_device(shared_expert_ids_to_device, num_devices)[0]
    total_experts_per_device = num_routed_experts_per_device + num_shared_experts_per_device

    device_to_shared_experts = [[] for _ in range(num_devices)]
    sorted_shared_ids = sorted(shared_expert_ids_to_device.keys())

    for shared_id in sorted_shared_ids:
        for device in shared_expert_ids_to_device[shared_id]:
            device_to_shared_experts[device].append(shared_id)

    # Get tensor dimensions for pre-allocation
    layers = routed_w0.shape[0]
    hidden_dim = routed_w0.shape[2]
    matmul_n = routed_w0.shape[3]

    # Pre-allocate output tensors
    total_experts = num_devices * total_experts_per_device
    output_w0 = torch.empty((layers, total_experts, hidden_dim, matmul_n), dtype=routed_w0.dtype)
    output_w1 = torch.empty((layers, total_experts, hidden_dim, matmul_n), dtype=routed_w1.dtype)
    output_w2 = torch.empty((layers, total_experts, matmul_n, hidden_dim), dtype=routed_w2.dtype)

    # Fill output tensors using direct indexing
    for d in range(num_devices):
        # Calculate output indices for this device
        start_idx = d * total_experts_per_device
        routed_end_idx = start_idx + num_routed_experts_per_device

        # Copy routed experts for this device using slice assignment
        routed_start = d * num_routed_experts_per_device
        routed_end = (d + 1) * num_routed_experts_per_device

        output_w0[:, start_idx:routed_end_idx, :, :] = routed_w0[:, routed_start:routed_end, :, :]
        output_w1[:, start_idx:routed_end_idx, :, :] = routed_w1[:, routed_start:routed_end, :, :]
        output_w2[:, start_idx:routed_end_idx, :, :] = routed_w2[:, routed_start:routed_end, :, :]

        # Copy shared experts for this device
        for i, shared_id in enumerate(device_to_shared_experts[d]):
            shared_idx = routed_end_idx + i
            output_w0[:, shared_idx : shared_idx + 1, :, :] = shared_w0[shared_id]
            output_w1[:, shared_idx : shared_idx + 1, :, :] = shared_w1[shared_id]
            output_w2[:, shared_idx : shared_idx + 1, :, :] = shared_w2[shared_id]

    return output_w0, output_w1, output_w2


####################################################################################################
# model specific and global constants
# must be consistent with moe_ring_common.h
BLOCK_TILES_W = 4
BLOCK_TILES_H = 7

DS_PAD_CORES = {1, 2, 4, 5, 7, 8, 10, 11}
DS_W0_W1_SHARD_VALS = [6, 5]
DS_W2_SHARD_VALS = {False: (2, 2), True: (3, 1)}  # mapped to pad core assignment

GPT_PAD_CORES = {2, 3, 6, 7, 10, 11}
GPT_W0_W1_SHARD_VALS = [8, 7]
GPT_W2_SHARD_VALS = {False: (4, 0), True: (3, 1)}
####################################################################################################


def prepare_w0_w1_tensor_for_moe_compute(
    torch_w0: "torch.Tensor",
    torch_w1: "torch.Tensor",
    L: int,
    E: int,
    K: int,
    N: int,
    shard_map: list[int],
):
    """
    Prepare the w0_w1 tensor input for moe_compute by interleaving chunks of w0 and w1 width-wise.

    Args:
        torch_w0: Weight tensor of shape (L, E, K, N)
        torch_w1: Weight tensor of shape (L, E, K, N)
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension
        shard_map: List of shard sizes for each core

    Returns:
        torch_w0_w1_interleaved: Interleaved tensor of shape (L, E, K, 4096)
    """
    import torch

    # Check that K and N are divisible by ttnn.TILE_SIZE
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Nt = N // ttnn.TILE_SIZE
    # in general, pad K up to a factor of transaction size (32*7)
    Kp = math.ceil(K // ttnn.TILE_SIZE / BLOCK_TILES_H) * ttnn.TILE_SIZE * BLOCK_TILES_H
    num_cores = len(shard_map)

    if K < Kp:
        padding = torch.zeros((L, E, Kp - K, N), dtype=torch_w0.dtype)
        working_torch_w0 = torch.concat([torch_w0, padding], dim=2)
        working_torch_w1 = torch.concat([torch_w1, padding], dim=2)
    else:
        working_torch_w0 = torch_w0
        working_torch_w1 = torch_w1

    # Reshape to expose chunks: (L, E, K, N) -> (L, E, K, Nt, ttnn.TILE_SIZE)
    w0_chunks = working_torch_w0.view(L, E, Kp, Nt, ttnn.TILE_SIZE)
    w1_chunks = working_torch_w1.view(L, E, Kp, Nt, ttnn.TILE_SIZE)

    # Stack w0 and w1 chunks together: (L, E, K, Nt, 2, ttnn.TILE_SIZE)
    # This puts w0_chunk_i and w1_chunk_i adjacent to each other
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)

    # Reshape to interleave: (L, E, K, Nt * 2 * ttnn.TILE_SIZE)
    # The order will be: w0_chunk_0, w1_chunk_0, w0_chunk_1, w1_chunk_1, ...
    torch_w0_w1_interleaved = stacked.view(L, E, Kp, Nt, 2 * ttnn.TILE_SIZE)

    # Permute to move Nt before K: (L, E, K, Nt, 2*TILE) -> (L, E, Nt, K, 2*TILE)
    torch_w0_w1_permuted = torch_w0_w1_interleaved.permute(0, 1, 3, 2, 4)

    each_shard = []
    max_shard_size = max(shard_map)
    if any(x not in [max_shard_size, max_shard_size - 1] for x in shard_map):
        raise RuntimeError(f"W0W1 shard sizes should differ by 1 at most: {shard_map}")

    # Pick appropriate number of column tiles for each core based on the ring position.
    start_tile = 0
    for num_tiles in shard_map:
        each_shard.append(torch_w0_w1_permuted[:, :, start_tile : start_tile + num_tiles, :, :])

        if num_tiles < max_shard_size:
            each_shard.append(torch.zeros(L, E, 1, Kp, 2 * ttnn.TILE_SIZE, dtype=torch_w0_w1_permuted.dtype))
        start_tile += num_tiles

    torch_w0_w1_reordered = torch.cat(each_shard, dim=2)
    all_groups_per_bank = torch_w0_w1_reordered.view(L, E, num_cores, -1, Kp, 2 * ttnn.TILE_SIZE)
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)

    groups_per_core = max_shard_size // 2

    torch_w0_w1_pair_2_tiles = all_groups_per_bank.view(num_cores, L, E, groups_per_core, -1, Kp, 2 * ttnn.TILE_SIZE)
    torch_w0_w1_pair_2_tiles = torch_w0_w1_pair_2_tiles.permute(0, 1, 2, 3, 5, 4, 6)
    torch_w0_w1_paired = torch_w0_w1_pair_2_tiles.reshape(num_cores, L, E, groups_per_core, -1, 4 * ttnn.TILE_SIZE)

    return torch_w0_w1_paired


def prepare_w2_tensor_for_moe_compute(
    torch_w2: "torch.Tensor",
    L: int,
    E: int,
    N: int,
    K: int,
    w2_shard_map: list[tuple[int, int]],
    w0_w1_shard_map: list[int],
) -> "torch.Tensor":
    """
    Prepare the w2 tensor input for moe_compute by padding and reordering tiles.

    Args:
        torch_w2: Weight tensor of shape (L, E, N, K)
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension
        w2_shard_map: List of tuples (last_group_tiles, last_group_pad_tiles) for each core
        w0_w1_shard_map: List of shard sizes from w0_w1 preparation

    Returns:
        torch_w2_reordered: Reordered tensor of shape (L, E, N_padded, 7680)
    """
    import torch

    # Check that N and K are divisible by ttnn.TILE_SIZE
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Kt = K // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    each_shard = []

    start_col = 0
    # groups are always 4 tiles wide in K, and full N
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        # Get the first 4 groups of 4 * 32 tiles.
        each_shard.append(torch_w2[:, :, :, start_col : start_col + (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE])
        start_col += (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
        each_shard.append(torch_w2[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        # Add padding for the last group.
        if last_group_pad_tiles > 0:
            each_shard.append(torch.zeros(L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_reordered = torch.cat(each_shard, dim=-1)
    all_groups_per_bank = torch_w2_reordered.view(L, E, N, num_cores, -1, 4 * ttnn.TILE_SIZE)

    # (L, E, N, 12, 5, 128) -> (12, L, E, 5, N, 128)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    # Group N in terms of tiles first
    N_grouped = all_groups_per_bank.view(
        num_cores, L, E, w2_groups_per_core, -1, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE
    )  # (12, L, E, num groups, 64, 32, 128)

    # Figure out the order of N tiles based on the ring position.
    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)

    # Figure out the starting position for each chunk
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(w0_w1_shard_map, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    # Assemble the number of such N tiles based on the ring position.
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))

        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(num_cores, L, E, w2_groups_per_core, -1, 4 * ttnn.TILE_SIZE)

    # Pad "N" dimension to make it divisible by 7 tiles, since we read 7 tiles at a time.
    Nt = N // ttnn.TILE_SIZE
    N_padding = math.ceil(Nt / BLOCK_TILES_H) * BLOCK_TILES_H * ttnn.TILE_SIZE - N
    padding = torch.zeros(num_cores, L, E, w2_groups_per_core, N_padding, 4 * ttnn.TILE_SIZE, dtype=torch_w2.dtype)
    all_groups_per_bank = torch.cat([N_reordered, padding], dim=4)
    return all_groups_per_bank


def prepare_w0_w1_tensor_with_bias(
    torch_w0: "torch.Tensor",
    torch_w1: "torch.Tensor",
    torch_b0: "torch.Tensor",
    torch_b1: "torch.Tensor",
    L: int,
    E: int,
    K: int,
    N: int,
    shard_map: list[int],
):
    """
    Prepare the w0_w1 tensor with bias by concatenating bias rows along K dimension,
    padding to transaction-aligned height, then delegating to prepare_w0_w1_tensor_for_moe_compute.

    Converts true PyTorch bias format (L, E, N) to kernel tile format (L, E, 32, N) with
    only the first row populated, then concatenates to weights along K dimension.

    The kernel reads W0/W1 in blocks of (W0_W1_TILES_PER_TXN * 2) tiles.
    With bias, K goes from K/32 tiles to (K/32 + 1) tiles. If (K/32 + 1) is not divisible by
    TILES_PER_TXN, the kernel reads extra padding tiles. The weight tensor must contain those
    padding tiles (zeros) so the DRAM reads don't overrun the expert boundary.

    Args:
        torch_w0: Weight tensor of shape (L, E, K, N)
        torch_w1: Weight tensor of shape (L, E, K, N)
        torch_b0: Bias tensor of shape (L, E, N) -- true PyTorch format
        torch_b1: Bias tensor of shape (L, E, N) -- true PyTorch format
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension
        shard_map: List of shard sizes for each core

    Returns:
        torch_w0_w1_paired: Prepared tensor with bias of shape (num_cores, L, E, groups_per_core, K_padded, 4*ttnn.TILE_SIZE)

    See also:
        Module docstring for full layout contract and constants that must match
        moe_ring_common.h; ``prepare_w0_w1_tensor_for_moe_compute`` for the non-bias path.
    """
    import torch

    # Check that K and N are divisible by ttnn.TILE_SIZE
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    # This constant must match moe_ring_common.h — it determines the DRAM read block alignment.
    K_tiles = K // ttnn.TILE_SIZE
    K_tiles_with_bias = K_tiles + 1
    K_with_bias = K_tiles_with_bias * ttnn.TILE_SIZE

    # Convert true PyTorch bias (L, E, N) to kernel tile format (L, E, 32, N) with only row 0 populated.
    torch_b0_tiled = torch.zeros(L, E, ttnn.TILE_SIZE, N, dtype=torch_b0.dtype)
    torch_b0_tiled[:, :, 0, :] = torch_b0
    torch_b1_tiled = torch.zeros(L, E, ttnn.TILE_SIZE, N, dtype=torch_b1.dtype)
    torch_b1_tiled[:, :, 0, :] = torch_b1

    torch_w0_b0 = torch.cat([torch_w0, torch_b0_tiled], dim=2)  # (L, E, K+32, N)
    torch_w1_b1 = torch.cat([torch_w1, torch_b1_tiled], dim=2)  # (L, E, K+32, N)

    return prepare_w0_w1_tensor_for_moe_compute(torch_w0_b0, torch_w1_b1, L, E, K_with_bias, N, shard_map)


def prepare_w2_tensor_with_bias(
    torch_w2: "torch.Tensor",
    torch_b2: "torch.Tensor",
    L: int,
    E: int,
    N: int,
    K: int,
    w2_shard_map: list[tuple[int, int]],
    w0_w1_shard_map: list[int],
) -> "torch.Tensor":
    """
    Prepare the w2 tensor with bias. The bias tile row is concatenated along N,
    but only the weight tiles are ring-rotated — the bias tile stays fixed at
    position N/32 for all cores (matching GPT-OSS behavior).

    Converts true PyTorch bias format (L, E, K) to kernel tile format (L, E, 32, K)
    with only the first row populated, then performs K-column sharding.

    Args:
        torch_w2: Weight tensor of shape (L, E, N, K)
        torch_b2: Bias tensor of shape (L, E, K) -- true PyTorch format
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension
        w2_shard_map: List of tuples (last_group_tiles, last_group_pad_tiles) for each core
        w0_w1_shard_map: List of shard sizes from w0_w1 preparation

    Returns:
        N_with_bias: Prepared tensor of shape (num_cores, L, E, groups_per_core, N_target, 4*ttnn.TILE_SIZE)

    See also:
        Module docstring for full layout contract and constants that must match
        moe_ring_common.h; ``prepare_w2_tensor_for_moe_compute`` for the non-bias path.
    """
    import torch

    # Check that N and K are divisible by ttnn.TILE_SIZE
    if N % ttnn.TILE_SIZE != 0:
        raise ValueError(f"N dimension ({N}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if K % ttnn.TILE_SIZE != 0:
        raise ValueError(f"K dimension ({K}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")

    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    # Convert true PyTorch bias (L, E, K) to kernel tile format (L, E, 32, K) with only row 0 populated.
    torch_b2_tiled = torch.zeros(L, E, ttnn.TILE_SIZE, K, dtype=torch_b2.dtype)
    torch_b2_tiled[:, :, 0, :] = torch_b2

    # Column-shard K dimension for weights
    each_shard = []
    start_col = 0
    # groups are always 4 tiles wide in K, and full N
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        # Get the first 4 groups of 4 * 32 tiles.
        each_shard.append(torch_w2[:, :, :, start_col : start_col + (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE])
        start_col += (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
        each_shard.append(torch_w2[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        # Add padding for the last group.
        if last_group_pad_tiles > 0:
            each_shard.append(torch.zeros(L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_reordered = torch.cat(each_shard, dim=-1)
    all_groups_per_bank = torch_w2_reordered.view(L, E, N, num_cores, -1, 4 * ttnn.TILE_SIZE)

    # (L, E, N, 12, groups_per_core, 128) -> (12, L, E, groups_per_core, N, 128)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    # Group N in terms of tiles (weight tiles only, no bias yet)
    N_grouped = all_groups_per_bank.view(
        num_cores, L, E, w2_groups_per_core, -1, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE
    )  # (12, L, E, num groups, Nt, 32, 128)

    # Figure out the order of N tiles based on the ring position.
    core_chunk_order = torch.tensor(list(reversed(range(num_cores)))).roll(1)

    # Figure out the starting position for each chunk
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(w0_w1_shard_map, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    # Assemble the number of such N tiles based on the ring position.
    for core_id in range(num_cores):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))

        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(num_cores, L, E, w2_groups_per_core, -1, 4 * ttnn.TILE_SIZE)

    # Now prepare bias tile row with the same K-column sharding
    b2_each_shard = []
    start_col = 0
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        b2_each_shard.append(
            torch_b2_tiled[:, :, :, start_col : start_col + (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE]
        )
        start_col += (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
        b2_each_shard.append(torch_b2_tiled[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        if last_group_pad_tiles > 0:
            b2_each_shard.append(
                torch.zeros(L, E, ttnn.TILE_SIZE, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_b2_tiled.dtype)
            )

    torch_b2_reordered = torch.cat(b2_each_shard, dim=-1)
    b2_groups_per_bank = torch_b2_reordered.view(L, E, ttnn.TILE_SIZE, num_cores, -1, 4 * ttnn.TILE_SIZE)
    b2_groups_per_bank = b2_groups_per_bank.permute(3, 0, 1, 4, 2, 5)  # (12, L, E, groups_per_core, 32, 128)

    # Concatenate bias tile row after weight tiles (NOT ring-rotated)
    N_with_bias = torch.cat([N_reordered, b2_groups_per_bank], dim=4)  # (12, L, E, groups_per_core, N+32, 128)

    # Pad "N+32" dimension so total height matches what dm0 expects.
    # We need to pad to make the total divisible by tiles_per_txn for the pipelined DRAM reads.
    N_total_tiles = Nt + 1  # Weight tiles + 1 bias tile
    # Pad to align with transaction boundary (7 tiles in A2A iteration)
    N_target_tiles = math.ceil(N_total_tiles / BLOCK_TILES_H) * BLOCK_TILES_H
    N_target = N_target_tiles * ttnn.TILE_SIZE
    N_padding = N_target - (N + ttnn.TILE_SIZE)

    if N_padding > 0:
        padding = torch.zeros(num_cores, L, E, w2_groups_per_core, N_padding, 4 * ttnn.TILE_SIZE, dtype=torch_w2.dtype)
        N_with_bias = torch.cat([N_with_bias, padding], dim=4)

    return N_with_bias


def get_weight_core_shard_maps(mesh_device, pad_cores, w0_w1_shard_vals, w2_shard_vals):
    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    sorted_dram_core_coords = []
    w0_w1_shard_map = []
    w2_shard_map = []
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        sorted_dram_core_coords.append(core2dram[core_coord])
        w0_w1_shard_map.append(w0_w1_shard_vals[ring_pos in pad_cores])
        w2_shard_map.append(w2_shard_vals[ring_pos in pad_cores])

    dram_core_coords = [ttnn.CoreCoord(c, 0) for c in sorted_dram_core_coords]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    return w0_w1_shard_map, w2_shard_map, dram_core_range_set


def get_weight_mem_configs(
    num_layers,
    experts_per_device,
    hidden_size,
    intermediate_size,
    w0_w1_shard_map,
    w2_shard_map,
    dram_core_range_set,
    has_bias=False,
):
    """
    Get memory configurations for W0/W1 and W2 weight tensors.

    When has_bias=True:
    - W0/W1: K dimension grows by 1 tile (for bias) and is padded to transaction boundary
    - W2: N dimension grows by 1 tile (for bias) and is padded to align with 7-tile reads

    Returns:
        tuple: (w0_w1_mem_config, w2_mem_config, K_for_shard, w2_N_total)
            - K_for_shard: The padded K dimension for W0/W1
            - w2_N_total: The padded N dimension for W2
    """

    # Check that hidden_size and intermediate_size are divisible by ttnn.TILE_SIZE
    if hidden_size % ttnn.TILE_SIZE != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})")
    if intermediate_size % ttnn.TILE_SIZE != 0:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) must be divisible by ttnn.TILE_SIZE ({ttnn.TILE_SIZE})"
        )

    # Calculate K dimension for W0/W1
    if has_bias:
        K_tiles = hidden_size // ttnn.TILE_SIZE
        K_tiles_with_bias = K_tiles + 1  # Add 1 tile for bias
        K_tiles_padded = math.ceil(K_tiles_with_bias / BLOCK_TILES_H) * BLOCK_TILES_H
        K_for_shard = K_tiles_padded * ttnn.TILE_SIZE
    else:
        # Without bias, just pad to BLOCK_TILES_H
        K_for_shard = math.ceil(hidden_size // ttnn.TILE_SIZE / BLOCK_TILES_H) * ttnn.TILE_SIZE * BLOCK_TILES_H

    # W0/W1 memory config
    w1_w0_groups_per_core = max(w0_w1_shard_map) // 2
    w0_w1_shard_height = num_layers * experts_per_device * w1_w0_groups_per_core * K_for_shard
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # Calculate N dimension for W2
    Nt = intermediate_size // ttnn.TILE_SIZE
    Ht = hidden_size // ttnn.TILE_SIZE
    w2_groups_per_core = math.ceil(Ht / (len(w2_shard_map) * sum(w2_shard_map[0])))

    if has_bias:
        # With bias: N grows by 1 tile, then pad to align with 7-tile reads
        Nt_with_bias = Nt + 1
        Nt_padded = math.ceil(Nt_with_bias / BLOCK_TILES_H) * BLOCK_TILES_H
        w2_N_total = Nt_padded * ttnn.TILE_SIZE
    else:
        # Without bias: just pad to 7-tile alignment
        w2_N_total = math.ceil(Nt / BLOCK_TILES_H) * BLOCK_TILES_H * ttnn.TILE_SIZE

    # W2 memory config
    w2_shard_height = num_layers * experts_per_device * w2_groups_per_core * w2_N_total
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    return w0_w1_mem_config, w2_mem_config, K_for_shard, w2_N_total
