# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Python helper utilities for ttnn.experimental.moe_compute """

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

    Nt = N // ttnn.TILE_SIZE  # 2048 / 32 = 64 chunks per tensor
    num_cores = len(shard_map)

    # Reshape to expose chunks: (L, E, K, N) -> (L, E, K, Nt, ttnn.TILE_SIZE)
    w0_chunks = torch_w0.view(L, E, K, Nt, ttnn.TILE_SIZE)
    w1_chunks = torch_w1.view(L, E, K, Nt, ttnn.TILE_SIZE)

    # Stack w0 and w1 chunks together: (L, E, K, Nt, 2, ttnn.TILE_SIZE)
    # This puts w0_chunk_i and w1_chunk_i adjacent to each other
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)

    # Reshape to interleave: (L, E, K, Nt * 2 * ttnn.TILE_SIZE) = (L, E, K, 4096)
    # The order will be: w0_chunk_0, w1_chunk_0, w0_chunk_1, w1_chunk_1, ...
    torch_w0_w1_interleaved = stacked.view(L, E, K, Nt, 2 * ttnn.TILE_SIZE)

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
            each_shard.append(torch.zeros(L, E, 1, K, 2 * ttnn.TILE_SIZE, dtype=torch_w0_w1_permuted.dtype))
        start_tile += num_tiles

    torch_w0_w1_reordered = torch.cat(each_shard, dim=2)  # (L, E, 5 * 8 + 1 * 8 + 6 * 4, K, 64)
    all_groups_per_bank = torch_w0_w1_reordered.view(L, E, num_cores, -1, K, 2 * ttnn.TILE_SIZE)  # (L, E, 12, 6, K, 64)
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)  # (12, L, E, 6, K, 64)

    groups_per_core = max_shard_size // 2

    # Let us further make the 6 as 3 and 64 as 128.
    torch_w0_w1_pair_2_tiles = all_groups_per_bank.view(num_cores, L, E, groups_per_core, -1, K, 2 * ttnn.TILE_SIZE)
    # (12, L, E, 3, 2, K, 64) -> (12, L, E, 3, K, 2, 64)
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

    Kt = K // ttnn.TILE_SIZE
    num_cores = len(w2_shard_map)
    w2_groups_per_core = math.ceil(Kt / (num_cores * sum(w2_shard_map[0])))

    # Separate the tensor into groups of 4 * 32 tiles and then 1 group of 2/3 * 32 tiles.
    each_shard = []

    start_col = 0
    for last_group_tiles, last_group_pad_tiles in w2_shard_map:
        # Get the first 4 groups of 4 * 32 tiles.
        each_shard.append(torch_w2[:, :, :, start_col : start_col + (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE])
        start_col += (w2_groups_per_core - 1) * 4 * ttnn.TILE_SIZE
        each_shard.append(torch_w2[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        # Add padding for the last group.
        if last_group_pad_tiles > 0:
            each_shard.append(torch.zeros(L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_reordered = torch.cat(each_shard, dim=-1)  # (L, E, N, 12 * (4 * 4 * 32 + 4 * 32))
    all_groups_per_bank = torch_w2_reordered.view(L, E, N, num_cores, -1, 4 * ttnn.TILE_SIZE)

    # (L, E, N, 12, 5, 128) -> (12, L, E, 5, N, 128)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    # Group N in terms of tiles first
    N_grouped = all_groups_per_bank.view(
        num_cores, L, E, w2_groups_per_core, -1, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE
    )  # (12, L, E, 5, 64, 32, 128)

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
    Nt = N // ttnn.TILE_SIZE  # 2048 / 32 = 64 chunks per tensor
    N_padding = math.ceil(Nt / 7) * 7 * ttnn.TILE_SIZE - N
    padding = torch.zeros(num_cores, L, E, w2_groups_per_core, N_padding, 4 * ttnn.TILE_SIZE, dtype=torch_w2.dtype)
    all_groups_per_bank = torch.cat([N_reordered, padding], dim=4)  # (12, L, E, 5, N + 192, 128)
    return all_groups_per_bank


DS_PAD_CORES = {1, 2, 4, 5, 7, 8, 10, 11}
DS_W0_W1_SHARD_VALS = [6, 5]
DS_W2_SHARD_VALS = {False: (2, 2), True: (3, 1)}  # mapped to pad core assignment

GPT_PAD_CORES = {2, 3, 6, 7, 10, 11}
GPT_W0_W1_SHARD_VALS = [8, 7]
GPT_W2_SHARD_VALS = {False: (4, 0), True: (3, 1)}


def get_weight_core_shard_maps(mesh_device, pad_cores, w0_w1_shard_vals, w2_shard_vals):
    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

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
    num_layers, experts_per_device, hidden_size, intermediate_size, w0_w1_shard_map, w2_shard_map, dram_core_range_set
):
    w1_w0_groups_per_core = max(w0_w1_shard_map) // 2
    w0_w1_shard_height = num_layers * experts_per_device * w1_w0_groups_per_core * hidden_size
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (num_layers, experts_per_device, N, hidden_size) -> padded and reordered to (12, num_layers, experts_per_device, 5, N + 192, 128)
    # ------------------------------------------------------------------------
    Nt = intermediate_size // ttnn.TILE_SIZE
    Ht = hidden_size // ttnn.TILE_SIZE
    w2_groups_per_core = math.ceil(Ht / (len(w2_shard_map) * sum(w2_shard_map[0])))

    w2_shard_height = num_layers * experts_per_device * w2_groups_per_core * math.ceil(Nt / 7) * 7 * ttnn.TILE_SIZE
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    return w0_w1_mem_config, w2_mem_config
