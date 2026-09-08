# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

Topology = ttnn._ttnn.operations.ccl.Topology

get_usable_topology = ttnn._ttnn.operations.ccl.get_usable_topology

# Experimental CCL enums for all_to_all_dispatch_metadata operation
DispatchAlgorithm = ttnn._ttnn.operations.experimental.ccl_experimental.DispatchAlgorithm
WorkerMode = ttnn._ttnn.operations.experimental.ccl_experimental.WorkerMode

# Experimental CCL enum for moe_compute operation
MoEActivationFunction = ttnn._ttnn.operations.experimental.ccl_experimental.MoEActivationFunction

# Experimental CCL enum for strided_all_gather_minimal_matmul_async operation
MMSignalAggregatorMode = ttnn._ttnn.operations.experimental.ccl_experimental.MMSignalAggregatorMode


def _preprocess_collective_golden_inputs(function_args, function_kwargs):
    input_tensor = function_args[0] if function_args else function_kwargs["input_tensor"]
    distributed_input = ttnn.decorators.distributed_golden_for_comparison(input_tensor)

    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    if function_args:
        function_args[0] = distributed_input
    else:
        function_kwargs["input_tensor"] = distributed_input
    return tuple(function_args), function_kwargs


def _get_collective_groups(topology, cluster_axis):
    """Return topology coordinates grouped along the collective axis."""

    mesh_coords = list(topology.mesh_coords)
    if cluster_axis is None:
        return [mesh_coords]

    groups = {}
    for mesh_coord in mesh_coords:
        coordinate = tuple(int(value) for value in mesh_coord)
        group_coordinate = coordinate[:cluster_axis] + coordinate[cluster_axis + 1 :]
        groups.setdefault(group_coordinate, []).append(mesh_coord)
    return list(groups.values())


def _topology_shard_dims(topology):
    return tuple(
        placement.dim if isinstance(placement, ttnn.PlacementShard) else None for placement in topology.placements
    )


def _ordered_distributed_shards(distributed_golden):
    if not isinstance(distributed_golden, ttnn.DistributedGolden):
        raise TypeError(f"Expected DistributedGolden, got {type(distributed_golden)}")
    if distributed_golden.shards is None:
        raise ValueError("Collective golden requires coordinate-keyed input shards")

    shards = dict(distributed_golden.shards)
    topology_coords = tuple(distributed_golden.topology.mesh_coords)
    if distributed_golden.global_value is not None and len(shards) != len(topology_coords):
        representative_shape = tuple(next(iter(shards.values())).shape)
        shard_shapes = {mesh_coord: representative_shape for mesh_coord in topology_coords}
        shards = ttnn.decompose_mesh_value(
            distributed_golden.global_value,
            topology=distributed_golden.topology,
            shard_shapes_by_mesh_coord=shard_shapes,
        )

    return {mesh_coord: shards[mesh_coord] for mesh_coord in topology_coords if mesh_coord in shards}


def _collective_groups(distributed_golden, input_shards, cluster_axis):
    groups = [
        group
        for group in _get_collective_groups(distributed_golden.topology, cluster_axis)
        if all(mesh_coord in input_shards for mesh_coord in group)
    ]
    if not groups:
        raise ValueError("Collective golden requires all shards for at least one collective group")
    return groups


def _output_compare_coords(distributed_golden, output_shards):
    if distributed_golden.compare_coords is None:
        return None
    return frozenset(mesh_coord for mesh_coord in distributed_golden.compare_coords if mesh_coord in output_shards)


def _distributed_collective_golden(
    output_shards,
    tensor_topology,
    mesh_shard_dims,
    *,
    compare_coords=None,
):
    placements = [
        ttnn.PlacementReplicate() if shard_dim is None else ttnn.PlacementShard(shard_dim)
        for shard_dim in mesh_shard_dims
    ]
    output_topology = tensor_topology.with_placements(placements)
    shards = dict(output_shards)
    global_value = None
    if set(shards) == set(output_topology.mesh_coords):
        global_value = ttnn.compose_mesh_value(shards_by_mesh_coord=shards, topology=output_topology)
    return ttnn.DistributedGolden(
        topology=output_topology,
        global_value=global_value,
        shards=shards,
        compare_coords=compare_coords,
    )


def _canonical_mesh_coord(topology, coordinate):
    coordinate_key = tuple(int(value) for value in coordinate)
    for mesh_coord in topology.mesh_coords:
        if tuple(int(value) for value in mesh_coord) == coordinate_key:
            return mesh_coord
    raise ValueError(f"Mesh coordinate {coordinate_key} is not present in the tensor topology")


def _normalize_dim(dim, rank):
    return dim if dim >= 0 else dim + rank


def _replace_matching_shards_with_replicas(mesh_shard_dims, dim, rank):
    normalized_dim = _normalize_dim(dim, rank)
    return tuple(
        None if shard_dim is not None and _normalize_dim(shard_dim, rank) == normalized_dim else shard_dim
        for shard_dim in mesh_shard_dims
    )


def _golden_function_all_broadcast(
    input_tensor,
    *args,
    cluster_axis=None,
    **kwargs,
):
    input_shards = _ordered_distributed_shards(input_tensor)
    input_shard_dims = _topology_shard_dims(input_tensor.topology)
    groups = _collective_groups(input_tensor, input_shards, cluster_axis)
    group_size = len(groups[0])
    per_result_device_outputs = [{} for _ in range(group_size)]
    for group in groups:
        for result_index, source_coord in enumerate(group):
            for destination_coord in group:
                per_result_device_outputs[result_index][destination_coord] = input_shards[source_coord]

    output_shard_dims = list(input_shard_dims)
    if cluster_axis is None:
        output_shard_dims = [None] * len(output_shard_dims)
    else:
        output_shard_dims[cluster_axis] = None
    return [
        _distributed_collective_golden(
            outputs,
            input_tensor.topology,
            output_shard_dims,
            compare_coords=_output_compare_coords(input_tensor, outputs),
        )
        for outputs in per_result_device_outputs
    ]


ttnn.attach_golden_function(
    ttnn.all_broadcast,
    golden_function=_golden_function_all_broadcast,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_all_gather(
    input_tensor,
    dim,
    *args,
    cluster_axis=None,
    **kwargs,
):
    import torch

    input_shards = _ordered_distributed_shards(input_tensor)
    input_shard_dims = _topology_shard_dims(input_tensor.topology)
    output_shards = {}
    for group in _collective_groups(input_tensor, input_shards, cluster_axis):
        gathered = torch.cat([input_shards[mesh_coord] for mesh_coord in group], dim=dim)
        for mesh_coord in group:
            output_shards[mesh_coord] = gathered

    input_rank = next(iter(input_shards.values())).ndim
    output_shard_dims = _replace_matching_shards_with_replicas(input_shard_dims, dim, input_rank)
    return _distributed_collective_golden(
        output_shards,
        input_tensor.topology,
        output_shard_dims,
        compare_coords=_output_compare_coords(input_tensor, output_shards),
    )


ttnn.attach_golden_function(
    ttnn.all_gather,
    golden_function=_golden_function_all_gather,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_all_reduce(
    input_tensor,
    *args,
    cluster_axis=None,
    **kwargs,
):
    import torch

    input_shards = _ordered_distributed_shards(input_tensor)
    input_shard_dims = _topology_shard_dims(input_tensor.topology)
    output_shards = {}
    for group in _collective_groups(input_tensor, input_shards, cluster_axis):
        reduced = torch.stack([input_shards[mesh_coord] for mesh_coord in group]).sum(dim=0)
        for mesh_coord in group:
            output_shards[mesh_coord] = reduced

    output_shard_dims = list(input_shard_dims)
    if cluster_axis is None:
        output_shard_dims = [None] * len(output_shard_dims)
    else:
        output_shard_dims[cluster_axis] = None
    return _distributed_collective_golden(
        output_shards,
        input_tensor.topology,
        output_shard_dims,
        compare_coords=_output_compare_coords(input_tensor, output_shards),
    )


ttnn.attach_golden_function(
    ttnn.all_reduce,
    golden_function=_golden_function_all_reduce,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_reduce_scatter(
    input_tensor,
    dim,
    *args,
    cluster_axis=None,
    **kwargs,
):
    import torch

    input_shards = _ordered_distributed_shards(input_tensor)
    mesh_shape = input_tensor.topology.distribution_shape
    input_shard_dims = _topology_shard_dims(input_tensor.topology)
    output_shards = {}
    for group in _collective_groups(input_tensor, input_shards, cluster_axis):
        reduced = torch.stack([input_shards[mesh_coord] for mesh_coord in group]).sum(dim=0)
        for mesh_coord, chunk in zip(group, torch.chunk(reduced, len(group), dim=dim)):
            output_shards[mesh_coord] = chunk

    input_rank = next(iter(input_shards.values())).ndim
    output_shard_dims = list(_replace_matching_shards_with_replicas(input_shard_dims, dim, input_rank))
    normalized_dim = _normalize_dim(dim, input_rank)
    if cluster_axis is None:
        for axis, dimension in enumerate(mesh_shape):
            output_shard_dims[axis] = normalized_dim if dimension > 1 else None
    else:
        output_shard_dims[cluster_axis] = normalized_dim
    return _distributed_collective_golden(
        output_shards,
        input_tensor.topology,
        output_shard_dims,
        compare_coords=_output_compare_coords(input_tensor, output_shards),
    )


ttnn.attach_golden_function(
    ttnn.reduce_scatter,
    golden_function=_golden_function_reduce_scatter,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_point_to_point(
    input_tensor,
    sender_coord,
    receiver_coord,
    *args,
    **kwargs,
):
    if input_tensor.shards is None:
        raise ValueError("Point-to-point golden requires coordinate-keyed input shards")
    sender_coord = _canonical_mesh_coord(input_tensor.topology, sender_coord)
    receiver_coord = _canonical_mesh_coord(input_tensor.topology, receiver_coord)
    return ttnn.DistributedGolden(
        topology=input_tensor.topology,
        shards={receiver_coord: input_tensor.shards[sender_coord].clone()},
        compare_coords=frozenset({receiver_coord}),
    )


ttnn.attach_golden_function(
    ttnn.point_to_point,
    golden_function=_golden_function_point_to_point,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_all_to_all_dispatch(
    input_tensor,
    expert_indices_tensor,
    expert_mapping_tensor,
    *args,
    **kwargs,
):
    import torch

    expert_indices = expert_indices_tensor[:, 0].long()
    expert_mapping = expert_mapping_tensor[0, 0].bool()
    num_devices = expert_mapping.shape[-1]

    selected_devices = expert_mapping[expert_indices].any(dim=2)
    dispatch_mask = selected_devices.permute(2, 0, 1)
    input_tokens = input_tensor[:, 0]
    output_tokens = torch.zeros(
        (num_devices, *input_tokens.shape),
        dtype=input_tokens.dtype,
        device=input_tokens.device,
    )
    output_tokens[dispatch_mask] = input_tokens.unsqueeze(0).expand(num_devices, -1, -1, -1)[dispatch_mask]
    ttnn.decorators.set_golden_comparison_config(
        output_tokens, method="allclose", scope="all", rtol=0.0, atol=0.0, mask=dispatch_mask
    )

    # Metadata is all-gathered; placeholder token rows are excluded from comparison.
    output_metadata = expert_indices_tensor[:, 0].unsqueeze(0).expand(num_devices, -1, -1, -1).clone()
    return output_tokens, output_metadata


ttnn.attach_golden_function(ttnn.all_to_all_dispatch, golden_function=_golden_function_all_to_all_dispatch)


def _golden_function_all_to_all_combine(
    input_tensor,
    expert_metadata_tensor,
    expert_mapping_tensor,
    *args,
    local_reduce=False,
    **kwargs,
):
    import torch

    expert_mapping = expert_mapping_tensor[0, 0].bool()
    # Metadata is replicated on its leading mesh-device axis. Select one complete
    # [batch, sequence, selected_experts] copy without dropping the batch axis.
    expert_metadata = expert_metadata_tensor[0].long()

    batch, sequence, selected_experts = expert_metadata.shape
    output = torch.zeros(
        (selected_experts, batch, sequence, input_tensor.shape[-1]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    populated_slots = torch.zeros((selected_experts, batch, sequence), dtype=torch.bool, device=input_tensor.device)

    # Reconstruct each sparse expert contribution in its original top-k slot.
    for batch_index in range(batch):
        for sequence_index in range(sequence):
            if local_reduce:
                selected = expert_metadata[batch_index, sequence_index]
                for device_index in range(expert_mapping.shape[-1]):
                    local_experts = torch.nonzero(expert_mapping[:, device_index], as_tuple=False).flatten()
                    for expert_index in local_experts:
                        matching_topk = torch.nonzero(selected == expert_index, as_tuple=False).flatten()
                        if matching_topk.numel() == 0:
                            continue
                        topk_index = int(matching_topk[0])
                        output[topk_index, batch_index, sequence_index] = input_tensor[
                            device_index, batch_index, sequence_index
                        ]
                        populated_slots[topk_index, batch_index, sequence_index] = True
                        break
            else:
                for topk_index in range(selected_experts):
                    expert_index = int(expert_metadata[batch_index, sequence_index, topk_index])
                    output[topk_index, batch_index, sequence_index] = input_tensor[
                        expert_index, batch_index, sequence_index
                    ]
                    populated_slots[topk_index, batch_index, sequence_index] = True

    ttnn.decorators.set_golden_comparison_config(
        output, method="allclose", scope="all", rtol=0.0, atol=0.0, mask=populated_slots
    )
    return output


ttnn.attach_golden_function(ttnn.all_to_all_combine, golden_function=_golden_function_all_to_all_combine)


def _preprocess_reduce_to_root_golden_inputs(function_args, function_kwargs):
    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    input_names = ("input_tensor_l", "input_tensor_s", "input_tensor_m")

    for index, input_name in enumerate(input_names):
        input_tensor = function_args[index] if index < len(function_args) else function_kwargs[input_name]
        distributed_input = ttnn.decorators.distributed_golden_for_comparison(input_tensor)
        if index < len(function_args):
            function_args[index] = distributed_input
        else:
            function_kwargs[input_name] = distributed_input

    return tuple(function_args), function_kwargs


def _golden_function_reduce_to_root(
    input_tensor_l,
    input_tensor_s,
    input_tensor_m,
    root_coord,
    *args,
    scale_fp32=1.0,
    **kwargs,
):
    import torch

    input_shards_l_by_coord = _ordered_distributed_shards(input_tensor_l)
    input_shards_s_by_coord = _ordered_distributed_shards(input_tensor_s)
    input_shards_m_by_coord = _ordered_distributed_shards(input_tensor_m)
    input_coords = [
        mesh_coord
        for mesh_coord in input_tensor_l.topology.mesh_coords
        if mesh_coord in input_shards_l_by_coord
        and mesh_coord in input_shards_s_by_coord
        and mesh_coord in input_shards_m_by_coord
    ]
    if len(input_coords) != 4:
        raise ValueError("reduce_to_root golden requires the operation's fixed four-device topology")
    input_shards_l = [input_shards_l_by_coord[mesh_coord] for mesh_coord in input_coords]
    input_shards_s = [input_shards_s_by_coord[mesh_coord] for mesh_coord in input_coords]
    input_shards_m = [input_shards_m_by_coord[mesh_coord] for mesh_coord in input_coords]

    tile_width = 32
    num_cores = input_shards_s[0].shape[-1] // tile_width
    if num_cores == 0 or input_shards_s[0].shape[-1] % tile_width != 0:
        raise ValueError("reduce_to_root golden requires tile-aligned S state")

    states = []
    for tensor_l, tensor_s, tensor_m in zip(input_shards_l, input_shards_s, input_shards_m):
        if tensor_s.shape != tensor_m.shape or tensor_l.shape[-1] % num_cores != 0:
            raise ValueError("reduce_to_root golden received incompatible L, S, and M state shapes")
        l_core_width = tensor_l.shape[-1] // num_cores
        states.append(
            (
                tensor_l.reshape(*tensor_l.shape[:-1], num_cores, l_core_width),
                tensor_s.reshape(*tensor_s.shape[:-1], num_cores, tile_width),
                tensor_m.reshape(*tensor_m.shape[:-1], num_cores, tile_width),
            )
        )

    def reduce_states(state_a, state_b):
        tensor_l_a, tensor_s_a, tensor_m_a = state_a
        tensor_l_b, tensor_s_b, tensor_m_b = state_b
        tensor_m = torch.maximum(tensor_m_a, tensor_m_b)
        scale_a = torch.exp((tensor_m_a - tensor_m) * scale_fp32)
        scale_b = torch.exp((tensor_m_b - tensor_m) * scale_fp32)
        tensor_s = tensor_s_a * scale_a + tensor_s_b * scale_b
        l_core_width = tensor_l_a.shape[-1]
        tensor_l = tensor_l_a * scale_a[..., :1].expand(*scale_a.shape[:-1], l_core_width)
        tensor_l += tensor_l_b * scale_b[..., :1].expand(*scale_b.shape[:-1], l_core_width)
        return tensor_l, tensor_s, tensor_m

    left_reduction = reduce_states(states[0], states[1])
    right_reduction = reduce_states(states[3], states[2])
    tensor_l, tensor_s, tensor_m = reduce_states(right_reduction, left_reduction)
    tensor_l = tensor_l / tensor_s[..., :1].expand(*tensor_l.shape)

    output_l = tensor_l.reshape(input_shards_l[0].shape)
    output_s = tensor_s.reshape(input_shards_s[0].shape)
    output_m = tensor_m.reshape(input_shards_m[0].shape)
    root_coord = _canonical_mesh_coord(input_tensor_l.topology, root_coord)

    def root_golden(output):
        return ttnn.DistributedGolden(
            topology=input_tensor_l.topology,
            shards={root_coord: output},
            compare_coords=frozenset({root_coord}),
        )

    return root_golden(output_l), root_golden(output_s), root_golden(output_m)


ttnn.attach_golden_function(
    ttnn.reduce_to_root,
    golden_function=_golden_function_reduce_to_root,
    preprocess_golden_function_inputs=_preprocess_reduce_to_root_golden_inputs,
)


def _golden_function_moe(
    input_tensor,
    expert_mask_tensor,
    topk_mask_tensor,
    k=32,
    *args,
    **kwargs,
):
    import torch

    if input_tensor.numel() == 0:
        output_shape = list(input_tensor.shape)
        output_shape[-1] = 1
        return torch.zeros(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    topk_values, topk_indices = torch.topk(input_tensor + expert_mask_tensor, k, dim=-1)
    topk_weights = torch.softmax(topk_values + topk_mask_tensor, dim=-1)
    # The operation returns the combined routing weight assigned to expert zero.
    return torch.sum(topk_weights * (topk_indices == 0), dim=-1, keepdim=True)


ttnn.attach_golden_function(ttnn.moe, golden_function=_golden_function_moe)


def _golden_function_moe_expert_token_remap(
    topk_tensor,
    expert_mapping_tensor,
    expert_metadata_tensor,
    *args,
    reduction_size=16,
    **kwargs,
):
    import torch
    import torch.nn.functional as F

    expert_mapping = expert_mapping_tensor[0, 0].bool()
    num_devices = expert_mapping.shape[-1]
    if topk_tensor.shape[0] == 1:
        topk_tensor = topk_tensor.expand(num_devices, -1, -1, -1)
    if expert_metadata_tensor.shape[0] == 1:
        expert_metadata_tensor = expert_metadata_tensor.expand(num_devices, -1, -1, -1)

    per_device_outputs = []
    for device_index in range(num_devices):
        local_experts = torch.nonzero(expert_mapping[:, device_index], as_tuple=False).flatten()
        local_weights = topk_tensor[device_index, ..., local_experts]
        selected = (expert_metadata_tensor[device_index].long().unsqueeze(-1) == local_experts).any(dim=-2)
        per_device_outputs.append(torch.where(selected, local_weights, 0))
    output_mapping = torch.stack(per_device_outputs)

    flattened_mapping = output_mapping.reshape(num_devices, -1, output_mapping.shape[-1])
    padding = (-flattened_mapping.shape[1]) % reduction_size
    if padding:
        flattened_mapping = F.pad(flattened_mapping, (0, 0, 0, padding))
    output_reduced = (
        flattened_mapping.reshape(num_devices, -1, reduction_size, output_mapping.shape[-1])
        .bool()
        .any(dim=2)
        .to(torch.int16)
        .unsqueeze(1)
    )
    return output_mapping, output_reduced


ttnn.attach_golden_function(
    ttnn.moe_expert_token_remap,
    golden_function=_golden_function_moe_expert_token_remap,
)


def _preprocess_moe_routing_remap_golden_inputs(function_args, function_kwargs):
    input_tensor = function_args[0] if function_args else function_kwargs["routing_weights_tensor"]
    distributed_input = ttnn.decorators.distributed_golden_for_comparison(input_tensor)
    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    if function_args:
        function_args[0] = distributed_input
    else:
        function_kwargs["routing_weights_tensor"] = distributed_input
    return tuple(function_args), function_kwargs


def _golden_function_moe_routing_remap(
    routing_weights_tensor,
    non_zero_weight_size,
    expert_parallel_size,
    cluster_axis,
    *args,
    **kwargs,
):
    import torch

    mesh_shape = routing_weights_tensor.topology.distribution_shape
    routing_weights = routing_weights_tensor.global_value
    if routing_weights is None:
        routing_weights = ttnn.compose_mesh_value(
            shards_by_mesh_coord=routing_weights_tensor.shards,
            topology=routing_weights_tensor.topology,
        )

    non_zero_indices = torch.nonzero(routing_weights.flatten(), as_tuple=False).flatten()
    local_non_zero_size = non_zero_weight_size // expert_parallel_size

    num_devices = 1
    for dimension in mesh_shape:
        num_devices *= dimension
    member_stride = 1
    for dimension in mesh_shape[cluster_axis + 1 :]:
        member_stride *= dimension

    output_shards = {}
    for device_index, mesh_coord in enumerate(routing_weights_tensor.topology.mesh_coords):
        member_index = (device_index // member_stride) % mesh_shape[cluster_axis]
        local_start = member_index * local_non_zero_size
        local_indices = non_zero_indices[local_start : local_start + local_non_zero_size]
        output = torch.zeros_like(routing_weights)
        output.flatten()[local_indices] = routing_weights.flatten()[local_indices]
        output_shards[mesh_coord] = output

    output_shard_dims = (0,) * len(mesh_shape)
    return _distributed_collective_golden(
        output_shards,
        routing_weights_tensor.topology,
        output_shard_dims,
        compare_coords=_output_compare_coords(routing_weights_tensor, output_shards),
    )


ttnn.attach_golden_function(
    ttnn.moe_routing_remap,
    golden_function=_golden_function_moe_routing_remap,
    preprocess_golden_function_inputs=_preprocess_moe_routing_remap_golden_inputs,
)

__all__ = [
    "Topology",
    "get_usable_topology",
    "DispatchAlgorithm",
    "WorkerMode",
    "MoEActivationFunction",
    "MMSignalAggregatorMode",
]
