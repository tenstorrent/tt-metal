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
    """Convert a collective mesh input into per-device Torch shards.
    Adds mesh shape and shard-mode metadata for global golden preprocessing.
    """

    input_tensor = function_args[0] if function_args else function_kwargs["input_tensor"]
    input_tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(input_tensor)]
    mesh_shape = tuple(input_tensor.device().shape)

    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    if function_args:
        function_args[0] = input_tensors
    else:
        function_kwargs["input_tensor"] = input_tensors
    function_kwargs["_ttnn_golden_mesh_shape"] = mesh_shape
    function_kwargs["_ttnn_global_golden_mesh_shards"] = True
    return tuple(function_args), function_kwargs


def _get_first_collective_group(input_tensors, mesh_shape, cluster_axis):
    """Select the first collective group along the requested mesh axis.
    Returns every input when no cluster axis restricts the group.
    """

    if cluster_axis is None:
        return input_tensors

    stride = 1
    for dimension in mesh_shape[cluster_axis + 1 :]:
        stride *= dimension
    return [input_tensors[index * stride] for index in range(mesh_shape[cluster_axis])]


def _golden_function_all_broadcast(
    input_tensor,
    *args,
    cluster_axis=None,
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    if _ttnn_golden_mesh_shape is None:
        return None

    # Each result broadcasts one rank's payload across the first cluster group.
    return _get_first_collective_group(input_tensor, _ttnn_golden_mesh_shape, cluster_axis)


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
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _ttnn_golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensor, _ttnn_golden_mesh_shape, cluster_axis)
    # The first rank receives every rank's shard concatenated along the requested dimension.
    return torch.cat(input_group, dim=dim)


ttnn.attach_golden_function(
    ttnn.all_gather,
    golden_function=_golden_function_all_gather,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_all_reduce(
    input_tensor,
    *args,
    cluster_axis=None,
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _ttnn_golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensor, _ttnn_golden_mesh_shape, cluster_axis)
    # Stable all-reduce always performs a sum and replicates it to the group.
    return torch.stack(input_group).sum(dim=0)


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
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _ttnn_golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensor, _ttnn_golden_mesh_shape, cluster_axis)
    reduced = torch.stack(input_group).sum(dim=0)
    # The first rank receives the first equal chunk of the reduced tensor.
    return torch.chunk(reduced, len(input_group), dim=dim)[0]


ttnn.attach_golden_function(
    ttnn.reduce_scatter,
    golden_function=_golden_function_reduce_scatter,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_mesh_partition(
    input_tensor,
    dim,
    cluster_axis=None,
    *args,
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _ttnn_golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensor, _ttnn_golden_mesh_shape, cluster_axis)
    # The first rank keeps the first equal partition of its local input.
    return torch.chunk(input_group[0], len(input_group), dim=dim)[0]


ttnn.attach_golden_function(
    ttnn.mesh_partition,
    golden_function=_golden_function_mesh_partition,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _mesh_coordinate_to_index(coordinate, mesh_shape):
    """Convert a row-major mesh coordinate to its flat device index.
    Uses each mesh dimension as the radix for the corresponding coordinate.
    """

    index = 0
    for value, dimension in zip(coordinate, mesh_shape):
        index = index * dimension + int(value)
    return index


def _golden_function_point_to_point(
    input_tensor,
    sender_coord,
    receiver_coord,
    *args,
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    if _ttnn_golden_mesh_shape is None:
        return None

    sender_index = _mesh_coordinate_to_index(sender_coord, _ttnn_golden_mesh_shape)
    receiver_index = _mesh_coordinate_to_index(receiver_coord, _ttnn_golden_mesh_shape)
    output = input_tensor[sender_index].clone()
    # Point-to-point initializes only the receiver shard of its fresh output tensor.
    output._ttnn_mesh_index = receiver_index
    return output


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
    """Convert reduce-to-root state tensors into per-device Torch shards.
    Preserves their shared mesh shape for local and global golden execution.
    """

    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    input_names = ("input_tensor_l", "input_tensor_s", "input_tensor_m")
    mesh_shape = None

    for index, input_name in enumerate(input_names):
        input_tensor = function_args[index] if index < len(function_args) else function_kwargs[input_name]
        if mesh_shape is None:
            mesh_shape = tuple(input_tensor.device().shape)
        input_tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(input_tensor)]
        if index < len(function_args):
            function_args[index] = input_tensors
        else:
            function_kwargs[input_name] = input_tensors

    function_kwargs["_ttnn_golden_mesh_shape"] = mesh_shape
    function_kwargs["_ttnn_global_golden_mesh_shards"] = True
    return tuple(function_args), function_kwargs


def _golden_function_reduce_to_root(
    input_tensor_l,
    input_tensor_s,
    input_tensor_m,
    root_coord,
    *args,
    scale_fp32=1.0,
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _ttnn_golden_mesh_shape is None:
        return None
    if len(input_tensor_l) != 4 or len(input_tensor_s) != 4 or len(input_tensor_m) != 4:
        raise ValueError("reduce_to_root golden requires the operation's fixed four-device topology")

    tile_width = 32
    num_cores = input_tensor_s[0].shape[-1] // tile_width
    if num_cores == 0 or input_tensor_s[0].shape[-1] % tile_width != 0:
        raise ValueError("reduce_to_root golden requires tile-aligned S state")

    states = []
    for tensor_l, tensor_s, tensor_m in zip(input_tensor_l, input_tensor_s, input_tensor_m):
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

    output_l = tensor_l.reshape(input_tensor_l[0].shape)
    output_s = tensor_s.reshape(input_tensor_s[0].shape)
    output_m = tensor_m.reshape(input_tensor_m[0].shape)
    root_index = _mesh_coordinate_to_index(root_coord, _ttnn_golden_mesh_shape)
    for output in (output_l, output_s, output_m):
        output._ttnn_mesh_index = root_index
    return output_l, output_s, output_m


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
    """Convert the replicated routing weights and retain their device mesh shape."""

    input_tensor = function_args[0] if function_args else function_kwargs["routing_weights_tensor"]
    golden_args, golden_kwargs = ttnn.decorators.default_preprocess_golden_function_inputs(
        function_args, function_kwargs
    )
    golden_kwargs["_ttnn_golden_mesh_shape"] = tuple(input_tensor.device().shape)
    return golden_args, golden_kwargs


def _golden_function_moe_routing_remap(
    routing_weights_tensor,
    non_zero_weight_size,
    expert_parallel_size,
    cluster_axis,
    *args,
    _ttnn_golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _ttnn_golden_mesh_shape is None:
        return None

    non_zero_indices = torch.nonzero(routing_weights_tensor.flatten(), as_tuple=False).flatten()
    local_non_zero_size = non_zero_weight_size // expert_parallel_size

    num_devices = 1
    for dimension in _ttnn_golden_mesh_shape:
        num_devices *= dimension
    member_stride = 1
    for dimension in _ttnn_golden_mesh_shape[cluster_axis + 1 :]:
        member_stride *= dimension

    per_device_outputs = []
    for device_index in range(num_devices):
        member_index = (device_index // member_stride) % _ttnn_golden_mesh_shape[cluster_axis]
        local_start = member_index * local_non_zero_size
        local_indices = non_zero_indices[local_start : local_start + local_non_zero_size]
        output = torch.zeros_like(routing_weights_tensor)
        output.flatten()[local_indices] = routing_weights_tensor.flatten()[local_indices]
        per_device_outputs.append(output)
    # Comparison composes the mesh output by concatenating its per-device [1, experts] rows.
    return torch.cat(per_device_outputs, dim=0)


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
