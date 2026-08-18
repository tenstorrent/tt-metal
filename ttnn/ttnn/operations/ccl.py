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
    input_tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(input_tensor)]
    mesh_shape = tuple(input_tensor.device().shape)

    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    if function_args:
        function_args[0] = input_tensors
    else:
        function_kwargs["input_tensor"] = input_tensors
    function_kwargs["_golden_mesh_shape"] = mesh_shape
    return tuple(function_args), function_kwargs


def _get_first_collective_group(input_tensors, mesh_shape, cluster_axis):
    if cluster_axis is None:
        return input_tensors

    stride = 1
    for dimension in mesh_shape[cluster_axis + 1 :]:
        stride *= dimension
    return [input_tensors[index * stride] for index in range(mesh_shape[cluster_axis])]


def _golden_function_all_broadcast(
    input_tensors,
    *args,
    cluster_axis=None,
    _golden_mesh_shape=None,
    **kwargs,
):
    if _golden_mesh_shape is None:
        return None

    # Each result broadcasts one rank's payload across the first cluster group.
    return _get_first_collective_group(input_tensors, _golden_mesh_shape, cluster_axis)


ttnn.attach_golden_function(
    ttnn.all_broadcast,
    golden_function=_golden_function_all_broadcast,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_all_gather(
    input_tensors,
    dim,
    *args,
    cluster_axis=None,
    _golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensors, _golden_mesh_shape, cluster_axis)
    # The first rank receives every rank's shard concatenated along the requested dimension.
    return torch.cat(input_group, dim=dim)


ttnn.attach_golden_function(
    ttnn.all_gather,
    golden_function=_golden_function_all_gather,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_all_reduce(
    input_tensors,
    *args,
    cluster_axis=None,
    _golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensors, _golden_mesh_shape, cluster_axis)
    # Stable all-reduce always performs a sum and replicates it to the group.
    return torch.stack(input_group).sum(dim=0)


ttnn.attach_golden_function(
    ttnn.all_reduce,
    golden_function=_golden_function_all_reduce,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_reduce_scatter(
    input_tensors,
    dim,
    *args,
    cluster_axis=None,
    _golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensors, _golden_mesh_shape, cluster_axis)
    reduced = torch.stack(input_group).sum(dim=0)
    # The first rank receives the first equal chunk of the reduced tensor.
    return torch.chunk(reduced, len(input_group), dim=dim)[0]


ttnn.attach_golden_function(
    ttnn.reduce_scatter,
    golden_function=_golden_function_reduce_scatter,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _golden_function_mesh_partition(
    input_tensors,
    dim,
    cluster_axis=None,
    *args,
    _golden_mesh_shape=None,
    **kwargs,
):
    import torch

    if _golden_mesh_shape is None:
        return None
    input_group = _get_first_collective_group(input_tensors, _golden_mesh_shape, cluster_axis)
    # The first rank keeps the first equal partition of its local input.
    return torch.chunk(input_group[0], len(input_group), dim=dim)[0]


ttnn.attach_golden_function(
    ttnn.mesh_partition,
    golden_function=_golden_function_mesh_partition,
    preprocess_golden_function_inputs=_preprocess_collective_golden_inputs,
)


def _mesh_coordinate_to_index(coordinate, mesh_shape):
    index = 0
    for value, dimension in zip(coordinate, mesh_shape):
        index = index * dimension + int(value)
    return index


def _golden_function_point_to_point(
    input_tensors,
    sender_coord,
    receiver_coord,
    *args,
    _golden_mesh_shape=None,
    **kwargs,
):
    if _golden_mesh_shape is None:
        return None

    sender_index = _mesh_coordinate_to_index(sender_coord, _golden_mesh_shape)
    receiver_index = _mesh_coordinate_to_index(receiver_coord, _golden_mesh_shape)
    # Comparison observes rank zero, replacing it only when rank zero is the receiver.
    return input_tensors[sender_index] if receiver_index == 0 else input_tensors[0]


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

    # Metadata is all-gathered while unspecified placeholder token rows are normalized to zero.
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
    num_devices = expert_mapping.shape[-1]
    if expert_metadata_tensor.shape[0] == num_devices:
        expert_metadata = expert_metadata_tensor[0].long()
    else:
        expert_metadata = expert_metadata_tensor[:, 0].long()

    batch, sequence, selected_experts = expert_metadata.shape
    output = torch.zeros(
        (selected_experts, batch, sequence, input_tensor.shape[-1]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # Reconstruct each sparse expert contribution in its original top-k slot.
    for batch_index in range(batch):
        for sequence_index in range(sequence):
            used_devices = set()
            for topk_index in range(selected_experts):
                expert_index = int(expert_metadata[batch_index, sequence_index, topk_index])
                device_index = int(torch.nonzero(expert_mapping[expert_index], as_tuple=False)[0])
                if local_reduce:
                    if device_index in used_devices:
                        continue
                    used_devices.add(device_index)
                    contribution_index = device_index
                else:
                    contribution_index = expert_index
                output[topk_index, batch_index, sequence_index] = input_tensor[
                    contribution_index, batch_index, sequence_index
                ]

    return output


ttnn.attach_golden_function(ttnn.all_to_all_combine, golden_function=_golden_function_all_to_all_combine)


def _golden_function_reduce_to_root(
    input_tensor_l,
    input_tensor_s,
    input_tensor_m,
    root_coord,
    *args,
    **kwargs,
):
    # Non-root outputs are implementation-owned, so retain the three host state tensors.
    return input_tensor_l, input_tensor_s, input_tensor_m


ttnn.attach_golden_function(ttnn.reduce_to_root, golden_function=_golden_function_reduce_to_root)


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


def _golden_function_moe_routing_remap(
    routing_weights_tensor,
    non_zero_weight_size,
    expert_parallel_size,
    cluster_axis,
    *args,
    **kwargs,
):
    import torch

    # Host comparison represents the first cluster member's contiguous share of non-zero weights.
    output = torch.zeros_like(routing_weights_tensor)
    non_zero_indices = torch.nonzero(routing_weights_tensor.flatten(), as_tuple=False).flatten()
    local_non_zero_size = non_zero_weight_size // expert_parallel_size
    local_indices = non_zero_indices[:local_non_zero_size]
    output.flatten()[local_indices] = routing_weights_tensor.flatten()[local_indices]
    return output


ttnn.attach_golden_function(ttnn.moe_routing_remap, golden_function=_golden_function_moe_routing_remap)

__all__ = [
    "Topology",
    "get_usable_topology",
    "DispatchAlgorithm",
    "WorkerMode",
    "MoEActivationFunction",
    "MMSignalAggregatorMode",
]
