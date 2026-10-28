# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

# The TTNN CCL ops (all_gather, reduce_scatter, all_reduce, ...) and their golden functions were removed
# for agent evaluation (TODO(nuked-op ccl)). The fabric topology types stay.
Topology = ttnn._ttnn.operations.ccl.Topology

get_usable_topology = ttnn._ttnn.operations.ccl.get_usable_topology


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
    # cluster_axis indexes the physical device mesh, so use the physical mesh shape (the
    # tensor's distribution shape may differ, e.g. a default replicated topology is 1D
    # even on a 2D device mesh).
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
    import math

    import torch

    if _ttnn_golden_mesh_shape is None:
        return None

    non_zero_indices = torch.nonzero(routing_weights_tensor.flatten(), as_tuple=False).flatten()
    local_non_zero_size = non_zero_weight_size // expert_parallel_size

    if cluster_axis < 0:
        cluster_axis += len(_ttnn_golden_mesh_shape)
    if cluster_axis < 0 or cluster_axis >= len(_ttnn_golden_mesh_shape):
        raise ValueError(f"Collective axis {cluster_axis} is invalid for mesh shape {_ttnn_golden_mesh_shape}")
    member_stride = 1
    for dimension in _ttnn_golden_mesh_shape[cluster_axis + 1 :]:
        member_stride *= dimension

    per_device_outputs = []
    for device_index in range(math.prod(_ttnn_golden_mesh_shape)):
        member_index = (device_index // member_stride) % _ttnn_golden_mesh_shape[cluster_axis]
        local_start = member_index * local_non_zero_size
        local_indices = non_zero_indices[local_start : local_start + local_non_zero_size]
        output = torch.zeros_like(routing_weights_tensor)
        output.flatten()[local_indices] = routing_weights_tensor.flatten()[local_indices]
        per_device_outputs.append(output)
    return torch.cat(per_device_outputs, dim=0)


ttnn.attach_golden_function(
    ttnn.moe_routing_remap,
    golden_function=_golden_function_moe_routing_remap,
    preprocess_golden_function_inputs=_preprocess_moe_routing_remap_golden_inputs,
)

__all__ = [
    "Topology",
    "get_usable_topology",
]
