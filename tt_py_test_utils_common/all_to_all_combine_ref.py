# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
from tt_py_test_utils_common.all_to_all_dispatch import (
    gen_expert_mapping,
    gen_tokens,
    get_expert_indices,
    get_metadata_tensor,
    get_output_tensor as get_sparse_tokens,
)


def get_experts_on_device(num_experts, expert_mapping, device):
    return [e for e in range(num_experts) if expert_mapping[0, 0, e, device] == 1]


def get_cluster_dims(replication_axis, mesh_shape):
    if replication_axis == 1:
        replication_dim = mesh_shape[0]
        replication_group = mesh_shape[1]
    elif replication_axis == 0:
        replication_dim = mesh_shape[1]
        replication_group = mesh_shape[0]
    else:
        assert replication_axis == -1
        replication_dim = 1
        replication_group = mesh_shape[0] * mesh_shape[1]

    return replication_dim, replication_group, mesh_shape[0] * mesh_shape[1]


def get_batch_cluster_idxr(replication_axis, batch):
    def _idxr(m0, m1, b):
        if replication_axis == 0:
            return m1 * batch + b
        elif replication_axis == 1:
            return m0 * batch + b
        else:
            return b

    return _idxr


def get_input_sparse_contribs(
    sparse_tokens, expert_indices, expert_mapping, mesh_shape, axis, apply_fake_expert=True, local_reduce=False
):
    # sparse tokens is [devices, batch, seq, hidden_size]
    # note, in the actual op batch*=replication_dim but the reference `sparse_tokens` is not doing that here
    # desired expert contributions tensor is [experts[/devices], batch*replicate_dim, seq, hidden_size]
    # we'll multiply the tokens by the index of their assigned expert to mock expert application.

    batch = expert_indices.shape[0]
    devices = expert_mapping.shape[-1]
    experts = expert_mapping.shape[-2]
    hidden_size = sparse_tokens.shape[-1]
    selected_experts_k = expert_indices.shape[-1]
    seq = sparse_tokens.shape[-2]

    assert experts % devices == 0
    experts_per_device = experts // devices

    if local_reduce:
        expert_dim = devices
        expert_idxr = lambda d, _: d

    else:
        expert_dim = experts
        expert_idxr = lambda d, local_idx: d * experts_per_device + local_idx

    input_contribs_tensor = torch.zeros([expert_dim, batch, seq, hidden_size])

    token_expert_count = 0
    for d in range(devices):
        experts_on_device = get_experts_on_device(experts, expert_mapping, d)
        assert len(experts_on_device) == experts_per_device
        for b in range(batch):
            for k in range(selected_experts_k):
                for s in range(seq):
                    expert_idx = expert_indices[b, 0, s, k].item()
                    if expert_idx not in experts_on_device:
                        continue

                    local_expert_idx = expert_idxr(d, experts_on_device.index(expert_idx))

                    # multiply by expert index to mock application of expert

                    if apply_fake_expert:
                        contrib = sparse_tokens[d, b, s, :] * (-1 if expert_idx == 0 else expert_idx)
                    else:
                        contrib = sparse_tokens[d, b, s, :]
                    input_contribs_tensor[local_expert_idx, b, s, :] += contrib

                    token_expert_count += 1

    assert token_expert_count == batch * seq * selected_experts_k
    return input_contribs_tensor


def get_output_combined_contribs(
    sparse_contribs, expert_indices, expert_mapping, mesh_shape, replication_axis, local_reduce=False
):
    # sparse_contribs is [E[/devices], b, seq, hidden]
    # output recalled contribs is [K, batch * replicate_dim [/devices], seq, hidden]
    batch = expert_indices.shape[0]
    experts = expert_mapping.shape[-2]
    selected_experts_k = expert_indices.shape[-1]
    hidden = sparse_contribs.shape[-1]
    seq = sparse_contribs.shape[-2]

    replication_dim, replication_group, devices = get_cluster_dims(replication_axis, mesh_shape)

    assert experts % devices == 0
    experts_per_device = experts // devices

    batch_rep_idxr = get_batch_cluster_idxr(replication_axis, batch)

    if local_reduce:
        local_contrib_idx_func = lambda d, _: d
    else:
        local_contrib_idx_func = lambda d, local_idx: d * experts_per_device + local_idx

    output_combined_contribs_tensor = torch.zeros(selected_experts_k, batch * replication_dim, seq, hidden).bfloat16()
    real_data_map = torch.zeros(output_combined_contribs_tensor.shape[:-1])

    total_token_expert_count = 0
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            d = m0 * mesh_shape[1] + m1
            device_expert_list = get_experts_on_device(experts, expert_mapping, d)

            for b in range(batch):
                for s in range(seq):
                    token_experts = expert_indices[b, 0, s, :].tolist()
                    for eg in device_expert_list:
                        if eg in token_experts:
                            k = token_experts.index(eg)
                        else:
                            continue

                        axis_batch_idx = batch_rep_idxr(m0, m1, b)
                        local_contrib_idx = local_contrib_idx_func(d, device_expert_list.index(eg))

                        sc = sparse_contribs[local_contrib_idx, b, s, :]
                        output_combined_contribs_tensor[k, axis_batch_idx, s, :] = sc

                        real_data_map[k, axis_batch_idx, s] = 1
                        total_token_expert_count += 1

                        if local_reduce:
                            break
    # assert total_token_expert_count == batch * (devices if local_reduce else selected_experts_k) * seq
    return output_combined_contribs_tensor, real_data_map


def gen_tensors(
    batch,
    experts,
    selected_experts_k,
    hidden_size,
    seq,
    mesh_shape,
    replication_axis,
    devices,
    scheme="random",
    local_reduce=False,
):
    torch.manual_seed(20)
    random.seed(20)
    # create input tokens
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, seq, mesh_shape, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)

    sparse_dispatched_tokens = get_sparse_tokens(input_tokens, expert_indices, expert_mapping, seq, mesh_shape)
    input_sparse_contribs_tensor = get_input_sparse_contribs(
        sparse_dispatched_tokens,
        expert_indices,
        expert_mapping,
        mesh_shape,
        replication_axis,
        local_reduce=local_reduce,
    )

    output_tensor, data_map = get_output_combined_contribs(
        input_sparse_contribs_tensor,
        expert_indices,
        expert_mapping,
        mesh_shape,
        replication_axis,
        local_reduce=local_reduce,
    )

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    # create expert indices
    return (
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        data_map,
    )
