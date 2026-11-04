# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn
from tt_py_test_utils_common.all_to_all_combine_ref import (
    gen_tensors as gen_tensors_combine,
    get_batch_cluster_idxr,
    get_cluster_dims,
    get_experts_on_device,
)


def device_mesh_iterator(mesh_shape):
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            device = m0 * mesh_shape[1] + m1
            yield m0, m1, device


def _unpack_dense_metadata_entry(entry, num_local_experts, seq):
    token_id = entry[0].item()
    k_entries = entry[1 : num_local_experts + 1]
    b = token_id // seq
    s = token_id % seq

    return k_entries, b, s


def gen_dense_metadata(batch, seq, experts, select_experts_k, mesh_shape, cluster_axis, metadata, expert_mapping):
    _, _, devices = get_cluster_dims(cluster_axis, mesh_shape)

    assert experts % devices == 0

    num_local_experts = experts // devices

    """
    struct Header {
       uint32_t token_id; // which token in source device's buffer
       uint32_t k[2]; // k+1 if not activated
       uint32_t expert_weight[2]; // bfloat16 scores
       uint32_t [3] _unused // 24 bytes padding
    }
    """

    metadata_entry_size = (
        2 * num_local_experts + 1 + 3
    )  # token id, k[experts], score[experts], 24 bytes worth of padding
    dense_metadata_buffer = (
        torch.ones([devices, batch * seq, metadata_entry_size], dtype=torch.uint32) * select_experts_k
    )

    dense_metadata_len = torch.zeros([devices], dtype=torch.uint32)
    active_token_counts = torch.zeros([experts], dtype=torch.int32)
    # 1 mapping value per token + 12 bytes padding
    dense_token_maps = torch.zeros([experts, batch * seq + 1, 4], dtype=torch.int32)

    for m0, m1, rec_d in device_mesh_iterator(mesh_shape):
        device_expert_list = get_experts_on_device(experts, expert_mapping, rec_d)

        for b in range(batch):
            for s in range(seq):
                k_entries = [select_experts_k] * num_local_experts
                for k in range(select_experts_k):
                    ek = metadata[0, b, s, k].item()
                    if ek in device_expert_list:
                        local_e_idx = device_expert_list.index(ek)
                        k_entries[local_e_idx] = k
                        global_e_idx = rec_d * num_local_experts + local_e_idx
                        dense_token_maps[global_e_idx, active_token_counts[global_e_idx], 0] = b * seq + s
                        active_token_counts[global_e_idx] += 1
                if any(map(lambda x: x != select_experts_k, k_entries)):
                    metadata_entry = torch.zeros([metadata_entry_size], dtype=torch.uint32)
                    metadata_entry[0] = b * seq + s
                    metadata_entry[1 : num_local_experts + 1] = torch.tensor(k_entries)
                    token_count = dense_metadata_len[rec_d].item()
                    dense_metadata_buffer[rec_d, token_count] = metadata_entry
                    dense_metadata_len[rec_d] = token_count + 1

    assert sum(active_token_counts) == batch * seq * select_experts_k
    return dense_metadata_buffer, dense_metadata_len, dense_token_maps, active_token_counts


def _active_token_core_split_counts_even(token_parallel_block_size, active_token_counts, token_parallel_core_dim):
    split_token_end_indexes = []
    for act in active_token_counts.tolist():
        end_index = [act // token_parallel_core_dim for _ in range(token_parallel_core_dim)]

        for rem in range(act % token_parallel_core_dim):
            end_index[rem] += 1

        split_token_end_indexes.append(list(reversed(end_index)))

    return split_token_end_indexes


def gen_dense_input_contribs(
    batch,
    seq,
    experts,
    select_experts_k,
    sparse_contribs_tensor,
    expert_mapping,
    dense_metadata_tensor,
    dense_metadata_len,
    active_token_counts,
    token_parallel_core_dim,
    mesh_shape,
    cluster_axis,
):
    num_local_experts = experts // expert_mapping.shape[-1]
    hidden_size = sparse_contribs_tensor.shape[-1]
    seq = sparse_contribs_tensor.shape[-2]

    dense_input_contribs_tensor = torch.ones([experts, batch * seq, hidden_size]).bfloat16()
    token_parallel_block_size = batch // token_parallel_core_dim
    block_counts = _active_token_core_split_counts_even(
        token_parallel_block_size, active_token_counts, token_parallel_core_dim
    )
    assert len(block_counts) == experts

    dense_contribs = 0
    for m0, m1, rec_d in device_mesh_iterator(mesh_shape):
        device_dense_idxs = [0] * num_local_experts
        device_blocked_dense_counts = [0] * num_local_experts
        for dt in range(dense_metadata_len[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, num_local_experts, seq)
            for local_e_idx, k_entry in enumerate(k_entries):
                if k_entry == select_experts_k:
                    continue

                global_e_idx = rec_d * num_local_experts + local_e_idx

                contrib = sparse_contribs_tensor[global_e_idx, b, s]
                assert not (contrib == torch.zeros([hidden_size])).all()

                dense_input_contribs_tensor[global_e_idx, device_dense_idxs[local_e_idx]] = contrib.bfloat16()

                if device_blocked_dense_counts[local_e_idx] == block_counts[global_e_idx][-1] - 1:
                    device_blocked_dense_counts[local_e_idx] = 0
                    if len(block_counts[global_e_idx]) > 1:
                        block_counts[global_e_idx].pop()
                    device_dense_idxs[local_e_idx] = (
                        token_parallel_core_dim - len(block_counts[global_e_idx])
                    ) * token_parallel_block_size

                else:
                    device_dense_idxs[local_e_idx] += 1
                    device_blocked_dense_counts[local_e_idx] += 1
                dense_contribs += 1

    assert dense_contribs == batch * seq * select_experts_k
    return dense_input_contribs_tensor


def gen_output_ref(
    batch,
    seq,
    experts,
    select_experts_k,
    expert_mapping,
    dense_input_contribs_tensor,
    dense_metadata_tensor,
    dense_metadata_len,
    active_token_counts,
    token_parallel_core_dim,
    mesh_shape,
    cluster_axis,
):
    cluster_factor, cluster_size, devices = get_cluster_dims(cluster_axis, mesh_shape)

    num_local_experts = experts // devices
    # num_cluster_experts = experts // cluster_factor
    hidden_size = dense_input_contribs_tensor.shape[-1]
    output_ref_tensor = torch.zeros(select_experts_k, batch * seq * cluster_factor, hidden_size).bfloat16()
    output_data_map = torch.zeros(output_ref_tensor.shape[:-1])

    token_parallel_block_size = batch // token_parallel_core_dim
    block_counts = _active_token_core_split_counts_even(
        token_parallel_block_size, active_token_counts, token_parallel_core_dim
    )

    batch_rep_idxr = get_batch_cluster_idxr(cluster_axis, batch)

    for m0, m1, rec_d in device_mesh_iterator(mesh_shape):
        device_dense_idxs = [0] * num_local_experts
        device_blocked_dense_counts = [0] * num_local_experts
        for dt in range(dense_metadata_len[rec_d]):
            dense_metadata_entry = dense_metadata_tensor[rec_d, dt]
            k_entries, b, s = _unpack_dense_metadata_entry(dense_metadata_entry, num_local_experts, seq)
            global_batch = batch_rep_idxr(m0, m1, b)

            for local_e_idx, k in enumerate(k_entries):
                if k == select_experts_k:
                    continue

                global_e_idx = rec_d * num_local_experts + local_e_idx

                output_ref_tensor[k, global_batch * seq + s] = dense_input_contribs_tensor[
                    global_e_idx, device_dense_idxs[local_e_idx]
                ]
                output_data_map[k, global_batch * seq + s] = 1

                if device_blocked_dense_counts[local_e_idx] == block_counts[global_e_idx][-1] - 1:
                    device_blocked_dense_counts[local_e_idx] = 0
                    if len(block_counts[global_e_idx]) > 1:
                        block_counts[global_e_idx].pop()
                    device_dense_idxs[local_e_idx] = (
                        token_parallel_core_dim - len(block_counts[global_e_idx])
                    ) * token_parallel_block_size

                else:
                    device_dense_idxs[local_e_idx] += 1
                    device_blocked_dense_counts[local_e_idx] += 1

    return output_ref_tensor, output_data_map


def gen_tensors(
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    mesh_shape,
    cluster_axis,
    devices,
    token_parallel_core_dim,
    scheme="random",
):
    _, input_sparse_contribs_tensor, expert_mapping, metadata_tensor, _, _ = gen_tensors_combine(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        cluster_axis,
        devices,
        scheme,
    )

    dense_metadata_tensor, dense_metadata_len, dense_token_maps, active_token_counts = gen_dense_metadata(
        batch, seq, experts, select_experts_k, mesh_shape, cluster_axis, metadata_tensor, expert_mapping
    )
    dense_contribs_tensor = gen_dense_input_contribs(
        batch,
        seq,
        experts,
        select_experts_k,
        input_sparse_contribs_tensor,
        expert_mapping,
        dense_metadata_tensor,
        dense_metadata_len,
        active_token_counts,
        token_parallel_core_dim,
        mesh_shape,
        cluster_axis,
    )
    output_ref, output_data_map = gen_output_ref(
        batch,
        seq,
        experts,
        select_experts_k,
        expert_mapping,
        dense_contribs_tensor,
        dense_metadata_tensor,
        dense_metadata_len,
        active_token_counts,
        token_parallel_core_dim,
        mesh_shape,
        cluster_axis,
    )

    return (
        dense_metadata_tensor,
        dense_token_maps,
        active_token_counts,
        dense_contribs_tensor,
        output_ref,
        output_data_map,
    )


FABRIC_PACKET_SIZE_BYTES = 4096

ELM_SIZE_BYTES = 2


def effective_data_parallel_core_dim(hidden_size, data_parallel_core_dim):
    """Data-parallel split the host layout uses: a token row is cut into at most one fabric packet per core, so a
    hidden size below data_parallel_core_dim packets gets fewer cores (bf16 4096 = two packets = 2 cores). The op
    must be called with this value and a worker grid sized by it, or its split disagrees with the sharded input."""
    hidden_bytes = hidden_size * ELM_SIZE_BYTES
    return min(data_parallel_core_dim, math.ceil(hidden_bytes / FABRIC_PACKET_SIZE_BYTES))


def get_sharded_dense_input(
    dense_contribs_tensor, core_range, token_parallel_core_dim, data_parallel_core_dim, device, cluster_axis
):
    data_parallel_core_dim = effective_data_parallel_core_dim(dense_contribs_tensor.shape[-1], data_parallel_core_dim)

    tt_dense_contribs = ttnn.from_torch(
        dense_contribs_tensor,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )

    shape0 = dense_contribs_tensor.shape

    local_experts = shape0[0] // math.prod(tuple(device.shape))
    num_tokens = shape0[1]
    hidden = shape0[2]

    assert shape0[2] % data_parallel_core_dim == 0

    # want [data_parallel_core_dim, token_parallel_core_dim, local_experts, num_tokens//token_parallel_core_dim, hidden//data_parallel_core_dim]

    shape1 = [
        local_experts,
        token_parallel_core_dim,
        num_tokens // token_parallel_core_dim,
        data_parallel_core_dim,
        hidden // data_parallel_core_dim,
    ]

    tt_dense_contribs = ttnn.reshape(tt_dense_contribs, shape1)
    tt_dense_contribs = ttnn.permute(tt_dense_contribs, [3, 1, 0, 2, 4])

    shard_shape = (local_experts * num_tokens // token_parallel_core_dim, hidden // data_parallel_core_dim)
    shard_spec = ttnn.ShardSpec(core_range, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    return tt_dense_contribs, mem_config


def _get_tt_sharded_dense_input(
    dense_contribs_tensor, core_range, token_parallel_core_dim, data_parallel_core_dim, device, cluster_axis
):
    tt_dense_contribs, mem_config = get_sharded_dense_input(
        dense_contribs_tensor, core_range, token_parallel_core_dim, data_parallel_core_dim, device, cluster_axis
    )
    return ttnn.interleaved_to_sharded(tt_dense_contribs, mem_config)


def _get_tt_dense_metadata(dense_metadata_tensor, mesh_device, memory_config=None):
    shape = dense_metadata_tensor.shape
    dense_metadata_tensor = dense_metadata_tensor.reshape([shape[0], shape[-2] * shape[-1]])
    return ttnn.from_torch(
        dense_metadata_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _get_tt_dense_token_maps(dense_token_maps_tensor, mesh_device):
    tt_dense_token_maps = ttnn.from_torch(
        dense_token_maps_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    shape = tt_dense_token_maps.shape
    return ttnn.reshape(tt_dense_token_maps, [shape[0], shape[1] * shape[2]])
