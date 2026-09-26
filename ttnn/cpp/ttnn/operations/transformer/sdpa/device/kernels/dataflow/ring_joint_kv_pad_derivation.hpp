// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runtime ring-work derivation for the metadata path. Keep the work masks in sync
// with build_ring_work_plan_impl in ring_joint_sdpa_program_factory.cpp.

#pragma once

#include <cstdint>

#include "chunked_prefill_utils.hpp"                                              // chunked_kv_global_tile_for_local
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_id_sequencer.hpp"  // RingIdSequencer
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

struct RingWorkMasks {
    uint32_t active_ring_iter_mask;
    uint32_t single_valid_kv_chunk_mask;
};

// logical_n = kv_actual_isl + chunk_global; logical_nt = div_up(logical_n, TILE_HEIGHT).
inline uint32_t compute_logical_nt(uint32_t kv_actual_isl, uint32_t chunk_global, uint32_t tile_height) {
    const uint32_t logical_n = kv_actual_isl + chunk_global;
    return (logical_n + tile_height - 1) / tile_height;
}

// Mirror of host kv_global_tile_for_host_ring_plan (ring_joint_sdpa_program_factory.cpp:174).
inline uint32_t kv_global_tile_for_ring_plan(
    bool is_chunked,
    uint32_t ring_id,
    uint32_t local_tile_start,
    uint32_t q_chunk_group_tile_count,
    uint32_t q_local_padded_tile_count,
    uint32_t kv_local_padded_tile_count) {
    if (is_chunked) {
        return chunked_kv_global_tile_for_local(
            ring_id, local_tile_start, q_chunk_group_tile_count, q_local_padded_tile_count);
    }
    return ring_id * kv_local_padded_tile_count + local_tile_start;
}

// Inputs to build_ring_work_masks_device. The sharded-joint trio defaults to the replicated-joint
// behaviour, where every selected joint chunk counts.
struct RingWorkMasksParams {
    uint32_t transport_rank;
    uint32_t ring_size;
    // Transport-to-tensor rank mapping (full-mesh snake ring); see tensor_rank_from_transport_rank.
    uint32_t mesh_rows;
    uint32_t mesh_cols;
    ttnn::ccl::snake_ring::Orientation snake_orientation;
    uint32_t backward_writes_expected;
    uint32_t forward_writes_expected;
    uint32_t num_local_k_chunks;
    uint32_t k_chunk_tile_count;
    uint32_t kv_local_padded_Nt;
    bool kernel_chunked;
    uint32_t q_chunk_group_tile_count;
    uint32_t q_local_padded_Nt;
    uint32_t logical_nt;
    uint32_t num_joint_k_chunks;
    uint32_t joint_seq_len;
    bool kv_pad_rotation_enabled;
    bool kernel_is_causal;
    bool is_balanced;
    // Sharded-joint tail (mirrors the host's valid_joint_kv_chunks loop).
    bool joint_is_sharded = false;
    uint32_t joint_local_padded_Nt = 0;
    uint32_t logical_lt = 0;
};

// Mirror of host build_ring_work_plan_impl (ring_joint_sdpa_program_factory.cpp:192). Walks the same
// ring-id order via RingIdSequencer, marks ring iterations with non-padded spatial / joint KV work, and
// applies the same causal unbalanced skip rule. logical_nt is the only per-chunk input.
template <bool FullMesh>
inline RingWorkMasks build_ring_work_masks_device(const RingWorkMasksParams& p) {
    const uint32_t transport_rank = p.transport_rank;
    const uint32_t ring_size = p.ring_size;
    const uint32_t mesh_rows = p.mesh_rows;
    const uint32_t mesh_cols = p.mesh_cols;
    const ttnn::ccl::snake_ring::Orientation snake_orientation = p.snake_orientation;
    const uint32_t backward_writes_expected = p.backward_writes_expected;
    const uint32_t forward_writes_expected = p.forward_writes_expected;
    const uint32_t num_local_k_chunks = p.num_local_k_chunks;
    const uint32_t k_chunk_tile_count = p.k_chunk_tile_count;
    const uint32_t kv_local_padded_Nt = p.kv_local_padded_Nt;
    const bool kernel_chunked = p.kernel_chunked;
    const uint32_t q_chunk_group_tile_count = p.q_chunk_group_tile_count;
    const uint32_t q_local_padded_Nt = p.q_local_padded_Nt;
    const uint32_t logical_nt = p.logical_nt;
    const uint32_t num_joint_k_chunks = p.num_joint_k_chunks;
    const uint32_t joint_seq_len = p.joint_seq_len;
    const bool kv_pad_rotation_enabled = p.kv_pad_rotation_enabled;
    const bool kernel_is_causal = p.kernel_is_causal;
    const bool is_balanced = p.is_balanced;
    const bool joint_is_sharded = p.joint_is_sharded;
    const uint32_t joint_local_padded_Nt = p.joint_local_padded_Nt;
    const uint32_t logical_lt = p.logical_lt;

    RingWorkMasks plan;
    plan.active_ring_iter_mask = 0;
    plan.single_valid_kv_chunk_mask = 0;

    RingIdSequencer seq(transport_rank, ring_size, backward_writes_expected, forward_writes_expected);
    const uint32_t tensor_rank = ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<FullMesh>(
        transport_rank, mesh_rows, mesh_cols, snake_orientation);
    auto noop_sync = [](uint32_t, uint32_t) {};

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        const uint32_t ring_id = ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<FullMesh>(
            seq.get_next_ring_id(noop_sync), mesh_rows, mesh_cols, snake_orientation);
        const bool has_joint_work = num_joint_k_chunks > 0 && joint_seq_len != 0;
        const bool joint_iter_selected = has_joint_work && (joint_is_sharded || ring_id == ring_size - 1);
        uint32_t valid_joint_kv_chunks = 0;
        if (joint_iter_selected) {
            if (joint_is_sharded) {
                for (uint32_t k = 0; k < num_joint_k_chunks; ++k) {
                    const uint32_t joint_global_start_tile = ring_id * joint_local_padded_Nt + k * k_chunk_tile_count;
                    if (joint_global_start_tile < logical_lt) {
                        valid_joint_kv_chunks++;
                    }
                }
            } else {
                valid_joint_kv_chunks = num_joint_k_chunks;
            }
        }
        const bool joint_contributes = valid_joint_kv_chunks > 0;
        uint32_t valid_spatial_kv_chunks = 0;
        for (uint32_t k_chunk = 0; k_chunk < num_local_k_chunks; ++k_chunk) {
            const uint32_t local_tile_start = k_chunk * k_chunk_tile_count;
            if (local_tile_start >= kv_local_padded_Nt) {
                continue;
            }
            if (kv_global_tile_for_ring_plan(
                    kernel_chunked,
                    ring_id,
                    local_tile_start,
                    q_chunk_group_tile_count,
                    q_local_padded_Nt,
                    kv_local_padded_Nt) < logical_nt) {
                valid_spatial_kv_chunks++;
            }
        }
        const uint32_t valid_kv_chunks = valid_spatial_kv_chunks + valid_joint_kv_chunks;
        const bool has_kv_work = (kernel_chunked && !kv_pad_rotation_enabled) || valid_spatial_kv_chunks > 0;
        const bool ring_iter_does_work =
            (has_kv_work || joint_contributes) && !(kernel_is_causal && tensor_rank < ring_id && !is_balanced);
        if (ring_iter_does_work) {
            plan.active_ring_iter_mask |= (1u << ring_iter);
        }
        if (valid_kv_chunks <= 1) {
            plan.single_valid_kv_chunk_mask |= (1u << ring_iter);
        }
    }
    return plan;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
