// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// On-device port of the host KV-pad-rotation derivations (ring_joint_sdpa_program_factory.cpp:
// build_kv_pad_q_mapping, build_ring_work_plan_impl, kv_global_tile_for_host_ring_plan, logical_nt /
// gather_valid_Ht). The trace-safe metadata path computes these in the SDPA reader from the per-chunk
// kv_actual_isl (metadata[1]) and hands them to the writer + compute via L1 (compute cannot NoC-read
// DRAM). All inputs other than kv_actual_isl are static per program. The host functions are the
// reference; the metadata==scalar bit-exact test guards against any divergence between them.

#pragma once

#include <cstdint>

#include "chunked_prefill_utils.hpp"    // chunked_kv_global_tile_for_local
#include "../../ring_id_sequencer.hpp"  // RingIdSequencer

namespace ttnn::operations::transformer::sdpa::ring_joint {

// 4 per-device Q-mapping tiles consumed by the compute kernel (kv_pad_q_*). See the host
// build_kv_pad_q_mapping for the derivation; here laid out as plain fields for the L1 handoff.
struct KvPadQMapping {
    uint32_t q_pre_wrap_start_tile;
    uint32_t q_pre_wrap_tile_count;
    uint32_t q_post_wrap_start_tile;
    uint32_t q_valid_tile_count;
};

struct RingWorkMasks {
    uint32_t active_ring_iter_mask;
    uint32_t single_valid_kv_chunk_mask;
};

inline uint32_t kv_pad_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t kv_pad_max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

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

// Mirror of host build_kv_pad_q_mapping (ring_joint_sdpa_program_factory.cpp:257). The current valid Q
// range is [kv_actual_tile_count, logical_tile_count), packed into this device's fixed Q slab; it may
// straddle one global chunk-group boundary. Host TT_FATAL invariants are validated host-side and are
// omitted here (the device path runs only after a successful host create).
inline KvPadQMapping build_kv_pad_q_mapping_device(
    uint32_t kv_actual_tile_count,
    uint32_t logical_tile_count,
    uint32_t ring_size,
    uint32_t q_local_padded_tile_count,
    uint32_t device_index) {
    const uint32_t q_chunk_group_tile_count = ring_size * q_local_padded_tile_count;
    const uint32_t first_group = kv_actual_tile_count / q_chunk_group_tile_count;
    const uint32_t last_group = (logical_tile_count - 1) / q_chunk_group_tile_count;

    // intersect_device_group: this device's Q slab within `group`, clamped to the valid range.
    uint32_t first_start = 0, first_count = 0;
    {
        const uint32_t block_start = first_group * q_chunk_group_tile_count + device_index * q_local_padded_tile_count;
        const uint32_t block_end = block_start + q_local_padded_tile_count;
        const uint32_t start = kv_pad_max_u32(kv_actual_tile_count, block_start);
        const uint32_t end = kv_pad_min_u32(logical_tile_count, block_end);
        if (end > start) {
            first_start = start;
            first_count = end - start;
        }
    }
    uint32_t second_start = 0, second_count = 0;
    if (last_group != first_group) {
        const uint32_t group = first_group + 1;
        const uint32_t block_start = group * q_chunk_group_tile_count + device_index * q_local_padded_tile_count;
        const uint32_t block_end = block_start + q_local_padded_tile_count;
        const uint32_t start = kv_pad_max_u32(kv_actual_tile_count, block_start);
        const uint32_t end = kv_pad_min_u32(logical_tile_count, block_end);
        if (end > start) {
            second_start = start;
            second_count = end - start;
        }
    }

    KvPadQMapping m;
    m.q_pre_wrap_start_tile = first_start;
    m.q_pre_wrap_tile_count = first_count;
    m.q_post_wrap_start_tile = second_start;
    m.q_valid_tile_count = first_count + second_count;
    return m;
}

// Mirror of host build_ring_work_plan_impl (ring_joint_sdpa_program_factory.cpp:192). Walks the same
// ring-id order via RingIdSequencer, marks ring iterations with non-padded spatial / joint KV work, and
// applies the same causal unbalanced skip rule. logical_nt is the only per-chunk input.
inline RingWorkMasks build_ring_work_masks_device(
    uint32_t device_index,
    uint32_t ring_size,
    uint32_t backward_writes_expected,
    uint32_t forward_writes_expected,
    uint32_t num_local_k_chunks,
    uint32_t k_chunk_tile_count,
    uint32_t kv_local_padded_Nt,
    bool kernel_chunked,
    uint32_t q_chunk_group_tile_count,
    uint32_t q_local_padded_Nt,
    uint32_t logical_nt,
    uint32_t num_joint_k_chunks,
    uint32_t joint_seq_len,
    bool kv_pad_rotation_enabled,
    bool kernel_is_causal,
    bool is_balanced) {
    RingWorkMasks plan;
    plan.active_ring_iter_mask = 0;
    plan.single_valid_kv_chunk_mask = 0;

    RingIdSequencer seq(device_index, ring_size, backward_writes_expected, forward_writes_expected);
    auto noop_sync = [](uint32_t, uint32_t) {};

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        const uint32_t ring_id = seq.get_next_ring_id(noop_sync);
        const bool joint_contributes = ring_id == ring_size - 1 && num_joint_k_chunks > 0 && joint_seq_len != 0;
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
        const uint32_t valid_kv_chunks = valid_spatial_kv_chunks + (joint_contributes ? num_joint_k_chunks : 0);
        const bool has_kv_work = (kernel_chunked && !kv_pad_rotation_enabled) || valid_spatial_kv_chunks > 0;
        const bool ring_iter_does_work =
            (has_kv_work || joint_contributes) && !(kernel_is_causal && device_index < ring_id && !is_balanced);
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
