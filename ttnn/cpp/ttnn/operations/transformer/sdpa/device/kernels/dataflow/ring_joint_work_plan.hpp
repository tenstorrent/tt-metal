// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single source of the ring_joint_sdpa / ring_mla per-chunk work plan, derived on-device: the ring-work
// masks, the KV-pad valid-Q mapping, logical_nt, and the ring global-tile mapping. The SDPA reader, writer,
// and compute kernels call these (the all-gather workers have their own gather-extent derivation and do not);
// the host factory computes none of them and keeps only validate_kv_pad_q_mapping (the scalar-path invariant
// asserts). On the trace-safe metadata path the SDPA reader computes the plan from the per-chunk kv_actual_isl
// (metadata[1]) and hands the compute-needed subset to the compute kernel via L1 (compute cannot NoC-read
// DRAM); the writer re-derives its own masks. Every input other than kv_actual_isl is static per program.

#pragma once

#include <cstdint>

#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"  // chunked_kv_global_tile_for_local
#include "ttnn/operations/transformer/sdpa/device/ring_id_sequencer.hpp"                       // RingIdSequencer

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Per-device valid-Q mapping (kv_pad_q_*) built by build_kv_pad_q_mapping_device below and consumed by
// the compute kernel; plain fields for the L1 handoff.
struct KvPadQMapping {
    uint32_t q_pre_wrap_start_tile = 0;
    uint32_t q_pre_wrap_tile_count = 0;
    uint32_t q_post_wrap_start_tile = 0;
    uint32_t q_valid_tile_count = 0;
};

struct RingWorkMasks {
    uint32_t active_ring_iter_mask = 0;
    uint32_t single_valid_kv_chunk_mask = 0;
};

// Element layout of the reader->compute work-plan handoff (cb_kv_pad_derived, tile 0): the SDPA reader packs
// these scalars via CoreLocalMem, compute reads them back with read_tile_value(cb, 0, slot). Index both sides
// through these names so a reorder can't silently desync the two kernels.
enum RingJointWorkPlanSlot : uint32_t {
    kDerivedLogicalNt = 0,
    kDerivedQPreWrapStart = 1,
    kDerivedQPreWrapCount = 2,
    kDerivedQPostWrapStart = 3,
    kDerivedQValidCount = 4,
    kDerivedActiveRingIterMask = 5,
    kNumWorkPlanSlots = 6,
};

inline uint32_t kv_pad_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline uint32_t kv_pad_max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

// logical_n = kv_actual_isl + chunk_global; logical_nt = div_up(logical_n, TILE_HEIGHT).
inline uint32_t compute_logical_nt(uint32_t kv_actual_isl, uint32_t chunk_global, uint32_t tile_height) {
    const uint32_t logical_n = kv_actual_isl + chunk_global;
    return (logical_n + tile_height - 1) / tile_height;
}

// Global KV tile index for a given (ring_id, local tile start) under the ring work-split -- chunked-prefill
// block-cyclic or the plain non-chunked ring_id * kv_local_padded stride.
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

// Per-device Q-mapping: the current valid Q range [kv_actual_tile_count, logical_tile_count), packed into
// this device's fixed Q slab; it may straddle one global chunk-group boundary. Host-only invariants are
// omitted here: the scalar path asserts them in validate_kv_pad_q_mapping (factory); the metadata path trusts
// the device metadata contract (its per-chunk values are never host-validated, by design).
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

// Static-per-program inputs to build_ring_work_masks_device -- everything except the five values that vary
// per device / per chunk (device_index, ring_size, backward/forward_writes_expected, logical_nt). All fields
// are compile-time kernel constants, so this is passed as a non-type template parameter (Cfg) rather than a
// runtime value: the reader/writer build one `constexpr RingWorkConfig` and the mask derivation's per-iter
// branches (causal skip, chunked/rotation gates) resolve at compile time. Literal aggregate => valid C++20 NTTP.
struct RingWorkConfig {
    uint32_t num_local_k_chunks;
    uint32_t k_chunk_tile_count;
    uint32_t kv_local_padded_Nt;
    bool kernel_chunked;
    uint32_t q_chunk_group_tile_count;
    uint32_t q_local_padded_Nt;
    uint32_t num_joint_k_chunks;
    uint32_t joint_seq_len;
    bool kv_pad_rotation_enabled;
    bool kernel_is_causal;
    bool is_balanced;
};

// Ring-work masks: walks the ring-id order via RingIdSequencer, marks ring iterations that carry non-padded
// spatial / joint KV work, and applies the causal unbalanced skip rule. logical_nt is the only per-chunk
// input; everything else comes from the compile-time Cfg (device-only -- the host derivation is validate-only).
template <RingWorkConfig Cfg>
inline RingWorkMasks build_ring_work_masks_device(
    uint32_t device_index,
    uint32_t ring_size,
    uint32_t backward_writes_expected,
    uint32_t forward_writes_expected,
    uint32_t logical_nt) {
    RingWorkMasks plan;
    plan.active_ring_iter_mask = 0;
    plan.single_valid_kv_chunk_mask = 0;

    RingIdSequencer seq(device_index, ring_size, backward_writes_expected, forward_writes_expected);
    auto noop_sync = [](uint32_t, uint32_t) {};

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        const uint32_t ring_id = seq.get_next_ring_id(noop_sync);
        const bool joint_contributes = ring_id == ring_size - 1 && Cfg.num_joint_k_chunks > 0 && Cfg.joint_seq_len != 0;
        uint32_t valid_spatial_kv_chunks = 0;
        for (uint32_t k_chunk = 0; k_chunk < Cfg.num_local_k_chunks; ++k_chunk) {
            const uint32_t local_tile_start = k_chunk * Cfg.k_chunk_tile_count;
            if (local_tile_start >= Cfg.kv_local_padded_Nt) {
                continue;
            }
            if (kv_global_tile_for_ring_plan(
                    Cfg.kernel_chunked,
                    ring_id,
                    local_tile_start,
                    Cfg.q_chunk_group_tile_count,
                    Cfg.q_local_padded_Nt,
                    Cfg.kv_local_padded_Nt) < logical_nt) {
                valid_spatial_kv_chunks++;
            }
        }
        const uint32_t valid_kv_chunks = valid_spatial_kv_chunks + (joint_contributes ? Cfg.num_joint_k_chunks : 0);
        const bool has_kv_work = (Cfg.kernel_chunked && !Cfg.kv_pad_rotation_enabled) || valid_spatial_kv_chunks > 0;
        const bool ring_iter_does_work =
            (has_kv_work || joint_contributes) && !(Cfg.kernel_is_causal && device_index < ring_id && !Cfg.is_balanced);
        if (ring_iter_does_work) {
            plan.active_ring_iter_mask |= (1u << ring_iter);
        }
        // "At most one valid KV chunk" (0 or 1): a deferred-norm ordering hint for the writer. Zero-work
        // iters also set it, but the writer only consults it after the active_ring_iter_mask skip.
        if (valid_kv_chunks <= 1) {
            plan.single_valid_kv_chunk_mask |= (1u << ring_iter);
        }
    }
    return plan;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
