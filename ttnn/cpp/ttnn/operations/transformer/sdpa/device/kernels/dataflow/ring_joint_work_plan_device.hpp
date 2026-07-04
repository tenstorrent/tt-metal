// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-side driver for the ring_joint per-chunk work plan: reads the trace-safe metadata tensor (when
// present) and runs the single-source derivations in ring_joint_work_plan.hpp, returning the handful of
// scalars the SDPA reader and writer each need. This is the device-only companion to ring_joint_work_plan.hpp
// (which stays host-includable for validate_kv_pad_q_mapping); it lives here because it pulls in the NoC /
// CircularBuffer read helper. The reader and writer both call derive_ring_work_plan so the derivation exists
// once; each uses the subset of RingWorkPlan it needs (reader: cache_slot + q_mapping; writer: single_valid).

#pragma once

#include <cstdint>

#include <tt-metalium/constants.hpp>  // tt::constants::TILE_HEIGHT
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_work_plan.hpp"  // derivations
#include "cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_metadata.hpp"  // read_ring_metadata

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Common runtime-arg layout on the metadata path (reader/writer contract with the host factory, which emits
// these via emplace_common_runtime_args): [0] = metadata tensor DRAM address (a Buffer* binding, re-patched
// on cache hits), [1] = kv_cache_num_layers, [2] = kv_cache_layer_idx. The reader reads all three (slot ->
// cache_batch_idx); the writer reads only [0].
constexpr uint32_t kMetadataAddrCommonArg = 0;
constexpr uint32_t kKvCacheNumLayersCommonArg = 1;
constexpr uint32_t kKvCacheLayerIdxCommonArg = 2;

// Everything the reader/writer derive per chunk. The reader consumes cache_slot (-> cache_batch_idx) and
// q_mapping (-> compute via cb_kv_pad_derived); the writer consumes masks.single_valid_kv_chunk_mask; both
// consume logical_nt and masks.active_ring_iter_mask.
struct RingWorkPlan {
    uint32_t cache_slot;            // metadata[0] (indexed cache slot); 0 when the metadata tensor isn't read
    uint32_t logical_nt;            // padded global KV tile count for this chunk
    uint32_t kv_actual_tile_count;  // valid (unpadded) KV tiles; 0 off the rotation path
    KvPadQMapping q_mapping;        // per-device valid-Q segment mapping
    RingWorkMasks masks;            // active_ring_iter_mask + single_valid_kv_chunk_mask
};

// Derive the per-chunk work plan. Compile-time parameters (identical derivation, pruned per path):
//   Cfg          -- the kernel's static RingWorkConfig (NTTP: every field stays compile-time).
//   ReadMetadata -- NoC-read the metadata tensor (for cache_slot and/or kv_actual). When false, cache_slot
//                   stays 0 and kv_actual comes from kv_actual_scalar (no-trace scalar path).
//   Rotation     -- derive logical_nt / kv_actual_tile_count from kv_actual. When false, logical_nt is the
//                   static host runtime arg (logical_nt_static) and kv_actual_tile_count is 0.
// `scratch_cb` is a real, currently-unused CB used as the metadata read's L1 landing (see read_ring_metadata
// for the offset-0 / real-CB platform-trap rationale).
template <RingWorkConfig Cfg, bool ReadMetadata, bool Rotation, typename MetaAccessorArgs>
inline RingWorkPlan derive_ring_work_plan(
    Noc& noc,
    const MetaAccessorArgs& meta_args,
    uint32_t metadata_addr,
    const CircularBuffer& scratch_cb,
    uint32_t kv_actual_scalar,
    uint32_t logical_nt_static,
    uint32_t ring_index,
    uint32_t ring_size,
    uint32_t backward_writes_expected,
    uint32_t forward_writes_expected) {
    RingWorkPlan plan;
    plan.cache_slot = 0;
    uint32_t kv_actual = kv_actual_scalar;
    if constexpr (ReadMetadata) {
        const auto md = ttnn::ring_attention::read_ring_metadata(noc, meta_args, metadata_addr, scratch_cb);
        plan.cache_slot = md.slot;
        kv_actual = md.kv_actual;
    }
    plan.logical_nt = logical_nt_static;
    plan.kv_actual_tile_count = 0;
    if constexpr (Rotation) {
        plan.kv_actual_tile_count = kv_actual / tt::constants::TILE_HEIGHT;
        // chunk_global = q_chunk_group (tiles) * TILE_HEIGHT; logical_n = kv_actual + chunk_global.
        plan.logical_nt = compute_logical_nt(
            kv_actual, Cfg.q_chunk_group_tile_count * tt::constants::TILE_HEIGHT, tt::constants::TILE_HEIGHT);
    }
    plan.q_mapping = build_kv_pad_q_mapping_device(
        plan.kv_actual_tile_count, plan.logical_nt, ring_size, Cfg.q_local_padded_Nt, ring_index);
    plan.masks = build_ring_work_masks_device<Cfg>(
        ring_index, ring_size, backward_writes_expected, forward_writes_expected, plan.logical_nt);
    return plan;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
