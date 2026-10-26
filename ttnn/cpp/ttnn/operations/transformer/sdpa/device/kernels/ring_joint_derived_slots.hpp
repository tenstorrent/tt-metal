// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

// Shared host/kernel contract for the trace-safe logical_n / logical_l transport: the cb_kv_pad_derived
// slot layout, the common-runtime-arg bases, the partial-tile forcing rule, and the tile-math helpers.
namespace ttnn::operations::transformer::sdpa::ring_joint {

// cb_kv_pad_derived slot layout. The reader fills these (compute cannot NoC-read DRAM) and compute reads
// them back by index. Slots 1-4 are the KV-pad Q-mapping (metadata path only); slots 6/7/8 are the
// logical-length transport (tensor path only). Both paths write slots 0 and 5.
constexpr uint32_t kDerivedLogicalNt = 0;
constexpr uint32_t kDerivedQPreWrapStartTile = 1;
constexpr uint32_t kDerivedQPreWrapTileCount = 2;
constexpr uint32_t kDerivedQPostWrapStartTile = 3;
constexpr uint32_t kDerivedQValidTileCount = 4;
constexpr uint32_t kDerivedActiveRingIterMask = 5;
constexpr uint32_t kDerivedGlobalNPartialCol = 6;
constexpr uint32_t kDerivedLogicalLt = 7;
constexpr uint32_t kDerivedJointLPartialCol = 8;
constexpr uint32_t kDerivedSlotCount = 9;

// Common-runtime-arg layout: the metadata block (when present) precedes the logical-length pair.
// slot_id addr, num_layers, layer_idx, batch count, kv_actual_isl addr
constexpr uint32_t kReaderMetadataCommonArgCount = 5;
constexpr uint32_t kWriterMetadataCommonArgCount = 1;  // kv_actual_isl addr
constexpr uint32_t kLogicalLengthCommonArgCount = 2;   // logical_n addr, logical_l addr (pair, when either is present)

// dynamic_local_n: the per-chip padded length N_local (in tiles) is one common runtime word per kernel, placed
// after every other common arg; every count below is derived from it on both sides.
constexpr uint32_t dynamic_local_n_reader_common_arg(
    bool slot_from_metadata, bool has_logical_length_tensor, bool has_kv_valid_end) {
    return (slot_from_metadata ? kReaderMetadataCommonArgCount : 0u) +
           (has_logical_length_tensor ? kLogicalLengthCommonArgCount : 0u) + (has_kv_valid_end ? 1u : 0u);
}
constexpr uint32_t dynamic_local_n_writer_common_arg(
    bool kv_pad_from_metadata, bool has_logical_length_tensor, bool has_kv_valid_end) {
    return (kv_pad_from_metadata ? kWriterMetadataCommonArgCount : 0u) +
           (has_logical_length_tensor ? kLogicalLengthCommonArgCount : 0u) + (has_kv_valid_end ? 1u : 0u);
}
constexpr uint32_t kDynamicLocalNComputeCommonArg = 0;

struct LocalNGeometry {
    uint32_t q_local_padded_Nt;
    uint32_t kv_local_padded_Nt;
    uint32_t num_local_q_chunks;
    uint32_t num_local_k_chunks;
    uint32_t num_q_chunks;  // local + joint
    uint32_t kv_region_Nt;
    uint32_t q_chunk_group_tile_count;
};

// Q and K/V share the per-chip length on the dynamic path (validated non-chunked).
constexpr LocalNGeometry derive_local_n_geometry(
    uint32_t n_local_tiles,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t kv_stripe_split,
    uint32_t ring_size,
    uint32_t num_joint_q_chunks) {
    const uint32_t kv_region_Nt = n_local_tiles / kv_stripe_split;
    return LocalNGeometry{
        n_local_tiles,
        n_local_tiles,
        ttnn::runtime_length::div_up(n_local_tiles, Sq_chunk_t),
        ttnn::runtime_length::div_up(n_local_tiles, Sk_chunk_t),
        ttnn::runtime_length::div_up(n_local_tiles, Sq_chunk_t) + num_joint_q_chunks,
        kv_region_Nt,
        kv_region_Nt * ring_size};
}

// A tile-aligned placeholder implies partial column 0, but a device tensor can land mid-tile at any
// dispatch, so the boundary tile must exist in the CB layout whenever a length tensor is in play. The
// kernels stamp the live column into it and gate the stamp off when the live column is 0.
constexpr bool partial_tile_present(uint32_t partial_col, bool has_length_tensor) {
    return (partial_col != 0) || has_length_tensor;
}

// Tile-math for the dataflow kernels' live-length narrowing. TILE_HEIGHT is 32 on this op's tiles.
constexpr uint32_t kTileHeight = 32;
constexpr uint32_t tiles_for(uint32_t len) { return (len + kTileHeight - 1) / kTileHeight; }
constexpr uint32_t tile_partial_col(uint32_t len) { return len % kTileHeight; }

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
