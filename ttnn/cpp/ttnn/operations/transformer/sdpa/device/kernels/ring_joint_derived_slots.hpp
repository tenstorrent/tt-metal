// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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
constexpr uint32_t kReaderMetadataCommonArgCount = 4;  // slot_id addr, num_layers, layer_idx, kv_actual_isl addr
constexpr uint32_t kWriterMetadataCommonArgCount = 1;  // kv_actual_isl addr

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
