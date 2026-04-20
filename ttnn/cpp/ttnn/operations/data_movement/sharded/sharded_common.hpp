// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <utility>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::detail {

// Utility function
uint32_t calculate_starting_idx_h(const Tensor& tensor, uint32_t num_slices, uint32_t slice_index);

// Compute chunked CB sizing for the DRAM-destination WIDTH-sharded TILE staging path
// in interleaved_to_sharded. When the output is DRAM sharded, the output CB is *not*
// globally allocated to the output buffer (DRAM buffers can't back an L1 CB) — it lives
// in local L1 as a staging buffer, so the reader/writer kernels process the shard in
// height-chunks that fit in an L1 budget.
//
// The chunking path is gated on three conditions, all of which must hold together:
//
//   1. dst_is_dram — L1-destination CBs are globally allocated to the output buffer, so
//      the full shard fits by construction and chunking is unnecessary.
//   2. is_tile_layout — tiled layout only. Row-major DRAM-sharded writes go through a
//      different reader/writer pair (*_stick_layout_*) that currently has the same
//      one-shot reserve/push pattern we fixed for TILE, but is not yet on any hot path
//      we care about (DRAM matmul weights are always tiled). When a large row-major
//      DRAM-sharded use case emerges, the same chunking pattern should be ported to the
//      stick-layout kernels.
//   3. is_width_sharded — DRAM sharding is effectively always WIDTH_SHARDED in practice
//      (one shard per DRAM bank, width split across banks). Height- and block-sharded
//      DRAM configs can be constructed but aren't realistic production paths and have
//      different per-core shard-shape-last handling; restricting to WIDTH keeps the
//      chunk math aligned with what's actually exercised.
//
// Returns {chunk_height_tiles, num_input_units}:
//   chunk_height_tiles — compile-time arg consumed by reader/writer kernels
//   num_input_units    — CB page count = chunk_height_tiles * num_units_per_shard_width
//
// Outside the gate, chunk_height_tiles == num_units_per_shard_height and
// num_input_units == num_units_per_shard (full shard, single pass — identical to the
// pre-chunking behavior).
std::pair<uint32_t, uint32_t> compute_staging_cb_chunk(
    const Tensor& input,
    bool dst_is_dram,
    bool is_tile_layout,
    bool is_width_sharded,
    bool convert_df,
    uint32_t input_page_size,
    uint32_t output_page_size,
    uint32_t scratch_cb_bytes,
    uint32_t num_units_per_shard_height,
    uint32_t num_units_per_shard_width,
    uint32_t num_units_per_shard);

struct WidthShardingReshardSegment {
    uint32_t write_size = 0;
    uint32_t read_offset = 0;
    uint32_t bank_id = 0;
    uint32_t write_offset = 0;
};

// Precompute a set of reads/writes for each core needed to perform a width-sharded reshard operations
std::tuple<std::vector<std::vector<WidthShardingReshardSegment>>, uint32_t, uint32_t, uint32_t>
compute_width_sharding_reshard_segments(
    const std::array<uint32_t, 2>& local_shard_shape,
    const std::array<uint32_t, 2>& remote_shard_shape,
    const std::vector<CoreCoord>& local_cores,
    const std::vector<CoreCoord>& remote_cores,
    const tt::tt_metal::BufferType& remote_buffer_type,
    const tt::CoreType& remote_core_type,
    tt::tt_metal::IDevice* device,
    uint32_t element_size);

}  // namespace ttnn::operations::data_movement::detail
