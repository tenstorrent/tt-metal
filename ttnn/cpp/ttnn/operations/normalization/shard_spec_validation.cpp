// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "shard_spec_validation.hpp"

#include <tt-metalium/math.hpp>

namespace ttnn::operations::normalization::detail {

using namespace tt::tt_metal;

void validate_sharded_input(const Tensor& a, const CoreCoord& program_grid, bool require_shard_width_tile_aligned) {
    TT_FATAL(a.is_sharded(), "validate_sharded_input called on non-sharded tensor");

    const auto& shard_spec = a.shard_spec().value();
    const auto& shard_shape = shard_spec.shape;
    const auto& shard_grid = shard_spec.grid;
    const uint32_t tile_h = a.tensor_spec().tile().get_height();
    const uint32_t tile_w = a.tensor_spec().tile().get_width();

    // 1. Grid hierarchy.
    TT_FATAL(shard_grid.num_cores() > 0, "Shard grid must have at least one core");

    const auto device_grid = a.device()->compute_with_storage_grid_size();
    const CoreRange device_range(CoreCoord{0, 0}, CoreCoord{device_grid.x - 1, device_grid.y - 1});
    const CoreRange program_range(CoreCoord{0, 0}, CoreCoord{program_grid.x - 1, program_grid.y - 1});
    TT_FATAL(
        device_range.contains(program_range),
        "program_config grid ({}x{}) must be contained within device grid ({}x{})",
        program_grid.x,
        program_grid.y,
        device_grid.x,
        device_grid.y);

    const auto bbox = shard_grid.bounding_box();
    const auto shard_offset = bbox.start_coord;
    const CoreRange shifted_shard_bbox(
        CoreCoord{0, 0}, CoreCoord{bbox.end_coord.x - shard_offset.x, bbox.end_coord.y - shard_offset.y});
    TT_FATAL(
        program_range.contains(shifted_shard_bbox),
        "shard_spec.grid size {}x{} does not fit within program_config grid {}x{}",
        bbox.end_coord.x - bbox.start_coord.x + 1,
        bbox.end_coord.y - bbox.start_coord.y + 1,
        program_grid.x,
        program_grid.y);

    // 2. Shard shape sanity.
    TT_FATAL(
        shard_shape[0] > 0 && shard_shape[1] > 0,
        "shard shape must be non-zero, got H={} W={}",
        shard_shape[0],
        shard_shape[1]);
    TT_FATAL(
        shard_shape[0] % tile_h == 0, "shard height {} must be divisible by tile height {}", shard_shape[0], tile_h);
    if (require_shard_width_tile_aligned) {
        TT_FATAL(
            shard_shape[1] % tile_w == 0, "shard width {} must be divisible by tile width {}", shard_shape[1], tile_w);
    }

    // 3 + 4. Shard count along each axis must match memory layout, and shard-padded volume must cover the tensor with
    // less than one shard of trailing pad on each axis.
    const auto num_cores = shard_grid.num_cores();
    const uint64_t W_phys = a.padded_shape()[-1];
    const uint64_t H_phys = a.physical_volume() / W_phys;
    const auto memory_layout = a.memory_config().memory_layout();

    uint32_t num_shards_h = 1;
    uint32_t num_shards_w = 1;
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_shards_h = num_cores;
    } else if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        num_shards_w = num_cores;
    } else {  // BLOCK_SHARDED
        num_shards_h = tt::div_up(static_cast<uint32_t>(H_phys), shard_shape[0]);
        num_shards_w = tt::div_up(static_cast<uint32_t>(W_phys), shard_shape[1]);
    }
    TT_FATAL(
        static_cast<uint64_t>(num_shards_h) * num_shards_w == num_cores,
        "Shard layout requires {}x{} = {} shards but shard grid has {} cores",
        num_shards_h,
        num_shards_w,
        static_cast<uint64_t>(num_shards_h) * num_shards_w,
        num_cores);

    const uint64_t shard_padded_h = static_cast<uint64_t>(num_shards_h) * shard_shape[0];
    const uint64_t shard_padded_w = static_cast<uint64_t>(num_shards_w) * shard_shape[1];
    TT_FATAL(
        shard_padded_h >= H_phys && (shard_padded_h - H_phys) < shard_shape[0],
        "Shard-padded height ({}x{} = {}) does not align with tensor height {}: trailing pad {} must be less than one "
        "shard height ({})",
        num_shards_h,
        shard_shape[0],
        shard_padded_h,
        H_phys,
        shard_padded_h - H_phys,
        shard_shape[0]);
    TT_FATAL(
        shard_padded_w >= W_phys && (shard_padded_w - W_phys) < shard_shape[1],
        "Shard-padded width ({}x{} = {}) does not align with tensor width {}: trailing pad {} must be less than one "
        "shard width ({})",
        num_shards_w,
        shard_shape[1],
        shard_padded_w,
        W_phys,
        shard_padded_w - W_phys,
        shard_shape[1]);
}

}  // namespace ttnn::operations::normalization::detail
