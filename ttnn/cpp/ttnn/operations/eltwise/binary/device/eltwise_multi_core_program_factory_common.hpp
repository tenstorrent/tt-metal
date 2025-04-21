// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include "binary_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

#include <tt-metalium/work_split.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::binary {

inline __attribute__((always_inline)) void set_eltwise_binary_runtime_args(
    const Tensor& a,
    const Tensor& b,
    const Tensor& output,
    tt::tt_metal::KernelDescriptor& binary_reader_kernel,
    tt::tt_metal::KernelDescriptor& unary_writer_kernel,
    tt::tt_metal::KernelDescriptor& eltwise_binary_kernel,
    const CoreRangeSet& all_device_cores,
    const uint32_t src0_single_tile_size,
    const uint32_t src1_single_tile_size,
    const uint32_t dst_single_tile_size) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    auto src_buffer_a = a.buffer();
    auto src_buffer_b = b.buffer();
    auto dst_buffer = output.buffer();

    CoreRangeSet all_cores, core_group_1, core_group_2;

    std::optional<ShardSpec> shard_spec = std::nullopt;
    std::optional<TensorMemoryLayout> sharded_layout = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool src1_sharded = b.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    bool block_or_width_sharded = false;
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
        block_or_width_sharded = a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        sharded_layout = a.memory_config().memory_layout;
    } else if (src1_sharded) {
        shard_spec = b.shard_spec().value();
        block_or_width_sharded = b.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        sharded_layout = b.memory_config().memory_layout;
    } else if (out_sharded) {
        shard_spec = output.shard_spec().value();
        block_or_width_sharded = output.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        sharded_layout = output.memory_config().memory_layout;
    }

    // zero_start_grid is a flag to indicate that we are using a single rectangular grid that starts at (0, 0)
    // as well as having the sharded tensors (if any) start at (0, 0)
    // This will run the original work/core distribution algorithms that are specifically for this setup, as these
    // are faster than the generic work/core distribution algorithms that work on arbitrary CoreRangeSets
    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid_size;
    if (all_device_cores.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (shard_spec.has_value()) {
                const auto& shard_start_coord = shard_spec->grid.ranges()[0].start_coord;
                if (shard_start_coord.x == 0 && shard_start_coord.y == 0) {
                    zero_start_grid = true;
                    compute_with_storage_grid_size = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
                }
            } else {
                zero_start_grid = true;
                compute_with_storage_grid_size = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
            }
        }
    }

    uint32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_cores, num_tiles_per_core_group_1, num_tiles_per_core_group_2, num_cores_total;
    if (zero_start_grid) {
        num_cores_total = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    } else {
        num_cores_total = all_device_cores.num_cores();
    }

    uint32_t block_size_per_core_group_1 = 1, block_size_per_core_group_2 = 1, max_block_size = 1;

    uint32_t block_cnt_per_core_group_1, block_cnt_per_core_group_2;

    bool row_major;
    uint32_t block_height = 0, block_width = 0, block_size = 0, output_width = 0, last_unpadded_block_height = 0,
             last_unpadded_block_width = 0;
    CoreCoord end_core;
    std::vector<CoreCoord> cores;

    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_tiles_per_core_group_1 = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        num_tiles_per_core_group_2 = 0;
        block_size_per_core_group_1 = find_max_block_size(num_tiles_per_core_group_1);
        max_block_size = block_size_per_core_group_1;

        block_cnt_per_core_group_1 = num_tiles_per_core_group_1 / block_size_per_core_group_1;
        block_cnt_per_core_group_2 = num_tiles_per_core_group_2 / block_size_per_core_group_2;
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        block_height = shard_spec.value().shape[0] / TILE_HEIGHT;
        block_width = shard_spec.value().shape[1] / TILE_WIDTH;
        if (block_or_width_sharded) {
            block_size = block_width * block_height;
            end_core = (*shard_spec.value().grid.ranges().begin()).end_coord;
            output_width = output.get_padded_shape()[-1] / TILE_WIDTH;
            uint32_t output_height = output.volume() / output.get_padded_shape()[-1] / TILE_HEIGHT;
            last_unpadded_block_height = block_height - (round_up(output_height, block_height) - output_height);
            last_unpadded_block_width = block_width - (round_up(output_width, block_width) - output_width);
        }
        if (zero_start_grid) {
            auto bbox = core_group_1.bounding_box();
            cores = grid_to_cores_with_noop(
                bbox.end_coord.x,
                bbox.end_coord.y,
                compute_with_storage_grid_size.x,
                compute_with_storage_grid_size.y,
                row_major);
        } else {
            cores = grid_to_cores_with_noop(all_cores, all_device_cores, row_major);
        }
    } else {
        row_major = true;
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            zero_start_grid ? tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major)
                            : tt::tt_metal::split_work_to_cores(all_device_cores, num_tiles, row_major);
        block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
        block_cnt_per_core_group_2 = num_tiles_per_core_group_2;
        if (zero_start_grid) {
            cores = grid_to_cores(
                num_cores_total, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major);
        } else {
            cores = corerange_to_cores(all_device_cores, {}, row_major);
        }
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores.at(i);
        auto& binary_reader_args = binary_reader_kernel.runtime_args[core.x][core.y];
        auto& eltwise_binary_args = eltwise_binary_kernel.runtime_args[core.x][core.y];
        auto& unary_writer_args = unary_writer_kernel.runtime_args[core.x][core.y];

        uint32_t num_tiles_per_core = 0;
        uint32_t block_cnt_per_core = 0;
        uint32_t block_size_per_core = 0;
        uint32_t num_shards_per_height = 0;
        uint32_t num_shards_per_width = 0;
        uint32_t start_id = 0;
        if (shard_spec.has_value()) {
            if (sharded_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
                num_shards_per_height = num_cores;
                num_shards_per_width = 1;
            } else if (sharded_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
                num_shards_per_width = num_cores;
                num_shards_per_height = 1;
            } else {  // block sharded
                auto bbox = core_group_1.bounding_box();
                if (shard_spec.value().orientation == ShardOrientation::ROW_MAJOR) {
                    num_shards_per_height = bbox.end_coord.y - bbox.start_coord.y + 1;
                    num_shards_per_width = bbox.end_coord.x - bbox.start_coord.x + 1;
                } else {
                    num_shards_per_height = bbox.end_coord.x - bbox.start_coord.x + 1;
                    num_shards_per_width = bbox.end_coord.y - bbox.start_coord.y + 1;
                }
            }
            start_id = (i / num_shards_per_width) * (block_height * block_width * num_shards_per_width) +
                       (i % num_shards_per_width) * block_width;
        } else {
            start_id = num_tiles_read;
        }

        if (i < g1_numcores) {
            num_tiles_per_core = num_tiles_per_core_group_1;
            block_cnt_per_core = block_cnt_per_core_group_1;
            block_size_per_core = block_size_per_core_group_1;
        } else if (i < num_cores) {
            num_tiles_per_core = num_tiles_per_core_group_2;
            block_cnt_per_core = block_cnt_per_core_group_2;
            block_size_per_core = block_size_per_core_group_2;
        } else {
            continue;
        }
        binary_reader_args = {
            src_buffer_a->address(),
            src_buffer_b->address(),
            num_tiles_per_core,
            start_id,
            block_height,
            block_width,
            num_shards_per_width,
            num_shards_per_width};
        eltwise_binary_args = {block_cnt_per_core, block_size_per_core};
        if (block_or_width_sharded and not out_sharded) {
            uint32_t unpadded_block_height = block_height;
            uint32_t unpadded_block_width = block_width;
            if (row_major) {
                if (core.x == end_core.x) {
                    unpadded_block_width = last_unpadded_block_width;
                }
                if (core.y == end_core.y) {
                    unpadded_block_height = last_unpadded_block_height;
                }
            } else {
                if (core.y == end_core.y) {
                    unpadded_block_width = last_unpadded_block_width;
                }
                if (core.x == end_core.x) {
                    unpadded_block_height = last_unpadded_block_height;
                }
            }
            unary_writer_args = {
                dst_buffer->address(),
                block_height,
                block_width,
                unpadded_block_height,
                unpadded_block_width,
                output_width,
                block_size,
                (i / num_shards_per_width) * (block_height * block_width * num_shards_per_width) +
                    (i % num_shards_per_width) * block_width,
                0};
        } else {
            unary_writer_args = {dst_buffer->address(), num_tiles_per_core, num_tiles_read};
        }
        num_tiles_read += num_tiles_per_core;
    }
}

}  // namespace ttnn::operations::binary
