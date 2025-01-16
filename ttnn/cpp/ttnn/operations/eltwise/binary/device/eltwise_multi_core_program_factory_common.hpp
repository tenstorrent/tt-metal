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

template <bool initialize_args>
inline __attribute__((always_inline)) void set_eltwise_binary_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const Tensor& output,
    const tt::tt_metal::KernelHandle binary_reader_kernel_id,
    const tt::tt_metal::KernelHandle unary_writer_kernel_id,
    const tt::tt_metal::KernelHandle eltwise_binary_kernel_id,
    const tt::tt_metal::CBHandle cb_src0,
    const tt::tt_metal::CBHandle cb_src1,
    const tt::tt_metal::CBHandle cb_output,
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
            output_width = output.get_legacy_shape()[-1] / TILE_WIDTH;
            uint32_t output_height = output.volume() / output.get_legacy_shape()[-1] / TILE_HEIGHT;
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

    std::vector<std::vector<uint32_t>> binary_reader_args;
    std::vector<std::vector<uint32_t>> eltwise_binary_args;
    std::vector<std::vector<uint32_t>> unary_writer_args;
    if constexpr (initialize_args) {
        binary_reader_args = {cores.size(), std::vector<uint32_t>(7)};
        eltwise_binary_args = {cores.size(), std::vector<uint32_t>(2)};
        if (block_or_width_sharded and not out_sharded) {
            unary_writer_args = {cores.size(), std::vector<uint32_t>(7)};
        } else {
            unary_writer_args = {cores.size(), std::vector<uint32_t>(3)};
        }
    }

    auto& cached_reader_args = GetRuntimeArgs(program, binary_reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, eltwise_binary_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, unary_writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores.at(i);
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
            // Zero out non-working cores RT args. Only necessary in override
            // since initialization pushes zero vectors to unused cores.
            if constexpr (!initialize_args) {
                auto& reader_args = cached_reader_args.at(core.x).at(core.y);
                reader_args[2] = 0;
                auto& eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
                eltwise_args[0] = 0;
                auto& writer_args = cached_writer_args.at(core.x).at(core.y);
                writer_args[1] = 0;
            }
            continue;
        }
        if constexpr (initialize_args) {
            binary_reader_args[i] = {
                src_buffer_a->address(),
                src_buffer_b->address(),
                num_tiles_per_core,
                start_id,
                block_height,
                block_width,
                num_shards_per_width,
                num_shards_per_width};
            eltwise_binary_args[i] = {block_cnt_per_core, block_size_per_core};
        } else {
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            reader_args[0] = src_buffer_a->address();
            reader_args[1] = src_buffer_b->address();
            reader_args[2] = num_tiles_per_core;
            reader_args[3] = start_id;
            reader_args[4] = block_height;
            reader_args[5] = block_width;
            reader_args[6] = num_shards_per_width;
            auto& eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            eltwise_args[0] = block_cnt_per_core;
            eltwise_args[1] = block_size_per_core;
        }
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
            if constexpr (initialize_args) {
                unary_writer_args[i] = {
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
                auto& writer_args = cached_writer_args.at(core.x).at(core.y);
                writer_args[0] = dst_buffer->address();
                writer_args[1] = block_height;
                writer_args[2] = block_width;
                writer_args[3] = unpadded_block_height;
                writer_args[4] = unpadded_block_width;
                writer_args[5] = output_width;
                writer_args[6] = block_size;
                writer_args[7] = (i / num_shards_per_width) * (block_height * block_width * num_shards_per_width) +
                                 (i % num_shards_per_width) * block_width;
                writer_args[8] = 0;
            }
        } else {
            if constexpr (initialize_args) {
                unary_writer_args[i] = {dst_buffer->address(), num_tiles_per_core, num_tiles_read};
            } else {
                auto& writer_args = cached_writer_args.at(core.x).at(core.y);
                writer_args[0] = dst_buffer->address();
                writer_args[1] = num_tiles_per_core;
                writer_args[2] = num_tiles_read;
            }
        }
        num_tiles_read += num_tiles_per_core;
    }

    if constexpr (initialize_args) {
        SetRuntimeArgs(program, binary_reader_kernel_id, cores, binary_reader_args);
        SetRuntimeArgs(program, eltwise_binary_kernel_id, cores, eltwise_binary_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);
    }

    if (src0_sharded) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            program, cb_src0, *src_buffer_a, num_tiles_per_core_group_1 * src0_single_tile_size);
    }
    if (src1_sharded) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            program, cb_src1, *src_buffer_b, num_tiles_per_core_group_1 * src1_single_tile_size);
    }
    if (out_sharded) {
        UpdateDynamicCircularBufferAddressAndTotalSize(
            program, cb_output, *dst_buffer, num_tiles_per_core_group_1 * dst_single_tile_size);
    }
}

}  // namespace ttnn::operations::binary
