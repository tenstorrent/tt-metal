// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

#include <optional>

namespace ttnn::operations::ternary::experimental {

template <bool initialize_args>
inline void set_eltwise_ternary_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle eltwise_ternary_kernel_id,
    const CoreRangeSet& all_device_cores) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    // zero_start_grid is a flag to indicate that we are using a single rectangular grid that starts at (0, 0)
    // as well as having the sharded tensors (if any) start at (0, 0)
    // This will run the original work/core distribution algorithms that are specifically for this setup, as these
    // are faster than the generic work/core distribution algorithms that work on arbitrary CoreRangeSets
    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid_size;
    if (all_device_cores.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            zero_start_grid = true;
            compute_with_storage_grid_size = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
        }
    }

    uint32_t num_tiles = a.volume() / TILE_HW;
    bool row_major = true;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        zero_start_grid ? tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major)
                        : tt::tt_metal::split_work_to_cores(all_device_cores, num_tiles, row_major);

    uint32_t num_cores_total = zero_start_grid ? compute_with_storage_grid_size.x * compute_with_storage_grid_size.y
                                               : all_device_cores.num_cores();
    auto cores =
        zero_start_grid
            ? grid_to_cores(
                  num_cores_total, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, row_major)
            : corerange_to_cores(all_device_cores, {}, row_major);

    // TODO: Don't use hardcoded size
    std::vector<std::vector<uint32_t>> binary_reader_args{cores.size(), std::vector<uint32_t>(7)};
    std::vector<std::vector<uint32_t>> eltwise_binary_args{cores.size(), std::vector<uint32_t>(2)};
    std::vector<std::vector<uint32_t>> unary_writer_args{cores.size(), std::vector<uint32_t>(3)};

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, eltwise_ternary_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    uint32_t block_size_per_core_group_1 = 1, block_size_per_core_group_2 = 1;
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores.at(i);

        uint32_t block_cnt_per_core = 0;
        uint32_t block_size_per_core = 0;
        uint32_t num_tiles_per_core = 0;

        if (i < core_group_1.num_cores()) {
            num_tiles_per_core = num_tiles_per_core_group_1;
            block_cnt_per_core = num_tiles_per_core_group_1;
            block_size_per_core = block_size_per_core_group_1;
        } else if (i < num_cores) {
            num_tiles_per_core = num_tiles_per_core_group_2;
            block_cnt_per_core = num_tiles_per_core_group_2;
            block_size_per_core = block_size_per_core_group_2;
        } else {
            // TODO: What is the use case?
            // Zero out non-working cores RT args. Only necessary in override
            // since initialization pushes zero vectors to unused cores.
            if constexpr (!initialize_args) {
                // RuntimeArgsData
                auto& reader_args = cached_reader_args.at(core.x).at(core.y);
                reader_args[2] = 0;
                auto& eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
                eltwise_args[0] = 0;
                auto& writer_args = cached_writer_args.at(core.x).at(core.y);
                writer_args[1] = 0;
            }
            continue;
        }

        binary_reader_args[i] = {
            a.buffer()->address(), b.buffer()->address(), c.buffer()->address(), num_tiles_per_core, num_tiles_read};
        eltwise_binary_args[i] = {block_cnt_per_core, block_size_per_core};
        unary_writer_args[i] = {output.buffer()->address(), num_tiles_per_core, num_tiles_read};

        if constexpr (!initialize_args) {
            auto& core_reader_args = cached_reader_args.at(core.x).at(core.y);
            std::ranges::copy(binary_reader_args[i], core_reader_args.data());

            auto& core_eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            std::ranges::copy(eltwise_binary_args[i], core_eltwise_args.data());

            auto& core_writer_args = cached_writer_args.at(core.x).at(core.y);
            std::ranges::copy(unary_writer_args[i], core_writer_args.data());
        }

        num_tiles_read += num_tiles_per_core;
    }

    if constexpr (initialize_args) {
        SetRuntimeArgs(program, reader_kernel_id, cores, binary_reader_args);
        SetRuntimeArgs(program, eltwise_ternary_kernel_id, cores, eltwise_binary_args);
        SetRuntimeArgs(program, writer_kernel_id, cores, unary_writer_args);
    }
}

}  // namespace ttnn::operations::ternary::experimental
