// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/where/device/kernels/dataflow/elemwise_reader_kernel_args.hpp"
#include "ttnn/operations/experimental/where/device/kernels/dataflow/elemwise_writer_kernel_args.hpp"
#include "ttnn/operations/experimental/where/device/kernels/compute/elemwise_where_kernel_args.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include <optional>

namespace ttnn::experimental::prim {

template <bool initialize_args>
inline void set_eltwise_ternary_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& condition_tensor,
    const Tensor& true_value_tensor,
    const Tensor& false_value_tensor,
    const Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle compute_kernel_id,
    const CoreRangeSet& all_device_cores) {
    using namespace ttnn::kernel::eltwise::where_args;
    using namespace ttnn::kernel_utils;
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

    uint32_t num_tiles = static_cast<uint32_t>(condition_tensor.physical_volume() / TILE_HW);
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

    std::vector<std::vector<uint32_t>> reader_args_array{
        cores.size(), std::vector<uint32_t>{amount_of_fields<ElemwiseReaderKernelArgs>(), 0}};
    std::vector<std::vector<uint32_t>> compute_args_array{
        cores.size(), std::vector<uint32_t>(amount_of_fields<ElemwiseComputeKernelArgs>(), 0)};
    std::vector<std::vector<uint32_t>> writer_args_array{
        cores.size(), std::vector<uint32_t>(amount_of_fields<ElemwiseWriterKernelArgs>(), 0)};

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    for (uint32_t i = 0, tile_ofs = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores.at(i);

        uint32_t block_cnt_per_core = 0;
        uint32_t block_size_per_core = 1;
        uint32_t num_tiles_per_core = 0;

        if (i < core_group_1.num_cores()) {
            num_tiles_per_core = num_tiles_per_core_group_1;
            block_cnt_per_core = num_tiles_per_core_group_1;
        } else if (i < num_cores) {
            num_tiles_per_core = num_tiles_per_core_group_2;
            block_cnt_per_core = num_tiles_per_core_group_2;
        } else {
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

        ElemwiseReaderKernelArgs read_kern_args = {
            .condition_tensor_base_addr = condition_tensor.buffer()->address(),
            .true_tensor_base_addr = true_value_tensor.buffer()->address(),
            .false_tensor_base_addr = false_value_tensor.buffer()->address(),
            .num_tiles = num_tiles_per_core,
            .tile_ofs = tile_ofs};

        ElemwiseComputeKernelArgs compute_kern_args = {
            .per_core_block_cnt = block_cnt_per_core, .per_core_block_size = block_size_per_core};

        ElemwiseWriterKernelArgs write_kern_args = {
            .dst_base_addr = output.buffer()->address(), .num_tiles = num_tiles_per_core, .tile_ofs = tile_ofs};

        reader_args_array[i] = to_vector(read_kern_args);
        compute_args_array[i] = to_vector(compute_kern_args);
        writer_args_array[i] = to_vector(write_kern_args);
        if constexpr (!initialize_args) {
            auto& core_reader_args = cached_reader_args.at(core.x).at(core.y);
            std::ranges::copy(reader_args_array[i], core_reader_args.data());

            auto& core_eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            std::ranges::copy(compute_args_array[i], core_eltwise_args.data());

            auto& core_writer_args = cached_writer_args.at(core.x).at(core.y);
            std::ranges::copy(writer_args_array[i], core_writer_args.data());
        }

        tile_ofs += num_tiles_per_core;
    }

    if constexpr (initialize_args) {
        SetRuntimeArgs(program, reader_kernel_id, cores, reader_args_array);
        SetRuntimeArgs(program, compute_kernel_id, cores, compute_args_array);
        SetRuntimeArgs(program, writer_kernel_id, cores, writer_args_array);
    }
}

}  // namespace ttnn::experimental::prim
