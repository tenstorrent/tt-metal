// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/add/device/kernels/dataflow/elemwise_reader_kernel_args.hpp"
#include "ttnn/operations/experimental/add/device/kernels/dataflow/elt_nd_sharded_add_reader_args.hpp"

#include "ttnn/operations/experimental/add/device/kernels/dataflow/elemwise_writer_kernel_args.hpp"
#include "ttnn/operations/experimental/add/device/kernels/compute/elemwise_add_kernel_args.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include <algorithm>
#include <optional>

namespace ttnn::experimental::prim {

template <bool initialize_args>
inline void set_eltwise_binary_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle compute_kernel_id,
    const CoreRangeSet& all_device_cores) {
    using namespace ttnn::kernel::eltwise::add_args;
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

    uint32_t num_tiles = static_cast<uint32_t>(a_tensor.physical_volume() / TILE_HW);
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

    // std::cout << "compute_with_storage_grid_size: " << compute_with_storage_grid_size.x << ", "
    //           << compute_with_storage_grid_size.y << std::endl;

    // std::cout << "cores: " << cores.size() << std::endl;

    // std::cout << "num_cores: " << num_cores << std::endl;
    // std::cout << "all_device_cores: " << all_device_cores.num_cores() << std::endl;
    // std::cout << "zero_start_grid: " << zero_start_grid << std::endl;
    // std::cout << "row_major: " << row_major << std::endl;
    // std::cout << "num_tiles: " << num_tiles << std::endl;
    // std::cout << "num_tiles_per_core_group_1: " << num_tiles_per_core_group_1 << std::endl;
    // std::cout << "num_tiles_per_core_group_2: " << num_tiles_per_core_group_2 << std::endl;
    // std::cout << "core_group_1: " << core_group_1.num_cores() << std::endl;
    // std::cout << "core_group_2: " << core_group_2.num_cores() << std::endl;

    // std::cout << "num_cores_total: " << num_cores_total << std::endl;

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

        uint32_t num_tiles_per_core = 0;

        if (i < core_group_1.num_cores()) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (i < num_cores) {
            num_tiles_per_core = num_tiles_per_core_group_2;
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
            .a_tensor_base_addr = a_tensor.buffer()->address(),
            .b_tensor_base_addr = b_tensor.buffer()->address(),
            .num_tiles = num_tiles_per_core,
            .tile_ofs = tile_ofs};

        ElemwiseComputeKernelArgs compute_kern_args = {.num_tiles = num_tiles_per_core, .tile_ofs = tile_ofs};

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

template <bool initialize_args>
inline void set_elt_nd_sharded_add_runtime_args(
    tt::tt_metal::Program& program,
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle compute_kernel_id,
    const CoreRangeSet& all_device_cores,
    const std::vector<CoreCoord>* ordered_cores_opt = nullptr) {
    using namespace ttnn::kernel::eltwise::add_nd_sharded_args;
    using namespace ttnn::kernel_utils;
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    auto fallback_cores = corerange_to_cores(all_device_cores, {}, true);
    const bool use_ordered =
        ordered_cores_opt && ordered_cores_opt->size() == fallback_cores.size();
    const std::vector<CoreCoord> cores = use_ordered ? *ordered_cores_opt : fallback_cores;
    auto distribution_spec = a_tensor.tensor_spec().compute_buffer_sharding_args().buffer_distribution_spec();
    TT_FATAL(distribution_spec.has_value(), "Buffer distribution spec must be set");

    std::vector<std::vector<uint32_t>> reader_args_array{
        cores.size(), std::vector<uint32_t>{amount_of_fields<ElemwiseReaderKernelArgs>(), 0}};
    std::vector<std::vector<uint32_t>> compute_args_array{
        cores.size(),
        std::vector<uint32_t>(amount_of_fields<ttnn::kernel::eltwise::add_args::ElemwiseComputeKernelArgs>(), 0)};
    std::vector<std::vector<uint32_t>> writer_args_array{
        cores.size(), std::vector<uint32_t>(amount_of_fields<ElemwiseWriterKernelArgs>(), 0)};

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    // const auto num_tiles_per_shard_width = distribution_spec->num_tiles_per_shard() / tt::constants::TILE_WIDTH;
    const auto num_tiles_per_shard_width = a_tensor.shard_spec()->shape[-1] / tt::constants::TILE_WIDTH;
    const auto num_tiles_per_shard_height = a_tensor.shard_spec()->shape[-2] / tt::constants::TILE_HEIGHT;
    const auto num_tiles_per_shard = num_tiles_per_shard_height * num_tiles_per_shard_width;

    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        const CoreCoord& core = cores[core_id];
        // Worker i from get_optimal_dram_bank_to_logical_worker_assignment is optimal for DRAM bank i.
        // Buffer shard k is on buffer core at index k (same order as allocation).
        const uint32_t start_shard_id = core_id;

        // Reader run-time args
        ElemwiseReaderKernelArgs reader_runtime_args = {
            .a_tensor_base_addr = a_tensor.buffer()->address(),
            .b_tensor_base_addr = b_tensor.buffer()->address(),
            .num_shards = distribution_spec->num_shards_per_core(core_id),
            .shard_id = start_shard_id,
            .next_shard_offset = static_cast<uint32_t>(cores.size())};

        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, ttnn::kernel_utils::to_vector(reader_runtime_args));

        // Compute processes all tiles for all shards this core handles
        const uint32_t num_shards_this_core = distribution_spec->num_shards_per_core(core_id);
        const uint32_t total_tiles_this_core = static_cast<uint32_t>(num_tiles_per_shard) * num_shards_this_core;
        ttnn::kernel::eltwise::add_args::ElemwiseComputeKernelArgs compute_kern_args = {
            .num_tiles = total_tiles_this_core, .tile_ofs = 0};

        const uint32_t num_cycles_per_shard =
            static_cast<uint32_t>(num_tiles_per_shard) / num_tiles_per_shard_width;  // tiles per shard / tiles per cycle
        ElemwiseWriterKernelArgs write_kern_args = {
            .dst_base_addr = output.buffer()->address(),
            .num_shards = num_shards_this_core,
            .shard_id = start_shard_id,
            .next_shard_offset = static_cast<uint32_t>(cores.size()),
            .num_cycles_per_shard = num_cycles_per_shard};

        reader_args_array[core_id] = to_vector(reader_runtime_args);
        compute_args_array[core_id] = to_vector(compute_kern_args);
        writer_args_array[core_id] = to_vector(write_kern_args);
        if constexpr (!initialize_args) {
            auto& core_reader_args = cached_reader_args.at(core.x).at(core.y);
            std::ranges::copy(reader_args_array[core_id], core_reader_args.data());

            auto& core_eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            std::ranges::copy(compute_args_array[core_id], core_eltwise_args.data());

            auto& core_writer_args = cached_writer_args.at(core.x).at(core.y);
            std::ranges::copy(writer_args_array[core_id], core_writer_args.data());
        }
    }

    if constexpr (initialize_args) {
        SetRuntimeArgs(program, reader_kernel_id, cores, reader_args_array);
        SetRuntimeArgs(program, compute_kernel_id, cores, compute_args_array);
        SetRuntimeArgs(program, writer_kernel_id, cores, writer_args_array);
    }
}

}  // namespace ttnn::experimental::prim
