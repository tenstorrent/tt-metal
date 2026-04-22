// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/add/device/dram_optimized/kernels/elemwise_args_kernel.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <optional>
#include <unordered_map>

namespace ttnn::experimental::prim {

/// Reorder cores so that they follow optimal DRAM bank assignment (core i is optimal for bank i).
/// When device supports get_optimal_dram_bank_to_logical_worker_assignment, cores are sorted by
/// their position in that list to improve DRAM locality.
inline std::vector<tt::tt_metal::CoreCoord> order_cores_by_optimal_dram(
    std::vector<tt::tt_metal::CoreCoord> cores, tt::tt_metal::IDevice* device, uint8_t noc = 0) {
    if (cores.empty()) {
        return cores;
    }
    auto* mesh_device = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
    if (mesh_device == nullptr) {
        return cores;
    }
    std::vector<tt::tt_metal::CoreCoord> optimal_ordered =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(static_cast<tt::tt_metal::NOC>(noc));
    std::unordered_map<uint32_t, uint32_t> core_to_priority;
    for (uint32_t i = 0; i < optimal_ordered.size(); ++i) {
        const auto& c = optimal_ordered[i];
        core_to_priority[c.x + (c.y << 16)] = i;
    }
    std::sort(
        cores.begin(),
        cores.end(),
        [&core_to_priority](const tt::tt_metal::CoreCoord& a, const tt::tt_metal::CoreCoord& b) {
            uint32_t key_a = a.x + (a.y << 16);
            uint32_t key_b = b.x + (b.y << 16);
            uint32_t pri_a = core_to_priority.count(key_a) ? core_to_priority.at(key_a) : 0xFFFFFFFFu;
            uint32_t pri_b = core_to_priority.count(key_b) ? core_to_priority.at(key_b) : 0xFFFFFFFFu;
            return pri_a < pri_b;
        });
    return cores;
}

template <bool initialize_args>
inline void set_eltwise_binary_runtime_args_for_dram_cores(
    tt::tt_metal::Program& program,
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle compute_kernel_id,
    const CoreRangeSet& all_device_cores) {
    using namespace eltwise_dram_optimized;
    using namespace ttnn::kernel_utils;
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    uint32_t num_tiles = static_cast<uint32_t>(a_tensor.physical_volume() / TILE_HW);

    bool row_major = true;  // TODO: make this configurable
    uint32_t num_cores_total = all_device_cores.num_cores();

    TT_FATAL(
        a_tensor.logical_shape()[-1] % tt::constants::TILE_HEIGHT == 0,
        "num_tiles mismatch, {} % {} != 0",
        a_tensor.logical_shape()[-1],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        a_tensor.logical_shape()[-2] % tt::constants::TILE_WIDTH == 0,
        "num_tiles mismatch, {} % {} != 0",
        a_tensor.logical_shape()[-2],
        tt::constants::TILE_WIDTH);

    // vector of cores
    auto cores = corerange_to_cores(all_device_cores, std::nullopt, row_major);

    std::vector<std::vector<uint32_t>> reader_args_array{cores.size()};
    std::vector<std::vector<uint32_t>> compute_args_array{cores.size()};
    std::vector<std::vector<uint32_t>> writer_args_array{cores.size()};

    auto& cached_reader_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& cached_eltwise_args = GetRuntimeArgs(program, compute_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, writer_kernel_id);

    std::vector<uint32_t> core_ids;
    for (uint32_t core_id = 0; core_id < num_cores_total; ++core_id) {
        const CoreCoord& core = cores.at(core_id);

        uint32_t num_tiles_per_core = num_tiles / num_cores_total + (core_id < num_tiles % num_cores_total ? 1 : 0);

        if constexpr (!initialize_args) {
            // RuntimeArgsData
            auto& reader_args = cached_reader_args.at(core.x).at(core.y);
            reader_args[2] = 0;
            auto& eltwise_args = cached_eltwise_args.at(core.x).at(core.y);
            eltwise_args[0] = 0;
            auto& writer_args = cached_writer_args.at(core.x).at(core.y);
            writer_args[1] = 0;
        }
        // std::cout << "Assigning runtime args to core: " << core.str() << " num_tiles_per_core: " <<
        // num_tiles_per_core
        //           << " tile_ofs: " << core_id << std::endl;

        uint32_t vc = core_id & 0x3;
        core_ids.push_back(core_id);
        for (uint32_t j = 0; j < core_id; ++j) {
            auto core_ = cores[j];

            if (core_.y == core.y and ((core_id & 0x3) == (core_ids[j] & 0x3))) {  // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        EltwiseReaderArgs read_kern_args = {
            .a_tensor_base_addr = a_tensor.buffer()->address(),
            .b_tensor_base_addr = b_tensor.buffer()->address(),
            .tile_ofs = core_id,
            .num_tiles = num_tiles_per_core,
            .vc = vc};

        EltwiseComputeArgs compute_kern_args = {.num_tiles = num_tiles_per_core, .vc = vc};

        EltwiseWriterArgs write_kern_args = {
            .dst_base_addr = output.buffer()->address(),
            .tile_ofs = core_id,
            .num_tiles = num_tiles_per_core,
            .vc = vc};

        reader_args_array[core_id] = to_vector(read_kern_args);
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

template <bool initialize_args>
inline void set_eltwise_binary_runtime_args_across_all_cores(
    tt::tt_metal::Program& program,
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const Tensor& output,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const tt::tt_metal::KernelHandle compute_kernel_id,
    const CoreRangeSet& all_device_cores) {
    using namespace eltwise_dram_optimized;
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

    std::vector<std::vector<uint32_t>> reader_args_array{
        cores.size(), std::vector<uint32_t>{amount_of_fields<EltwiseReaderArgs>(), 0}};
    std::vector<std::vector<uint32_t>> compute_args_array{
        cores.size(), std::vector<uint32_t>(amount_of_fields<EltwiseComputeArgs>(), 0)};
    std::vector<std::vector<uint32_t>> writer_args_array{
        cores.size(), std::vector<uint32_t>(amount_of_fields<EltwiseWriterArgs>(), 0)};

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
        // std::cout << "Assigning runtime args to core: " << core.str() << " num_tiles_per_core: " <<
        // num_tiles_per_core
        //           << " tile_ofs: " << tile_ofs << std::endl;

        EltwiseReaderArgs read_kern_args = {
            .a_tensor_base_addr = a_tensor.buffer()->address(),
            .b_tensor_base_addr = b_tensor.buffer()->address(),
            .tile_ofs = tile_ofs,
            .num_tiles = num_tiles_per_core,
        };
        EltwiseComputeArgs compute_kern_args = {.num_tiles = num_tiles_per_core};

        EltwiseWriterArgs write_kern_args = {
            .dst_base_addr = output.buffer()->address(),
            .tile_ofs = tile_ofs,
            .num_tiles = num_tiles_per_core,
        };

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
