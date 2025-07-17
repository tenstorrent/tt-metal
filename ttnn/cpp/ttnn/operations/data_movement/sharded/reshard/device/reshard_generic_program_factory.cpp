// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_generic_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "reshard_common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks reshard_multi_core_generic(const Tensor& input, Tensor& output) {
    auto device = input.device();
    auto output_core_to_page_range_pair = get_core_page_ranges(input.buffer(), output.buffer());

    tt::tt_metal::Program program{};

    auto input_shard_spec = input.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();
    auto all_cores = output_shard_spec.grid;
    auto grid = input.buffer()->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                  : device->compute_with_storage_grid_size();
    auto input_core_type = input.buffer()->core_type();
    uint32_t dst_cb_index = 16;
    auto cores =
        corerange_to_cores(all_cores, std::nullopt, output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    uint32_t total_size, page_size, unit_size;
    auto output_shard_shape = output_shard_spec.shape;
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    if (input.layout() == Layout::TILE) {
        page_size = tt::tt_metal::detail::TileSize(data_format);
        unit_size = page_size;
        total_size = output_shard_spec.numel() / TILE_HW * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        page_size = output.padded_shape()[-1] * output.element_size();
        total_size = output_shard_shape[0] * unit_size;
    }

    tt::tt_metal::KernelHandle kernel_id_0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig({dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size}));

    tt::tt_metal::KernelHandle kernel_id_1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig({dst_cb_index, (uint32_t)grid.x, (uint32_t)grid.y, page_size}));

    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    std::vector<uint32_t> physical_core_coords;
    physical_core_coords.reserve(grid.x * grid.y);
    for (uint32_t i = 0; i < grid.x; i++) {
        auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(i, 0), input_core_type);
        physical_core_coords.push_back(physical_input_core.x);
    }
    for (uint32_t i = 0; i < grid.y; i++) {
        auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(0, i), input_core_type);
        physical_core_coords.push_back(physical_input_core.y);
    }

    for (const auto& core : cores) {
        auto page_stride_vector = output_core_to_page_range_pair.at(core);
        uint32_t num_ranges = page_stride_vector.size();
        auto runtime_args_0 = get_runtime_args_for_given_ranges(
            physical_core_coords,
            page_stride_vector,
            0,
            input.buffer()->address(),
            0,
            tt::div_up(page_stride_vector.size(), 2));
        auto output_page_offset =
            runtime_args_0[physical_core_coords.size() + 1];  // offset is equivalent to number of pages output in
                                                              // previous risc core
        tt::tt_metal::SetRuntimeArgs(program, kernel_id_0, core, runtime_args_0);
        auto runtime_args_1 = get_runtime_args_for_given_ranges(
            physical_core_coords,
            page_stride_vector,
            output_page_offset,
            input.buffer()->address(),
            tt::div_up(page_stride_vector.size(), 2),
            page_stride_vector.size());
        tt::tt_metal::SetRuntimeArgs(program, kernel_id_1, core, runtime_args_1);
    }

    auto override_runtime_arguments_callback = [kernel_id_0, kernel_id_1, cb_dst0, grid, cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors.at(0);
        const auto& output = output_tensors.at(0);
        uint32_t input_addr = input.buffer()->address();
        auto& runtime_args_0_by_core = GetRuntimeArgs(program, kernel_id_0);
        auto& runtime_args_1_by_core = GetRuntimeArgs(program, kernel_id_1);
        for (auto core : cores) {
            auto& runtime_args_0 = runtime_args_0_by_core[core.x][core.y];
            auto& runtime_args_1 = runtime_args_1_by_core[core.x][core.y];
            runtime_args_0[grid.x + grid.y] = input_addr;
            runtime_args_1[grid.x + grid.y] = input_addr;
        }
        UpdateDynamicCircularBufferAddress(program, cb_dst0, *output.buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail
