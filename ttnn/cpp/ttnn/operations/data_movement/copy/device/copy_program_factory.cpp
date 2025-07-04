// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/util.hpp>
#include "copy_device_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>
#include <algorithm>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

operation::ProgramWithCallbacks copy_multi_core(const Tensor& input, const Tensor& output, bool backwards) {
    tt::tt_metal::Program program{};

    bool tilized = output.layout() == Layout::TILE;
    bool sharded = input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = tilized ? tt::tt_metal::detail::TileSize(input_cb_data_format)
                                       : input.padded_shape()[-1] * input.element_size();
    uint32_t full_input_row = input_unit_size;
    if (sharded && !tilized) {
        input_unit_size = input.memory_config().shard_spec()->shape[1] * input.element_size();
    }
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size = tilized ? tt::tt_metal::detail::TileSize(output_cb_data_format)
                                        : output.padded_shape()[-1] * output.element_size();
    uint32_t full_output_row = output_unit_size;
    if (sharded && !tilized) {
        output_unit_size = output.memory_config().shard_spec()->shape[1] * output.element_size();
    }

    bool convert_dtype = input_cb_data_format != output_cb_data_format;

    uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.padded_shape()[-1];

    tt::tt_metal::IDevice* device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_units);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_units = 2;
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_units * aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;  // same as input cb
    if (convert_dtype) {
        output_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_units = 2;
        uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
        tt::tt_metal::CircularBufferConfig output_cb_config =
            tt::tt_metal::CircularBufferConfig(
                num_output_units * aligned_output_unit_size, {{output_cb_index, output_cb_data_format}})
                .set_page_size(output_cb_index, aligned_output_unit_size);
        auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    }

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        reader_compile_time_args = {(uint32_t)src_is_dram};
        writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_unit_size);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(input_unit_size) : 0;
        reader_compile_time_args = {
            (std::uint32_t)src0_cb_index,
            (std::uint32_t)src_is_dram,
            (std::uint32_t)src_stick_size_is_power_of_two,
            (std::uint32_t)src_log2_stick_size};
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_unit_size);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(output_unit_size) : 0;
        writer_compile_time_args = {
            (std::uint32_t)output_cb_index,
            (std::uint32_t)dst_is_dram,
            (std::uint32_t)dst_stick_size_is_power_of_two,
            (std::uint32_t)dst_log2_stick_size};
    }
    std::map<std::string, std::string> kernel_defines;
    if (sharded) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input, writer_compile_time_args);
        shard_builder::extend_sharding_compile_time_args(input, reader_compile_time_args);
    }
    if (backwards) {
        kernel_defines["BACKWARDS"] = "1";
    }
    std::string reader_rm_path =
        sharded ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_stick_start_id.cpp"
                : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_start_id.cpp"
                : reader_rm_path,
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    std::string writer_rm_path =
        sharded ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_stick_start_id.cpp"
                : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp"
                : writer_rm_path,
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    if (convert_dtype) {
        std::vector<uint32_t> compute_kernel_args_group_1 = {num_units_per_core_group_1};
        auto eltwise_unary_kernel_group_1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp",
            core_group_1,
            tt::tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_1});

        if (!core_group_2.ranges().empty()) {
            std::vector<uint32_t> compute_kernel_args_group_2 = {num_units_per_core_group_2};
            auto eltwise_unary_kernel_group_2 = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp",
                core_group_2,
                tt::tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_2});
        }
    }

    uint32_t start_id = 0;
    if (backwards) {
        start_id = num_units - 1;
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        if (tilized) {
            std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), num_units_per_core, start_id};
            std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), num_units_per_core, start_id};
            if (sharded) {
                shard_builder::extend_sharding_run_time_args(input, reader_runtime_args);
                shard_builder::extend_sharding_run_time_args(input, writer_runtime_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        } else {
            std::vector<uint32_t> reader_runtime_args = {
                src_buffer->address(), input_unit_size, num_units_per_core, start_id, full_input_row / input_unit_size};
            std::vector<uint32_t> writer_runtime_args = {
                dst_buffer->address(),
                output_unit_size,
                num_units_per_core,
                start_id,
                full_output_row / output_unit_size};
            if (sharded) {
                shard_builder::extend_sharding_run_time_args(input, reader_runtime_args);
                shard_builder::extend_sharding_run_time_args(input, writer_runtime_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        }
        if (backwards) {
            start_id -= num_units_per_core;
        } else {
            start_id += num_units_per_core;
        }
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement
