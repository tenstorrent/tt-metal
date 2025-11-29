// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_program_factory.hpp"

#include <math.h>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::data_movement::move::program {

MoveProgramFactory::cached_program_t MoveProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::constants;

    const Tensor& input = tensor_args.input_tensor;
    const tensor_return_value_t& output = tensor_return_value;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    MoveProgramFactory::shared_variables_t shared_vars{};

    const bool tilized = output.layout() == Layout::TILE;
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t unit_size =
        tilized ? tt::tile_size(cb_data_format) : output.padded_shape()[-1] * output.element_size();

    const uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.padded_shape()[-1];

    const auto compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_units);

    shared_vars.num_cores = num_cores;
    shared_vars.num_cores_y = num_cores_y;

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t num_input_units = 2;
    const uint32_t aligned_unit_size = round_up_to_mul32(unit_size);

    const tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * aligned_unit_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, aligned_unit_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        writer_compile_time_args = {(std::uint32_t)src0_cb_index};
    } else {
        reader_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)unit_size};
        writer_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)unit_size};
    }

    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    const std::string reader_rm_path =
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_start_id.cpp"
                : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_rm_path, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    const std::string writer_rm_path =
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp"
                : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_rm_path, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    shared_vars.reader_kernel_id = unary_reader_kernel_id;
    shared_vars.writer_kernel_id = unary_writer_kernel_id;

    uint32_t start_id = 0;
    const uint32_t g1_numcores = core_group_1.num_cores();
    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        if (tilized) {
            std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), num_units_per_core, start_id};
            std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), num_units_per_core, start_id};
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        } else {
            uint32_t full_row = unit_size;
            std::vector<uint32_t> reader_runtime_args = {
                src_buffer->address(), unit_size, num_units_per_core, start_id, full_row / unit_size};
            std::vector<uint32_t> writer_runtime_args = {
                dst_buffer->address(), unit_size, num_units_per_core, start_id, full_row / unit_size};
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        }

        start_id += num_units_per_core;
    }

    return {std::move(program), std::move(shared_vars)};
}

void MoveProgramFactory::override_runtime_arguments(
    MoveProgramFactory::cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const Tensor& input = tensor_args.input_tensor;
    const tensor_return_value_t& output = tensor_return_value;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;
    const tt::tt_metal::KernelHandle reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const tt::tt_metal::KernelHandle writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    const CoreCoord compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (const CoreCoord& core : cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

        // Update buffer addresses (first argument for both reader and writer)
        reader_runtime_args[0] = src_buffer->address();
        writer_runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement::move::program
