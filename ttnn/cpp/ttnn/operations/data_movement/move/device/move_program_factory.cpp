// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

MoveProgramFactory::cached_program_t MoveProgramFactory::create(
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& /*tensor_args*/,
    Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const MoveInputTensorSnapshot& input = operation_attributes.input_snapshot;
    Tensor& output = tensor_return_value;
    const bool backwards = operation_attributes.backwards;
    Program program{};

    const bool tilized = input.layout == Layout::TILE;
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype);
    uint32_t unit_size = tilized ? tt::tile_size(cb_data_format) : input.padded_shape[-1] * input.element_size;

    const uint32_t num_units =
        tilized ? input.physical_volume / TILE_HW : input.physical_volume / input.padded_shape[-1];

    IDevice* device = output.device();

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, num_units);

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t num_input_units = 2;
    const uint32_t alignment = input.buffer_alignment;
    const uint32_t aligned_unit_size = tt::align(unit_size, alignment);
    const CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_units * aligned_unit_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, aligned_unit_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);

    Buffer* src_buffer = input.buffer.get();
    Buffer* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        writer_compile_time_args = {(std::uint32_t)src0_cb_index};
    } else {
        reader_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)unit_size};
        writer_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)unit_size};
    }
    std::map<std::string, std::string> kernel_defines;
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    if (backwards) {
        kernel_defines["BACKWARDS"] = "1";
    }

    const KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_start_id.cpp"
                : "ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    const KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp"
                : "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    uint32_t start_id = 0;
    if (backwards) {
        start_id = num_units - 1;
    }

    const uint32_t g1_numcores = core_group_1.num_cores();
    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        if (tilized) {
            const std::vector<uint32_t> reader_runtime_args = {input.buffer_address, num_units_per_core, start_id};
            const std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), num_units_per_core, start_id};
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        } else {
            const std::vector<uint32_t> reader_runtime_args = {
                input.buffer_address, unit_size, num_units_per_core, start_id, 1};
            const std::vector<uint32_t> writer_runtime_args = {
                dst_buffer->address(), unit_size, num_units_per_core, start_id, 1};
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        }
        if (backwards) {
            start_id -= num_units_per_core;
        } else {
            start_id += num_units_per_core;
        }
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .cores = cores}};
}

void MoveProgramFactory::override_runtime_arguments(
    MoveProgramFactory::cached_program_t& cached_program,
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& /*tensor_args*/,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    Program& program = cached_program.program;
    const KernelHandle unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    const KernelHandle unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const std::vector<CoreCoord>& cores = cached_program.shared_variables.cores;

    const MoveInputTensorSnapshot& input = operation_attributes.input_snapshot;
    Buffer* const dst_buffer = tensor_return_value.buffer();

    for (const CoreCoord& core : cores) {
        {
            RuntimeArgsData& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = input.buffer_address;
        }

        {
            RuntimeArgsData& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
