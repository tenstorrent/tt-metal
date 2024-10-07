// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"

namespace ttnn::operations::data_movement::clone {
CloneOperation::ProgramFactory::cached_program_t CloneOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::detail;

    const auto& input = tensor_args.input;
    Program program = Program();

    bool tilized = output.get_layout() == Layout::TILE;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.get_dtype());
    bool convert_dtype = input_cb_data_format != output_cb_data_format;
    uint32_t input_unit_size =
        tilized ? TileSize(input_cb_data_format) : input.get_legacy_shape()[-1] * input.element_size();
    uint32_t output_unit_size =
        tilized ? TileSize(output_cb_data_format) : output.get_legacy_shape()[-1] * output.element_size();

    uint32_t num_units =
        tilized ? output.volume() / constants::TILE_HW : output.volume() / output.get_legacy_shape()[-1];

    Device* device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_units);

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_units = 2;
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    auto cb_src0_config =
        CircularBufferConfig(num_input_units * aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;
    if (convert_dtype) {
        output_cb_index = CB::c_out0;
        uint32_t num_output_units = 2;
        uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
        auto output_cb_config =
            CircularBufferConfig(
                num_output_units * aligned_output_unit_size, {{output_cb_index, output_cb_data_format}})
                .set_page_size(output_cb_index, aligned_output_unit_size);
        auto cb_output = CreateCircularBuffer(program, all_cores, output_cb_config);
    }

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;

    vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        reader_compile_time_args = {(uint32_t)src_is_dram};
        writer_compile_time_args = {(uint32_t)output_cb_index, (uint32_t)dst_is_dram};
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_unit_size);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (uint32_t)log2(input_unit_size) : 0;
        reader_compile_time_args = {
            (uint32_t)src0_cb_index,
            (uint32_t)src_is_dram,
            (uint32_t)src_stick_size_is_power_of_two,
            (uint32_t)src_log2_stick_size};
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_unit_size);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (uint32_t)log2(output_unit_size) : 0;
        writer_compile_time_args = {
            (uint32_t)output_cb_index,
            (uint32_t)dst_is_dram,
            (uint32_t)dst_stick_size_is_power_of_two,
            (uint32_t)dst_log2_stick_size};
    }
    map<string, string> kernel_defines;
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    if (convert_dtype) {
        vector<uint32_t> compute_kernel_args_group_1 = {num_units_per_core_group_1};
        auto eltwise_unary_kernel_group_1 = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp",
            core_group_1,
            ComputeConfig{.compile_args = compute_kernel_args_group_1});

        if (!core_group_2.ranges().empty()) {
            vector<uint32_t> compute_kernel_args_group_2 = {num_units_per_core_group_2};
            auto eltwise_unary_kernel_group_2 = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp",
                core_group_2,
                ComputeConfig{.compile_args = compute_kernel_args_group_2});
        }
    }

    uint32_t start_id = 0;
    uint32_t g1_numcores = core_group_1.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        if (tilized) {
            SetRuntimeArgs(
                program, unary_reader_kernel_id, core, {src_buffer->address(), num_units_per_core, start_id});
            SetRuntimeArgs(
                program, unary_writer_kernel_id, core, {dst_buffer->address(), num_units_per_core, start_id});
        } else {
            SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {src_buffer->address(), input_unit_size, num_units_per_core, start_id});
            SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {dst_buffer->address(), output_unit_size, num_units_per_core, start_id});
        }
        start_id += num_units_per_core;
    }

    return {std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cores}};
}

void CloneOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& program = cached_program.program;
    const auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    const auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto cores = cached_program.shared_variables.cores;

    auto src_buffer_address = tensor_args.input.buffer()->address();
    auto dst_buffer_address = output.buffer()->address();
    for (const auto& core : cores) {
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer_address;
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer_address;
        }
    }
}
}  // namespace ttnn::operations::data_movement::clone
