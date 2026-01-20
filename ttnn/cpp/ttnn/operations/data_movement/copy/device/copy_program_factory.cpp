// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <tt-metalium/work_split.hpp>
#include "copy_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::prim {

CopyProgramFactory::cached_program_t CopyProgramFactory::create(
    const CopyParams& operation_attributes, const CopyInputs& tensor_args, Tensor& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const bool backwards = operation_attributes.backwards;
    Program program{};

    const bool tilized = output.layout() == Layout::TILE;
    const bool sharded = input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = tilized ? tt::tile_size(input_cb_data_format)
                                       : input.padded_shape()[-1] * input.element_size();
    const uint32_t full_input_row = input_unit_size;
    if (sharded && !tilized) {
        input_unit_size = input.memory_config().shard_spec()->shape[1] * input.element_size();
    }
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size = tilized ? tt::tile_size(output_cb_data_format)
                                        : output.padded_shape()[-1] * output.element_size();
    const uint32_t full_output_row = output_unit_size;
    if (sharded && !tilized) {
        output_unit_size = output.memory_config().shard_spec()->shape[1] * output.element_size();
    }

    const bool convert_dtype = input_cb_data_format != output_cb_data_format;

    const uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.padded_shape()[-1];

    IDevice* device = output.device();

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, num_units);

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t num_input_units = 2;
    const uint32_t input_alignment = input.buffer()->alignment();
    const uint32_t aligned_input_unit_size = tt::align(input_unit_size, input_alignment);
    const CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_units * aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, aligned_input_unit_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;  // same as input cb
    if (convert_dtype) {
        output_cb_index = tt::CBIndex::c_16;
        const uint32_t num_output_units = 2;
        const uint32_t output_alignment = output.buffer()->alignment();
        const uint32_t aligned_output_unit_size = tt::align(output_unit_size, output_alignment);
        const CircularBufferConfig output_cb_config =
            CircularBufferConfig(
                num_output_units * aligned_output_unit_size, {{output_cb_index, output_cb_data_format}})
                .set_page_size(output_cb_index, aligned_output_unit_size);
        CreateCircularBuffer(program, all_cores, output_cb_config);
    }

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        writer_compile_time_args = {(std::uint32_t)output_cb_index};
    } else {
        reader_compile_time_args = {(std::uint32_t)src0_cb_index, (std::uint32_t)input_unit_size};
        writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)output_unit_size};
    }
    std::map<std::string, std::string> kernel_defines;
    if (sharded) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input, writer_compile_time_args);
        shard_builder::extend_sharding_compile_time_args(input, reader_compile_time_args);
    } else {
        TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    }
    if (backwards) {
        kernel_defines["BACKWARDS"] = "1";
    }
    const std::string reader_rm_path =
        sharded ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_stick_start_id.cpp"
                : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp";
    const KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_start_id.cpp"
                : reader_rm_path,
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    const std::string writer_rm_path =
        sharded ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_stick_start_id.cpp"
                : "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
    const KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp"
                : writer_rm_path,
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    if (convert_dtype) {
        const std::vector<uint32_t> compute_kernel_args_group_1 = {num_units_per_core_group_1};
        CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp",
            core_group_1,
            ComputeConfig{.compile_args = compute_kernel_args_group_1});

        if (!core_group_2.ranges().empty()) {
            const std::vector<uint32_t> compute_kernel_args_group_2 = {num_units_per_core_group_2};
            CreateKernel(
                program,
                "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/eltwise_copy.cpp",
                core_group_2,
                ComputeConfig{.compile_args = compute_kernel_args_group_2});
        }
    }

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
            std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), num_units_per_core, start_id};
            std::vector<uint32_t> writer_runtime_args = {dst_buffer->address(), num_units_per_core, start_id};
            if (sharded) {
                shard_builder::extend_sharding_run_time_args(input, reader_runtime_args);
                shard_builder::extend_sharding_run_time_args(input, writer_runtime_args);
            }
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
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

void CopyProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const CopyParams& /*operation_attributes*/,
    const CopyInputs& tensor_args,
    Tensor& output) {
    using namespace tt::tt_metal;

    Program& program = cached_program.program;
    const KernelHandle unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    const KernelHandle unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const std::vector<CoreCoord>& cores = cached_program.shared_variables.cores;

    Buffer* const src_buffer = tensor_args.input.buffer();
    Buffer* const dst_buffer = output.buffer();

    for (const CoreCoord& core : cores) {
        {
            RuntimeArgsData& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            RuntimeArgsData& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
