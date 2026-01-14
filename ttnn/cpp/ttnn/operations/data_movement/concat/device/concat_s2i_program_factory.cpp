// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2i_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <vector>

namespace ttnn::operations::data_movement::concat::program {

ConcatS2IProgramFactory::cached_program_t ConcatS2IProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    Tensor& output = tensor_return_value;
    Program program = CreateProgram();

    // CoreRangeSet all_cores({CoreRange(CoreCoord(0,0), compute_with_storage_grid_size)});

    const uint32_t num_output_rows = output.padded_shape()[-1];
    const uint32_t num_input_tensors = input_tensors.size();

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const CoreRangeSet all_cores = input_tensors[0].shard_spec().value().grid;

    const uint32_t input_unit_size = input_tensors[0].shard_spec().value().shape[1] * input_tensors[0].element_size();
    // input CBs
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        constexpr uint32_t input_num_units_per_shard_width = 1;
        const ShardSpec& shard_spec = input_tensors[input_id].shard_spec().value();
        const uint32_t num_input_units = shard_spec.shape[0] * input_num_units_per_shard_width;
        const uint32_t input_page_size = round_up_to_mul32(input_unit_size);
        CircularBufferConfig input_cb_config =
            CircularBufferConfig(num_input_units * input_page_size, {{input_id, cb_data_format}})
                .set_page_size(input_id, input_page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        CreateCircularBuffer(program, all_cores, input_cb_config);
    }

    std::vector<uint32_t> reader_compile_time_args = {num_input_tensors};
    std::vector<uint32_t> writer_compile_time_args = {num_input_tensors, input_unit_size};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2i_width.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/writer_s2i_width.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    const bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
    const auto input_cores = input_tensors[0].shard_spec().value().grid;
    const uint32_t num_output_rows_per_core = tt::div_up(num_output_rows, input_cores.num_cores());

    uint32_t core_id = 0;
    for (const CoreCoord& core : cores) {
        const ShardSpec& input_shard_spec = input_tensors[0].shard_spec().value();
        uint32_t curr_num_output_rows = (input_cores.contains(core)) ? num_output_rows_per_core : 0;

        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(num_input_tensors * 2);
        std::vector<uint32_t> writer_runtime_args = {
            output.buffer()->address(),
            core_id,
            curr_num_output_rows,
            num_input_tensors * input_shard_spec.shape[0],
            input_shard_spec.shape[0]};
        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            reader_runtime_args.push_back(input_id);
            reader_runtime_args.push_back(input_shard_spec.shape[0]);
            writer_runtime_args.push_back(input_id);
        }
        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        core_id++;
    }

    return {
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .reader_kernel_id = unary_reader_kernel_id,
         .writer_kernel_id = unary_writer_kernel_id,
         .all_cores = all_cores}};
}

void ConcatS2IProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    const bool row_wise = tensor_args.input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    Buffer* dst_buffer = tensor_return_value.buffer();
    auto cores = corerange_to_cores(shared_vars.all_cores, std::nullopt, row_wise);
    auto input_cores = tensor_args.input_tensors[0].shard_spec().value().grid;
    const uint32_t num_output_rows = tensor_return_value.padded_shape()[-1];
    const uint32_t num_output_rows_per_core = tt::div_up(num_output_rows, input_cores.num_cores());

    for (const CoreCoord& core : cores) {
        uint32_t curr_num_input_tensors;
        uint32_t curr_num_output_rows;
        if (input_cores.contains(core)) {
            curr_num_input_tensors = shared_vars.num_input_tensors;
            curr_num_output_rows = num_output_rows_per_core;
        } else {
            curr_num_input_tensors = 0;
            curr_num_output_rows = 0;
        }

        std::vector<uint32_t> reader_runtime_args = {curr_num_input_tensors};
        std::vector<uint32_t> writer_runtime_args = {
            dst_buffer->address(), curr_num_input_tensors, curr_num_output_rows};
        for (uint32_t input_id = 0; input_id < shared_vars.num_input_tensors; input_id++) {
            UpdateDynamicCircularBufferAddress(program, input_id, *dst_buffer);
            const ShardSpec& input_shard_spec = tensor_args.input_tensors[input_id].shard_spec().value();
            reader_runtime_args.push_back(input_id);
            reader_runtime_args.push_back(input_shard_spec.shape[1]);
            writer_runtime_args.push_back(input_id);
        }
        SetRuntimeArgs(program, shared_vars.reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, shared_vars.writer_kernel_id, core, writer_runtime_args);
    }
}

}  // namespace ttnn::operations::data_movement::concat::program
