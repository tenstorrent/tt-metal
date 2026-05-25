// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "iterative_topk_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk {

IterativeTopkDeviceOperation::ShardedProgramFactory::cached_program_t
IterativeTopkDeviceOperation::ShardedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    auto& output_values = tensor_return_value[0];
    auto& output_indices = tensor_return_value[1];

    auto* device = input.device();
    TT_FATAL(device != nullptr, "Device must be non-null");
    tt::tt_metal::Program program{};

    uint32_t width = input.logical_shape()[-1];
    uint32_t k = operation_attributes.k;

    const auto& shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t shard_height = shard_spec.shape[0];

    uint32_t input_page_size = input.buffer()->aligned_page_size();
    uint32_t output_values_page_size = output_values.buffer()->aligned_page_size();
    uint32_t output_indices_page_size = output_indices.buffer()->aligned_page_size();

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_out_values = tt::CBIndex::c_1;
    constexpr auto cb_out_indices = tt::CBIndex::c_2;

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    auto values_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_values.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_indices.dtype());

    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(shard_height * input_page_size, {{cb_input, input_data_format}})
            .set_page_size(cb_input, input_page_size)
            .set_globally_allocated_address(*input.buffer());
    auto cb_input_handle = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    tt::tt_metal::CircularBufferConfig cb_out_values_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height * output_values_page_size, {{cb_out_values, values_data_format}})
            .set_page_size(cb_out_values, output_values_page_size)
            .set_globally_allocated_address(*output_values.buffer());
    auto cb_out_values_handle = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_values_config);

    tt::tt_metal::CircularBufferConfig cb_out_indices_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height * output_indices_page_size, {{cb_out_indices, indices_data_format}})
            .set_page_size(cb_out_indices, output_indices_page_size)
            .set_globally_allocated_address(*output_indices.buffer());
    auto cb_out_indices_handle = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_indices_config);

    std::vector<uint32_t> reader_compile_time_args = {
        cb_input,
        shard_height,
    };

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/iterative_topk/device/kernels/dataflow/"
        "reader_iterative_topk_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        cb_input,
        cb_out_values,
        cb_out_indices,
        width,
        k,
        shard_height,
        input_page_size,
        output_values_page_size,
        output_indices_page_size,
    };

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/iterative_topk/device/kernels/dataflow/"
        "writer_iterative_topk_sharded.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    return {
        std::move(program),
        {reader_kernel_id, writer_kernel_id, cb_input_handle, cb_out_values_handle, cb_out_indices_handle, cores}};
}

void IterativeTopkDeviceOperation::ShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;

    UpdateDynamicCircularBufferAddress(
        program, cached_program.shared_variables.cb_input_handle, *tensor_args.input.buffer());
    UpdateDynamicCircularBufferAddress(
        program, cached_program.shared_variables.cb_out_values_handle, *tensor_return_value[0].buffer());
    UpdateDynamicCircularBufferAddress(
        program, cached_program.shared_variables.cb_out_indices_handle, *tensor_return_value[1].buffer());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::iterative_topk
