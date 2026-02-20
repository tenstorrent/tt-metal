// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_multi_program_factory.hpp"

#include <algorithm>
#include <numeric>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

namespace {

uint32_t find_greatest_common_page_size(std::vector<uint32_t>& stick_sizes, uint32_t alignment) {
    TT_FATAL(!stick_sizes.empty(), "Need at least one stick size to find page size");
    uint32_t page_size = tt::align(stick_sizes[0], alignment);
    for (size_t idx = 1; idx < stick_sizes.size(); idx++) {
        const uint32_t padded_stick_size = tt::align(stick_sizes[idx], alignment);
        page_size = std::gcd(page_size, padded_stick_size);
    }
    return page_size;
}

}  // namespace

ConcatS2SMultiProgramFactory::cached_program_t ConcatS2SMultiProgramFactory::create(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    Tensor& output = tensor_return_value;
    TT_FATAL(dim == 2 || dim == 3, "Sharded concat only supports dim=2 or 3");
    const bool is_height_concat = dim == 2;

    Program program = CreateProgram();

    const uint32_t num_input_tensors = input_tensors.size();
    const uint32_t cb_dst_id = 16;
    TT_FATAL(num_input_tensors <= cb_dst_id, "Not enough circular buffer for {} inputs.", num_input_tensors);
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    // Assume inputs and output have the same element size and alignment.
    const uint32_t element_size = input_tensors[0].element_size();
    const uint32_t alignment = input_tensors[0].buffer()->alignment();

    uint32_t page_size;
    uint32_t elements_per_page_width;
    uint32_t elements_per_page_height;
    if (rm_layout) {
        std::vector<uint32_t> all_stick_sizes;
        all_stick_sizes.reserve(input_tensors.size() + 1);
        all_stick_sizes.push_back(output.shard_spec().value().shape[1]);
        std::transform(
            input_tensors.begin(), input_tensors.end(), std::back_inserter(all_stick_sizes), [](const Tensor& tensor) {
                return tensor.element_size() * tensor.shard_spec().value().shape[1];
            });
        page_size = find_greatest_common_page_size(all_stick_sizes, alignment);
        elements_per_page_width = page_size / element_size;
        elements_per_page_height = 1;
    } else {
        page_size = tt::tile_size(cb_data_format);
        elements_per_page_width = TILE_WIDTH;
        elements_per_page_height = TILE_HEIGHT;
    }

    std::vector<CBHandle> cb_inputs(num_input_tensors);
    std::vector<uint32_t> input_num_pages_per_stick(num_input_tensors);
    std::vector<uint32_t> input_num_sticks(num_input_tensors);
    std::vector<uint32_t> input_write_offsets(num_input_tensors);

    // Assume inputs and output have the same sharding grid.
    const auto all_cores = input_tensors[0].shard_spec().value().grid;

    // Input CBs
    uint32_t curr_input_write_offset = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_pages_per_stick[input_id] = tt::div_up(shard_spec.shape[1], elements_per_page_width);
        input_num_sticks[input_id] = tt::div_up(shard_spec.shape[0], elements_per_page_height);
        input_write_offsets[input_id] = curr_input_write_offset;

        const uint32_t input_num_pages = input_num_pages_per_stick[input_id] * input_num_sticks[input_id];
        const CircularBufferConfig input_cb_config =
            CircularBufferConfig(page_size * input_num_pages, {{input_id, cb_data_format}})
                .set_page_size(input_id, page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_inputs[input_id] = CreateCircularBuffer(program, all_cores, input_cb_config);

        curr_input_write_offset +=
            page_size * (is_height_concat ? input_num_pages : input_num_pages_per_stick[input_id]);
    }

    // Output CB
    const auto output_shard_spec = output.shard_spec().value();
    const uint32_t output_num_pages_per_stick = tt::div_up(output_shard_spec.shape[1], elements_per_page_width);
    const uint32_t output_num_sticks = tt::div_up(output_shard_spec.shape[0], elements_per_page_height);
    const CircularBufferConfig output_cb_config =
        CircularBufferConfig(page_size * output_num_sticks * output_num_pages_per_stick, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = CreateCircularBuffer(program, all_cores, output_cb_config);

    const uint32_t output_stride = page_size * output_num_pages_per_stick;
    const std::vector<uint32_t> compile_time_args = {cb_dst_id, page_size, output_stride, num_input_tensors};

    std::vector<uint32_t> runtime_args_0;
    std::vector<uint32_t> runtime_args_1;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto input_num_sticks_per_risc = tt::div_up(input_num_sticks[input_id], 2);
        runtime_args_0.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_0.push_back(input_num_sticks_per_risc);
        runtime_args_0.push_back(input_write_offsets[input_id]);
        runtime_args_0.push_back(0);
        runtime_args_1.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_1.push_back(input_num_sticks[input_id] - input_num_sticks_per_risc);
        runtime_args_1.push_back(input_write_offsets[input_id] + (output_stride * input_num_sticks_per_risc));
        runtime_args_1.push_back(page_size * input_num_pages_per_stick[input_id] * input_num_sticks_per_risc);
    }

    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp",
        all_cores,
        ReaderDataMovementConfig(compile_time_args));

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp",
        all_cores,
        WriterDataMovementConfig(compile_time_args));

    SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, runtime_args_0);
    SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, runtime_args_1);

    return {
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .cb_inputs = cb_inputs,
         .cb_output = cb_output,
         .all_cores = all_cores}};
}

void ConcatS2SMultiProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    for (uint32_t input_id = 0; input_id < shared_vars.num_input_tensors; input_id++) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_vars.cb_inputs[input_id], *tensor_args.input_tensors[input_id].buffer());
    }
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, shared_vars.cb_output, *tensor_return_value.buffer());
}

}  // namespace ttnn::prim
