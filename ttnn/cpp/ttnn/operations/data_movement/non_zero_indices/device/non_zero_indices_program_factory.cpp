// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/non_zero_indices/device/non_zero_indices_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

NonZeroIndicesProgramFactory::cached_program_t NonZeroIndicesProgramFactory::create(
    const NonzeroParams& /*operation_attributes*/, const NonzeroInputs& tensor_args, NonzeroResult& output_tensors) {
    const auto& input = tensor_args.input;
    const auto& out_num_indices = std::get<0>(output_tensors);
    const auto& out_indices = std::get<1>(output_tensors);

    tt::tt_metal::Program program{};

    uint32_t alignment_base = 32 / input.element_size();
    // we want per core to be aligned to aligment_base per core

    uint32_t aligned_elements = tt::div_up(input.padded_shape()[-1], alignment_base) * alignment_base;
    uint32_t actual_elements = input.padded_shape()[-1];

    CoreCoord core = {0, 0};

    uint32_t input_cb_index = 0;
    uint32_t output_cb_index_0 = 1;
    uint32_t output_cb_index_1 = 2;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::UINT32);

    uint32_t page_size = actual_elements * input.element_size();
    uint32_t rounded_page_size = round_up_to_mul32(page_size);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(2 * rounded_page_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, rounded_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(2 * 32, {{output_cb_index_0, output_cb_data_format}})
            .set_page_size(output_cb_index_0, 32);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);

    uint32_t dst_page_size = actual_elements * 4;
    uint32_t dst_rounded_page_size = round_up_to_mul32(dst_page_size);
    tt::tt_metal::CircularBufferConfig cb_dst1_config =
        tt::tt_metal::CircularBufferConfig(2 * dst_rounded_page_size, {{output_cb_index_1, output_cb_data_format}})
            .set_page_size(output_cb_index_1, dst_rounded_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_dst1_config);

    std::map<std::string, std::string> defines;
    defines["NUM_BYTES"] = std::to_string(input.element_size());

    // Create Kernel
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)output_cb_index_0,
        (std::uint32_t)output_cb_index_1,
    };
    TensorAccessorArgs(*input.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*out_num_indices.buffer()).append_to(compile_time_args);
    TensorAccessorArgs(*out_indices.buffer()).append_to(compile_time_args);

    const std::array run_time_args = {
        (std::uint32_t)input.buffer()->address(),
        (std::uint32_t)out_num_indices.buffer()->address(),
        (std::uint32_t)out_indices.buffer()->address(),
        (std::uint32_t)aligned_elements,
        (std::uint32_t)actual_elements,
        (std::uint32_t)input.element_size()};

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/non_zero_indices/device/kernels/dataflow/"
        "non_zero_indices_sc_reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, run_time_args);

    return cached_program_t{std::move(program), {kernel_id, core, page_size}};
}

void NonZeroIndicesProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const NonzeroParams& /*operation_attributes*/,
    const NonzeroInputs& tensor_args,
    NonzeroResult& output_tensors) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    auto& kernel_id = shared_vars.kernel_id;
    auto& core = shared_vars.core;

    const auto& input = tensor_args.input;
    const auto& out_num_indices = std::get<0>(output_tensors);
    const auto& out_indices = std::get<1>(output_tensors);

    uint32_t alignment_base = 32 / input.element_size();
    uint32_t aligned_elements = tt::div_up(input.padded_shape()[-1], alignment_base) * alignment_base;
    uint32_t actual_elements = input.padded_shape()[-1];
    auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_id, core);
    runtime_args[0] = input.buffer()->address();
    runtime_args[1] = out_num_indices.buffer()->address();
    runtime_args[2] = out_indices.buffer()->address();
    runtime_args[3] = aligned_elements;
    runtime_args[4] = actual_elements;
    runtime_args[5] = input.element_size();
}

}  // namespace ttnn::prim
