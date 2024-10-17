// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nonzero_device_operation.hpp"

namespace ttnn::operations::nonzero {
NonzeroOperation::ProgramFactory::cached_program_t NonzeroOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    const auto& input = tensor_args.input;
    const auto& out_num_indices = outputs[0];
    const auto& out_indices = outputs[1];

    uint32_t alignment_base = 32 / input.element_size();
    // we want per core to be aligned to aligment_base per core

    uint32_t aligned_elements = tt::div_up(input.get_legacy_shape()[-1], alignment_base) * alignment_base;
    uint32_t actual_elements = input.get_legacy_shape()[-1];

    CoreCoord core = {0, 0};

    // Create program
    Program program = Program();

    // Create circular buffer
    uint32_t input_cb_index = 0;
    uint32_t output_cb_index_0 = 1;
    uint32_t output_cb_index_1 = 2;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::UINT32);
    bool src_is_dram = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram_0 = out_num_indices.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram_1 = out_indices.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t page_size = actual_elements * input.element_size();
    uint32_t rounded_page_size = round_up_to_mul32(page_size);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(2 * rounded_page_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, rounded_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(2 * 32, {{output_cb_index_0, output_cb_data_format}})
            .set_page_size(output_cb_index_0, 32);
    auto cb_dst0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);

    uint32_t dst_page_size = actual_elements * 4;
    uint32_t dst_rounded_page_size = round_up_to_mul32(dst_page_size);
    tt::tt_metal::CircularBufferConfig cb_dst1_config =
        tt::tt_metal::CircularBufferConfig(2 * dst_rounded_page_size, {{output_cb_index_1, output_cb_data_format}})
            .set_page_size(output_cb_index_1, dst_rounded_page_size);
    auto cb_dst1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_dst1_config);

    std::map<string, string> defines;
    defines["NUM_BYTES"] = std::to_string(input.element_size());

    // Create write kernel
    std::vector<uint32_t> compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)output_cb_index_0,
        (std::uint32_t)output_cb_index_1,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)out_is_dram_0,
        (std::uint32_t)out_is_dram_1,
    };

    std::vector<uint32_t> run_time_args = {
        (std::uint32_t)input.buffer()->address(),
        (std::uint32_t)out_num_indices.buffer()->address(),
        (std::uint32_t)out_indices.buffer()->address(),
        (std::uint32_t)aligned_elements,
        (std::uint32_t)actual_elements,
        (std::uint32_t)input.element_size()};

    auto kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/nonzero/device/kernels/dataflow/nonzero_sc_reader.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(compile_time_args, defines));

    // Set runtime arguments
    tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, run_time_args);

    return {std::move(program), {kernel_id}};
}

void NonzeroOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    const auto& program = cached_program.program;
    const auto& kernel_id = cached_program.shared_variables.kernel_id;

    auto output_0 = outputs[0];
    auto output_1 = outputs[1];
    auto input = tensor_args.input;
    uint32_t alignment_base = 32 / input.element_size();
    uint32_t aligned_elements = tt::div_up(input.get_legacy_shape()[-1], alignment_base) * alignment_base;
    uint32_t actual_elements = input.get_legacy_shape()[-1];
    CoreCoord core{0, 0};
    auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, kernel_id, core);
    runtime_args[0] = input.buffer()->address();
    runtime_args[1] = output_0.buffer()->address();
    runtime_args[2] = output_1.buffer()->address();
    runtime_args[3] = aligned_elements;
    runtime_args[4] = actual_elements;
    runtime_args[5] = input.element_size();
}
}  // namespace ttnn::operations::nonzero
