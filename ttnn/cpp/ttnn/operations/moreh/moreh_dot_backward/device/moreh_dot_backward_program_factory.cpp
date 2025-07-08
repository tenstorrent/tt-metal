// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_backward_device_operation.hpp"
#include <tt-metalium/util.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {
MorehDotBackwardOperation::SingleCore::cached_program_t MorehDotBackwardOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;
    const auto& input_grad = tensor_return_value.at(0);
    const auto& other_grad = tensor_return_value.at(1);
    Program program{};
    CoreCoord core = {0, 0};
    const uint32_t core_num = 1;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    auto* src0_buffer = output_grad.buffer();
    auto* src1_buffer = input.buffer();
    auto* src2_buffer = other.buffer();

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    float scaler = 1.0f;
    const auto& a_shape_wo_padding = input.logical_shape();
    uint32_t pad_h = a_shape_wo_padding[2] % tt::constants::TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % tt::constants::TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (tt::constants::TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (tt::constants::TILE_WIDTH) : (pad_w);

    IDevice* device = input.device();

    const uint32_t in0_t = 2;
    const uint32_t in1_t = 2;
    const uint32_t in2_t = 2;
    const uint32_t out0_t = 2;
    const uint32_t out1_t = 2;

    CreateCircularBuffer(
        program,
        std::set<CoreRange>{CoreRange(core, core)},
        cb_data_format,
        {
            {CBIndex::c_0, in0_t},
            {CBIndex::c_1, in1_t},
            {CBIndex::c_2, in2_t},
            {CBIndex::c_16, out0_t},
            {CBIndex::c_17, out1_t},
        });
    bool has_input_grad = input_grad.has_value();
    bool has_other_grad = other_grad.has_value();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)is_dram(src0_buffer), (std::uint32_t)is_dram(src1_buffer), (std::uint32_t)is_dram(src2_buffer)};

    bool dst0_is_dram = false;
    bool dst1_is_dram = false;
    uint32_t dst0_address = 0;
    uint32_t dst1_address = 0;

    if (has_input_grad) {
        const auto& input_grad_tensor = input_grad.value();
        auto* dst0_buffer = input_grad_tensor.buffer();
        TT_ASSERT(dst0_buffer != nullptr, "input_grad buffer should be allocated on device!");
        dst0_is_dram = is_dram(dst0_buffer);
        dst0_address = dst0_buffer->address();
    }

    if (has_other_grad) {
        const auto& other_grad_tensor = other_grad.value();
        auto* dst1_buffer = other_grad_tensor.buffer();
        TT_ASSERT(dst1_buffer != nullptr, "other_grad buffer should be allocated on device!");
        dst1_is_dram = is_dram(dst1_buffer);
        dst1_address = dst1_buffer->address();
    }

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)CBIndex::c_16,
        (std::uint32_t)CBIndex::c_17,
        (std::uint32_t)dst0_is_dram,
        (std::uint32_t)dst1_is_dram,
    };

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/reader_moreh_dot_backward.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/writer_moreh_dot_backward.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, core, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, core, writer_compile_time_args);

    std::vector<uint32_t> compute_kernel_args = {};
    std::map<std::string, std::string> compute_defines;

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/moreh_dot_backward.cpp";
    const auto compute_kernel_id =
        CreateComputeKernel(program, compute_kernel_file, {core, core_num, compute_kernel_args}, compute_defines);

    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {(std::uint32_t)has_input_grad,
         (std::uint32_t)has_other_grad,
         src0_buffer->address(),
         src1_buffer->address(),
         src2_buffer->address(),
         num_tiles,
         0});

    SetRuntimeArgs(
        program, compute_kernel_id, core, {(std::uint32_t)has_input_grad, (std::uint32_t)has_other_grad, num_tiles});

    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {(std::uint32_t)has_input_grad, (std::uint32_t)has_other_grad, dst0_address, dst1_address, num_tiles, 0});

    return {
        std::move(program), {.unary_reader_kernel_id = reader_kernel_id, .unary_writer_kernel_id = writer_kernel_id}};
}

void MorehDotBackwardOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& output_grad_buffer = tensor_args.output_grad.buffer();
    const auto& input_buffer = tensor_args.input.buffer();
    const auto& other_buffer = tensor_args.other.buffer();
    const auto input_grad_buffer = tensor_return_value.at(0);
    const auto other_grad_buffer = tensor_return_value.at(1);

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, CoreCoord{0, 0});
        runtime_args[2] = output_grad_buffer->address();
        runtime_args[3] = input_buffer->address();
        runtime_args[4] = other_buffer->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, CoreCoord{0, 0});
        if (input_grad_buffer.has_value()) {
            runtime_args[2] = input_grad_buffer.value().buffer()->address();
        }
        if (other_grad_buffer.has_value()) {
            runtime_args[3] = other_grad_buffer.value().buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::moreh::moreh_dot_backward
