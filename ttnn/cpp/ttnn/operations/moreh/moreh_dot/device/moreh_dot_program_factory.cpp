// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::moreh::moreh_dot {
MorehDotOperation::SingleCore::cached_program_t MorehDotOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto src0_buffer = input_a.buffer();
    auto src1_buffer = input_b.buffer();
    auto dst_buffer = output.buffer();
    float scaler = 1.0f;

    Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_a.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input_a.physical_volume() / tt::constants::TILE_HW;
    const auto& a_shape_wo_padding = input_a.logical_shape();
    uint32_t pad_h = a_shape_wo_padding[2] % tt::constants::TILE_HEIGHT;
    uint32_t pad_w = a_shape_wo_padding[3] % tt::constants::TILE_WIDTH;
    uint32_t mask_h = (pad_h == 0) ? (tt::constants::TILE_HEIGHT) : (pad_h);
    uint32_t mask_w = (pad_w == 0) ? (tt::constants::TILE_WIDTH) : (pad_w);

    tt::tt_metal::IDevice* device = input_a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t in0_t = 2;   // a
    const uint32_t in1_t = 2;   // b
    const uint32_t in2_t = 1;   // scaler
    const uint32_t out0_t = 2;  // out
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    CoreCoord core = {0, 0};

    CreateCircularBuffer(
        program,
        std::set<CoreRange>{CoreRange(core, core)},
        cb_data_format,
        {
            {CBIndex::c_0, in0_t},
            {CBIndex::c_1, in1_t},
            {CBIndex::c_2, in2_t},
            {CBIndex::c_16, out0_t},
            {CBIndex::c_24, im0_t},
            {CBIndex::c_25, im1_t},
        });

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)is_dram(src0_buffer),
        (std::uint32_t)is_dram(src1_buffer),
        *reinterpret_cast<uint32_t*>(&scaler)};

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)CBIndex::c_16, (std::uint32_t)is_dram(dst_buffer)};
    const auto reader_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/reader_moreh_dot.cpp";
    const auto writer_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/writer_moreh_dot.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, core, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, core, writer_compile_time_args);

    std::vector<uint32_t> compute_kernel_args = {};
    std::map<std::string, std::string> compute_defines;
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    const uint32_t core_num = 1;
    const auto compute_kernel_file = "ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/kernels/moreh_dot.cpp";
    const auto compute_kernel_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core, core_num, compute_kernel_args},
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_buffer->address(), src1_buffer->address(), num_tiles, 0, mask_h, mask_w});
    SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles, 1});
    SetRuntimeArgs(program, writer_kernel_id, core, {output.buffer()->address(), 1, 0});

    const std::vector<Tensor> input_tensors = {input_a, input_b};

    return {
        std::move(program), {.unary_reader_kernel_id = reader_kernel_id, .unary_writer_kernel_id = writer_kernel_id}};
}

void MorehDotOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    auto src_buffer_a = input_a.buffer();
    auto src_buffer_b = input_b.buffer();
    auto dst_buffer = output.buffer();

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = src_buffer_a->address();
        runtime_args[1] = src_buffer_b->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::moreh::moreh_dot
