// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/common/fp8_quant_common.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

namespace common = ttnn::operations::experimental::deepseek_prefill::fp8_quant_common;

PerTokenCastToFp8ProgramFactory::cached_program_t PerTokenCastToFp8ProgramFactory::create(
    const PerTokenCastToFp8Params& /*operation_attributes*/,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input_tensor;
    const auto& [output_e4m3, output_scale] = tensor_return_value;

    const auto& input_shape = input.logical_shape();
    auto [M, H] = common::infer_M_H(input_shape);
    TT_FATAL(M % constants::TILE_HEIGHT == 0, "per_token_cast_to_fp8: M={} must be divisible by TILE_HEIGHT=32", M);
    TT_FATAL(H % common::SCALE_GROUP_SIZE == 0, "per_token_cast_to_fp8: H={} must be a multiple of 128", H);

    const uint32_t tile_rows = M / constants::TILE_HEIGHT;

    auto* src_buffer = input.buffer();
    auto* dst_e4m3_buffer = output_e4m3.buffer();
    auto* dst_scale_buffer = output_scale.buffer();

    Program program{};

    auto* device = input.device();
    auto compute_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2] =
        split_work_to_cores(compute_grid, tile_rows);

    const DataFormat input_df = datatype_to_dataformat_converter(input.dtype());
    const DataFormat fp8_df = DataFormat::Fp8_e4m3;
    const DataFormat scale_df = DataFormat::Float32;

    const uint32_t input_element_size = input.element_size();
    const uint32_t stick_size_input = H * input_element_size;
    const uint32_t stick_size_fp8 = H;
    const uint32_t scale_write_size_bytes = dst_scale_buffer->aligned_page_size();
    const uint32_t is_fp32 = (input.dtype() == DataType::FLOAT32) ? 1 : 0;

    constexpr uint32_t cb_in_rm_idx = CBIndex::c_0;
    constexpr uint32_t cb_scratch_fp8_idx = CBIndex::c_1;
    constexpr uint32_t cb_scale_const_idx = CBIndex::c_2;

    // cb_in_rm: 32 RM pages (one per stick) of input dtype.
    CircularBufferConfig cb_in_rm_cfg =
        CircularBufferConfig(constants::TILE_HEIGHT * stick_size_input, {{cb_in_rm_idx, input_df}})
            .set_page_size(cb_in_rm_idx, stick_size_input);
    CreateCircularBuffer(program, all_cores, cb_in_rm_cfg);

    // cb_scratch_fp8: L1 region for 32 converted fp8 sticks (= TILE_HEIGHT * H bytes total).
    const uint32_t fp8_scratch_bytes = constants::TILE_HEIGHT * stick_size_fp8;
    CircularBufferConfig cb_scratch_fp8_cfg = CircularBufferConfig(fp8_scratch_bytes, {{cb_scratch_fp8_idx, fp8_df}})
                                                  .set_page_size(cb_scratch_fp8_idx, fp8_scratch_bytes);
    CreateCircularBuffer(program, all_cores, cb_scratch_fp8_cfg);

    // cb_scale_const: 1 page sized to the scale buffer's aligned_page_size.
    CircularBufferConfig cb_scale_const_cfg =
        CircularBufferConfig(scale_write_size_bytes, {{cb_scale_const_idx, scale_df}})
            .set_page_size(cb_scale_const_idx, scale_write_size_bytes);
    CreateCircularBuffer(program, all_cores, cb_scale_const_cfg);

    // Reader kernel (RISCV_1).
    std::vector<uint32_t> reader_ct_args = {cb_in_rm_idx, stick_size_input};
    TensorAccessorArgs(src_buffer).append_to(reader_ct_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "reader_per_token_cast_to_fp8.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Writer kernel (RISCV_0) — does bf16/fp32 -> fp8 conversion in software.
    std::vector<uint32_t> writer_ct_args = {
        cb_in_rm_idx,
        cb_scratch_fp8_idx,
        cb_scale_const_idx,
        input_element_size,
        stick_size_input,
        stick_size_fp8,
        scale_write_size_bytes,
        is_fp32,
        H};
    TensorAccessorArgs(dst_e4m3_buffer).append_to(writer_ct_args);
    TensorAccessorArgs(dst_scale_buffer).append_to(writer_ct_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/dataflow/"
        "writer_per_token_cast_to_fp8.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // No compute kernel: conversion happens in the writer (software path).

    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
    uint32_t row_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = all_cores_vec[i];
        uint32_t rows_for_core =
            core_group_1.contains(core) ? rows_per_core_g1 : (core_group_2.contains(core) ? rows_per_core_g2 : 0);

        SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), row_offset, rows_for_core});
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {dst_e4m3_buffer->address(), dst_scale_buffer->address(), row_offset, rows_for_core});
        row_offset += rows_for_core;
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, /*compute_kernel_id=*/0, std::move(all_cores_vec)}};
}

void PerTokenCastToFp8ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PerTokenCastToFp8Params& /*operation_attributes*/,
    const PerTokenCastToFp8Inputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    const auto& [output_e4m3, output_scale] = tensor_return_value;
    uint32_t src_addr = tensor_args.input_tensor.buffer()->address();
    uint32_t dst_e4m3_addr = output_e4m3.buffer()->address();
    uint32_t dst_scale_addr = output_scale.buffer()->address();

    for (const auto& core : shared.all_cores_vec) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
        reader_args[0] = src_addr;
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[0] = dst_e4m3_addr;
        writer_args[1] = dst_scale_addr;
    }
}

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8
