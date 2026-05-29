// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/common/fp8_quant_common.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

namespace common = ttnn::operations::experimental::deepseek_prefill::fp8_quant_common;

PerTokenCastBackProgramFactory::cached_program_t PerTokenCastBackProgramFactory::create(
    const PerTokenCastBackParams& operation_attributes,
    const PerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& e4m3 = tensor_args.input_e4m3;
    const auto& scale_in = tensor_args.input_scale;
    auto& output = tensor_return_value;

    const auto& shape = e4m3.logical_shape();
    auto [M, H] = common::infer_M_H(shape);
    TT_FATAL(M % constants::TILE_HEIGHT == 0, "per_token_cast_back: M={} must be divisible by TILE_HEIGHT=32", M);
    TT_FATAL(H % common::SCALE_GROUP_SIZE == 0, "per_token_cast_back: H={} must be a multiple of 128", H);

    const uint32_t tile_rows = M / constants::TILE_HEIGHT;

    auto* src_e4m3_buffer = e4m3.buffer();
    auto* src_scale_buffer = scale_in.buffer();
    auto* dst_buffer = output.buffer();

    Program program{};

    auto* device = e4m3.device();
    auto compute_grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2] =
        split_work_to_cores(compute_grid, tile_rows);

    const DataFormat fp8_df = DataFormat::Fp8_e4m3;
    const DataFormat output_df = datatype_to_dataformat_converter(operation_attributes.output_dtype);

    const uint32_t fp8_stick_size = H;  // 1 byte per fp8 element
    const uint32_t out_element_size = output.element_size();
    const uint32_t out_stick_size = H * out_element_size;
    const uint32_t is_fp32 = (operation_attributes.output_dtype == DataType::FLOAT32) ? 1 : 0;

    constexpr uint32_t cb_in_fp8_idx = CBIndex::c_0;
    constexpr uint32_t cb_scratch_idx = CBIndex::c_1;

    // cb_in_fp8: 32 fp8 RM sticks (one per row of the tile-row).
    CircularBufferConfig cb_in_fp8_cfg =
        CircularBufferConfig(constants::TILE_HEIGHT * fp8_stick_size, {{cb_in_fp8_idx, fp8_df}})
            .set_page_size(cb_in_fp8_idx, fp8_stick_size);
    CreateCircularBuffer(program, all_cores, cb_in_fp8_cfg);

    // cb_scratch: holds 32 converted output sticks (bf16/fp32 RM). Sized as a single page.
    const uint32_t scratch_bytes = constants::TILE_HEIGHT * out_stick_size;
    CircularBufferConfig cb_scratch_cfg =
        CircularBufferConfig(scratch_bytes, {{cb_scratch_idx, output_df}}).set_page_size(cb_scratch_idx, scratch_bytes);
    CreateCircularBuffer(program, all_cores, cb_scratch_cfg);

    // Reader kernel (RISCV_1).
    std::vector<uint32_t> reader_ct_args = {cb_in_fp8_idx, fp8_stick_size};
    TensorAccessorArgs(src_e4m3_buffer).append_to(reader_ct_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/"
        "reader_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Writer kernel (RISCV_0) — does fp8 -> bf16/fp32 conversion in software.
    std::vector<uint32_t> writer_ct_args = {
        cb_in_fp8_idx, cb_scratch_idx, fp8_stick_size, out_element_size, out_stick_size, is_fp32};
    TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/"
        "writer_per_token_cast_back.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // No compute kernel: conversion happens in the writer (RISC-V software path).

    auto all_cores_vec = corerange_to_cores(all_cores, num_cores, true);
    uint32_t row_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = all_cores_vec[i];
        uint32_t rows_for_core =
            core_group_1.contains(core) ? rows_per_core_g1 : (core_group_2.contains(core) ? rows_per_core_g2 : 0);

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src_e4m3_buffer->address(), src_scale_buffer->address(), row_offset, rows_for_core});
        SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), row_offset, rows_for_core});
        row_offset += rows_for_core;
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, /*compute_kernel_id=*/0, std::move(all_cores_vec)}};
}

void PerTokenCastBackProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PerTokenCastBackParams& /*operation_attributes*/,
    const PerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    uint32_t src_e4m3_addr = tensor_args.input_e4m3.buffer()->address();
    uint32_t src_scale_addr = tensor_args.input_scale.buffer()->address();
    uint32_t dst_addr = tensor_return_value.buffer()->address();

    for (const auto& core : shared.all_cores_vec) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
        reader_args[0] = src_e4m3_addr;
        reader_args[1] = src_scale_addr;
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
        writer_args[0] = dst_addr;
    }
}

}  // namespace ttnn::experimental::prim::per_token_cast_back
