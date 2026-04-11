// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample_bicubic_program_factory.hpp"

#include <cmath>
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <ttnn/operations/cb_utils.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include "upsample/device/upsample_device_operation_types.hpp"

namespace ttnn::prim {

UpsampleBicubicProgramFactory::cached_program_t UpsampleBicubicProgramFactory::create(
    const UpsampleParams& operation_attributes, const Tensor& input, Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::Program{};

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    auto* const device = output_tensor.device();

    const auto& input_shape = input.logical_shape();
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t channels = input_shape[3];

    const uint32_t output_height =
        static_cast<uint32_t>(std::floor(input_height * operation_attributes.scale_factor_h));
    const uint32_t output_width = static_cast<uint32_t>(std::floor(input_width * operation_attributes.scale_factor_w));

    const uint32_t batch_size = input_shape[0];
    const uint32_t total_output_pixels = batch_size * output_height * output_width;

    const uint32_t aligned_input_page_size = input.buffer()->aligned_page_size();
    const uint32_t input_stick_nbytes = channels * input.element_size();

    // Channel blocking: process channels in blocks that fit in tile reduction
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    const uint32_t max_block_bytes = MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * input.element_size();
    const uint32_t input_block_size_bytes = std::min(input_stick_nbytes, max_block_bytes);
    const uint32_t num_blocks = static_cast<uint32_t>(
        std::ceil(static_cast<float>(channels) / (MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH)));
    const uint32_t in_ntiles_c = tt::div_up(channels, tt::constants::TILE_WIDTH);

    // Compute kernel config
    const auto compute_config_tuple =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    const tt::tt_metal::MathFidelity math_fidelity = std::get<0>(compute_config_tuple);
    const bool math_approx_mode = std::get<1>(compute_config_tuple);
    const bool fp32_dest_acc_en = std::get<2>(compute_config_tuple);

    // Work distribution
    const tt::tt_metal::CoreCoord compute_grid_size = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, pixels_per_core_group_1, pixels_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_pixels);

    const std::vector<tt::tt_metal::CoreCoord> logical_cores =
        tt::tt_metal::corerange_to_cores(all_cores, std::nullopt, true);

    // --- Circular Buffers ---
    uint32_t next_cb_index = tt::CBIndex::c_0;

    // Input CB: holds 4 neighbor partial sticks per group
    const auto [input_cb_index, input_cb_handle] =
        tt::tt_metal::create_cb(next_cb_index++, program, all_cores, input_block_size_bytes, 4 * 2, cb_data_format);

    // Scalar CB: holds 4 BF16 weights per group in tile format
    const auto [scalar_cb_index, scalar_cb_handle] =
        tt::tt_metal::create_cb(next_cb_index++, program, all_cores, tt::tile_size(cb_data_format), 2, cb_data_format);

    // Output CB: holds output sticks (tile-width pages), ring-buffered
    const uint32_t out_cb_page_size = tt::constants::TILE_WIDTH * output_tensor.element_size();
    const auto [output_cb_index, output_cb_handle] =
        tt::tt_metal::create_cb(next_cb_index++, program, all_cores, out_cb_page_size, in_ntiles_c * 2, cb_data_format);

    // --- Reader Kernel ---
    // Precompute Q16.16 fixed-point ratios on host: ratio = input_size / output_size
    constexpr int32_t FIXED_ONE_Q16 = 1 << 16;
    const int32_t ratio_h_fixed =
        static_cast<int32_t>((static_cast<float>(input_height) / static_cast<float>(output_height)) * FIXED_ONE_Q16);
    const int32_t ratio_w_fixed =
        static_cast<int32_t>((static_cast<float>(input_width) / static_cast<float>(output_width)) * FIXED_ONE_Q16);

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,
        scalar_cb_index,
        input_height,
        input_width,
        output_height,
        output_width,
        input_block_size_bytes,
        aligned_input_page_size,
        num_blocks,
        input_stick_nbytes,
        static_cast<uint32_t>(ratio_h_fixed),  // [10] Q16.16 ratio_h
        static_cast<uint32_t>(ratio_w_fixed),  // [11] Q16.16 ratio_w
    };
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_upsample_bicubic.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // --- Compute Kernel ---
    std::vector<uint32_t> compute_compile_time_args = {
        input_cb_index,
        scalar_cb_index,
        output_cb_index,
        in_ntiles_c,
        num_blocks,
    };

    const tt::tt_metal::ReduceOpMath reduce_op = tt::tt_metal::ReduceOpMath::SUM;
    const tt::tt_metal::ReduceOpDim reduce_dim = tt::tt_metal::ReduceOpDim::H;

    const tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bicubic.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)});

    // --- Writer Kernel ---
    const uint32_t aligned_output_page_size = output_tensor.buffer()->aligned_page_size();
    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,
        aligned_output_page_size,
        in_ntiles_c,
        num_blocks,
    };
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    const tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_bicubic.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // --- Runtime Arguments ---
    uint32_t pixels_processed = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord& core = logical_cores[i];
        const uint32_t pixels_this_core =
            core_group_1.contains(core) ? pixels_per_core_group_1 : pixels_per_core_group_2;

        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {input.buffer()->address(), pixels_this_core, pixels_processed});
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {pixels_this_core});
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output_tensor.buffer()->address(), pixels_this_core, pixels_processed});

        pixels_processed += pixels_this_core;
    }

    return {
        std::move(program),
        UpsampleBicubicSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .all_cores = all_cores,
            .num_cores = num_cores}};
}

void UpsampleBicubicProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const UpsampleParams& /* operation_attributes */,
    const Tensor& input,
    Tensor& output_tensor) {
    tt::tt_metal::Program& program = cached_program.program;
    const tt::tt_metal::KernelHandle reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const tt::tt_metal::KernelHandle writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;

    auto* const device = input.device();
    const tt::tt_metal::CoreCoord compute_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_grid_size.y;

    for (uint32_t i = 0; i < num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core)[0] = input.buffer()->address();
        tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core)[0] = output_tensor.buffer()->address();
    }
}

}  // namespace ttnn::prim
