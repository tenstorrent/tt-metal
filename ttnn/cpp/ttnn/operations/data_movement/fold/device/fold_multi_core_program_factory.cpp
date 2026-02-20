// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "fold_device_op.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

Fold::MultiCore::cached_program_t fold_multi_core(
    const Tensor& input, const Tensor& output, uint32_t stride_h, uint32_t stride_w) {
    Program program = CreateProgram();

    auto all_cores = input.shard_spec()->grid;
    auto shard_shape = input.shard_spec()->shape;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t pixel_size = shard_shape[1] * input.element_size();
    uint32_t num_pixels = shard_shape[0];
    uint32_t num_dst_pixels = num_pixels / (stride_h * stride_w);

    // chunk consists of channel values of stride_w neighboring pixels along the W dimension
    uint32_t width = input.padded_shape()[2];
    uint32_t chunk_size = stride_w * pixel_size;
    uint32_t dst_pixel_size = stride_h * chunk_size;
    uint32_t num_dst_rows = num_pixels / (width * stride_h);
    uint32_t pixels_per_dst_row = stride_h * width;

    // input CB
    uint32_t cb_src0_index = tt::CBIndex::c_0;
    uint32_t aligned_pixel_size = tt::align(pixel_size, hal::get_l1_alignment());
    auto src_cb_config = CircularBufferConfig(num_pixels * aligned_pixel_size, {{cb_src0_index, cb_data_format}})
                             .set_page_size(cb_src0_index, aligned_pixel_size)
                             .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output CB
    uint32_t cb_dst0_index = tt::CBIndex::c_16;
    uint32_t aligned_dst_pixel_size = tt::align(dst_pixel_size, hal::get_l1_alignment());
    auto dst_cb_config =
        CircularBufferConfig(num_dst_pixels * aligned_dst_pixel_size, {{cb_dst0_index, cb_data_format}})
            .set_page_size(cb_dst0_index, aligned_dst_pixel_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = CreateCircularBuffer(program, all_cores, dst_cb_config);

    std::vector<uint32_t> compile_time_args = {
        cb_src0_index,
        cb_dst0_index,
        pixel_size,
        aligned_pixel_size,
        aligned_dst_pixel_size,
        stride_w * aligned_pixel_size,
        width * aligned_pixel_size,
        stride_h,
        stride_w,
        num_dst_rows,
        width / stride_w,
        pixels_per_dst_row * aligned_pixel_size,
        input.element_size(),
        true,
    };
    // Setup kernel
    // Set build optimization level to Os. O2 was slower.
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2s_row_major.cpp",
        all_cores,
        WriterDataMovementConfig(compile_time_args, {}, {}, tt::tt_metal::KernelBuildOptLevel::Os));

    compile_time_args[13] = false;  // is_reader = false for writer
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2s_row_major.cpp",
        all_cores,
        ReaderDataMovementConfig(compile_time_args, {}, {}, tt::tt_metal::KernelBuildOptLevel::Os));

    return {std::move(program), {reader_kernel_id, writer_kernel_id, stride_h, stride_w, cb_src0, cb_dst0}};
}

Fold::MultiCore::cached_program_t Fold::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    return fold_multi_core(
        tensor_args.input_tensor, output_tensor, operation_attributes.stride_h, operation_attributes.stride_w);
}

void Fold::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    auto& cb_dst0 = cached_program.shared_variables.cb_dst0;

    auto& program = cached_program.program;
    const auto& input_tensor = tensor_args.input_tensor;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_dst0, *dst_buffer);
}

}  // namespace ttnn::operations::data_movement
