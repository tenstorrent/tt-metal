// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

#include "fold_device_op.hpp"
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::data_movement {

cached_program_t fold_multi_core(
    const Tensor& input, const Tensor& output, uint32_t stride_h, uint32_t stride_w) {
    auto program = CreateProgram();
    Device* device = output.device();

    auto all_cores = input.shard_spec()->grid;
    auto shard_shape = input.shard_spec()->shape;

    DataFormat cb_data_format = datatype_to_dataformat_converter(input.get_dtype());

    uint32_t pixel_size = shard_shape[1] * input.element_size();
    uint32_t num_pixels = shard_shape[0];
    uint32_t num_dst_pixels = num_pixels / (stride_h * stride_w);

    // chunk consists of channel values of stride_w neighboring pixels along the W dimension
    uint32_t width = input.get_legacy_shape()[2];
    uint32_t chunk_size = stride_w * pixel_size;
    uint32_t row_size = width * pixel_size;
    uint32_t dst_pixel_size = stride_h * chunk_size;
    uint32_t dst_row_size = stride_h * row_size;
    uint32_t num_dst_rows = num_pixels / (width * stride_h);
    uint32_t pixels_per_dst_row = stride_h * width;

    // input CB
    uint32_t cb_src0_index = CB::c_in0;
    uint32_t aligned_pixel_size = round_up_to_mul32(pixel_size);
    auto src_cb_config = CircularBufferConfig(num_pixels * aligned_pixel_size, {{cb_src0_index, cb_data_format}})
                             .set_page_size(cb_src0_index, aligned_pixel_size)
                             .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = CreateCircularBuffer(program, all_cores, src_cb_config);

    // output CB
    uint32_t cb_dst0_index = CB::c_out0;
    uint32_t aligned_dst_pixel_size = round_up_to_mul32(dst_pixel_size);
    auto dst_cb_config =
        CircularBufferConfig(num_dst_pixels * aligned_dst_pixel_size, {{cb_dst0_index, cb_data_format}})
            .set_page_size(cb_dst0_index, aligned_dst_pixel_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = CreateCircularBuffer(program, all_cores, dst_cb_config);

    // Setup kernel
    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2s_row_major.cpp",
        all_cores,
        WriterDataMovementConfig({cb_src0_index, cb_dst0_index}));

    // Writer run-time args
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        all_cores,
        {
            pixel_size,
            aligned_pixel_size,
            aligned_dst_pixel_size,
            num_pixels,
            num_dst_pixels,
            stride_w * aligned_pixel_size,
            width * aligned_pixel_size,
            stride_h,
            stride_w,
            num_dst_rows,
            width / stride_w,
            pixels_per_dst_row * aligned_pixel_size,
        });

    return { std::move(program), {writer_kernel_id, stride_h, stride_w, cb_src0, cb_dst0} };
}

cached_program_t Fold::MultiCore::create(const operation_attributes_t& operation_attributes,
                                         const tensor_args_t& tensor_args,
                                         tensor_return_value_t& output_tensor) {
    return fold_multi_core(tensor_args.input_tensor, output_tensor, operation_attributes.stride_h, operation_attributes.stride_w);
}

void Fold::MultiCore::override_runtime_arguments(cached_program_t& cached_program,
                                                const operation_attributes_t& operation_attributes,
                                                const tensor_args_t& tensor_args,
                                                tensor_return_value_t& output_tensor) {

        auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
        auto stride_h = cached_program.shared_variables.stride_h;
        auto stride_w = cached_program.shared_variables.stride_w;
        auto cb_src0 = cached_program.shared_variables.cb_src0;
        auto cb_dst0 = cached_program.shared_variables.cb_dst0;

        auto program = cached_program.program;
        auto input_tensor = tensor_args.input_tensor;

        auto shard_shape = input_tensor.shard_spec()->shape;
        auto all_cores = input_tensor.shard_spec()->grid;

        DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.get_dtype());

        uint32_t pixel_size = shard_shape[1] * input_tensor.element_size();
        uint32_t num_pixels = shard_shape[0];
        uint32_t num_dst_pixels = num_pixels / (stride_h * stride_w);

        uint32_t width = input_tensor.get_legacy_shape()[2];
        uint32_t chunk_size = stride_w * pixel_size;
        uint32_t row_size = width * pixel_size;
        uint32_t dst_pixel_size = stride_h * chunk_size;
        uint32_t dst_row_size = stride_h * row_size;
        uint32_t num_dst_rows = num_pixels / (width * stride_h);
        uint32_t cb_pages_per_dst_row = stride_h * width;

        uint32_t aligned_pixel_size = round_up_to_mul32(pixel_size);
        uint32_t aligned_dst_pixel_size = round_up_to_mul32(dst_pixel_size);

        auto src_buffer = input_tensor.buffer();
        auto dst_buffer = output_tensor.buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, cb_dst0, *dst_buffer);

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            all_cores,
            {
                pixel_size,
                aligned_pixel_size,
                aligned_dst_pixel_size,
                num_pixels,
                num_dst_pixels,
                stride_w * aligned_pixel_size,
                width * aligned_pixel_size,
                stride_h,
                stride_w,
                num_dst_rows,
                width / stride_w,
                cb_pages_per_dst_row,
            });
}

}  // namespace ttnn::operations::data_movement
