// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

using namespace tt::tt_metal;

ProgramDescriptor Fold::MultiCore::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = output_tensor;
    const uint32_t stride_h = operation_attributes.stride_h;
    const uint32_t stride_w = operation_attributes.stride_w;

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

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    // Input CB — globally allocated to the sharded input buffer.
    // The descriptor framework patches the CB address on cache hits via cb.buffer.
    const uint32_t cb_src0_index = tt::CBIndex::c_0;
    const uint32_t aligned_pixel_size = tt::align(pixel_size, hal::get_l1_alignment());
    {
        CBDescriptor cb_src0;
        cb_src0.total_size = num_pixels * aligned_pixel_size;
        cb_src0.core_ranges = all_cores;
        cb_src0.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_src0_index),
            .data_format = cb_data_format,
            .page_size = aligned_pixel_size,
        });
        cb_src0.buffer = src_buffer;
        desc.cbs.push_back(std::move(cb_src0));
    }

    // Output CB — globally allocated to the sharded output buffer.
    const uint32_t cb_dst0_index = tt::CBIndex::c_16;
    const uint32_t aligned_dst_pixel_size = tt::align(dst_pixel_size, hal::get_l1_alignment());
    {
        CBDescriptor cb_dst0;
        cb_dst0.total_size = num_dst_pixels * aligned_dst_pixel_size;
        cb_dst0.core_ranges = all_cores;
        cb_dst0.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_dst0_index),
            .data_format = cb_data_format,
            .page_size = aligned_dst_pixel_size,
        });
        cb_dst0.buffer = dst_buffer;
        desc.cbs.push_back(std::move(cb_dst0));
    }

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
        true,  // is_reader (overwritten below for the writer kernel)
    };

    // Writer kernel: shares the same source as the reader (a single kernel file selects
    // its role from the last compile-time arg). Build optimization level Os was faster
    // than O2 when this kernel was originally tuned.
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2s_row_major.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.opt_level = KernelBuildOptLevel::Os;
    desc.kernels.push_back(std::move(writer_desc));

    compile_time_args[13] = false;  // is_reader = false for the reader-data-movement variant
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2s_row_major.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.opt_level = KernelBuildOptLevel::Os;
    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::operations::data_movement
