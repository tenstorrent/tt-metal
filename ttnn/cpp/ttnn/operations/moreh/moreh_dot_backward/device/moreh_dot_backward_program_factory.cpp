// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_backward_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_dot_backward {

using namespace tt::tt_metal;

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/reader_moreh_dot_backward.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/writer_moreh_dot_backward.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/kernels/moreh_dot_backward.cpp";

ProgramDescriptor MorehDotBackwardOperation::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;
    const auto& input_grad = tensor_return_value.at(0);
    const auto& other_grad = tensor_return_value.at(1);

    CoreCoord core = {0, 0};
    CoreRangeSet core_set(CoreRange(core, core));

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    auto* src0_buffer = output_grad.buffer();
    auto* src1_buffer = input.buffer();
    auto* src2_buffer = other.buffer();

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    const uint32_t in0_t = 2;
    const uint32_t in1_t = 2;
    const uint32_t in2_t = 2;
    const uint32_t out0_t = 2;
    const uint32_t out1_t = 2;

    bool has_input_grad = input_grad.has_value();
    bool has_other_grad = other_grad.has_value();

    ProgramDescriptor desc;

    // Circular buffers
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out1_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_17,
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });

    // Reader kernel
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(src0_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(src1_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(src2_buffer).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.emplace_back(
        core,
        KernelDescriptor::CoreRuntimeArgs{
            static_cast<uint32_t>(has_input_grad),
            static_cast<uint32_t>(has_other_grad),
            src0_buffer->address(),
            src1_buffer->address(),
            src2_buffer->address(),
            num_tiles,
            0u});

    // Writer kernel
    uint32_t dst0_address = 0;
    uint32_t dst1_address = 0;

    if (has_input_grad) {
        const auto& input_grad_tensor = input_grad.value();
        auto* dst0_buffer = input_grad_tensor.buffer();
        TT_ASSERT(dst0_buffer != nullptr, "input_grad buffer should be allocated on device!");
        dst0_address = dst0_buffer->address();
    }

    if (has_other_grad) {
        const auto& other_grad_tensor = other_grad.value();
        auto* dst1_buffer = other_grad_tensor.buffer();
        TT_ASSERT(dst1_buffer != nullptr, "other_grad buffer should be allocated on device!");
        dst1_address = dst1_buffer->address();
    }

    KernelDescriptor::CompileTimeArgs writer_ct_args = {
        static_cast<uint32_t>(tt::CBIndex::c_16),
        static_cast<uint32_t>(tt::CBIndex::c_17),
    };
    TensorAccessorArgs(has_input_grad ? input_grad.value().buffer() : nullptr).append_to(writer_ct_args);
    TensorAccessorArgs(has_other_grad ? other_grad.value().buffer() : nullptr).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.emplace_back(
        core,
        KernelDescriptor::CoreRuntimeArgs{
            static_cast<uint32_t>(has_input_grad),
            static_cast<uint32_t>(has_other_grad),
            dst0_address,
            dst1_address,
            num_tiles,
            0u});

    // Compute kernel
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = std::move(core_set);
    compute_desc.config = ComputeConfigDescriptor{};
    compute_desc.runtime_args.emplace_back(
        core,
        KernelDescriptor::CoreRuntimeArgs{
            static_cast<uint32_t>(has_input_grad), static_cast<uint32_t>(has_other_grad), num_tiles});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_dot_backward
