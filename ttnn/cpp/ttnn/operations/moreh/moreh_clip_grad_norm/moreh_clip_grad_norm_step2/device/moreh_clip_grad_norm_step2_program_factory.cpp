// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <vector>

#include "moreh_clip_grad_norm_step2_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2 {

using namespace tt::tt_metal;

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord) {
    auto p = std::floor(ord);
    auto fractional_part = ord - p;
    const bool p_is_negative = p < 0.0f;
    uint32_t integer_part = static_cast<uint32_t>(std::abs(p));
    return std::make_tuple(integer_part, fractional_part, p_is_negative);
}

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/"
    "reader_moreh_clip_grad_norm_step2.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/"
    "writer_moreh_clip_grad_norm_step2.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/device/kernels/"
    "moreh_clip_grad_norm_step2_kernel.cpp";

ProgramDescriptor MorehClipGradNormStep2Operation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& total_norm) {
    const auto& tmp_pow_sum = tensor_args.tmp_pow_sum;
    auto norm_type = operation_attributes.norm_type;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_tiles = tmp_pow_sum.physical_volume() / tt::constants::TILE_HW;

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(1.0f / norm_type);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord single_core = {0, 0};
    CoreRangeSet core_set(CoreRange(single_core, single_core));

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(==tmp_pow_sum)
    const uint32_t in1_t = 1;  // decimal

    // x^p * exp(log(x) * decimal)
    const uint32_t out0_t = 1;  // output(==total_norm)

    const uint32_t im0_t = 1;  // Sum[tmp_pow_sum](==x)
    const uint32_t im1_t = 1;  // x^p
    const uint32_t im2_t = 1;  // log(x)
    const uint32_t im3_t = 1;  // exp(log(x) * decimal)

    const auto cb_data_format = datatype_to_dataformat_converter(total_norm.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // input(==tmp_pow_sum)
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // decimal
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // output(==total_norm)
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_24, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // Sum[tmp_pow_sum](==x)
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_25, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // x^p
    desc.cbs.push_back(CBDescriptor{
        .total_size = im2_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_26, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // log(x)
    desc.cbs.push_back(CBDescriptor{
        .total_size = im3_t * cb_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_27, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // exp(log(x) * decimal)

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*tmp_pow_sum.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*total_norm.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = std::move(core_set);
    compute_desc.compile_time_args = {static_cast<uint32_t>(num_tiles)};
    compute_desc.config = ComputeConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = tmp_pow_sum.buffer()->address();
    const auto output_addr = total_norm.buffer()->address();

    // reader
    reader_desc.runtime_args.emplace_back(
        single_core,
        KernelDescriptor::CoreRuntimeArgs{
            input_addr, static_cast<uint32_t>(num_tiles), *reinterpret_cast<uint32_t*>(&decimal)});

    // writer
    writer_desc.runtime_args.emplace_back(single_core, KernelDescriptor::CoreRuntimeArgs{output_addr});

    // compute
    compute_desc.runtime_args.emplace_back(
        single_core,
        KernelDescriptor::CoreRuntimeArgs{static_cast<uint32_t>(num_tiles), p, static_cast<uint32_t>(p_is_negative)});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step2
