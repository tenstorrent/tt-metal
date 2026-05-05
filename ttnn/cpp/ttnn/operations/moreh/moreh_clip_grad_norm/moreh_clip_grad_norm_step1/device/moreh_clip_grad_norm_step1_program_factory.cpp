// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <vector>

#include "moreh_clip_grad_norm_step1_device_operation.hpp"
#include <tt_stl/assert.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1 {

using namespace tt::tt_metal;

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord) {
    auto p = std::floor(ord);
    auto fractional_part = ord - p;
    const bool p_is_negative = p < 0.0f;
    uint32_t integer_part = static_cast<uint32_t>(std::abs(p));
    return std::make_tuple(integer_part, fractional_part, p_is_negative);
}

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/"
    "reader_moreh_clip_grad_norm_step1.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/"
    "writer_moreh_clip_grad_norm_step1.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/kernels/"
    "moreh_clip_grad_norm_step1_kernel.cpp";

ProgramDescriptor MorehClipGradNormStep1Operation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tmp_pow_sum) {
    const auto& inputs = tensor_args.inputs;
    auto norm_type = operation_attributes.norm_type;
    auto tile_offset_of_tmp_pow_sum = operation_attributes.tile_offset_of_tmp_pow_sum;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = tmp_pow_sum.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_inputs = static_cast<uint32_t>(inputs.size());

    std::vector<std::pair<uint32_t, uint32_t>> origin_hw_vec;
    origin_hw_vec.reserve(num_inputs);

    for (uint32_t j = 0; j < num_inputs; ++j) {
        const auto& input_shape_without_padding = inputs.at(j).logical_shape();
        origin_hw_vec.emplace_back(input_shape_without_padding[2], input_shape_without_padding[3]);
    }

    auto [p, decimal, p_is_negative] = get_p_decimal_p_is_negative(norm_type);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_x = grid.x;
    const auto num_cores_y = grid.y;
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_inputs_per_core_group_1,
         num_inputs_per_core_group_2] = split_work_to_cores(grid, num_inputs);
    TT_FATAL(core_group_2.ranges().empty(), "core_group_2 must be empty");
    TT_FATAL(num_inputs_per_core_group_1 == 1, "num_inputs_per_core_group_1 must be 1");
    TT_FATAL(num_inputs_per_core_group_2 == 0, "num_inputs_per_core_group_2 must be 0");

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;  // input(==x)
    const uint32_t in1_t = 1;  // one
    const uint32_t in2_t = 1;  // decimal
    const uint32_t in3_t = 2;  // mask_h_w

    const uint32_t out0_t = 1;  // output(==y)

    const uint32_t im0_t = 1;  // |x|
    const uint32_t im1_t = 1;  // |x|^p
    const uint32_t im2_t = 1;  // Add[|x|^p * exp(log(|x|) * decimal)]
    const uint32_t im3_t = 1;  // log(|x|)
    const uint32_t im4_t = 1;  // exp(log(|x|) * decimal)
    const uint32_t im5_t = 1;  // |x|^p * exp(log(|x|) * decimal)

    const auto cb_data_format = datatype_to_dataformat_converter(tmp_pow_sum.dtype());
    const uint32_t cb_tile_size = tile_size(cb_data_format);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // input(==x)
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // one
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // decimal
    desc.cbs.push_back(CBDescriptor{
        .total_size = in3_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // mask_h_w
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // output(==y)
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_24, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // |x|
    desc.cbs.push_back(CBDescriptor{
        .total_size = im1_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_25, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // |x|^p
    desc.cbs.push_back(CBDescriptor{
        .total_size = im2_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_26, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // Add[|x|^p * exp(log(|x|) * decimal)]
    desc.cbs.push_back(CBDescriptor{
        .total_size = im3_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_27, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // log(|x|)
    desc.cbs.push_back(CBDescriptor{
        .total_size = im4_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_28, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // exp(log(|x|) * decimal)
    desc.cbs.push_back(CBDescriptor{
        .total_size = im5_t * cb_tile_size,
        .core_ranges = core_group_1,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_29, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // |x|^p * exp(log(|x|) * decimal)

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // Use inputs.at(0) for compile-time accessor args (all inputs share same buffer layout)
    KernelDescriptor::CompileTimeArgs reader_ct_args;
    TensorAccessorArgs(*inputs.at(0).buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_group_1;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores_to_be_used);

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*tmp_pow_sum.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_group_1;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_to_be_used);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_group_1;
    compute_desc.compile_time_args = {num_inputs_per_core_group_1};
    compute_desc.defines = {{"REDUCE_OP", "PoolType::SUM"}, {"REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"}};
    compute_desc.config = ComputeConfigDescriptor{};
    compute_desc.runtime_args.reserve(num_cores_to_be_used);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto output_addr = tmp_pow_sum.buffer()->address();
    auto cores = grid_to_cores(num_cores_to_be_used, num_cores_x, num_cores_y, false);

    uint32_t tile_offset = tile_offset_of_tmp_pow_sum;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);

        const auto& input = inputs.at(i);
        const auto input_addr = input.buffer()->address();
        const auto num_tiles = static_cast<uint32_t>(input.physical_volume()) / tt::constants::TILE_HW;
        const auto [origin_h, origin_w] = origin_hw_vec.at(i);

        // reader
        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                input_addr, num_tiles, std::bit_cast<uint32_t>(decimal), origin_h, origin_w});

        // writer
        writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{output_addr, tile_offset});

        // compute
        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{num_tiles, p, static_cast<uint32_t>(p_is_negative), origin_h, origin_w});

        tile_offset++;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm_step1
