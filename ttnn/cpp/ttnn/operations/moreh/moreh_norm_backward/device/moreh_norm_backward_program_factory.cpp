// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm_backward_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_norm_backward {

using namespace tt::tt_metal;

std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}

void get_tensor_dim(ttnn::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = (shape[idx] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }
}

ttnn::Shape get_output_grad_shape(
    const Tensor& output_grad, const Tensor& input_grad, const ttnn::SmallVector<int64_t>& dims, const bool& keepdim) {
    if (keepdim) {
        return output_grad.logical_shape();
    }

    auto shape = input_grad.logical_shape();
    auto rank = shape.rank();
    for (auto dim : dims) {
        TT_FATAL(dim < rank, "dim {} < rank {}", dim, rank);
        shape[dim] = 1;
    }
    return shape;
}

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/reader_moreh_norm_backward.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/writer_moreh_norm_backward.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward/device/kernels/moreh_norm_backward_kernel.cpp";

ProgramDescriptor MorehNormBackwardOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;
    const auto& output_grad = tensor_args.output_grad;
    const auto p = operation_attributes.p;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto& input_grad_shape = input_grad.logical_shape();
    const auto input_grad_rank = input_grad_shape.rank();

    ttnn::SmallVector<uint32_t> input_grad_dim(input_grad_rank, 1);
    get_tensor_dim(input_grad_dim, input_grad_shape);
    auto output_grad_shape =
        get_output_grad_shape(output_grad, input_grad, operation_attributes.dims, operation_attributes.keepdim);

    ttnn::SmallVector<uint32_t> output_grad_dim(input_grad_rank, 1);
    get_tensor_dim(output_grad_dim, output_grad_shape);

    ttnn::SmallVector<uint32_t> need_bcast_dim(input_grad_rank, 0);
    for (auto i = 0; i < input_grad_rank; ++i) {
        auto idx = input_grad_rank - 1 - i;
        need_bcast_dim[i] = (output_grad_shape[idx] != input_grad_shape[idx]);
    }

    const auto num_input_grad_tiles = input_grad.physical_volume() / tt::constants::TILE_HW;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(output_grad.device()->arch(), operation_attributes.compute_kernel_config);

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_p_minus_one, decimal_minus_one, p_minus_one_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(p - 1.0f);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = split_work_to_cores(grid, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const uint32_t cb_tile_size = tile_size(cb_data_format);
    const uint32_t intermed_tile_size = tile_size(intermed_data_format);

    const uint32_t in0_t{1};  // input(==x)
    const uint32_t in1_t{1};  // output(==y)
    const uint32_t in2_t{1};  // output_grad(==dy)
    const uint32_t in3_t{1};  // decimal

    // (x^(p - 1) * y * dy) / y^p
    const uint32_t out0_t{1};  // input_grad(==dx)

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // input
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // output
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // output_grad
    desc.cbs.push_back(CBDescriptor{
        .total_size = in3_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // decimal
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_16, .data_format = cb_data_format, .page_size = cb_tile_size}}},
    });  // input_grad

    // Intermediate CBs (c_24 through c_31)
    for (uint8_t cb_idx = static_cast<uint8_t>(tt::CBIndex::c_24); cb_idx <= static_cast<uint8_t>(tt::CBIndex::c_31);
         ++cb_idx) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = 1 * intermed_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_idx, .data_format = intermed_data_format, .page_size = intermed_tile_size}}},
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args = {static_cast<uint32_t>(input_grad_rank)};
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*output.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*output_grad.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores_to_be_used);

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(*input_grad.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_to_be_used);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::Defines compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    ComputeConfigDescriptor compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // Compute kernel for core_group_1
    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {num_cols_per_core_group_1, need_bcast_dim[0], need_bcast_dim[1]};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = compute_config;

    // Compute kernel for core_group_2 (may be empty)
    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = COMPUTE_KERNEL_PATH;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {num_cols_per_core_group_2, need_bcast_dim[0], need_bcast_dim[1]};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = compute_config;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        KernelDescriptor::CoreRuntimeArgs reader_rt_args{
            input.buffer()->address(),
            output.buffer()->address(),
            output_grad.buffer()->address(),
            *reinterpret_cast<uint32_t*>(&decimal),
            num_tiles_per_core,
            tile_offset};
        reader_rt_args.insert(reader_rt_args.end(), output_grad_dim.begin(), output_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), input_grad_dim.begin(), input_grad_dim.end());
        reader_rt_args.insert(reader_rt_args.end(), need_bcast_dim.begin(), need_bcast_dim.end());

        reader_desc.runtime_args.emplace_back(core, std::move(reader_rt_args));

        // writer
        writer_desc.emplace_runtime_args(core, {input_grad.buffer(), num_tiles_per_core, tile_offset});

        // compute — runtime args go to the correct kernel descriptor
        if (core_group_1.contains(core)) {
            compute_desc_1.emplace_runtime_args(
                core,
                {num_tiles_per_core,
                 floored_p,
                 static_cast<uint32_t>(p_is_negative),
                 floored_p_minus_one,
                 static_cast<uint32_t>(p_minus_one_is_negative)});
        } else {
            compute_desc_2.emplace_runtime_args(
                core,
                {num_tiles_per_core,
                 floored_p,
                 static_cast<uint32_t>(p_is_negative),
                 floored_p_minus_one,
                 static_cast<uint32_t>(p_minus_one_is_negative)});
        }

        tile_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_norm_backward
