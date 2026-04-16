// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad {

using namespace tt::tt_metal;
using namespace tt::constants;

static void push_cb(
    ProgramDescriptor& desc,
    uint32_t num_tiles,
    const CoreRangeSet& cores,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t tile_size) {
    if (num_tiles > 0) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * tile_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_index,
                .data_format = data_format,
                .page_size = tile_size,
            }}},
        });
    }
}

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "reader_moreh_layer_norm_backward_gamma_beta_grad.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "writer_moreh_layer_norm_backward_gamma_beta_grad.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp";

ProgramDescriptor MorehLayerNormBackwardGammaBetaGradOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const std::optional<const Tensor>& gamma_grad = output_tensor.at(0);
    const std::optional<const Tensor>& beta_grad = output_tensor.at(1);

    auto normalized_dims = operation_attributes.normalized_dims;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.padded_shape();
    const auto output_grad_shape_without_padding = output_grad.logical_shape();

    const bool is_lastdim_layer_norm = normalized_dims == 1;
    const bool is_groupnorm = false;

    const auto origin_H = output_grad_shape_without_padding[-2];
    const auto origin_W = output_grad_shape_without_padding[-1];

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && is_lastdim_layer_norm;
    const uint32_t mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const auto mean_rstd_shape_without_padding = mean.logical_shape();
    auto mean_rstd_height = mean_rstd_shape_without_padding[-2];
    auto mean_rstd_width = mean_rstd_shape_without_padding[-1];

    auto num_inner = compute_inner(output_grad_shape, normalized_dims);
    auto num_outer = compute_outer(output_grad_shape, normalized_dims);

    const bool gamma_grad_has_value = gamma_grad.has_value();
    const bool beta_grad_has_value = beta_grad.has_value();
    TT_FATAL(gamma_grad_has_value || beta_grad_has_value, "gamma_grad and beta_grad must have values");

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            split_work_to_cores(grid, num_inner);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                  // output_grad(==dy)
    const uint32_t in1_t = 1;                  // input(==x)
    const uint32_t in2_t = 1;                  // mean
    const uint32_t in3_t = 1;                  // rstd
    const uint32_t in4_t = 1;                  // scaler
    const uint32_t in5_t = do_mask_h ? 1 : 0;  // mask_h

    const uint32_t out0_t = 1;  // gamma_grad(==dgamma)
    const uint32_t out1_t = 1;  // beta_grad(==dbeta)

    const uint32_t im0_t = 1;  // output(==y)
    const uint32_t im1_t = 1;  // y * dy
    const uint32_t im2_t = 1;  // Add[dy]
    const uint32_t im3_t = 1;  // Add[y * dy]
    const uint32_t im4_t = 1;  // x - mean
    const uint32_t im5_t = 1;  // dycopy

    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const auto intermed_single_tile_size = tt::tile_size(intermed_cb_format);

    ProgramDescriptor desc;

    push_cb(desc, in0_t, all_cores, tt::CBIndex::c_0, cb_data_format, single_tile_size);    // output_grad(==dy)
    push_cb(desc, in1_t, all_cores, tt::CBIndex::c_1, cb_data_format, single_tile_size);    // input(==x)
    push_cb(desc, in2_t, all_cores, tt::CBIndex::c_2, cb_data_format, single_tile_size);    // mean
    push_cb(desc, in3_t, all_cores, tt::CBIndex::c_3, cb_data_format, single_tile_size);    // rstd
    push_cb(desc, in4_t, all_cores, tt::CBIndex::c_4, cb_data_format, single_tile_size);    // scaler
    push_cb(desc, in5_t, all_cores, tt::CBIndex::c_5, cb_data_format, single_tile_size);    // mask_h
    push_cb(desc, out0_t, all_cores, tt::CBIndex::c_16, cb_data_format, single_tile_size);  // gamma_grad
    push_cb(desc, out1_t, all_cores, tt::CBIndex::c_17, cb_data_format, single_tile_size);  // beta_grad
    push_cb(desc, im0_t, all_cores, tt::CBIndex::c_24, intermed_cb_format, intermed_single_tile_size);  // y
    push_cb(desc, im1_t, all_cores, tt::CBIndex::c_25, intermed_cb_format, intermed_single_tile_size);  // y*dy
    push_cb(desc, im2_t, all_cores, tt::CBIndex::c_26, intermed_cb_format, intermed_single_tile_size);  // Add[dy]
    push_cb(desc, im3_t, all_cores, tt::CBIndex::c_27, intermed_cb_format, intermed_single_tile_size);  // Add[y*dy]
    push_cb(desc, im4_t, all_cores, tt::CBIndex::c_28, intermed_cb_format, intermed_single_tile_size);  // x-mean
    push_cb(desc, im5_t, all_cores, tt::CBIndex::c_29, intermed_cb_format, intermed_single_tile_size);  // dycopy

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args{
        static_cast<uint32_t>(gamma_grad_has_value), static_cast<uint32_t>(do_mask_h)};
    TensorAccessorArgs(output_grad.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(mean.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(rstd.buffer()).append_to(reader_ct_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines compute_defines;
    compute_defines.emplace_back("REDUCE_OP", "PoolType::SUM");
    compute_defines.emplace_back("REDUCE_DIM", "ReduceDim::REDUCE_COL");
    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores);

    KernelDescriptor::CompileTimeArgs writer_ct_args{
        static_cast<uint32_t>(gamma_grad_has_value), static_cast<uint32_t>(beta_grad_has_value)};
    TensorAccessorArgs(gamma_grad.has_value() ? gamma_grad->buffer() : nullptr).append_to(writer_ct_args);
    TensorAccessorArgs(beta_grad.has_value() ? beta_grad->buffer() : nullptr).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = std::move(all_cores);
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    ComputeConfigDescriptor compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {
        num_cols_per_core_group_1,
        origin_H,
        origin_W,
        num_outer,
        num_inner,
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(beta_grad_has_value),
        static_cast<uint32_t>(is_lastdim_layer_norm),
        static_cast<uint32_t>(is_groupnorm)};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = compute_config;

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = COMPUTE_KERNEL_PATH;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {
            num_cols_per_core_group_2,
            origin_H,
            origin_W,
            num_outer,
            num_inner,
            static_cast<uint32_t>(gamma_grad_has_value),
            static_cast<uint32_t>(beta_grad_has_value),
            static_cast<uint32_t>(is_lastdim_layer_norm),
            static_cast<uint32_t>(is_groupnorm)};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = compute_config;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_addr = input.buffer()->address();
    const auto mean_addr = mean.buffer()->address();
    const auto rstd_addr = rstd.buffer()->address();

    const auto gamma_grad_addr = gamma_grad_has_value ? gamma_grad.value().buffer()->address() : 0;
    const auto beta_grad_addr = beta_grad_has_value ? beta_grad.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                output_grad_addr,
                input_addr,
                mean_addr,
                rstd_addr,
                num_cols_per_core,
                num_outer,
                num_inner,
                tile_offset,
                mask_h,
                normalized_dims,
                mean_rstd_height,
                mean_rstd_width});

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{gamma_grad_addr, beta_grad_addr, num_cols_per_core, tile_offset});

        tile_offset += num_cols_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad
