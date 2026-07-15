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

tt::tt_metal::ProgramDescriptor
MorehLayerNormBackwardGammaBetaGradOperation::MorehLayerNormBackwardGammaBetaGradFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const std::optional<const Tensor>& gamma_grad = output_tensor.at(0);
    const std::optional<const Tensor>& beta_grad = output_tensor.at(1);

    auto normalized_dims = operation_attributes.normalized_dims;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    using namespace tt::constants;
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

    const auto mean_rstd_shape = mean.padded_shape();
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
            tt::tt_metal::split_work_to_cores(grid, num_inner);

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

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;

    ProgramDescriptor desc;

    auto push_cb = [&](uint8_t cb_index, uint32_t num_tiles, tt::DataFormat fmt) {
        if (num_tiles == 0) {
            // Preserve original behavior: skip zero-size CBs.
            return;
        }
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * tile_size(fmt),
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_index,
                .data_format = fmt,
                .page_size = tile_size(fmt),
            }}},
        });
    };

    push_cb(static_cast<uint8_t>(CBIndex::c_0), in0_t, cb_data_format);       // output_grad(==dy)
    push_cb(static_cast<uint8_t>(CBIndex::c_1), in1_t, cb_data_format);       // input(==x)
    push_cb(static_cast<uint8_t>(CBIndex::c_2), in2_t, cb_data_format);       // mean
    push_cb(static_cast<uint8_t>(CBIndex::c_3), in3_t, cb_data_format);       // rstd
    push_cb(static_cast<uint8_t>(CBIndex::c_4), in4_t, cb_data_format);       // scaler
    push_cb(static_cast<uint8_t>(CBIndex::c_5), in5_t, cb_data_format);       // mask_h
    push_cb(static_cast<uint8_t>(CBIndex::c_16), out0_t, cb_data_format);     // gamma_grad(==dgamma)
    push_cb(static_cast<uint8_t>(CBIndex::c_17), out1_t, cb_data_format);     // beta_grad(==dbeta)
    push_cb(static_cast<uint8_t>(CBIndex::c_24), im0_t, intermed_cb_format);  // output(==y)
    push_cb(static_cast<uint8_t>(CBIndex::c_25), im1_t, intermed_cb_format);  // y * dy
    push_cb(static_cast<uint8_t>(CBIndex::c_26), im2_t, intermed_cb_format);  // Add[dy]
    push_cb(static_cast<uint8_t>(CBIndex::c_27), im3_t, intermed_cb_format);  // Add[y * dy]
    push_cb(static_cast<uint8_t>(CBIndex::c_28), im4_t, intermed_cb_format);  // x - mean
    push_cb(static_cast<uint8_t>(CBIndex::c_29), im5_t, intermed_cb_format);  // dycopy

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{
        static_cast<uint32_t>(gamma_grad_has_value), static_cast<uint32_t>(do_mask_h)};
    TensorAccessorArgs(output_grad.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(mean.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(rstd.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{
        static_cast<uint32_t>(gamma_grad_has_value), static_cast<uint32_t>(beta_grad_has_value)};
    TensorAccessorArgs(gamma_grad.has_value() ? gamma_grad->buffer() : nullptr).append_to(writer_compile_time_args);
    TensorAccessorArgs(beta_grad.has_value() ? beta_grad->buffer() : nullptr).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines_map{};
    std::map<std::string, std::string> compute_defines_map{};
    compute_defines_map["REDUCE_OP"] = "PoolType::SUM";
    compute_defines_map["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";
    if (fp32_dest_acc_en) {
        reader_defines_map["FP32_DEST_ACC_EN"] = "1";
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
    }

    KernelDescriptor::Defines reader_defines(reader_defines_map.begin(), reader_defines_map.end());
    KernelDescriptor::Defines compute_defines(compute_defines_map.begin(), compute_defines_map.end());

    const auto* const reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
        "reader_moreh_layer_norm_backward_gamma_beta_grad.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
        "writer_moreh_layer_norm_backward_gamma_beta_grad.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = reader_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
        "moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
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
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_2;
    const bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = compute_kernel_file;
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
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto* const output_grad_buf = output_grad.buffer();
    auto* const input_buf = input.buffer();
    auto* const mean_buf = mean.buffer();
    auto* const rstd_buf = rstd.buffer();

    // Pass the output grad tensors as Buffer* (not raw ->address()) so the program-cache fast hit
    // path patches their addresses when they are reallocated across calls. nullptr is valid for an
    // absent optional output.
    auto* const gamma_grad_buf = gamma_grad_has_value ? gamma_grad.value().buffer() : nullptr;
    auto* const beta_grad_buf = beta_grad_has_value ? beta_grad.value().buffer() : nullptr;

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

        reader_desc.emplace_runtime_args(
            core,
            {output_grad_buf,
             input_buf,
             mean_buf,
             rstd_buf,
             num_cols_per_core,
             num_outer,
             num_inner,
             tile_offset,
             mask_h,
             normalized_dims,
             mean_rstd_height,
             mean_rstd_width});

        writer_desc.emplace_runtime_args(core, {gamma_grad_buf, beta_grad_buf, num_cols_per_core, tile_offset});

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
