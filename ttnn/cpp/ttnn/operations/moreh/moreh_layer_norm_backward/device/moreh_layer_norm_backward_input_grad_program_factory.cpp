// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <string>

#include "moreh_layer_norm_backward_input_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad {

tt::tt_metal::ProgramDescriptor
MorehLayerNormBackwardInputGradOperation::MorehLayerNormBackwardInputGradFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    auto normalized_dims = operation_attributes.normalized_dims;

    const std::optional<const Tensor>& gamma = tensor_args.gamma;

    auto compute_kernel_config =
        init_device_compute_kernel_config(output_grad.device()->arch(), operation_attributes.compute_kernel_config);

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
    const auto output_grad_rank = output_grad_shape.rank();

    const bool is_lastdim_layer_norm = normalized_dims == 1;
    const bool is_groupnorm = false;

    const auto origin_H = output_grad_shape_without_padding[-2];
    const auto origin_W = output_grad_shape_without_padding[-1];

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layer_norm;
    const uint32_t mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    const uint32_t mask_w = do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH;

    const auto mean_rstd_shape = mean.padded_shape();
    const auto mean_rstd_shape_without_padding = mean.logical_shape();
    auto mean_rstd_height = mean_rstd_shape_without_padding[-2];
    auto mean_rstd_width = mean_rstd_shape_without_padding[-1];

    auto normalized_numel = 1.0f;
    for (uint32_t i = output_grad_rank - normalized_dims; i < output_grad_rank; i++) {
        auto size = output_grad_shape_without_padding[i];
        normalized_numel *= size;
    }

    auto n = static_cast<float>(normalized_numel);
    auto recip_n = 1.0f / n;

    auto num_inner = compute_inner(output_grad_shape, normalized_dims);
    auto num_outer = compute_outer(output_grad_shape, normalized_dims);

    const bool gamma_has_value = gamma.has_value();

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_outer);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);
    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                                 // output_grad(==dy)
    const uint32_t in1_t = 1;                                 // input(==x)
    const uint32_t in2_t = 1;                                 // mean
    const uint32_t in3_t = 1;                                 // rstd
    const uint32_t in4_t = 1;                                 // scaler
    const uint32_t in5_t = 2;                                 // n_recip_n
    const uint32_t in6_t = gamma_has_value ? 1 : 0;           // gamma
    const uint32_t in7_t = (do_mask_h || do_mask_w) ? 2 : 0;  // mask_h_w

    // dx = ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
    const uint32_t out0_t = 1;  // input_grad(==dx)

    uint32_t im0_t = num_inner;  // copy output_grad(==dycopy)
    uint32_t im1_t = num_inner;  // output(==y)
    const uint32_t im2_t = 1;    // Sum[dy]
    const uint32_t im3_t = 1;    // Sum[y * dy]
    const uint32_t im4_t = 1;    // rstd / n

    const uint32_t im5_t = 1;
    const uint32_t im6_t = 1;
    uint32_t im7_t = 1;

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const auto intermed_single_tile_size = tt::tile_size(intermed_cb_format);

    const uint32_t cb_usage =
        ((in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + in7_t + out0_t) * single_tile_size) +
        ((im0_t + im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) * intermed_single_tile_size);
    const uint32_t available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(tt::LogTest, "Large moreh_layer_norm_backward_input_grad algorithm is selected.");
        im0_t = 1;
        im1_t = 1;
        im7_t = 0;
    } else {
        log_info(tt::LogTest, "Small moreh_layer_norm_backward_input_grad algorithm is selected.");
    }

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
    push_cb(static_cast<uint8_t>(CBIndex::c_5), in5_t, cb_data_format);       // n_recip_n
    push_cb(static_cast<uint8_t>(CBIndex::c_6), in6_t, cb_data_format);       // gamma
    push_cb(static_cast<uint8_t>(CBIndex::c_7), in7_t, cb_data_format);       // mask_h_w
    push_cb(static_cast<uint8_t>(CBIndex::c_16), out0_t, cb_data_format);     // input_grad(==dx)
    push_cb(static_cast<uint8_t>(CBIndex::c_24), im0_t, intermed_cb_format);  // copy output_grad(==dy or dy * gamma)
    push_cb(static_cast<uint8_t>(CBIndex::c_25), im1_t, intermed_cb_format);  // output(==y)
    push_cb(static_cast<uint8_t>(CBIndex::c_26), im2_t, intermed_cb_format);  // Sum[dy]
    push_cb(static_cast<uint8_t>(CBIndex::c_27), im3_t, intermed_cb_format);  // Sum[y * dy]
    push_cb(static_cast<uint8_t>(CBIndex::c_28), im4_t, intermed_cb_format);  // rstd / n
    push_cb(static_cast<uint8_t>(CBIndex::c_29), im5_t, intermed_cb_format);
    push_cb(static_cast<uint8_t>(CBIndex::c_30), im6_t, intermed_cb_format);
    push_cb(static_cast<uint8_t>(CBIndex::c_31), im7_t, intermed_cb_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{
        static_cast<uint32_t>(gamma_has_value), static_cast<uint32_t>(do_mask_h), static_cast<uint32_t>(do_mask_w)};
    TensorAccessorArgs(output_grad.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(mean.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(rstd.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gamma.has_value() ? gamma->buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(input_grad.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines_map{};
    std::map<std::string, std::string> compute_defines_map{};
    compute_defines_map["REDUCE_OP"] = "PoolType::AVG";
    if (is_lastdim_layer_norm) {
        compute_defines_map["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    } else {
        compute_defines_map["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";
    }
    if (fp32_dest_acc_en) {
        reader_defines_map["FP32_DEST_ACC_EN"] = "1";
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
    }

    KernelDescriptor::Defines reader_defines(reader_defines_map.begin(), reader_defines_map.end());
    KernelDescriptor::Defines compute_defines(compute_defines_map.begin(), compute_defines_map.end());

    const auto* const reader_kernel_file =
        use_large_algorithm ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                              "reader_moreh_layer_norm_backward_input_grad_large.cpp"
                            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                              "reader_moreh_layer_norm_backward_input_grad_small.cpp";

    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
        "writer_moreh_layer_norm_backward_input_grad.cpp";

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
        use_large_algorithm ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                              "moreh_layer_norm_backward_input_grad_large_kernel.cpp"
                            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                              "moreh_layer_norm_backward_input_grad_small_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {
        num_rows_per_core_group_1,
        origin_H,
        origin_W,
        num_inner,
        static_cast<uint32_t>(gamma_has_value),
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
            num_rows_per_core_group_2,
            origin_H,
            origin_W,
            num_inner,
            static_cast<uint32_t>(gamma_has_value),
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

    // Pass gamma as Buffer* (not raw ->address()) so the program-cache fast hit path patches its
    // address when gamma is reallocated across calls. nullptr is valid for the absent optional.
    auto* const gamma_buf = gamma_has_value ? gamma.value().buffer() : nullptr;

    auto* const input_grad_buf = input_grad.buffer();

    const uint32_t n_u = std::bit_cast<uint32_t>(n);
    const uint32_t recip_n_u = std::bit_cast<uint32_t>(recip_n);

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.emplace_runtime_args(
            core,
            {output_grad_buf,
             input_buf,
             mean_buf,
             rstd_buf,
             gamma_buf,
             num_rows_per_core,
             num_inner,
             tile_offset,
             n_u,
             recip_n_u,
             mask_h,
             mask_w,
             normalized_dims,
             mean_rstd_height,
             mean_rstd_width});

        writer_desc.emplace_runtime_args(core, {input_grad_buf, num_rows_per_core, num_inner, tile_offset});

        tile_offset += num_rows_per_core * num_inner;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad
