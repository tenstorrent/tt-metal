// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {

tt::tt_metal::ProgramDescriptor
MorehGroupNormBackwardGammaBetaGradOperation::MorehGroupNormBackwardGammaBetaGradFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    using namespace tt;
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const auto& gamma_grad = outputs[0];
    const auto& beta_grad = outputs[1];
    auto num_groups = operation_attributes.num_groups;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.padded_shape();

    const auto n = output_grad_shape[0];
    const auto c = output_grad_shape[1];
    const auto h = output_grad_shape[2];
    const auto w = output_grad_shape[3];

    const auto origin_output_grad_shape = output_grad.logical_shape();

    const auto origin_h = origin_output_grad_shape[2];
    const auto origin_w = origin_output_grad_shape[3];

    const bool is_groupnorm = true;
    const bool is_lastdim_layernorm = false;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;

    const auto batch = n;
    const auto HtWt = Ht * Wt;
    const auto num_inner_tiles = batch * HtWt;  // inner_size

    const bool gamma_grad_has_value = gamma_grad.has_value();
    const bool beta_grad_has_value = beta_grad.has_value();

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
         num_channels_per_core_group_1,
         num_channels_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_channels);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_channels_per_core_group_1: {}", num_channels_per_core_group_1);
    log_debug(LogTest, "num_channels_per_core_group_2: {}", num_channels_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                  // output_grad(==dy)
    const uint32_t in1_t = 1;                  // input(==x)
    const uint32_t in2_t = 1;                  // mean
    const uint32_t in3_t = 1;                  // rstd
    const uint32_t in4_t = 1;                  // one
    const uint32_t in5_t = do_mask_h ? 1 : 0;  // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;  // mask_w

    const uint32_t out0_t = gamma_grad_has_value ? 1 : 0;  // gamma_grad(==dgamma)
    const uint32_t out1_t = beta_grad_has_value ? 1 : 0;   // beta_grad(==dbeta)

    const uint32_t im0_t = 1;  // output(==y)
    const uint32_t im1_t = 1;  // y * dy
    const uint32_t im2_t = 1;  // Add[dy]
    const uint32_t im3_t = 1;  // Add[y * dy]
    const uint32_t im4_t = 1;  // x - mean
    const uint32_t im5_t = 1;  // dycopy

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);

    ProgramDescriptor desc;

    auto add_cb = [&](CBIndex cb_index, uint32_t num_tiles) {
        if (num_tiles == 0) {
            return;
        }
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index),
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    };

    add_cb(CBIndex::c_0, in0_t);    // output_grad(==dy)
    add_cb(CBIndex::c_1, in1_t);    // input(==x)
    add_cb(CBIndex::c_2, in2_t);    // mean
    add_cb(CBIndex::c_3, in3_t);    // rstd
    add_cb(CBIndex::c_4, in4_t);    // one
    add_cb(CBIndex::c_5, in5_t);    // mask_h
    add_cb(CBIndex::c_6, in6_t);    // mask_w
    add_cb(CBIndex::c_16, out0_t);  // gamma_grad(==dgamma)
    add_cb(CBIndex::c_17, out1_t);  // beta_grad(==dbeta)
    add_cb(CBIndex::c_24, im0_t);   // output(==y)
    add_cb(CBIndex::c_25, im1_t);   // y * dy
    add_cb(CBIndex::c_26, im2_t);   // Add[dy]
    add_cb(CBIndex::c_27, im3_t);   // Add[y * dy]
    add_cb(CBIndex::c_28, im4_t);   // x - mean
    add_cb(CBIndex::c_29, im5_t);   // dycopy

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::string reader_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
        "reader_moreh_group_norm_backward_gamma_beta_grad.cpp");
    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
        "writer_moreh_group_norm_backward_gamma_beta_grad.cpp");

    KernelDescriptor::CompileTimeArgs reader_compile_time_args{static_cast<uint32_t>(gamma_grad_has_value)};
    TensorAccessorArgs(*output_grad.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*mean.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*rstd.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{
        static_cast<uint32_t>(gamma_grad_has_value), static_cast<uint32_t>(beta_grad_has_value)};
    TensorAccessorArgs(gamma_grad_has_value ? gamma_grad.value().buffer() : nullptr)
        .append_to(writer_compile_time_args);
    TensorAccessorArgs(beta_grad_has_value ? beta_grad.value().buffer() : nullptr).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::Defines compute_defines{
        {"REDUCE_OP", "PoolType::SUM"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"},
    };

    const std::string compute_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
        "moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp");

    auto make_compute_args = [&](uint32_t num_channels_per_core) -> KernelDescriptor::CompileTimeArgs {
        return {
            num_channels_per_core,
            origin_h,
            origin_w,
            num_inner_tiles,
            Wt,
            static_cast<uint32_t>(gamma_grad_has_value),
            static_cast<uint32_t>(beta_grad_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};
    };

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = make_compute_args(num_channels_per_core_group_1);
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{};

    bool has_core_group_2 = !core_group_2.ranges().empty();
    KernelDescriptor compute_desc_2;
    if (has_core_group_2) {
        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = make_compute_args(num_channels_per_core_group_2);
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{};
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto* const output_grad_buf = output_grad.buffer();
    auto* const input_buf = input.buffer();
    auto* const mean_buf = mean.buffer();
    auto* const rstd_buf = rstd.buffer();

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_channels_per_core;
        if (core_group_1.contains(core)) {
            num_channels_per_core = num_channels_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_channels_per_core = num_channels_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.emplace_runtime_args(
            core,
            {output_grad_buf,
             input_buf,
             mean_buf,
             rstd_buf,
             tile_offset,
             num_channels_per_core,
             num_inner_tiles,
             num_channels,
             num_groups,
             origin_h,
             origin_w});

        // writer
        KernelDescriptor::RTArgList writer_args;
        if (gamma_grad_has_value) {
            writer_args.push_back(gamma_grad.value().buffer());
        } else {
            writer_args.push_back(0u);
        }
        if (beta_grad_has_value) {
            writer_args.push_back(beta_grad.value().buffer());
        } else {
            writer_args.push_back(0u);
        }
        writer_args.push_back(tile_offset);
        writer_args.push_back(num_channels_per_core);
        writer_args.push_back(num_inner_tiles);
        writer_args.push_back(batch);
        writer_desc.emplace_runtime_args(core, writer_args);

        tile_offset += num_channels_per_core * HtWt;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
