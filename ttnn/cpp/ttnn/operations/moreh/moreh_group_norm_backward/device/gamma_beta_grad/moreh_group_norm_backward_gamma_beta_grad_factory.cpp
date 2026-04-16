// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

inline void push_cb_gamma_beta(
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
                .buffer_index = cb_index, .data_format = data_format, .page_size = tile_size}}},
        });
    }
}

static constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
    "reader_moreh_group_norm_backward_gamma_beta_grad.cpp";
static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
    "writer_moreh_group_norm_backward_gamma_beta_grad.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp";

ProgramDescriptor MorehGroupNormBackwardGammaBetaGradOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
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
         num_channels_per_core_group_2] = split_work_to_cores(grid, num_channels);

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

    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);

    ProgramDescriptor desc;

    push_cb_gamma_beta(
        desc, in0_t, all_cores, tt::CBIndex::c_0, cb_data_format, single_tile_size);  // output_grad(==dy)
    push_cb_gamma_beta(desc, in1_t, all_cores, tt::CBIndex::c_1, cb_data_format, single_tile_size);  // input(==x)
    push_cb_gamma_beta(desc, in2_t, all_cores, tt::CBIndex::c_2, cb_data_format, single_tile_size);  // mean
    push_cb_gamma_beta(desc, in3_t, all_cores, tt::CBIndex::c_3, cb_data_format, single_tile_size);  // rstd
    push_cb_gamma_beta(desc, in4_t, all_cores, tt::CBIndex::c_4, cb_data_format, single_tile_size);  // one
    push_cb_gamma_beta(desc, in5_t, all_cores, tt::CBIndex::c_5, cb_data_format, single_tile_size);  // mask_h
    push_cb_gamma_beta(desc, in6_t, all_cores, tt::CBIndex::c_6, cb_data_format, single_tile_size);  // mask_w
    push_cb_gamma_beta(
        desc, out0_t, all_cores, tt::CBIndex::c_16, cb_data_format, single_tile_size);  // gamma_grad(==dgamma)
    push_cb_gamma_beta(
        desc, out1_t, all_cores, tt::CBIndex::c_17, cb_data_format, single_tile_size);  // beta_grad(==dbeta)
    push_cb_gamma_beta(desc, im0_t, all_cores, tt::CBIndex::c_24, cb_data_format, single_tile_size);  // output(==y)
    push_cb_gamma_beta(desc, im1_t, all_cores, tt::CBIndex::c_25, cb_data_format, single_tile_size);  // y * dy
    push_cb_gamma_beta(desc, im2_t, all_cores, tt::CBIndex::c_26, cb_data_format, single_tile_size);  // Add[dy]
    push_cb_gamma_beta(desc, im3_t, all_cores, tt::CBIndex::c_27, cb_data_format, single_tile_size);  // Add[y * dy]
    push_cb_gamma_beta(desc, im4_t, all_cores, tt::CBIndex::c_28, cb_data_format, single_tile_size);  // x - mean
    push_cb_gamma_beta(desc, im5_t, all_cores, tt::CBIndex::c_29, cb_data_format, single_tile_size);  // dycopy

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_ct_args{static_cast<uint32_t>(gamma_grad_has_value)};
    TensorAccessorArgs(output_grad.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(mean.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(rstd.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores_to_be_used);

    KernelDescriptor::CompileTimeArgs writer_ct_args{
        static_cast<uint32_t>(gamma_grad_has_value), static_cast<uint32_t>(beta_grad_has_value)};
    TensorAccessorArgs(gamma_grad_has_value ? gamma_grad.value().buffer() : nullptr).append_to(writer_ct_args);
    TensorAccessorArgs(beta_grad_has_value ? beta_grad.value().buffer() : nullptr).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = std::move(all_cores);
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_to_be_used);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::Defines compute_defines = {
        {"REDUCE_OP", "PoolType::SUM"}, {"REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"}};

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {
        num_channels_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        Wt,
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(beta_grad_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{};

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = COMPUTE_KERNEL_PATH;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {
            num_channels_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            Wt,
            static_cast<uint32_t>(gamma_grad_has_value),
            static_cast<uint32_t>(beta_grad_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{};
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

        // reader
        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                output_grad_addr,
                input_addr,
                mean_addr,
                rstd_addr,
                tile_offset,
                num_channels_per_core,
                num_inner_tiles,
                num_channels,
                num_groups,
                origin_h,
                origin_w});

        // writer
        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                gamma_grad_addr, beta_grad_addr, tile_offset, num_channels_per_core, num_inner_tiles, batch});

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
