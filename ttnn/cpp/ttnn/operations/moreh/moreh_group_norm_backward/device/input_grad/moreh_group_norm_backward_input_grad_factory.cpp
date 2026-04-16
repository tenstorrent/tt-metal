// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "moreh_group_norm_backward_input_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

// Helper: only add a CB descriptor when num_tiles > 0
inline void push_cb_input_grad(
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

ProgramDescriptor MorehGroupNormBackwardInputGradOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const auto& input_grad = outputs;
    auto gamma = tensor_args.gamma;
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

    const bool is_lastdim_layernorm = false;
    const bool is_groupnorm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const bool gamma_has_value = gamma.has_value();

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
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = split_work_to_cores(grid, num_rows);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_rows_per_core_group_1: {}", num_rows_per_core_group_1);
    log_debug(LogTest, "num_rows_per_core_group_2: {}", num_rows_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{1};
    const uint32_t in1_t{1};
    const uint32_t in2_t{1};
    const uint32_t in3_t{1};
    const uint32_t in4_t{1};
    const uint32_t in5_t{2};
    const uint32_t in6_t = gamma_has_value ? 1 : 0;
    const uint32_t in7_t = (do_mask_h || do_mask_w) ? 2 : 0;

    const uint32_t out0_t{1};

    uint32_t im0_t{num_inner_tiles};
    uint32_t im1_t{num_inner_tiles};
    const uint32_t im2_t{1};
    const uint32_t im3_t{1};
    const uint32_t im4_t{1};
    const uint32_t im5_t{1};
    const uint32_t im6_t{1};
    uint32_t im7_t{1};

    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);

    const auto cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + in7_t + out0_t + im0_t + im1_t +
                           im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                          single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_layer_norm_backward_input_grad algorithm is selected.");
        im0_t = 1;
        im1_t = 1;
        im7_t = 0;
    } else {
        log_info(LogTest, "Small moreh_layer_norm_backward_input_grad algorithm is selected.");
    }

    ProgramDescriptor desc;

    push_cb_input_grad(desc, in0_t, all_cores, tt::CBIndex::c_0, cb_data_format, single_tile_size);  // output_grad
    push_cb_input_grad(desc, in1_t, all_cores, tt::CBIndex::c_1, cb_data_format, single_tile_size);  // input
    push_cb_input_grad(desc, in2_t, all_cores, tt::CBIndex::c_2, cb_data_format, single_tile_size);  // mean
    push_cb_input_grad(desc, in3_t, all_cores, tt::CBIndex::c_3, cb_data_format, single_tile_size);  // rstd
    push_cb_input_grad(desc, in4_t, all_cores, tt::CBIndex::c_4, cb_data_format, single_tile_size);  // one
    push_cb_input_grad(desc, in5_t, all_cores, tt::CBIndex::c_5, cb_data_format, single_tile_size);  // inner_size(==n)
    push_cb_input_grad(desc, in6_t, all_cores, tt::CBIndex::c_6, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, in7_t, all_cores, tt::CBIndex::c_7, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, out0_t, all_cores, tt::CBIndex::c_16, cb_data_format, single_tile_size);  // input_grad
    push_cb_input_grad(desc, im0_t, all_cores, tt::CBIndex::c_24, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, im1_t, all_cores, tt::CBIndex::c_25, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, im2_t, all_cores, tt::CBIndex::c_26, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, im3_t, all_cores, tt::CBIndex::c_27, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, im4_t, all_cores, tt::CBIndex::c_28, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, im5_t, all_cores, tt::CBIndex::c_29, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, im6_t, all_cores, tt::CBIndex::c_30, cb_data_format, single_tile_size);
    push_cb_input_grad(desc, im7_t, all_cores, tt::CBIndex::c_31, cb_data_format, single_tile_size);

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const char* reader_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
              "reader_moreh_group_norm_backward_input_grad_large.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
              "reader_moreh_group_norm_backward_input_grad_small.cpp";

    static constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
        "writer_moreh_group_norm_backward_input_grad.cpp";

    KernelDescriptor::CompileTimeArgs reader_ct_args{static_cast<uint32_t>(gamma_has_value)};
    TensorAccessorArgs(output_grad.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(mean.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(rstd.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(gamma_has_value ? gamma->buffer() : nullptr).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores_to_be_used);

    KernelDescriptor::CompileTimeArgs writer_ct_args;
    TensorAccessorArgs(input_grad.buffer()).append_to(writer_ct_args);

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

    const char* compute_kernel_file = use_large_algorithm
                                          ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                                            "moreh_layer_norm_backward_input_grad_large_kernel.cpp"
                                          : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                                            "moreh_layer_norm_backward_input_grad_small_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{};

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {
            num_rows_per_core_group_2,
            origin_h,
            origin_w,
            num_inner_tiles,
            static_cast<uint32_t>(gamma_has_value),
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

    const auto input_grad_addr = input_grad.buffer()->address();

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
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
                gamma_addr,
                tile_offset,
                num_rows_per_core,
                num_inner_tiles,
                num_channels,
                num_groups,
                origin_h,
                origin_w});

        // writer
        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{input_grad_addr, tile_offset, num_rows_per_core, num_inner_tiles});

        tile_offset += num_rows_per_core * num_inner_tiles;
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
