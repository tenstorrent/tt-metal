// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

inline uint32_t get_block_size(uint32_t num_tiles, uint32_t max_block_size) {
    uint32_t block_size{1};
    for (uint32_t current_block_size = max_block_size; current_block_size >= 1; current_block_size >>= 1) {
        if (num_tiles % current_block_size == 0) {
            block_size = current_block_size;
            break;
        }
    }
    return block_size;
}

namespace ttnn::operations::moreh::moreh_group_norm {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

// Helper: only add a CB descriptor when num_tiles > 0 (mirrors moreh helper behavior)
static void push_cb_if_nonzero(
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

ProgramDescriptor MorehGroupNormOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    const auto& input = tensor_args.input;
    auto gamma = tensor_args.gamma;
    auto beta = tensor_args.beta;
    auto mean = outputs[1];
    auto rstd = outputs[2];

    auto& output = outputs[0].value();

    auto num_groups = operation_attributes.num_groups;
    auto eps = operation_attributes.eps;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();

    const auto n = input_shape[0];
    const auto c = input_shape[1];
    const auto h = input_shape[2];
    const auto w = input_shape[3];

    const auto origin_input_shape = input.logical_shape();

    const auto origin_h = origin_input_shape[2];
    const auto origin_w = origin_input_shape[3];

    const bool is_lastdim_layernorm = false;
    const bool is_group_norm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    TT_FATAL(
        num_channels % num_groups == 0,
        "Group norm requires num_channels ({}) to be divisible by num_groups ({})",
        num_channels,
        num_groups);
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const auto f_c = static_cast<float>(num_channels) / static_cast<float>(num_groups);
    const auto f_ht = static_cast<float>(origin_h) / static_cast<float>(TILE_HEIGHT);
    const auto f_wt = static_cast<float>(origin_w) / static_cast<float>(TILE_WIDTH);
    auto scaler = 1.0f / (static_cast<float>(TILE_WIDTH) * sqrt(f_c * f_ht * f_wt));

    const bool gamma_has_value = gamma.has_value();
    const bool beta_has_value = beta.has_value();
    const bool mean_has_value = mean.has_value();
    const bool rstd_has_value = rstd.has_value();

    constexpr uint32_t MAX_BLOCK_SIZE = 8;
    const uint32_t block_size = get_block_size(num_inner_tiles, MAX_BLOCK_SIZE);

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
    log_debug(LogTest, "block_size: {}", block_size);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner_tiles;                         // input
    const uint32_t in1_t = 1;                                 // scaler
    const uint32_t in2_t = 1;                                 // epsilon
    const uint32_t in3_t = gamma_has_value ? block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                 // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                 // mask_w

    const uint32_t out0_t = block_size;              // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = num_inner_tiles;                                                 // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);

    const auto cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t + im0_t +
                           im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                          single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_group_norm algorithm is selected.");
        in0_t = block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(LogTest, "Small moreh_group_norm algorithm is selected.");
    }

    ProgramDescriptor desc;

    // Push CBs — only create when num_tiles > 0 (mirrors CreateCircularBuffer helper behavior)
    push_cb_if_nonzero(desc, in0_t, all_cores, tt::CBIndex::c_0, cb_data_format, single_tile_size);    // input
    push_cb_if_nonzero(desc, in1_t, all_cores, tt::CBIndex::c_1, cb_data_format, single_tile_size);    // scaler
    push_cb_if_nonzero(desc, in2_t, all_cores, tt::CBIndex::c_2, cb_data_format, single_tile_size);    // eps
    push_cb_if_nonzero(desc, in3_t, all_cores, tt::CBIndex::c_3, cb_data_format, single_tile_size);    // gamma
    push_cb_if_nonzero(desc, in4_t, all_cores, tt::CBIndex::c_4, cb_data_format, single_tile_size);    // beta
    push_cb_if_nonzero(desc, in5_t, all_cores, tt::CBIndex::c_5, cb_data_format, single_tile_size);    // mask_h
    push_cb_if_nonzero(desc, in6_t, all_cores, tt::CBIndex::c_6, cb_data_format, single_tile_size);    // mask_w
    push_cb_if_nonzero(desc, out0_t, all_cores, tt::CBIndex::c_16, cb_data_format, single_tile_size);  // output
    push_cb_if_nonzero(desc, out1_t, all_cores, tt::CBIndex::c_17, cb_data_format, single_tile_size);  // mean
    push_cb_if_nonzero(desc, out2_t, all_cores, tt::CBIndex::c_18, cb_data_format, single_tile_size);  // rstd
    push_cb_if_nonzero(desc, im0_t, all_cores, tt::CBIndex::c_24, cb_data_format, single_tile_size);   // E[x]
    push_cb_if_nonzero(desc, im1_t, all_cores, tt::CBIndex::c_25, cb_data_format, single_tile_size);   // x - E[x]
    push_cb_if_nonzero(desc, im2_t, all_cores, tt::CBIndex::c_26, cb_data_format, single_tile_size);   // (x - E[x])^2
    push_cb_if_nonzero(
        desc, im3_t, all_cores, tt::CBIndex::c_27, cb_data_format, single_tile_size);  // Sum[(x - E[x])^2]
    push_cb_if_nonzero(desc, im4_t, all_cores, tt::CBIndex::c_28, cb_data_format, single_tile_size);  // Var[x]
    push_cb_if_nonzero(desc, im5_t, all_cores, tt::CBIndex::c_29, cb_data_format, single_tile_size);  // 1/sqrt(Var+eps)
    push_cb_if_nonzero(desc, im6_t, all_cores, tt::CBIndex::c_30, cb_data_format, single_tile_size);  // y*gamma+beta
    push_cb_if_nonzero(desc, im7_t, all_cores, tt::CBIndex::c_31, cb_data_format, single_tile_size);  // Sum[x]

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const char* reader_kernel_file = use_large_algorithm ? "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/"
                                                           "kernels/dataflow/reader_moreh_group_norm_large.cpp"
                                                         : "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/"
                                                           "kernels/dataflow/reader_moreh_group_norm_small.cpp";

    static constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/writer_moreh_group_norm.cpp";

    KernelDescriptor::CompileTimeArgs reader_ct_args{
        static_cast<uint32_t>(gamma_has_value), static_cast<uint32_t>(beta_has_value)};
    TensorAccessorArgs(input.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(gamma_has_value ? gamma->buffer() : nullptr).append_to(reader_ct_args);
    TensorAccessorArgs(beta_has_value ? beta->buffer() : nullptr).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores_to_be_used);

    KernelDescriptor::CompileTimeArgs writer_ct_args{
        static_cast<uint32_t>(mean_has_value), static_cast<uint32_t>(rstd_has_value)};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(mean_has_value ? mean.value().buffer() : nullptr).append_to(writer_ct_args);
    TensorAccessorArgs(rstd_has_value ? rstd.value().buffer() : nullptr).append_to(writer_ct_args);

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

    const char* compute_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {
        num_rows_per_core_group_1,
        origin_h,
        origin_w,
        num_inner_tiles,
        block_size,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_group_norm)};
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
            block_size,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_group_norm)};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{};
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();

    const auto output_addr = output.buffer()->address();
    const auto mean_addr = mean_has_value ? mean.value().buffer()->address() : 0;
    const auto rstd_addr = rstd_has_value ? rstd.value().buffer()->address() : 0;

    const auto gamma_addr = gamma_has_value ? gamma.value().buffer()->address() : 0;
    const auto beta_addr = beta_has_value ? beta.value().buffer()->address() : 0;

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
                input_addr,
                gamma_addr,
                beta_addr,
                *reinterpret_cast<uint32_t*>(&scaler),
                *reinterpret_cast<uint32_t*>(&eps),
                tile_offset,
                num_rows_per_core,
                num_inner_tiles,
                num_channels,
                origin_h,
                origin_w,
                block_size});

        // writer
        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                output_addr,
                mean_addr,
                rstd_addr,
                tile_offset,
                num_rows_per_core,
                num_inner_tiles,
                num_groups,
                block_size});

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

}  // namespace ttnn::operations::moreh::moreh_group_norm
