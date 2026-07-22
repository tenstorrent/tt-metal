// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rmsnorm_post_all_gather_device_operation.hpp"

#include <bit>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor FusedRMSNormPostAllGatherProgramFactory::create_descriptor(
    const FusedRmsnormPostAllGatherParams& operation_attributes,
    const FusedRmsnormPostAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& stats_tensor = tensor_args.stats_tensor;
    const auto& weight_tensor = tensor_args.weight;
    const auto& transformation_mat = tensor_args.transformation_mat;
    const auto& rope_cos = tensor_args.rope_cos;
    const auto& rope_sin = tensor_args.rope_sin;

    const float eps = operation_attributes.eps;
    const uint32_t num_heads = operation_attributes.num_heads;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    const bool has_weight = weight_tensor.has_value();
    const bool fuse_rope = transformation_mat.has_value();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t W = input_shape[-1];
    const uint32_t H = input_shape[-2];

    const uint32_t num_tile_cols = W / TILE_WIDTH;
    const uint32_t num_tile_rows = H / TILE_HEIGHT;
    const uint32_t head_dim_tiles = num_tile_cols / num_heads;

    const uint32_t stats_tiles_cols = stats_tensor.padded_shape()[-1] / TILE_WIDTH;
    // AllGather results in a tensor with num_devices columns of tiles
    const uint32_t num_devices = stats_tiles_cols;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");

    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "num_tile_cols: {}", num_tile_cols);
    log_debug(tt::LogOp, "stats_tiles_cols: {}", stats_tiles_cols);
    log_debug(tt::LogOp, "num_devices: {}", num_devices);
    log_debug(tt::LogOp, "has_weight: {}", has_weight);
    log_debug(tt::LogOp, "fuse_rope: {}", fuse_rope);
    log_debug(tt::LogOp, "num_heads: {}", num_heads);
    log_debug(tt::LogOp, "head_dim_tiles: {}", head_dim_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    //////////////////////////////////////////////////////////////////////////
    IDevice* device = input_tensor.device();
    const auto grid_size = device->compute_with_storage_grid_size();
    const auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    const CoreRangeSet core_grid_set(core_grid);
    const uint32_t num_cores = core_grid.size();

    ////////////////////////////////////////////////////////////////////////////
    //                Circular Buffer Data Format Setup
    //////////////////////////////////////////////////////////////////////////
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t dst_reg_count = get_dest_reg_count(compute_kernel_config);

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat stats_data_format = tt::tt_metal::datatype_to_dataformat_converter(stats_tensor.dtype());
    tt::DataFormat intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat reduce_scalar_data_format =
        (input_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    tt::DataFormat weight_data_format =
        has_weight ? tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.value().dtype())
                   : tt::DataFormat::Float16_b;

    tt::DataFormat transformation_mat_data_format =
        fuse_rope ? tt::tt_metal::datatype_to_dataformat_converter(transformation_mat.value().dtype())
                  : tt::DataFormat::Float16_b;
    tt::DataFormat rope_cos_data_format = fuse_rope
                                              ? tt::tt_metal::datatype_to_dataformat_converter(rope_cos.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat rope_sin_data_format = fuse_rope
                                              ? tt::tt_metal::datatype_to_dataformat_converter(rope_sin.value().dtype())
                                              : tt::DataFormat::Float16_b;

    uint32_t input_tile_size = tt::tile_size(input_data_format);
    uint32_t output_tile_size = tt::tile_size(output_data_format);
    uint32_t intermediate_tile_size = tt::tile_size(intermediate_data_format);
    uint32_t reduce_scalar_tile_size = tt::tile_size(reduce_scalar_data_format);
    uint32_t stats_tile_size = tt::tile_size(stats_data_format);
    uint32_t weight_tile_size = tt::tile_size(weight_data_format);
    uint32_t transformation_mat_tile_size = tt::tile_size(transformation_mat_data_format);
    uint32_t rope_cos_tile_size = tt::tile_size(rope_cos_data_format);
    uint32_t rope_sin_tile_size = tt::tile_size(rope_sin_data_format);

    log_debug(tt::LogOp, "input_data_format: {}", input_data_format);
    log_debug(tt::LogOp, "output_data_format: {}", output_data_format);
    log_debug(tt::LogOp, "stats_data_format: {}", stats_data_format);
    log_debug(tt::LogOp, "intermediate_data_format: {}", intermediate_data_format);
    log_debug(tt::LogOp, "reduce_scalar_data_format: {}", reduce_scalar_data_format);
    log_debug(tt::LogOp, "weight_data_format: {}", weight_data_format);
    log_debug(tt::LogOp, "transformation_mat_data_format: {}", transformation_mat_data_format);
    log_debug(tt::LogOp, "rope_cos_data_format: {}", rope_cos_data_format);
    log_debug(tt::LogOp, "rope_sin_data_format: {}", rope_sin_data_format);

    // Optional tensor buffers passed as Buffer* so the framework patches their
    // addresses on cache hits (BufferBinding). Plain uint32_t goes stale when
    // the allocator moves the buffer on a subsequent dispatch (cf. #44565).
    Buffer* const weight_buffer = has_weight ? weight_tensor.value().buffer() : nullptr;
    Buffer* const transformation_mat_buffer = fuse_rope ? transformation_mat.value().buffer() : nullptr;
    Buffer* const rope_cos_buffer = fuse_rope ? rope_cos.value().buffer() : nullptr;
    Buffer* const rope_sin_buffer = fuse_rope ? rope_sin.value().buffer() : nullptr;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    //////////////////////////////////////////////////////////////////////////
    /*
    CB 0: input
    CB 1: stats
    CB 2: reduce scalar
    CB 3: epsilon
    CB 4: reduce result
    CB 5: output

    if (has_weight)
        CB 6: weight

    if (has_weight or fuse_rope)
        CB 7: intermediate hold x * RMS before applying weight or rope

    if (fuse_rope)
        CB 8: transformation mat
        CB 9: rope cos
        CB 10: rope sin
        CB 11: rotated input
    */

    const uint32_t input_cb_id = tt::CBIndex::c_0;
    const uint32_t stats_cb_id = tt::CBIndex::c_1;
    const uint32_t reduce_scalar_cb_id = tt::CBIndex::c_2;
    const uint32_t epsilon_cb_id = tt::CBIndex::c_3;
    const uint32_t reduce_result_cb_id = tt::CBIndex::c_4;
    const uint32_t output_cb_id = tt::CBIndex::c_5;
    const uint32_t weight_cb_id = tt::CBIndex::c_6;
    const uint32_t intermediate_cb_id = tt::CBIndex::c_7;
    const uint32_t transformation_mat_cb_id = tt::CBIndex::c_8;
    const uint32_t rope_cos_cb_id = tt::CBIndex::c_9;
    const uint32_t rope_sin_cb_id = tt::CBIndex::c_10;
    const uint32_t rotated_input_cb_id = tt::CBIndex::c_11;

    constexpr uint32_t double_buffer_constant = 2;
    const uint32_t input_cb_num_tiles = dst_reg_count * double_buffer_constant;
    const uint32_t stats_cb_num_tiles = stats_tiles_cols * double_buffer_constant;
    const uint32_t reduce_scalar_cb_num_tiles = 1;
    const uint32_t epsilon_cb_num_tiles = 1;
    const uint32_t reduce_result_cb_num_tiles = 1;
    const uint32_t output_cb_num_tiles = dst_reg_count * double_buffer_constant;
    // kernels use weight_cb in granularity of dst_reg_count, so we need to size appropriately
    const uint32_t weight_cb_num_tiles = tt::round_up(num_tile_cols, dst_reg_count);
    const uint32_t intermediate_cb_num_tiles = dst_reg_count;
    const uint32_t transformation_mat_cb_num_tiles = 1;
    const uint32_t rope_cos_sin_cb_num_tiles = head_dim_tiles;

    const uint32_t epsilon_packed = std::bit_cast<uint32_t>(eps);

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_id,
        stats_cb_id,
        weight_cb_id,
        reduce_scalar_cb_id,
        epsilon_cb_id,
        transformation_mat_cb_id,
        rope_cos_cb_id,
        rope_sin_cb_id,
        num_tile_cols,
        dst_reg_count,
        stats_tiles_cols,
        W * num_devices,  // reduce_factor
        epsilon_packed,
        static_cast<uint32_t>(has_weight),
        static_cast<uint32_t>(fuse_rope),
        head_dim_tiles};

    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(stats_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(has_weight ? weight_tensor.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(fuse_rope ? transformation_mat.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(fuse_rope ? rope_cos.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(fuse_rope ? rope_sin.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_id,
        num_tile_cols,
        dst_reg_count,
        head_dim_tiles,
        num_heads,
        num_tile_rows,
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    bool use_legacy_rsqrt = false;
    std::vector<uint32_t> compute_args = {
        input_cb_id,
        stats_cb_id,
        weight_cb_id,
        reduce_scalar_cb_id,
        epsilon_cb_id,
        reduce_result_cb_id,
        intermediate_cb_id,
        output_cb_id,
        transformation_mat_cb_id,
        rope_cos_cb_id,
        rope_sin_cb_id,
        rotated_input_cb_id,
        num_tile_cols,
        dst_reg_count,
        stats_tiles_cols,
        static_cast<uint32_t>(use_legacy_rsqrt),
        static_cast<uint32_t>(has_weight),
        static_cast<uint32_t>(fuse_rope),
        head_dim_tiles};

    const auto* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/"
        "rmsnorm_post_allgather.cpp";

    const uint32_t num_tile_rows_per_core = tt::div_up(num_tile_rows, num_cores);

    const auto cores = corerange_to_cores(core_grid, num_cores, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Reader kernel
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/"
        "rms_post_allgather_reader.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = core_grid_set;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/"
        "rms_post_allgather_writer.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = core_grid_set;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    // Compute kernel
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = compute_kernel_file;
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = core_grid_set;
    compute_kernel_desc.compile_time_args = std::move(compute_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};

    // Build runtime args per core
    reader_kernel_desc.runtime_args.reserve(num_cores);
    writer_kernel_desc.runtime_args.reserve(num_cores);
    compute_kernel_desc.runtime_args.reserve(num_cores);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);

        const uint32_t tile_row_start = std::min(core_id * num_tile_rows_per_core, num_tile_rows);
        const uint32_t tile_row_end = std::min(tile_row_start + num_tile_rows_per_core, num_tile_rows);
        const uint32_t num_tile_rows_to_process = tile_row_end - tile_row_start;

        reader_kernel_desc.emplace_runtime_args(
            core,
            {input_tensor.buffer(),
             stats_tensor.buffer(),
             weight_buffer,
             transformation_mat_buffer,
             rope_cos_buffer,
             rope_sin_buffer,
             tile_row_start,
             tile_row_end});

        compute_kernel_desc.emplace_runtime_args(core, {num_tile_rows_to_process});

        writer_kernel_desc.emplace_runtime_args(core, {output_tensor.buffer(), tile_row_start, tile_row_end});
    }

    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));
    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));
    program_descriptor.kernels.push_back(std::move(compute_kernel_desc));

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = input_cb_num_tiles * input_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_id),
            .data_format = input_data_format,
            .page_size = input_tile_size}}}});

    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = stats_cb_num_tiles * stats_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(stats_cb_id),
            .data_format = stats_data_format,
            .page_size = stats_tile_size}}}});

    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = reduce_scalar_cb_num_tiles * reduce_scalar_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reduce_scalar_cb_id),
            .data_format = reduce_scalar_data_format,
            .page_size = reduce_scalar_tile_size}}}});

    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = epsilon_cb_num_tiles * reduce_scalar_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(epsilon_cb_id),
            .data_format = reduce_scalar_data_format,
            .page_size = reduce_scalar_tile_size}}}});

    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = reduce_result_cb_num_tiles * intermediate_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reduce_result_cb_id),
            .data_format = intermediate_data_format,
            .page_size = intermediate_tile_size}}}});

    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = output_cb_num_tiles * output_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_id),
            .data_format = output_data_format,
            .page_size = output_tile_size}}}});

    if (has_weight) {
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = weight_cb_num_tiles * weight_tile_size,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(weight_cb_id),
                .data_format = weight_data_format,
                .page_size = weight_tile_size}}}});
    }

    if (has_weight || fuse_rope) {
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = intermediate_cb_num_tiles * intermediate_tile_size,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(intermediate_cb_id),
                .data_format = intermediate_data_format,
                .page_size = intermediate_tile_size}}}});
    }

    if (fuse_rope) {
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = transformation_mat_cb_num_tiles * transformation_mat_tile_size,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(transformation_mat_cb_id),
                .data_format = transformation_mat_data_format,
                .page_size = transformation_mat_tile_size}}}});

        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = rope_cos_sin_cb_num_tiles * rope_cos_tile_size,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rope_cos_cb_id),
                .data_format = rope_cos_data_format,
                .page_size = rope_cos_tile_size}}}});

        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = rope_cos_sin_cb_num_tiles * rope_sin_tile_size,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rope_sin_cb_id),
                .data_format = rope_sin_data_format,
                .page_size = rope_sin_tile_size}}}});

        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = intermediate_cb_num_tiles * input_tile_size,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(rotated_input_cb_id),
                .data_format = input_data_format,
                .page_size = input_tile_size}}}});
    }

    return program_descriptor;
}

}  // namespace ttnn::experimental::prim
