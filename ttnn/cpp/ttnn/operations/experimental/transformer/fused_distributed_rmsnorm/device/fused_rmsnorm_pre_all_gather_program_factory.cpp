// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rmsnorm_pre_all_gather_device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor FusedRMSNormPreAllGatherProgramFactory::create_descriptor(
    const FusedRmsnormPreAllGatherParams& operation_attributes,
    const FusedRmsnormPreAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const uint32_t num_heads = operation_attributes.num_heads;

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t W = input_shape[-1];
    const uint32_t folded_H = input_tensor.physical_volume() / W;

    const uint32_t num_tile_cols = W / TILE_WIDTH;
    const uint32_t num_tile_rows = folded_H / TILE_HEIGHT;
    // num_heads == 1: legacy mode. Reduce all num_tile_cols → 1 stat tile per row.
    // num_heads >  1: per-head mode. Reduce head_dim_tiles tiles → 1 stat tile per head;
    //                 emit num_heads stat tiles per row.
    const uint32_t head_dim_tiles = num_tile_cols / num_heads;
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "folded_H: {}", folded_H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "num_tile_cols: {}", num_tile_cols);
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
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t dst_reg_count = get_dest_reg_count(compute_kernel_config);

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat reduce_scalar_data_format =
        (input_tensor.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat intermediate_data_format = tt::DataFormat::Float32;
    uint32_t input_tile_size = tt::tile_size(input_data_format);
    uint32_t output_tile_size = tt::tile_size(output_data_format);
    uint32_t intermediate_tile_size = tt::tile_size(intermediate_data_format);
    uint32_t reduce_scalar_tile_size = tt::tile_size(reduce_scalar_data_format);

    log_debug(tt::LogOp, "input_data_format: {}", input_data_format);
    log_debug(tt::LogOp, "output_data_format: {}", output_data_format);
    log_debug(tt::LogOp, "intermediate_data_format: {}", intermediate_data_format);
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug(tt::LogOp, "dst_reg_count: {}", dst_reg_count);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    /*
    CB 0: input
    CB 1: reduce scalar
    CB 2: partial sum
    CB 3: output
    */
    // Per-head mode emits one stat tile per head; legacy mode emits one per row.
    const uint32_t output_tiles_per_row = num_heads;

    const uint32_t double_buffer_constant = 2;
    const uint32_t input_cb_num_tiles = dst_reg_count * double_buffer_constant;
    const uint32_t reduce_scalar_cb_num_tiles = 1;
    const uint32_t intermediate_cb_num_tiles = 1;
    const uint32_t output_cb_num_tiles = output_tiles_per_row * double_buffer_constant;

    const uint32_t num_tile_rows_per_core = tt::div_up(num_tile_rows, num_cores);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_grid: {}", core_grid);
    log_debug(tt::LogOp, "num_tile_rows_per_core: {}", num_tile_rows_per_core);

    const uint32_t input_cb_id = tt::CBIndex::c_0;
    const uint32_t reduce_scalar_cb_id = tt::CBIndex::c_1;
    const uint32_t intermediate_cb_id = tt::CBIndex::c_2;
    const uint32_t output_cb_id = tt::CBIndex::c_3;

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_id,
        reduce_scalar_cb_id,
        num_tile_cols,
        dst_reg_count,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_id, output_tiles_per_row};
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_args = {
        input_cb_id,
        reduce_scalar_cb_id,
        intermediate_cb_id,
        output_cb_id,
        num_tile_cols,
        dst_reg_count,
        num_heads,
        head_dim_tiles,
    };

    const auto* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/"
        "rmsnorm_pre_allgather.cpp";

    const auto cores = corerange_to_cores(core_grid, num_cores, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Reader kernel
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/"
        "rms_pre_allgather_reader.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = core_grid_set;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/"
        "rms_pre_allgather_writer.cpp";
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

        reader_kernel_desc.emplace_runtime_args(core, {input_tensor.buffer(), tile_row_start, tile_row_end});

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
        .total_size = reduce_scalar_cb_num_tiles * reduce_scalar_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reduce_scalar_cb_id),
            .data_format = reduce_scalar_data_format,
            .page_size = reduce_scalar_tile_size}}}});

    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = intermediate_cb_num_tiles * intermediate_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(intermediate_cb_id),
            .data_format = intermediate_data_format,
            .page_size = intermediate_tile_size}}}});

    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = output_cb_num_tiles * output_tile_size,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_id),
            .data_format = output_data_format,
            .page_size = output_tile_size}}}});

    return program_descriptor;
}

}  // namespace ttnn::experimental::prim
