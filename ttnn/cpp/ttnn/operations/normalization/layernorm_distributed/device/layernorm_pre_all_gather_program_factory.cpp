// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/math.hpp"

#include <map>
#include <string>

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;

namespace ttnn::prim {

// =============================================================================
// LayerNormPreAllGatherProgramFactory - Normal (non-Welford, non-2D) operation
// =============================================================================

tt::tt_metal::ProgramDescriptor LayerNormPreAllGatherProgramFactory::create_descriptor(
    const LayerNormPreAllGatherParams& operation_attributes,
    const LayerNormPreAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const bool fuse_pre_add = b.has_value();
    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "H: {}", H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat scaler_cb_data_format =
        in_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (fuse_pre_add) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b->dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t scaler_tile_size = tt::tile_size(scaler_cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();
    auto b_addr = fuse_pre_add ? b->buffer()->address() : 0;

    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;  // reduce scalar
    const uint32_t res_tiles = Wt * double_buffer_constant;    // residual b
    const uint32_t fused_tiles = Wt;                           // a + b

    const uint32_t intermed0_tiles = Wt * double_buffer_constant;  // x^2
    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    TT_FATAL(
        W <= tile_width * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now).",
        W,
        tile_width,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        in0_tiles,
        block_size);
    TT_FATAL(
        intermed0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        intermed0_tiles,
        block_size);

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_group_1: {}", core_group_1.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    log_debug(tt::LogOp, "core_group_2: {}", core_group_2.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    std::vector<uint32_t> reader_compile_time_args = {
        block_size,
    };
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    if (fuse_pre_add) {
        tt::tt_metal::TensorAccessorArgs(b->buffer()).append_to(reader_compile_time_args);
    }

    std::vector<uint32_t> writer_compile_time_args = {writer_block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;
    reader_defines["FUSE_PRE_ADD"] = fuse_pre_add ? "1" : "0";
    compute_defines["FUSE_PRE_ADD"] = fuse_pre_add ? "1" : "0";

    std::vector<uint32_t> compute_args = {Wt, block_size};

    const auto* compute_kernel_file =
        is_rmsnorm ? "ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/"
                     "rmsnorm_pre_allgather.cpp"
                   : "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
                     "layernorm_pre_allgather.cpp";

    // Build runtime args per core
    KernelDescriptor::RuntimeArgs reader_runtime_args;
    KernelDescriptor::RuntimeArgs writer_runtime_args;
    KernelDescriptor::RuntimeArgs compute_runtime_args;
    reader_runtime_args.reserve(num_cores);
    writer_runtime_args.reserve(num_cores);
    compute_runtime_args.reserve(num_cores);

    uint32_t curr_row = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        std::vector<uint32_t> reader_args = {a_addr, num_tile_rows_per_core, Wt, in_tile_offset};
        if (fuse_pre_add) {
            reader_args.push_back(b_addr);
        }
        reader_runtime_args.emplace_back(core, std::move(reader_args));
        compute_runtime_args.emplace_back(core, std::vector<uint32_t>{num_tile_rows_per_core});
        writer_runtime_args.emplace_back(
            core, std::vector<uint32_t>{dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset});

        curr_row += num_tile_rows_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Reader kernel
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.defines = KernelDescriptor::Defines(reader_defines.begin(), reader_defines.end());
    reader_kernel_desc.runtime_args = std::move(reader_runtime_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));

    // Writer kernel
    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.runtime_args = std::move(writer_runtime_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));

    // Compute kernel
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = compute_kernel_file;
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_args);
    compute_kernel_desc.defines = KernelDescriptor::Defines(compute_defines.begin(), compute_defines.end());
    compute_kernel_desc.runtime_args = std::move(compute_runtime_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    program_descriptor.kernels.push_back(std::move(compute_kernel_desc));

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    // c_in0 -> a
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * in_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in_data_format,
            .page_size = in_single_tile_size}}}});

    // c_in1 -> reduce scalar
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in1_tiles * scaler_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_tile_size}}}});

    if (fuse_pre_add) {
        // c_5 -> residual b. Sized in residual's own data format so a residual with a different
        // dtype than the input is read correctly; add_tiles handles the per-operand format.
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = res_tiles * inb_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                .data_format = inb_data_format,
                .page_size = inb_single_tile_size}}}});
        // c_3 -> fused a + b (compute kernel writes into this and downstream consumes)
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = fused_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = cb_data_format,
                .page_size = single_tile_size}}}});
    }

    // LN and RMS shared intermediates
    // c_intermed0 -> x^2
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = intermed0_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = cb_data_format,
            .page_size = single_tile_size}}}});

    // Output
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = out0_tiles * out_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_14),
            .data_format = out_data_format,
            .page_size = out_single_tile_size}}}});

    return program_descriptor;
}

// =============================================================================
// LayerNormPreAllGather2DProgramFactory - 2D core grid operation
// =============================================================================

tt::tt_metal::ProgramDescriptor LayerNormPreAllGather2DProgramFactory::create_descriptor(
    const LayerNormPreAllGatherParams& operation_attributes,
    const LayerNormPreAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const bool fuse_pre_add = b.has_value();
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;

    uint32_t num_tile_rows = NC * Ht;

    IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    uint32_t block_size = 1;
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat scaler_cb_data_format =
        in_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (fuse_pre_add) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b->dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t scaler_tile_size = tt::tile_size(scaler_cb_data_format);

    auto a_addr = a.buffer()->address();
    auto dst_addr = output.buffer()->address();
    auto b_addr = fuse_pre_add ? b->buffer()->address() : 0;

    const uint32_t double_buffer_constant = 2;
    const uint32_t in0_tiles = Wt * double_buffer_constant;
    const uint32_t in1_tiles = 1;  // reduce scalar
    const uint32_t res_tiles = Wt * double_buffer_constant;    // residual b
    const uint32_t fused_tiles = Wt;                           // a + b

    const uint32_t intermed0_tiles = Wt * double_buffer_constant;  // x^2
    uint32_t out0_tiles = 1;

    TT_FATAL(
        W <= tile_width * in0_tiles,
        "W ({}) exceeds the maximum supported size of tile buffer ({} * {}, kernel limitation right now).",
        W,
        tile_width,
        in0_tiles);
    TT_FATAL(
        in0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        in0_tiles,
        block_size);
    TT_FATAL(
        intermed0_tiles % block_size == 0,
        "Size of buffer ({}) must be divisible by the size of block ({}) used by the reader and compute kernel.",
        intermed0_tiles,
        block_size);

    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t max_cores_y = grid_size.y;
    uint32_t cores_x = std::min(max_cores_y, num_tile_rows);
    while (num_tile_rows % cores_x != 0 && cores_x > 1) {
        cores_x--;
    }
    uint32_t tiles_per_core_x = num_tile_rows / cores_x;
    uint32_t cores_y = std::min(max_cores_y, Wt);
    while (Wt % cores_y != 0 && cores_y > 1) {
        cores_y--;
    }
    uint32_t tiles_per_core_y = Wt / cores_y;

    CoreRange all_cores_range({0, 0}, {cores_x - 1, cores_y - 1});
    CoreRangeSet all_cores = CoreRangeSet(std::vector{all_cores_range});

    std::vector<CoreRange> merge_core_ranges_vec;
    for (uint32_t x = 0; x < cores_x; ++x) {
        CoreCoord merge_core = {x, 0};
        merge_core_ranges_vec.emplace_back(CoreRange(merge_core, merge_core));
    }
    CoreRangeSet merge_cores(merge_core_ranges_vec);

    // Translate CreateSemaphore(...) to SemaphoreDescriptor with id 0.
    constexpr uint32_t reducer_semaphore_id = 0;

    std::vector<uint32_t> reader_compile_time_args = {
        block_size,
        reducer_semaphore_id,
        cores_y,
    };
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    if (fuse_pre_add) {
        tt::tt_metal::TensorAccessorArgs(b->buffer()).append_to(reader_compile_time_args);
    }

    std::vector<uint32_t> writer_compile_time_args = {writer_block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;
    reader_defines["FUSE_PRE_ADD"] = fuse_pre_add ? "1" : "0";
    compute_defines["FUSE_PRE_ADD"] = fuse_pre_add ? "1" : "0";

    std::vector<uint32_t> compute_args = {tiles_per_core_x, tiles_per_core_y, block_size, cores_y};

    // Build runtime args per core
    KernelDescriptor::RuntimeArgs reader_runtime_args;
    KernelDescriptor::RuntimeArgs writer_runtime_args;
    KernelDescriptor::RuntimeArgs compute_runtime_args;

    for (uint32_t x = 0; x < cores_x; ++x) {
        for (uint32_t y = 0; y < cores_y; ++y) {
            CoreCoord core = {x, y};
            bool is_merge_core = y == 0;
            const auto merge_core = device->worker_core_from_logical_core({x, 0});

            uint32_t num_tile_rows_per_core = tiles_per_core_x;

            uint32_t in_tile_offset = (x * Wt) + (y * tiles_per_core_y);
            uint32_t out_tile_offset = x * out0_tiles;

            std::vector<uint32_t> reader_args = {
                a_addr,
                tiles_per_core_x,
                tiles_per_core_y,
                in_tile_offset,
                static_cast<uint32_t>(is_merge_core),
                static_cast<uint32_t>(merge_core.x),
                static_cast<uint32_t>(merge_core.y),
                y};
            if (fuse_pre_add) {
                reader_args.push_back(b_addr);
            }
            reader_runtime_args.emplace_back(core, std::move(reader_args));
            compute_runtime_args.emplace_back(core, std::vector<uint32_t>{static_cast<uint32_t>(is_merge_core)});
            if (is_merge_core) {
                writer_runtime_args.emplace_back(
                    core, std::vector<uint32_t>{dst_addr, num_tile_rows_per_core * out0_tiles, out_tile_offset});
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Semaphore: reducer_semaphore_id (id 0)
    program_descriptor.semaphores.push_back(SemaphoreDescriptor{
        .id = reducer_semaphore_id, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});

    // Reader kernel
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_layernorm_preallgather_2d.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.defines = KernelDescriptor::Defines(reader_defines.begin(), reader_defines.end());
    reader_kernel_desc.runtime_args = std::move(reader_runtime_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));

    // Writer kernel (only on merge cores)
    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = merge_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.runtime_args = std::move(writer_runtime_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));

    // Compute kernel
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_2d.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_args);
    compute_kernel_desc.defines = KernelDescriptor::Defines(compute_defines.begin(), compute_defines.end());
    compute_kernel_desc.runtime_args = std::move(compute_runtime_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    program_descriptor.kernels.push_back(std::move(compute_kernel_desc));

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    // c_in0 -> a
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * in_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in_data_format,
            .page_size = in_single_tile_size}}}});

    // c_in1 -> reduce scalar
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in1_tiles * scaler_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_tile_size}}}});

    if (fuse_pre_add) {
        // c_5 -> residual b. Sized in residual's own data format so a residual with a different
        // dtype than the input is read correctly; add_tiles handles the per-operand format.
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = res_tiles * inb_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                .data_format = inb_data_format,
                .page_size = inb_single_tile_size}}}});
        // c_3 -> fused a + b (compute kernel writes into this and downstream consumes)
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = fused_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = cb_data_format,
                .page_size = single_tile_size}}}});
    }

    // LN and RMS shared intermediates
    // c_intermed0 -> x^2
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = intermed0_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = cb_data_format,
            .page_size = single_tile_size}}}});

    // c_intermed1 (CB 15)
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = tiles_per_core_y * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_15),
            .data_format = cb_data_format,
            .page_size = single_tile_size}}}});

    // c_out (CB 16) - per-core partial output
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = out0_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = cb_data_format,
            .page_size = single_tile_size}}}});

    // c_zero (CB 13)
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = out0_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_13),
            .data_format = cb_data_format,
            .page_size = single_tile_size}}}});

    // c_out_final (CB 14) - only on merge cores
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = out0_tiles * out_single_tile_size,
        .core_ranges = merge_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_14),
            .data_format = out_data_format,
            .page_size = out_single_tile_size}}}});

    return program_descriptor;
}

}  // namespace ttnn::prim
