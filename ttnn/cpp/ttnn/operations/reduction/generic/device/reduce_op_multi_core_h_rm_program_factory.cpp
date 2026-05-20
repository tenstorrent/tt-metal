// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>
#include <bit>
#include <limits>
#include <map>

namespace {

float get_pad_value(tt::tt_metal::ReduceOpMath reduce_math) {
    using tt::tt_metal::ReduceOpMath;
    return reduce_math == ReduceOpMath::MAX   ? -std::numeric_limits<float>::infinity()
           : reduce_math == ReduceOpMath::MIN ? std::numeric_limits<float>::infinity()
                                              : 0.0f;
}

uint32_t dense_rm_padding_identity_bits(tt::DataFormat df, tt::tt_metal::ReduceOpMath op) {
    const float v = get_pad_value(op);
    if (df == tt::DataFormat::Float32) {
        return std::bit_cast<uint32_t>(v);
    }
    const uint16_t bf16 = std::bit_cast<uint16_t>(bfloat16::truncate(v));
    return static_cast<uint32_t>(bf16);
}

}  // namespace

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ReduceDeviceOperation::ReduceMultiCoreHRmProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& logical_shape = a.logical_shape();
    const auto& padded_shape = a.padded_shape();
    const uint32_t W_logical = logical_shape[3];
    const uint32_t W_padded = padded_shape[3];
    const uint32_t H_logical = logical_shape[2];
    const uint32_t NC = logical_shape[1] * logical_shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t Wt = (W_padded + tile_width - 1) / tile_width;

    // Detect WIDTH_SHARDED early; it overrides W_logical and Wt used throughout.
    const bool use_width_sharding =
        a.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED &&
        output.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;

    const uint32_t shard_W = use_width_sharding ? a.shard_spec().value().shape[1] : W_logical;
    // For the sharded path, the effective logical width and tile count are shard-relative.
    // effective_Wt uses ceil so a partial last tile (shard_W not a multiple of tile_width) is not dropped.
    const uint32_t effective_W_logical = use_width_sharding ? shard_W : W_logical;
    const uint32_t effective_Wt = use_width_sharding ? ((shard_W + tile_width - 1) / tile_width) : Wt;

    TT_FATAL(
        effective_Wt != 0, "Width in tiles (Wt) must be non-zero (W_padded={}, tile_width={})", W_padded, tile_width);
    TT_FATAL(NC != 0, "Batch size NC must be non-zero");

    constexpr uint32_t k_max_tiles_per_chunk = 8;
    const uint32_t wt_tiles_per_chunk = std::min<uint32_t>(k_max_tiles_per_chunk, effective_Wt);
    constexpr uint32_t nc_per_reduce = 1;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);
    const uint32_t datum_size = tt::datum_size(dst_cb_data_format);
    const uint32_t src_datum_size = tt::datum_size(src0_cb_data_format);

    tt_metal::IDevice* device = a.device();
    // const uint32_t src_rm_page_size = a.buffer()->page_size();
    // const uint32_t logical_row_bytes = effective_W_logical * src_datum_size;
    const uint32_t chunk_row_bytes = wt_tiles_per_chunk * tile_width * src_datum_size;
    const uint32_t rm_rows_per_tile = tile_height;
    const uint32_t rm_staging_page_size = rm_rows_per_tile * chunk_row_bytes;
    // TT_FATAL(
    //     logical_row_bytes <= src_rm_page_size,
    //     "Reduce H (dense RM): logical row size {} bytes exceeds RM page size {} (effective_W_logical={})",
    //     logical_row_bytes,
    //     src_rm_page_size,
    //     effective_W_logical);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_output_tile_cols = NC * effective_Wt;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cols_per_core_group_1, num_cols_per_core_group_2;
    if (use_width_sharding) {
        // Each shard core handles the same number of output tile columns (NC * Wt_shard).
        all_cores = a.shard_spec().value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_cols_per_core_group_1 = num_output_tile_cols;
        num_cols_per_core_group_2 = 0;
        num_cores = all_cores.num_cores();
    } else if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_output_tile_cols);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tile_cols);
    }
    TT_FATAL(num_cores > 0, "Reduce H (dense RM) requires at least one worker core");

    const uint32_t Ht_total = (H_logical + rm_rows_per_tile - 1) / rm_rows_per_tile;
    const uint32_t ht_tiles_per_chunk = std::min<uint32_t>(k_max_tiles_per_chunk, std::max(1U, Ht_total));

    ProgramDescriptor desc;

    constexpr uint32_t cb_rm = tt::CBIndex::c_24;
    constexpr uint32_t num_rm_pages = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_rm_pages * rm_staging_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_rm),
            .data_format = src0_cb_data_format,
            .page_size = rm_staging_page_size,
        }}},
    });

    constexpr uint32_t cb_clear_value = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_clear_value),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    const uint32_t num_input_tiles = std::max(2U, wt_tiles_per_chunk * ht_tiles_per_chunk * nc_per_reduce);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(0),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * scaler_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });

    const uint32_t num_output_tiles = std::max(2U, wt_tiles_per_chunk);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_3),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    constexpr uint32_t cb_acc = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = wt_tiles_per_chunk * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_acc),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    const uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    tt_metal::Buffer* src_buffer = a.buffer();
    const uint32_t padding_identity_bits =
        dense_rm_padding_identity_bits(src0_cb_data_format, operation_attributes.math_op);
    std::vector<uint32_t> reader_compile_time_args = {
        std::bit_cast<uint32_t>(operation_attributes.scaler),
        effective_W_logical,  // W_logical — shard-relative for WIDTH_SHARDED
        src_datum_size,
        padding_identity_bits,
        effective_Wt,  // Wt — shard-relative for WIDTH_SHARDED
        wt_tiles_per_chunk,
        rm_rows_per_tile,
        ht_tiles_per_chunk,
        H_logical,
    };
    if (use_width_sharding) {
        // Slot 9: shard row stride in bytes, 16B-aligned. The reader uses direct local L1
        // addressing (UnicastEndpoint with my_x/my_y). Aligning ensures every row in the shard
        // starts at a 16B-aligned L1 address for NOC DMA; for valid inputs the align() is a no-op
        // since the 16B check in validate_on_program_cache_miss already enforces alignment.
        reader_compile_time_args.push_back(tt::align(shard_W * src_datum_size, hal::get_l1_alignment()));
    } else {
        TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    }

    tt_metal::Buffer* dst_buffer = output.buffer();
    std::vector<uint32_t> writer_compile_time_args = {
        datum_size, effective_Wt, effective_W_logical, wt_tiles_per_chunk};
    if (use_width_sharding) {
        // Slot 4: output shard row stride in bytes, 16B-aligned (same reasoning as reader slot 9).
        writer_compile_time_args.push_back(tt::align(shard_W * datum_size, hal::get_l1_alignment()));
    } else {
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    }

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::H);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    if (use_width_sharding) {
        reduce_defines["SHARDED_WIDTH_INPUT"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_rm.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "writer_reduce_h_rm_scalar.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        Ht_total,  // Ht (total tile rows along H for this op)
        effective_Wt,
        nc_per_reduce,
        wt_tiles_per_chunk,
        ht_tiles_per_chunk,
        post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
    };

    const std::string compute_kernel =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm.cpp";

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = compute_kernel;
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_g1.defines = {reduce_defines.begin(), reduce_defines.end()};
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        KernelDescriptor d;
        d.kernel_source = compute_kernel;
        d.source_type = KernelDescriptor::SourceType::FILE_PATH;
        d.core_ranges = core_group_2;
        d.compile_time_args = compute_kernel_args_group_1;
        d.defines = {reduce_defines.begin(), reduce_defines.end()};
        d.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(d);
    }

    constexpr bool k_split_row_wise = true;
    std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, k_split_row_wise);
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);

    for (uint32_t i = 0, output_tiles_seen = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_output_tiles_local = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_local = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_local = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // For WIDTH_SHARDED each core owns a private L1 shard that starts at page 0; the start
        // offset is always 0. For interleaved, cores partition the global output range so the
        // start advances by how many tile-columns each previous core handled.
        const uint32_t start_offset = use_width_sharding ? 0u : output_tiles_seen;

        // Concrete addresses (not Buffer*) keep buffer_bindings empty so mesh program-cache fast
        // paths still re-apply per-core integer runtime args (see apply_descriptor_runtime_args).
        reader_desc.emplace_runtime_args(
            core,
            {
                src_buffer->address(),
                num_output_tiles_local,
                start_offset,
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                dst_buffer->address(),
                num_output_tiles_local,
                start_offset,
            });

        if (core_group_1.contains(core)) {
            compute_desc_g1.emplace_runtime_args(core, {num_output_tiles_local, start_offset});
        } else if (compute_desc_g2.has_value()) {
            compute_desc_g2->emplace_runtime_args(core, {num_output_tiles_local, start_offset});
        } else {
            TT_THROW("Reduce H (dense RM): core in core_group_2 but no second compute descriptor");
        }

        output_tiles_seen += num_output_tiles_local;

        if (!use_width_sharding && i == num_cores - 1) {
            TT_FATAL(
                output_tiles_seen == num_output_tile_cols,
                "Reduce H (dense RM) assigned {} output tile columns across cores, expected {}",
                output_tiles_seen,
                num_output_tile_cols);
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

}  // namespace ttnn::prim
