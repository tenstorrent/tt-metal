// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <bit>
#include <cmath>
#include <limits>
#include <map>

#include <tt-metalium/bfloat16.hpp>

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

tt::tt_metal::ProgramDescriptor ReduceDeviceOperation::ReduceMultiCoreWRmProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& logical_shape = a.logical_shape();
    const auto& padded_shape = a.padded_shape();
    uint32_t W_logical = logical_shape[3];
    uint32_t W_padded = padded_shape[3];
    uint32_t H = logical_shape[2], NC = logical_shape[1] * logical_shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    // Tilize must cover the full row-major page width (padded last dim).
    uint32_t Wt = (W_padded + tile_width - 1) / tile_width;
    const uint32_t wt_tiles_per_chunk = std::min<uint32_t>(8, Wt);

    // Keep NC fixed to 1 for dense RM W path (same convention as tiled multi_core_w factory).
    const uint32_t nc_per_reduce = 1;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);
    uint32_t datum_size = tt::datum_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();
    const uint32_t src_rm_page_size = a.buffer()->page_size();
    uint32_t src_datum_size = tt::datum_size(src0_cb_data_format);
    const uint32_t logical_row_bytes = W_logical * src_datum_size;
    const uint32_t chunk_row_bytes = wt_tiles_per_chunk * tile_width * src_datum_size;
    const uint32_t rm_rows_per_tile = tile_height;
    const uint32_t rm_staging_page_size = rm_rows_per_tile * chunk_row_bytes;
    TT_FATAL(
        logical_row_bytes <= src_rm_page_size,
        "Dense RM reduce: logical row size {} bytes exceeds RM page size {} (W_logical={}, dtype)",
        logical_row_bytes,
        src_rm_page_size,
        W_logical);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_rows = NC * H;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_rows_per_core_group_1, num_rows_per_core_group_2;
    // Row-major tensors use one buffer page per logical row along W; linear page_id matches flattened
    // (N,C,H) order. Match split_work_to_cores / corerange_to_cores row-wise traversal — same as
    // typecast_rm_chunked_program_factory — so each core's row chunk stays aligned with golden output.
    constexpr bool k_split_rows_row_wise = true;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_rows, k_split_rows_row_wise);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, k_split_rows_row_wise);
    }

    ProgramDescriptor desc;

    constexpr uint32_t cb_rm = tt::CBIndex::c_24;
    uint32_t num_rm_pages = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_rm_pages * rm_staging_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_rm),
            .data_format = src0_cb_data_format,
            .page_size = rm_staging_page_size,
        }}},
    });

    // Template CB filled with per-element padding identity (same as tail fill): 0 for sum/mean, -inf for max, etc.
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

    const uint32_t ht_per_core_group_1 = (num_rows_per_core_group_1 + rm_rows_per_tile - 1) / rm_rows_per_tile;
    const uint32_t ht_per_core_group_2 = (num_rows_per_core_group_2 + rm_rows_per_tile - 1) / rm_rows_per_tile;
    const uint32_t max_ht_per_core = std::max(ht_per_core_group_1, ht_per_core_group_2);
    // Chunk packed tile-rows (same cap as W tile chunking); at least 1 slab per iteration.
    const uint32_t ht_tiles_per_chunk = std::min<uint32_t>(8, std::max(1U, max_ht_per_core));
    // Staging for tilize: ht_in_chunk × wt_in_chunk tiles before each ReduceInputBlockShape reduce.
    uint32_t num_input_tiles = std::max(2U, wt_tiles_per_chunk * ht_tiles_per_chunk * nc_per_reduce);
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
        .total_size = scaler_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });

    uint32_t num_output_tiles = std::max(2U, ht_tiles_per_chunk);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_3),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    // Partial per logical row across W chunks (one tile per row in the H chunk).
    constexpr uint32_t cb_acc = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ht_tiles_per_chunk * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_acc),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    tt_metal::Buffer* src_buffer = a.buffer();
    const uint32_t padding_identity_bits =
        dense_rm_padding_identity_bits(src0_cb_data_format, operation_attributes.math_op);
    std::vector<uint32_t> reader_compile_time_args = {
        std::bit_cast<uint32_t>(operation_attributes.scaler),
        W_logical,
        src_datum_size,
        padding_identity_bits,
        Wt,
        wt_tiles_per_chunk,
        rm_rows_per_tile,
        ht_tiles_per_chunk,
    };
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    tt_metal::Buffer* dst_buffer = output.buffer();
    std::vector<uint32_t> writer_compile_time_args = {datum_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, ReduceOpDim::W);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_rm_w.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_reduce_w_rm_scalar.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_args_g1 = {
        ht_per_core_group_1,  // Ht
        Wt,
        nc_per_reduce,
        wt_tiles_per_chunk,
        ht_tiles_per_chunk,
        post_mul_scaler_bits,
    };

    const std::string compute_kernel =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_rm_w.cpp";

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source = compute_kernel;
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_args_g1;
    compute_desc_g1.defines = {reduce_defines.begin(), reduce_defines.end()};
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_g2 = {
            ht_per_core_group_2,  // Ht
            Wt,
            nc_per_reduce,
            wt_tiles_per_chunk,
            ht_tiles_per_chunk,
            post_mul_scaler_bits,
        };
        KernelDescriptor d;
        d.kernel_source = compute_kernel;
        d.source_type = KernelDescriptor::SourceType::FILE_PATH;
        d.core_ranges = core_group_2;
        d.compile_time_args = compute_args_g2;
        d.defines = {reduce_defines.begin(), reduce_defines.end()};
        d.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(d);
    }

    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W_padded={}, tile_width={})", W_padded, tile_width);

    uint32_t out_dim_divider = 1;
    std::vector<CoreCoord> cores = corerange_to_cores(all_cores, std::nullopt, k_split_rows_row_wise);

    for (uint32_t i = 0, num_rows_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        // Use concrete addresses (not Buffer*) so resolve_bindings() leaves kernel buffer_bindings
        // empty. Otherwise the mesh adapter's cache-hit fast path only patches addresses and never
        // re-runs apply_descriptor_runtime_args — per-core integer args would stay stuck from the
        // first cached program (wrong pages / PCC ~0). Slow path rebuilds full RT args every dispatch.
        reader_desc.emplace_runtime_args(
            core,
            {
                a.buffer()->address(),
                num_rows_per_core,
                num_rows_read,
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                output.buffer()->address(),
                num_rows_per_core / out_dim_divider,
                num_rows_read / out_dim_divider,
            });
        num_rows_read += num_rows_per_core;
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
