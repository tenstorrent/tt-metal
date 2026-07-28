// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

// Single-tile (Wt == 1) path. The Wt >= 2 path implements HF rotate_half via
// inter-tile half-swap + scalar negation, which collapses when Wt == 1 (half_Wt
// == 0). Here we instead use matmul_tiles(input, trans_mat) with an in-L1
// transformation matrix that encodes [[0, I], [-I, 0]].
ProgramDescriptor create_single_tile_descriptor(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    ProgramDescriptor desc;

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    auto& output = tensor_return_value;
    const auto& token_idx = operation_attributes.token_idx;

    const auto input_tile = input.tensor_spec().tile();
    const auto input_tile_height = input_tile.get_height();
    const auto input_tile_hw = input_tile.get_tile_hw();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = input_tile.get_tile_size(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = input_tile.get_tile_size(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = input_tile.get_tile_size(sin_cb_data_format);

    // trans_mat is constructed in L1 by the reader and is always bf16.
    tt::DataFormat trans_mat_cb_data_format =
        (input_cb_data_format == tt::DataFormat::Bfp8_b) ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    uint32_t trans_mat_single_tile_size = input_tile.get_tile_size(trans_mat_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = input_tile.get_tile_size(output_cb_data_format);

    constexpr uint32_t Wt = 1;
    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.padded_shape()[-1] * sizeof(bfloat16);

    tt::tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_rows_per_core_group_1, num_rows_per_core_group_2;
    CoreRangeSet all_cores, core_group_1, core_group_2;

    bool in_sharded = input.shard_spec().has_value();
    bool out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    uint32_t num_input_tiles, num_output_tiles;

    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_rows_per_core_group_1 = shard_spec.value().shape[0] / input_tile_height;
        num_rows_per_core_group_2 = 0;
        num_input_tiles =
            in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / input_tile_hw : 2 * Wt;
        num_output_tiles =
            out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / input_tile_hw : 2 * Wt;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, row_major);
        num_input_tiles = 2 * Wt;
        num_output_tiles = num_input_tiles;
    }

    constexpr uint8_t input_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
        .buffer = in_sharded ? input.buffer() : nullptr,
    });

    // trans_mat CB at the slot the Wt>=2 path uses for "rotated input".
    constexpr uint8_t trans_mat_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = trans_mat_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = trans_mat_cb_index,
            .data_format = trans_mat_cb_data_format,
            .page_size = trans_mat_single_tile_size,
            .tile = input_tile,
        }}},
    });

    uint32_t num_cos_sin_tiles = token_idx.has_value() ? Wt : 2 * Wt;
    constexpr uint8_t cos_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * cos_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
            .tile = input_tile,
        }}},
    });

    constexpr uint8_t sin_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * sin_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
            .tile = input_tile,
        }}},
    });

    uint32_t num_interm_tiles = 1;
    constexpr uint8_t rotated_input_interm_cb_index = tt::CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_input_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
    });

    // Keep sin/cos intermediates at input format regardless of sincos format.
    // The packer format stays stable across matmul / mul / add, avoiding
    // fragile pack_reconfig sequences after mm_init for mixed precision.
    constexpr uint8_t cos_interm_cb_index = tt::CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
    });

    constexpr uint8_t sin_interm_cb_index = tt::CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
            .tile = input_tile,
        }}},
        .buffer = out_sharded ? output.buffer() : nullptr,
    });

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);
    constexpr uint8_t untilized_cos_interm_cb_index = tt::CBIndex::c_27;
    constexpr uint8_t untilized_cos_sync_cb_index = tt::CBIndex::c_5;
    constexpr uint8_t untilized_sin_interm_cb_index = tt::CBIndex::c_28;
    constexpr uint8_t untilized_sin_sync_cb_index = tt::CBIndex::c_6;
    constexpr uint8_t retilized_cos_cb_index = tt::CBIndex::c_29;
    constexpr uint8_t retilized_sin_cb_index = tt::CBIndex::c_30;
    KernelDescriptor::Defines reader_kernel_defines, writer_kernel_defines, compute_kernel_defines;
    if (token_idx.has_value()) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * cos_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = retilized_cos_cb_index,
                .data_format = cos_cb_data_format,
                .page_size = cos_single_tile_size,
                .tile = input_tile,
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * sin_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = retilized_sin_cb_index,
                .data_format = sin_cb_data_format,
                .page_size = sin_single_tile_size,
                .tile = input_tile,
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * scalar_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{
                CBFormatDescriptor{
                    .buffer_index = untilized_cos_interm_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_cos_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
            }},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * scalar_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{
                CBFormatDescriptor{
                    .buffer_index = untilized_sin_interm_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_sin_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
            }},
        });
        reader_kernel_defines.emplace_back("DECODE_MODE", "1");
        writer_kernel_defines.emplace_back("DECODE_MODE", "1");
        compute_kernel_defines.emplace_back("DECODE_MODE", "1");
    }

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    if (in_sharded) {
        reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)cos_cb_index,
            (std::uint32_t)sin_cb_index,
            (std::uint32_t)trans_mat_cb_index,
            (std::uint32_t)Ht,
            (std::uint32_t)HtWt,
        };
        tt::tt_metal::TensorAccessorArgs(*cos_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*sin_buffer).append_to(reader_compile_time_args);
    } else {
        reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)cos_cb_index,
            (std::uint32_t)sin_cb_index,
            (std::uint32_t)trans_mat_cb_index,
            (std::uint32_t)Ht,
            (std::uint32_t)HtWt,
        };
        tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*cos_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*sin_buffer).append_to(reader_compile_time_args);
    }
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    if (token_idx.has_value()) {
        writer_compile_time_args.insert(
            writer_compile_time_args.end(),
            {untilized_cos_interm_cb_index,
             untilized_cos_sync_cb_index,
             untilized_sin_interm_cb_index,
             untilized_sin_sync_cb_index});
    }
    if (out_sharded) {
        writer_kernel_defines.emplace_back("OUT_SHARDED", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        in_sharded ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                     "reader_rotary_embedding_single_tile_interleaved_start_id_sharded.cpp"
                   : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                     "reader_rotary_embedding_single_tile_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_kernel_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
        "writer_rotary_embedding_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_kernel_defines);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)trans_mat_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)num_rows_per_core_group_1};
    if (token_idx.has_value()) {
        compute_kernel_args_group_1.insert(
            compute_kernel_args_group_1.end(),
            {(std::uint32_t)untilized_cos_interm_cb_index,
             (std::uint32_t)untilized_cos_sync_cb_index,
             (std::uint32_t)untilized_sin_interm_cb_index,
             (std::uint32_t)untilized_sin_sync_cb_index,
             (std::uint32_t)retilized_cos_cb_index,
             (std::uint32_t)retilized_sin_cb_index});
    }

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
        "rotary_embedding_single_tile.cpp";
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_g1.defines = compute_kernel_defines;
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = compute_kernel_args_group_1;
        compute_kernel_args_group_2[8] = num_rows_per_core_group_2;
        KernelDescriptor g2;
        g2.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
            "rotary_embedding_single_tile.cpp";
        g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        g2.core_ranges = core_group_2;
        g2.compile_time_args = std::move(compute_kernel_args_group_2);
        g2.defines = compute_kernel_defines;
        g2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(g2);
    }

    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (token_idx.has_value()) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        if (!token_idx.has_value()) {
            cos_sin_start_id = num_tiles_written % HtWt;
        }
        if (in_sharded) {
            reader_desc.emplace_runtime_args(
                core, {cos_buffer, sin_buffer, num_rows_per_core, num_tiles_written / Wt % Ht, cos_sin_start_id});
        } else {
            reader_desc.emplace_runtime_args(
                core,
                {src_buffer,
                 cos_buffer,
                 sin_buffer,
                 num_rows_per_core,
                 num_tiles_written,
                 num_tiles_written / Wt % Ht,
                 cos_sin_start_id});
        }

        writer_desc.emplace_runtime_args(
            core, {dst_buffer, num_rows_per_core * Wt, num_tiles_written, cos_sin_offset, Wt, Wbytes});
        num_tiles_written += num_rows_per_core * Wt;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

ProgramDescriptor create_multi_tile_descriptor(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    ProgramDescriptor desc;

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    auto& output = tensor_return_value;
    const auto& token_idx = operation_attributes.token_idx;

    const auto input_tile = input.tensor_spec().tile();
    const auto input_tile_width = input_tile.get_width();
    const auto input_tile_height = input_tile.get_height();
    const auto input_tile_hw = input_tile.get_tile_hw();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = input_tile.get_tile_size(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = input_tile.get_tile_size(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = input_tile.get_tile_size(sin_cb_data_format);

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = input_tile.get_tile_size(scalar_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = input_tile.get_tile_size(output_cb_data_format);

    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / input_tile_height;
    uint32_t Ht = input.padded_shape()[-2] / input_tile_height;
    uint32_t Wt = input.padded_shape()[-1] / input_tile_width;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.padded_shape()[-1] * sizeof(bfloat16);

    tt::tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_rows_per_core_group_1, num_rows_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    bool in_sharded = input.shard_spec().has_value();
    bool out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    uint32_t num_input_tiles, num_output_tiles;

    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_rows_per_core_group_1 = shard_spec.value().shape[0] / input_tile_height;
        num_rows_per_core_group_2 = 0;
        num_input_tiles =
            in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / input_tile_hw : 2 * Wt;
        num_output_tiles =
            out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / input_tile_hw : 2 * Wt;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, row_major);
        num_input_tiles = 2 * Wt;
        num_output_tiles = num_input_tiles;
    }

    constexpr uint8_t input_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
        .buffer = in_sharded ? input.buffer() : nullptr,
    });

    constexpr uint8_t rotated_input_cb_index = tt::CBIndex::c_1;
    uint32_t num_rotated_input_tiles = 2 * Wt;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_rotated_input_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
    });

    uint32_t num_cos_sin_tiles = token_idx.has_value() ? Wt : 2 * Wt;
    constexpr uint8_t cos_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * cos_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
            .tile = input_tile,
        }}},
    });

    constexpr uint8_t sin_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * sin_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
            .tile = input_tile,
        }}},
    });

    // Used for bcast scalar
    constexpr uint8_t src_scalar_cb_index = tt::CBIndex::c_4;
    uint32_t num_scalar_tiles = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_scalar_tiles * scalar_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src_scalar_cb_index,
            .data_format = scalar_cb_data_format,
            .page_size = scalar_single_tile_size,
            .tile = input_tile,
        }}},
    });

    uint32_t num_interm_tiles = 1;
    constexpr uint8_t rotated_input_interm_cb_index = tt::CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_input_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile,
        }}},
    });

    constexpr uint8_t cos_interm_cb_index = tt::CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * cos_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
            .tile = input_tile,
        }}},
    });

    constexpr uint8_t sin_interm_cb_index = tt::CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * sin_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_interm_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
            .tile = input_tile,
        }}},
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;  // output operands start at index 16
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
            .tile = input_tile,
        }}},
        .buffer = out_sharded ? output.buffer() : nullptr,
    });

    constexpr uint8_t untilized_cos_interm_cb_index = tt::CBIndex::c_27;
    constexpr uint8_t untilized_cos_sync_cb_index = tt::CBIndex::c_5;
    constexpr uint8_t untilized_sin_interm_cb_index = tt::CBIndex::c_28;
    constexpr uint8_t untilized_sin_sync_cb_index = tt::CBIndex::c_6;
    constexpr uint8_t retilized_cos_cb_index = tt::CBIndex::c_29;
    constexpr uint8_t retilized_sin_cb_index = tt::CBIndex::c_30;
    KernelDescriptor::Defines reader_kernel_defines, writer_kernel_defines, compute_kernel_defines;
    if (token_idx.has_value()) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * cos_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = retilized_cos_cb_index,
                .data_format = cos_cb_data_format,
                .page_size = cos_single_tile_size,
                .tile = input_tile,
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * sin_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = retilized_sin_cb_index,
                .data_format = sin_cb_data_format,
                .page_size = sin_single_tile_size,
                .tile = input_tile,
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * scalar_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{
                CBFormatDescriptor{
                    .buffer_index = untilized_cos_interm_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_cos_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
            }},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * scalar_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{
                CBFormatDescriptor{
                    .buffer_index = untilized_sin_interm_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_sin_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
                    .tile = input_tile,
                },
            }},
        });
        reader_kernel_defines.emplace_back("DECODE_MODE", "1");
        writer_kernel_defines.emplace_back("DECODE_MODE", "1");
        compute_kernel_defines.emplace_back("DECODE_MODE", "1");
    }

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    if (in_sharded) {
        reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)rotated_input_cb_index,
            (std::uint32_t)cos_cb_index,
            (std::uint32_t)sin_cb_index,
            (std::uint32_t)src_scalar_cb_index,
            (std::uint32_t)bfloat16_scalar,
            (std::uint32_t)Ht,
            (std::uint32_t)Wt,
            (std::uint32_t)HtWt,
            (std::uint32_t)half_Wt * input_single_tile_size,
        };
        tt::tt_metal::TensorAccessorArgs(*cos_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*sin_buffer).append_to(reader_compile_time_args);
    } else {
        reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)rotated_input_cb_index,
            (std::uint32_t)cos_cb_index,
            (std::uint32_t)sin_cb_index,
            (std::uint32_t)src_scalar_cb_index,
            (std::uint32_t)bfloat16_scalar,
            (std::uint32_t)Ht,
            (std::uint32_t)Wt,
            (std::uint32_t)HtWt,
            (std::uint32_t)half_Wt,
        };
        tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*cos_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*sin_buffer).append_to(reader_compile_time_args);
    }
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    if (token_idx.has_value()) {
        writer_compile_time_args.insert(
            writer_compile_time_args.end(),
            {untilized_cos_interm_cb_index,
             untilized_cos_sync_cb_index,
             untilized_sin_interm_cb_index,
             untilized_sin_sync_cb_index});
    }

    if (out_sharded) {
        writer_kernel_defines.emplace_back("OUT_SHARDED", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        in_sharded ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                     "reader_rotary_embedding_interleaved_start_id_sharded.cpp"
                   : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                     "reader_rotary_embedding_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_kernel_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
        "writer_rotary_embedding_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_kernel_defines);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)num_rows_per_core_group_1,
        (std::uint32_t)Wt,
        (std::uint32_t)half_Wt};
    if (token_idx.has_value()) {
        compute_kernel_args_group_1.insert(
            compute_kernel_args_group_1.end(),
            {(std::uint32_t)untilized_cos_interm_cb_index,
             (std::uint32_t)untilized_cos_sync_cb_index,
             (std::uint32_t)untilized_sin_interm_cb_index,
             (std::uint32_t)untilized_sin_sync_cb_index,
             (std::uint32_t)retilized_cos_cb_index,
             (std::uint32_t)retilized_sin_cb_index});
    }

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
        "rotary_embedding.cpp";
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_g1.defines = compute_kernel_defines;
    // NOTE: legacy create() left math_fidelity/fp32_dest_acc_en unset for the g1
    // ComputeConfig in the multi-tile path; preserve those defaults here.
    compute_desc_g1.config = ComputeConfigDescriptor{};

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = compute_kernel_args_group_1;
        compute_kernel_args_group_2[9] = num_rows_per_core_group_2;
        KernelDescriptor g2;
        g2.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
            "rotary_embedding.cpp";
        g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        g2.core_ranges = core_group_2;
        g2.compile_time_args = std::move(compute_kernel_args_group_2);
        g2.defines = compute_kernel_defines;
        g2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(g2);
    }

    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (token_idx.has_value()) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    uint32_t g1_numcores = core_group_1.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = 0;
        if (i < g1_numcores) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else {
            num_rows_per_core = num_rows_per_core_group_2;
        }

        if (!token_idx.has_value()) {
            cos_sin_start_id = num_tiles_written % HtWt;
        }
        if (in_sharded) {
            reader_desc.emplace_runtime_args(
                core, {cos_buffer, sin_buffer, num_rows_per_core, num_tiles_written / Wt % Ht, cos_sin_start_id});
        } else {
            reader_desc.emplace_runtime_args(
                core,
                {src_buffer,
                 cos_buffer,
                 sin_buffer,
                 num_rows_per_core,
                 num_tiles_written,
                 num_tiles_written / Wt % Ht,
                 cos_sin_start_id});
        }

        writer_desc.emplace_runtime_args(
            core, {dst_buffer, num_rows_per_core * Wt, num_tiles_written, cos_sin_offset, Wt, Wbytes});
        num_tiles_written += num_rows_per_core * Wt;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_g1));
    if (compute_desc_g2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_g2));
    }

    return desc;
}

}  // namespace

ProgramDescriptor RotaryEmbeddingProgramFactory::create_descriptor(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    if (tensor_args.input.padded_shape()[-1] / TILE_WIDTH == 1) {
        return create_single_tile_descriptor(operation_attributes, tensor_args, tensor_return_value);
    }
    return create_multi_tile_descriptor(operation_attributes, tensor_args, tensor_return_value);
}

}  // namespace ttnn::experimental::prim
