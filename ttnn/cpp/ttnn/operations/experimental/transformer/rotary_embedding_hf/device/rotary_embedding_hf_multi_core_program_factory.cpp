// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf_multi_core_program_factory.hpp"
#include <bit>
#include <string>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

ProgramDescriptor create_single_tile_prefill_descriptor(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    ProgramDescriptor desc;

    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    tt::DataFormat trans_mat_cb_data_format = input_cb_data_format == tt::DataFormat::Bfp8_b
                                                  ? tt::DataFormat::Bfp8_b
                                              : input_cb_data_format == tt::DataFormat::Float32
                                                  ? tt::DataFormat::Float32
                                                  : tt::DataFormat::Float16_b;
    uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    constexpr uint32_t Wt = 1;
    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

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
        num_rows_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_rows_per_core_group_2 = 0;
        num_input_tiles = in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        num_output_tiles = out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
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
        }}},
        .buffer = in_sharded ? input.buffer() : nullptr,
    });

    constexpr uint8_t trans_mat_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = trans_mat_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = trans_mat_cb_index,
            .data_format = trans_mat_cb_data_format,
            .page_size = trans_mat_single_tile_size,
        }}},
    });

    constexpr uint8_t cos_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cos_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
        }}},
    });

    constexpr uint8_t sin_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = sin_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
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
        }}},
    });

    constexpr uint8_t cos_interm_cb_index = tt::CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
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
        }}},
        .buffer = out_sharded ? output.buffer() : nullptr,
    });

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
    writer_compile_time_args.push_back(1);  // Wt: tiles per row
    writer_compile_time_args.push_back(1);  // page stride between rows

    KernelDescriptor::Defines writer_kernel_defines;
    if (out_sharded) {
        writer_kernel_defines.emplace_back("OUT_SHARDED", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        in_sharded ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/"
                     "dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id_sharded.cpp"
                   : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/"
                     "dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
        "writer_rotary_embedding_hf_interleaved.cpp";
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
        (std::uint32_t)num_rows_per_core_group_1,
    };

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
        "rotary_embedding_single_tile.cpp";
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
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
        g2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(g2);
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t cos_sin_start_id = num_tiles_written % HtWt;

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

        writer_desc.emplace_runtime_args(core, {dst_buffer, num_rows_per_core * Wt, num_tiles_written});
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
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    ProgramDescriptor desc;

    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    // Wt_in = tiles per row in the input/output buffer; cos_Wt = tiles per row of cos/sin (== rotary_dim / TILE_WIDTH
    // in the in-place mode, validated in the device op).
    const uint32_t Wt_in = input.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t cos_Wt = cos.padded_shape()[-1] / TILE_WIDTH;
    // inplace: explicit rotary_dim > 0 -> rotate the first rotary_dim columns of the input buffer in place (shared
    // interleaved reader/writer, the non-rotated tiles are neither read nor written).
    const bool inplace = operation_attributes.rotary_dim > 0;
    // passthrough: rotary_dim == 0 and cos/sin narrower than the input (validated in the device op: prefill,
    // interleaved) -> rotate the first cos-width tiles of every row into a new output and pass the rest through
    // (partial-rotary kernels + pass CB).
    const bool passthrough = !inplace && cos_Wt < Wt_in;
    // Wt = rotated tiles per row.
    uint32_t Wt = (inplace || passthrough) ? cos_Wt : Wt_in;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;  // cos/sin tiles per head-slab (cos/sin last dim == Wt * 32)

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
        num_rows_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_rows_per_core_group_2 = 0;
        num_input_tiles = in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        num_output_tiles = out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
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
        }}},
    });

    uint32_t num_cos_sin_tiles = 2 * Wt;
    constexpr uint8_t cos_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * cos_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
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
        }}},
    });

    constexpr uint8_t src_scalar_cb_index = tt::CBIndex::c_4;
    uint32_t num_scalar_tiles = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_scalar_tiles * scalar_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src_scalar_cb_index,
            .data_format = scalar_cb_data_format,
            .page_size = scalar_single_tile_size,
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
        }}},
        .buffer = out_sharded ? output.buffer() : nullptr,
    });

    constexpr uint8_t pass_cb_index = tt::CBIndex::c_5;
    if (passthrough) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = 2 * (Wt_in - Wt) * input_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = pass_cb_index,
                .data_format = input_cb_data_format,
                .page_size = input_single_tile_size,
            }}},
        });
    }

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
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
    reader_compile_time_args.push_back(Wt_in);  // page stride between rows of the input (== Wt when not partial)

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    writer_compile_time_args.push_back(Wt);     // tiles written per row
    writer_compile_time_args.push_back(Wt_in);  // page stride between rows of the output (== Wt when not partial)

    KernelDescriptor::Defines writer_kernel_defines;
    KernelDescriptor::Defines reader_kernel_defines;
    if (out_sharded) {
        writer_kernel_defines.emplace_back("OUT_SHARDED", "1");
    }
    if (passthrough) {
        for (auto* d : {&reader_kernel_defines, &writer_kernel_defines}) {
            d->emplace_back("ROT_WT_IN", std::to_string(Wt_in));
            d->emplace_back("ROT_WT", std::to_string(Wt));
            d->emplace_back("PASS_CB", std::to_string(static_cast<uint32_t>(pass_cb_index)));
        }
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        passthrough ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
                      "reader_rotary_embedding_hf_partial_interleaved.cpp"
                    : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
                      "reader_rotary_embedding_hf_interleaved.cpp";
    reader_desc.defines = std::move(reader_kernel_defines);
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        passthrough ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
                      "writer_rotary_embedding_hf_partial_interleaved.cpp"
                    : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
                      "writer_rotary_embedding_hf_interleaved.cpp";
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

    KernelDescriptor compute_desc_g1;
    compute_desc_g1.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/"
        "rotary_embedding_hf.cpp";
    compute_desc_g1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_g1.core_ranges = core_group_1;
    compute_desc_g1.compile_time_args = compute_kernel_args_group_1;
    compute_desc_g1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    std::optional<KernelDescriptor> compute_desc_g2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = compute_kernel_args_group_1;
        compute_kernel_args_group_2[9] = num_rows_per_core_group_2;

        KernelDescriptor g2;
        g2.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/"
            "rotary_embedding_hf.cpp";
        g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        g2.core_ranges = core_group_2;
        g2.compile_time_args = std::move(compute_kernel_args_group_2);
        g2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
        };
        compute_desc_g2 = std::move(g2);
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

    for (uint32_t i = 0, rows_done = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = i < g1_numcores ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t cos_sin_start_id = (rows_done * Wt) % HtWt;
        uint32_t start_page = rows_done * Wt_in;  // first input/output page of this core's first row

        reader_desc.emplace_runtime_args(
            core,
            {src_buffer, cos_buffer, sin_buffer, num_rows_per_core, start_page, rows_done % Ht, cos_sin_start_id});

        // The partial pass-through writer counts rows; the shared writer counts tiles.
        writer_desc.emplace_runtime_args(
            core, {dst_buffer, passthrough ? num_rows_per_core : num_rows_per_core * Wt, start_page});
        rows_done += num_rows_per_core;
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

ProgramDescriptor RotaryEmbeddingHfMultiCore::create_descriptor(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    if (input.padded_shape()[-1] / TILE_WIDTH == 1) {
        return create_single_tile_prefill_descriptor(operation_attributes, tensor_args, output);
    }
    return create_multi_tile_descriptor(operation_attributes, tensor_args, output);
}

}  // namespace ttnn::experimental::prim
