// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <variant>
#include <vector>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

// Work distribution, shared by create_*_descriptor (cache miss) and override_runtime_arguments
// (cache hit) so the hit path targets exactly the cores the miss path emitted args for.
// `Wt` is the per-row tile count (1 on the single-tile path).
struct RotaryWorkSplit {
    bool row_major = true;
    uint32_t num_cores = 0;
    uint32_t num_cores_x = 0;
    uint32_t num_cores_y = 0;
    uint32_t num_rows_per_core_group_1 = 0;
    uint32_t num_rows_per_core_group_2 = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    bool in_sharded = false;
    bool out_sharded = false;
    uint32_t num_input_tiles = 0;
    uint32_t num_output_tiles = 0;
};

RotaryWorkSplit compute_rotary_work_split(const Tensor& input, const Tensor& output, uint32_t Wt) {
    RotaryWorkSplit w;
    w.in_sharded = input.shard_spec().has_value();
    w.out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = w.in_sharded ? input.shard_spec() : output.shard_spec();

    auto compute_with_storage_grid_size = input.device()->compute_with_storage_grid_size();
    w.num_cores_x = compute_with_storage_grid_size.x;
    w.num_cores_y = compute_with_storage_grid_size.y;

    if (shard_spec.has_value()) {
        w.row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        w.all_cores = shard_spec.value().grid;
        w.num_cores = w.all_cores.num_cores();
        w.core_group_1 = w.all_cores;
        w.core_group_2 = CoreRangeSet();
        w.num_rows_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        w.num_rows_per_core_group_2 = 0;
        w.num_input_tiles = w.in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        w.num_output_tiles =
            w.out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        auto bbox = w.all_cores.bounding_box();
        w.num_cores_x = bbox.end_coord.x + 1;
        w.num_cores_y = bbox.end_coord.y + 1;
    } else {
        uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
        w.row_major = true;
        std::tie(
            w.num_cores,
            w.all_cores,
            w.core_group_1,
            w.core_group_2,
            w.num_rows_per_core_group_1,
            w.num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, w.row_major);
        w.num_input_tiles = 2 * Wt;
        w.num_output_tiles = w.num_input_tiles;
    }
    return w;
}

// One core's reader/writer runtime args. Buffer* slots are buffer base addresses (registered as
// bindings by emplace_runtime_args on a miss, written as the current address on a hit); uint32_t
// slots are plain values.
using RotaryRtArgs = std::vector<std::variant<uint32_t, Buffer*>>;

struct RotaryPerCoreArgs {
    std::vector<CoreCoord> cores;
    std::vector<RotaryRtArgs> reader;
    std::vector<RotaryRtArgs> writer;
};

// SINGLE SOURCE OF TRUTH for the per-core reader/writer runtime args. Both descriptor variants use
// the same arg layout, and both create_*_descriptor (cache miss) and override_runtime_arguments
// (cache hit) go through here -- so the re-applied values cannot drift from the built layout and no
// arg index is hard-coded on the hit path.
RotaryPerCoreArgs build_rotary_runtime_args(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    const Tensor& output,
    const RotaryWorkSplit& work,
    uint32_t Wt) {
    const auto& input = tensor_args.input;
    const auto& token_idx = operation_attributes.token_idx;

    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.padded_shape()[-1] * sizeof(bfloat16);

    // Decode mode derives both cos/sin offsets from token_idx, whose value compute_program_hash
    // excludes; prefill derives cos_sin_start_id from the (hashed) shapes instead.
    uint32_t cos_sin_offset = token_idx.has_value() ? token_idx.value() % TILE_HEIGHT * Wbytes : 0;

    auto* src_buffer = input.buffer();
    auto* cos_buffer = tensor_args.cos.buffer();
    auto* sin_buffer = tensor_args.sin.buffer();
    auto* dst_buffer = output.buffer();

    RotaryPerCoreArgs result;
    result.cores = grid_to_cores(work.num_cores, work.num_cores_x, work.num_cores_y, work.row_major);
    result.reader.reserve(result.cores.size());
    result.writer.reserve(result.cores.size());

    uint32_t g1_numcores = work.core_group_1.num_cores();
    for (uint32_t i = 0, num_tiles_written = 0; i < work.num_cores; ++i) {
        uint32_t num_rows_per_core = i < g1_numcores ? work.num_rows_per_core_group_1 : work.num_rows_per_core_group_2;
        uint32_t cos_sin_start_id =
            token_idx.has_value() ? token_idx.value() / TILE_HEIGHT * Wt : num_tiles_written % HtWt;

        if (work.in_sharded) {
            result.reader.push_back(
                {cos_buffer, sin_buffer, num_rows_per_core, num_tiles_written / Wt % Ht, cos_sin_start_id});
        } else {
            result.reader.push_back(
                {src_buffer,
                 cos_buffer,
                 sin_buffer,
                 num_rows_per_core,
                 num_tiles_written,
                 num_tiles_written / Wt % Ht,
                 cos_sin_start_id});
        }
        result.writer.push_back({dst_buffer, num_rows_per_core * Wt, num_tiles_written, cos_sin_offset, Wt, Wbytes});
        num_tiles_written += num_rows_per_core * Wt;
    }
    return result;
}

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

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    // trans_mat is constructed in L1 by the reader and is always bf16.
    tt::DataFormat trans_mat_cb_data_format =
        (input_cb_data_format == tt::DataFormat::Bfp8_b) ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
    uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    constexpr uint32_t Wt = 1;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    tt::tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const auto work = compute_rotary_work_split(input, output, Wt);
    const bool in_sharded = work.in_sharded;
    const bool out_sharded = work.out_sharded;
    const CoreRangeSet& all_cores = work.all_cores;
    const CoreRangeSet& core_group_1 = work.core_group_1;
    const CoreRangeSet& core_group_2 = work.core_group_2;
    const uint32_t num_rows_per_core_group_1 = work.num_rows_per_core_group_1;
    const uint32_t num_rows_per_core_group_2 = work.num_rows_per_core_group_2;
    const uint32_t num_input_tiles = work.num_input_tiles;
    const uint32_t num_output_tiles = work.num_output_tiles;

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

    // trans_mat CB at the slot the Wt>=2 path uses for "rotated input".
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

    uint32_t num_cos_sin_tiles = token_idx.has_value() ? Wt : 2 * Wt;
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
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * sin_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = retilized_sin_cb_index,
                .data_format = sin_cb_data_format,
                .page_size = sin_single_tile_size,
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
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_cos_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
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
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_sin_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
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

    // Built via the shared builder so create_descriptor (cache miss) and
    // override_runtime_arguments (cache hit) stay byte-identical.
    const auto per_core = build_rotary_runtime_args(operation_attributes, tensor_args, output, work, Wt);
    reader_desc.runtime_args.reserve(per_core.cores.size());
    writer_desc.runtime_args.reserve(per_core.cores.size());
    for (size_t i = 0; i < per_core.cores.size(); ++i) {
        reader_desc.emplace_runtime_args(per_core.cores[i], per_core.reader[i]);
        writer_desc.emplace_runtime_args(per_core.cores[i], per_core.writer[i]);
    }

    // Kernel push order: reader (0), writer (1), compute (2[, 3]) -- override_runtime_arguments
    // patches by these indices.
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

    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;

    tt::tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const auto work = compute_rotary_work_split(input, output, Wt);
    const bool in_sharded = work.in_sharded;
    const bool out_sharded = work.out_sharded;
    const CoreRangeSet& all_cores = work.all_cores;
    const CoreRangeSet& core_group_1 = work.core_group_1;
    const CoreRangeSet& core_group_2 = work.core_group_2;
    const uint32_t num_rows_per_core_group_1 = work.num_rows_per_core_group_1;
    const uint32_t num_rows_per_core_group_2 = work.num_rows_per_core_group_2;
    const uint32_t num_input_tiles = work.num_input_tiles;
    const uint32_t num_output_tiles = work.num_output_tiles;

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

    uint32_t num_cos_sin_tiles = token_idx.has_value() ? Wt : 2 * Wt;
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

    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;  // output operands start at index 16
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
            }}},
        });

        desc.cbs.push_back(CBDescriptor{
            .total_size = Wt * sin_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = retilized_sin_cb_index,
                .data_format = sin_cb_data_format,
                .page_size = sin_single_tile_size,
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
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_cos_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
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
                },
                CBFormatDescriptor{
                    .buffer_index = untilized_sin_sync_cb_index,
                    .data_format = scalar_cb_data_format,
                    .page_size = scalar_single_tile_size,
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

    // Built via the shared builder so create_descriptor (cache miss) and
    // override_runtime_arguments (cache hit) stay byte-identical.
    const auto per_core = build_rotary_runtime_args(operation_attributes, tensor_args, output, work, Wt);
    reader_desc.runtime_args.reserve(per_core.cores.size());
    writer_desc.runtime_args.reserve(per_core.cores.size());
    for (size_t i = 0; i < per_core.cores.size(); ++i) {
        reader_desc.emplace_runtime_args(per_core.cores[i], per_core.reader[i]);
        writer_desc.emplace_runtime_args(per_core.cores[i], per_core.writer[i]);
    }

    // Kernel push order: reader (0), writer (1), compute (2[, 3]) -- override_runtime_arguments
    // patches by these indices.
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

void RotaryEmbeddingProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Patch the cached program in place; never rebuild the descriptor here (this runs on EVERY cache
    // hit, so a rebuild would pay the cache-miss host cost -- work split, CoreRangeSet, arch queries,
    // TensorAccessorArgs, kernel sources -- on every dispatch).
    //
    // Two classes of state must be re-applied.  (1) token_idx's VALUE is excluded from
    // compute_program_hash so successive decode positions reuse one program; cos_sin_offset and
    // cos_sin_start_id derive from it and would otherwise stay frozen at the first token, silently
    // reading the wrong cos/sin rows.  (2) The override supersedes resolve_bindings, so every buffer
    // address is ours to re-apply too.  Everything else (per-core row counts, Wt, Wbytes, Ht, HtWt) is
    // derived from shapes/memory configs the hash includes and is therefore identical on a hit.
    const auto& input = tensor_args.input;
    // Matches create_descriptor's variant selection: Wt == 1 is the single-tile path.
    const uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
    const auto work = compute_rotary_work_split(input, tensor_return_value, Wt);
    const auto per_core = build_rotary_runtime_args(operation_attributes, tensor_args, tensor_return_value, work, Wt);

    // Kernel push order in both descriptor variants: reader (0), writer (1), compute (2[, 3]).  The
    // compute kernels take no runtime args, so there is nothing to patch on them.
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;

    auto apply = [&program](uint32_t kernel_idx, const CoreCoord& core, const RotaryRtArgs& args) {
        auto& data = tt::tt_metal::GetRuntimeArgs(program, kernel_idx, core);
        for (uint32_t arg_idx = 0; arg_idx < static_cast<uint32_t>(args.size()); ++arg_idx) {
            const auto& slot = args[arg_idx];
            data[arg_idx] = std::holds_alternative<Buffer*>(slot)
                                ? static_cast<uint32_t>(std::get<Buffer*>(slot)->address())
                                : std::get<uint32_t>(slot);
        }
    };

    for (size_t i = 0; i < per_core.cores.size(); ++i) {
        apply(kReaderKernelIdx, per_core.cores[i], per_core.reader[i]);
        apply(kWriterKernelIdx, per_core.cores[i], per_core.writer[i]);
    }

    // The only globally-allocated CBs are the sharded input (c_0) and sharded output (c_16); re-point
    // them at the current buffers, whose base addresses move with reallocation.
    for (const auto& cb : program.circular_buffers()) {
        if (!cb->globally_allocated()) {
            continue;
        }
        const auto& indices = cb->buffer_indices();
        if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_0))) {
            UpdateDynamicCircularBufferAddress(program, cb->id(), *input.buffer());
        } else if (indices.contains(static_cast<uint8_t>(tt::CBIndex::c_16))) {
            UpdateDynamicCircularBufferAddress(program, cb->id(), *tensor_return_value.buffer());
        }
    }
}

}  // namespace ttnn::experimental::prim
