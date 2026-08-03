// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_sharded_retile_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TilizeMultiCoreShardedRetileProgramFactory::create_descriptor(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    const Tile& input_tile = a.tensor_spec().tile();
    const Tile& output_tile = operation_attributes.tile;

    const uint32_t in_tile_width = input_tile.get_width();
    const uint32_t in_tile_height = input_tile.get_height();
    const uint32_t out_tile_width = output_tile.get_width();
    const uint32_t out_tile_height = output_tile.get_height();

    TT_FATAL(
        in_tile_width == TILE_WIDTH && out_tile_width == TILE_WIDTH,
        "Retile requires tile width {}, got input {} and output {}",
        TILE_WIDTH,
        in_tile_width,
        out_tile_width);
    const bool shrink = in_tile_height >= out_tile_height;
    TT_FATAL(
        shrink ? (in_tile_height % out_tile_height) == 0 : (out_tile_height % in_tile_height) == 0,
        "Retile requires one tile height to divide the other exactly; got {} -> {}",
        in_tile_height,
        out_tile_height);

    TT_FATAL(a.is_sharded(), "Sharded retile program factory requires a sharded input");

    const auto& shard_spec = a.shard_spec().value();
    const uint32_t shard_height = shard_spec.shape[0];
    const uint32_t shard_width = shard_spec.shape[1];
    const CoreRangeSet& all_cores = shard_spec.grid;

    TT_FATAL(
        shard_width % in_tile_width == 0,
        "Sharded retile requires shard width {} divisible by tile width {}",
        shard_width,
        in_tile_width);
    TT_FATAL(
        shard_height % in_tile_height == 0,
        "Sharded retile requires shard height {} divisible by input tile height {}",
        shard_height,
        in_tile_height);
    TT_FATAL(
        shard_height % out_tile_height == 0,
        "Sharded retile requires shard height {} divisible by output tile height {}",
        shard_height,
        out_tile_height);

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t input_single_tile_size = input_tile.get_tile_size(input_cb_data_format);
    const uint32_t output_single_tile_size = output_tile.get_tile_size(output_cb_data_format);
    const uint32_t mid_page_size = input_single_tile_size;
    // The intermediate stays in the input data format (conversion happens on the final pack), so the
    // consumer view sizes an output tile in the input format, not the output format.
    const uint32_t out_tile_size_input_fmt = output_tile.get_tile_size(input_cb_data_format);

    const bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                              output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src0_buffer != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const bool output_is_interleaved = output.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;

    // A retile leaves element dimensions unchanged, so each core's shard maps to whole tile-rows on
    // both sides; only the tiling of those elements changes. Work is per-core and independent.
    const uint32_t tiles_per_block = shard_width / in_tile_width;
    const uint32_t num_input_tile_rows = shard_height / in_tile_height;
    const uint32_t num_output_tile_rows = shard_height / out_tile_height;
    const uint32_t num_tiles_per_shard_in = num_input_tile_rows * tiles_per_block;
    const uint32_t num_tiles_per_shard_out = num_output_tile_rows * tiles_per_block;

    const uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);
    // One output block occupies `ratio` input tile-rows of RM in the grow case, one otherwise.
    const uint32_t mid_pages_per_out_block = (shrink ? 1u : ratio) * tiles_per_block;

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t mid_cb_index = tt::CBIndex::c_1;       // input tile geometry (untilize producer)
    constexpr uint32_t mid_view_cb_index = tt::CBIndex::c_2;  // output tile geometry (tilize consumer), aliases c_1
    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;

    const TileDescriptor input_tile_descriptor(input_tile);
    const TileDescriptor output_tile_descriptor(output_tile);

    ProgramDescriptor desc;

    // Input CB (tiled, input tile shape) — aliased to the input shard buffer for zero-copy read.
    {
        CBDescriptor cb_src0;
        cb_src0.total_size = num_tiles_per_shard_in * input_single_tile_size;
        cb_src0.core_ranges = all_cores;
        cb_src0.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
            .tile = input_tile_descriptor,
        });
        cb_src0.buffer = src0_buffer;
        desc.cbs.push_back(std::move(cb_src0));
    }

    // c_1 and c_2 are two views over one shared intermediate L1 region (avoids an L1 copy between
    // untilize and tilize). They must be separate CBs because face geometry is fixed per-CB at
    // program-creation time: c_1 carries the input tile shape for pack_untilize to write into, c_2
    // the output tile shape so llk_unpack_tilize reads the correct number of RM rows.
    desc.cbs.push_back(CBDescriptor{
        .total_size = 2 * mid_pages_per_out_block * mid_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(mid_cb_index),
                .data_format = input_cb_data_format,
                .page_size = mid_page_size,
                .tile = input_tile_descriptor,
            },
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(mid_view_cb_index),
                .data_format = input_cb_data_format,
                .page_size = out_tile_size_input_fmt,
                .tile = output_tile_descriptor,
            },
        }},
    });

    // Output CB (tiled, output tile shape):
    //   Sharded output  → aliased to the output shard buffer (zero-copy write); full shard size.
    //   Interleaved output → local CB sized to a couple of output tile-rows; writer drains it via
    //     TensorAccessor as the compute kernel produces rows.
    {
        CBDescriptor cb_output;
        const uint32_t out_cb_tiles = output_is_interleaved ? (2u * tiles_per_block) : num_tiles_per_shard_out;
        cb_output.total_size = out_cb_tiles * output_single_tile_size;
        cb_output.core_ranges = all_cores;
        cb_output.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_descriptor,
        });
        if (!output_is_interleaved) {
            cb_output.buffer = dst_buffer;
        }
        desc.cbs.push_back(std::move(cb_output));
    }

    // Reader: sharded unary — reads from the local input shard CB (zero-copy).
    {
        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = {src0_cb_index};
        reader_desc.config = ReaderConfigDescriptor{};
        for (const auto& core : corerange_to_cores(all_cores)) {
            reader_desc.emplace_runtime_args(core, {num_tiles_per_shard_in});
        }
        desc.kernels.push_back(std::move(reader_desc));
    }

    // Writer: sharded (zero-copy) or interleaved (TensorAccessor scatter).
    if (output_is_interleaved) {
        // HEIGHT_SHARDED with ROW_MAJOR orientation: each core's shard maps to a contiguous tile
        // range in the output, so start_id = i * num_tiles_per_shard_out.
        std::vector<uint32_t> writer_ct_args = {output_cb_index};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};

        const auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
        uint32_t tile_start_id = 0;
        for (const auto& core : cores) {
            writer_desc.emplace_runtime_args(core, {dst_buffer, num_tiles_per_shard_out, tile_start_id});
            tile_start_id += num_tiles_per_shard_out;
        }
        desc.kernels.push_back(std::move(writer_desc));
    } else {
        // Zero-copy: output CB is aliased to the output shard buffer; writer just synchronises.
        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = all_cores;
        writer_desc.compile_time_args = {output_cb_index};
        writer_desc.config = WriterConfigDescriptor{};
        for (const auto& core : corerange_to_cores(all_cores)) {
            writer_desc.emplace_runtime_args(core, {num_tiles_per_shard_out});
        }
        desc.kernels.push_back(std::move(writer_desc));
    }

    // Compute: the same retile kernel as the interleaved path (per-core, independent).
    {
        std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        if (fp32_llk_acc) {
            unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[mid_cb_index] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[mid_view_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        }

        KernelDescriptor compute_desc;
        compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/retile.cpp";
        compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc.core_ranges = all_cores;
        compute_desc.compile_time_args = {
            tiles_per_block,
            src0_cb_index,
            mid_cb_index,
            mid_view_cb_index,
            output_cb_index,
            in_tile_height,
            out_tile_height,
            out_tile_size_input_fmt,
            mid_page_size,
        };
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_llk_acc,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
        // All shards are the same size, so every core does identical work. num_input_blocks is in
        // input tile-rows; all rows are real (no grow-case height padding within a shard).
        for (const auto& core : corerange_to_cores(all_cores)) {
            compute_desc.emplace_runtime_args(core, {num_input_tile_rows, num_input_tile_rows});
        }
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
