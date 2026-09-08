// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_sharded_retile_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize/device/tilize_device_operation.hpp"

#include <tt-metalium/experimental/program_descriptor_patching.hpp>

#include <numeric>

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
    const uint32_t input_shard_height = shard_spec.shape[0];
    const uint32_t shard_width = shard_spec.shape[1];
    const CoreRangeSet& all_cores = shard_spec.grid;

    // Output shard geometry. For an interleaved output we don't have an output shard spec; each
    // core's output range in the interleaved buffer mirrors its input shard's row count under the
    // *output* tile shape (input and output cover the same logical rows), so derive it from the
    // input shard's real row count expressed in output tiles. For a sharded output we take the
    // dimensions directly from the output shard spec — critically, the output shard height can
    // differ from the input shard height (e.g. width-sharded logical H=1 padded to a tile: the
    // input shard is 32 rows tall, the output shard 1 row tall). Using the input shard height for
    // the output CB (as the old code did) allocated a CB many times larger than the output shard's
    // L1 bank, tripping `total_size <= max_size_` in circular_buffer_config.
    const bool output_is_interleaved = output.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
    uint32_t output_shard_height = 0;
    if (output_is_interleaved) {
        // Interleaved output only supports the shrink/height-sharded ROW_MAJOR path below; each
        // core's writer emits `num_real_output_tile_rows * tiles_per_block` tiles into a
        // contiguous interleaved range. We treat the output shard height as covering the same
        // rows the input shard does (rounded up to the output tile height).
        output_shard_height = ((input_shard_height + out_tile_height - 1) / out_tile_height) * out_tile_height;
    } else {
        const auto& out_shard_spec = output.shard_spec().value();
        output_shard_height = out_shard_spec.shape[0];
        TT_FATAL(
            out_shard_spec.shape[1] == shard_width,
            "Sharded retile requires input and output shard widths to match ({} vs {})",
            shard_width,
            out_shard_spec.shape[1]);
        TT_FATAL(out_shard_spec.grid == all_cores, "Sharded retile requires input and output shard grids to match");
    }

    TT_FATAL(
        shard_width % in_tile_width == 0,
        "Sharded retile requires shard width {} divisible by tile width {}",
        shard_width,
        in_tile_width);
    TT_FATAL(
        input_shard_height % in_tile_height == 0,
        "Sharded retile requires input shard height {} divisible by input tile height {}",
        input_shard_height,
        in_tile_height);
    TT_FATAL(
        output_shard_height % out_tile_height == 0,
        "Sharded retile requires output shard height {} divisible by output tile height {}",
        output_shard_height,
        out_tile_height);

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    // The intermediate is row-major, so it cannot be a block-float format. If the input is
    // already block-float, unpack it to bfloat16; otherwise keep the input dtype. Conversion
    // to the output dtype happens on the final pack (see retile.cpp).
    auto intermediate_dtype = is_block_float(a.dtype()) ? tt::tt_metal::DataType::BFLOAT16 : a.dtype();
    tt::DataFormat mid_cb_data_format = datatype_to_dataformat_converter(intermediate_dtype);
    const uint32_t input_single_tile_size = input_tile.get_tile_size(input_cb_data_format);
    const uint32_t output_single_tile_size = output_tile.get_tile_size(output_cb_data_format);
    const uint32_t mid_input_page_size = input_tile.get_tile_size(mid_cb_data_format);
    const uint32_t mid_output_page_size = output_tile.get_tile_size(mid_cb_data_format);

    const bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                              output.dtype() == DataType::FLOAT32 || output.dtype() == DataType::FP8_E4M3 ||
                              output.dtype() == DataType::BFLOAT8_B;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(src0_buffer != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // A retile leaves the element dimensions unchanged. Each core's input shard and output shard
    // therefore cover the same *logical* rows of the tensor, but the padded row count on each side
    // is rounded up to its own tile height — for width-sharded tensors those two padded heights
    // can differ (e.g. logical H=1: input shard 32 rows under a (32,32) tile, output shard 1 row
    // under a (1,32) tile). We stage through a shared L1 mid buffer whose row extent covers the
    // *taller* of the two shards; the shorter side either only fills its real rows (grow: input
    // real rows < mid rows, remainder is zero-filled) or only drains its real rows (shrink:
    // output real rows < mid rows, surplus tiles are simply not written).
    const uint32_t tiles_per_block = shard_width / in_tile_width;
    const uint32_t mid_height = std::max(input_shard_height, output_shard_height);
    TT_FATAL(
        mid_height % in_tile_height == 0 && mid_height % out_tile_height == 0,
        "Sharded retile: max shard height {} must be divisible by both tile heights ({}, {})",
        mid_height,
        in_tile_height,
        out_tile_height);
    const uint32_t num_input_tile_rows = mid_height / in_tile_height;  // rows in mid (input tile view)
    const uint32_t num_tiles_per_shard_in = (input_shard_height / in_tile_height) * tiles_per_block;
    const uint32_t num_tiles_per_shard_out = (output_shard_height / out_tile_height) * tiles_per_block;

    // Real (non-padded) row counts per core. For a width-sharded tensor every core carries the
    // whole logical H, so the real rows are min(logical_H, shard_height). For height/block-sharded
    // tensors we require the shard grid to divide the tensor evenly and treat every shard row as
    // real (the interleaved retile factory handles ragged trailing shards; if that becomes needed
    // here we can generalise later). The compute kernel uses these to cap the untilize loop
    // (grow case) and the tilize/write loop (shrink case).
    const auto& logical_shape = a.logical_shape();
    const uint32_t logical_h = logical_shape.rank() >= 2 ? logical_shape[-2] : 1;
    const bool is_width_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const uint32_t real_rows_per_core = is_width_sharded ? std::min(logical_h, input_shard_height) : input_shard_height;
    const uint32_t num_real_input_tile_rows = (real_rows_per_core + in_tile_height - 1) / in_tile_height;
    const uint32_t num_real_output_tile_rows = (real_rows_per_core + out_tile_height - 1) / out_tile_height;

    // Width chunking. For very wide shards the mid CB (row-major intermediate) would exceed L1
    // if sized for the full shard width. We cap the per-chunk width at MAX_CHUNK_ELEMS elements
    // (MAX_CHUNK_TILES tiles) and process the shard in `num_width_chunks` passes; the compute
    // kernel manually seeks the aliased src/out CB pointers per (chunk, tile-row). We pick the
    // largest divisor of tiles_per_block that fits under the cap so num_chunks divides evenly and
    // the compute kernel needs only one template instantiation per pass.
    constexpr uint32_t MAX_CHUNK_ELEMS = 256;
    constexpr uint32_t MAX_CHUNK_TILES = MAX_CHUNK_ELEMS / TILE_WIDTH;
    static_assert(MAX_CHUNK_TILES > 0, "MAX_CHUNK_ELEMS must be at least one tile wide");
    auto compute_chunk_tiles = [MAX_CHUNK_TILES](uint32_t total_tiles) {
        const uint32_t cap = std::min<uint32_t>(total_tiles, MAX_CHUNK_TILES);
        for (uint32_t cw = cap; cw > 0; --cw) {
            if (total_tiles % cw == 0) {
                return cw;
            }
        }
        return 1u;
    };
    // Interleaved-output writer reads pages sequentially from the output CB and scatters them by
    // global tile id, so it requires push_backs in natural (tile-row-major) shard order. Width
    // chunking would emit pages in (chunk, row) order instead, breaking that contract — keep the
    // legacy single-pass path for the interleaved case. Sharded output uses zero-copy writes into
    // the aliased L1 buffer, so the kernel can seek the write pointer per (chunk, row) safely.
    const uint32_t chunk_tiles = output_is_interleaved ? tiles_per_block : compute_chunk_tiles(tiles_per_block);
    const uint32_t num_width_chunks = tiles_per_block / chunk_tiles;
    TT_FATAL(
        chunk_tiles * num_width_chunks == tiles_per_block,
        "Sharded retile: chunk_tiles ({}) must evenly divide tiles_per_block ({})",
        chunk_tiles,
        tiles_per_block);

    // Compute untilizes `num_input_tile_rows` (mid-height) rows into mid, then tilizes them.
    // Aliased c_1/c_2 page sizes can differ, so the allocation must be a multiple of both.
    // Mid CB holds a single width-chunk's worth of tiles; the kernel pops between chunks.
    const uint32_t mid_input_pages = num_input_tile_rows * chunk_tiles;
    const uint32_t mid_size_align = std::lcm(mid_input_page_size, mid_output_page_size);
    const uint32_t mid_total_size =
        ((mid_input_pages * mid_input_page_size + mid_size_align - 1) / mid_size_align) * mid_size_align;

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
        .total_size = mid_total_size,
        .core_ranges = all_cores,
        .format_descriptors = {{
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(mid_cb_index),
                .data_format = mid_cb_data_format,
                .page_size = mid_input_page_size,
                .tile = input_tile_descriptor,
            },
            CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(mid_view_cb_index),
                .data_format = mid_cb_data_format,
                .page_size = mid_output_page_size,
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
            chunk_tiles,
            src0_cb_index,
            mid_cb_index,
            mid_view_cb_index,
            output_cb_index,
            in_tile_height,
            out_tile_height,
            mid_output_page_size,
            mid_input_page_size,
            tiles_per_block,
            input_single_tile_size,
            output_single_tile_size,
            num_width_chunks,
        };
        compute_desc.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_llk_acc,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
        // All shards are the same size, so every core does identical work. num_input_blocks is
        // the mid buffer's tile-row count (in input tile geometry) — the untilize loop iterates
        // that many times, zero-filling any rows past the real input (grow case). The two
        // real-row caps let the kernel stop untilizing / tilizing at the logical H (they equal
        // num_input_tile_rows / num_output_tile_rows-of-mid when there is no padding on either
        // side, matching the previous behaviour when input and output shard heights agree).
        for (const auto& core : corerange_to_cores(all_cores)) {
            compute_desc.emplace_runtime_args(
                core, {num_input_tile_rows, num_real_input_tile_rows, num_real_output_tile_rows});
        }
        desc.kernels.push_back(std::move(compute_desc));
    }

    return desc;
}

void TilizeMultiCoreShardedRetileProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const TilizeParams& /*operation_attributes*/,
    const TilizeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Sharded input rides on a buffer-backed CB, so the reader carries no address. The writer only
    // does when the output is interleaved; otherwise it is CB-backed too.
    const bool output_is_interleaved =
        tensor_return_value.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
    if (output_is_interleaved) {
        patch_tilize_kernel_slot0(program, 1, tensor_return_value.buffer()->address());
    }

    // CBs are matched positionally against create_descriptor's push order.
    ProgramDescriptor cb_addr_only;
    cb_addr_only.cbs.push_back(CBDescriptor{.buffer = tensor_args.input_tensor.buffer()});
    cb_addr_only.cbs.push_back(CBDescriptor{});  // retile scratch, not tensor-backed
    cb_addr_only.cbs.push_back(CBDescriptor{.buffer = output_is_interleaved ? nullptr : tensor_return_value.buffer()});
    apply_descriptor_runtime_args(program, cb_addr_only);  // override-rebuild-ok: cb-addr-only
}

}  // namespace ttnn::prim
