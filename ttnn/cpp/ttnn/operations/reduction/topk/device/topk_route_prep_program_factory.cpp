// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Program factory for topk_route_prep (fused untilize + lowest-finite-bf16 clamp).
//
// Work split (split_blocks_for_tilize-style, over blocks instead of single tiles):
// the padded input is a grid of 32x32 tiles, total_tile_rows x width_tiles. Each
// tile-row is cut into blocks of bw_full = min(8, width_tiles) tiles (8 = the
// bf16 half-sync DEST capacity AND the pack_untilize max block width) plus one
// bw_last remainder block; blocks never cross a tile-row. The flat block list
// (tile-row-major, so each core's blocks cover a CONTIGUOUS tile range — the
// stock reader_unary_interleaved_start_id reader works unchanged) is split
// across the full worker grid with a single cliff core, exactly like the
// untilize parallelize-column factory's split.
//
// Per block, compute copy_tile's the tiles into DEST, floors each at the lowest
// finite bf16 (unary_max_tile), and pack_untilize's DEST into one output-CB page;
// the writer scatters that page's logical sticks into the ROW_MAJOR output.

#include "topk_route_prep_device_operation.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include <algorithm>
#include <bit>
#include <string>
#include <vector>

using namespace tt::constants;

namespace ttnn::operations::reduction::topk_route_prep::program {

namespace {

constexpr uint32_t input_cb_index = tt::CBIndex::c_0;  // c_0: hardcoded in the reused reader
constexpr uint32_t output_cb_index = tt::CBIndex::c_16;

// Kernel indices follow create_descriptor's push_back order.
constexpr uint32_t kReaderKernelIdx = 0;
constexpr uint32_t kComputeKernelIdx = 1;
constexpr uint32_t kWriterKernelIdx = 2;

// Per-core runtime arg slots. Must match the emplace_runtime_args order below.
namespace reader_arg {
constexpr uint32_t SRC_ADDR = 0;
constexpr uint32_t NTILES = 1;
constexpr uint32_t START_ID = 2;
}  // namespace reader_arg
namespace compute_arg {
constexpr uint32_t NBLOCKS = 0;
constexpr uint32_t START_BLOCK = 1;
constexpr uint32_t NBLOCKS_PER_ROW = 2;
}  // namespace compute_arg
namespace writer_arg {
constexpr uint32_t DST_ADDR = 0;
constexpr uint32_t NBLOCKS = 1;
constexpr uint32_t START_BLOCK = 2;
constexpr uint32_t NBLOCKS_PER_ROW = 3;
constexpr uint32_t TILE_ROWS_PER_BATCH = 4;
constexpr uint32_t LOGICAL_ROWS = 5;
constexpr uint32_t LOGICAL_WIDTH = 6;
}  // namespace writer_arg

// 8 = bf16 half-sync DEST tile capacity (fp32_dest_acc_en=false, dst_full_sync_en=false —
// the ComputeConfig below) and simultaneously pack_untilize's max block width in that mode.
// The compute kernel static_asserts this against DEST_AUTO_LIMIT.
constexpr uint32_t max_block_width_tiles = 8;

// The clamp constant, as the fp32 bit pattern unary_max_tile takes (SFPU scalar params are
// bit-cast floats). The lowest finite bf16 is 0xFF7F (sign=1, exp=0xFE, mantissa=0x7F, value
// -3.3895313892515355e38); widened to fp32 by appending 16 zero mantissa bits: 0xFF7F0000.
// Must stay bit-identical to what run_topk_large_indices_route documents (topk.cpp) — the
// routed pipeline's -inf index parity depends on the op's input being floored exactly here.
constexpr float lowest_finite_bf16 = -3.3895313892515355e38f;
constexpr uint32_t clamp_bits = std::bit_cast<uint32_t>(lowest_finite_bf16);
static_assert(clamp_bits == 0xFF7F0000u, "clamp constant must be the lowest finite bf16 widened to fp32");

struct PrepWorkSplit {
    uint32_t width_tiles = 0;      // W_p / 32
    uint32_t total_tile_rows = 0;  // padded volume / W_p / 32 (all batches)
    uint32_t bw_full = 0;          // full block width in tiles
    uint32_t bw_last = 0;          // last (remainder) block width in tiles, in [1, bw_full]
    uint32_t nblocks_per_row = 0;
    uint32_t nblocks = 0;
};

PrepWorkSplit compute_work_split(const Tensor& input) {
    PrepWorkSplit split;
    const auto& padded = input.padded_shape();
    split.width_tiles = padded[-1] / TILE_WIDTH;
    split.total_tile_rows = (input.physical_volume() / padded[-1]) / TILE_HEIGHT;
    split.bw_full = std::min(max_block_width_tiles, split.width_tiles);
    split.nblocks_per_row = tt::div_up(split.width_tiles, split.bw_full);
    split.bw_last = split.width_tiles - (split.nblocks_per_row - 1) * split.bw_full;
    split.nblocks = split.total_tile_rows * split.nblocks_per_row;
    return split;
}

// Tile index of the first tile of block `b` (blocks are tile-row-major, so a contiguous block
// range maps to a contiguous tile range).
uint32_t first_tile_of_block(const PrepWorkSplit& split, uint32_t b) {
    return (b / split.nblocks_per_row) * split.width_tiles + (b % split.nblocks_per_row) * split.bw_full;
}

struct PrepCoreArgs {
    CoreCoord core;
    uint32_t ntiles = 0;
    uint32_t start_tile = 0;
    uint32_t nblocks_this_core = 0;
    uint32_t start_block = 0;
};

// Per-core runtime layout. create_descriptor and override_runtime_arguments both call this so
// the arg slots cannot drift. The program hash pins the block widths, the stick size, and the
// core partition; logical rows and tile_rows_per_batch are not in that hash.
struct PrepRuntimeLayout {
    PrepWorkSplit split;
    tt::tt_metal::CoreRangeSet all_cores;
    std::vector<PrepCoreArgs> cores;
    uint32_t tile_rows_per_batch = 0;
    uint32_t logical_rows = 0;
    uint32_t logical_width = 0;
};

PrepRuntimeLayout build_runtime_layout(const Tensor& input) {
    PrepRuntimeLayout layout;
    layout.split = compute_work_split(input);
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto block_split = ttnn::split_blocks_for_tilize(CoreCoord(grid.x, grid.y), layout.split.nblocks);
    layout.all_cores = block_split.all_cores;
    layout.tile_rows_per_batch = input.padded_shape()[-2] / TILE_HEIGHT;
    layout.logical_rows = input.logical_shape()[-2];
    layout.logical_width = input.logical_shape()[-1];

    // split_blocks_for_tilize(CoreCoord, ...) places core i at (i % grid.x, i / grid.x), cliff
    // last; enumerate in that exact order so the contiguous block partition lines up.
    layout.cores.reserve(block_split.ncores);
    uint32_t start_block = 0;
    for (uint32_t i = 0; i < block_split.ncores; ++i) {
        const bool is_cliff = block_split.nblocks_per_core_cliff > 0 && i + 1 == block_split.ncores;
        const uint32_t nblocks_this_core = is_cliff ? block_split.nblocks_per_core_cliff : block_split.nblocks_per_core;
        const uint32_t start_tile = first_tile_of_block(layout.split, start_block);
        const uint32_t ntiles_this_core =
            first_tile_of_block(layout.split, start_block + nblocks_this_core) - start_tile;
        layout.cores.push_back(PrepCoreArgs{
            .core = CoreCoord{i % grid.x, i / grid.x},
            .ntiles = ntiles_this_core,
            .start_tile = start_tile,
            .nblocks_this_core = nblocks_this_core,
            .start_block = start_block,
        });
        start_block += nblocks_this_core;
    }
    return layout;
}

}  // namespace

tt::tt_metal::ProgramDescriptor TopkRoutePrepProgramFactory::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto layout = build_runtime_layout(input);
    const auto& split = layout.split;
    const auto& all_cores = layout.all_cores;

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr, "topk_route_prep input tensor has no device buffer");
    TT_FATAL(output_buffer != nullptr, "topk_route_prep output tensor has no device buffer");

    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    tt::tt_metal::ProgramDescriptor desc;
    desc.cbs.reserve(2);
    desc.kernels.reserve(3);

    // Input CB: tile pages, double-buffered against one full block.
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2 * split.bw_full * tile_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = tile_bytes,
        }}},
    });

    // Output CB: one page per BLOCK (uniform bw_full-sized pages so pack_untilize's contiguous
    // block write can never straddle the CB wrap; a bw_last block simply leaves the page tail
    // unused), double-buffered. Each page holds the untilized block: 32 sticks of bw*32 elements.
    const uint32_t output_page_bytes = split.bw_full * tile_bytes;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2 * output_page_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = output_page_bytes,
        }}},
    });

    // Reader: reused BY PATH, with the same compile args the untilize parallelize-column factory
    // passes (TensorAccessorArgs only; CB c_0 and its page size come from the CB interface).
    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_args);
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    // Compute: fused clamp + pack_untilize. CLAMP_BITS is the fp32 bit pattern documented above.
    // bf16 half-sync: DEST holds 8 tiles == max_block_width_tiles above.
    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_route_prep_untilize_clamp.cpp";
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {split.bw_full, split.bw_last, input_cb_index, output_cb_index};
    compute_desc.defines = {{"CLAMP_BITS", std::to_string(clamp_bits) + "u"}};
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .math_approx_mode = false,
    };

    std::vector<uint32_t> writer_compile_args = {split.bw_full, split.bw_last, output_cb_index};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_topk_route_prep_stick_layout.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_args);
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    for (const auto& core_args : layout.cores) {
        const auto& core = core_args.core;
        reader_desc.emplace_runtime_args(core, {input_buffer, core_args.ntiles, core_args.start_tile});
        compute_desc.emplace_runtime_args(
            core, {core_args.nblocks_this_core, core_args.start_block, split.nblocks_per_row});
        writer_desc.emplace_runtime_args(
            core,
            {output_buffer,
             core_args.nblocks_this_core,
             core_args.start_block,
             split.nblocks_per_row,
             layout.tile_rows_per_batch,
             layout.logical_rows,
             layout.logical_width});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

void TopkRoutePrepProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr, "topk_route_prep input tensor has no device buffer");
    TT_FATAL(output_buffer != nullptr, "topk_route_prep output tensor has no device buffer");

    const auto layout = build_runtime_layout(input);
    const uint32_t input_address = input_buffer->address();
    const uint32_t output_address = output_buffer->address();

    for (const auto& core_args : layout.cores) {
        const auto& core = core_args.core;
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        reader_args[reader_arg::SRC_ADDR] = input_address;
        reader_args[reader_arg::NTILES] = core_args.ntiles;
        reader_args[reader_arg::START_ID] = core_args.start_tile;

        auto& compute_args = tt::tt_metal::GetRuntimeArgs(program, kComputeKernelIdx, core);
        compute_args[compute_arg::NBLOCKS] = core_args.nblocks_this_core;
        compute_args[compute_arg::START_BLOCK] = core_args.start_block;
        compute_args[compute_arg::NBLOCKS_PER_ROW] = layout.split.nblocks_per_row;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
        writer_args[writer_arg::DST_ADDR] = output_address;
        writer_args[writer_arg::NBLOCKS] = core_args.nblocks_this_core;
        writer_args[writer_arg::START_BLOCK] = core_args.start_block;
        writer_args[writer_arg::NBLOCKS_PER_ROW] = layout.split.nblocks_per_row;
        writer_args[writer_arg::TILE_ROWS_PER_BATCH] = layout.tile_rows_per_batch;
        writer_args[writer_arg::LOGICAL_ROWS] = layout.logical_rows;
        writer_args[writer_arg::LOGICAL_WIDTH] = layout.logical_width;
    }
}

}  // namespace ttnn::operations::reduction::topk_route_prep::program
