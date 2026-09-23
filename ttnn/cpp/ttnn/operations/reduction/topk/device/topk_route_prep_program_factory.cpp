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
#include <map>
#include <string>

using namespace tt::constants;

namespace ttnn::operations::reduction::topk_route_prep::program {

namespace {

constexpr uint32_t input_cb_index = tt::CBIndex::c_0;  // c_0: hardcoded in the reused reader
constexpr uint32_t output_cb_index = tt::CBIndex::c_16;

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

// (Re)computes every runtime arg from the tensors. Called by create() and, on cache hits, by
// override_runtime_arguments(); the program hash pins every structural input (core partition,
// block widths, stick size), so only addresses and the logical R/W clamps can differ here.
void set_runtime_args(
    tt::tt_metal::Program& program,
    const TopkRoutePrepSharedVariables& shared,
    const Tensor& input,
    const Tensor& output) {
    const auto split = compute_work_split(input);
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto block_split = ttnn::split_blocks_for_tilize(CoreCoord(grid.x, grid.y), split.nblocks);

    const uint32_t tile_rows_per_batch = input.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t logical_rows = input.logical_shape()[-2];
    const uint32_t logical_width = input.logical_shape()[-1];

    uint32_t start_block = 0;
    for (uint32_t i = 0; i < shared.cores.size(); ++i) {
        const CoreCoord& core = shared.cores[i];
        const bool is_cliff = block_split.nblocks_per_core_cliff > 0 && i + 1 == shared.cores.size();
        const uint32_t nblocks_this_core = is_cliff ? block_split.nblocks_per_core_cliff : block_split.nblocks_per_core;
        const uint32_t ntiles_this_core =
            first_tile_of_block(split, start_block + nblocks_this_core) - first_tile_of_block(split, start_block);

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            core,
            {input.buffer()->address(),                  // src_addr
             ntiles_this_core,                           // ntiles
             first_tile_of_block(split, start_block)});  // start_id

        tt::tt_metal::SetRuntimeArgs(
            program, shared.compute_kernel_id, core, {nblocks_this_core, start_block, split.nblocks_per_row});

        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.writer_kernel_id,
            core,
            {output.buffer()->address(),  // dst_addr
             nblocks_this_core,
             start_block,
             split.nblocks_per_row,
             tile_rows_per_batch,
             logical_rows,
             logical_width});

        start_block += nblocks_this_core;
    }
}

}  // namespace

TopkRoutePrepProgramFactory::cached_program_t TopkRoutePrepProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    auto program = tt::tt_metal::CreateProgram();

    const auto split = compute_work_split(input);
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto block_split = ttnn::split_blocks_for_tilize(CoreCoord(grid.x, grid.y), split.nblocks);
    const auto& all_cores = block_split.all_cores;

    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    // Input CB: tile pages, double-buffered against one full block.
    const auto input_cb_config = tt::tt_metal::CircularBufferConfig(
                                     2 * split.bw_full * tile_bytes, {{input_cb_index, tt::DataFormat::Float16_b}})
                                     .set_page_size(input_cb_index, tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);

    // Output CB: one page per BLOCK (uniform bw_full-sized pages so pack_untilize's contiguous
    // block write can never straddle the CB wrap; a bw_last block simply leaves the page tail
    // unused), double-buffered. Each page holds the untilized block: 32 sticks of bw*32 elements.
    const uint32_t output_page_bytes = split.bw_full * tile_bytes;
    const auto output_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * output_page_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, output_page_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // Reader: reused BY PATH, with the same compile args the untilize parallelize-column factory
    // passes (TensorAccessorArgs only; CB c_0 and its page size come from the CB interface).
    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_args);
    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Compute: fused clamp + pack_untilize. CLAMP_BITS is the fp32 bit pattern documented above.
    std::map<std::string, std::string> compute_defines;
    compute_defines["CLAMP_BITS"] = std::to_string(clamp_bits) + "u";
    const std::vector<uint32_t> compute_compile_args = {split.bw_full, split.bw_last, input_cb_index, output_cb_index};
    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_route_prep_untilize_clamp.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{// bf16 half-sync: DEST holds 8 tiles == max_block_width_tiles above.
                                    .fp32_dest_acc_en = false,
                                    .dst_full_sync_en = false,
                                    .compile_args = compute_compile_args,
                                    .defines = compute_defines});

    std::vector<uint32_t> writer_compile_args = {split.bw_full, split.bw_last, output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_args);
    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_topk_route_prep_stick_layout.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // split_blocks_for_tilize(CoreCoord, ...) places core i at (i % grid.x, i / grid.x), cliff
    // last; enumerate in that exact order so the contiguous block partition lines up.
    std::vector<CoreCoord> cores;
    cores.reserve(block_split.ncores);
    for (uint32_t i = 0; i < block_split.ncores; ++i) {
        cores.push_back(CoreCoord{i % grid.x, i / grid.x});
    }

    TopkRoutePrepSharedVariables shared{
        .reader_kernel_id = reader_kernel,
        .compute_kernel_id = compute_kernel,
        .writer_kernel_id = writer_kernel,
        .cores = std::move(cores)};
    set_runtime_args(program, shared, input, output);

    return cached_program_t{std::move(program), std::move(shared)};
}

void TopkRoutePrepProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    set_runtime_args(
        cached_program.program, cached_program.shared_variables, tensor_args.input_tensor, tensor_return_value);
}

}  // namespace ttnn::operations::reduction::topk_route_prep::program
