// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_buffers.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::operations::matmul {

namespace {

using tt::tt_metal::TensorMemoryLayout;

bool sharded(TensorMemoryLayout layout) { return layout != TensorMemoryLayout::INTERLEAVED; }

bool transposed(const tt::tt_metal::Tile& tile) {
    return tile.get_transpose_of_faces() && tile.get_transpose_within_face();
}

// Tiles read from DRAM are padded to the DRAM alignment (e.g. bfp8 32x16 = 544 B on Blackhole's 64 B), and the
// readers write them to L1 at that stride. A view of a sharded tensor keeps the natural size.
uint32_t entry_size(uint32_t tile_bytes, bool natural, uint32_t dram_alignment) {
    return natural ? tile_bytes : tt::align(tile_bytes, dram_alignment);
}

// What every factory shares: the partials format, the output and partials buffers (one region unless the format,
// an untilized multi-subblock output or a sharded output's block count forces them apart), the bias reload alias
// and the transposed copy of in0. `out_sharded` and `out_block_tiles` are the factory's own notions.
void finish(
    const BufferContext& c,
    uint32_t out_block_tiles,
    uint32_t out_block_w,
    uint32_t out_subblock_w,
    bool out_sharded,
    bool sharded_out_needs_own_partials,
    uint32_t out_shard_tiles,
    MatmulBuffers& r) {
    r.out = {
        c.output_tile.get_tile_size(c.output_format), out_sharded ? out_shard_tiles : out_block_tiles, out_sharded};
    // Before a subblock is chosen (out_subblock_w == 0), assume several across the block: that can only
    // overestimate L1
    const bool several_subblocks_w = out_subblock_w == 0 || out_block_w / out_subblock_w > 1;
    r.share_out_interm =
        !(sharded_out_needs_own_partials || r.interm0_format != c.output_format ||
          (c.untilize_out && several_subblocks_w));
    r.interm0 = {
        c.output_tile.get_tile_size(r.interm0_format),
        r.share_out_interm ? r.out.num_entries : out_block_tiles,
        r.share_out_interm && out_sharded};
    r.bias_reload_alias = c.fp32_dest_acc_en && r.interm0_format == tt::DataFormat::Float32 && c.bias_tile_bytes != 0;
    if (transposed(c.in0_tile)) {
        r.in0_transposed = {r.in0.entry_size, r.in0.num_entries, false};
    }
}

MatmulBuffers start(const BufferContext& c, uint32_t in0_block_w) {
    MatmulBuffers r;
    // Partials accumulate in L1 whenever they are kept between K blocks
    r.packer_l1_acc_en = c.packer_l1_acc && c.Kt / in0_block_w > 1;
    // fp32 partials with fp32 accumulation; else bf16 in L1, or the output format without L1 accumulation
    r.interm0_format = c.fp32_dest_acc_en   ? tt::DataFormat::Float32
                       : r.packer_l1_acc_en ? tt::DataFormat::Float16_b
                                            : c.output_format;
    return r;
}

constexpr uint32_t MCAST_BUFFERING = operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;

// A sharded output can share its region with the partials only when the core computes a single output block:
// spill and reload advance the region's pointers by one block, which only wraps back to the start when the region
// holds exactly one block (#58046).
bool several_output_blocks(const Blocking& b) { return b.per_core_M != b.out_block_h || b.per_core_N != b.out_block_w; }

}  // namespace

BufferContext buffer_context(
    const tt::tt_metal::MeshTensor& in0,
    const tt::tt_metal::MeshTensor& in1,
    const tt::tt_metal::MeshTensor& out,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    uint32_t bias_tile_bytes,
    bool bias_sharded,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    bool untilize_out,
    uint32_t Mt,
    uint32_t Kt) {
    BufferContext c{
        .in0_tile = in0_tile,
        .in1_tile = in1_tile,
        .output_tile = output_tile,
        .in0_format = tt::tt_metal::datatype_to_dataformat_converter(in0.dtype()),
        .in1_format = tt::tt_metal::datatype_to_dataformat_converter(in1.dtype()),
        .output_format = tt::tt_metal::datatype_to_dataformat_converter(out.dtype()),
        .bias_tile_bytes = bias_tile_bytes,
        .bias_sharded = bias_sharded,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .packer_l1_acc = packer_l1_acc,
        .untilize_out = untilize_out,
        .in0_layout = in0.memory_config().memory_layout(),
        .in1_layout = in1.memory_config().memory_layout(),
        .out_layout = out.memory_config().memory_layout(),
        .in1_in_dram = in1.mesh_buffer().device_local_config().buffer_type == tt::tt_metal::BufferType::DRAM,
        .dram_alignment = tt::tt_metal::hal::get_dram_alignment(),
        .Mt = Mt,
        .Kt = Kt,
    };
    if (in0.memory_config().is_sharded() && in0.shard_spec().has_value()) {
        c.in0_shard_width = in0.shard_spec()->shape[1] / in0_tile.get_width();
    }
    if (in1.memory_config().is_sharded() && in1.shard_spec().has_value()) {
        c.in1_shard_height = in1.shard_spec()->shape[0] / in1_tile.get_height();
    }
    return c;
}

uint32_t MatmulBuffers::allocated_bytes() const {
    return in0.allocated_bytes() + in1.allocated_bytes() + in0_sharded.allocated_bytes() + out.allocated_bytes() +
           (share_out_interm ? 0 : interm0.allocated_bytes()) + bias.allocated_bytes() +
           in0_transposed.allocated_bytes();
}

MatmulBuffers mcast_2d_buffers(const BufferContext& c, const Blocking& b, uint32_t batches) {
    MatmulBuffers r = start(c, b.in0_block_w);
    const uint32_t num_blocks = c.Kt / b.in0_block_w;
    // A is read in place when height-sharded, extracted from a block shard otherwise; B in place when in L1
    const bool in0_block_sharded = c.in0_layout == TensorMemoryLayout::BLOCK_SHARDED;
    const bool in0_height_sharded = c.in0_layout == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool in1_sharded =
        c.in1_layout == TensorMemoryLayout::WIDTH_SHARDED || c.in1_layout == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool out_sharded = c.out_layout == TensorMemoryLayout::BLOCK_SHARDED;

    const uint32_t in0_tile = c.in0_tile.get_tile_size(c.in0_format);
    const uint32_t in1_tile = c.in1_tile.get_tile_size(c.in1_format);
    // Inputs are double-buffered unless the program runs a single K block once
    const uint32_t buffering = batches * num_blocks > 1 ? MCAST_BUFFERING : 1;
    r.in0 = {
        entry_size(in0_tile, in0_block_sharded || in0_height_sharded, c.dram_alignment),
        b.out_block_h * b.in0_block_w * buffering,
        in0_height_sharded};
    r.in1 = {
        entry_size(in1_tile, in1_sharded, c.dram_alignment),
        b.out_block_w * b.in0_block_w * buffering,
        in1_sharded && !c.in1_in_dram};
    if (in0_block_sharded) {
        r.in0_sharded = {in0_tile, b.per_core_M * c.in0_shard_width, true};
    }
    if (c.bias_tile_bytes != 0) {
        r.bias = {entry_size(c.bias_tile_bytes, false, c.dram_alignment), b.out_block_w, false};
    }
    finish(
        c,
        b.out_block_h * b.out_block_w,
        b.out_block_w,
        b.out_subblock_w,
        out_sharded,
        out_sharded && several_output_blocks(b),
        b.per_core_M * b.per_core_N,
        r);
    return r;
}

MatmulBuffers mcast_1d_in0_buffers(
    const BufferContext& c, const Blocking& b, uint32_t in0_batches, uint32_t in1_batches) {
    MatmulBuffers r = start(c, b.in0_block_w);
    const uint32_t num_blocks = c.Kt / b.in0_block_w;
    const bool in0_sharded = sharded(c.in0_layout);
    const bool in1_sharded = sharded(c.in1_layout);
    const bool out_sharded = sharded(c.out_layout);

    const uint32_t in0_tile = c.in0_tile.get_tile_size(c.in0_format);
    const uint32_t in1_tile = c.in1_tile.get_tile_size(c.in1_format);
    // in0 is multicast from a copy (extracted from A's shard when A is sharded); B's shard is read in place
    r.in0 = {
        entry_size(in0_tile, in0_sharded, c.dram_alignment),
        b.out_block_h * b.in0_block_w * (in0_batches * num_blocks > 1 ? MCAST_BUFFERING : 1),
        false};
    if (in0_sharded) {
        r.in0_sharded = {in0_tile, b.per_core_M * c.in0_shard_width, true};
    }
    r.in1 = {
        entry_size(in1_tile, in1_sharded, c.dram_alignment),
        in1_sharded ? b.per_core_N * c.in1_shard_height
                    : b.out_block_w * b.in0_block_w * (in1_batches * num_blocks > 1 ? MCAST_BUFFERING : 1),
        in1_sharded};
    if (c.bias_tile_bytes != 0) {
        r.bias = {entry_size(c.bias_tile_bytes, c.bias_sharded, c.dram_alignment), b.out_block_w, c.bias_sharded};
    }
    finish(
        c,
        b.out_block_h * b.out_block_w,
        b.out_block_w,
        b.out_subblock_w,
        out_sharded,
        out_sharded && several_output_blocks(b),
        b.per_core_M * b.per_core_N,
        r);
    return r;
}

MatmulBuffers mcast_1d_in1_buffers(
    const BufferContext& c, const Blocking& b, uint32_t in0_batches, uint32_t in1_batches) {
    MatmulBuffers r = start(c, b.in0_block_w);
    const uint32_t num_blocks = c.Kt / b.in0_block_w;
    const bool in0_sharded = sharded(c.in0_layout);
    const bool out_sharded = sharded(c.out_layout);

    const uint32_t in0_tile = c.in0_tile.get_tile_size(c.in0_format);
    const uint32_t in1_tile = c.in1_tile.get_tile_size(c.in1_format);
    const uint32_t in0_block_tiles = b.out_block_h * b.in0_block_w;
    // A sharded A is read in place, unless K is split across blocks of its shard or the core has several
    // block rows and columns (a row block is needed twice, and advancing would land on the wrong one): then each
    // block is copied out of the shard
    const bool extract = in0_sharded && (c.in0_shard_width / b.in0_block_w > 1 ||
                                         (b.per_core_M / b.out_block_h > 1 && b.per_core_N / b.out_block_w > 1));
    uint32_t in0_entries = in0_block_tiles;
    if (in0_batches == 1 && in1_batches > 1) {
        in0_entries = b.per_core_M * num_blocks * b.in0_block_w;  // A resident across B's batches
    } else if (in0_sharded) {
        in0_entries = num_blocks * b.per_core_M * b.in0_block_w * in0_batches;
    } else if (in0_batches * num_blocks > 1) {
        in0_entries *= 2;
    }
    r.in0 = {entry_size(in0_tile, in0_sharded, c.dram_alignment), in0_entries, in0_sharded && !extract};
    if (extract) {
        r.in0_sharded = {in0_tile, in0_block_tiles, true};
    }
    r.in1 = {
        entry_size(in1_tile, false, c.dram_alignment),
        b.out_block_w * b.in0_block_w * (in1_batches * num_blocks > 1 ? 2 : 1),
        false};
    if (c.bias_tile_bytes != 0) {
        r.bias = {entry_size(c.bias_tile_bytes, false, c.dram_alignment), b.out_block_w, false};
    }
    finish(
        c,
        b.out_block_h * b.out_block_w,
        b.out_block_w,
        b.out_subblock_w,
        out_sharded,
        out_sharded && several_output_blocks(b),
        b.per_core_M * b.per_core_N,
        r);
    return r;
}

MatmulBuffers reuse_buffers(const BufferContext& c, const Blocking& b) {
    MatmulBuffers r = start(c, b.in0_block_w);
    const uint32_t num_blocks = c.Kt / b.in0_block_w;
    const bool in0_sharded = sharded(c.in0_layout);
    const bool in1_sharded = sharded(c.in1_layout);
    const bool out_sharded = sharded(c.out_layout);
    // A block of several whole batch matrices
    const uint32_t batches_per_block = b.per_core_M > c.Mt ? b.per_core_M / c.Mt : 1;
    const uint32_t per_core_M_per_batch = b.per_core_M > c.Mt ? c.Mt : b.per_core_M;

    const uint32_t in0_tile = c.in0_tile.get_tile_size(c.in0_format);
    const uint32_t in1_tile = c.in1_tile.get_tile_size(c.in1_format);
    // Sharded operands are read in place, all of K at once; interleaved ones are double-buffered
    r.in0 = {
        entry_size(in0_tile, in0_sharded, c.dram_alignment),
        in0_sharded ? b.per_core_M * c.Kt : per_core_M_per_batch * b.in0_block_w * 2,
        in0_sharded};
    r.in1 = {
        entry_size(in1_tile, in1_sharded, c.dram_alignment),
        b.per_core_N * b.in0_block_w * (in1_sharded ? num_blocks * batches_per_block : 2),
        in1_sharded};
    if (c.bias_tile_bytes != 0) {
        // The whole per-batch [M, N] bias block, loaded once
        r.bias = {c.bias_tile_bytes, per_core_M_per_batch * b.per_core_N, false};
    }
    const uint32_t block_tiles = b.per_core_M * b.per_core_N;
    finish(c, block_tiles, b.per_core_N, b.out_subblock_w, out_sharded, false, block_tiles, r);
    return r;
}

}  // namespace ttnn::operations::matmul
