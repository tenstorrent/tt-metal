// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

// Sizes of the matmul factories' dataflow (circular) buffers, as functions of plain data. Each factory sizes its
// buffers with these, and the program config selection uses the same functions to check what fits L1, so the two
// can't disagree.
namespace ttnn::operations::matmul {

// The per-core work a program config describes (in tiles). Reuse's output block is the whole per-core block.
struct Blocking {
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;
    uint32_t in0_block_w = 0;
    uint32_t out_block_h = 0;
    uint32_t out_block_w = 0;
    uint32_t out_subblock_h = 0;
    uint32_t out_subblock_w = 0;
};

// Everything besides the blocking that the factories' buffer sizes depend on
struct BufferContext {
    tt::tt_metal::Tile in0_tile;     // A's tile as the matmul reads it (after transpose_a)
    tt::tt_metal::Tile in1_tile;     // B's tile as the matmul reads it
    tt::tt_metal::Tile output_tile;  // in0_tile's height x in1_tile's width
    tt::DataFormat in0_format = tt::DataFormat::Float16_b;
    tt::DataFormat in1_format = tt::DataFormat::Float16_b;
    tt::DataFormat output_format = tt::DataFormat::Float16_b;
    uint32_t bias_tile_bytes = 0;  // 0 without a fused bias
    bool bias_sharded = false;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = false;
    bool untilize_out = false;
    tt::tt_metal::TensorMemoryLayout in0_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    tt::tt_metal::TensorMemoryLayout in1_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    tt::tt_metal::TensorMemoryLayout out_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    bool in1_in_dram = true;
    uint32_t in0_shard_width = 0;   // tiles; a sharded A's shard width
    uint32_t in1_shard_height = 0;  // tiles; a sharded B's shard height
    uint32_t dram_alignment = 32;   // bytes; tiles read from DRAM are padded to it
    uint32_t Mt = 0;                // per batch
    uint32_t Kt = 0;
};

// The context of matmul(in0, in1) -> out as the factories see it
BufferContext buffer_context(
    const tt::tt_metal::MeshTensor& in0,
    const tt::tt_metal::MeshTensor& in1,
    const tt::tt_metal::MeshTensor& out,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    uint32_t bias_tile_bytes,  // 0 without a fused bias
    bool bias_sharded,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    bool untilize_out,
    uint32_t Mt,
    uint32_t Kt);

// One buffer: entries of entry_size bytes. A borrowed buffer is a view of a tensor already in L1.
struct BufferSize {
    uint32_t entry_size = 0;
    uint32_t num_entries = 0;
    bool borrowed = false;

    uint32_t allocated_bytes() const { return borrowed ? 0 : entry_size * num_entries; }
};

// A factory's buffers on a core with work. Buffers a factory doesn't create have no entries.
struct MatmulBuffers {
    bool packer_l1_acc_en = false;  // partials accumulate in L1 between K blocks
    tt::DataFormat interm0_format = tt::DataFormat::Float16_b;
    bool share_out_interm = false;   // the partials live in the output buffer
    bool bias_reload_alias = false;  // an fp32 view of the partials for the bias reload (no extra memory)
    BufferSize in0;
    BufferSize in1;
    BufferSize in0_sharded;  // a sharded A's resident shard, when the factory reads A from it
    BufferSize out;
    BufferSize interm0;
    BufferSize bias;
    BufferSize in0_transposed;  // when A's tiles are transposed

    // L1 the factory allocates for these on a core with work
    uint32_t allocated_bytes() const;
};

// `batches`: how many batches the program loops over per operand (1 when the batch is fused into M)
MatmulBuffers mcast_2d_buffers(const BufferContext& context, const Blocking& blocking, uint32_t batches);
MatmulBuffers mcast_1d_in0_buffers(
    const BufferContext& context, const Blocking& blocking, uint32_t in0_batches, uint32_t in1_batches);
MatmulBuffers mcast_1d_in1_buffers(
    const BufferContext& context, const Blocking& blocking, uint32_t in0_batches, uint32_t in1_batches);
MatmulBuffers reuse_buffers(const BufferContext& context, const Blocking& blocking);

}  // namespace ttnn::operations::matmul
