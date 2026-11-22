// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Host/device constants shared by the sparse_sdpa_msa factory and its JIT kernels: the reader->writer gather
// request record, the per-token control record, the compile-time-argument counts, and the per-core K/V
// block-cache limits. Constexpr only, so both compilation paths agree. The CB page sizes that hold the records
// are a host concern (message_page_bytes in the device operation); the kernels address the records by word.
namespace sparse_sdpa_msa {

// cb_kreq page: one gather request from the reader to the writer.
namespace kreq {
constexpr uint32_t BLOCK_ID = 0;  // logical block id; the writer applies the block-cyclic remap itself
constexpr uint32_t FLAGS = 1;     // LAST | FETCH
constexpr uint32_t SLOT = 2;      // block-cache slot to fill (0 on the streamed path)
constexpr uint32_t WORD_COUNT = 3;

constexpr uint32_t LAST = 1u << 0;   // the token's last block: the writer drains the output after it
constexpr uint32_t FETCH = 1u << 1;  // fetch the lower tile halves (every block when streaming; misses when cached)
}  // namespace kreq

// cb_ctrl page: per-token control from the reader to compute.
namespace ctrl {
constexpr uint32_t ACTIVE_BLOCKS = 0;  // blocks selected for this token
constexpr uint32_t DIAG_CHUNK = 1;     // causal: chunk index of the diagonal block (sentinel = no token mask)
constexpr uint32_t BOUNDARY_TILE = 2;  // causal: key-tiles >= this are fully masked in the diagonal block
constexpr uint32_t BOUNDARY_COL = 3;   // causal: partial-column boundary within BOUNDARY_TILE (0 = none)
constexpr uint32_t WORD_COUNT = 4;
}  // namespace ctrl

// Circular buffers beside the recipe compute layout (CBs 0-16 and 29-31, recipe_compute_program; the kernels use
// recipe_ring_cb's names for those). The K CB holds one block in the recipe's K^T layout (block tile (r, c) at
// c * k_tiles + r), the V CB one block as stored; with the block cache each holds the cache's slots.
namespace recipe_cb {
constexpr uint32_t q_rm = 20;    // Q rows (row-major, reader -> compute tilize)
constexpr uint32_t idx = 21;     // reader-internal: one token's block-id row (uint32)
constexpr uint32_t ctrl = 22;    // reader -> compute per token: active block count + causal geometry (ctrl:: words)
constexpr uint32_t kreq = 23;    // reader -> writer gather request (kreq)
constexpr uint32_t kack = 24;    // writer -> reader: its lower tile halves landed
constexpr uint32_t out_rm = 25;  // untilized row-major out (compute -> writer)
constexpr uint32_t slot = 26;    // block cache only: per-block slot id reader -> compute (depth = reader run-ahead)
}  // namespace recipe_cb

// Compile-time argument indices, one enum per kernel. The factory fills each slot by name and the kernel reads it
// by the same name, so the two cannot drift; COUNT is where the kernel's TensorAccessorArgs block begins.
namespace reader_ct {
enum : uint32_t {
    H_LOGICAL,
    H,
    S,
    TOPK,
    N_KV,
    Q_ROW_BYTES,
    IDX_ROW_BYTES,
    K_TILES_PER_BLOCK,
    V_TILES_PER_BLOCK,
    K_HALF,
    V_HALF,
    K_TILE_BYTES,
    V_TILE_BYTES,
    CAUSAL_MASK_ENABLED,
    BLOCK_SIZE,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    KV_CACHE_SLOTS,
    D_TILES,  // K's tile columns (the K^T layout's row count)
    Q_TILES,  // Q tile rows compute processes (padding rows past H zero)
    COUNT
};
}  // namespace reader_ct

namespace writer_ct {
enum : uint32_t {
    H_LOGICAL,
    S,
    N_KV,
    ROW_BYTES,
    BLOCK_TILES,
    K_TILES_PER_BLOCK,
    V_TILES_PER_BLOCK,
    K_HALF,
    V_HALF,
    K_TILE_BYTES,
    V_TILE_BYTES,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    KV_CACHE_SLOTS,
    D_TILES,
    COUNT
};
}  // namespace writer_ct

// The K^T position of tile i (DRAM order: row tile i / d_tiles, column i % d_tiles) of a block of k_tiles row tiles.
constexpr uint32_t kt_tile_position(uint32_t i, uint32_t d_tiles, uint32_t k_tiles) {
    return (i % d_tiles) * k_tiles + i / d_tiles;
}

// Per-core K/V block cache.
constexpr uint32_t KV_CACHE_SLOTS_MAX = 64;  // bounds the reader's per-block residency scan
constexpr uint32_t KV_CACHE_SLOT_DEPTH = 2;  // cb_slot depth: the reader runs one block ahead of compute
// A cache needs two slots: the reader never evicts the previous block's slot (compute may still read it), and a
// one-slot cache would have no run-ahead anyway. The host floor is max(this, topk): fewer slots than
// one query selects give no reuse.
constexpr uint32_t KV_CACHE_SLOTS_MIN = 2;

}  // namespace sparse_sdpa_msa
