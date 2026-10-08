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
    CB_Q_RM,
    CB_K_IN,
    CB_V_IN,
    CB_IDX,
    CB_CTRL,
    CB_KREQ,
    CB_KACK,
    K_TILE_BYTES,
    V_TILE_BYTES,
    CAUSAL_MASK_ENABLED,
    BLOCK_SIZE,
    CB_VMASK,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    KV_CACHE_SLOTS,
    CB_K_CACHE,
    CB_V_CACHE,
    CB_SLOT,
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
    CB_OUT_RM,
    CB_SCALE,
    CB_COL_IDENTITY,
    CB_K_IN,
    CB_V_IN,
    CB_KREQ,
    CB_KACK,
    K_TILE_BYTES,
    V_TILE_BYTES,
    CAUSAL_MASK_ENABLED,
    CB_NEGINF,
    BLOCK_CYCLIC,
    BC_CHUNK_LOCAL,
    BC_SP,
    BC_SHARD_STRIDE_GAP,
    BC_SLAB_STRIDE_GAP,
    KV_CACHE_SLOTS,
    CB_K_CACHE,
    CB_V_CACHE,
    COUNT
};
}  // namespace writer_ct

namespace compute_ct {
enum : uint32_t {
    H,
    DHT,
    VDHT,
    SKT,
    SCALE_FP32,
    CB_Q_RM,
    CB_Q_IN,
    CB_K_IN,
    CB_V_IN,
    CB_SCALE,
    CB_QK_IM,
    CB_MAX_A,
    CB_MAX_B,
    CB_SUM_A,
    CB_SUM_B,
    CB_OUT_A,
    CB_OUT_B,
    CB_CORR,
    CB_OUT_IM,
    CB_OUT_RM,
    CB_CTRL,
    CB_COL_IDENTITY,
    CB_RECIP_SCRATCH,
    QSB,
    CAUSAL_MASK_ENABLED,
    CB_NEGINF,
    CB_VMASK,
    KV_CACHE_SLOTS,
    CB_K_CACHE,
    CB_V_CACHE,
    CB_SLOT,
    COUNT
};
}  // namespace compute_ct

// Per-core K/V block cache.
constexpr uint32_t KV_CACHE_SLOTS_MAX = 64;      // bounds the reader's per-block residency scan
constexpr uint32_t KV_CACHE_SLOT_DEPTH = 2;      // cb_slot depth: the reader runs one block ahead of compute
// A cache needs two slots: the reader never evicts the previous block's slot (compute may still read it), and a
// one-slot cache would have no run-ahead anyway. Fewer fitting selects the streamed kernels.
constexpr uint32_t KV_CACHE_SLOTS_MIN = 2;

}  // namespace sparse_sdpa_msa
