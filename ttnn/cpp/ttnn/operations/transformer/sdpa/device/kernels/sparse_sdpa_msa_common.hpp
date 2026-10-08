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

// Compile-time arguments each kernel decodes positionally before its TensorAccessorArgs block. The factory checks
// its vectors against these counts before appending the accessors, so an argument added or dropped on one side
// fails at program creation instead of shifting the accessor decode.
constexpr uint32_t READER_CT_ARGS = 33;
constexpr uint32_t WRITER_CT_ARGS = 28;
constexpr uint32_t COMPUTE_CT_ARGS = 31;

// Per-core K/V block cache.
constexpr uint32_t KV_CACHE_SLOTS_MAX = 64;      // bounds the reader's per-block residency scan
constexpr uint32_t KV_CACHE_SLOT_DEPTH_MAX = 2;  // blocks the reader may run ahead of compute

}  // namespace sparse_sdpa_msa
