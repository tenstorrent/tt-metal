// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Refinement 4 — Causal masking (mask_mode=causal), shared reader/compute logic.
//
// Causal SDPA requires self-attention (S_q == S_kv → SQT == SKVT). For a Q-block
// at query-chunk `qc` and a KV-block at key-chunk `kc` (chunks measured in tiles
// of SQ_CHUNK_T / SK_CHUNK_T), the online-softmax recurrence only needs the
// KV-blocks that contain at least one key tile at or before the block's last
// query tile — all strictly-future KV-blocks are fully −inf and are BLOCK-SKIPPED
// (≈ half the KV work). Within the processed range, only the diagonal-straddling
// block(s) carry a non-trivial mask; the fully-past blocks are all-zero and need
// no mask add at all.
//
// These two predicates are the SINGLE SOURCE OF TRUTH shared by the reader (which
// generates + pushes cb_mask_in only for straddling blocks and truncates its KV
// read loop) and the compute kernel (which adds the mask only for straddling
// blocks and truncates its KV compute loop). Reader and compute MUST agree
// tile-for-tile or the cb_k_in / cb_v_in / cb_mask_in FIFOs deadlock.

#pragma once

#include <cstdint>

namespace sdpa_causal {

// Number of KV-blocks to process for query-chunk `qc` (block-skip strictly-future
// blocks). q_high_tile = (qc+1)·sq_ct is the exclusive upper query-tile bound of
// the block; a KV-block kc is processed iff kc·sk_ct < q_high_tile, i.e. for
// kc in [0, ceil(q_high_tile / sk_ct)). Capped at k_num_chunks for safety.
inline uint32_t kc_count(uint32_t qc, uint32_t sq_ct, uint32_t sk_ct, uint32_t k_num_chunks) {
    const uint32_t q_high_tile = (qc + 1u) * sq_ct;
    const uint32_t cnt = (q_high_tile + sk_ct - 1u) / sk_ct;
    return cnt < k_num_chunks ? cnt : k_num_chunks;
}

// Does KV-block (qc, kc) straddle / cross the causal diagonal (needs the mask)?
// A block is fully-past (all keys strictly before all queries → all-zero mask,
// no add needed) iff its last key tile < its first query tile:
//   (kc+1)·sk_ct <= qc·sq_ct.
// So it needs masking iff (kc+1)·sk_ct > qc·sq_ct.
inline bool needs_mask(uint32_t qc, uint32_t kc, uint32_t sq_ct, uint32_t sk_ct) {
    return (kc + 1u) * sk_ct > qc * sq_ct;
}

}  // namespace sdpa_causal
