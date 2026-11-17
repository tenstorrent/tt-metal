// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tail_writes_both_nocs (perf experiment): split the LAST block's output write stream between the writer
// (BRISC / NoC1) and the reader (NCRISC / NoC0), which is idle once its K-walk is done.
//
// The output CB keeps ONE consumer (the writer: cb_wait_front / cb_pop_front). Per batch of the tail the
// writer posts the batch's L1 address (mailbox) and a sequence number, issues its own share, flushes, then
// waits for the reader's `done` >= seq before popping (the reader's share has left L1). The reader, after
// its walk, waits for each post, issues its share on NoC0, flushes and publishes `done`. Both words are
// program semaphores (zero-initialized by dispatch), used as plain local L1 words.
//
// TAIL_SPLIT_MODE (define; absent = 0):
//   0 off (baseline: the writer writes everything on NoC1)
//   1 column cut: the reader owns tiles n < blk_cols * TAIL_NC_PCT / 100 of every row, the writer the rest
//   2 tile alternation: the reader owns even n, the writer odd n
//   3 row alternation inside a batch: the reader owns even rows (needs batch_rows >= 2)
//   4 per-core column cut: the reader owns tiles n < ncut (a per-core runtime arg, host-chosen from the
//     core's position: the NoC whose route to the DRAM banks is shorter for this core takes more)
#pragma once
#include <cstdint>

#ifndef TAIL_SPLIT_MODE
#define TAIL_SPLIT_MODE 0
#endif
#ifndef TAIL_NC_PCT
#define TAIL_NC_PCT 50
#endif

namespace tail_split {
constexpr uint32_t mode = TAIL_SPLIT_MODE;
constexpr bool active = mode != 0;

// Does the reader (NCRISC) own tile (row r of the batch, column n of blk_cols)?
FORCE_INLINE bool reader_owns(uint32_t r, uint32_t n, uint32_t blk_cols, uint32_t ncut) {
    if constexpr (mode == 1) {
        return n < (blk_cols * TAIL_NC_PCT) / 100;
    } else if constexpr (mode == 2) {
        return (n & 1) == 0;
    } else if constexpr (mode == 3) {
        return (r & 1) == 0;
    } else if constexpr (mode == 4) {
        return n < ncut;
    } else {
        return false;
    }
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

// Writer side: publish batch `seq` (1-based) at L1 address `l1`.
FORCE_INLINE void post(uint32_t mbox_addr, uint32_t seq_addr, uint32_t l1, uint32_t seq) {
    *word(mbox_addr) = l1;
    *word(seq_addr) = seq;
}

FORCE_INLINE void wait_at_least(uint32_t addr, uint32_t v) {
    do {
        invalidate_l1_cache();
    } while (*word(addr) < v);
}

FORCE_INLINE uint32_t read(uint32_t addr) {
    invalidate_l1_cache();
    return *word(addr);
}
}  // namespace tail_split
