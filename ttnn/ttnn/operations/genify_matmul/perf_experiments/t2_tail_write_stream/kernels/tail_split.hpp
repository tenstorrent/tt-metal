// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tail write split (Perf, tail_writes_both_nocs): split the LAST block's output write stream between the writer
// (BRISC / NoC1) and the reader (NCRISC / NoC0), which is idle once its K-walk is done.
// Scope: the LAST block-batch of the core's last block (Refinement 12). Its output is produced only after the
// reader's final K-block push, so the reader can always serve it; holding an EARLIER batch of a multi-batch
// block (block_batches > 1) for the reader deadlocks (output CB full -> compute stalls -> reader's input
// reserve never returns).
//
// The output CB keeps ONE consumer (the writer: cb_wait_front / cb_pop_front). Per batch of the tail the
// writer posts the batch's L1 address (mailbox) and a sequence number, issues its own share, flushes, then
// waits for the reader's `done` >= seq before popping (the reader's share has left L1). The reader, after
// its walk, waits for each post, issues its share on NoC0, flushes and publishes `done`. Both words are
// program semaphores (zero-initialized by dispatch), used as plain local L1 words.
//
// TAIL_SPLIT_MODE (define; absent = 0) / TAIL_NC_PCT:
//   0 off (baseline: the writer writes everything on NoC1)
//   1 column cut: the reader owns tiles n < blk_cols * TAIL_NC_PCT / 100 of every row, the writer the rest
#pragma once
#include <cstdint>

#ifndef TAIL_SPLIT_MODE
#define TAIL_SPLIT_MODE 0
#endif
#ifndef TAIL_NC_PCT
#define TAIL_NC_PCT 50
#endif

// t2_tail_write_stream per-core policy table (host-computed, keyed by this core's virtual NoC coords):
// TWS_TAB_Y<y> nibble x: bits 0..2 = reader tiles of every row (overrides TAIL_NC_PCT), bit 3 = the writer's DRAM
// endpoint NoC table; TWS_RTAB_Y<y> bit x = the reader's endpoint NoC table.
#ifdef TWS_TAB
namespace tws {
constexpr uint64_t tab[12] = {
    TWS_TAB_Y0,
    TWS_TAB_Y1,
    TWS_TAB_Y2,
    TWS_TAB_Y3,
    TWS_TAB_Y4,
    TWS_TAB_Y5,
    TWS_TAB_Y6,
    TWS_TAB_Y7,
    TWS_TAB_Y8,
    TWS_TAB_Y9,
    TWS_TAB_Y10,
    TWS_TAB_Y11};
constexpr uint32_t rtab[12] = {
    TWS_RTAB_Y0,
    TWS_RTAB_Y1,
    TWS_RTAB_Y2,
    TWS_RTAB_Y3,
    TWS_RTAB_Y4,
    TWS_RTAB_Y5,
    TWS_RTAB_Y6,
    TWS_RTAB_Y7,
    TWS_RTAB_Y8,
    TWS_RTAB_Y9,
    TWS_RTAB_Y10,
    TWS_RTAB_Y11};
FORCE_INLINE uint32_t nib() { return (uint32_t)(tab[my_y[0] % 12] >> (4 * my_x[0])) & 0xF; }
FORCE_INLINE uint32_t reader_tiles() { return nib() & 7; }
FORCE_INLINE uint8_t wr_ep() { return (nib() >> 3) & 1; }
FORCE_INLINE uint8_t rd_ep() { return (rtab[my_y[0] % 12] >> my_x[0]) & 1; }
}  // namespace tws
#endif

namespace tail_split {
constexpr uint32_t mode = TAIL_SPLIT_MODE;
constexpr bool active = mode != 0;

// Does the reader (NCRISC) own tile n of a row of blk_cols valid tiles?
FORCE_INLINE bool reader_owns(uint32_t n, uint32_t blk_cols) {
    if constexpr (mode == 1) {
#ifdef TWS_TAB
        return n < tws::reader_tiles();
#else
        return n < (blk_cols * TAIL_NC_PCT) / 100;
#endif
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
