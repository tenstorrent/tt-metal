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
#include "api/dataflow/dataflow_api.h"

#ifndef TAIL_SPLIT_MODE
#define TAIL_SPLIT_MODE 0
#endif
#ifndef TAIL_NC_PCT
#define TAIL_NC_PCT 50
#endif
// GM_TAIL_POLICY (t2_tail_write_stream, host TAIL_POLICY): the host's per-core tail policy for the bank-local split
// with a DRAM output -- the reader's tile count of every tail row (instead of TAIL_NC_PCT) and the DRAM endpoint
// table the writer addresses (0: the NoC0 endpoints, 1: its own NoC1 endpoints) -- arrives as two runtime args right
// after the output chunk table, on both RISCs. Absent (0): TAIL_NC_PCT and each RISC's own endpoints.
#ifndef GM_TAIL_POLICY
#define GM_TAIL_POLICY 0
#endif

namespace tail_split {
constexpr uint32_t mode = TAIL_SPLIT_MODE;
constexpr bool active = mode != 0;

// The per-core tail policy (GM_TAIL_POLICY), or the defaults.
struct Policy {
    uint32_t reader_tiles;  // GM_TAIL_POLICY: the reader's tiles of every tail row (else unused)
    uint8_t wr_ep_noc;      // the DRAM endpoint table the writer's output writes address
};

// Reads the policy's two runtime args at `arg` (advanced) under GM_TAIL_POLICY; else the defaults, nothing read.
FORCE_INLINE Policy read_policy(uint32_t& arg) {
    if constexpr (GM_TAIL_POLICY != 0) {
        const uint32_t tiles = get_arg_val<uint32_t>(arg++);
        const uint32_t ep = get_arg_val<uint32_t>(arg++);
        return {tiles, static_cast<uint8_t>(ep)};
    } else {
        return {0, static_cast<uint8_t>(noc_index)};
    }
}

// Does the reader (NCRISC) own tile n of a row of blk_cols valid tiles?
FORCE_INLINE bool reader_owns(uint32_t n, uint32_t blk_cols, const Policy& p) {
    if constexpr (mode == 1) {
        if constexpr (GM_TAIL_POLICY != 0) {
            return n < p.reader_tiles;
        } else {
            return n < (blk_cols * TAIL_NC_PCT) / 100;
        }
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
