// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block split of a core's m rows, shared by reader, compute and writer: blocks 0..num_full-1 have full_rows
// (block_h) rows, blocks num_full..num_blocks-1 split the rest evenly, larger first. The host picks num_full.

#pragma once

#include <cstdint>

struct RowBlocks {
    uint32_t num_blocks;
    uint32_t num_full;
    uint32_t full_rows;
    uint32_t rest_q;
    uint32_t rest_r;

    RowBlocks(uint32_t core_rows, uint32_t num_blocks_, uint32_t num_full_, uint32_t full_rows_) :
        num_blocks(num_blocks_), num_full(num_full_), full_rows(full_rows_), rest_q(0), rest_r(0) {
        if (num_blocks > num_full) {
            const uint32_t rest = core_rows - num_full * full_rows;
            rest_q = rest / (num_blocks - num_full);
            rest_r = rest % (num_blocks - num_full);
        }
    }
    uint32_t rows(uint32_t idx) const {
        return idx < num_full ? full_rows : rest_q + (idx - num_full < rest_r ? 1 : 0);
    }
    uint32_t start(uint32_t idx) const {
        if (idx <= num_full) {
            return idx * full_rows;
        }
        const uint32_t j = idx - num_full;
        return num_full * full_rows + j * rest_q + (j < rest_r ? j : rest_r);
    }
    // Subblock class of block idx: 0 = full, 1 = rest_q + 1 rows, 2 = rest_q rows.
    uint32_t kind(uint32_t idx) const { return idx < num_full ? 0 : (rows(idx) == rest_q + 1 ? 1 : 2); }
};
