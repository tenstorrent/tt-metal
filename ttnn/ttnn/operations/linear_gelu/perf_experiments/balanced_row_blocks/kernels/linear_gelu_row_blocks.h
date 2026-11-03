// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row-block split of a core's m rows, shared by reader, compute and writer. Block 0 has first_rows (the host keeps
// it at block_h so its compute hides the W delivery); blocks 1..num_blocks-1 split the rest evenly, larger first,
// so no 1-row tail block runs 1x1 subblocks.

#pragma once

#include <cstdint>

struct RowBlocks {
    uint32_t num_blocks;
    uint32_t first_rows;
    uint32_t rest_q;
    uint32_t rest_r;

    RowBlocks(uint32_t core_rows, uint32_t num_blocks_, uint32_t first_rows_) :
        num_blocks(num_blocks_), first_rows(first_rows_), rest_q(0), rest_r(0) {
        if (num_blocks > 1) {
            const uint32_t rest = core_rows - first_rows;
            rest_q = rest / (num_blocks - 1);
            rest_r = rest % (num_blocks - 1);
        }
    }
    uint32_t rows(uint32_t idx) const { return idx == 0 ? first_rows : rest_q + (idx - 1 < rest_r ? 1 : 0); }
    uint32_t start(uint32_t idx) const {
        if (idx == 0) {
            return 0;
        }
        const uint32_t j = idx - 1;
        return first_rows + j * rest_q + (j < rest_r ? j : rest_r);
    }
};
