// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// embedding_gen: the unit -> block geometry shared by the reader and the writer (op_design.md, "Unit -> block
// geometry"). One definition, so both kernels walk the same blocks from the same runtime arguments of plan_blocks.

#pragma once

#include <stdint.h>

namespace embedding_gen {

struct Block {
    uint32_t b;       // batch
    uint32_t tg;      // token group inside the batch (the output tile-row when the output is tiled)
    uint32_t s0;      // first token of the block in its batch
    uint32_t n_tok;   // tokens in this block (ragged at the end of a batch)
    uint32_t c0;      // first column unit
    uint32_t n_cols;  // column units in this block (ragged in the last chunk)
};

// The shape and the blocking of one call: the common runtime arguments both kernels receive.
struct Geometry {
    uint32_t S;
    uint32_t Dt;
    uint32_t block_tokens;
    uint32_t block_cols;
    uint32_t num_col_chunks;
    uint32_t groups_per_batch;

    // unit u = (token group g, column chunk cc), g = (batch b, group tg inside the batch).
    inline Block block(uint32_t unit) const {
        const uint32_t g = unit / num_col_chunks;
        const uint32_t cc = unit - g * num_col_chunks;
        Block blk;
        blk.b = g / groups_per_batch;
        blk.tg = g - blk.b * groups_per_batch;
        blk.s0 = blk.tg * block_tokens;
        blk.n_tok = (S - blk.s0) < block_tokens ? (S - blk.s0) : block_tokens;
        blk.c0 = cc * block_cols;
        blk.n_cols = (Dt - blk.c0) < block_cols ? (Dt - blk.c0) : block_cols;
        return blk;
    }
};

}  // namespace embedding_gen
