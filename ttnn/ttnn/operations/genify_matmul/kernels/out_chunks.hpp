// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Split output (Refinement 13, `chunks`): the logical output (..., M, N_out) is written as GM_OUT_CHUNKS
// separate interleaved tensors of equal width `row_tiles` = N_out / GM_OUT_CHUNKS tiles; output tile column
// gc lands in chunk gc / row_tiles at local column gc % row_tiles. Every chunk shares one memory config,
// dtype and shape, so one TensorAccessorArgs serves all of them -- only the base address differs (the chunk
// table: chunk 0's address is the kernel's existing output-address arg, chunks 1.. follow in the RT args).
//
// The tiles a core computes are unchanged; only the store address changes. A row segment of a block that
// straddles a chunk boundary is written as one run per chunk (the chunk is located once per block, not per
// tile). GM_OUT_CHUNKS absent / 1: one chunk of width N_out -- byte-identical page arithmetic to the
// unsplit output.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#ifndef GM_OUT_CHUNKS
#define GM_OUT_CHUNKS 1
#endif

namespace out_chunks {
constexpr uint32_t num = GM_OUT_CHUNKS;
static_assert(num >= 1);

struct Table {
    uint32_t row_tiles;  // output tiles per tile row of ONE chunk (N_out tiles when num == 1)
    uint32_t addr[num];  // per-chunk buffer base address
};

// Chunk 0's address is `addr0`; chunks 1 .. num-1 are read from RT args starting at `arg` (advanced).
FORCE_INLINE Table read_table(uint32_t addr0, uint32_t row_tiles, uint32_t& arg) {
    Table t{};
    t.row_tiles = row_tiles;
    t.addr[0] = addr0;
    for (uint32_t i = 1; i < num; ++i) {
        t.addr[i] = get_arg_val<uint32_t>(arg++);
    }
    return t;
}

// Where a block's first output column lands: chunk index and column inside it.
struct Col {
    uint32_t chunk;
    uint32_t local;
};

FORCE_INLINE Col locate(const Table& t, uint32_t gcol) {
    if constexpr (num == 1) {
        return {0, gcol};
    } else {
        const uint32_t c = gcol / t.row_tiles;
        return {c, gcol - c * t.row_tiles};
    }
}

// Write `cols` consecutive output tiles of tile row `trow` (batch * tensor_m_tiles + row), starting at block
// column `start` (from `locate`), from L1 tiles packed `tile_bytes` apart at `l1`. `owns(n)` selects the tiles
// this RISC writes (n = 0 .. cols-1, the block-relative column); the others are skipped.
template <typename Args, typename Owns>
FORCE_INLINE void write_row(
    const Args& args,
    const Table& t,
    uint32_t page_bytes,
    uint32_t tile_bytes,
    uint32_t l1,
    uint32_t trow,
    Col start,
    uint32_t cols,
    Owns owns) {
    uint32_t c = start.chunk;
    uint32_t lc = start.local;
    uint32_t n = 0;
    while (n < cols) {
        const auto acc = TensorAccessor(args, t.addr[c], page_bytes);
        uint32_t seg = cols - n;
        if constexpr (num > 1) {
            if (seg > t.row_tiles - lc) {
                seg = t.row_tiles - lc;
            }
        }
        const uint32_t page = trow * t.row_tiles + lc;
        for (uint32_t j = 0; j < seg; ++j) {
            if (owns(n + j)) {
                noc_async_write(l1, acc.get_noc_addr(page + j), tile_bytes);
            }
            l1 += tile_bytes;
        }
        n += seg;
        ++c;
        lc = 0;
    }
}
}  // namespace out_chunks
