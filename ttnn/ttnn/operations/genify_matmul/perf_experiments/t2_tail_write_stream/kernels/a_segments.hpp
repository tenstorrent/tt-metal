// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A as a list of K segments (Refinement 14): A = concat(A_0, ..., A_{s-1}, dim=-1) is never materialized.
// Every segment is an interleaved (..., M, K_i) tensor with the same leading dims, dtype and memory config,
// so one TensorAccessorArgs serves all of them -- only the base address and the row pitch K_i differ.
// Global K tile kt lands in segment s (cum_s <= kt < cum_s + kt_s) at local tile kt - cum_s; tile row `trow`
// of segment s starts at page trow * kt_s.
//
// The segment table: segment 0's address is the kernel's existing A-address arg; with GM_A_SEGMENTS = s > 1
// the RT args carry [kt_0, .., kt_{s-1}, addr_1, .., addr_{s-1}]. A K-block row that straddles a segment
// boundary is read as one run per segment (the segment is located once per K-block, not per tile).
// GM_A_SEGMENTS absent / 1: callers keep their single-tensor path byte-identical (see `num`).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#ifndef GM_A_SEGMENTS
#define GM_A_SEGMENTS 1
#endif

namespace a_segments {
constexpr uint32_t num = GM_A_SEGMENTS;
static_assert(num >= 1);

struct Table {
    uint32_t kt[num];    // K tiles per segment (the segment's row pitch in pages)
    uint32_t addr[num];  // per-segment buffer base address
};

// Segment 0's address is `addr0`; with one segment its width is the whole K (`k_tiles`). Otherwise the widths
// and segments 1 .. num-1's addresses are read from RT args starting at `arg` (advanced).
FORCE_INLINE Table read_table(uint32_t addr0, uint32_t k_tiles, uint32_t& arg) {
    Table t{};
    t.addr[0] = addr0;
    t.kt[0] = k_tiles;
    if constexpr (num > 1) {
        for (uint32_t i = 0; i < num; ++i) {
            t.kt[i] = get_arg_val<uint32_t>(arg++);
        }
        for (uint32_t i = 1; i < num; ++i) {
            t.addr[i] = get_arg_val<uint32_t>(arg++);
        }
    }
    return t;
}

// Where global K tile `gk` lands: segment index and K tile inside it.
struct Pos {
    uint32_t seg;
    uint32_t local;
};

FORCE_INLINE Pos locate(const Table& t, uint32_t gk) {
    uint32_t s = 0;
    if constexpr (num > 1) {
        while (s + 1 < num && gk >= t.kt[s]) {
            gk -= t.kt[s];
            ++s;
        }
    }
    return {s, gk};
}

// Read `k_valid` consecutive K tiles of A tile row `trow`, starting at `start` (from `locate`), into L1 tiles
// packed `tile_bytes` apart at `l1`: one run per segment the row segment touches.
template <typename Args>
FORCE_INLINE void read_row(
    const Args& args,
    const Table& t,
    uint32_t page_bytes,
    uint32_t tile_bytes,
    uint32_t l1,
    uint32_t trow,
    Pos start,
    uint32_t k_valid) {
    uint32_t s = start.seg;
    uint32_t lk = start.local;
    uint32_t k = 0;
    while (k < k_valid) {
        const auto acc = TensorAccessor(args, t.addr[s], page_bytes);
        uint32_t run = k_valid - k;
        if (run > t.kt[s] - lk) {
            run = t.kt[s] - lk;
        }
        const uint32_t page = trow * t.kt[s] + lk;
        for (uint32_t j = 0; j < run; ++j) {
            noc_async_read(acc.get_noc_addr(page + j), l1, tile_bytes);
            l1 += tile_bytes;
        }
        k += run;
        ++s;
        lk = 0;
    }
}
}  // namespace a_segments
