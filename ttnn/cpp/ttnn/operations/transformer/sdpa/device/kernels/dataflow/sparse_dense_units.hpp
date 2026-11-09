// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dense rows split across devices by key shard (dense_blocks). Every device computes every dense unit (owner shard o,
// head h, dense Q block b of o) against its own K/V shard. Units are head-major: u = h x n_dense_blocks + the block's
// place in the shard-ordered list of every owner's dense blocks; a core holds units of one head, so the cores of a head
// can share its local K/V pieces by group multicast; the raw partial states move toward the owner one ring hop
// at a time, are merged on the way, and the owner normalizes them into its output. The reader, compute and writer
// walk the same batches (dense_for_each_batch) and the program factory builds the same records.

#pragma once

#include <cstdint>

// With dense units compute's ring-iteration body and the writer's output block write run from two call sites (the ring
// loop and the dense batches): keep one copy of each. Without them (SPARSE_DENSE_UNITS undefined) nothing changes.
#if defined(SPARSE_DENSE_UNITS)
#define SPARSE_DENSE_ONE_COPY __attribute__((noinline))
#else
#define SPARSE_DENSE_ONE_COPY
#endif

namespace sparse_dense {

// Tiles of one partial in the emit / receive order: out (Sq x vDHt), per-column sums (Sq), row max (Sq). bf16.
template <uint32_t Sq_chunk_t, uint32_t vDHt>
constexpr uint32_t partial_tiles() {
    return Sq_chunk_t * vDHt + 2 * Sq_chunk_t;
}

// Hop buffers (staging, receive): unit u's partial_tiles pages are consecutive pages of one DRAM bank (page p is in
// bank p % n_banks), so one read and few fabric packets move a whole unit. Units are padded to a multiple of n_banks.
inline uint32_t banked_page(uint32_t u, uint32_t t, uint32_t partial_tiles, uint32_t n_banks) {
    return ((u / n_banks) * partial_tiles + t) * n_banks + u % n_banks;
}

// Per owner shard, kRecordWords runtime words.
constexpr uint32_t kRecordWords = 8;
// role: what this device does with the owner's units after its own partial (and any merges).
constexpr uint32_t kRoleOwner = 0;     // normalize into the output
constexpr uint32_t kRoleEmitPlus = 1;  // emit, sent to the tensor-rank + 1 neighbour
constexpr uint32_t kRoleEmitMinus = 2;
struct Owner {
    uint32_t shard;           // o
    uint32_t n_blocks;        // dense Q blocks of o
    uint32_t block_offset;    // o's first block in the list of every owner's dense blocks (unit u: u % n_dense_blocks)
    uint32_t slab_out_row;    // o's slab in the dense-Q buffer (tile row)
    uint32_t slab_row_start;  // local Q tile row of o's slab start
    uint32_t slab_rows;       // o's slab rows (tiles)
    uint32_t up_sides;        // bit 0: a partial arrives from the - neighbour, bit 1: from the + neighbour
    uint32_t role;
    uint32_t n_up() const { return (up_sides & 1u) + ((up_sides >> 1) & 1u); }
};

template <typename ReadArg>
Owner read_owner(uint32_t base, uint32_t k, ReadArg read_arg) {
    const uint32_t a = base + k * kRecordWords;
    return Owner{
        read_arg(a + 0),
        read_arg(a + 1),
        read_arg(a + 2),
        read_arg(a + 3),
        read_arg(a + 4),
        read_arg(a + 5),
        read_arg(a + 6),
        read_arg(a + 7)};
}

// Local Q block of the j-th block of shard s inside a range (in_dense(global block)).
template <typename InDense>
uint32_t dense_block(uint32_t s, uint32_t j, uint32_t nb, uint32_t nb_pad, InDense in_dense) {
    for (uint32_t b = 0; b < nb; ++b) {
        if (in_dense(s * nb_pad + b)) {
            if (j == 0) {
                return b;
            }
            --j;
        }
    }
    return 0;
}

// Local Q block of the c-th block of shard s outside every range: the resident blocks when this shard's dense rows
// are split across the devices.
template <typename InDense>
uint32_t resident_block(uint32_t s, uint32_t c, uint32_t nb, uint32_t nb_pad, InDense in_dense) {
    for (uint32_t b = 0; b < nb; ++b) {
        if (!in_dense(s * nb_pad + b)) {
            if (c == 0) {
                return b;
            }
            --c;
        }
    }
    return 0;
}

// The core's units [du_start, du_end), all of one head (the factory never splits a core across heads), in batches of
// at most `slots`, never across owners: fn(k, owner, first unit, end unit, first batch of the owner, last batch of the
// owner). Unit u is block u % n_dense_blocks - owner.block_offset of the owner, head u / n_dense_blocks.
template <typename ReadArg, typename Fn>
void for_each_batch(
    uint32_t n_owners,
    uint32_t records_base,
    uint32_t du_start,
    uint32_t du_end,
    uint32_t n_dense_blocks,
    uint32_t slots,
    ReadArg read_arg,
    Fn fn) {
    const uint32_t base = du_start - du_start % n_dense_blocks;
    for (uint32_t k = 0; k < n_owners; ++k) {
        const Owner o = read_owner(records_base, k, read_arg);
        const uint32_t start = base + o.block_offset;
        const uint32_t lo = du_start > start ? du_start : start;
        const uint32_t hi = du_end < start + o.n_blocks ? du_end : start + o.n_blocks;
        for (uint32_t u0 = lo; u0 < hi; u0 += slots) {
            const uint32_t u1 = hi - u0 < slots ? hi : u0 + slots;
            fn(k, o, u0, u1, u0 == lo, u1 == hi);
        }
    }
}

}  // namespace sparse_dense
