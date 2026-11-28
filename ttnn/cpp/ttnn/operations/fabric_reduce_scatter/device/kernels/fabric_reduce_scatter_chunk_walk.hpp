// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The chunk walk of one block for one port or final core (the generic op's _CHUNK_WALK, as every
// fabric_reduce_scatter kernel called it: no rotation, the whole bank set). The core owns banks first,
// first + stride, ... and visits them round-robin; a chunk is n <= run_pages pages b + (m + t) * B,
// consecutive in bank b. f(page, n, idx): the chunk's first page (block-local), its page count and its
// index in the walk.
template <typename F>
FORCE_INLINE void for_each_chunk(
    uint32_t shard_pages, uint32_t first_bank, uint32_t bank_stride, uint32_t num_banks, uint32_t run_pages, F&& f) {
    // num_banks is a compile-time constant at every call site, so the per-bank divisions fold to shifts.
    const uint32_t max_per_bank = (shard_pages + num_banks - 1) / num_banks;
    uint32_t idx = 0;
    for (uint32_t m = 0; m < max_per_bank; m += run_pages) {
        for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
            const uint32_t per_bank = b < shard_pages ? (shard_pages - b + num_banks - 1) / num_banks : 0;
            if (m >= per_bank) {
                continue;
            }
            const uint32_t n = (per_bank - m) < run_pages ? (per_bank - m) : run_pages;
            f(b + m * num_banks, n, idx++);
        }
    }
}
