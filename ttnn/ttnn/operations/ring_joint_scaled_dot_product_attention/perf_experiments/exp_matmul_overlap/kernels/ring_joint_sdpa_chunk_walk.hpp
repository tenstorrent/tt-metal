// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// TRANSPORT chunk walk shared by the port reader and sender (from examples/fabric_all_gather `for_each_chunk`,
// part 0 / no rotation). One part (the K or V pages of one (batch, kv head) entry) of `part_pages` pages, walked
// on this port's bank set (banks first, first+stride, ...; links split an entry by bank). A chunk is n <= run_pages
// pages b + (m + t) * B, consecutive in one DRAM bank of the source and of the landing slot (both bases are whole
// pages, so stride-B runs stay in one bank). f(page, n): page relative to the part's first page.
#pragma once

#include <cstdint>

template <typename F>
FORCE_INLINE void for_each_chunk(
    uint32_t part_pages, uint32_t first_bank, uint32_t bank_stride, uint32_t num_banks, uint32_t run_pages, F&& f) {
    const uint32_t max_per_bank = (part_pages + num_banks - 1) / num_banks;
    for (uint32_t m = 0; m < max_per_bank; m += run_pages) {
        for (uint32_t b = first_bank; b < num_banks; b += bank_stride) {
            const uint32_t per_bank = b < part_pages ? (part_pages - b + num_banks - 1) / num_banks : 0;
            if (m >= per_bank) {
                continue;
            }
            const uint32_t n = (per_bank - m) < run_pages ? (per_bank - m) : run_pages;
            f(b + m * num_banks, n);
        }
    }
}
