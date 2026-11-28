// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_copy {

// Moves the pages [block_first, block_first + block_len) between DRAM and the L1 region at l1_base,
// one NoC transfer per bank run (pages p, p + num_banks, ... sit contiguously in one bank).
// A page's L1 slot follows walk order, not page order, so the reader and the writer must both walk
// through this function with the same arguments.
template <bool IsRead, typename Accessor>
FORCE_INLINE void transfer_block(
    const Accessor& accessor, uint32_t block_first, uint32_t block_len, uint32_t walk_rotation, uint32_t l1_base) {
    // The bank stride of the accessor's default page size; the CB page size matches it on the host.
    const uint32_t page_bytes = accessor.get_aligned_page_size();
    // A run must stop at this block's end: bank contiguity continues into other blocks and other cores.
    const uint32_t block_end = block_first + block_len;
    const uint32_t num_banks = accessor.contiguous_page_stride();
    const uint32_t num_runs = block_len < num_banks ? block_len : num_banks;

    uint32_t k = walk_rotation % num_runs;
    uint32_t l1_addr = l1_base;
    for (uint32_t i = 0; i < num_runs; ++i) {
        const uint32_t page = block_first + k;
        const uint32_t run_bytes = accessor.num_contiguous_pages(page, block_end) * page_bytes;
        if constexpr (IsRead) {
            noc_async_read(accessor.get_noc_addr(page), l1_addr, run_bytes);
        } else {
            noc_async_write(l1_addr, accessor.get_noc_addr(page), run_bytes);
        }
        l1_addr += run_bytes;
        if (++k == num_runs) {
            k = 0;
        }
    }
}

// Bank-major enumeration (BANK_AFFINITY): position j walks every page of bank 0 in bank order, then
// bank 1, ... With N pages over B banks, bank b holds n_b = q + (b < r) pages (q = N / B, r = N % B),
// so a core's contiguous j range sits in one bank (two at a boundary) and each block is one long
// run per bank touched instead of one short run per bank. Moves positions [j_first, j_first + len)
// to / from the L1 region at l1_base in j order.
template <bool IsRead, typename Accessor>
FORCE_INLINE void transfer_block_bank_major(
    const Accessor& accessor, uint32_t j_first, uint32_t len, uint32_t tensor_pages, uint32_t l1_base) {
    const uint32_t page_bytes = accessor.get_aligned_page_size();
    const uint32_t num_banks = accessor.contiguous_page_stride();
    const uint32_t q = tensor_pages / num_banks;
    const uint32_t r = tensor_pages % num_banks;
    const uint32_t big_banks_end = r * (q + 1);  // positions held by the r banks with q + 1 pages

    const uint32_t j_end = j_first + len;
    uint32_t j = j_first;
    uint32_t l1_addr = l1_base;
    while (j < j_end) {
        uint32_t bank, slot, bank_pages;
        if (j < big_banks_end) {
            bank = j / (q + 1);
            slot = j - bank * (q + 1);
            bank_pages = q + 1;
        } else {
            const uint32_t rest = j - big_banks_end;
            bank = r + rest / q;
            slot = rest - (bank - r) * q;
            bank_pages = q;
        }
        const uint32_t want = (j_end - j) < (bank_pages - slot) ? (j_end - j) : (bank_pages - slot);
        const uint32_t page = slot * num_banks + bank;
        // == want when bank pages are contiguous; 1 when the accessor says they are not.
        const uint32_t run = accessor.num_contiguous_pages(page, page + (want - 1) * num_banks + 1);
        const uint32_t run_bytes = run * page_bytes;
        if constexpr (IsRead) {
            noc_async_read(accessor.get_noc_addr(page), l1_addr, run_bytes);
        } else {
            noc_async_write(l1_addr, accessor.get_noc_addr(page), run_bytes);
        }
        l1_addr += run_bytes;
        j += run;
    }
}

// The one walk both kernels call: same arguments on both sides keep a page's L1 slot identical.
template <bool IsRead, typename Accessor>
FORCE_INLINE void walk_block(
    const Accessor& accessor,
    uint32_t block_first,
    uint32_t block_len,
    uint32_t walk_rotation,
    uint32_t bank_major,
    uint32_t tensor_pages,
    uint32_t l1_base) {
    if (bank_major) {
        transfer_block_bank_major<IsRead>(accessor, block_first, block_len, tensor_pages, l1_base);
    } else {
        transfer_block<IsRead>(accessor, block_first, block_len, walk_rotation, l1_base);
    }
}

}  // namespace g6_copy
