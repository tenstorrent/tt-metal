// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The pad fill's ONE store primitive, shared by the reader (row-major fill of
// the input block, op_design.md §8.3) and the writer (R7's output-format
// rewrite of the pad positions inside a packed tile). Two call sites, one
// implementation — the two fills must agree byte-for-byte about what
// "pad_value" means or a padded cast would disagree with itself.

#pragma once

// Fill `bytes` of L1 at `addr` with `pad_word`.
//
// `pad_word` is the fill value already replicated across the 32-bit word by the
// host (`pad_fill_word`), so the fast path is a word-store loop. The unaligned
// head/tail exist because a region can start at the real row WIDTH, which is a
// multiple of the element size but need not be a multiple of 4 (e.g. an odd W
// at bf16, or any W at uint8). The element size always DIVIDES 4, so the word
// repeats with a period that divides 4 and byte `A` of the fill is
// `pad_word >> ((A & 3) * 8)` — the phase is carried by the address itself.
FORCE_INLINE void fill_pad_region(uint32_t addr, uint32_t bytes, uint32_t pad_word) {
    const uint32_t end = addr + bytes;
    while (addr < end && (addr & 3u) != 0u) {
        *reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr) = (pad_word >> ((addr & 3u) * 8)) & 0xFFu;
        ++addr;
    }
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const uint32_t n_words = (end - addr) >> 2;
    for (uint32_t i = 0; i < n_words; ++i) {
        words[i] = pad_word;
    }
    addr += n_words << 2;
    while (addr < end) {
        *reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr) = (pad_word >> ((addr & 3u) * 8)) & 0xFFu;
        ++addr;
    }
}
