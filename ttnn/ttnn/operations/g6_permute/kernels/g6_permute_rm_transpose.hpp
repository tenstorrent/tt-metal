// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

// Core-local transpose of ROW_MAJOR bfloat16 pages for the blocks_rm_gather writer. RISC-V loads and
// stores only: the values never pass through an unpacker or packer, so every bit pattern survives. Plain (non-volatile)
// accesses: the caller fences them against the CB handshake before and the NoC writes after. Raw dataflow: kernel_lib
// has no block helper that transposes halfwords between strided L1 rows.
namespace g6_permute {

// One word of the page pairs (2q, 2q + 1), page 2q's word at src + 2q * src_page_stride, becomes word q
// of two rows, the even one at `row`. Every load is issued before the first store, so the L1 load
// latencies overlap (kernels build at -Os, which does not unroll).
template <uint32_t src_page_stride, uint32_t dst_row_stride, uint32_t... q>
FORCE_INLINE void transpose_word_pairs(uint32_t src, uint32_t row, std::integer_sequence<uint32_t, q...>) {
    const uint32_t a[] = {*reinterpret_cast<const tt_l1_ptr uint32_t*>(src + 2 * q * src_page_stride)...};
    const uint32_t b[] = {*reinterpret_cast<const tt_l1_ptr uint32_t*>(src + (2 * q + 1) * src_page_stride)...};
    ((*reinterpret_cast<tt_l1_ptr uint32_t*>(row + q * sizeof(uint32_t)) = (a[q] & 0xFFFFu) | (b[q] << 16)), ...);
    ((*reinterpret_cast<tt_l1_ptr uint32_t*>(row + dst_row_stride + q * sizeof(uint32_t)) =
          (a[q] >> 16) | (b[q] & 0xFFFF0000u)),
     ...);
}

// `pages` pages of `values` values, page p at src + p * src_page_stride, become `values` rows of
// `pages` values, row v at dst + v * dst_row_stride: value v of page p lands at position p of row v.
// Even `values`: word m of pages p and p + 1 holds values 2m (low half) and 2m + 1 of each, which
// make one word of rows 2m and 2m + 1, so the L1 accesses are half the halfword walk's. Pages walk
// outer so each page's words are loaded in address order (data-cache lines); `pair_batch` page pairs
// are walked together so their loads overlap. An odd `pages` leaves one halfword per row.
template <uint32_t pages, uint32_t values, uint32_t src_page_stride, uint32_t dst_row_stride, uint32_t pair_batch = 1>
FORCE_INLINE void transpose_pages(uint32_t src, uint32_t dst) {
    static_assert(src_page_stride % sizeof(uint32_t) == 0 && dst_row_stride % sizeof(uint32_t) == 0);
    static_assert(dst_row_stride >= pages * sizeof(uint16_t), "a row holds one value per page");
    static_assert(pair_batch >= 1);
    if constexpr (values % 2 == 0) {
        uint32_t p2 = 0;
        if constexpr (pair_batch > 1) {
            for (; p2 + pair_batch <= pages / 2; p2 += pair_batch) {
                uint32_t word = src + 2 * p2 * src_page_stride;
                uint32_t row = dst + p2 * sizeof(uint32_t);
                for (uint32_t m = 0; m < values / 2; ++m) {
                    transpose_word_pairs<src_page_stride, dst_row_stride>(
                        word, row, std::make_integer_sequence<uint32_t, pair_batch>{});
                    word += sizeof(uint32_t);
                    row += 2 * dst_row_stride;
                }
            }
        }
        for (; p2 < pages / 2; ++p2) {
            const auto* page_a = reinterpret_cast<const tt_l1_ptr uint32_t*>(src + 2 * p2 * src_page_stride);
            const auto* page_b = reinterpret_cast<const tt_l1_ptr uint32_t*>(src + (2 * p2 + 1) * src_page_stride);
            uint32_t row = dst + p2 * sizeof(uint32_t);
            for (uint32_t m = 0; m < values / 2; ++m) {
                const uint32_t a = page_a[m];
                const uint32_t b = page_b[m];
                *reinterpret_cast<tt_l1_ptr uint32_t*>(row) = (a & 0xFFFFu) | (b << 16);
                *reinterpret_cast<tt_l1_ptr uint32_t*>(row + dst_row_stride) = (a >> 16) | (b & 0xFFFF0000u);
                row += 2 * dst_row_stride;
            }
        }
        if constexpr (pages % 2 != 0) {
            const auto* page_a = reinterpret_cast<const tt_l1_ptr uint32_t*>(src + (pages - 1) * src_page_stride);
            uint32_t row = dst + (pages - 1) * sizeof(uint16_t);
            for (uint32_t m = 0; m < values / 2; ++m) {
                const uint32_t a = page_a[m];
                *reinterpret_cast<tt_l1_ptr uint16_t*>(row) = a & 0xFFFFu;
                *reinterpret_cast<tt_l1_ptr uint16_t*>(row + dst_row_stride) = a >> 16;
                row += 2 * dst_row_stride;
            }
        }
    } else {
        for (uint32_t v = 0; v < values; ++v) {
            const auto* in = reinterpret_cast<const tt_l1_ptr uint16_t*>(src) + v;
            auto* out = reinterpret_cast<tt_l1_ptr uint16_t*>(dst + v * dst_row_stride);
            for (uint32_t p = 0; p < pages; ++p) {
                out[p] = in[p * (src_page_stride / sizeof(uint16_t))];
            }
        }
    }
}

}  // namespace g6_permute
