// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

// riscv_transpose bake-off: core-local transposes of ROW_MAJOR bfloat16 pages on plain RISC-V loads
// and stores (bit-exact by construction: no unpacker / packer / FPU on the path). Usable on BRISC,
// NCRISC and the three TRISCs alike (no dataflow or LLK calls).
//
// `pages` pages of `values` bf16 values, page p at src + p * src_stride, become `values` rows, row v at
// dst + v * dst_stride, value v of page p at position p of row v. Every function transposes the page
// range [p_begin, p_end) of the unit, so several RISCs can share one unit (each writes its own column
// band of every row).
//
// Raw RISC-V (no helper): kernel_lib has no block helper that transposes halfwords between strided L1
// rows; the mechanism exploited is the RISC-V load/store schedule itself.
//
// Implementations:
//   IMPL_BASE  word pairs of two pages -> one word of two rows (the op's transpose_pages), pair_batch
//              page pairs walked together. 2 loads + 2 stores + 6 ALU per 4 values.
//   IMPL_LINE  word pairs, one page pair at a time, `batch` words (cache lines of 4 words) of each page
//              loaded before the first store: loads hit the 4 x 16 B L1 data cache.
//   IMPL_HALF  one page at a time: `batch` words loaded, each word stored as two halfwords (low half to
//              row 2m, high half to row 2m + 1). 1 load + 2 stores + 1 ALU per 2 values; any C, any
//              page boundary (odd C: the last value by one halfword load).
namespace rvt {

constexpr uint32_t IMPL_BASE = 0;
constexpr uint32_t IMPL_LINE = 1;
constexpr uint32_t IMPL_HALF = 2;
// IMPL_BLK: batch = (PB << 8) | WB, see blk_tile.
constexpr uint32_t IMPL_BLK = 3;
// IMPL_BLKV: BLK with volatile loads and stores (program-order memory accesses); IMPL_BLKS: volatile
// stores only.
constexpr uint32_t IMPL_BLKV = 4;
constexpr uint32_t IMPL_BLKS = 5;

template <typename T>
FORCE_INLINE T ld(uint32_t addr) {
    return *reinterpret_cast<const tt_l1_ptr T*>(addr);
}
template <typename T>
FORCE_INLINE void st(uint32_t addr, T v) {
    *reinterpret_cast<tt_l1_ptr T*>(addr) = v;
}

// Volatile accessors: the compiler keeps volatile accesses in program order, so a tile's loads walk each
// page's words in address order and its stores fill each row's 16 B line with consecutive stores.
template <typename T, bool vol>
FORCE_INLINE T ldv(uint32_t addr) {
    if constexpr (vol) {
        return *reinterpret_cast<const volatile tt_l1_ptr T*>(addr);
    } else {
        return *reinterpret_cast<const tt_l1_ptr T*>(addr);
    }
}
template <typename T, bool vol>
FORCE_INLINE void stv(uint32_t addr, T v) {
    if constexpr (vol) {
        *reinterpret_cast<volatile tt_l1_ptr T*>(addr) = v;
    } else {
        *reinterpret_cast<tt_l1_ptr T*>(addr) = v;
    }
}

// ---------------------------------------------------------------------------------------------- BASE
template <uint32_t src_stride, uint32_t dst_stride, uint32_t... q>
FORCE_INLINE void base_word_pairs(uint32_t src, uint32_t row, std::integer_sequence<uint32_t, q...>) {
    const uint32_t a[] = {ld<uint32_t>(src + 2 * q * src_stride)...};
    const uint32_t b[] = {ld<uint32_t>(src + (2 * q + 1) * src_stride)...};
    (st<uint32_t>(row + q * 4, (a[q] & 0xFFFFu) | (b[q] << 16)), ...);
    (st<uint32_t>(row + dst_stride + q * 4, (a[q] >> 16) | (b[q] & 0xFFFF0000u)), ...);
}

// Page pairs [q_begin, q_end) (pairs of the even-C word walk). An odd last page is handled by the
// caller of the whole unit (base_unit).
template <uint32_t values, uint32_t src_stride, uint32_t dst_stride, uint32_t pair_batch>
FORCE_INLINE void base_pairs(uint32_t src, uint32_t dst, uint32_t q_begin, uint32_t q_end) {
    static_assert(values % 2 == 0);
    uint32_t q = q_begin;
    if constexpr (pair_batch > 1) {
        for (; q + pair_batch <= q_end; q += pair_batch) {
            uint32_t word = src + 2 * q * src_stride;
            uint32_t row = dst + q * 4;
            for (uint32_t m = 0; m < values / 2; ++m) {
                base_word_pairs<src_stride, dst_stride>(word, row, std::make_integer_sequence<uint32_t, pair_batch>{});
                word += 4;
                row += 2 * dst_stride;
            }
        }
    }
    for (; q < q_end; ++q) {
        const auto* page_a = reinterpret_cast<const tt_l1_ptr uint32_t*>(src + 2 * q * src_stride);
        const auto* page_b = reinterpret_cast<const tt_l1_ptr uint32_t*>(src + (2 * q + 1) * src_stride);
        uint32_t row = dst + q * 4;
        for (uint32_t m = 0; m < values / 2; ++m) {
            const uint32_t a = page_a[m];
            const uint32_t b = page_b[m];
            st<uint32_t>(row, (a & 0xFFFFu) | (b << 16));
            st<uint32_t>(row + dst_stride, (a >> 16) | (b & 0xFFFF0000u));
            row += 2 * dst_stride;
        }
    }
}

// ---------------------------------------------------------------------------------------------- LINE
template <uint32_t dst_stride, uint32_t... k>
FORCE_INLINE void line_words(uint32_t pa, uint32_t pb, uint32_t row, std::integer_sequence<uint32_t, k...>) {
    const uint32_t a[] = {ld<uint32_t>(pa + 4 * k)...};
    const uint32_t b[] = {ld<uint32_t>(pb + 4 * k)...};
    ((st<uint32_t>(row + 2 * k * dst_stride, (a[k] & 0xFFFFu) | (b[k] << 16)),
      st<uint32_t>(row + (2 * k + 1) * dst_stride, (a[k] >> 16) | (b[k] & 0xFFFF0000u))),
     ...);
}

template <uint32_t values, uint32_t src_stride, uint32_t dst_stride, uint32_t batch>
FORCE_INLINE void line_pairs(uint32_t src, uint32_t dst, uint32_t q_begin, uint32_t q_end) {
    static_assert(values % 2 == 0);
    constexpr uint32_t words = values / 2;
    constexpr uint32_t full = words / batch * batch;
    uint32_t pa = src + 2 * q_begin * src_stride;
    uint32_t row = dst + q_begin * 4;
    for (uint32_t q = q_begin; q < q_end; ++q) {
        uint32_t w = pa;
        uint32_t r = row;
        for (uint32_t m = 0; m < full; m += batch) {
            line_words<dst_stride>(w, w + src_stride, r, std::make_integer_sequence<uint32_t, batch>{});
            w += 4 * batch;
            r += 2 * batch * dst_stride;
        }
        if constexpr (words % batch != 0) {
            line_words<dst_stride>(w, w + src_stride, r, std::make_integer_sequence<uint32_t, words % batch>{});
        }
        pa += 2 * src_stride;
        row += 4;
    }
}

// ---------------------------------------------------------------------------------------------- HALF
template <uint32_t dst_stride, uint32_t... k>
FORCE_INLINE void half_words(uint32_t page, uint32_t col, std::integer_sequence<uint32_t, k...>) {
    const uint32_t w[] = {ld<uint32_t>(page + 4 * k)...};
    ((st<uint16_t>(col + 2 * k * dst_stride, static_cast<uint16_t>(w[k])),
      st<uint16_t>(col + (2 * k + 1) * dst_stride, static_cast<uint16_t>(w[k] >> 16))),
     ...);
}

template <uint32_t values, uint32_t src_stride, uint32_t dst_stride, uint32_t batch>
FORCE_INLINE void half_pages(uint32_t src, uint32_t dst, uint32_t p_begin, uint32_t p_end) {
    constexpr uint32_t words = values / 2;
    constexpr uint32_t full = words / batch * batch;
    uint32_t page = src + p_begin * src_stride;
    uint32_t col = dst + p_begin * 2;
    for (uint32_t p = p_begin; p < p_end; ++p) {
        uint32_t w = page;
        uint32_t r = col;
        for (uint32_t m = 0; m < full; m += batch) {
            half_words<dst_stride>(w, r, std::make_integer_sequence<uint32_t, batch>{});
            w += 4 * batch;
            r += 2 * batch * dst_stride;
        }
        if constexpr (words % batch != 0) {
            half_words<dst_stride>(w, r, std::make_integer_sequence<uint32_t, words % batch>{});
        }
        if constexpr (values % 2 != 0) {
            st<uint16_t>(col + (values - 1) * dst_stride, ld<uint16_t>(page + 2 * (values - 1)));
        }
        page += src_stride;
        col += 2;
    }
}

// Odd-C fallback of the word-pair impls: the op's halfword walk over pages [p_begin, p_end).
template <uint32_t values, uint32_t src_stride, uint32_t dst_stride>
FORCE_INLINE void halfword_walk(uint32_t src, uint32_t dst, uint32_t p_begin, uint32_t p_end) {
    for (uint32_t v = 0; v < values; ++v) {
        const auto* in = reinterpret_cast<const tt_l1_ptr uint16_t*>(src) + v;
        auto* out = reinterpret_cast<tt_l1_ptr uint16_t*>(dst + v * dst_stride);
        for (uint32_t p = p_begin; p < p_end; ++p) {
            out[p] = in[p * (src_stride / 2)];
        }
    }
}

// ----------------------------------------------------------------------------------------------- BLK
// Register-blocked tile: PB pages (even) x WB consecutive words of each page are loaded page by page
// (each page's words in address order, so a 16 B data-cache line serves 4 loads), then the 2 * WB rows
// they feed are stored row by row, PB / 2 consecutive words each (consecutive stores to one 16 B line,
// which the store path takes much cheaper than scattered stores).
template <uint32_t src_stride, uint32_t dst_stride, uint32_t PB, uint32_t WB, bool vld, bool vst, uint32_t... i>
FORCE_INLINE void blk_tile(uint32_t src, uint32_t dst, std::integer_sequence<uint32_t, i...>) {
    // i walks loads page-major: page i / WB, word i % WB.
    const uint32_t a[] = {ldv<uint32_t, vld>(src + (i / WB) * src_stride + 4 * (i % WB))...};
    // i walks stores row-major: row r = i / (PB / 2) (word k = r / 2 of the pages, high half when r odd),
    // word j = i % (PB / 2) of the row, from pages 2j and 2j + 1.
    constexpr uint32_t J = PB / 2;
    (stv<uint32_t, vst>(
         dst + (i / J) * dst_stride + 4 * (i % J),
         ((i / J) % 2 == 0)
             ? ((a[(2 * (i % J)) * WB + (i / J) / 2] & 0xFFFFu) | (a[(2 * (i % J) + 1) * WB + (i / J) / 2] << 16))
             : ((a[(2 * (i % J)) * WB + (i / J) / 2] >> 16) | (a[(2 * (i % J) + 1) * WB + (i / J) / 2] & 0xFFFF0000u))),
     ...);
}

// Page groups of PB pages from page-pair q_begin up to q_end (the pairs left over below a whole group go
// through base_pairs).
template <uint32_t values, uint32_t src_stride, uint32_t dst_stride, uint32_t PB, uint32_t WB, bool vld, bool vst>
FORCE_INLINE void blk_pairs(uint32_t src, uint32_t dst, uint32_t q_begin, uint32_t q_end) {
    static_assert(values % 2 == 0 && PB % 2 == 0 && PB * WB <= 2 * PB * WB);
    constexpr uint32_t words = values / 2;
    constexpr uint32_t full = words / WB * WB;
    constexpr uint32_t QB = PB / 2;
    uint32_t q = q_begin;
    for (; q + QB <= q_end; q += QB) {
        uint32_t page = src + 2 * q * src_stride;
        uint32_t row = dst + 4 * q;
        for (uint32_t m = 0; m < full; m += WB) {
            blk_tile<src_stride, dst_stride, PB, WB, vld, vst>(
                page, row, std::make_integer_sequence<uint32_t, PB * WB>{});
            page += 4 * WB;
            row += 2 * WB * dst_stride;
        }
        if constexpr (words % WB != 0) {
            blk_tile<src_stride, dst_stride, PB, words % WB, vld, vst>(
                page, row, std::make_integer_sequence<uint32_t, PB*(words % WB)>{});
        }
    }
    if (q < q_end) {
        base_pairs<values, src_stride, dst_stride, 1>(src, dst, q, q_end);
    }
}

// ---------------------------------------------------------------------------------------------- ROWS
// Row-range split for several RISCs on one unit: word row m (rows 2m and 2m + 1, values 2m and 2m + 1 of
// every page) is one work item; an odd C adds item m = C / 2 (row C - 1, one halfword per page). A RISC
// transposes items [m_lo, m_hi) of the unit over all pages: whole groups of PB pages through blk_tile
// (PB pages x 1 word, program-order volatile accesses, each row written in PB / 2 consecutive words),
// the pairs left over in one smaller tile, an odd last page by halfwords.
template <uint32_t... j>
FORCE_INLINE void half_row_store(uint32_t row, const uint32_t* h, std::integer_sequence<uint32_t, j...>) {
    (stv<uint32_t, true>(row + 4 * j, h[2 * j] | (h[2 * j + 1] << 16)), ...);
}

template <uint32_t src_stride, uint32_t dst_stride, uint32_t PB, uint32_t... i>
FORCE_INLINE void half_row_tile(uint32_t src, uint32_t row, std::integer_sequence<uint32_t, i...>) {
    // PB pages: halfword at src + p * src_stride (value C - 1) -> PB / 2 words of row C - 1.
    static_assert(sizeof...(i) == PB && PB % 2 == 0);
    const uint32_t h[] = {static_cast<uint32_t>(ldv<uint16_t, true>(src + i * src_stride))...};
    half_row_store(row, h, std::make_integer_sequence<uint32_t, PB / 2>{});
}

template <uint32_t pages, uint32_t values, uint32_t src_stride, uint32_t dst_stride, uint32_t PB>
FORCE_INLINE void transpose_rows(uint32_t src, uint32_t dst, uint32_t m_lo, uint32_t m_hi) {
    static_assert(PB % 2 == 0 && PB >= 2);
    static_assert(src_stride % 4 == 0 && dst_stride % 16 == 0);
    static_assert(dst_stride >= pages * 2, "a row holds one value per page");
    constexpr uint32_t words = values / 2;
    constexpr uint32_t G = pages / PB;
    constexpr uint32_t rem_pairs = (pages % PB) / 2;
    constexpr bool odd_page = pages % 2 != 0;
    for (uint32_t m = m_lo; m < m_hi; ++m) {
        if (m < words) {
            const uint32_t page = src + 4 * m;
            const uint32_t row = dst + 2 * m * dst_stride;
            for (uint32_t g = 0; g < G; ++g) {
                blk_tile<src_stride, dst_stride, PB, 1, true, true>(
                    page + g * PB * src_stride, row + g * PB * 2, std::make_integer_sequence<uint32_t, PB>{});
            }
            if constexpr (rem_pairs > 0) {
                blk_tile<src_stride, dst_stride, 2 * rem_pairs, 1, true, true>(
                    page + G * PB * src_stride,
                    row + G * PB * 2,
                    std::make_integer_sequence<uint32_t, 2 * rem_pairs>{});
            }
            if constexpr (odd_page) {
                const uint32_t a = ldv<uint32_t, true>(page + (pages - 1) * src_stride);
                stv<uint16_t, true>(row + 2 * (pages - 1), static_cast<uint16_t>(a));
                stv<uint16_t, true>(row + dst_stride + 2 * (pages - 1), static_cast<uint16_t>(a >> 16));
            }
        } else if constexpr (values % 2 != 0) {
            const uint32_t page = src + 2 * (values - 1);
            const uint32_t row = dst + (values - 1) * dst_stride;
            for (uint32_t g = 0; g < G; ++g) {
                half_row_tile<src_stride, dst_stride, PB>(
                    page + g * PB * src_stride, row + g * PB * 2, std::make_integer_sequence<uint32_t, PB>{});
            }
            if constexpr (rem_pairs > 0) {
                half_row_tile<src_stride, dst_stride, 2 * rem_pairs>(
                    page + G * PB * src_stride,
                    row + G * PB * 2,
                    std::make_integer_sequence<uint32_t, 2 * rem_pairs>{});
            }
            if constexpr (odd_page) {
                stv<uint16_t, true>(row + 2 * (pages - 1), ldv<uint16_t, true>(page + (pages - 1) * src_stride));
            }
        }
    }
}

// Work items of a unit for transpose_rows, and the item bounds of share [cum, cum') of `total`.
template <uint32_t values>
constexpr uint32_t row_items() {
    return (values + 1) / 2;
}
template <uint32_t values>
constexpr uint32_t row_bound(uint32_t cum, uint32_t total) {
    return row_items<values>() * cum / total;
}

// ---------------------------------------------------------------------------------------- dispatcher
// Pages [p_begin, p_end) of a unit of `pages` pages. Word-pair impls (BASE, LINE) need even C and an
// even p_begin; their odd last page (p_end == pages odd) goes through one halfword per row.
template <uint32_t pages, uint32_t values, uint32_t src_stride, uint32_t dst_stride, uint32_t impl, uint32_t batch>
FORCE_INLINE void transpose_range(uint32_t src, uint32_t dst, uint32_t p_begin, uint32_t p_end) {
    static_assert(src_stride % 4 == 0 && dst_stride % 4 == 0);
    static_assert(dst_stride >= pages * 2, "a row holds one value per page");
    if constexpr (impl == IMPL_HALF) {
        half_pages<values, src_stride, dst_stride, batch>(src, dst, p_begin, p_end);
    } else if constexpr (values % 2 != 0) {
        halfword_walk<values, src_stride, dst_stride>(src, dst, p_begin, p_end);
    } else {
        const uint32_t q_end = p_end / 2;
        if constexpr (impl == IMPL_BASE) {
            base_pairs<values, src_stride, dst_stride, batch>(src, dst, p_begin / 2, q_end);
        } else if constexpr (impl == IMPL_BLK || impl == IMPL_BLKV || impl == IMPL_BLKS) {
            blk_pairs<
                values,
                src_stride,
                dst_stride,
                (batch >> 8),
                (batch & 0xFF),
                impl == IMPL_BLKV,
                impl != IMPL_BLK>(src, dst, p_begin / 2, q_end);
        } else {
            line_pairs<values, src_stride, dst_stride, batch>(src, dst, p_begin / 2, q_end);
        }
        if constexpr (pages % 2 != 0) {
            if (p_end == pages) {
                const uint32_t page = src + (pages - 1) * src_stride;
                uint32_t row = dst + (pages - 1) * 2;
                for (uint32_t m = 0; m < values / 2; ++m) {
                    const uint32_t a = ld<uint32_t>(page + 4 * m);
                    st<uint16_t>(row, static_cast<uint16_t>(a));
                    st<uint16_t>(row + dst_stride, static_cast<uint16_t>(a >> 16));
                    row += 2 * dst_stride;
                }
            }
        }
    }
}

// The page bounds of share [cum_lo, cum_hi) of `total` over a unit of `pages` pages. Word-pair impls
// cut at page pairs (an odd last page stays with the last share).
template <uint32_t pages, uint32_t values, uint32_t impl>
constexpr uint32_t share_bound(uint32_t cum, uint32_t total) {
    if constexpr (impl != IMPL_HALF && values % 2 == 0) {
        return cum == total ? pages : 2 * ((pages / 2) * cum / total);
    } else {
        return pages * cum / total;
    }
}

}  // namespace rvt
