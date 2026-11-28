// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// riscv_transpose microbench body: impl < 8 is an rvt::transpose_range impl; 8+ are L1 access probes of the
// same volume as one unit (1024 words): 8 = word stores at 4 B stride, 9 = word stores at dst_stride
// (row walk), 10 = word loads at 4 B stride, 11 = halfword stores at dst_stride.
#include "rvt_transpose.hpp"

namespace rvt_micro {

template <uint32_t stride, uint32_t... k>
FORCE_INLINE void store8(uint32_t a, uint32_t v, std::integer_sequence<uint32_t, k...>) {
    (rvt::st<uint32_t>(a + k * stride, v + k), ...);
}
template <uint32_t stride, uint32_t... k>
FORCE_INLINE void hstore8(uint32_t a, uint32_t v, std::integer_sequence<uint32_t, k...>) {
    (rvt::st<uint16_t>(a + k * stride, static_cast<uint16_t>(v + k)), ...);
}
template <uint32_t... k>
FORCE_INLINE uint32_t load8(uint32_t a, std::integer_sequence<uint32_t, k...>) {
    const uint32_t w[] = {rvt::ld<uint32_t>(a + 4 * k)...};
    return (w[k] ^ ...);
}

template <uint32_t pages, uint32_t values, uint32_t src_stride, uint32_t dst_stride, uint32_t impl, uint32_t batch>
FORCE_INLINE uint32_t run_unit(uint32_t src, uint32_t dst) {
    constexpr auto seq8 = std::make_integer_sequence<uint32_t, 8>{};
    uint32_t acc = 0;
    if constexpr (impl < 8) {
        rvt::transpose_range<pages, values, src_stride, dst_stride, impl, batch>(src, dst, 0, pages);
    } else if constexpr (impl == 8) {
        for (uint32_t i = 0; i < 128; ++i) {
            store8<4>(dst + 32 * i, i, seq8);
        }
    } else if constexpr (impl == 9) {
        // 32 rows x 32 words, column-major walk: word q of rows 0..31 (8 rows per call).
        for (uint32_t q = 0; q < 32; ++q) {
            for (uint32_t r = 0; r < 32; r += 8) {
                store8<dst_stride>(dst + r * dst_stride + 4 * q, q, seq8);
            }
        }
    } else if constexpr (impl == 10) {
        for (uint32_t i = 0; i < 128; ++i) {
            acc += load8(src + 32 * i, seq8);
        }
    } else if constexpr (impl == 11) {
        for (uint32_t q = 0; q < 32; ++q) {
            for (uint32_t r = 0; r < 32; r += 8) {
                hstore8<dst_stride>(dst + r * dst_stride + 2 * q, q, seq8);
            }
        }
    } else if constexpr (impl == 12) {
        // 2 consecutive words per row, then the next row.
        for (uint32_t q = 0; q < 32; q += 2) {
            for (uint32_t r = 0; r < 32; r += 4) {
                const uint32_t a = dst + r * dst_stride + 4 * q;
                rvt::st<uint32_t>(a, q);
                rvt::st<uint32_t>(a + 4, q);
                rvt::st<uint32_t>(a + dst_stride, q);
                rvt::st<uint32_t>(a + dst_stride + 4, q);
                rvt::st<uint32_t>(a + 2 * dst_stride, q);
                rvt::st<uint32_t>(a + 2 * dst_stride + 4, q);
                rvt::st<uint32_t>(a + 3 * dst_stride, q);
                rvt::st<uint32_t>(a + 3 * dst_stride + 4, q);
            }
        }
    } else if constexpr (impl == 13) {
        // two rows alternating word by word.
        for (uint32_t r = 0; r < 32; r += 2) {
            for (uint32_t q = 0; q < 32; q += 4) {
                const uint32_t a = dst + r * dst_stride + 4 * q;
                rvt::st<uint32_t>(a, q);
                rvt::st<uint32_t>(a + dst_stride, q);
                rvt::st<uint32_t>(a + 4, q);
                rvt::st<uint32_t>(a + dst_stride + 4, q);
                rvt::st<uint32_t>(a + 8, q);
                rvt::st<uint32_t>(a + dst_stride + 8, q);
                rvt::st<uint32_t>(a + 12, q);
                rvt::st<uint32_t>(a + dst_stride + 12, q);
            }
        }
    } else if constexpr (impl == 14) {
        // column walk: word m of 8 pages per batch (every load a different line).
        for (uint32_t m = 0; m < 16; ++m) {
            for (uint32_t p = 0; p < 64; p += 8) {
                const uint32_t a = src + p * src_stride + 4 * m;
                uint32_t w[8];
                for (uint32_t k = 0; k < 8; ++k) {
                    w[k] = rvt::ld<uint32_t>(a + k * src_stride);
                }
                acc += w[0] ^ w[1] ^ w[2] ^ w[3] ^ w[4] ^ w[5] ^ w[6] ^ w[7];
            }
        }
    }
    return acc;
}

}  // namespace rvt_micro
