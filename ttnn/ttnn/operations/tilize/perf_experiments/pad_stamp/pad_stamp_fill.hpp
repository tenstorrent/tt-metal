// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// EXPERIMENT ONLY (perf bake-off "pad_stamp"). Not part of the op.
//
// A copy of ttnn/ttnn/operations/tilize/kernels/tilize_fill.hpp with the L1 fill
// loop parameterized by an IMPLEMENTATION id, so the bake-off can price the
// store-side of the pad stamp against the op's current one. Everything else is
// byte-identical to the shipped header, so impl 0 IS the honest baseline.
//
// Namespaced `pad_stamp` (not `tilize_kernels`) so a TU can include this AND the
// real header without a redefinition.

#pragma once

#include <cstdint>
#include <type_traits>

namespace pad_stamp {

// --- fill implementations ---------------------------------------------------
// 0 = BASELINE. Exactly the shipped loop: one volatile 32-bit store per
//     iteration, with element-sized stores for the unaligned head/tail.
// 1 = UNROLLED. Same volatile 32-bit stores in the same order, 8 per loop
//     iteration off one base register, so the per-store add/compare/branch is
//     amortized 8x. Still volatile (no merging, no memset call, no reordering)
//     — this changes only the loop overhead, never the store width or the bytes
//     written.
constexpr uint32_t FILL_BASE = 0;
constexpr uint32_t FILL_UNROLL = 1;

template <uint32_t elem_bytes, uint32_t impl = FILL_BASE>
FORCE_INLINE void fill_l1_with_val(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(elem_bytes == 1 || elem_bytes == 2 || elem_bytes == 4, "unsupported element width");
    using elem_t =
        std::conditional_t<elem_bytes == 1, uint8_t, std::conditional_t<elem_bytes == 2, uint16_t, uint32_t>>;

    const uint32_t end_addr = start_addr + n_bytes;
    const uint32_t start_addr_4B = (start_addr + 3u) & ~3u;
    const uint32_t end_addr_4B = end_addr & ~3u;

    uint32_t val_4B = val;
    if constexpr (elem_bytes == 1) {
        const uint32_t b = val & 0xFFu;
        val_4B = (b << 24) | (b << 16) | (b << 8) | b;
    } else if constexpr (elem_bytes == 2) {
        const uint32_t h = val & 0xFFFFu;
        val_4B = (h << 16) | h;
    }

    auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
    auto* const e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
    if constexpr (impl == FILL_UNROLL) {
        while (p + 8 <= e) {
            p[0] = val_4B;
            p[1] = val_4B;
            p[2] = val_4B;
            p[3] = val_4B;
            p[4] = val_4B;
            p[5] = val_4B;
            p[6] = val_4B;
            p[7] = val_4B;
            p += 8;
        }
    }
    for (; p < e; ++p) {
        *p = val_4B;
    }

    if constexpr (elem_bytes < 4) {
        const elem_t v = static_cast<elem_t>(val);
        for (auto* q = reinterpret_cast<volatile tt_l1_ptr elem_t*>(start_addr);
             q < reinterpret_cast<volatile tt_l1_ptr elem_t*>(start_addr_4B);
             ++q) {
            *q = v;
        }
        for (auto* q = reinterpret_cast<volatile tt_l1_ptr elem_t*>(end_addr_4B);
             q < reinterpret_cast<volatile tt_l1_ptr elem_t*>(end_addr);
             ++q) {
            *q = v;
        }
    }
}

// Re-stamp the pad region of ONE tiled output tile — verbatim from the shipped
// header apart from the `impl` pass-through.
template <uint32_t tile_h, uint32_t tile_w, uint32_t elem_bytes, uint32_t impl = FILL_BASE>
FORCE_INLINE void fill_tile_pad(uint32_t tile_addr, uint32_t valid_rows, uint32_t valid_cols, uint32_t word) {
    constexpr uint32_t FACE_H = tile_h < 16 ? tile_h : 16;
    constexpr uint32_t FACE_W = 16;
    static_assert(tile_h % FACE_H == 0 && tile_w % FACE_W == 0, "fill_tile_pad needs whole faces");
    constexpr uint32_t FACES_PER_ROW = tile_w / FACE_W;
    constexpr uint32_t FACE_ROWS = tile_h / FACE_H;
    constexpr uint32_t FACE_ELEMS = FACE_H * FACE_W;

    if (valid_rows == 0 || valid_cols == 0) {
        fill_l1_with_val<elem_bytes, impl>(tile_addr, tile_h * tile_w * elem_bytes, word);
        return;
    }
    if (valid_rows >= tile_h && valid_cols >= tile_w) {
        return;
    }

    for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
        uint32_t vr = (valid_rows > fr * FACE_H) ? (valid_rows - fr * FACE_H) : 0;
        if (vr > FACE_H) {
            vr = FACE_H;
        }
        for (uint32_t fc = 0; fc < FACES_PER_ROW; ++fc) {
            uint32_t vc = (valid_cols > fc * FACE_W) ? (valid_cols - fc * FACE_W) : 0;
            if (vc > FACE_W) {
                vc = FACE_W;
            }
            const uint32_t face_addr = tile_addr + (fr * FACES_PER_ROW + fc) * FACE_ELEMS * elem_bytes;
            if (vr == 0 || vc == 0) {
                fill_l1_with_val<elem_bytes, impl>(face_addr, FACE_ELEMS * elem_bytes, word);
                continue;
            }
            if (vc < FACE_W) {
                for (uint32_t rr = 0; rr < vr; ++rr) {
                    fill_l1_with_val<elem_bytes, impl>(
                        face_addr + (rr * FACE_W + vc) * elem_bytes, (FACE_W - vc) * elem_bytes, word);
                }
            }
            if (vr < FACE_H) {
                fill_l1_with_val<elem_bytes, impl>(
                    face_addr + vr * FACE_W * elem_bytes, (FACE_H - vr) * FACE_W * elem_bytes, word);
            }
        }
    }
}

}  // namespace pad_stamp
