// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fills the remainder lanes of a TILE tensor's partial border tiles with the pad value in place, so the
// whole-tile pad kernel can run next. A tile is four 16x16 faces: element (r, c) is in face (r/16)*2 + c/16
// at offset (r%16)*16 + c%16, so row 16 does not follow row 15 in memory.
//
// RT args: [dst_addr, start, n, Ht, Wt, h_mod, w_mod]: border tiles start..start+n in plane-major order (bottom
// tile row when h_mod, then right tile column when w_mod; the corner belongs to the row). Deriving tiles from the
// range keeps the arg count constant, well under the per-kernel runtime-arg cap however many tiles a core owns.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#if !defined(ARCH_QUASAR)
#include "ckernel.h"
#endif

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start = get_arg_val<uint32_t>(1);
    uint32_t n = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);
    uint32_t h_mod = get_arg_val<uint32_t>(5);
    uint32_t w_mod = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t elem_size = get_compile_time_arg_val(2);
    constexpr uint32_t packed_val = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pad = get_compile_time_arg_val(4);
    constexpr auto args = TensorAccessorArgs<5>();
    const auto s = TensorAccessor(args, dst_addr, tile_bytes);

    Noc noc;
    CircularBuffer scratch_cb(cb_scratch);
    CircularBuffer pad_cb(cb_pad);

    scratch_cb.reserve_back(1);
    uint32_t l1 = scratch_cb.get_write_ptr();
    // The tile is staged at the scratch CB's WRITE pointer (never pushed), so
    // address the staging buffer explicitly rather than through the CB object,
    // whose read-pointer resolution would point elsewhere.
    CoreLocalMem<uint32_t> scratch(l1);
    CoreLocalMem<volatile uint16_t> fill16(l1);
    CoreLocalMem<volatile uint32_t> fill32(l1);
    constexpr uint32_t packed_val16 = (packed_val & 0xFFFF) | ((packed_val & 0xFFFF) << 16);

    // One immutable face of the fill value is enough to source every aligned
    // row-tail write below. Keep it separate from scratch, which the partial-
    // column fallback overwrites with input tile contents.
    pad_cb.reserve_back(1);
    uint32_t pad_l1 = pad_cb.get_write_ptr();
    CoreLocalMem<volatile uint32_t> pad_words(pad_l1);
    constexpr uint32_t pad_word = elem_size == 2 ? packed_val16 : packed_val;
    constexpr uint32_t pad_num_words = 16 * 16 * elem_size / 4;
    constexpr uint32_t tile_words = 32 * 32 * elem_size / 4;
    for (uint32_t word = 0; word < pad_num_words; ++word) {
        pad_words[word] = pad_word;
    }
#if !defined(ARCH_QUASAR)
    // The RISC and the NoC are separate L1 clients and a store can retire
    // before it lands, so the NoC could source stale bytes. A blocking load of
    // the last store drains every earlier one; Wormhole matches a load to a
    // store by start address only, so it must name that store exactly.
    (void)ckernel::load_blocking(&pad_words[pad_num_words - 1]);
#endif
    CoreLocalMem<uint32_t> pad_source(pad_l1);

    if (n == 0) {
        return;
    }
    const uint32_t HtWt = Ht * Wt;
    const uint32_t bottom = h_mod ? Wt : 0;
    const uint32_t per_plane = bottom + (w_mod ? (h_mod ? Ht - 1 : Ht) : 0);
    uint32_t plane_base = (start / per_plane) * HtWt;
    uint32_t j = start % per_plane;

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t tid, row_from, col_from;
        if (j < bottom) {
            tid = plane_base + (Ht - 1) * Wt + j;
            row_from = h_mod;
            col_from = (w_mod && j == Wt - 1) ? w_mod : 32;
        } else {
            tid = plane_base + (j - bottom) * Wt + Wt - 1;
            row_from = 32;
            col_from = w_mod;
        }
        if (++j == per_plane) {
            j = 0;
            plane_base += HtWt;
        }

        // With no partial column, invalid rows form one contiguous suffix in
        // each of the four faces. Write those spans directly; valid lanes never
        // enter L1 and need no read-modify-write.
        if (col_from == 32) {
            for (uint32_t fr = 0; fr < 2; ++fr) {
                uint32_t face_row_from = row_from > fr * 16 ? row_from - fr * 16 : 0;
                if (face_row_from > 16) {
                    face_row_from = 16;
                }
                uint32_t rows = 16 - face_row_from;
                uint32_t bytes = rows * 16 * elem_size;
                if (bytes == 0) {
                    continue;
                }
                for (uint32_t fc = 0; fc < 2; ++fc) {
                    uint32_t offset = ((fr * 2 + fc) * 256 + face_row_from * 16) * elem_size;
                    noc.async_write(
                        pad_source, s, bytes, {.offset_bytes = 0}, {.page_id = tid, .offset_bytes = offset});
                }
            }
            noc.async_write_barrier();
            continue;
        }

        noc.async_read(s, scratch, tile_bytes, {.page_id = tid, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();

        for (uint32_t fr = 0; fr < 2; ++fr) {
            for (uint32_t fc = 0; fc < 2; ++fc) {
                uint32_t face_base = (fr * 2 + fc) * 256;
                for (uint32_t r = 0; r < 16; ++r) {
                    uint32_t grow = fr * 16 + r;
                    uint32_t c_start = 0;
                    if (grow < row_from) {
                        uint32_t face_col = fc * 16;
                        if (col_from >= face_col + 16) {
                            c_start = 16;
                        } else if (col_from > face_col) {
                            c_start = col_from - face_col;
                        }
                    }
                    uint32_t idx = face_base + r * 16 + c_start;
                    if constexpr (elem_size == 2) {
                        // Rows start on an even element, so after the odd head
                        // the halfword pairs end exactly at column 16.
                        if ((idx & 1) && c_start < 16) {
                            fill16[idx++] = (uint16_t)(packed_val & 0xFFFF);
                            ++c_start;
                        }
                        for (; c_start < 16; c_start += 2, idx += 2) {
                            fill32[idx / 2] = packed_val16;
                        }
                    } else {
                        for (; c_start < 16; ++c_start, ++idx) {
                            fill32[idx] = packed_val;
                        }
                    }
                }
            }
        }
#if !defined(ARCH_QUASAR)
        // Same store/NoC race as the pad face. With col_from < 32 the last
        // tile row always fills through column 31, so the final store ends the
        // tile: a 32-bit word, or for 2-byte elements a lone halfword when
        // that row's fill starts at odd column 15.
        const uint32_t last_row_c_start = (row_from == 32 && col_from > 16) ? col_from - 16 : 0;
        if (elem_size == 2 && last_row_c_start == 15) {
            (void)ckernel::load_blocking(&fill16[2 * tile_words - 1]);
        } else {
            (void)ckernel::load_blocking(&fill32[tile_words - 1]);
        }
#endif

        noc.async_write(scratch, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tid, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
