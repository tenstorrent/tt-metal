// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// In-place fill of a TILE tensor's partial-tile REMAINDER lanes with a packed
// constant, so a partial-input TILE pad can then use the fast whole-tile kernel
// (mirrors tt-metal FillPad + Pad). One data-movement kernel per assigned
// border tile: read tile -> fill L-shaped remainder (grow>=row_from ||
// gcol>=col_from) -> write the same tile back. Touches ONLY boundary tiles.
//
// Tiles are stored as four 16x16 faces (face0=rows0-15/cols0-15,
// face1=rows0-15/cols16-31, face2=rows16-31/cols0-15, face3=rows16-31/cols16-31);
// element (grow,gcol) lives at index (fr*2+fc)*256 + r*16 + c with fr=grow/16,
// fc=gcol/16, r=grow%16, c=gcol%16. row 16 is NOT after row 15 in memory.
//
// CT args: cb_scratch, tile_bytes, elem_size(2|4), packed_val, cb_pad,
//          then TensorAccessorArgs.
// RT args: [dst_addr, n], then n triples (tile_id, row_from, col_from).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n = get_arg_val<uint32_t>(1);

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
    for (uint32_t word = 0; word < pad_num_words; ++word) {
        pad_words[word] = pad_word;
    }
    CoreLocalMem<uint32_t> pad_source(pad_l1);

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t tid = get_arg_val<uint32_t>(2 + 3 * i + 0);
        uint32_t row_from = get_arg_val<uint32_t>(2 + 3 * i + 1);
        uint32_t col_from = get_arg_val<uint32_t>(2 + 3 * i + 2);

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
                        if ((idx & 1) && c_start < 16) {
                            fill16[idx++] = (uint16_t)(packed_val & 0xFFFF);
                            ++c_start;
                        }
                        for (; c_start + 1 < 16; c_start += 2, idx += 2) {
                            fill32[idx / 2] = packed_val16;
                        }
                        if (c_start < 16) {
                            fill16[idx] = (uint16_t)(packed_val & 0xFFFF);
                        }
                    } else {
                        for (; c_start < 16; ++c_start, ++idx) {
                            fill32[idx] = packed_val;
                        }
                    }
                }
            }
        }

        noc.async_write(scratch, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tid, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
