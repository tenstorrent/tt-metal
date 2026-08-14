// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NCRISC / NOC0).
//
// Reads ROW_MAJOR sticks from an interleaved tensor into cb_input_sticks, one
// block at a time. A block is 1 tile-row x WT_CHUNK tile-columns (op_design.md
// §1): TILE_H sticks of WT_CHUNK*32*elem bytes each, written at L1 stride
// row_bytes, with ONE barrier per block (master.md B7).
//
// Two regimes, selected by a compile-time arg (op_design.md §5.1):
//
//   R_ALIGNED — the hot path. Delegates verbatim to the library helper
//               dataflow_kernel_lib::read_sticks_for_tilize<TILE granularity>.
//
//   R_PAD     — HELPER SUBSTITUTION, justified: the library helper cannot fill.
//               Its contract (tilize_helpers_dataflow.hpp:50-52) states that for
//               a partial block "untouched rows contain stale data", and
//               .inl:120-123 reads only row_bytes while advancing L1 by the
//               padded stride, leaving the W tail untouched. There is no fill
//               parameter and no other kernel_lib helper covers a value-filled
//               read, while the pad oracle compares the pad region exactly. The
//               pad branch therefore uses raw dataflow (TensorAccessor +
//               noc_async_read + an L1 fill), keeping the helper's block
//               structure and one-barrier-per-block policy.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {

// Alignment-aware L1 fill: 4-byte stores for the aligned middle, element-sized
// stores for the unaligned head/tail (rv32 faults on unaligned wide stores).
// `val` carries the fill in the INPUT element format in its low bytes; it is
// replicated across the 32-bit store word, so a sub-word element fills every
// position (a value written once per word is invisible at 0 and garbage
// otherwise).
template <uint32_t elem_bytes>
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

    for (auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
         ptr < reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
         ++ptr) {
        *ptr = val_4B;
    }

    if constexpr (elem_bytes < 4) {
        const elem_t v = static_cast<elem_t>(val);
        for (auto* ptr = reinterpret_cast<volatile tt_l1_ptr elem_t*>(start_addr);
             ptr < reinterpret_cast<volatile tt_l1_ptr elem_t*>(start_addr_4B);
             ++ptr) {
            *ptr = v;
        }
        for (auto* ptr = reinterpret_cast<volatile tt_l1_ptr elem_t*>(end_addr_4B);
             ptr < reinterpret_cast<volatile tt_l1_ptr elem_t*>(end_addr);
             ++ptr) {
            *ptr = v;
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_sticks = 0;
    constexpr uint32_t TILE_W = 32;  // a tile is always 32 wide (hardware fact, not a knob)

    constexpr uint32_t regime = get_compile_time_arg_val(0);
    constexpr uint32_t tile_h = get_compile_time_arg_val(1);
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(2);  // the W block factor
    constexpr uint32_t nt_h = get_compile_time_arg_val(3);
    constexpr uint32_t nth_per_img = get_compile_time_arg_val(4);
    constexpr uint32_t h_in = get_compile_time_arg_val(5);
    constexpr uint32_t n_img_in = get_compile_time_arg_val(6);
    constexpr uint32_t w_in_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(8);
    constexpr auto src_args = TensorAccessorArgs<9>();

    // Every byte quantity below derives from the WT_CHUNK knob — one source.
    constexpr uint32_t row_bytes = wt_chunk * TILE_W * elem_bytes;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t pad_word = get_arg_val<uint32_t>(3);

    if (num_blocks == 0) {
        return;
    }

    const auto accessor = TensorAccessor(src_args, src_addr);

    if constexpr (regime == 0) {
        // ── R_ALIGNED ────────────────────────────────────────────────────
        // Blocks are W-chunk-major, so a run of consecutive blocks that shares
        // one W chunk is one helper call over run*TILE_H contiguous sticks.
        uint32_t block = start_block;
        uint32_t remaining = num_blocks;
        while (remaining > 0) {
            const uint32_t wc = block / nt_h;
            const uint32_t row = block % nt_h;
            uint32_t run = nt_h - row;
            if (run > remaining) {
                run = remaining;
            }
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks, dataflow_kernel_lib::TilizeGranularity::TILE>(
                accessor,
                /*total_num_rows=*/run * tile_h,
                /*row_bytes=*/row_bytes,
                /*start_page=*/row * tile_h,
                /*byte_offset_within_page=*/wc * row_bytes);
            block += run;
            remaining -= run;
        }
    } else {
        // ── R_PAD ────────────────────────────────────────────────────────
        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t block = start_block + i;
            const uint32_t wc = block / nt_h;
            const uint32_t row = block % nt_h;
            const uint32_t img = row / nth_per_img;
            const uint32_t row_in_img = (row % nth_per_img) * tile_h;
            const uint32_t col_off = wc * row_bytes;

            // Bytes of real data this block's rows carry (W tail beyond it).
            uint32_t valid_bytes = 0;
            if (col_off < w_in_bytes) {
                valid_bytes = w_in_bytes - col_off;
                if (valid_bytes > row_bytes) {
                    valid_bytes = row_bytes;
                }
            }

            cb_reserve_back(cb_input_sticks, wt_chunk);
            uint32_t l1_addr = get_write_ptr(cb_input_sticks);

            for (uint32_t r = 0; r < tile_h; ++r) {
                const uint32_t src_row = row_in_img + r;
                // H tail and whole pad tiles: no source row at all.
                const uint32_t n_read = (img < n_img_in && src_row < h_in) ? valid_bytes : 0;
                if (n_read > 0) {
                    noc_async_read(accessor.get_noc_addr(img * h_in + src_row, col_off), l1_addr, n_read);
                }
                if (n_read < row_bytes) {
                    fill_l1_with_val<elem_bytes>(l1_addr + n_read, row_bytes - n_read, pad_word);
                }
                l1_addr += row_bytes;
            }

            noc_async_read_barrier();
            cb_push_back(cb_input_sticks, wt_chunk);
        }
    }
}
