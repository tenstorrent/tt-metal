// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for tile-based YUV conversion (per-unit L1-resident RGB).
//
// The unit of work is one (row-group, T-tile): a row-group is 2 H-rows -> 1 UV
// row.  For each unit the reader reads its 2 RGB rows (all 3 channels, one
// T-tile) from DRAM ONCE into an L1 scratch buffer, then re-arranges from L1
// (no DRAM re-read) to feed the compute kernel's three sub-passes:
//   Y:  flat 32-stick tiles of the 2 rows.
//   Cb: for each UV tile, the 4 2x2-corner sticks per channel (compute sums
//       them; the reader-side 0.25 pre-scale of the chroma weights turns the
//       sum into the 2x2 average).
//   Cr: same 4-corner arrangement, different (resident) scalar coefficients.
//
// The 12 scalar coefficient tiles (Y/Cb/Cr x wr/wg/wb/off) are generated once
// up front and kept resident (never popped) since they are the same for every
// unit.
//
// Compile-time args:
//   [0] cb_R_rm, [1] cb_G_rm, [2] cb_B_rm  — row-major bf16 channel CBs
//   [3] cb_scratch                          — per-unit RGB residency (raw L1)
//   [4] cb_scalar_base                      — first of 12 resident scalar CBs
//   [5] num_t_tiles, [6] T, [7] W, [8] W2, [9] HW
//   [10] y_tiles (= ceil(2W/32)), [11] uv_tiles (= ceil(W2/32))
//   [12..] TensorAccessorArgs for input
//
// Runtime args:
//   [0] src_addr
//   [1..12] coefficients as bf16 packed in upper 16 bits
//           (Y wr,wg,wb,off; Cb wr,wg,wb,off pre-scaled *0.25 except off; Cr ...)
//   [13] unit_start, [14] unit_count

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;
constexpr uint32_t FULL_TILE_BYTES = TILE_W * 2;      // 64 bytes: one stick's T-tile (32 bf16)
constexpr uint32_t PAGE_BYTES = TILE_H * TILE_W * 2;  // 2048: one row-major channel tile

// Zero a full 2048-byte channel page (pads partial spatial/T).
FORCE_INLINE void zero_page(uint32_t l1_base) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_base);
    for (uint32_t i = 0; i < PAGE_BYTES / 4; i++) {
        p[i] = 0;
    }
}

// Local L1->L1 copy via the NOC DMA engine (async; caller barriers before use).
// Scalar RISC copies here are far too slow and make the reader the bottleneck.
FORCE_INLINE void local_read(uint32_t src_l1, uint32_t dst_l1, uint32_t nbytes) {
    noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], src_l1), dst_l1, nbytes);
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_R_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_G_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_B_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scalar_base = get_compile_time_arg_val(4);
    constexpr uint32_t num_t_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t T = get_compile_time_arg_val(6);
    constexpr uint32_t W = get_compile_time_arg_val(7);
    constexpr uint32_t W2 = get_compile_time_arg_val(8);
    constexpr uint32_t HW = get_compile_time_arg_val(9);
    constexpr uint32_t y_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t uv_tiles = get_compile_time_arg_val(11);
    constexpr auto src_tensor_args = TensorAccessorArgs<12>();

    constexpr uint32_t last_tile_elems = T - (num_t_tiles - 1) * TILE_W;
    constexpr uint32_t last_tile_bytes = last_tile_elems * 2;
    constexpr uint32_t y_sticks = 2 * W;

    // Generate the 12 resident scalar tiles once (Y, Cb, Cr) x (wr, wg, wb, off).
    for (uint32_t k = 0; k < 12; k++) {
        generate_bcast_unary_scalar(CircularBuffer(cb_scalar_base + k), get_arg_val<uint32_t>(1 + k));
    }

    const uint32_t unit_start = get_arg_val<uint32_t>(13);
    const uint32_t unit_count = get_arg_val<uint32_t>(14);

    const auto src = TensorAccessor(src_tensor_args, src_addr);
    const uint32_t scratch_base = get_write_ptr(cb_scratch);
    const uint32_t cb_ids[3] = {cb_R_rm, cb_G_rm, cb_B_rm};

    for (uint32_t u = unit_start; u < unit_start + unit_count; u++) {
        const uint32_t g = u / num_t_tiles;
        const uint32_t tt = u % num_t_tiles;
        const bool is_last_t = (tt == num_t_tiles - 1) && (last_tile_elems < TILE_W);
        const uint32_t read_bytes = is_last_t ? last_tile_bytes : FULL_TILE_BYTES;
        const uint32_t byte_off = tt * FULL_TILE_BYTES;

        // ---- 1. Read this unit's 2 RGB rows from DRAM into L1 scratch (once) ----
        // scratch layout: [c][s][32 T], s in [0, 2W); spatial_global = 2*g*W + s.
        for (uint32_t c = 0; c < 3; c++) {
            for (uint32_t s = 0; s < y_sticks; s++) {
                uint32_t spatial = 2 * g * W + s;
                uint32_t src_row = c * HW + spatial;
                uint32_t dst = scratch_base + (c * y_sticks + s) * FULL_TILE_BYTES;
                noc_async_read(src.get_noc_addr(src_row, byte_off), dst, read_bytes);
            }
        }
        noc_async_read_barrier();

        // ---- 2. Y sub-pass: emit flat 32-stick tiles from scratch ----
        // A full tile (32 sticks, full T) is 32 contiguous 64-byte sticks in
        // scratch -> a single 2048-byte NOC copy.  Partial edges fall back to
        // zero-pad + per-stick copies.
        for (uint32_t yt = 0; yt < y_tiles; yt++) {
            uint32_t base = yt * TILE_H;
            uint32_t sticks = (base + TILE_H <= y_sticks) ? TILE_H : (y_sticks - base);
            bool full = (sticks == TILE_H) && !is_last_t;
            for (uint32_t c = 0; c < 3; c++) {
                cb_reserve_back(cb_ids[c], 1);
                uint32_t l1 = get_write_ptr(cb_ids[c]);
                uint32_t src_base = scratch_base + (c * y_sticks + base) * FULL_TILE_BYTES;
                if (full) {
                    local_read(src_base, l1, PAGE_BYTES);
                } else {
                    zero_page(l1);
                    for (uint32_t s = 0; s < sticks; s++) {
                        local_read(src_base + s * FULL_TILE_BYTES, l1 + s * FULL_TILE_BYTES, read_bytes);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_ids[c], 1);
            }
        }

        // ---- 3 & 4. Cb and Cr sub-passes: 4-corner gather from scratch ----
        // Both planes consume the identical 2x2-corner arrangement (differing
        // only in the resident scalar coefficients the compute kernel applies).
        // Corner sticks are strided in scratch, so these are per-stick copies.
        for (uint32_t plane = 0; plane < 2; plane++) {
            for (uint32_t ut = 0; ut < uv_tiles; ut++) {
                uint32_t base = ut * TILE_H;
                uint32_t sticks = (base + TILE_H <= W2) ? TILE_H : (W2 - base);
                bool pad = (sticks < TILE_H) || is_last_t;
                for (uint32_t c = 0; c < 3; c++) {
                    for (uint32_t corner = 0; corner < 4; corner++) {
                        cb_reserve_back(cb_ids[c], 1);
                        uint32_t l1 = get_write_ptr(cb_ids[c]);
                        if (pad) {
                            zero_page(l1);
                        }
                        uint32_t row = corner >> 1;  // local H-row within the 2-row group (0 or 1)
                        uint32_t woff = corner & 1;  // W offset within the 2x2 block (0 or 1)
                        for (uint32_t s = 0; s < sticks; s++) {
                            uint32_t w_uv = base + s;
                            uint32_t col = 2 * w_uv + woff;
                            uint32_t scratch_idx = row * W + col;  // spatial within the 2 rows
                            uint32_t src_stick = scratch_base + (c * y_sticks + scratch_idx) * FULL_TILE_BYTES;
                            local_read(src_stick, l1 + s * FULL_TILE_BYTES, read_bytes);
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_ids[c], 1);
                    }
                }
            }
        }
    }
}
