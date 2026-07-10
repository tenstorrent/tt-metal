// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for tile-based YUV conversion (Strategy 2: 32-stick packed tiles).
//
// Assembles 32 spatial sticks per tile into row-major bf16 CBs for each
// channel (R, G, B).  Also generates scalar coefficient tiles.
//
// Three phases: Y, Cb, Cr.  For each phase, generates scalar tiles, then
// fills channel data.  The compute kernel pops scalar tiles between phases,
// so the reader blocks on cb_reserve_back until the compute is ready.
//
// Y pass:  for each batch of 32 Y positions, fills cb_R_rm/cb_G_rm/cb_B_rm
//          with 32 rows × 32 columns of bf16 data (one T-tile at a time).
// UV pass: for each UV tile, sends 4 corner pages per channel into the
//          same channel CBs (compute accumulates them for 2×2 averaging).
//
// Compile-time args:
//   [0] cb_R_rm, [1] cb_G_rm, [2] cb_B_rm  — row-major bf16 output CBs
//   [3] cb_wr, [4] cb_wg, [5] cb_wb, [6] cb_off — scalar tile CBs
//   [7] cb_out_rm  — unused by reader, listed for index consistency
//   [8]  num_t_tiles  — ceil(T / 32)
//   [9]  T
//   [10] H, [11] W, [12] H2, [13] W2
//   [14] HW  (= H * W)
//   [15..] TensorAccessorArgs for input
//
// Runtime args:
//   [0]  src_addr
//   [1..4]   Y  coefficients as bf16 packed in upper 16 bits
//   [5..8]   Cb coefficients (wr*0.25, wg*0.25, wb*0.25, offset)
//   [9..12]  Cr coefficients (wr*0.25, wg*0.25, wb*0.25, offset)
//   [13] y_start, [14] y_count, [15] uv_start, [16] uv_count

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;

template <typename T>
FORCE_INLINE void read_stick_into_row(
    const T& src_accessor,
    uint32_t src_row,
    uint32_t byte_off,
    uint32_t read_bytes,
    uint32_t l1_base,
    uint32_t row_idx) {
    uint32_t dst = l1_base + row_idx * TILE_W * 2;
    noc_async_read(src_accessor.get_noc_addr(src_row, byte_off), dst, read_bytes);
}

// Zero a full 2048-byte page.
FORCE_INLINE void zero_page(uint32_t l1_base, uint32_t page_bytes) {
    volatile uint32_t* p32 = reinterpret_cast<volatile uint32_t*>(l1_base);
    for (uint32_t i = 0; i < page_bytes / 4; i++) {
        p32[i] = 0;
    }
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_R_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_G_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_B_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_wr = get_compile_time_arg_val(3);
    constexpr uint32_t cb_wg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_wb = get_compile_time_arg_val(5);
    constexpr uint32_t cb_off = get_compile_time_arg_val(6);
    constexpr uint32_t num_t_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr uint32_t H = get_compile_time_arg_val(10);
    constexpr uint32_t W = get_compile_time_arg_val(11);
    constexpr uint32_t H2 = get_compile_time_arg_val(12);
    constexpr uint32_t W2 = get_compile_time_arg_val(13);
    constexpr uint32_t HW = get_compile_time_arg_val(14);
    constexpr auto src_tensor_args = TensorAccessorArgs<15>();

    constexpr uint32_t FULL_TILE_BYTES = TILE_W * 2;  // 64 bytes per tile row
    constexpr uint32_t last_tile_elems = T - (num_t_tiles - 1) * TILE_W;
    constexpr uint32_t last_tile_bytes = last_tile_elems * 2;
    constexpr uint32_t PAGE_BYTES = TILE_H * TILE_W * 2;  // 2048

    const auto src = TensorAccessor(src_tensor_args, src_addr);

    const uint32_t y_start = get_arg_val<uint32_t>(13);
    const uint32_t y_count = get_arg_val<uint32_t>(14);
    const uint32_t uv_start = get_arg_val<uint32_t>(15);
    const uint32_t uv_count = get_arg_val<uint32_t>(16);

    const uint32_t y_end = y_start + y_count;
    const uint32_t uv_end = uv_start + uv_count;
    const uint32_t y_batches = (y_count + TILE_H - 1) / TILE_H;

    // ---- Phase 1: Y pass ----
    // Generate Y scalar tiles.  Compute will wait_front, use them, then pop.
    generate_bcast_unary_scalar(CircularBuffer(cb_wr),get_arg_val<uint32_t>(1));
    generate_bcast_unary_scalar(CircularBuffer(cb_wg),get_arg_val<uint32_t>(2));
    generate_bcast_unary_scalar(CircularBuffer(cb_wb),get_arg_val<uint32_t>(3));
    generate_bcast_unary_scalar(CircularBuffer(cb_off),get_arg_val<uint32_t>(4));

    for (uint32_t batch = 0; batch < y_batches; batch++) {
        uint32_t base_spatial = y_start + batch * TILE_H;
        uint32_t sticks_in_batch = (base_spatial + TILE_H <= y_end) ? TILE_H : (y_end - base_spatial);

        for (uint32_t tt = 0; tt < num_t_tiles; tt++) {
            const bool is_last = (tt == num_t_tiles - 1) && (last_tile_elems < TILE_W);
            const uint32_t read_bytes = is_last ? last_tile_bytes : FULL_TILE_BYTES;
            const uint32_t byte_off = tt * FULL_TILE_BYTES;

            const uint32_t cb_ids[3] = {cb_R_rm, cb_G_rm, cb_B_rm};
            for (uint32_t ch = 0; ch < 3; ch++) {
                cb_reserve_back(cb_ids[ch], 1);
                uint32_t l1_base = get_write_ptr(cb_ids[ch]);
                zero_page(l1_base, PAGE_BYTES);

                for (uint32_t s = 0; s < sticks_in_batch; s++) {
                    uint32_t spatial = base_spatial + s;
                    uint32_t src_row = ch * HW + spatial;
                    read_stick_into_row(src, src_row, byte_off, read_bytes, l1_base, s);
                }
                noc_async_read_barrier();
                cb_push_back(cb_ids[ch], 1);
            }
        }
    }

    // ---- Phases 2 & 3: Cb and Cr UV passes ----
    for (uint32_t pass = 0; pass < 2; pass++) {
        uint32_t coeff_base = 5 + pass * 4;

        // Push new scalar tiles for this UV plane.
        // Compute will have popped the previous set, freeing space in these CBs.
        generate_bcast_unary_scalar(CircularBuffer(cb_wr),get_arg_val<uint32_t>(coeff_base + 0));
        generate_bcast_unary_scalar(CircularBuffer(cb_wg),get_arg_val<uint32_t>(coeff_base + 1));
        generate_bcast_unary_scalar(CircularBuffer(cb_wb),get_arg_val<uint32_t>(coeff_base + 2));
        generate_bcast_unary_scalar(CircularBuffer(cb_off),get_arg_val<uint32_t>(coeff_base + 3));

        uint32_t uv_batches = (uv_count + TILE_H - 1) / TILE_H;

        for (uint32_t batch = 0; batch < uv_batches; batch++) {
            uint32_t base_uv = uv_start + batch * TILE_H;
            uint32_t sticks_in_batch = (base_uv + TILE_H <= uv_end) ? TILE_H : (uv_end - base_uv);

            for (uint32_t tt = 0; tt < num_t_tiles; tt++) {
                const bool is_last = (tt == num_t_tiles - 1) && (last_tile_elems < TILE_W);
                const uint32_t read_bytes = is_last ? last_tile_bytes : FULL_TILE_BYTES;
                const uint32_t byte_off = tt * FULL_TILE_BYTES;

                // For each channel, send 4 corner pages.
                const uint32_t cb_ids[3] = {cb_R_rm, cb_G_rm, cb_B_rm};
                for (uint32_t ch = 0; ch < 3; ch++) {
                    for (uint32_t corner = 0; corner < 4; corner++) {
                        cb_reserve_back(cb_ids[ch], 1);
                        uint32_t l1_base = get_write_ptr(cb_ids[ch]);
                        zero_page(l1_base, PAGE_BYTES);

                        for (uint32_t s = 0; s < sticks_in_batch; s++) {
                            uint32_t uv_idx = base_uv + s;
                            uint32_t h_uv = uv_idx / W2;
                            uint32_t w_uv = uv_idx % W2;
                            uint32_t h = 2 * h_uv + (corner >> 1);
                            uint32_t w = 2 * w_uv + (corner & 1);
                            uint32_t src_row = ch * HW + h * W + w;
                            read_stick_into_row(src, src_row, byte_off, read_bytes, l1_base, s);
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_ids[ch], 1);
                    }
                }
            }
        }
    }
}
