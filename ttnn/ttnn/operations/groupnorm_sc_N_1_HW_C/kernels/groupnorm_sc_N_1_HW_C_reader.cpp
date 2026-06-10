// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for groupnorm_sc_N_1_HW_C (multi-core GroupNorm over (N, 1, HW, C)).
//
// Startup:
//   - push the reduce scaler tile once: 1/sqrt(HW * Cg). REDUCE_SCALAR applies
//     the scaler twice (row then col), so SUM * (1/sqrt(N))^2 = mean.
//   - read gamma / beta (Wt tiles each, host-tilized (1,1,1,C) -> 1 x Wt tile
//     row) once; they persist in their CBs for the whole kernel (HeldBulk).
//     Every core reads its own gamma/beta copy from DRAM.
//
// Work split (interleaved-parallel): one work unit = one (n, g) group; this
// core handles group ids [start_group, start_group + num_groups_here) with
// n = id / G, g = id % G. Per group the compute kernel makes THREE streaming
// passes over the same Ht x Wg tile slab (mean, variance, normalize), so the
// reader streams the slab three times. Tile index: n*Ht*Wt + r*Wt + g*Wg + c.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Fill one bf16 tile with a 0/1 mask: 1.0 where (row < valid_rows && col < valid_cols),
// 0 elsewhere. Tile faces are sequential 16x16 blocks: f0 (r0-15, c0-15), f1 (r0-15,
// c16-31), f2 (r16-31, c0-15), f3 (r16-31, c16-31). bf16 pairs pack little-endian —
// the lower 16 bits of each u32 hold the even (first) column of the pair.
static void fill_mask_tile(uint32_t l1_addr, uint32_t valid_rows, uint32_t valid_cols) {
    constexpr uint32_t ONE_PAIR = 0x3F803F80u;  // [1.0, 1.0]
    constexpr uint32_t ONE_LO = 0x00003F80u;    // [1.0, 0.0] (low half = even column)
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t face_r = 0; face_r < 2; ++face_r) {
        const uint32_t fr = valid_rows > face_r * 16 ? (valid_rows - face_r * 16 > 16 ? 16 : valid_rows - face_r * 16)
                                                     : 0;  // valid rows in this face
        for (uint32_t face_c = 0; face_c < 2; ++face_c) {
            const uint32_t fc = valid_cols > face_c * 16
                                    ? (valid_cols - face_c * 16 > 16 ? 16 : valid_cols - face_c * 16)
                                    : 0;  // valid cols in this face
            const uint32_t full_pairs = fc / 2;
            const bool odd = (fc & 1) != 0;
            for (uint32_t r = 0; r < 16; ++r) {
                const bool row_valid = r < fr;
                for (uint32_t pair = 0; pair < 8; ++pair) {
                    uint32_t v = 0;
                    if (row_valid) {
                        v = pair < full_pairs ? ONE_PAIR : (odd && pair == full_pairs ? ONE_LO : 0u);
                    }
                    *p++ = v;
                }
            }
        }
    }
}

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_group = get_arg_val<uint32_t>(3);
    const uint32_t num_groups_here = get_arg_val<uint32_t>(4);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Wg = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_BETA = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t inv_sqrt_n_bits = get_compile_time_arg_val(6);
    constexpr uint32_t hw_tail = get_compile_time_arg_val(7);  // HW % 32 (0 = aligned)
    constexpr uint32_t c_tail = get_compile_time_arg_val(8);   // C % 32 (0 = aligned; > 0 implies G == 1)

    // Accessors declared unconditionally, chained offsets (placeholders when absent).
    constexpr auto input_args = TensorAccessorArgs<9>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_input_tiles = 0;
    constexpr uint32_t cb_gamma_tiles = 1;
    constexpr uint32_t cb_beta_tiles = 2;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_mask_interior = 9;
    constexpr uint32_t cb_mask_tail = 10;

    // Scaler: non-standard value (1/sqrt(HW*Cg) combines the SCALAR double-apply
    // with the 1/N mean factor) -> prepare_reduce_scaler, pool-type-aware overload.
    const float inv_sqrt_n = __builtin_bit_cast(float, inv_sqrt_n_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_SCALAR>(
        inv_sqrt_n);

    // Tail masks (Refinement 2): one Wg-tile mask row per variant, generated once,
    // held by compute for the whole kernel (HeldBulk). Interior rows only need the
    // C-tail column mask in the last tile; the HW-tail row masks every tile of the
    // last tile row (corner tile combines both). c_tail > 0 implies G == 1, so the
    // last tile of the group slab IS the last tile of C.
    if constexpr (c_tail > 0) {
        const uint32_t mask_tile_bytes = get_tile_size(cb_mask_interior);
        cb_reserve_back(cb_mask_interior, Wg);
        uint32_t l1_addr = get_write_ptr(cb_mask_interior);
        for (uint32_t t = 0; t < Wg; ++t) {
            fill_mask_tile(l1_addr, 32, t + 1 == Wg ? c_tail : 32);
            l1_addr += mask_tile_bytes;
        }
        cb_push_back(cb_mask_interior, Wg);
    }
    if constexpr (hw_tail > 0) {
        const uint32_t mask_tile_bytes = get_tile_size(cb_mask_tail);
        cb_reserve_back(cb_mask_tail, Wg);
        uint32_t l1_addr = get_write_ptr(cb_mask_tail);
        for (uint32_t t = 0; t < Wg; ++t) {
            fill_mask_tile(l1_addr, hw_tail, (c_tail > 0 && t + 1 == Wg) ? c_tail : 32);
            l1_addr += mask_tile_bytes;
        }
        cb_push_back(cb_mask_tail, Wg);
    }

    const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);

    // gamma / beta: Wt tiles each, read once, pushed in bulk; never re-read.
    if constexpr (HAS_GAMMA) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
        const auto gamma = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
        cb_reserve_back(cb_gamma_tiles, Wt);
        uint32_t l1_addr = get_write_ptr(cb_gamma_tiles);
        for (uint32_t t = 0; t < Wt; ++t) {
            noc_async_read_tile(t, gamma, l1_addr);
            l1_addr += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_tiles, Wt);
    }
    if constexpr (HAS_BETA) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta_tiles);
        const auto beta = TensorAccessor(beta_args, beta_addr, beta_tile_bytes);
        cb_reserve_back(cb_beta_tiles, Wt);
        uint32_t l1_addr = get_write_ptr(cb_beta_tiles);
        for (uint32_t t = 0; t < Wt; ++t) {
            noc_async_read_tile(t, beta, l1_addr);
            l1_addr += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta_tiles, Wt);
    }

    // Group loop: 3 streaming passes over the Ht x Wg slab per group.
    // Batched per row chunk (Wg tiles, one barrier) — cb_input_tiles is sized
    // 2*Wg pages so chunk batching preserves the double-buffered overlap.
    // (n, g) tracked incrementally from start_group: one div/mod at startup.
    uint32_t n = start_group / G;
    uint32_t g = start_group % G;
    for (uint32_t i = 0; i < num_groups_here; ++i) {
        const uint32_t group_base = n * Ht * Wt + g * Wg;
        for (uint32_t pass = 0; pass < 3; ++pass) {
            for (uint32_t r = 0; r < Ht; ++r) {
                const uint32_t row_base = group_base + r * Wt;
                cb_reserve_back(cb_input_tiles, Wg);
                uint32_t l1_addr = get_write_ptr(cb_input_tiles);
                for (uint32_t c = 0; c < Wg; ++c) {
                    noc_async_read_tile(row_base + c, input, l1_addr);
                    l1_addr += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_tiles, Wg);
            }
        }
        if (++g == G) {
            g = 0;
            ++n;
        }
    }
}
