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

// Fill one bf16 tile with a 0/1 mask: 1.0 where (row < valid_rows && col in
// [col_start, col_end)), 0 elsewhere. Tile faces are sequential 16x16 blocks:
// f0 (r0-15, c0-15), f1 (r0-15, c16-31), f2 (r16-31, c0-15), f3 (r16-31,
// c16-31). bf16 pairs pack little-endian — the lower 16 bits of each u32 hold
// the even (first) column of the pair.
static void fill_mask_tile_range(uint32_t l1_addr, uint32_t valid_rows, uint32_t col_start, uint32_t col_end) {
    constexpr uint32_t ONE_LO = 0x00003F80u;  // [1.0, 0.0] (low half = even column)
    constexpr uint32_t ONE_HI = 0x3F800000u;  // [0.0, 1.0]
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t face_r = 0; face_r < 2; ++face_r) {
        const uint32_t fr = valid_rows > face_r * 16 ? (valid_rows - face_r * 16 > 16 ? 16 : valid_rows - face_r * 16)
                                                     : 0;  // valid rows in this face
        for (uint32_t face_c = 0; face_c < 2; ++face_c) {
            const uint32_t cbase = face_c * 16;
            for (uint32_t r = 0; r < 16; ++r) {
                const bool row_valid = r < fr;
                for (uint32_t pair = 0; pair < 8; ++pair) {
                    uint32_t v = 0;
                    if (row_valid) {
                        const uint32_t c_even = cbase + 2 * pair;
                        if (c_even >= col_start && c_even < col_end) {
                            v |= ONE_LO;
                        }
                        if (c_even + 1 >= col_start && c_even + 1 < col_end) {
                            v |= ONE_HI;
                        }
                    }
                    *p++ = v;
                }
            }
        }
    }
}

static void fill_mask_tile(uint32_t l1_addr, uint32_t valid_rows, uint32_t valid_cols) {
    fill_mask_tile_range(l1_addr, valid_rows, 0, valid_cols);
}

// Zero `n_bytes` (multiple of 4) at l1_addr.
static void zero_l1(uint32_t l1_addr, uint32_t n_bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < n_bytes / 4; ++i) {
        *p++ = 0;
    }
}

// Set row 0 of a (zeroed) tile to `val_bits` (f32 bits) over cols [col_start,
// col_end). Faces are sequential 16x16 blocks; row 0 lives in faces 0 (cols
// 0-15) and 1 (cols 16-31), at the start of each face. Only row 0 is written —
// the Row-broadcast consumer reads row 0 only, the rest stays zero.
template <bool stat_f32>
static void set_row0_range(uint32_t tile_addr, uint32_t col_start, uint32_t col_end, uint32_t val_bits) {
    for (uint32_t c = col_start; c < col_end; ++c) {
        const uint32_t face = c / 16;
        const uint32_t idx = c % 16;
        if constexpr (stat_f32) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr + face * 1024);
            p[idx] = val_bits;
        } else {
            // bf16: pairs pack little-endian, even column in the low half.
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr + face * 512);
            const uint32_t word = idx / 2;
            const uint32_t bf16 = val_bits >> 16;
            uint32_t v = p[word];
            v = (idx & 1) ? ((v & 0x0000FFFFu) | (bf16 << 16)) : ((v & 0xFFFF0000u) | bf16);
            p[word] = v;
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
    constexpr uint32_t c_tail = get_compile_time_arg_val(8);   // C % 32 (0 = aligned)
    // Cluster path (Refinement 3): Cg % 32 != 0 && G > 1 — work unit is one
    // (n, cluster) of lcm(Cg, 32) channels; group/tile boundaries align at
    // cluster edges.
    constexpr bool GROUPS_NA = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t CLUSTER_CH = get_compile_time_arg_val(10);  // lcm(Cg, 32)
    constexpr uint32_t WC_FULL = get_compile_time_arg_val(11);     // cluster_ch / 32 (cluster_t0 stride)
    constexpr uint32_t NUM_CLUSTERS = get_compile_time_arg_val(12);
    constexpr uint32_t Cg = get_compile_time_arg_val(13);
    constexpr uint32_t C = get_compile_time_arg_val(14);
    constexpr bool STAT_F32 = get_compile_time_arg_val(15) != 0;
    constexpr bool MASK_OUT = get_compile_time_arg_val(16) != 0;  // bf8b out + HW tail (cluster path)
    constexpr uint32_t WS_MAX = get_compile_time_arg_val(17);     // mask frame width (tiles)
    [[maybe_unused]] constexpr uint32_t CHUNK_ROWS = get_compile_time_arg_val(18);  // compute-side reduce block rows

    // Accessors declared unconditionally, chained offsets (placeholders when absent).
    constexpr auto input_args = TensorAccessorArgs<19>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_input_tiles = 0;
    constexpr uint32_t cb_gamma_tiles = 1;
    constexpr uint32_t cb_beta_tiles = 2;
    constexpr uint32_t cb_scaler = 8;
    constexpr uint32_t cb_mask_interior = 9;
    constexpr uint32_t cb_mask_tail = 10;
    constexpr uint32_t cb_mean_export = 11;
    constexpr uint32_t cb_rstd_export = 12;
    constexpr uint32_t cb_mean_row = 13;
    constexpr uint32_t cb_rstd_row = 14;
    constexpr uint32_t cb_mask_ones = 15;
    constexpr uint32_t cb_mask_rows = 17;

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
    if constexpr (!GROUPS_NA && c_tail > 0) {
        const uint32_t mask_tile_bytes = get_tile_size(cb_mask_interior);
        cb_reserve_back(cb_mask_interior, Wg);
        uint32_t l1_addr = get_write_ptr(cb_mask_interior);
        for (uint32_t t = 0; t < Wg; ++t) {
            fill_mask_tile(l1_addr, 32, t + 1 == Wg ? c_tail : 32);
            l1_addr += mask_tile_bytes;
        }
        cb_push_back(cb_mask_interior, Wg);
    }
    if constexpr (!GROUPS_NA && hw_tail > 0) {
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

    // Streams `width` tiles per row chunk, `Ht` rows, starting at tile column
    // `col0` of batch n. One barrier per chunk; cb_input_tiles is sized 2x the
    // widest chunk so batching preserves the double-buffered overlap.
    auto stream_rows = [&](uint32_t n, uint32_t col0, uint32_t width) {
        const uint32_t base = n * Ht * Wt + col0;
        for (uint32_t r = 0; r < Ht; ++r) {
            const uint32_t row_base = base + r * Wt;
            cb_reserve_back(cb_input_tiles, width);
            uint32_t l1_addr = get_write_ptr(cb_input_tiles);
            for (uint32_t c = 0; c < width; ++c) {
                noc_async_read_tile(row_base + c, input, l1_addr);
                l1_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, width);
        }
    };

    if constexpr (!GROUPS_NA) {
        // Group loop: 3 streaming passes over the Ht x Wg slab per group.
        // (n, g) tracked incrementally from start_group: one div/mod at startup.
        uint32_t n = start_group / G;
        uint32_t g = start_group % G;
        for (uint32_t i = 0; i < num_groups_here; ++i) {
            for (uint32_t pass = 0; pass < 3; ++pass) {
                stream_rows(n, g * Wg, Wg);
            }
            if (++g == G) {
                g = 0;
                ++n;
            }
        }
    } else {
        // Cluster loop (Refinement 3): work unit = (n, cluster). Per cluster:
        // per group, push the group's 0/1 column masks, stream passes 1 and 2
        // over the group's tile span, then collect the group's mean/rstd
        // scalars from compute and scatter them into per-column row-vector
        // tiles. After all groups: push the row vectors (+ optional bf8b
        // output masks) and stream pass 3 over the whole cluster.
        //
        // CB-wrap rule: multi-tile reserve/push frames must all be the same
        // width per CB, or a frame straddles the buffer end and writes run off
        // it. Input is therefore streamed per tile; masks push WS_MAX frames;
        // row vectors push WC_FULL frames (zero-padded for capped clusters).
        const uint32_t stat_tile_bytes = get_tile_size(cb_mean_row);
        const uint32_t mask_tile_bytes = get_tile_size(cb_mask_interior);

        auto stream_rows_per_tile = [&](uint32_t n, uint32_t col0, uint32_t width) {
            const uint32_t base = n * Ht * Wt + col0;
            for (uint32_t r = 0; r < Ht; ++r) {
                const uint32_t row_base = base + r * Wt;
                for (uint32_t c = 0; c < width; ++c) {
                    cb_reserve_back(cb_input_tiles, 1);
                    noc_async_read_tile(row_base + c, input, get_write_ptr(cb_input_tiles));
                    noc_async_read_barrier();
                    cb_push_back(cb_input_tiles, 1);
                }
            }
        };

        uint32_t n = start_group / NUM_CLUSTERS;
        uint32_t cl = start_group % NUM_CLUSTERS;
        for (uint32_t i = 0; i < num_groups_here; ++i) {
            const uint32_t cluster_c0 = cl * CLUSTER_CH;
            const uint32_t Ccl = (C - cluster_c0 < CLUSTER_CH) ? (C - cluster_c0) : CLUSTER_CH;
            const uint32_t Gc = Ccl / Cg;
            const uint32_t Wcu = (Ccl + 31) / 32;
            const uint32_t cluster_t0 = cl * WC_FULL;

            cb_reserve_back(cb_mean_row, WC_FULL);
            cb_reserve_back(cb_rstd_row, WC_FULL);
            const uint32_t mean_row_addr = get_write_ptr(cb_mean_row);
            const uint32_t rstd_row_addr = get_write_ptr(cb_rstd_row);
            zero_l1(mean_row_addr, WC_FULL * stat_tile_bytes);
            zero_l1(rstd_row_addr, WC_FULL * stat_tile_bytes);

            for (uint32_t g = 0; g < Gc; ++g) {
                const uint32_t c0 = g * Cg;  // cluster-relative channel range
                const uint32_t c1 = c0 + Cg;
                const uint32_t t0 = c0 / 32;  // cluster-relative tile span
                const uint32_t Wsg = (c1 - 1) / 32 - t0 + 1;

                // Group masks: interior (all rows) and HW-tail variant.
                // WS_MAX-wide frames; tiles beyond Wsg are left unfilled.
                cb_reserve_back(cb_mask_interior, WS_MAX);
                uint32_t m_addr = get_write_ptr(cb_mask_interior);
                for (uint32_t t = 0; t < Wsg; ++t) {
                    const uint32_t lo = c0 > (t0 + t) * 32 ? c0 - (t0 + t) * 32 : 0;
                    const uint32_t hi = c1 - (t0 + t) * 32 < 32 ? c1 - (t0 + t) * 32 : 32;
                    fill_mask_tile_range(m_addr, 32, lo, hi);
                    m_addr += mask_tile_bytes;
                }
                cb_push_back(cb_mask_interior, WS_MAX);
                if constexpr (hw_tail > 0) {
                    cb_reserve_back(cb_mask_tail, WS_MAX);
                    m_addr = get_write_ptr(cb_mask_tail);
                    for (uint32_t t = 0; t < Wsg; ++t) {
                        const uint32_t lo = c0 > (t0 + t) * 32 ? c0 - (t0 + t) * 32 : 0;
                        const uint32_t hi = c1 - (t0 + t) * 32 < 32 ? c1 - (t0 + t) * 32 : 32;
                        fill_mask_tile_range(m_addr, hw_tail, lo, hi);
                        m_addr += mask_tile_bytes;
                    }
                    cb_push_back(cb_mask_tail, WS_MAX);
                }

                stream_rows_per_tile(n, cluster_t0 + t0, Wsg);  // pass 1
                stream_rows_per_tile(n, cluster_t0 + t0, Wsg);  // pass 2

                // Collect this group's mean/rstd, scatter into row vectors.
                cb_wait_front(cb_mean_export, 1);
                uint32_t bits;
                if constexpr (STAT_F32) {
                    bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_mean_export));
                } else {
                    bits = uint32_t(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_mean_export)))
                           << 16;
                }
                cb_pop_front(cb_mean_export, 1);
                for (uint32_t t = t0; t * 32 < c1; ++t) {
                    const uint32_t lo = c0 > t * 32 ? c0 - t * 32 : 0;
                    const uint32_t hi = c1 - t * 32 < 32 ? c1 - t * 32 : 32;
                    set_row0_range<STAT_F32>(mean_row_addr + t * stat_tile_bytes, lo, hi, bits);
                }
                cb_wait_front(cb_rstd_export, 1);
                if constexpr (STAT_F32) {
                    bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_rstd_export));
                } else {
                    bits = uint32_t(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_rstd_export)))
                           << 16;
                }
                cb_pop_front(cb_rstd_export, 1);
                for (uint32_t t = t0; t * 32 < c1; ++t) {
                    const uint32_t lo = c0 > t * 32 ? c0 - t * 32 : 0;
                    const uint32_t hi = c1 - t * 32 < 32 ? c1 - t * 32 : 32;
                    set_row0_range<STAT_F32>(rstd_row_addr + t * stat_tile_bytes, lo, hi, bits);
                }
            }
            cb_push_back(cb_mean_row, WC_FULL);
            cb_push_back(cb_rstd_row, WC_FULL);

            if constexpr (MASK_OUT) {
                // Cluster-wide output row masks are column-independent — one
                // scalar tile per variant (all-ones interior; HW-tail rows).
                // Padding columns are already zeroed via rstd_row = 0.
                cb_reserve_back(cb_mask_ones, 1);
                fill_mask_tile_range(get_write_ptr(cb_mask_ones), 32, 0, 32);
                cb_push_back(cb_mask_ones, 1);
                cb_reserve_back(cb_mask_rows, 1);
                fill_mask_tile_range(get_write_ptr(cb_mask_rows), hw_tail, 0, 32);
                cb_push_back(cb_mask_rows, 1);
            }

            stream_rows_per_tile(n, cluster_t0, Wcu);  // pass 3
            if (++cl == NUM_CLUSTERS) {
                cl = 0;
                ++n;
            }
        }
    }
}
