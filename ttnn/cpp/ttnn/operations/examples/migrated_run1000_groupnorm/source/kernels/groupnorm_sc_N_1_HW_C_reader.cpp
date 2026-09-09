// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C reader — regime `cluster_parallel_two_pass`.
//
// Responsibilities (see op_design.md):
//   1. build_mask_tiles()  — once per kernel. One bf16 tile per (group, channel-tile)
//      pair inside a cluster, carrying 1.0 on the channel lanes that belong to the
//      group and 0.0 elsewhere, ROW-REPLICATED down all 32 rows so every consumer
//      multiply is BroadcastDim::None. This is the whole partial-channel mechanism:
//      no host mask tensor, built purely from Cg / groups_per_cluster CT args.
//   2. the reduce scaler tile (all-1.0, REDUCE_SCALAR) — once per kernel.
//   3. per work unit: gamma/beta ingest, expanded to fp32 row-replicated tiles.
//   4. per work unit: the input block stream, TWICE (moment pass, then apply pass).
//      A unit is (batch n, cluster k); the HW axis is walked in blocks of
//      BLOCK_HW_TILES — the live block knob shared with compute and the writer.
//
// PERF LAMP (Phase 0): the mask tiles and the affine tiles are patterned constants
// with no kernel_lib generator, so their non-zero lanes are hand-written with RISC
// stores after a NoC zero-fill (per the "patterned constant with no helper" rule).
// The mask build is once per kernel; the affine expansion is once per work unit and
// is a candidate for hoisting (it is cluster-invariant across batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "../dependencies/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_mask_tiles = 2;
constexpr uint32_t cb_scaler_ones = 3;
constexpr uint32_t cb_gamma_tiles = 10;
constexpr uint32_t cb_beta_tiles = 11;
constexpr uint32_t cb_rm_sticks = 12;
constexpr uint32_t cb_affine_scratch = 15;
constexpr uint32_t cb_row_mask = 16;

// --- geometry (shared geom_ct prefix) ---------------------------------------
constexpr uint32_t tensor_hw_tiles = get_compile_time_arg_val(0);
constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(1);
constexpr uint32_t cluster_c_tiles = get_compile_time_arg_val(2);
constexpr uint32_t groups_per_cluster = get_compile_time_arg_val(3);
constexpr uint32_t num_clusters = get_compile_time_arg_val(4);
constexpr uint32_t BLOCK_HW_TILES = get_compile_time_arg_val(5);
constexpr uint32_t num_hw_blocks = get_compile_time_arg_val(6);
constexpr uint32_t Cg = get_compile_time_arg_val(7);
constexpr uint32_t cluster_channels = get_compile_time_arg_val(8);
constexpr uint32_t num_mask_tiles = get_compile_time_arg_val(9);
// (10) max_span — compute-side only.
constexpr uint32_t hw_tail = get_compile_time_arg_val(11);
constexpr uint32_t c_tail = get_compile_time_arg_val(12);

// (13) regime_id, (14) hw_split_factor — the reader needs neither: its HW slice
// arrives as the `hw_tile_start` runtime arg and its extent is num_hw_blocks.

// Refinement 4 lever 1 — residency fast path. When set, cb_input_tiles holds the
// core's WHOLE HW assignment, so the input is streamed ONCE per unit instead of
// once per pass: one DRAM crossing instead of two. Host-predicated on L1
// capacity; 0 keeps the unchanged streaming schedule.
constexpr uint32_t resident_input = get_compile_time_arg_val(15);

constexpr uint32_t has_gamma = get_compile_time_arg_val(16);
constexpr uint32_t has_beta = get_compile_time_arg_val(17);
constexpr uint32_t input_is_rm = get_compile_time_arg_val(18);
constexpr uint32_t in_elem = get_compile_time_arg_val(19);
constexpr uint32_t affine_is_rm = get_compile_time_arg_val(20);
constexpr uint32_t affine_elem = get_compile_time_arg_val(21);
constexpr uint32_t C_channels = get_compile_time_arg_val(22);
constexpr uint32_t HW_rows = get_compile_time_arg_val(23);
// Bytes of ONE affine page (a whole (1,1,1,C) RM row, or one tile page).
// Host-derived so bfp8_b's 1088-byte tile is never guessed as 32*32*elem.
constexpr uint32_t affine_page_bytes = get_compile_time_arg_val(24);

constexpr auto input_args = TensorAccessorArgs<25>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
[[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_HW = 32;
constexpr uint16_t BF16_ONE = 0x3F80;

// Element index of (row, col) inside a 32x32 tile laid out as 4 16x16 faces.
FORCE_INLINE uint32_t tile_elem_index(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >> 4) << 1) | (c >> 4);
    return (face << 8) + ((r & 15) << 4) + (c & 15);
}

FORCE_INLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }
FORCE_INLINE uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }

// ---------------------------------------------------------------------------
// 1. Mask tiles
// ---------------------------------------------------------------------------
void build_mask_tiles() {
    cb_reserve_back(cb_mask_tiles, num_mask_tiles);

    const uint32_t tile_bytes = get_tile_size(cb_mask_tiles);
    {
        Noc noc;
        CircularBuffer cb_obj(cb_mask_tiles);
        noc.async_write_zeros(cb_obj, num_mask_tiles * tile_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
    }

    volatile tt_l1_ptr uint16_t* base = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_mask_tiles));
    const uint32_t tile_stride = tile_bytes >> 1;  // in uint16 elements

    uint32_t mi = 0;
    for (uint32_t j = 0; j < groups_per_cluster; ++j) {
        const uint32_t lo = j * Cg;
        const uint32_t hi = lo + Cg;
        const uint32_t t0 = lo / TILE_HW;
        const uint32_t t1 = (hi - 1) / TILE_HW;
        for (uint32_t t = t0; t <= t1; ++t) {
            volatile tt_l1_ptr uint16_t* tile = base + mi * tile_stride;
            const uint32_t tbase = t * TILE_HW;
            const uint32_t c_lo = umax(lo, tbase) - tbase;
            const uint32_t c_hi = umin(hi, tbase + TILE_HW) - tbase;
            for (uint32_t c = c_lo; c < c_hi; ++c) {
                for (uint32_t r = 0; r < TILE_HW; ++r) {
                    tile[tile_elem_index(r, c)] = BF16_ONE;
                }
            }
            ++mi;
        }
    }
    cb_push_back(cb_mask_tiles, num_mask_tiles);
}

// ---------------------------------------------------------------------------
// 1b. HW row-tail mask (Refinement 1 — `hw_non_aligned`)
// ---------------------------------------------------------------------------
// One bf16 tile carrying 1.0 on the rows of the LAST HW tile that are real
// spatial positions (`row < hw_tail`) and 0.0 on the padding rows, replicated
// across all 32 channel lanes. Built once per kernel and never popped; the
// compute kernel multiplies the last HW tile of the moment pass by it so the
// padding rows contribute nothing to Sigma-x / Sigma-x^2. `n_g = Cg * HW`
// already uses the TRUE row count, so nothing else changes.
//
// hw_tail == 0 => the axis is tile-aligned and this whole path is compiled out.
void build_row_mask() {
    if constexpr (hw_tail != 0) {
        cb_reserve_back(cb_row_mask, 1);
        const uint32_t tile_bytes = get_tile_size(cb_row_mask);
        {
            Noc noc;
            CircularBuffer cb_obj(cb_row_mask);
            noc.async_write_zeros(cb_obj, tile_bytes, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        }
        volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_row_mask));
        for (uint32_t r = 0; r < hw_tail; ++r) {
            for (uint32_t c = 0; c < TILE_HW; ++c) {
                tile[tile_elem_index(r, c)] = BF16_ONE;
            }
        }
        cb_push_back(cb_row_mask, 1);
    }
}

// ---------------------------------------------------------------------------
// 3. Affine ingest
// ---------------------------------------------------------------------------

// Row-replicate one fp32 channel value down all 32 rows of column `c` of `tile`.
FORCE_INLINE void splat_channel(volatile tt_l1_ptr uint32_t* tile, uint32_t c, uint32_t bits) {
    for (uint32_t r = 0; r < TILE_HW; ++r) {
        tile[tile_elem_index(r, c)] = bits;
    }
}

void fill_affine_const(uint32_t cb_dst, uint32_t bits) {
    cb_reserve_back(cb_dst, cluster_c_tiles);
    volatile tt_l1_ptr uint32_t* base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_dst));
    const uint32_t tile_stride = get_tile_size(cb_dst) >> 2;
    for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
        volatile tt_l1_ptr uint32_t* tile = base + t * tile_stride;
        for (uint32_t c = 0; c < TILE_HW; ++c) {
            splat_channel(tile, c, bits);
        }
    }
    cb_push_back(cb_dst, cluster_c_tiles);
}

// --- Refinement 2: bfloat8_b affine ----------------------------------------
// `affine_elem == 1` is the host's sentinel for "this operand is bfp8_b".
// A bfp8_b tile is 64 bytes of shared exponents (16 per 16x16 face, one per
// face-row, faces in order 0..3) followed by 4 x 256 bytes of sign+mantissa,
// face-major then row-major. Ported from the reference decode in
// tt_metal/impl/data_format/blockfloat_common.cpp (convert_bfp_to_u32,
// Bfp8_b branch): the stored 7-bit magnitude carries an explicit leading 1,
// so it is normalized back into an IEEE fp32 hidden-bit mantissa.
constexpr uint32_t BFP8_EXP_SECTION_BYTES = 64;
constexpr uint32_t BFP8_FACE_DATA_BYTES = 256;

FORCE_INLINE uint32_t bfp8_to_f32_bits(uint32_t shared_exp, uint32_t data) {
    const uint32_t sign = data >> 7;
    uint32_t man = data & 0x7F;
    if (man == 0) {
        return sign << 31;  // +/- 0.0
    }
    uint32_t shift_cnt = 0;
    while ((man & 0x40) == 0) {
        man <<= 1;
        ++shift_cnt;
    }
    // Shift once more to drop the (now explicit) leading 1 -> hidden bit.
    man = (man << 1) & 0x7F;
    if (shared_exp <= shift_cnt) {
        return sign << 31;  // underflow to zero rather than wrapping the exponent
    }
    const uint32_t exp = shared_exp - shift_cnt;
    return (sign << 31) | (exp << 23) | (man << 16);
}

// fp32 bits of the element at (row 0, channel `c`) of a bfp8_b tile page.
FORCE_INLINE uint32_t bfp8_row0_bits(uint32_t scratch_addr, uint32_t c) {
    volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch_addr);
    const uint32_t face = c >> 4;  // row 0 lives in faces 0 (cols 0-15) and 1 (cols 16-31)
    const uint32_t col = c & 15;
    const uint32_t shared_exp = p[face * 16];  // face-row 0
    const uint32_t datum = p[BFP8_EXP_SECTION_BYTES + face * BFP8_FACE_DATA_BYTES + col];
    return bfp8_to_f32_bits(shared_exp, datum);
}

// Widen one source element (bf16 or fp32) at index `i` of the scratch to fp32 bits.
FORCE_INLINE uint32_t affine_bits_at(uint32_t scratch_addr, uint32_t i) {
    if constexpr (affine_elem == 2) {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_addr);
        return static_cast<uint32_t>(p[i]) << 16;
    } else {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
        return p[i];
    }
}

template <typename Acc>
void load_affine_tiles(uint32_t cb_dst, const Acc& acc, uint32_t cluster_c0_tile, uint32_t default_bits) {
    cb_reserve_back(cb_dst, cluster_c_tiles);
    volatile tt_l1_ptr uint32_t* base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_dst));
    const uint32_t tile_stride = get_tile_size(cb_dst) >> 2;
    const uint32_t scratch = get_write_ptr(cb_affine_scratch);

    if constexpr (affine_is_rm == 1) {
        // (1,1,1,C) row-major: one page of C elements. Read only this cluster's
        // slice; the byte offset is a multiple of 32*affine_elem so it is aligned.
        noc_async_read(
            acc.get_noc_addr(0, cluster_c0_tile * TILE_HW * affine_elem), scratch, cluster_channels * affine_elem);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
            volatile tt_l1_ptr uint32_t* tile = base + t * tile_stride;
            for (uint32_t c = 0; c < TILE_HW; ++c) {
                // Refinement 1: with C % 32 != 0 the trailing channel tile has
                // padding lanes with no gamma/beta element behind them. Splat
                // the identity there instead of reading past the (1,1,1,C) row.
                const uint32_t ch = (cluster_c0_tile + t) * TILE_HW + c;
                splat_channel(tile, c, (ch < C_channels) ? affine_bits_at(scratch, t * TILE_HW + c) : default_bits);
            }
        }
    } else {
        // (1,1,1,C) tiled: one tile page per channel-tile; the channels live in
        // row 0, i.e. face-0 elements 0..15 and face-1 elements 256..271.
        for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
            noc_async_read_page(cluster_c0_tile + t, acc, scratch);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* tile = base + t * tile_stride;
            for (uint32_t c = 0; c < TILE_HW; ++c) {
                const uint32_t ch = (cluster_c0_tile + t) * TILE_HW + c;
                uint32_t bits = default_bits;
                if (ch < C_channels) {
                    if constexpr (affine_elem == 1) {
                        bits = bfp8_row0_bits(scratch, c);  // Refinement 2: bfloat8_b affine
                    } else {
                        bits = affine_bits_at(scratch, tile_elem_index(0, c));
                    }
                }
                splat_channel(tile, c, bits);
            }
        }
    }
    cb_push_back(cb_dst, cluster_c_tiles);
}

// ---------------------------------------------------------------------------
// 4. Input block stream
// ---------------------------------------------------------------------------

template <typename Acc>
void stream_input_pass(const Acc& in, uint32_t n, uint32_t cluster_c0_tile, uint32_t hw_tile_start) {
    for (uint32_t b = 0; b < num_hw_blocks; ++b) {
        // `hw_tile_start` is 0 in REGIME_CLUSTER_PARALLEL and this core's slice
        // origin in REGIME_HW_SPLIT. The host snaps BLOCK_HW_TILES to a divisor
        // of the per-core extent, so every block is exactly BLOCK_HW_TILES tall.
        const uint32_t r0 = hw_tile_start + b * BLOCK_HW_TILES;
        const uint32_t rows_this = BLOCK_HW_TILES;

        if constexpr (input_is_rm == 1) {
            // Channel-major sub-sticks: for each channel-tile t, stream every
            // 32-channel row slice of the block so compute can tilize it into
            // `rows_this` contiguous tiles.
            // One reserve / barrier / push per (channel-tile, HW block): the whole
            // block's sticks are issued back-to-back so `32 * rows_this` reads are
            // in flight per barrier instead of 32. `tilize` consumes them 32 at a
            // time regardless, and the CB capacity (RM_DEPTH * 32 * BLOCK_HW_TILES)
            // is an exact multiple of this push size, so no push wraps mid-block.
            const uint32_t stick_bytes = TILE_HW * in_elem;
            for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
                const uint32_t c0 = (cluster_c0_tile + t) * TILE_HW;
                const uint32_t byte_off = c0 * in_elem;
                // Refinement 1 (`c_non_aligned`): the trailing channel tile is
                // partial, so only `valid_ch` elements of each stick exist in
                // the (N,1,HW,C) row-major page. Reading a full 32-channel
                // stick there would run off the end of the last page.
                const uint32_t valid_ch = (c0 + TILE_HW <= C_channels) ? TILE_HW : (C_channels - c0);
                const uint32_t read_bytes = valid_ch * in_elem;
                const uint32_t block_sticks = TILE_HW * rows_this;
                cb_reserve_back(cb_rm_sticks, block_sticks);
                uint32_t l1 = get_write_ptr(cb_rm_sticks);
                for (uint32_t r = 0; r < rows_this; ++r) {
                    for (uint32_t row = 0; row < TILE_HW; ++row) {
                        // Refinement 1 (`hw_non_aligned`): the last HW tile's
                        // padding rows have no page; re-read the last REAL row
                        // so the lanes hold finite data (the compute-side row
                        // mask zeroes their contribution to the moments).
                        const uint32_t abs_row = (r0 + r) * TILE_HW + row;
                        const uint32_t src_row = (abs_row < HW_rows) ? abs_row : (HW_rows - 1);
                        noc_async_read(in.get_noc_addr(n * HW_rows + src_row, byte_off), l1, read_bytes);
                        l1 += stick_bytes;
                    }
                }
                noc_async_read_barrier();
                if constexpr (c_tail != 0) {
                    // Zero the padding lanes of the partial trailing stick so
                    // they are finite; the group masks then zero them out of
                    // every reduction (a stale NaN would survive `x * 0`).
                    if (valid_ch != TILE_HW) {
                        uint32_t p = get_write_ptr(cb_rm_sticks);
                        for (uint32_t s = 0; s < block_sticks; ++s) {
                            if constexpr (in_elem == 2) {
                                volatile tt_l1_ptr uint16_t* q = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(p);
                                for (uint32_t c = valid_ch; c < TILE_HW; ++c) {
                                    q[c] = 0;
                                }
                            } else {
                                volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p);
                                for (uint32_t c = valid_ch; c < TILE_HW; ++c) {
                                    q[c] = 0;
                                }
                            }
                            p += stick_bytes;
                        }
                    }
                }
                cb_push_back(cb_rm_sticks, block_sticks);
            }
        } else {
            const uint32_t tile_bytes = get_tile_size(cb_input_tiles);
            for (uint32_t t = 0; t < cluster_c_tiles; ++t) {
                cb_reserve_back(cb_input_tiles, rows_this);
                uint32_t l1 = get_write_ptr(cb_input_tiles);
                for (uint32_t r = 0; r < rows_this; ++r) {
                    const uint32_t page = (n * tensor_hw_tiles + r0 + r) * tensor_c_tiles + cluster_c0_tile + t;
                    noc_async_read_page(page, in, l1);
                    l1 += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_tiles, rows_this);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t unit_start = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);
    const uint32_t hw_tile_start = get_arg_val<uint32_t>(5);

    build_mask_tiles();
    build_row_mask();
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler_ones,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_SCALAR>();

    const uint32_t input_page_bytes = (input_is_rm == 1) ? (C_channels * in_elem) : get_tile_size(cb_input_tiles);
    const auto in = TensorAccessor(input_args, in_addr, input_page_bytes);

    // Refinement 4 lever 2 — the gamma/beta tiles depend only on the CLUSTER,
    // not on the batch index, so they are ingested once per distinct cluster a
    // core owns instead of once per (n, cluster) unit. The compute kernel
    // derives the identical predicate from the same `unit % num_clusters`, so
    // the wait/pop counts stay matched with no extra plumbing.
    uint32_t prev_k = 0xFFFFFFFFu;

    for (uint32_t u = 0; u < units; ++u) {
        const uint32_t unit = unit_start + u;
        const uint32_t n = unit / num_clusters;
        const uint32_t k = unit % num_clusters;
        const uint32_t cluster_c0_tile = k * cluster_c_tiles;

        // --- affine (cluster-invariant; hoisted out of the batch loop) ------
        if (k != prev_k) {
            if constexpr (has_gamma == 1) {
                const auto g = TensorAccessor(gamma_args, gamma_addr, affine_page_bytes);
                load_affine_tiles(cb_gamma_tiles, g, cluster_c0_tile, 0x3F800000u);
            } else {
                fill_affine_const(cb_gamma_tiles, 0x3F800000u);  // 1.0f
            }
            if constexpr (has_beta == 1) {
                const auto bt = TensorAccessor(beta_args, beta_addr, affine_page_bytes);
                load_affine_tiles(cb_beta_tiles, bt, cluster_c0_tile, 0u);
            } else {
                fill_affine_const(cb_beta_tiles, 0u);  // 0.0f
            }
            prev_k = k;
        }

        // --- input stream ---------------------------------------------------
        // Streaming path: once for the moment pass, once for the apply pass.
        // Residency path: ONCE — the whole assignment stays in cb_input_tiles
        // and the compute kernel's moment pass reads it without popping.
        stream_input_pass(in, n, cluster_c0_tile, hw_tile_start);
        if constexpr (resident_input == 0) {
            stream_input_pass(in, n, cluster_c0_tile, hw_tile_start);
        }
    }
}
