// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm worker reader (NCRISC / NoC0). Block operations (op_design.md -> Block schedule):
//   constants            1.0 reduce scaler; LN: W-tail 0/1 mask (tail member), n_j partition-width tile
//   load_gamma_beta_slice  this chip's / member's columns of the full-W gamma (and beta) -> compact row-0 pages
//   load_x_block         crt x cw tiles of the own slice (one reserve / barrier / push per tile-row; W tail zeroed)
//   gather_partials_block  after both hubs multicast the stats table into this core's cb_stats_table and announce
//                        it complete: per tile-row, local NoC copies place each record as ROW j (= partition
//                        q*k + m) of a Float32 gather tile in cb_gather_rows (rows >= P stay 0); compute transposes
//                        them exactly in DEST
//
// Raw dataflow (documented): the record gather / placement are NoC copies no helper expresses (op_design.md
// "Helpers considered and rejected"; implementation notes: row-form gather).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "agrn_worker_args.hpp"

using namespace agrn;

namespace {

constexpr uint32_t x_elem_bytes = x_page_bytes / (TILE * TILE) >= 2 ? x_page_bytes / (TILE * TILE) : 1;

// fp32 bits of a small non-negative integer (exact below 2^24).
inline uint32_t u32_to_f32_bits(uint32_t v) {
    if (v == 0) {
        return 0;
    }
    uint32_t e = 31 - __builtin_clz(v);
    uint32_t mant = (e > 23) ? (v >> (e - 23)) : (v << (23 - e));
    return ((e + 127) << 23) | (mant & 0x7FFFFF);
}

// Word offset of element (row, col) in a 32x32 Float32 tile (4 faces of 16x16).
inline uint32_t f32_tile_word(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}

// Zero columns [v, 32) of a tile in L1 (v = valid columns, 1..31). Whole faces through the NoC zero engine,
// partial face rows by RISC-V stores (base op mechanism, rms_norm_reader.cpp zero_w_tail_of_tile).
template <uint32_t elem_bytes>
inline void zero_w_tail_of_tile(const Noc& noc, const CircularBuffer& cb, uint32_t tile_offset_bytes, uint32_t v) {
    constexpr uint32_t face_bytes = FACE * FACE * elem_bytes;
    constexpr uint32_t face_row_bytes = FACE * elem_bytes;
    const uint32_t tile_addr = cb.get_write_ptr() + tile_offset_bytes;
    if (v <= FACE) {
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = tile_offset_bytes + 3 * face_bytes});
        noc.write_zeros_l1_barrier();
        if (v < FACE) {
            for (uint32_t f = 0; f < 4; f += 2) {
                for (uint32_t row = 0; row < FACE; ++row) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                        tile_addr + f * face_bytes + row * face_row_bytes + v * elem_bytes, (FACE - v) * elem_bytes, 0);
                }
            }
        }
    } else {
        for (uint32_t f = 1; f < 4; f += 2) {
            for (uint32_t row = 0; row < FACE; ++row) {
                dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                    tile_addr + f * face_bytes + row * face_row_bytes + (v - FACE) * elem_bytes,
                    (TILE - v) * elem_bytes,
                    0);
            }
        }
    }
}

// load_gamma_beta_slice: the member's columns of the full-W weight into compact pages of `cb` (page j = local
// tile-column w_start + j; row 0 of face 0 at +0, row 0 of face 1 at +face_bytes; compute reads them
// ROW-broadcast). Global column of local column lc is p * W_local + lc (beta at its own global column).
// Source rows (a stick, or row 0 of a TILE weight tile, 32 datums) are read into a 64 B-aligned scratch with a
// 128 B stride, then copied element-wise (handles any W_local offset; aligned slices copy whole rows).
template <uint32_t cb, typename Acc>
inline void load_affine_slice(const Acc& acc, uint32_t p, const WorkerId& w) {
    constexpr uint32_t eb = gamma_elem_bytes;
    constexpr uint32_t g_face_bytes = FACE * FACE * eb;
    const uint32_t c0 = p * W_local + w.w_start * TILE;
    const uint32_t lc_end = ((w.w_start + w.cw) * TILE) < W_local ? (w.w_start + w.cw) * TILE : W_local;
    const uint32_t c_end = p * W_local + lc_end;  // exclusive
    const uint32_t s0 = c0 / TILE;
    const uint32_t s1 = (c_end - 1) / TILE;
    const uint32_t scratch = (get_write_ptr(cb_gamma_rows) + 63) & ~63u;
    for (uint32_t s = s0; s <= s1; ++s) {
        const uint32_t row = scratch + (s - s0) * 128;
        if constexpr (gamma_is_tile) {
            noc_async_read(acc.get_noc_addr(s, 0), row, FACE * eb);
            noc_async_read(acc.get_noc_addr(s, g_face_bytes), row + 64, FACE * eb);
        } else {
            noc_async_read(acc.get_noc_addr(s, 0), row, TILE * eb);
        }
    }
    noc_async_read_barrier();
    cb_reserve_back(cb, w.cw);
    const uint32_t base = get_write_ptr(cb);
    for (uint32_t j = 0; j < w.cw; ++j) {
        const uint32_t page = base + j * gamma_page_bytes;
        for (uint32_t e = 0; e < TILE; ++e) {
            const uint32_t lc = (w.w_start + j) * TILE + e;
            const uint32_t dst = page + (e < FACE ? e * eb : g_face_bytes + (e - FACE) * eb);
            uint32_t src = 0;
            if (lc < W_local) {
                const uint32_t c = p * W_local + lc;
                const uint32_t ee = c % TILE;
                const uint32_t row = scratch + (c / TILE - s0) * 128;
                if constexpr (gamma_is_tile) {
                    src = row + (ee < FACE ? ee * eb : 64 + (ee - FACE) * eb);
                } else {
                    src = row + ee * eb;
                }
            }
            if constexpr (eb == 4) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) =
                    src ? *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src) : 0u;
            } else {
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst) =
                    src ? *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src) : static_cast<uint16_t>(0);
            }
        }
    }
    cb_push_back(cb, w.cw);
}

// Fill gather tiles [0, n_tiles) of `cb` (Float32, zeroed) with n_j for partitions j < n_parts (all 32 rows).
template <typename WidthFn>
inline void fill_partition_widths(const Noc& noc, uint32_t cb, uint32_t n_tiles, uint32_t n_parts, WidthFn width) {
    CircularBuffer cb_consts(cb);
    cb_reserve_back(cb, n_tiles);
    for (uint32_t t = 0; t < n_tiles; ++t) {
        noc.async_write_zeros(cb_consts, TILE * TILE * 4, {.offset_bytes = t * TILE * TILE * 4});
    }
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
    for (uint32_t j = 0; j < n_parts; ++j) {
        const uint32_t bits = u32_to_f32_bits(width(j));
        for (uint32_t r = 0; r < TILE; ++r) {
            t[(j / TILE) * TILE * TILE + f32_tile_word(r, j % TILE)] = bits;
        }
    }
    cb_push_back(cb, n_tiles);
}

// Valid columns of member m's slice (member widths; the slice's W tail sits in the last member).
inline uint32_t member_valid_cols(uint32_t m) {
    const uint32_t ws = m * cw_nom;
    const uint32_t wtiles = (Wlt - ws) < cw_nom ? (Wlt - ws) : cw_nom;
    uint32_t n = wtiles * TILE;
    if (ws + wtiles == Wlt) {
        n -= TILE - w_valid_last;
    }
    return n;
}

// Place M records (one half h of one (row, stat) block: M consecutive 64 B half-records at L1 `src`) as ROWS
// j0 .. j0 + M - 1 of the gather tiles at `gtiles` (row j -> tile j / 32, face (j % 32) / 16 * 2 + h, row j % 16):
// one local NoC copy per run that stays inside a 16-row face.
inline void place_records(uint32_t src, uint32_t gtiles, uint32_t j0, uint32_t M, uint32_t h) {
    constexpr uint32_t f32_tile_bytes = TILE * TILE * 4;
    uint32_t m0 = 0;
    while (m0 < M) {
        const uint32_t j = j0 + m0;
        const uint32_t jr = j & 31;
        const uint32_t room = FACE - (jr & 15);
        const uint32_t n = (M - m0) < room ? (M - m0) : room;
        const uint32_t dst = gtiles + (j >> 5) * f32_tile_bytes + ((jr >> 4) * 512 + (jr & 15) * FACE + h * 256) * 4;
        noc_async_read(get_noc_addr(src + m0 * (RECORD_BYTES / 2)), dst, n * (RECORD_BYTES / 2));
        m0 += n;
    }
}

// Reserve and zero n Float32 gather tiles of cb_gather_rows (rows >= P must read 0); returns their base.
inline uint32_t reserve_zeroed_gather_tiles(const Noc& noc, uint32_t n) {
    constexpr uint32_t f32_tile_bytes = TILE * TILE * 4;
    CircularBuffer cb_g(cb_gather_rows);
    cb_reserve_back(cb_gather_rows, n);
    for (uint32_t t = 0; t < n; ++t) {
        noc.async_write_zeros(cb_g, f32_tile_bytes, {.offset_bytes = t * f32_tile_bytes});
    }
    noc.write_zeros_l1_barrier();
    return get_write_ptr(cb_gather_rows);
}

}  // namespace

void kernel_main() {
    const WorkerId w = my_worker_id();
    if (!w.active) {
        return;
    }
    constexpr auto x_args = TensorAccessorArgs<WORKER_CT_ARGS>();
    constexpr auto dst_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t p = get_common_arg_val<uint32_t>(CRT_P);
    const bool tail = member_owns_tail(w);

    Noc noc;
    // ---- load_x_block ----
    if constexpr (x_sharded) {
        // zero-copy: cb_x is backed on the resident shard (crt x cw tiles, row pitch cw); announce it in place. The
        // W tail of the caller's shard is not zeroed: compute masks it (cb_mask, below).
        // (WIDTH: row pitch = the shard width cw_nom; a ragged last member has cw < cw_nom valid tiles per row)
        const uint32_t x_tiles = w.crt * (pre_combine ? cw_nom : w.cw);
        cb_reserve_back(cb_x, x_tiles);
        cb_push_back(cb_x, x_tiles);
        if constexpr (pre_combine) {
            cb_reserve_back(cb_x_stats, x_tiles);  // the same shard, the local-stats pass's own view
            cb_push_back(cb_x_stats, x_tiles);
        }
    } else {
        AGRN_ZONE("rd_x");
        const auto x_acc = TensorAccessor(x_args, get_common_arg_val<uint32_t>(CRT_X_ADDR), x_page_bytes);
        CircularBuffer cb(cb_x);
        for (uint32_t r = 0; r < w.crt; ++r) {
            cb_reserve_back(cb_x, w.cw);
            const uint32_t base = get_write_ptr(cb_x);
            const uint32_t first = (w.row_start + r) * Wlt + w.w_start;
            for (uint32_t j = 0; j < w.cw; ++j) {
                noc_async_read(x_acc.get_noc_addr(first + j), base + j * x_page_bytes, x_page_bytes);
            }
            noc_async_read_barrier();
            if (tail) {
                zero_w_tail_of_tile<x_elem_bytes>(noc, cb, (w.cw - 1) * x_page_bytes, w_valid_last);
            }
            cb_push_back(cb_x, w.cw);
        }
    }

    // ---- constants ----
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    if constexpr (x_sharded && w_valid_last < TILE) {
        // sharded input, tail member: the mask BLOCK (one row-0 mask per tile-column: ones, then the tail tile's
        // valid columns) every sum over x multiplies by (compute) — the caller's shard keeps its padding
        if (tail) {
            for (uint32_t j = 0; j + 1 < w.cw; ++j) {
                dataflow_kernel_lib::prepare_reduce_mask<cb_mask, ckernel::ReduceDim::REDUCE_ROW>(TILE);
            }
            dataflow_kernel_lib::prepare_reduce_mask<cb_mask, ckernel::ReduceDim::REDUCE_ROW>(w_valid_last);
        }
    } else if constexpr (is_ln) {
        // LN T: the 0/1 mask of the W-tail tile (AccumulateViaAdd only_partial)
        if (tail) {
            dataflow_kernel_lib::prepare_reduce_mask<cb_mask, ckernel::ReduceDim::REDUCE_ROW>(w_valid_last);
        }
    }
    if constexpr (is_ln) {
        if constexpr (pre_combine) {
            // on-chip pre-combine: partition j = member j; cross-chip: partition q = chip q (every chip: W_local)
            fill_partition_widths(noc, cb_stat_consts, gt_pc, K, [](uint32_t j) { return member_valid_cols(j); });
            fill_partition_widths(noc, cb_stat_consts_x, gather_tiles, P, [](uint32_t) { return W_local; });
        } else {
            // n_j: valid columns of partition j = q*K + m (member m's width; every chip has the same W_local)
            fill_partition_widths(
                noc, cb_stat_consts, gather_tiles, P, [](uint32_t j) { return member_valid_cols(j % K); });
        }
    }

    constexpr uint32_t f32_tile_bytes = TILE * TILE * 4;
    // ---- zero the gather tiles once, off the critical path (rows >= P must read 0) ----
    // (pre_combine: the gather CB holds one row chunk and is reused, so each chunk zeroes its own tiles)
    uint32_t gather_base = 0;
    if constexpr (!pre_combine) {
        gather_base = reserve_zeroed_gather_tiles(noc, w.crt * nstats * gather_tiles);
    }

    // ---- load_gamma_beta_slice ----
    {
        AGRN_ZONE("rd_gamma");
        if constexpr (has_gamma) {
            const auto g_acc =
                TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(CRT_GAMMA_ADDR), gamma_src_page_bytes);
            load_affine_slice<cb_gamma>(g_acc, p, w);
        }
        if constexpr (has_beta) {
            const auto b_acc =
                TensorAccessor(beta_args, get_common_arg_val<uint32_t>(CRT_BETA_ADDR), gamma_src_page_bytes);
            load_affine_slice<cb_beta>(b_acc, p, w);
        }
    }

    const uint32_t table = get_write_ptr(cb_stats_table);           // same address on every core of every chip
    constexpr uint32_t row_rec_bytes = KT * nstats * RECORD_BYTES;  // one tile-row of one slot
    constexpr uint32_t slot_bytes = R * row_rec_bytes;
    // Each 128 B record (32 per-row values of one partition j = q * KT + m and one stat s) becomes ROW j of gather
    // tile (s, j / 32): values 0..15 -> face (j%32)/16*2, row j%16; values 16..31 -> the next face. The writers store
    // a slot's members half-split ([values 0..15 of m = 0..KT-1][values 16..31 ...]), so the KT rows of one slot land
    // with one contiguous NoC read per half (split only where they cross a 16-row face boundary): no RISC-V copy, few
    // large requests to the hubs. Compute transposes the tiles (exact, in DEST).
    // gather_partials (rows [tr0, tr0 + n) into the gather tiles at gbase0): the hubs multicast every slot of the
    // table into this core's own cb_stats_table before stats-ready (Refinement 3), so the records are placed straight
    // from local L1 (no request to the hubs, no landing copy).
    auto gather_partials = [&](uint32_t tr0, uint32_t n, uint32_t gbase0) {
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t tr = tr0 + r;
            const uint32_t gbase = gbase0 + r * nstats * gather_tiles * f32_tile_bytes;
            for (uint32_t q = 0; q < G; ++q) {
                for (uint32_t s = 0; s < nstats; ++s) {
                    for (uint32_t h = 0; h < 2; ++h) {
                        place_records(
                            table + q * slot_bytes + tr * row_rec_bytes + ((s * 2 + h) * KT) * (RECORD_BYTES / 2),
                            gbase + s * gather_tiles * f32_tile_bytes,
                            q * KT,
                            KT,
                            h);
                    }
                }
            }
            noc_async_read_barrier();
        }
    };
    auto wait_stats_ready = [&]() {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_ready));
        AGRN_ZONE("rd_wait_stats");
        volatile tt_l1_ptr uint32_t* ready_bwd =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_ready_bwd));
        noc_semaphore_wait_min(ready, 1);  // both hubs: own slot + every remote slot of the table landed
        noc_semaphore_wait_min(ready_bwd, 1);
    };

    if constexpr (!pre_combine) {
        // ---- gather_partials_block ----
        wait_stats_ready();
        AGRN_ZONE("rd_gather");
        for (uint32_t r = 0; r < w.crt; ++r) {
            gather_partials(w.row_start + r, 1, gather_base + r * nstats * gather_tiles * f32_tile_bytes);
            cb_push_back(cb_gather_rows, nstats * gather_tiles);
        }
    } else {
        // ---- row owner: pre-combine gather, then the cross-chip gather of its rows ----
        const uint32_t orows = owner_rows(w.m);
        const uint32_t orow0 = w.m * rows_per_owner;
        if (orows > 0) {
            volatile tt_l1_ptr uint32_t* pre = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_pre));
            noc_semaphore_wait_min(pre, K);  // every member's records of my rows landed in my pre table
            const uint32_t pre_table = get_write_ptr(cb_pre_table);
            for (uint32_t r0 = 0; r0 < orows; r0 += row_chunk) {
                const uint32_t n = min_u32(row_chunk, orows - r0);
                const uint32_t gbase0 = reserve_zeroed_gather_tiles(noc, n * nstats * gt_pc);
                for (uint32_t r = 0; r < n; ++r) {
                    const uint32_t gbase = gbase0 + r * nstats * gt_pc * f32_tile_bytes;
                    for (uint32_t s = 0; s < nstats; ++s) {
                        for (uint32_t h = 0; h < 2; ++h) {
                            place_records(
                                pre_table + (((r0 + r) * nstats + s) * 2 + h) * K * (RECORD_BYTES / 2),
                                gbase + s * gt_pc * f32_tile_bytes,
                                0,
                                K,
                                h);
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_gather_rows, n * nstats * gt_pc);
            }
            wait_stats_ready();
            for (uint32_t r0 = 0; r0 < orows; r0 += row_chunk) {
                const uint32_t n = min_u32(row_chunk, orows - r0);
                const uint32_t gbase0 = reserve_zeroed_gather_tiles(noc, n * nstats * gather_tiles);
                gather_partials(orow0 + r0, n, gbase0);
                cb_push_back(cb_gather_rows, n * nstats * gather_tiles);
            }
        }
        // ---- every member: the final per-row stats of all its rows, one row chunk at a time (column 0) ----
        volatile tt_l1_ptr uint32_t* fin = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_final));
        noc_semaphore_wait_min(fin, n_owners);
        const uint32_t final_table = get_write_ptr(cb_final_table);
        auto to_col0 = [&](uint32_t cb, uint32_t tr0, uint32_t n, uint32_t s) {
            cb_reserve_back(cb, n);
            const uint32_t base = get_write_ptr(cb);
            for (uint32_t r = 0; r < n; ++r) {
                volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    final_table + ((tr0 + r) * nfinal + s) * RECORD_BYTES);
                volatile tt_l1_ptr uint32_t* t =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + r * f32_tile_bytes);
                for (uint32_t i = 0; i < TILE; ++i) {
                    t[f32_tile_word(i, 0)] = rec[i];
                }
            }
            cb_push_back(cb, n);
        };
        for (uint32_t r0 = 0; r0 < w.crt; r0 += row_chunk) {
            const uint32_t n = min_u32(row_chunk, w.crt - r0);
            if constexpr (is_ln) {
                to_col0(cb_final_mean, r0, n, 0);
                to_col0(cb_final_inv, r0, n, 1);
            } else {
                to_col0(cb_final_inv, r0, n, 0);
            }
        }
    }
}
