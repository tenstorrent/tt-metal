// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm worker reader (NCRISC / NoC0). Block operations (op_design.md -> Block schedule):
//   constants            1.0 reduce scaler; LN: W-tail 0/1 mask (tail member), n_j partition-width tile
//   load_gamma_beta_slice  this chip's / member's columns of the full-W gamma (and beta) -> compact row-0 pages
//   load_x_block         crt x cw tiles of the own slice (one reserve / barrier / push per tile-row; W tail zeroed)
//   gather_partials_block  after both hubs announce the stats table complete: per tile-row, G reads of the
//                        k*nstats records of my rows into the landing scratch, scattered into Float32 gather tiles
//                        (column j = partition q*k + m; columns >= P stay 0)
//   LN mean broadcast    exact fp32 broadcast of the combined row mean (column 0 -> every column), by word copies:
//                        an FPU broadcast would truncate the mean to tf32 (op_design.md Key Risks: LN precision)
//
// Raw dataflow (documented): the record gather / scatter and the mean broadcast are word-level L1 copies no helper
// expresses (op_design.md "Helpers considered and rejected").

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

    // ---- constants ----
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    Noc noc;
    if constexpr (is_ln) {
        if (tail) {
            dataflow_kernel_lib::prepare_reduce_mask<cb_mask, ckernel::ReduceDim::REDUCE_ROW>(w_valid_last);
        }
        // n_j: valid columns of partition j = q*K + m (member m's width; every chip has the same W_local).
        CircularBuffer cb_consts(cb_stat_consts);
        cb_reserve_back(cb_stat_consts, 1);
        noc.async_write_zeros(cb_consts, TILE * TILE * 4, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stat_consts));
        for (uint32_t j = 0; j < P; ++j) {
            const uint32_t m = j % K;
            const uint32_t ws = m * cw_nom;
            const uint32_t wtiles = (Wlt - ws) < cw_nom ? (Wlt - ws) : cw_nom;
            uint32_t n = wtiles * TILE;
            if (ws + wtiles == Wlt) {
                n -= TILE - w_valid_last;
            }
            const uint32_t bits = u32_to_f32_bits(n);
            for (uint32_t r = 0; r < TILE; ++r) {
                t[f32_tile_word(r, j)] = bits;
            }
        }
        cb_push_back(cb_stat_consts, 1);
    }

    // ---- load_gamma_beta_slice ----
    if constexpr (has_gamma) {
        const auto g_acc =
            TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(CRT_GAMMA_ADDR), gamma_src_page_bytes);
        load_affine_slice<cb_gamma>(g_acc, p, w);
    }
    if constexpr (has_beta) {
        const auto b_acc = TensorAccessor(beta_args, get_common_arg_val<uint32_t>(CRT_BETA_ADDR), gamma_src_page_bytes);
        load_affine_slice<cb_beta>(b_acc, p, w);
    }

    // ---- load_x_block ----
    {
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

    // ---- gather_partials_block ----
    {
        volatile tt_l1_ptr uint32_t* ready =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stats_ready));
        noc_semaphore_wait_min(ready, 2);  // both hubs: own slot + every remote slot of the table landed
    }
    const uint32_t table = get_common_arg_val<uint32_t>(CRT_TABLE_ADDR);
    const uint32_t hub_mask = get_common_arg_val<uint32_t>(CRT_HUB_MASK);
    const uint32_t fwd_hub = get_common_arg_val<uint32_t>(CRT_FWD_HUB);
    const uint32_t bwd_hub = get_common_arg_val<uint32_t>(CRT_BWD_HUB);
    constexpr uint32_t row_rec_bytes = K * nstats * RECORD_BYTES;  // one tile-row of one slot
    constexpr uint32_t slot_bytes = R * row_rec_bytes;
    constexpr uint32_t f32_tile_bytes = TILE * TILE * 4;
    const uint32_t landing = get_write_ptr(cb_record_landing);
    CircularBuffer cb_g(cb_gathered);
    for (uint32_t r = 0; r < w.crt; ++r) {
        const uint32_t tr = w.row_start + r;
        cb_reserve_back(cb_gathered, nstats * gather_tiles);
        const uint32_t gbase = get_write_ptr(cb_gathered);
        for (uint32_t t = 0; t < nstats * gather_tiles; ++t) {
            noc.async_write_zeros(cb_g, f32_tile_bytes, {.offset_bytes = t * f32_tile_bytes});
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t q = 0; q < G; ++q) {
            const uint32_t hub = ((hub_mask >> q) & 1u) ? bwd_hub : fwd_hub;
            const uint64_t src = get_noc_addr(hub & 0xFFFF, hub >> 16, table + q * slot_bytes + tr * row_rec_bytes);
            noc_async_read(src, landing + q * row_rec_bytes, row_rec_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < P; ++j) {  // j = q * K + m: the slot's records are [m][s]
            for (uint32_t s = 0; s < nstats; ++s) {
                volatile tt_l1_ptr uint32_t* rec =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(landing + (j * nstats + s) * RECORD_BYTES);
                volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    gbase + (s * gather_tiles + (j >> 5)) * f32_tile_bytes);
                const uint32_t c = j & 31;
                for (uint32_t i = 0; i < TILE; ++i) {
                    tile[f32_tile_word(i, c)] = rec[i];
                }
            }
        }
        cb_push_back(cb_gathered, nstats * gather_tiles);
    }

    // ---- LN: exact broadcast of the combined row mean ----
    if constexpr (is_ln) {
        for (uint32_t r = 0; r < w.crt; ++r) {
            cb_wait_front(cb_mean_col, 1);
            cb_reserve_back(cb_mean_b, 1);
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_mean_col));
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mean_b));
            for (uint32_t i = 0; i < TILE; ++i) {
                const uint32_t v = src[f32_tile_word(i, 0)];
                volatile tt_l1_ptr uint32_t* row0 = dst + f32_tile_word(i, 0);
                volatile tt_l1_ptr uint32_t* row1 = dst + f32_tile_word(i, FACE);
                for (uint32_t c = 0; c < FACE; ++c) {
                    row0[c] = v;
                    row1[c] = v;
                }
            }
            cb_push_back(cb_mean_b, 1);
            cb_pop_front(cb_mean_col, 1);
        }
    }
}
