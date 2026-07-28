// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (NCRISC / NoC0) — op_design.md §4.1.
//
// Per core: a disjoint range of tile-rows, looped in row-blocks of HT_BLOCK
// tile-rows.  Each row-block is read as NW chunks of WT_CHUNK W-tiles; the
// number of reader passes over a row-block is the ONE structural difference
// between the two residency regimes:
//
//   X_RESIDENT     -> 1 pass  (whole HT_BLOCK x Wt strip stays in L1)
//   streaming       -> 2 passes (pass A feeds the reduce, pass B feeds the scale)
//
// Every count below is a function of the block knobs (HT_BLOCK / WT_CHUNK / NW)
// — never of a whole-op dimension.
//
// TILE input is read with whole-tile pages through a TensorAccessor and one
// barrier per chunk (batched, coalescing the chunk into a single NoC burst
// train).  ROW_MAJOR input goes through
// dataflow_kernel_lib::read_sticks_for_tilize, whose byte_offset_within_page
// argument IS the WT_CHUNK knob on the read side.
//
// Helper substitution note: the TILE path uses TensorAccessor +
// noc_async_read_page directly because no dataflow helper can express a
// tile-page read of an already-tiled tensor (read_sticks_for_tilize is
// stick-indexed and feeds the tilize helper). op_design.md §7 mandates exactly
// this.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_input_rm = 3;
constexpr uint32_t cb_gamma_rm = 4;
}  // namespace

namespace dkl = dataflow_kernel_lib;

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(1) != 0;
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(2) != 0;
    constexpr bool X_RESIDENT = get_compile_time_arg_val(3) != 0;
    constexpr bool GAMMA_RESIDENT = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(5) != 0;
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // ---- geometry ----
    constexpr uint32_t W_VALID_LAST = get_compile_time_arg_val(11);
    constexpr uint32_t CHUNK_ROW_BYTES = get_compile_time_arg_val(12);
    constexpr uint32_t LAST_ROW_BYTES = get_compile_time_arg_val(13);
    constexpr uint32_t G_CHUNK_ROW_BYTES = get_compile_time_arg_val(14);
    constexpr uint32_t G_LAST_ROW_BYTES = get_compile_time_arg_val(15);
    constexpr uint32_t TOTAL_STICKS = get_compile_time_arg_val(16);

    constexpr auto in_args = TensorAccessorArgs<17>();
    constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    static_assert(WT_LAST == WT_CHUNK, "reader assumes uniform chunk widths");

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(3);

    // ---- 1. scaler / partial-W mask: one tile, pushed once, never popped ----
    if constexpr (HAS_PARTIAL_W) {
        dkl::prepare_reduce_mask<cb_scaler, ckernel::ReduceDim::REDUCE_ROW>(W_VALID_LAST);
    } else {
        dkl::calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    }

    const auto in_acc = TensorAccessor(in_args, src_addr);
    [[maybe_unused]] const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr);
    const uint32_t in_tile_bytes = get_tile_size(cb_input_tiles);

    // ---- 2. gamma: reuse-shared operand, read once when it fits L1 (§1.1) ----
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
        if constexpr (IS_RM_GAMMA) {
            for (uint32_t wc = 0; wc < NW; ++wc) {
                const uint32_t rb = (wc + 1 == NW) ? G_LAST_ROW_BYTES : G_CHUNK_ROW_BYTES;
                dkl::read_sticks_for_tilize<cb_gamma_rm, dkl::TilizeGranularity::ROW>(
                    gamma_acc, /*total_num_rows=*/1, rb, /*start_page=*/0, wc * G_CHUNK_ROW_BYTES);
            }
        } else {
            const uint32_t gt = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, WT);
            uint32_t addr = get_write_ptr(cb_gamma);
            for (uint32_t t = 0; t < WT; ++t) {
                noc_async_read_page(t, gamma_acc, addr);
                addr += gt;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, WT);
        }
    }

    // ---- per-chunk readers -------------------------------------------------

    // TILE: ht x WT_CHUNK whole tile pages, one barrier for the whole chunk.
    auto read_input_chunk_tile = [&](uint32_t wc, uint32_t row0, uint32_t ht) {
        const uint32_t n = ht * WT_CHUNK;
        cb_reserve_back(cb_input_tiles, n);
        uint32_t addr = get_write_ptr(cb_input_tiles);
        for (uint32_t h = 0; h < ht; ++h) {
            const uint32_t base_tile = (row0 + h) * WT + wc * WT_CHUNK;
            for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                noc_async_read_page(base_tile + t, in_acc, addr);
                addr += in_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_tiles, n);
    };

    // ROW_MAJOR: one row page per stick; `valid_rows` clamps the read to the
    // sticks that actually exist (non-tile-aligned H), and the missing rows of
    // the tile-row block are pushed unread so the tilize helper always consumes
    // whole 32-row blocks (their stale content lands in H-padding rows the
    // writer never writes back).
    auto read_input_chunk_rm = [&](uint32_t wc, uint32_t row0, uint32_t ht, uint32_t valid_rows) {
        const uint32_t rb = (wc + 1 == NW) ? LAST_ROW_BYTES : CHUNK_ROW_BYTES;
        dkl::read_sticks_for_tilize<cb_input_rm, dkl::TilizeGranularity::ROW>(
            in_acc, valid_rows, rb, row0 * 32u, wc * CHUNK_ROW_BYTES);
        const uint32_t pad_rows = ht * 32u - valid_rows;
        if (pad_rows != 0) {
            cb_reserve_back(cb_input_rm, pad_rows);
            cb_push_back(cb_input_rm, pad_rows);
        }
    };

    auto read_input_chunk = [&](uint32_t wc, uint32_t row0, uint32_t ht, uint32_t valid_rows) {
        if constexpr (IS_RM) {
            read_input_chunk_rm(wc, row0, ht, valid_rows);
        } else {
            read_input_chunk_tile(wc, row0, ht);
        }
    };

    auto read_gamma_chunk = [&](uint32_t wc) {
        if constexpr (IS_RM_GAMMA) {
            const uint32_t rb = (wc + 1 == NW) ? G_LAST_ROW_BYTES : G_CHUNK_ROW_BYTES;
            dkl::read_sticks_for_tilize<cb_gamma_rm, dkl::TilizeGranularity::ROW>(
                gamma_acc, /*total_num_rows=*/1, rb, /*start_page=*/0, wc * G_CHUNK_ROW_BYTES);
        } else {
            const uint32_t gt = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, WT_CHUNK);
            uint32_t addr = get_write_ptr(cb_gamma);
            for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                noc_async_read_page(wc * WT_CHUNK + t, gamma_acc, addr);
                addr += gt;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, WT_CHUNK);
        }
    };

    // ---- 3. row-block loop -------------------------------------------------
    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        const uint32_t row0 = start_tile_row + hb * HT_BLOCK;
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }

        uint32_t valid_rows = ht * 32u;
        if constexpr (IS_RM) {
            const uint32_t remaining = TOTAL_STICKS - row0 * 32u;
            if (remaining < valid_rows) {
                valid_rows = remaining;
            }
        }

        // pass A — feeds square -> chunked SUM.
        if constexpr (X_RESIDENT && !IS_RM) {
            // Resident strip: one coalesced ht x Wt read, a single barrier.
            const uint32_t n = ht * WT;
            cb_reserve_back(cb_input_tiles, n);
            uint32_t addr = get_write_ptr(cb_input_tiles);
            for (uint32_t h = 0; h < ht; ++h) {
                const uint32_t base_tile = (row0 + h) * WT;
                for (uint32_t t = 0; t < WT; ++t) {
                    noc_async_read_page(base_tile + t, in_acc, addr);
                    addr += in_tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, n);
        } else {
            for (uint32_t wc = 0; wc < NW; ++wc) {
                read_input_chunk(wc, row0, ht, valid_rows);
            }
        }

        // pass B — re-read x only when it is not resident; stream gamma when it
        // is not resident.
        if constexpr (!X_RESIDENT || (HAS_GAMMA && !GAMMA_RESIDENT)) {
            for (uint32_t wc = 0; wc < NW; ++wc) {
                if constexpr (!X_RESIDENT) {
                    read_input_chunk(wc, row0, ht, valid_rows);
                }
                if constexpr (HAS_GAMMA && !GAMMA_RESIDENT) {
                    read_gamma_chunk(wc);
                }
            }
        }
    }
}
