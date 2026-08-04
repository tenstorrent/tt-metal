// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for rms_norm (NCRISC, NoC0).
//
// Per core, per row-block, per width chunk it stages:
//   * TILE build      : whole x tiles         -> cb_input_tiles
//   * ROW_MAJOR build : padded x sticks       -> cb_input_sticks (compute tilizes)
// plus, once at boot, the reduce scaler; plus gamma (once per core in the
// RESIDENT regime, once per width chunk in STREAM).
//
// Loop nest mirrors the compute kernel exactly (op_design.md section 7):
//     for blk in blocks: for pass in {A} or {A,B}: for c in chunks
// Pass B is re-read ONLY in the STREAM regime (X_RESIDENT == false); in the
// RESIDENT regime cb_input_tiles is held across both passes, so x is read once.
//
// Helper-usage notes
// ------------------
// * scaler CB          -> dataflow_kernel_lib::prepare_[partial_]reduce_scalers
//                         (ReduceTile datapath) or prepare_reduce_mask
//                         (AccumulateViaAdd datapath), pool-type-aware
//                         overloads (PoolType::SUM, ReduceDim::REDUCE_ROW).
// * ROW_MAJOR staging  -> dataflow_kernel_lib::read_sticks_for_tilize at TILE
//                         granularity, which is exactly the contract of
//                         compute_kernel_lib::tilize<WT_CHUNK>(rows).
// * TILE staging + gamma reads are TensorAccessor + noc_async_read_tile: the
//   dataflow tilize helper covers neither whole-tile interleaved reads nor the
//   gamma slot (op_design.md section 6.1).
//
// One raw-API addition beyond the design's table: a ONE-TIME zero of the whole
// cb_input_sticks ring at boot, via noc.async_write_zeros (the device zero
// API), gated on PARTIAL_W != 0.  Reason (R3): the L1 pad lanes of a staged
// ROW_MAJOR row are never written by a stick read, so whatever L1 garbage was
// there survives into the reduce.  The partial scaler multiplies pad lanes by
// zero, and inf*0 / nan*0 = NaN would poison the whole row.  Zeroing the ring
// once establishes the invariant "every pad byte is either zero or real tensor
// data" (later reads only ever overwrite with tensor values), so no per-block
// zeroing is needed.  H-tail rows need no zeroing: a padding row's reduction
// and output are confined to that row and the writer never writes it.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_gamma_sticks = 5;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(4);
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(5);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(6);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(7);
    constexpr uint32_t ELEM_BYTES = get_compile_time_arg_val(8);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(9);
    constexpr uint32_t R_RM = get_compile_time_arg_val(10);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(11);
    constexpr uint32_t REDUCE_ACC_VIA_ADD = get_compile_time_arg_val(12);
    constexpr auto x_args = TensorAccessorArgs<13>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    constexpr bool RM = (IS_TILE == 0);
    constexpr bool HAS_G = (HAS_GAMMA != 0);
    constexpr bool G_RM = (GAMMA_IS_RM != 0);
    // X_RESIDENT == GAMMA_RESIDENT == (NUM_W_CHUNKS == 1): one source of truth.
    constexpr bool X_RESIDENT = (NUM_W_CHUNKS == 1);
    constexpr uint32_t NUM_PASSES = X_RESIDENT ? 1 : 2;

    // Bytes of one full width chunk of a row-major stick, and of the last one
    // (short by the tile padding when W is not tile-aligned).
    constexpr uint32_t CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * ELEM_BYTES;
    constexpr uint32_t LAST_CHUNK_ROW_BYTES = W_ELEMS * ELEM_BYTES - (NUM_W_CHUNKS - 1) * CHUNK_ROW_BYTES;
    constexpr uint32_t G_CHUNK_ROW_BYTES = WT_CHUNK * TILE_DIM * GAMMA_ELEM_BYTES;
    constexpr uint32_t G_LAST_CHUNK_ROW_BYTES = W_ELEMS * GAMMA_ELEM_BYTES - (NUM_W_CHUNKS - 1) * G_CHUNK_ROW_BYTES;

    // ---- runtime work assignment -----------------------------------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start = get_arg_val<uint32_t>(2);  // this core's first tile-row
    const uint32_t num_rows = get_arg_val<uint32_t>(3);   // tile-rows owned by this core

    const auto x_acc = TensorAccessor(x_args, x_addr);

    // ---- boot: what cb_scaler carries, per reduce datapath ----------------
    // Value is exactly 1.0 everywhere; 1/W is applied in fp32 by the compute
    // finalize, never folded into a bf16 scaler (R4).
    //
    //   ReduceTile       aligned : [full scaler]                   -> 1 tile
    //                    partial : [full scaler, partial scaler]   -> 2 tiles
    //   AccumulateViaAdd aligned : [scaler] (unused by the datapath, but keeps
    //                              the boot SrcB format real)      -> 1 tile
    //                    partial : [0/1 mask]                      -> 1 tile
    // The tile COUNT is the descriptor's SCALER_TILES, which the compute kernel
    // pops -- this branch must agree with it (asserted host-side).
    if constexpr (REDUCE_ACC_VIA_ADD != 0) {
        if constexpr (PARTIAL_W != 0) {
            // 0/1 mask in the row-0 broadcast layout AccumulateViaAdd's masked
            // accumulating broadcast-mul consumes for the last width tile.
            dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ckernel::ReduceDim::REDUCE_ROW>(PARTIAL_W);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        }
    } else if constexpr (PARTIAL_W != 0) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            PARTIAL_W>(1.0f);
    } else {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            1.0f);
    }

    // ---- boot: establish the pad-lane invariant on the RM staging ring ----
    if constexpr (RM && PARTIAL_W != 0) {
        Noc noc;
        DataflowBuffer stage_dfb(cb_input_sticks);
        noc.async_write_zeros(stage_dfb, stage_dfb.get_total_size_bytes());
        noc.write_zeros_l1_barrier();
    }

    // ---- gamma: one chunk's worth of tiles (or sticks) --------------------
    // In RESIDENT this runs once per core before the row-block loop and the
    // tiles are never popped; in STREAM it runs per pass-B chunk.
    auto stage_gamma_chunk = [&](uint32_t c) {
        if constexpr (HAS_G) {
            const auto g_acc = TensorAccessor(gamma_args, gamma_addr);
            if constexpr (G_RM) {
                // gamma is a single stick; row 0 of the staged tile-row is the
                // only row BroadcastDim::Row reads.
                const uint32_t row_bytes = (c + 1 == NUM_W_CHUNKS) ? G_LAST_CHUNK_ROW_BYTES : G_CHUNK_ROW_BYTES;
                dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks>(
                    g_acc,
                    /*total_num_rows=*/1,
                    row_bytes,
                    /*start_page=*/0,
                    /*byte_offset_within_page=*/c * G_CHUNK_ROW_BYTES);
            } else {
                const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_tiles);
                cb_reserve_back(cb_gamma_tiles, WT_CHUNK);
                uint32_t l1_addr = get_write_ptr(cb_gamma_tiles);
                for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                    noc_async_read_tile(c * WT_CHUNK + w, g_acc, l1_addr);
                    l1_addr += gamma_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma_tiles, WT_CHUNK);
            }
        }
    };

    if constexpr (X_RESIDENT) {
        stage_gamma_chunk(0);
    }

    // ---- one width chunk of one row-block of x ---------------------------
    // Transaction granularity is WT_CHUNK tiles (one tile-row of the chunk):
    // a single knob-derived unit that divides every CB ring by construction,
    // and >= 4 tiles per barrier whenever the block allows it.
    const uint32_t x_tile_bytes = get_tile_size(cb_input_tiles);
    auto stage_x_chunk = [&](uint32_t first_tile_row, uint32_t rows, uint32_t c) {
        if constexpr (RM) {
            const uint32_t stick_start = first_tile_row * TILE_DIM;
            uint32_t sticks = rows * TILE_DIM;
            if (stick_start + sticks > R_RM) {
                sticks = R_RM - stick_start;  // short final tile-row
            }
            const uint32_t row_bytes = (c + 1 == NUM_W_CHUNKS) ? LAST_CHUNK_ROW_BYTES : CHUNK_ROW_BYTES;
            dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks>(
                x_acc, sticks, row_bytes, stick_start, /*byte_offset_within_page=*/c * CHUNK_ROW_BYTES);
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t tile_base = (first_tile_row + r) * WT + c * WT_CHUNK;
                cb_reserve_back(cb_input_tiles, WT_CHUNK);
                uint32_t l1_addr = get_write_ptr(cb_input_tiles);
                for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                    noc_async_read_tile(tile_base + w, x_acc, l1_addr);
                    l1_addr += x_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_tiles, WT_CHUNK);
            }
        }
    };

    // ---- row-block loop ---------------------------------------------------
    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        const uint32_t rows = (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
        const uint32_t first_tile_row = row_start + r0;

        for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
            for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
                stage_x_chunk(first_tile_row, rows, c);
                // STREAM: gamma is chunked and re-read for every pass-B chunk.
                if constexpr (!X_RESIDENT) {
                    if (pass == 1) {
                        stage_gamma_chunk(c);
                    }
                }
            }
        }
    }
}
