// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-core W-split reader for rms_norm (op_design.md §5).
//
// The input W-slice is already resident in L1 (zero-copy sharded cb_x_in), so the
// reader does NOT read x. It only:
//   * prepares the 1/W reduce scaler (full tile + partial tile when W is not
//     tile-aligned; the compute-side partial-holder routes the partial tile), and
//   * reads this core's gamma W-slice (vwt tiles) ONCE into cb_gamma (held,
//     reused across every tile-row) — the same batched tiled-gamma read as the
//     interleaved TILE-gamma path (Refinement 2), offset to the core's W-slice.
//
// gamma is (1,1,1,W) -> Wt tiles in one tile-row; this core reads tiles
// [w_tile_start, w_tile_start + vwt).
//   TILE gamma : read whole tiles straight into cb_gamma (compute skips tilize).
//   RM gamma   : read the W-slice as row-major sticks into cb_gamma_sticks (one
//                tile-wide page per read) so compute tilizes vwt tiles into
//                cb_gamma — the cross-core mirror of the interleaved RM-gamma
//                knob-turn (Refinement 2/4a). gamma stays interleaved in DRAM.
//
// X_FROM_DRAM (Refinement 4a, logical wide-interleaved / decode W-split): the input
// is an INTERLEAVED tensor whose W is split across K cores (no physical shard). This
// reader reads this core's W/K slice (per_w_t tiles of every tile-row) from DRAM into
// cb_x_in via TensorAccessor (tile_id = t*Wt + w_tile_start + w) so the same
// cross-core combine/compute runs. When X_FROM_DRAM=0 the slice is a zero-copy
// sharded CB and the reader does not touch x.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_x_sticks = 0;  // RM input: tile-padded sticks (loopback repack) -> compute tilize
constexpr uint32_t cb_x_in = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_gamma_sticks = 4;
constexpr uint32_t cb_shard_in = 8;  // RM input: zero-copy alias of the resident RM W-slice (stick pages)
constexpr uint32_t TILE_DIM = 32;
}  // namespace

void kernel_main() {
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t inv_N_bits = get_compile_time_arg_val(1);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t partial_w = get_compile_time_arg_val(3);
    constexpr uint32_t gamma_page = get_compile_time_arg_val(4);
    constexpr bool GAMMA_IS_RM = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t gamma_elem = get_compile_time_arg_val(6);
    constexpr uint32_t origin_W = get_compile_time_arg_val(7);
    constexpr bool X_FROM_DRAM = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t Wt = get_compile_time_arg_val(9);         // full tile-row width (tiles)
    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(10);  // tile-rows this core handles
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(11);   // cb_x_in W-tiles per tile-row
    constexpr uint32_t in_page = get_compile_time_arg_val(12);
    // IS_RM (Refinement 4b): the resident W-slice is ROW-MAJOR (arbitrary sub-tile
    // width). This reader loopback-repacks the shard sticks into tile-padded cb_x_sticks
    // and reads the gamma W-slice at an ELEMENT column offset (not a tile boundary).
    constexpr bool IS_RM = get_compile_time_arg_val(13) != 0;
    constexpr uint32_t ELEM = get_compile_time_arg_val(14);               // element bytes (RM loopback math)
    constexpr uint32_t SHARD_STICK_BYTES = get_compile_time_arg_val(15);  // resident RM shard stick stride

    constexpr auto gamma_args = TensorAccessorArgs<16>();
    constexpr auto in_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(1);  // IS_RM: g0 = first global tile (w_offset//32)
    const uint32_t vwt = get_arg_val<uint32_t>(2);           // IS_RM: reduce tile count (ceil(valid_end/32))
    const uint32_t in_addr = get_arg_val<uint32_t>(3);
    const uint32_t valid_cols = get_arg_val<uint32_t>(4);        // IS_RM: valid columns in this core's slice
    const uint32_t valid_rows_total = get_arg_val<uint32_t>(5);  // IS_RM: valid rows in this core's shard
    const uint32_t reduce_partial_w = get_arg_val<uint32_t>(6);  // IS_RM: valid_end % 32 (0 => no partial)
    const uint32_t phase = get_arg_val<uint32_t>(7);             // IS_RM: w_offset % 32 (leading tile offset)

    // scaler = 1/origin_W so the SUM-reduce over the LOCAL slice yields the slice's
    // (1/W)-scaled partial; summing the K partials across cores gives mean(x^2).
    const float scaler_f = __builtin_bit_cast(float, inv_N_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f);  // tile 0: full scaler
    if constexpr (IS_RM) {
        // RM: EVERY core's last reduce-tile is sub-tile-wide, so always emit tile 1.
        // When this core has no partial (valid_cols % 32 == 0) tile 1 is a full scaler
        // duplicate (unused — compute routes ReducePartialScaler::none()).
        if (reduce_partial_w != 0) {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    scaler_f, reduce_partial_w);  // tile 1: partial scaler (zeros padded lanes)
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    scaler_f);  // tile 1: full scaler duplicate
        }
    } else if constexpr (HAS_PARTIAL_W) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler_f, partial_w);  // tile 1: partial scaler (zeros padded lanes)
    }

    // ---- RM-input sharded (Refinement 4b): gamma W-slice + loopback x repack ----
    if constexpr (IS_RM) {
        constexpr uint32_t PADDED_ROW_BYTES = PER_W_T * TILE_DIM * ELEM;
        const uint32_t phase_bytes = phase * ELEM;  // leading tile offset of this core's slice
        if constexpr (HAS_GAMMA) {
            // gamma (1,1,1,W) is interleaved DRAM. Read this core's `vwt` valid tiles at
            // the tile-ALIGNED global column (g0+wt)*32 (w_tile_start = g0), so the DRAM
            // read is aligned — a sub-tile column offset faults. Tile wt covers global
            // columns [(g0+wt)*32, ...), matching x's local tile wt after phase-align.
            const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page);
            for (uint32_t wt = 0; wt < vwt; ++wt) {
                const uint32_t col0 = (w_tile_start + wt) * TILE_DIM;
                uint32_t cols = (col0 < origin_W) ? (origin_W - col0) : 0;
                if (cols > TILE_DIM) {
                    cols = TILE_DIM;
                }
                dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks>(
                    gamma_accessor, /*total_num_rows=*/1, cols * gamma_elem, /*start_page=*/0, col0 * gamma_elem);
            }
        }

        // Zero the cb_x_sticks backing ONCE. The loopback below writes only the valid
        // columns [phase, phase+valid_cols) of each stick, so the leading [0,phase) and
        // trailing pad columns stay 0 (nan-safe: leading zeros contribute 0 to the SUM
        // reduce, trailing lanes are masked by the partial scaler). Pad rows of a partial
        // last tile-row are never written -> stay 0 (harmless).
        {
            const uint32_t zbase = get_write_ptr(cb_x_sticks);
            const uint32_t znbytes = get_local_cb_interface(cb_x_sticks).fifo_num_pages * get_tile_size(cb_x_sticks);
            volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbase);
            for (uint32_t k = 0; k < (znbytes >> 2); ++k) {
                zp[k] = 0;
            }
        }
        // Loopback-repack the resident RM shard (cb_shard_in, contiguous sticks at
        // SHARD_STICK_BYTES stride) into tile-padded cb_x_sticks (PADDED_ROW_BYTES
        // stride) at column offset `phase`, so the compute-side tilize consumes it
        // exactly like the interleaved RM path. Reading only valid_cols*ELEM per stick
        // keeps shard tensor-padding out of the reduce.
        const uint32_t shard_base = get_read_ptr(cb_shard_in);
        const uint32_t vc_bytes = valid_cols * ELEM;
        for (uint32_t t = 0; t < HT_LOCAL; ++t) {
            uint32_t valid_rows = (valid_rows_total > t * TILE_DIM) ? (valid_rows_total - t * TILE_DIM) : 0;
            if (valid_rows > TILE_DIM) {
                valid_rows = TILE_DIM;
            }
            cb_reserve_back(cb_x_sticks, PER_W_T);
            const uint32_t dst = get_write_ptr(cb_x_sticks) + phase_bytes;
            for (uint32_t s = 0; s < valid_rows; ++s) {
                const uint32_t src = shard_base + (t * TILE_DIM + s) * SHARD_STICK_BYTES;
                noc_async_read(
                    get_noc_addr(my_x[noc_index], my_y[noc_index], src), dst + s * PADDED_ROW_BYTES, vc_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_x_sticks, PER_W_T);
        }
        return;
    }

    // Logical W-split: read this core's W/K slice from interleaved DRAM into cb_x_in.
    // Whole slice held resident (compute indexes both passes over it) — one coalesced
    // barrier. The ragged last core's padding slots [vwt, per_w_t) are left un-read
    // (pass 1 reads only vwt; pass-2 padding output is not written back).
    if constexpr (X_FROM_DRAM) {
        MaybeDeviceZoneScope("xc_rd_x");
        const auto in_accessor = TensorAccessor(in_args, in_addr, in_page);
        const uint32_t tile_bytes = get_tile_size(cb_x_in);
        const uint32_t shard_tiles = HT_LOCAL * PER_W_T;
        cb_reserve_back(cb_x_in, shard_tiles);
        const uint32_t base = get_write_ptr(cb_x_in);
        for (uint32_t t = 0; t < HT_LOCAL; ++t) {
            uint32_t l1 = base + t * PER_W_T * tile_bytes;
            for (uint32_t w = 0; w < vwt; ++w) {
                noc_async_read_tile(t * Wt + w_tile_start + w, in_accessor, l1);
                l1 += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_x_in, shard_tiles);
    }

    if constexpr (HAS_GAMMA) {
        MaybeDeviceZoneScope("xc_rd_gamma");
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr, gamma_page);
        if constexpr (GAMMA_IS_RM) {
            // gamma (1,1,1,W) is one row -> page 0. Read this core's W-slice one
            // tile-column at a time (row_bytes <= one tile width) so each read
            // pushes exactly ONE tile-wide page; compute tilizes vwt of them.
            // Columns beyond origin_W are gamma tensor-padding (output discarded),
            // so clamp cols but still emit vwt pages. Matches the interleaved
            // RM-gamma read (base page 0, per-tile column byte offset).
            for (uint32_t wt = 0; wt < vwt; ++wt) {
                const uint32_t col0 = (w_tile_start + wt) * TILE_DIM;
                uint32_t cols = (col0 < origin_W) ? (origin_W - col0) : 0;
                if (cols > TILE_DIM) {
                    cols = TILE_DIM;
                }
                dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_sticks>(
                    gamma_accessor, /*total_num_rows=*/1, cols * gamma_elem, /*start_page=*/0, col0 * gamma_elem);
            }
        } else {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, vwt);
            uint32_t gl1 = get_write_ptr(cb_gamma);
            for (uint32_t wt = 0; wt < vwt; ++wt) {
                noc_async_read_tile(w_tile_start + wt, gamma_accessor, gl1);
                gl1 += gamma_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, vwt);
        }
    }
}
