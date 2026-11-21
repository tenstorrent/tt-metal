// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_adaln — compute.
//
// Stats phase (stats_chain, perf tournament 4; stats_publish_fuse, round 3 tournament 5): per tile row, streamed
// behind the reader's chunked push (the reader pushes a row `chunk` tiles at a time, one NoC read trid per chunk):
//   pivot      p = bf16(mean of the row's first pivot_tiles landed tiles) — needs only the first chunk, not the
//              whole row's mean. (p = x[:, 0] was measured and rejected: an outlier first column breaks precision.)
//   dev        the whole row in ONE DEST acquire (dev_pass_macro, tournament 6): pivot tile copied into D0 first,
//              then per window D[1 + j] = x - p (FPU sub, srcB Col-bcast p) and the SFPU folds every same-row vector
//              of the window into 8 + 8 DEST-resident STRIP accumulators (sum (x - p), sum (x - p)^2 per 4-row strip,
//              packed into D0; p stashed in D0's unused parity-1 vectors); after the last window one SFPU pass
//              lane-reduces each strip (8-lane rotate-add in fp32 LREGs) and writes the payload, one pack of the
//              row's payload -> cb_local_stats. Both DEST modes.
// Raw LLK:
//   - the pivot reduce bypasses compute_kernel_lib::reduce (no tile-base offset / rotated column order for an
//     un-popped, partially-landed row); same calls as the helper's FPU REDUCE_ROW path.
//   - the dev + publish pass bypasses eltwise_chain and compute_kernel_lib::reduce (and, since dev_pass_macro, the
//     sfpu_reduce<SUM, REDUCE_ROW> call the reduce helper's Accurate fp32 path makes): (a) its accumulators live in
//     DEST across windows of one acquire (no chain element accumulates an SFPU result into a separate DEST slot across
//     calls; reduce() takes its input from a CB, which would need the pack -> unpack round trip this removes), (b) the
//     accumulators are 4-row STRIP vectors (every same-row SFPU vector of a tile folded into one, 16 vectors in ONE
//     DEST tile instead of two whole accumulator tiles), which frees a work slot per window (fp32 3, 16-bit 7) and
//     cuts the per-window accumulator traffic 4x, and no helper folds lanes / faces of different tiles into one
//     vector, (c) the row sum is an 8-lane rotate-add (SFPSHFT2 SUBVEC_SHFLROR1) of the 16 strip vectors in fp32 LREGs
//     inside the publish routine, where sfpu_reduce<ROW> reduces two whole tiles and stores the sums back to DEST
//     first (16-bit DEST: rounding the row sums to bf16 before the publish), and (d) pivot copy, dev pass and
//     publish share one acquire (the helpers each own an acquire).
//     Measured (BH, perf_experiments/dev_pass_macro): isolated stats stage (pivot + dev + publish, one resident row of
//     16 tiles, warm) fp32 3283 -> 2434 ns, 16-bit 2760 -> 2361 ns; 48 tiles fp32 8150 -> 5522 ns. Whole op,
//     same-process alternating A/B vs HEAD 78f3167cc31, median of 3: 1x1x256x3072 dynamic_wb fp32 14.51 -> 13.96 us
//     (bf16 output bit-identical), 16-bit 14.44 -> 14.10 us, 256x1024 wb fp32 8.79 -> 8.45 us; payload ready after
//     the row's last chunk lands 1790 -> 1399 ns (fp32), 1711 -> 1338 ns (16-bit). Statistics: fp32 payload equal to
//     HEAD within fp32 summation order; 16-bit M2 error vs float64 1.5e-2 -> 8.3e-3 (max), 0.217 -> 0.0075 at offset
//     1000 (row sums no longer rounded to bf16 in DEST before the publish).
//   What would close it: a DEST-resident accumulate element (SFPU sum / sum-of-squares into fixed slots across calls,
//   with a row-strip accumulator layout) and a row-reduce element over SFPU-register (strip) inputs.
//
// Block schedule (op_design.md "Block schedule"), per block of block_rows x block_width tiles:
//   stats + publish (above, per row) SFPU (column 0): d = sum(x-p)/n, M2 = s2 - sum(x-p) d; emit one payload tile per
//                       row: [r_hi, r_lo | M2_hi, M2_lo] (fp32 DEST) | [p, M2 | d] (16-bit DEST)  -> cb_local_stats
//   (exchange_block on the writer: streamed all-gather of S payloads, own slot first      -> cb_gathered_stats)
//   (compact_stats_payload: the publish packs the payload's stats into face 0 rows 0..7 — one SFPU vector per stat
//    for all 32 rows, SFPSHFT2 lane rotations — so the writer sends only that 1/8-tile prefix)
//   combine_block       SFPU Chan combine in own pivot frame on the compact vectors, one fold per landed slot: m,
//                       M2, rstd (expanded back to column 0)                             -> cb_row_stats
//   normalize_block     fp32 DEST: D = c'_k (SFPU LREG preload), D += x * rstd (one FPU MAC), then [D * w + b on
//                       the SFPU]. 16-bit DEST: D = x - p (FPU, srcB Col-bcast p), then (D * rstd + c'') [* w + b] on
//                       the SFPU in fp32 LREGs. w / b come from per-call P tiles (cb_wbp, built in the all-gather
//                       idle), 16-bit rstd / c'' from per-row S tiles (cb_nstats). Pushed per DEST window  ->
//                       cb_output_tiles Columns are visited from the per-core start column `rot` (RT arg 2) and wrap —
//                       the writer stores in that order (DRAM bank stagger, Perf 2 dram_bw).
//
// Raw-LLK deviation (publish, combine_block): custom SFPI routines (layer_norm_adaln_stats_sfpu.hpp) driven by
// copy_tile into one dst-sync window per tile row. Helper gap:
//   - eltwise_chain elements are a compile-time list, so the combine's fold over a CT-variable number of peers S
//     (accumulators live in DEST across peer copies) cannot be one chain call; splitting it would pack fp32
//     intermediates to L1 and back for every peer.
//   - the payload carries several stats per tile in the column-0 lanes of columns 0/1 and faces 0/2 | 1/3; no helper
//     addresses a lane / face offset as a second operand / result, and the binary SFPU *_tile APIs only run all four
//     faces (VectorMode::RC).
//   - both ops emit outputs into one CB at out-of-order indices; the chain rejects two PackTiles targeting one CB.
// Measured (BH, 1x1x256x3072 dynamic_wb fp32, whole op): baseline library-op combine 39.7 us; same library ops on
// faces 0/2 with copy re-init hoisted 34.2 us; fused SFPU routines 31.1 us; fused + packed payload 28.7 us.
// combine_lean (perf tournament round 3, BH, whole op, same-process A/B vs HEAD): the routines visit only the 8
// column-0 SFPU vectors (VectorMode::None) and apply rsqrt in place (rsqrt_tile_init + rsqrt_tile(VectorMode::C)
// bypassed: same non-approximate body and constants, bit-identical column 0); the payload is one tile per row in both
// DEST modes (fp32: tf32-exact hi/lo split copied through srcA instead of a per-tile unpack-to-dest handshake;
// 16-bit: [p, M2 | d]); row 0's pivot copy runs before the gather wait. 256x3072 wb fp32 17.4 -> 16.2 us (bf16 output
// bit-identical), bf16 DEST 18.8 -> 15.9 us.
// streamed_fold (round 3): the writer pushes cb_gathered_stats slot by slot as each peer's flag lands (raw dataflow,
// see the writer) and the combine folds each landed slot at once (cumulative CB waits), so only one fold + finalize
// follow the last arrival; both DEST modes fold in one pass (16-bit: running sums kept compensated, hi + lo in the
// column-0 / column-1 lanes, finalize rounds m and var + eps to nearest). Helper gap: no helper folds a CB whose pages
// arrive incrementally into DEST-resident accumulators. r5_merge stack 256x3072 wb fp32 16.00 -> 15.68 us, noaff
// 16-bit 14.58 -> 13.68 us (rel-RMS 0.0090 -> 0.0063).
// What would close it: a chain element for a runtime-count fold over CB tiles, face-offset operands, multi-slot pack.
//
// Raw-LLK deviation (normalize_block, fp32 DEST): (x - p - m) * rstd is one standard FPU multiply-accumulate per tile,
// D = c'_k; D += x * rstd (ELWMUL accumulates into DEST; srcA = x straight from the unpacker, srcB = rstd Col bcast,
// HiFi4), with c'_k = -(p + m) * rstd built once per tile row (SFPU, fp32) and held in SFPU LREG0..7 (one vector per
// 4-row strip), then SFPSTOREd into each window's DEST slots. Replaces sub_tiles_bcast + two dest-reuse ops (each a
// DEST -> srcA MOVD2A round trip per face, which also truncates the running value to tf32). Helper gap: no helper
// preloads DEST from SFPU registers or exposes an accumulate-into-DEST multiply with a broadcast operand, and an
// unpack-to-dest copy of c' (the library way to preload DEST) costs more than the ops it removes (measured: +96 ns
// per tile). (c' ~ -mean * rstd does not survive a 16-bit DEST: that mode uses the SFPU path below.)
// Measured (BH, 3 alternating reps, whole op, vs an identical-code A/A control): math-TRISC normalize 256x3072 wb
// 5.68 -> 4.79 us; whole op 2048x3072 wb fp32 -5.8%, 8192x256 wb -2.8%, 256x3072 wb -1.6% (write-limited after),
// 2304x3072 flat (write wall). Precision improves (no tf32 truncations of the running value): rel-RMS 0.0060 ->
// 0.0017 (noaff, offset 0), got/true median 0.995 -> 0.9997.
// What would close it: a DEST-preload-from-SFPU-registers element plus an accumulate flag on BinaryFpu with broadcast.
//
// Raw-LLK deviation (normalize_block, sfpu_normalize): the affine tail and the whole 16-bit normalize run as custom
// SFPI routines (layer_norm_adaln_stats_sfpu.hpp nz_*) on fp32 LREGs instead of FPU dest-reuse binaries
// (DestReuseBinary / dest-reuse raw calls: each a DEST -> srcA MOVD2A round trip, HiFi4 multiplies at 4 fidelity
// phases). Operands reach the SFPU as strip vectors at fixed DEST positions: per-column w / b from P tiles
// (row-replicated 4-row strips, 3 columns per tile, built once per call: Row-bcast + SFPU strip copies), per-row
// rstd / c'' from S tiles (16-bit; merged into the P slot by one FPU add), c' (fp32) stashed in P face 3 across the
// affine. Helper gap: eltwise_chain has no SFPU element that reads a second operand from a DEST position other than
// the same tile/face (no strip-broadcast operand from another slot), no multi-operand SFPU MAD element
// (D * a + b with a, b from DEST), and no way to keep values live in LREGs across chain calls / DEST windows.
// Measured (BH, whole op, same process vs HEAD, median of 3 / single sweep runs): math-TRISC normalize 256x3072 wb
// fp32 4.71 -> 4.11 us, wb 16-bit 5.54 -> 3.27 us; whole op 256x3072 wb fp32 -3.9%, wb 16-bit -9.0%, noaff 16-bit
// -4.0%, 8192x256 wb fp32 -3.6%, wb 16-bit -8.7%; in the r5_merge stack 256x3072 wb 16-bit 15.74 -> 14.59 us, 32x3072
// wb fp32 12.23 -> 11.46 us. Precision improves (fp32 affine: rel-RMS 0.0044 -> 0.0017; 16-bit 0.0055 -> 0.0033).
// What would close it: an SFPU chain element with DEST-strip operands (row / column broadcast from another slot) and
// a MAD form, plus LREG-resident operands across windows.
// Raw-LLK deviation (normalize_block, normalize_sfpu_macro): the SFPU MADs of the normalize (fp32 affine D * w + b;
// 16-bit (D * rstd + c'') [* w + b]) are issued as SFPLOADMACRO sequences (layer_norm_adaln_stats_sfpu.hpp nm_*): one
// macro instruction per DEST vector loads it, runs the MAD from an instruction template (operands in fixed LREGs) and
// stores it back on later cycles, so the SFPU retires one vector per instruction slot instead of separate SFPLOAD /
// SFPMAD / SFPSTORE slots (16-bit affine: macro + one explicit MAD + w / b loads, 4 slots per vector). Same fp32 MADs
// in the same order, so the output is bit-identical to the sfpi routines it replaces. Helper gap: no compute helper or
// eltwise_chain element exposes SFPLOADMACRO pipelining for a user MAD, and the existing LLK macro users
// (typecast / where / mul_int32 / reduce) are fixed functions. Measured (BH, math-TRISC c_normalize zone, 1x1x256x3072,
// 16 tiles per core): dynamic_wb fp32 4000 -> 3290 ns, dynamic_wb 16-bit 3271 -> 2948 ns, no-affine 16-bit 2481 ->
// 1542 ns; whole op (median of 3, alternating) 256x3072 noaff 16-bit -6.4%, 32x3072 wb fp32 -6.5%, 8192x256 wb fp32
// -2.5%; 256x3072 wb fp32 and the 2304x3072 focus flat (writer / DRAM bound after normalize).
// What would close it: a MAD chain element with a pipelined (load-macro) implementation and DEST-strip operands.
// normalize_window_fixed (Perf 4 of tournament 10): per-row fixed cost of normalize_block cut, same arithmetic (output
// bit-identical): fp32 DEST builds c' inside the row's first window acquire (no empty acquire / commit / release),
// computes only the 8 strip vectors it loads (nz_cprime_strips, was all 32), drops a redundant srcB reconfig and bcast
// re-init, and waits on cb_row_stats only at its first use so the P copy and pivot broadcast overlap the combine's
// pack -> L1 -> unpack round trip; 16-bit DEST issues the window's x - p subtractions before the P + S adds (the first
// window overlaps the S tiles' round trip), skips zeroing S faces 0..2 (each DEST half is cleared on release) and
// reconfigures the packer only on a format change. Measured (BH, math c_normalize, 1 rep): 256x3072 wb fp32 1864 ->
// 1598 ns, 992x1024 wb fp32 3244 -> 2904, 256x3072 wb 16-bit 1989 -> 1591; whole op (median of 3, alternating)
// 992x1024 wb fp32 -2.6%, 256x3072 wb fp32 -2.6%, wb 16-bit -2.8%, noaff -3.6 to -3.9%, 256x1024 -5.5%, 1024x64 -6%.
// (No new helper bypass: these edits stay inside the existing raw-LLK normalize path above.)
// Deviation from op_design.md: the dev pass keeps [sum (x - p), sum (x - p)^2] in DEST as 4-row strip accumulators and
// lane-reduces them in SFPU registers (no cb_dev_sq_acc / cb_dev_sq), and publishes per row inside the same acquire.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "layer_norm_adaln_stats_sfpu.hpp"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"

using namespace compute_kernel_lib;

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_reduce_scaler = 1;
constexpr uint32_t cb_pivot = 3;
constexpr uint32_t cb_local_stats = 5;
constexpr uint32_t cb_gathered_stats = 6;
constexpr uint32_t cb_row_stats = 7;
constexpr uint32_t cb_weight_rows = 10;
constexpr uint32_t cb_bias_rows = 11;
constexpr uint32_t cb_output_tiles = 14;
constexpr uint32_t cb_wbp = 12;     // affine: w / b P tiles (row-replicated strips, 3 columns per tile), per call
constexpr uint32_t cb_nstats = 13;  // 16-bit DEST: per-row rstd / c'' strip tile(s) for the SFPU normalize

// Copy one tile of `cb` into DEST slot `dst`, re-pointing the unpacker at cb's format first.
ALWI void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

// One call of a column-0 stats routine (VectorMode::None: the routine visits its 8 column-0 vectors itself; DEST base =
// tile 0, routines address their slots absolutely).
// Usage: STATS_SFPU_CALL((fn<T...>), extra args...) — parentheses keep template commas out of the macro split.
#define STATS_STRIP(...) __VA_ARGS__
#define STATS_SFPU_CALL(FN, ...)                 \
    MATH((_llk_math_eltwise_binary_sfpu_params_( \
        ckernel::sfpu::STATS_STRIP FN, 0u, 0u, 0u, VectorMode::None, ##__VA_ARGS__)))

ALWI void stats_sfpu_init() { MATH((ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::unused>())); }

// DEST tiles in one sync half (fp32: 4, 16-bit: 8).
constexpr uint32_t stats_dest_slots = DST_ACCUM_MODE ? 4 : 8;

// Streamed fold (perf_experiments/streamed_fold): one SFPU call per landed payload, both DEST modes.
template <uint32_t n_slice_bits>
ALWI void fold_one(bool first) {
    if constexpr (DST_ACCUM_MODE) {
        if (first) {
            STATS_SFPU_CALL((cf_fold_fp32_c<true>), n_slice_bits);
        } else {
            STATS_SFPU_CALL((cf_fold_fp32_c<false>), n_slice_bits);
        }
    } else {
        if (first) {
            STATS_SFPU_CALL((cf_fold_b16_c<true>), n_slice_bits);
        } else {
            STATS_SFPU_CALL((cf_fold_b16_c<false>), n_slice_bits);
        }
    }
}

// wide_groups ragged slices: slot j's slice has n_j elements per row (runtime), so the fold accumulates
// sum n_j delta_j and sum M2_j + n_j delta_j^2 (cf_fold_*_w); the finalize then runs with n = 1, inv_s = 1 / W.
ALWI void fold_one_w(bool first, uint32_t n_bits) {
    if constexpr (DST_ACCUM_MODE) {
        if (first) {
            STATS_SFPU_CALL((cf_fold_fp32_w<true>), n_bits);
        } else {
            STATS_SFPU_CALL((cf_fold_fp32_w<false>), n_bits);
        }
    } else {
        if (first) {
            STATS_SFPU_CALL((cf_fold_b16_w<true>), n_bits);
        } else {
            STATS_SFPU_CALL((cf_fold_b16_w<false>), n_bits);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t block_height = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);
    constexpr uint32_t num_w_slices = get_compile_time_arg_val(2);
    constexpr bool has_affine = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t n_slice_bits = get_compile_time_arg_val(4);
    constexpr uint32_t inv_n_slice_bits = get_compile_time_arg_val(5);
    constexpr uint32_t inv_slices_bits = get_compile_time_arg_val(6);
    constexpr uint32_t inv_width_bits = get_compile_time_arg_val(7);
    constexpr uint32_t stats_per_row = get_compile_time_arg_val(8);  // payload tiles per row tile (1)
    [[maybe_unused]] constexpr bool dev_sum_pass = get_compile_time_arg_val(9) != 0;
    static_assert(stats_per_row == 1, "payload: one tile per row, [r_hi, r_lo | M2_hi, M2_lo] | [p, M2 | d]");
    constexpr uint32_t chunk = get_compile_time_arg_val(10);  // reader push granularity within a row
    // Pivot: bf16 mean of the row's first pivot_tiles landed tiles (one reduce, packed bf16 -> cb_pivot).
    constexpr uint32_t pivot_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t inv_pivot_n_bits = get_compile_time_arg_val(12);
    constexpr bool ragged = get_compile_time_arg_val(13) != 0;  // RT args 3.. = per-slot n_j bits
    constexpr uint32_t one_bits = 0x3f800000u;
    constexpr uint32_t fin_n_bits = ragged ? one_bits : n_slice_bits;

    const uint32_t group_rows = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);
    const uint32_t rot = get_arg_val<uint32_t>(2);  // per-core start column (DRAM bank stagger; see the writer)

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer reduce_scaler(cb_reduce_scaler);
    DataflowBuffer pivot(cb_pivot);
    DataflowBuffer local_stats(cb_local_stats);
    DataflowBuffer gathered_stats(cb_gathered_stats);
    DataflowBuffer row_stats(cb_row_stats);
    DataflowBuffer output_tiles(cb_output_tiles);
    [[maybe_unused]] DataflowBuffer weight_rows(cb_weight_rows);
    [[maybe_unused]] DataflowBuffer bias_rows(cb_bias_rows);
    [[maybe_unused]] DataflowBuffer wbp(cb_wbp);
    [[maybe_unused]] DataflowBuffer nstats(cb_nstats);
    // P tiles: 3 columns each; 16-bit DEST pads to an even count (a window of 6 always reads two).
    constexpr uint32_t n_wbp = DST_ACCUM_MODE ? (block_width + 2) / 3 : 2 * ((block_width + 5) / 6);

    compute_kernel_hw_startup(cb_input_tiles, cb_reduce_scaler, cb_pivot);

    uint32_t rows_done = 0;
    for (uint32_t block_idx = 0; rows_done < group_rows; ++block_idx) {
        const uint32_t block_rows = (group_rows - rows_done) < block_height ? (group_rows - rows_done) : block_height;

        // ---------------- per-row stats, streamed behind the reader's chunked load (see the head comment) ------
        for (uint32_t r = 0; r < block_rows; ++r) {
            const uint32_t base = r * block_width;
            uint32_t landed = 0;
            {
                MaybeDeviceZoneScope("c_pivot");
                landed = ((pivot_tiles + chunk - 1) / chunk) * chunk;
                landed = landed > block_width ? block_width : landed;
                {
                    MaybeDeviceZoneScope("c_wait_input");  // starved on the reader's first chunk of row r
                    input_tiles.wait_front(base + landed);
                }
                // p = bf16(mean of the first pivot_tiles landed tiles): same reduce + scale as HEAD's mean pass,
                // packed straight to the bf16 pivot CB (HEAD: fp32 r, then a copy -> bf16 pack).
                reconfig_data_format(cb_reduce_scaler, cb_input_tiles);
                pack_reconfig_data_format(cb_pivot);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_input_tiles, cb_reduce_scaler, cb_pivot);
                reduce_scaler.wait_front(1);
                pivot.reserve_back(1);
                tile_regs_acquire();
                for (uint32_t t = 0; t < pivot_tiles; ++t) {
                    uint32_t c = rot + t;
                    c = c >= block_width ? c - block_width : c;
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_input_tiles, cb_reduce_scaler, base + c, 0, 0);
                }
                binop_with_scalar_tile_init();
                mul_unary_tile(0, inv_pivot_n_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_pivot);
                tile_regs_release();
                pivot.push_back(1);
                reduce_uninit(cb_input_tiles);
                pivot.wait_front(r + 1);
            }
            // dev + publish, one DEST acquire per row (stats_publish_fuse; dev_pass_macro strip accumulators):
            // D0 = the row's 8 + 8 strip accumulators s_g / q_g (sum (x - p), sum (x - p)^2 per 4-row strip, every
            // same-row vector of a tile folded into one; layer_norm_adaln_stats_sfpu.hpp ds_acc_strip), D1.. = x - p
            // work tiles (FPU sub, srcB Col-bcast p). The pivot tile is copied into D0 first and the first strip call
            // stashes p in D0's parity-1 vectors, so after the row's last window only the lane-reduce publish
            // (ds_publish_compact: 8-lane row sums in fp32 LREGs, payload algebra, compaction to the 1/8-tile prefix)
            // and the pack remain.
            constexpr uint32_t NWK = stats_dest_slots - 1;  // work slots per window (fp32: 3, 16-bit: 7)
            constexpr uint32_t N0 = block_width < NWK ? block_width : NWK;
            constexpr uint32_t NREM = (block_width - N0) % NWK;
            {
                MaybeDeviceZoneScope("c_dev_sq");
                if (r == 0) {
                    local_stats.reserve_back(block_rows);
                }
                tile_regs_acquire();
                copy_from(cb_pivot, r, 0);  // D0 = pivot tile (column 0 = p), stashed by the first strip call
                reconfig_data_format(cb_input_tiles, cb_pivot);
                sub_bcast_cols_init(cb_input_tiles, cb_pivot);
                stats_sfpu_init();
                for (uint32_t t0 = 0; t0 < block_width; t0 += NWK) {
                    const uint32_t n = (block_width - t0) < NWK ? (block_width - t0) : NWK;
                    if (t0 + n > landed) {
                        landed = ((t0 + n + chunk - 1) / chunk) * chunk;
                        if (landed > block_width) {
                            landed = block_width;
                        }
                        input_tiles.wait_front(base + landed);
                    }
                    if (t0 != 0) {
                        MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));  // SFPU done reading D1..
                    }
                    for (uint32_t j = 0; j < n; ++j) {
                        uint32_t c = rot + t0 + j;
                        c = c >= block_width ? c - block_width : c;
                        sub_tiles_bcast<BroadcastType::COL>(cb_input_tiles, cb_pivot, base + c, r, 1 + j);
                    }
                    if (t0 == 0) {
                        STATS_SFPU_CALL((ds_acc_strip<N0, true, true>));
                    } else if (n == NWK) {
                        STATS_SFPU_CALL((ds_acc_strip<NWK, false>));
                    } else {
                        if constexpr (NREM > 0) {
                            STATS_SFPU_CALL((ds_acc_strip<NREM, false>));
                        }
                    }
                }
            }
            // publish: one tile per row tile; only column 0 of faces 0/2 is meaningful, so the payload also uses
            // column 1 and faces 1/3 (the same SFPU lanes, see layer_norm_adaln_stats_sfpu.hpp "combine_lean"):
            //   fp32 DEST   : [r_hi, r_lo | M2_hi, M2_lo]  (tf32-exact hi/lo split: the combine copies the payload
            //                 through srcA instead of unpack-to-dest, and re-forms r = r_hi + r_lo, M2 = M2_hi + M2_lo)
            //   16-bit DEST : [p, M2 | d]                  (d = sum(x - p) / n; r itself is too coarse to carry)
            //   d = sum(x - p) / n, M2 = s2 - sum(x - p) * d, r = p + d — one SFPU pass over the strip accumulators.
            {
                MaybeDeviceZoneScope("c_publish");
                // stats_tail: lane-reduce + payload algebra + compaction in one SFPU pass (p from D0's parity-1 stash)
                STATS_SFPU_CALL((ds_publish_compact<1>), inv_n_slice_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_local_stats);
                pack_tile<true>(0, cb_local_stats, r);
                tile_regs_release();
                if (r + 1 == block_rows) {
                    local_stats.push_back(block_rows);
                }
            }
        }

        // ---------------- combine_block: Chan's parallel combine of S slices, in own pivot frame ----------------
        // One pass over the peers, pivot-relative (every term O(std)), both DEST modes:
        //   delta_j = r_j - p_own (fp32) | (p_j - p_own) + d_j (16-bit);  m = (1/S) sum delta_j;
        //   M2 = sum_j [M2_j + n delta_j^2] - n S m^2
        // Streamed (perf_experiments/streamed_fold): the writer pushes cb_gathered_stats one slot at a time (own
        // payload first, then the peers in landing order, see exchange_streamed) and each slot is folded as soon as it
        // lands (cumulative waits — free once landed), so after the last peer lands only one fold + finalize remain.
        // Multi-row blocks fold row-outer: only row 0 overlaps the exchange.
        // DEST (compact vectors, compact_stats_payload): D0 units 0 / 1 = [sum delta, sum q] (16-bit: units 0..3 =
        // their hi / lo compensated pairs), D1 unit 0 = p_own, D2 = the landed slot's compact payload. Each fold is one
        // SFPU vector per stat; the finalize applies rsqrt in place (no rsqrt_tile pass) and expands m / rstd to
        // column 0. Output [m.., rstd..].
        const uint32_t slot_tiles = block_rows;
        // Per-row prologue: D1 = p_own, unpacker re-pointed at the payload. Needs only the local pivot, so row 0's runs
        // before the all-gather rendezvous (hidden behind c_wait_gathered).
        auto combine_prologue = [&](uint32_t k) {
            tile_regs_acquire();
            copy_from(cb_pivot, k, 1);
            reconfig_data_format_srca(cb_gathered_stats);
            copy_init(cb_gathered_stats);
            stats_sfpu_init();
            STATS_SFPU_CALL((cf_compact_pivot));  // D1 unit 0 = compact p_own
        };
        if constexpr (has_affine) {
            if (block_idx == 0) {
                // sfpu_normalize: P tiles (w / b as row-replicated strips, 3 columns per tile in logical column order;
                // see layer_norm_adaln_stats_sfpu.hpp), built once per call while compute would otherwise idle on the
                // stats all-gather. Per column: w, b Row-bcast into slots 0, 1, then SFPU strip copies into slot 2.
                MaybeDeviceZoneScope("c_wbp_prep");
                weight_rows.wait_front(block_width);
                bias_rows.wait_front(block_width);
                wbp.reserve_back(n_wbp);
                pack_reconfig_data_format(cb_wbp);
                reconfig_data_format_srcb(cb_weight_rows);
                unary_bcast_init<BroadcastType::ROW>(cb_weight_rows);
                stats_sfpu_init();
                for (uint32_t i = 0; i < n_wbp; ++i) {
                    tile_regs_acquire();
                    MATH((_llk_math_eltwise_sfpu_start_(0)));
                    MATH((ckernel::sfpu::nz_wbp_zero_face3()));  // 16-bit: S lands in face 3 via an FPU add
                    MATH((_llk_math_eltwise_sfpu_done_()));
                    for (uint32_t g = 0; g < 3; ++g) {
                        uint32_t c = 3 * i + g;
                        if (c >= block_width) {
                            break;
                        }
                        c += rot;
                        c = c >= block_width ? c - block_width : c;
                        unary_bcast<BroadcastType::ROW>(cb_weight_rows, c, 0);
                        unary_bcast<BroadcastType::ROW>(cb_bias_rows, c, 1);
                        MATH((_llk_math_eltwise_sfpu_start_(0)));
                        if (g == 0) {
                            MATH((ckernel::sfpu::nz_wbp_prep<0>()));
                        } else if (g == 1) {
                            MATH((ckernel::sfpu::nz_wbp_prep<1>()));
                        } else {
                            MATH((ckernel::sfpu::nz_wbp_prep<2>()));
                        }
                        MATH((_llk_math_eltwise_sfpu_done_()));
                        MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(2, cb_wbp);
                    tile_regs_release();
                }
                wbp.push_back(n_wbp);
            }
        }
        row_stats.reserve_back(2 * block_rows);
        pack_reconfig_data_format(cb_row_stats);
        combine_prologue(0);
        {
            MaybeDeviceZoneScope("c_wait_gathered");  // first slot (own payload), seen from compute
            gathered_stats.wait_front(slot_tiles);
        }
        {
            MaybeDeviceZoneScope("c_combine");  // includes the waits on the later slots
            for (uint32_t k = 0; k < block_rows; ++k) {
                if (k > 0) {
                    combine_prologue(k);
                }
                for (uint32_t j = 0; j < num_w_slices; ++j) {
                    gathered_stats.wait_front((j + 1) * slot_tiles);      // free once landed (rows k > 0)
                    copy_tile(cb_gathered_stats, j * slot_tiles + k, 2);  // D2 = slot j's payload
                    if constexpr (ragged) {
                        fold_one_w(j == 0, get_arg_val<uint32_t>(3 + j));
                    } else {
                        fold_one<n_slice_bits>(j == 0);
                    }
                }
                STATS_SFPU_CALL((cf_finalize_c), fin_n_bits, inv_slices_bits, inv_width_bits, eps_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_row_stats, k);
                pack_tile<true>(1, cb_row_stats, block_rows + k);
                tile_regs_release();
            }
            row_stats.push_back(2 * block_rows);
            gathered_stats.pop_front(num_w_slices * slot_tiles);
        }

        // ---------------- normalize_block (sfpu_normalize) ----------------
        //   fp32 DEST  : D = c'_k (SFPU LREG preload), D += x * rstd (one FPU MAC), then [D = D * w + b on the SFPU,
        //                w / b from the window's P tile]. Window = 3 x tiles + 1 P tile (P face 3 = c' stash).
        //   16-bit DEST: D = x - p (FPU), then y = (D * rstd + c'') [* w + b] on the SFPU in fp32 LREGs, c'' = -m *
        //   rstd,
        //                rstd / c'' from the per-row stats tile(s) S (cb_nstats). Affine: 6 x + 2 (P + S) slots, one
        //                FPU add lands P and S in one slot; no affine: 7 x + 1 S slot.
        // Each window is pushed as soon as it is packed, so the writer stores a row while the rest of it normalizes.
        {
            MaybeDeviceZoneScope("c_normalize");
            constexpr uint32_t nb = DST_ACCUM_MODE ? (has_affine ? 3 : DEST_AUTO_LIMIT) : (has_affine ? 6 : 7);
            constexpr uint32_t s_slot = nb;  // first P / S slot
            // Logical tile c of a row is physical column (c + rot) mod block_width — the order the writer stores in.
            auto phys_col = [rot](uint32_t c) -> uint32_t {
                const uint32_t p = c + rot;
                return p >= block_width ? p - block_width : p;
            };
            input_tiles.wait_front(block_rows * block_width);
            // row_stats ([m_0 .. m_bh-1, rstd_0 .. rstd_bh-1]) is waited on at its first use, so the stats-independent
            // work of the first window (P copy, pivot broadcast) overlaps the combine's pack -> L1 -> unpack round
            // trip.
            if constexpr (has_affine) {
                wbp.wait_front(n_wbp);
            }
            pack_reconfig_data_format(cb_output_tiles);
            reconfig_data_format(cb_input_tiles, cb_pivot);
            stats_sfpu_init();
            MATH((ckernel::sfpu::nm_macro_init()));  // SFPLOADMACRO templates / sequences (clobbers LREG0 only)
            for (uint32_t k = 0; k < block_rows; ++k) {
                if constexpr (DST_ACCUM_MODE) {
                    for (uint32_t c0 = 0; c0 < block_width; c0 += nb) {
                        const uint32_t n = (block_width - c0) < nb ? (block_width - c0) : nb;
                        output_tiles.reserve_back(n);
                        tile_regs_acquire();
                        if constexpr (has_affine) {
                            copy_init(cb_wbp);
                            copy_tile(cb_wbp, c0 / 3, s_slot);  // P: w / b strips of columns c0 .. c0 + 2
                        }
                        if (c0 == 0) {
                            // c'_k = -(p + m) * rstd (fp32) -> LREG0..7, built in the row's first window (slots 0..2
                            // are overwritten by the c' stores below). p, m, rstd reach DEST through the same srcB
                            // broadcast path the MAC uses (bf16-exact p; tf32 m, rstd = the rstd the MAC uses). Only
                            // the 8 strip vectors nz_load_cprime reads are computed. srcB enters as cb_pivot.
                            unary_bcast_init<BroadcastType::COL>(cb_pivot);
                            unary_bcast<BroadcastType::COL>(cb_pivot, k, 0);
                            reconfig_data_format_srcb(cb_pivot, cb_row_stats);
                            row_stats.wait_front(2 * block_rows);  // cumulative: free after row 0
                            unary_bcast<BroadcastType::COL>(cb_row_stats, k, 1);
                            unary_bcast<BroadcastType::COL>(cb_row_stats, block_rows + k, 2);
                            MATH((_llk_math_eltwise_sfpu_start_(0)));
                            MATH((ckernel::sfpu::nz_cprime_strips()));
                            MATH((ckernel::sfpu::nz_load_cprime()));
                        }
                        // D[j] = c'_k
                        for (uint32_t j = 0; j < n; ++j) {
                            MATH((_llk_math_eltwise_sfpu_start_(j)));
                            MATH((ckernel::sfpu::nz_store_cprime_tile()));
                        }
                        if constexpr (has_affine) {
                            MATH((_llk_math_eltwise_sfpu_start_(0)));
                            MATH((ckernel::sfpu::nz_stash_cprime()));  // the affine below clobbers LREG0..7
                        }
                        MATH((_llk_math_eltwise_sfpu_done_()));
                        MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
                        // D[j] += x * rstd   (srcA = x, srcB = rstd Col-bcast; ELWMUL accumulates into DEST)
                        mul_bcast_cols_init(cb_input_tiles, cb_row_stats);
                        for (uint32_t j = 0; j < n; ++j) {
                            mul_tiles_bcast_cols(
                                cb_input_tiles, cb_row_stats, k * block_width + phys_col(c0 + j), block_rows + k, j);
                        }
                        if constexpr (has_affine) {
                            // D[j] = D[j] * w + b on the SFPU (fp32), then restore c' in LREG0..7
                            MATH((_llk_math_eltwise_sfpu_start_(0)));
                            MATH((ckernel::sfpu::nm_affine_window<3>(n)));
                            MATH((ckernel::sfpu::nz_unstash_cprime()));
                            MATH((_llk_math_eltwise_sfpu_done_()));
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t j = 0; j < n; ++j) {
                            pack_tile(j, cb_output_tiles);
                        }
                        tile_regs_release();
                        output_tiles.push_back(n);
                    }
                    reconfig_data_format_srcb(cb_row_stats, cb_pivot);  // entry state for the next row / block
                    continue;
                }
                // 16-bit DEST: per-row stats tile(s) S (rstd / c'' strip vectors at fixed DEST positions, zero
                // elsewhere)
                constexpr uint32_t ns = has_affine ? 2 : 1;
                nstats.reserve_back(ns);
                row_stats.wait_front(2 * block_rows);  // cumulative: free after row 0
                tile_regs_acquire();
                reconfig_data_format_srcb(cb_pivot, cb_row_stats);
                unary_bcast_init<BroadcastType::COL>(cb_row_stats);
                unary_bcast<BroadcastType::COL>(cb_row_stats, k, 0);
                unary_bcast<BroadcastType::COL>(cb_row_stats, block_rows + k, 1);
                stats_sfpu_init();
                MATH((_llk_math_eltwise_sfpu_start_(0)));
                if constexpr (has_affine) {
                    MATH((ckernel::sfpu::nz_stats16_build()));  // slot2 = S_r, slot3 = S_c (face 3 each)
                } else {
                    MATH((ckernel::sfpu::nz_stats16_build_noaff()));  // slot2 = [0 | 0 | rstd | c'']
                }
                MATH((_llk_math_eltwise_sfpu_done_()));
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output_tiles, cb_nstats);
                pack_tile(2, cb_nstats);
                if constexpr (has_affine) {
                    pack_tile(3, cb_nstats);
                }
                pack_reconfig_data_format(cb_nstats, cb_output_tiles);
                tile_regs_release();
                nstats.push_back(ns);
                reconfig_data_format_srcb(cb_row_stats, cb_pivot);  // S / P are bf16 like the pivot
                for (uint32_t c0 = 0; c0 < block_width; c0 += nb) {
                    const uint32_t n = (block_width - c0) < nb ? (block_width - c0) : nb;
                    output_tiles.reserve_back(n);
                    tile_regs_acquire();
                    // D[j] = x - p (srcA = x, srcB = pivot Col-bcast; exact in DEST). Issued before the P + S slots
                    // so the row's first window overlaps the S tiles' pack -> L1 -> unpack round trip.
                    sub_bcast_cols_init(cb_input_tiles, cb_pivot);
                    for (uint32_t j = 0; j < n; ++j) {
                        sub_tiles_bcast<BroadcastType::COL>(
                            cb_input_tiles, cb_pivot, k * block_width + phys_col(c0 + j), k, j);
                    }
                    nstats.wait_front(ns);  // cumulative: free after the row's first window
                    if constexpr (has_affine) {
                        // slots 6, 7 = P (w / b strips, faces 0..2) + S (row stats, face 3): one FPU add each
                        add_init(cb_wbp, cb_nstats);
                        add_tiles(cb_wbp, cb_nstats, c0 / 3, 0, s_slot);
                        add_tiles(cb_wbp, cb_nstats, c0 / 3 + 1, 1, s_slot + 1);
                    } else {
                        copy_init(cb_nstats);
                        copy_tile(cb_nstats, 0, s_slot);
                    }
                    MATH((_llk_math_eltwise_sfpu_start_(0)));
                    if constexpr (has_affine) {
                        MATH((ckernel::sfpu::nm_full16_window(n)));  // (D * rstd + c'') * w + b
                    } else {
                        MATH((ckernel::sfpu::nm_norm16_window<7>(n)));  // D * rstd + c''
                    }
                    MATH((_llk_math_eltwise_sfpu_done_()));
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < n; ++j) {
                        pack_tile(j, cb_output_tiles);
                    }
                    tile_regs_release();
                    output_tiles.push_back(n);
                }
                nstats.pop_front(ns);
            }
            input_tiles.pop_front(block_rows * block_width);
            row_stats.pop_front(2 * block_rows);
            pivot.pop_front(block_rows);
        }
        rows_done += block_rows;
    }

    // Release the call-resident operands.
    if constexpr (has_affine) {
        weight_rows.pop_front(block_width);
        bias_rows.pop_front(block_width);
        wbp.pop_front(n_wbp);
    }
    reduce_scaler.pop_front(1);
}
