// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// argmax_sfpu_tile_compute.cpp — SFPU TILE-layout last-dim argmax, PHASE 1.
//
// Lane-parallel vertical reduce: for each 32-row tile-row pass, reduce this
// core's `w_count` input tiles down to ONE (max value, winning tile index)
// candidate pair per DST lane — i.e. per (row, column) position — using the
// argmax_nc_compute.cpp op chain:
//
//   DST slot 0: running max_val  (fp32; bf16 inputs widen exactly)
//   DST slot 1: running win_tile (uint32 = LOCAL index of the winning tile t;
//               the lane's own column c is implicit — the consumer
//               reconstructs the element index as (w_start + t) * 32 + c)
//   DST slots 2/3: scratch (new_val / new_tile-idx, gt mask)
//
//   per input tile t (5 DST-wide ops):
//     copy_tile             new_val <- tile t
//     gt_binary_tile        mask = (new_val > max_val)          [IEEE fp32 >]
//     where<Float32>        max_val  = mask ? new_val : max_val
//     fill_tile_int<UInt32> scratch  = t
//     where<Int32>          win_tile = mask ? t : win_tile
//
// Storing the winning TILE index (a constant per step, like the NC kernel's
// k) instead of the global element index removes the need for a persistent
// column-ramp tile in DST and an add_int32 step: lane c IS column c. The
// uint32-CB/SrcA corruption documented in argmax_nc_compute.cpp is never in
// play — indices are materialized in DST via fill_tile_int and only ever
// PACKED out. A single where_tile_init() serves both the fp32 and int32
// updates (the NC kernel's SFPCONFIG-macro-collision dodge, kept verbatim).
//
// The candidate pair is packed out once per pass (max values -> bf16 CB,
// winning tile indices -> UInt32 CB); PHASE 2 — the horizontal reduce of the
// 32 per-column candidates per row, plus the cross-core merge when running
// multicore — is 32 scalar lexicographic compares per row on the dataflow
// RISC (reader_argmax_sfpu_tile.cpp). Phase 2 is O(32) per row vs phase 1's
// O(32 * w_count) per lane, so it amortizes to noise for large widths.
//
// SEMANTICS (documented divergence from the incumbent scalar TILE reader;
// all silicon-measured on Blackhole, planted special-value probes):
//   The pipeline is IEEE-compare-on-fp32 behind a bf16 special-value gasket:
//   * NaN behaves as SAME-SIGNED INFINITY end-to-end. A qNaN anywhere in the
//     row WINS the argmax (same index the incumbent reports for single-NaN
//     rows) but the max-value output reads 0x7F80 (+inf), not the NaN
//     payload; -NaN acts as -inf and never wins. A row holding both a +inf
//     and a later NaN reports the +inf's index (they tie as +inf; the
//     incumbent's bit-pattern order picks the NaN).
//   * -0 flushes to +0 in the value output; +0/-0 compare equal, so the
//     FIRST zero's index is kept (the incumbent's total order prefers a
//     later +0 over an earlier -0).
//   * Denormals flush to zero before the compare (the incumbent ranks them
//     normally) — same family as the known Blackhole eltwise min-normal
//     flush behavior.
//   * The max-value output carries a +2^-127 additive pack bias, visible
//     only when the winner's magnitude is below ~2^-118; compare order and
//     the index are unaffected.
//   Everything finite and normal — including every exact tie — matches the
//   incumbent's bfloat16_greater + smallest-index semantics bit-for-bit:
//   strict-gt per lane keeps the lowest tile, and phase 2's lexicographic
//   rule keeps the lowest global index across columns (and cores).
//   PRECEDENT: ttnn.argmax already ships this divergence class between its
//   own paths — the NC compute kernel (dim < rank-2) uses the same IEEE
//   gt_binary_tile chain while the scalar readers use the bfloat16_greater
//   bit order, so NaN/signed-zero corners already differ by dispatched dim.
//
// Compile-time args: cb_in, cb_res_val, cb_res_idx, num_passes.
// Runtime args: [0] w_count — this core's tile count per pass (>= 1).
// =============================================================================

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr auto cb_in = static_cast<tt::CBIndex>(get_compile_time_arg_val(0));
    constexpr auto cb_res_val = static_cast<tt::CBIndex>(get_compile_time_arg_val(1));
    constexpr auto cb_res_idx = static_cast<tt::CBIndex>(get_compile_time_arg_val(2));
    constexpr uint32_t num_passes = get_compile_time_arg_val(3);

    const uint32_t w_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst_max = 0;
    constexpr uint32_t dst_argmax = 1;
    constexpr uint32_t dst_scratch_a = 2;
    constexpr uint32_t dst_scratch_b = 3;

    DataflowBuffer dfb_in(cb_in);
    DataflowBuffer dfb_val(cb_res_val);
    DataflowBuffer dfb_idx(cb_res_idx);

    // One-time hardware config, then the copy-op pipeline init that feeds the
    // SFPU chain below. (This pair is exactly what the deprecated
    // init_sfpu(icb, ocb) forwarded to; same model as argmax_nc_compute.cpp.)
    compute_kernel_hw_startup(cb_in, cb_res_val);
    copy_init(cb_in);
    // Single where_tile_init for BOTH updates (fp32 max / int32 argmax) —
    // where_tile_init and binary_max_min_init both write SFPCONFIG macros 0
    // and 1, so they cannot coexist; see argmax_nc_compute.cpp.
    gt_binary_tile_init();
    where_tile_init();
    fill_tile_init();

    for (uint32_t pass = 0; pass < num_passes; ++pass) {
        tile_regs_acquire();

        // --- init from tile 0: max_val <- tile0, win_tile <- 0 ---
        dfb_in.wait_front(onetile);
        copy_tile(cb_in, 0, dst_max);
        dfb_in.pop_front(onetile);
        fill_tile_int<DataFormat::UInt32>(dst_argmax, 0u);

        // --- reduce tiles t = 1 .. w_count-1 (5 DST-wide ops per tile) ---
        for (uint32_t t = 1; t < w_count; ++t) {
            dfb_in.wait_front(onetile);
            copy_tile(cb_in, 0, dst_scratch_a);
            dfb_in.pop_front(onetile);

            gt_binary_tile(dst_scratch_a, dst_max, dst_scratch_b);
            where_tile<DataFormat::Float32>(dst_scratch_b, dst_scratch_a, dst_max, dst_max);
            fill_tile_int<DataFormat::UInt32>(dst_scratch_a, t);
            where_tile<DataFormat::Int32>(dst_scratch_b, dst_scratch_a, dst_argmax, dst_argmax);
        }

        tile_regs_commit();

        dfb_val.reserve_back(onetile);
        dfb_idx.reserve_back(onetile);
        tile_regs_wait();
        if (pass > 0) {
            // Passes after the first left the packer configured for the
            // UInt32 index CB; restore the value format first.
            pack_reconfig_data_format(cb_res_idx, cb_res_val);
        }
        pack_tile(dst_max, cb_res_val);
        pack_reconfig_data_format(cb_res_val, cb_res_idx);
        pack_tile(dst_argmax, cb_res_idx);
        tile_regs_release();
        dfb_val.push_back(onetile);
        dfb_idx.push_back(onetile);
    }
}
