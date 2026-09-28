// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — reduce compute (TRISC), core (0,2).
//
// Arrival-ordered incremental N-way SUM over the FULL P-tile shard (op_design.md
// "Compute Phases"). Order-agnostic by construction: it counts N contributions of P
// tiles in g-granules; the READER decides arrival order. All-helper: every phase is
// a documented accumulate-helper call (SUM needs no scaler, so the reference's raw
// bcast-scalar pass is deleted; the drain is the helper's degenerate copy).
//
//   C0  binary_op_init_common (hw startup — accumulate_helpers pre-condition,
//       accum.hpp:116-117; NOT interchangeable with compute_kernel_hw_startup) +
//       BlockAccumulate::arm over (cb_contributions, cb_accumulator, cb_accumulator).
//   C1  Seed: contribution 0 (own shard) copied into the resident accumulator via
//       P/g x sum_blocks(num_blocks=1) — a documented copy of block 0 (accum.hpp:217).
//       pop_input=true is load-bearing (R4): the default false deadlocks the reader
//       on cb_reserve_back once cb_contributions fills.
//   C2  acc.rearm(): sum_blocks leaves add_tiles_init in acc_to_dest mode and the
//       formats possibly reprogrammed (R3); rearm restores both.
//   C3  Incremental accumulate: (N-1) x P/g x acc.run(g). IN-PLACE cb_b == cb_out is
//       sound: run() pops a and b BEFORE reserving out (verified ordering), so with
//       capacity exactly P the reserve always finds g free pages. FIFO order = dense
//       page order preserved every pass, keeping the adds positionally aligned (R11).
//   C4  Drain: P/g x sum_blocks(cb_accumulator, cb_summed, 1, g, true) — a degenerate
//       helper copy streaming the final sum to the writer. Runs strictly AFTER the
//       last run(); nothing follows, so its acc_to_dest post-condition is moot (R3).
//
// All CBs share the input dtype, so the boot init's data formats cover every phase
// with zero mid-kernel reconfig (C2's rearm() re-establishes them anyway).
//
// CT args: [cb_contributions, cb_accumulator, cb_summed, ring_size, P, g]. No rt args.

#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_contributions = get_compile_time_arg_val(0);
    constexpr uint32_t cb_accumulator = get_compile_time_arg_val(1);
    constexpr uint32_t cb_summed = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);  // N contributions
    constexpr uint32_t P = get_compile_time_arg_val(4);          // tiles per shard = output tiles
    constexpr uint32_t g = get_compile_time_arg_val(5);          // granule (divides P)

    static_assert(g > 0 && P % g == 0, "all_reduce: granule must divide the shard tile count (R5)");
    static_assert(
        g <= compute_kernel_lib::DEST_AUTO_LIMIT,
        "all_reduce: granule exceeds DEST capacity (4 under fp32_dest_acc_en + SyncHalf)");
    constexpr uint32_t chunks = P / g;

    // C0 — hardware startup stays with the kernel (helper banner: binary_op_init_common
    // and compute_kernel_hw_startup are NOT interchangeable; the accumulate helpers
    // deliberately never pick one). Then arm once, outside the loops (R12: one armed
    // accumulator per kernel).
    binary_op_init_common(cb_contributions, cb_accumulator, cb_summed);
    auto acc = compute_kernel_lib::BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g);

    // C1 — seed the resident accumulator with the own contribution (a block-0 copy).
    // The seed goes through cb_contributions + sum_blocks, never a reader push into
    // cb_accumulator (R2: single-producer invariant — compute owns that CB's ring).
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        compute_kernel_lib::sum_blocks(
            cb_contributions, cb_accumulator, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true);
    }

    // C2 — restore after sum_blocks's acc_to_dest + format post-condition (R3).
    acc.rearm();

    // C3 — one incremental pass per remaining contribution, the moment it streams in.
    // Arrival-major on purpose (R15): pass k overlaps the fabric flight of arrival k+1.
    for (uint32_t k = 1; k < ring_size; ++k) {
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            acc.run(g);
        }
    }

    // C4 — drain the resident final sum to the writer via the helper's degenerate
    // copy (num_blocks == 1). pop_input=true empties cb_accumulator (R4).
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        compute_kernel_lib::sum_blocks(
            cb_accumulator, cb_summed, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true);
    }
}
