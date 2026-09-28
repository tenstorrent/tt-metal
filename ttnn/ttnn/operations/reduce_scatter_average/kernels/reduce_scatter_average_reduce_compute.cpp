// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter_average — reduce compute (TRISC), core (0,2).
//
// Arrival-ordered incremental N-way mean over this device's S-tile output slice
// (op_design.md "Compute Phases"). Order-agnostic by construction: it counts N
// contributions of S tiles in g-granules; the READER decides arrival order.
//
//   C0  binary_op_init_common (hw startup — accumulate_helpers pre-condition) +
//       BlockAccumulate::arm over (cb_contributions, cb_accumulator, cb_accumulator).
//   C1  Seed: contribution 0 (own slice) copied into the resident accumulator via
//       S/g x sum_blocks(num_blocks=1) — a copy of block 0. pop_input=true is
//       load-bearing (R4): the default false deadlocks the reader on
//       cb_reserve_back once cb_contributions fills.
//   C2  acc.rearm(): sum_blocks leaves add_tiles_init in acc_to_dest mode and the
//       formats possibly reprogrammed (R3); rearm restores both.
//   C3  Incremental accumulate: (N-1) x S/g x acc.run(g). IN-PLACE cb_b == cb_out
//       is sound: run() pops a and b BEFORE reserving out (verified ordering), so
//       with capacity exactly S the reserve always finds g free pages. FIFO order
//       = walker order preserved every pass, keeping the adds positionally aligned.
//   C4  1/N scale (raw mul_tiles_bcast_scalar — the designated eltwise helper is
//       absent from this clone; every other helper considered is rejected with
//       citations in op_design.md "Raw-API justifications"): init_short strictly
//       AFTER the last run() (R10 — it reprograms the binary-op state), then S/g
//       granule passes cb_accumulator x cb_scaler(0,0) -> cb_averaged. cb_scaler is
//       waited once (count 1) and never popped. All CBs share the input dtype, so
//       the boot init's data formats cover C4 with no mid-kernel reconfig (R12).
//
// CT args: [cb_contributions, cb_scaler, cb_accumulator, cb_averaged, ring_size, S, g]

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_contributions = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_accumulator = get_compile_time_arg_val(2);
    constexpr uint32_t cb_averaged = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);  // N contributions
    constexpr uint32_t S = get_compile_time_arg_val(5);          // slice tiles
    constexpr uint32_t g = get_compile_time_arg_val(6);          // granule (divides S)

    static_assert(g > 0 && S % g == 0, "reduce_scatter_average: granule must divide the slice tile count");
    static_assert(
        g <= compute_kernel_lib::DEST_AUTO_LIMIT,
        "reduce_scatter_average: granule exceeds DEST capacity (4 under fp32_dest_acc_en + SyncHalf)");
    constexpr uint32_t chunks = S / g;

    // C0 — hardware startup stays with the kernel (helper banner: binary_op_init_common
    // and compute_kernel_hw_startup are NOT interchangeable; the accumulate helpers
    // deliberately never pick one). Then arm once, outside the loops.
    binary_op_init_common(cb_contributions, cb_accumulator, cb_averaged);
    auto acc = compute_kernel_lib::BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g);

    // C1 — seed the resident accumulator with the own contribution (a block-0 copy).
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        compute_kernel_lib::sum_blocks(
            cb_contributions, cb_accumulator, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true);
    }

    // C2 — restore after sum_blocks's acc_to_dest post-condition (R3).
    acc.rearm();

    // C3 — one incremental pass per remaining contribution, the moment it streams in.
    for (uint32_t k = 1; k < ring_size; ++k) {
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            acc.run(g);
        }
    }

    // C4 — broadcast-scalar multiply by 1/N: accumulator (full 32x32 tiles) x
    // cb_scaler tile 0 element (0,0). The scaler is persistent: one wait, no pop.
    cb_wait_front(cb_scaler, 1);
    mul_tiles_bcast_scalar_init_short(cb_accumulator, cb_scaler);
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        cb_wait_front(cb_accumulator, g);
        tile_regs_acquire();
        for (uint32_t t = 0; t < g; ++t) {
            mul_tiles_bcast_scalar(cb_accumulator, cb_scaler, t, 0, t);
        }
        tile_regs_commit();
        // Pop before reserving out (the accumulate helpers' verified ordering).
        cb_pop_front(cb_accumulator, g);
        cb_reserve_back(cb_averaged, g);
        tile_regs_wait();
        for (uint32_t t = 0; t < g; ++t) {
            pack_tile(t, cb_averaged);  // in-order pack mode: tiles append per push window
        }
        tile_regs_release();
        cb_push_back(cb_averaged, g);
    }
}
