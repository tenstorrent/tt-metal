// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated wide-untilize bench (compute only, one core). The bf8_b / bf16 input tile-rows are resident in a
// shard-backed CB (the kernel pushes them itself: the reader stand-in), the bf16 row-major slab lands in a
// shard-backed CB of one 2 KB page per tile (cb_halo_slab's page). `reps` M-blocks, each one full untilize call
// (init .. uninit), as conv2d_3x3's compute_untilize runs once per M-block. fp32 DEST, half sync (the op's contract).
//
// variant 0: compute_kernel_lib::untilize<ct> exactly as conv2d_3x3 calls it (baseline)
// variant 1: wide_untilize::untilize_base (replica of the helper's fast_untilize loop), 2/3: its NoPack / PackOnly
// variant 10: wide_untilize::untilize_wide, CB sync per section group; 11: per batch; 14: hybrid (input per batch,
// output reserved per batch and pushed per group)
// variant 19: lazy sync (input waited per group, popped per batch; output reserved per batch, pushed per group)
// variant 16: hybrid + folded unpack MOP (half-row mode); 17/18: NoPack of 16 / of hybrid
// variant 12/13/15: NoPack / PackOnly / UnpackOnly (no math copy, no pack) diagnostics of 10
// tilize_rows > 0: after each untilize, tilize that many tilize_w-wide RM tile-rows (CB 1 -> CB 2), as the op's next
// pack phase does, so a leaked packer state shows up as a wrong tilize.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "wide_untilize.hpp"

namespace ckl = compute_kernel_lib;
namespace wu = wide_untilize;

void kernel_main() {
    constexpr uint32_t CB_IN = 0;
    constexpr uint32_t CB_OUT = 16;
    constexpr uint32_t CB_RM = 1;
    constexpr uint32_t CB_TILES = 2;
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    constexpr uint32_t reps = get_compile_time_arg_val(3);
    constexpr uint32_t batch_rows = get_compile_time_arg_val(4);
    constexpr uint32_t tilize_rows = get_compile_time_arg_val(5);
    constexpr uint32_t tilize_w = get_compile_time_arg_val(6);
    constexpr uint32_t pages = rows * ct;

    compute_kernel_hw_startup(CB_IN, CB_OUT);
    DataflowBuffer in(CB_IN);
    DataflowBuffer out(CB_OUT);
    DataflowBuffer rm(CB_RM);
    DataflowBuffer tiles(CB_TILES);

    for (uint32_t rep = 0; rep < reps; ++rep) {
        cb_reserve_back(CB_IN, pages);
        cb_push_back(CB_IN, pages);
        {
            MaybeDeviceZoneScope("compute_untilize");
            pack_reconfig_l1_acc(0);
            if constexpr (variant == 0) {
                ckl::untilize<ct, CB_IN, CB_OUT>(in, out, rows);
            } else if constexpr (variant == 1) {
                wu::untilize_base<ct, CB_IN, CB_OUT, wu::Diag::None>(in, out, rows);
            } else if constexpr (variant == 2) {
                wu::untilize_base<ct, CB_IN, CB_OUT, wu::Diag::NoPack>(in, out, rows);
            } else if constexpr (variant == 3) {
                wu::untilize_base<ct, CB_IN, CB_OUT, wu::Diag::PackOnly>(in, out, rows);
            } else if constexpr (variant == 10) {
                wu::untilize_wide<ct, CB_IN, CB_OUT>(in, out, rows, batch_rows);
            } else if constexpr (variant == 11) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::None, wu::Sync::Batch>(in, out, rows, batch_rows);
            } else if constexpr (variant == 14) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::None, wu::Sync::Hybrid>(in, out, rows, batch_rows);
            } else if constexpr (variant == 16) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::None, wu::Sync::Hybrid, true>(in, out, rows, batch_rows);
            } else if constexpr (variant == 17) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::NoPack, wu::Sync::Hybrid, true>(
                    in, out, rows, batch_rows);
            } else if constexpr (variant == 18) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::NoPack, wu::Sync::Hybrid>(in, out, rows, batch_rows);
            } else if constexpr (variant == 19) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::None, wu::Sync::Lazy>(in, out, rows, batch_rows);
            } else if constexpr (variant == 15) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::UnpackOnly>(in, out, rows, batch_rows);
            } else if constexpr (variant == 12) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::NoPack>(in, out, rows, batch_rows);
            } else if constexpr (variant == 13) {
                wu::untilize_wide<ct, CB_IN, CB_OUT, wu::Diag::PackOnly>(in, out, rows, batch_rows);
            }
        }
        cb_wait_front(CB_OUT, pages);
        cb_pop_front(CB_OUT, pages);
        if constexpr (tilize_rows > 0) {
            cb_reserve_back(CB_RM, tilize_rows * tilize_w);
            cb_push_back(CB_RM, tilize_rows * tilize_w);
            ckl::tilize<tilize_w, CB_RM, CB_TILES>(rm, tiles, tilize_rows);
            cb_wait_front(CB_TILES, tilize_rows * tilize_w);
            cb_pop_front(CB_TILES, tilize_rows * tilize_w);
        }
    }
}
