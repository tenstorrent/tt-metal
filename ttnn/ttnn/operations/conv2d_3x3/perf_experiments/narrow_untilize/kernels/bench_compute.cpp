// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated untilize bench (compute only, one core). The bf8_b input tile-rows are resident in a shard-backed CB
// (the kernel pushes them itself: the reader stand-in), the bf16 row-major slab lands in a shard-backed CB of one
// 2 KB page per tile (the gather stand-in pops it). `reps` M-blocks, each one full untilize call (init .. uninit),
// as conv2d_3x3's compute_untilize does once per M-block.
//
// variant 0: compute_kernel_lib::untilize<ct> exactly as conv2d_3x3 calls it (baseline)
// variant 1: narrow_untilize::untilize_1wide, regular untilize pack MOP (ct == 1)
// variant 2: narrow_untilize::untilize_1wide, contiguous-stream pack (ct == 1)
// variant 3: narrow_untilize::untilize_wide_batched (ct >= 2)
// variant 4: as 2 with CB sync per DEST section instead of per batch
// variant 5: narrow_untilize::untilize_1wide, strided pack (Last=1 per row, ch1 Y addressing), CB sync per section
// variant 6: diagnostic pack-only on variant 5
// variants 1/2 sync the CBs once per batch
// variant 10/11: diagnostics on variant 1 (no pack / pack only); 13: pack-only diagnostic on variant 2
// NU_TILIZE_ROWS > 0 (compile arg 6): after each untilize, tilize 1-tile RM rows (the op's next pack phase on its
// tap-major path) and check them: this is what caught the ch0-Y exit-state bug (see narrow_untilize.hpp).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "narrow_untilize.hpp"

namespace ckl = compute_kernel_lib;
namespace nu = narrow_untilize;

void kernel_main() {
    constexpr uint32_t CB_IN = 0;
    constexpr uint32_t CB_OUT = 16;
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t rows = get_compile_time_arg_val(2);
    constexpr uint32_t reps = get_compile_time_arg_val(3);
    constexpr uint32_t batch_rows = get_compile_time_arg_val(4);
    constexpr uint32_t rows_per_section = get_compile_time_arg_val(5);
    constexpr uint32_t pages = rows * ct;
    // > 0: after each untilize, tilize this many 1-tile RM tile-rows (CB 1 -> CB 2), the op's next pack phase.
    constexpr uint32_t tilize_rows = get_compile_time_arg_val(6);
    constexpr uint32_t CB_RM = 1;
    constexpr uint32_t CB_TILES = 2;

    compute_kernel_hw_startup(CB_IN, CB_OUT);
    DataflowBuffer in(CB_IN);
    DataflowBuffer out(CB_OUT);
    DataflowBuffer rm(CB_RM);
    DataflowBuffer tiles(CB_TILES);

    for (uint32_t rep = 0; rep < reps; ++rep) {
        cb_reserve_back(CB_IN, pages);
        cb_push_back(CB_IN, pages);
        {
            MaybeDeviceZoneScope("bench_untilize");
            pack_reconfig_l1_acc(0);
            if constexpr (variant == 0) {
                ckl::untilize<ct, CB_IN, CB_OUT>(in, out, rows);
            } else if constexpr (variant == 1) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::RowMop, rows_per_section, nu::Diag::None, true>(
                    in, out, rows, batch_rows);
            } else if constexpr (variant == 2) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::Contig, rows_per_section, nu::Diag::None, true>(
                    in, out, rows, batch_rows);
            } else if constexpr (variant == 4) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::Contig, rows_per_section>(in, out, rows, batch_rows);
            } else if constexpr (variant == 5) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::Strided, rows_per_section>(in, out, rows, batch_rows);
            } else if constexpr (variant == 6) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::Strided, rows_per_section, nu::Diag::PackOnly>(
                    in, out, rows, batch_rows);
            } else if constexpr (variant == 3) {
                nu::untilize_wide_batched<ct, CB_IN, CB_OUT>(in, out, rows, batch_rows);
            } else if constexpr (variant == 10) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::RowMop, rows_per_section, nu::Diag::NoPack>(
                    in, out, rows, batch_rows);
            } else if constexpr (variant == 11) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::RowMop, rows_per_section, nu::Diag::PackOnly>(
                    in, out, rows, batch_rows);
            } else if constexpr (variant == 13) {
                nu::untilize_1wide<CB_IN, CB_OUT, nu::Pack::Contig, rows_per_section, nu::Diag::PackOnly>(
                    in, out, rows, batch_rows);
            }
        }
        cb_wait_front(CB_OUT, pages);
        cb_pop_front(CB_OUT, pages);
        if constexpr (tilize_rows > 0) {
            cb_reserve_back(CB_RM, tilize_rows);
            cb_push_back(CB_RM, tilize_rows);
            ckl::tilize<1, CB_RM, CB_TILES>(rm, tiles, tilize_rows);
            cb_wait_front(CB_TILES, tilize_rows);
            cb_pop_front(CB_TILES, tilize_rows);
        }
    }
}
