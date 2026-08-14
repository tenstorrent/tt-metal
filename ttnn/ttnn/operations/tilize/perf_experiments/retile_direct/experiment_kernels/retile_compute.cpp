// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Bake-off compute arm for the `retile_direct` idea. Three shapes:
//
//   ARM 0                 — the op's real compute kernel, verbatim (the reader
//                           produced ROW-MAJOR sticks, so the tilize LLK still
//                           has the permutation-into-tile job).
//   ARMS 1/2/5/6          — the reader produced cb_output_tiles ITSELF, in the
//                           output dtype, so there is nothing left to compute.
//                           An empty kernel (the op still launches three kernels;
//                           this one returns before the first CB touch).
//   ARMS 3/4 (the CAST-capable widening) — the reader produced OUTPUT-SHAPED
//                           tiles in the INPUT dtype in cb_input_sticks; compute
//                           owns the dtype cast alone, as a DATACOPY pass
//                           (unpack -> DST -> pack with both register formats
//                           reconfigured) instead of a tilize. `copy_tiles` is
//                           the kernel_lib helper for exactly this, so no LLK is
//                           hand-written here.
//
//     cb_input_sticks is declared by the host with page_size = tile_h*32*elem_in
//     and TileDescriptor(tile_h, 32) in the INPUT dtype — i.e. it is already a
//     legal TILE page of the OUTPUT geometry in the input format, which is the
//     structural reason the datacopy is expressible at all.

#include <cstdint>

#ifndef RETILE_ARM
#define RETILE_ARM 0
#endif

#if RETILE_ARM == 3 || RETILE_ARM == 4

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = 0;
    constexpr uint32_t cb_output_tiles = 16;

    constexpr uint32_t wt_chunk = get_compile_time_arg_val(0);
    constexpr uint32_t needs_cast = get_compile_time_arg_val(1);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);
    if (num_blocks == 0) {
        return;
    }
    MaybeDeviceZoneScope("compute_tilize");
    if constexpr (needs_cast) {
        compute_kernel_lib::copy_tiles<
            compute_kernel_lib::CopyInputPolicy::WaitAndPop,
            compute_kernel_lib::CopyDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_input_sticks, cb_output_tiles, num_blocks * wt_chunk);
    } else {
        compute_kernel_lib::copy_tiles<
            compute_kernel_lib::CopyInputPolicy::WaitAndPop,
            compute_kernel_lib::CopyDataFormatReconfig::NONE>(cb_input_sticks, cb_output_tiles, num_blocks * wt_chunk);
    }
}

#elif RETILE_ARM != 0

void kernel_main() {}

#else

#include "ttnn/ttnn/operations/tilize/kernels/tilize_compute.cpp"

#endif
