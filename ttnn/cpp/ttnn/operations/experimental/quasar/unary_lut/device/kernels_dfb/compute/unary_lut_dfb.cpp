// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) compute kernel for the unary piecewise-LUT
// activation. The UNARY analog of binary_ng's eltwise_binary_no_bcast_dfb.cpp:
// ONE input DFB (in0) instead of two, and the compute is an embedded
// piecewise-polynomial LUT SFPU evaluation (Horner) instead of add_tiles.
//
// Pipeline per chunk:
//   in0 (DFB) --copy_tile--> DST --SFPU LUT eval--> DST --pack_tile--> out (DFB)
//
// The LUT (boundaries + per-segment coefficients) is baked in at compile time via
// the LUT_* defines emitted by the program factory (unary_lut_sfpu.h). NO range
// reduction in this slice.
//
// Runtime arg: num_tiles (this core's shard tile count).
// Compile-time arg: num_tiles_per_cycle (tiles per DST acquire; bounded by DST capacity).

#include <cstdint>

#include "api/compute/common_globals.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// The embedded LUT SFPU evaluator (ckernel::sfpu::calculate_lut_activation). MATH thread only.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "unary_lut_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_out(dfb::out);

    // Configure unpacker (SrcA datacopy) + packer + SFPU for the in0->out DFBs.
    init_sfpu(dfb::in0, dfb::out);
    copy_tile_to_dst_init_short(dfb::in0);

    // Initialize the SFPU. SfpuType is unused on Quasar; on WH/BH it selects the
    // SFPU op slot (we run a custom callback, so the slot value is immaterial).
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::unused>()));

    auto process_tiles = [&](uint32_t n) {
        dfb_in0.wait_front(n);
        dfb_out.reserve_back(n);

        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            // Datacopy in0[i] -> DST[i] (SrcA -> Dest / MOVA2D).
            copy_tile(dfb::in0, i, i);
            // Apply the embedded piecewise-LUT activation in place on DST[i].
            MATH((SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_lut_activation,
                (8 /* ITERATIONS = SFPU_ITERATIONS */),
                i,
                VectorMode::RC)));
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb::out);
        }
        tile_regs_release();

        dfb_out.push_back(n);
        dfb_in0.pop_front(n);
    };

    const uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }
    const uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles(remainder);
    }
}
