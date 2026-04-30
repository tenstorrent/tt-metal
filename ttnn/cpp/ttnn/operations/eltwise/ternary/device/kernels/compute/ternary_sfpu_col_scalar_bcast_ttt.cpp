// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PARTIAL MIGRATION: CB wait/pop lifecycle stays raw because broadcast vs
// non-broadcast CBs use different lifecycle scopes (broadcast = outer
// wait/pop, non-broadcast = inner wait/pop). The DEST acquire/copy/op/pack
// block migrates to V2 helper via TernaryMacroOp + NoWaitNoPop CopyTiles
// (caller owns CB lifecycle).

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
struct TernaryMacroOp : compute_kernel_lib::TernaryOp<
                            TernaryMacroOp,
                            compute_kernel_lib::Dst::D0,
                            compute_kernel_lib::Dst::D1,
                            compute_kernel_lib::Dst::D2,
                            compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) {
        TERNARY_SFPU_OP_FUNC(i0, i1, i2, out_idx);
    }
};
}  // namespace

template <tt::CBIndex predicate_cb, tt::CBIndex true_cb, tt::CBIndex false_cb, tt::CBIndex cb_out>
ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    // Outer wait for broadcast CBs (consumed once per process_tile invocation).
#if BCAST_A
    cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_wait_front(true_cb, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_wait_front(false_cb, num_tiles_per_cycle);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Inner wait for non-broadcast CBs.
#if !BCAST_A
        cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_wait_front(true_cb, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_wait_front(false_cb, num_tiles_per_cycle);
#endif

        // Caller owns CB lifecycle (NoWaitNoPop). Helper handles DEST acquire,
        // copy×3, ternary op, pack, release. Reserve/push for cb_out is
        // emitted by the pipeline (single-tile per call).
        compute_kernel_lib::eltwise_pipeline<static_cast<uint32_t>(cb_out)>(
            num_tiles_per_cycle,
            compute_kernel_lib::eltwise_chain(
                compute_kernel_lib::CopyTile<
                    static_cast<uint32_t>(predicate_cb),
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::CopyTilePolicy::NoWaitNoPop>{},
                compute_kernel_lib::CopyTile<
                    static_cast<uint32_t>(true_cb),
                    compute_kernel_lib::Dst::D1,
                    compute_kernel_lib::CopyTilePolicy::NoWaitNoPop>{},
                compute_kernel_lib::CopyTile<
                    static_cast<uint32_t>(false_cb),
                    compute_kernel_lib::Dst::D2,
                    compute_kernel_lib::CopyTilePolicy::NoWaitNoPop>{},
                TernaryMacroOp{}));

#if !BCAST_A
        cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_pop_front(false_cb, num_tiles_per_cycle);
#endif
    }

#if BCAST_A
    cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_pop_front(true_cb, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_pop_front(false_cb, num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb = tt::CBIndex::c_0;
    constexpr auto true_cb = tt::CBIndex::c_1;
    constexpr auto false_cb = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(predicate_cb, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile<predicate_cb, true_cb, false_cb, cb_out>(tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile<predicate_cb, true_cb, false_cb, cb_out>(remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}
