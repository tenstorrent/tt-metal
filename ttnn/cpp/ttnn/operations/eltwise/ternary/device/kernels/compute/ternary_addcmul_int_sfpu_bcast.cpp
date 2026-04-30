// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PARTIAL MIGRATION: BCAST-aware CB wait/pop lifecycle stays raw (broadcast =
// outer scope, non-broadcast = inner scope). DEST acquire + addcmul-int chain
// migrated to V2 helper via NoWaitNoPop CopyTiles + FillInt + MulIntBinary +
// AddIntBinary.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.hpp"

template <tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex cb_in2, tt::CBIndex cb_out>
ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle, uint32_t scalar_arg) {
    using namespace ckernel;
    using compute_kernel_lib::AddIntBinary;
    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::CopyTilePolicy;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;
    using compute_kernel_lib::FillInt;
    using compute_kernel_lib::MulIntBinary;

#if BCAST_A
    cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
#if !BCAST_A
        cb_wait_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_wait_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_wait_front(cb_in2, num_tiles_per_cycle);
#endif

        eltwise_pipeline<static_cast<uint32_t>(cb_out)>(
            num_tiles_per_cycle,
            eltwise_chain(
                CopyTile<static_cast<uint32_t>(cb_in0), Dst::D0, CopyTilePolicy::NoWaitNoPop>{},
                CopyTile<static_cast<uint32_t>(cb_in1), Dst::D1, CopyTilePolicy::NoWaitNoPop>{},
                CopyTile<static_cast<uint32_t>(cb_in2), Dst::D2, CopyTilePolicy::NoWaitNoPop>{},
                FillInt<ADDCMUL_DATA_FORMAT, Dst::D3>{{}, scalar_arg},
                MulIntBinary<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D1, Dst::D3>{},
                MulIntBinary<ADDCMUL_DATA_FORMAT, Dst::D3, Dst::D2, Dst::D2>{},
                AddIntBinary<ADDCMUL_DATA_FORMAT, Dst::D0, Dst::D2, Dst::D0>{}));

#if !BCAST_A
        cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if !BCAST_B
        cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if !BCAST_C
        cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
    }

#if BCAST_A
    cb_pop_front(cb_in0, num_tiles_per_cycle);
#endif
#if BCAST_B
    cb_pop_front(cb_in1, num_tiles_per_cycle);
#endif
#if BCAST_C
    cb_pop_front(cb_in2, num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "addcmul_int_bcast path runs one tile per chain invocation");

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckernel::compute_kernel_hw_startup(cb_in0, cb_out);
    ckernel::init_sfpu(cb_in0, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile<cb_in0, cb_in1, cb_in2, cb_out>(tile_freq, tile_start, num_tiles_per_cycle, scalar_arg);
    }
    if (remaining_iterations > 0) {
        process_tile<cb_in0, cb_in1, cb_in2, cb_out>(remaining_iterations, tile_start, num_tiles_per_cycle, scalar_arg);
    }
}
