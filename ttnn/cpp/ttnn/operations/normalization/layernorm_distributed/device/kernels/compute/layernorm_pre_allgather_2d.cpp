// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes rmsnorm statistics.
For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 */

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    constexpr uint32_t NCHt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(3);
    bool is_merge_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t cb_x2 = tt::CBIndex::c_6;  // x**2
    constexpr uint32_t cb_zero = tt::CBIndex::c_13;

    binary_op_init_common(cb_inp, cb_reduce, cb_x2);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * x**2
         *
         * Migrated: same-CB BinaryFpu(Mul) over Wt with chain BlockSize=blk.
         * CumulativeWaitNoPop matches cb_wait_front(cb_inp, wt+blk) per-iter
         * cumulative grow without popping — caller pops cb_inp(Wt) after the
         * downstream sum(x²) reduce completes. same_cb dedup avoids duplicate
         * waits when CbA==CbB==cb_inp.
         */
        compute_kernel_lib::eltwise_chain<blk>(
            Wt,
            compute_kernel_lib::BinaryFpu<
                cb_inp,
                cb_inp,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::CopyTilePolicy::CumulativeWaitNoPop,
                compute_kernel_lib::CopyTilePolicy::CumulativeWaitNoPop,
                compute_kernel_lib::CbIndexMode::BlockIter,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::CbIndexMode::BlockIter>{},
            compute_kernel_lib::PackTile<
                cb_x2,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::PackTilePolicy::UpfrontReservePushAtEnd,
                compute_kernel_lib::PackTileIndexMode::BlockIter,
                compute_kernel_lib::PackTileReconfig::Output>{});

        /*
         * sum(x**2)
         */
        // BulkWaitBulkPop: All Wt tiles already in CB (see cumulative wait above)
        compute_kernel_lib::
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_x2, cb_reduce, cb_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));
        cb_pop_front(cb_inp, Wt);
        cb_pop_front(cb_reduce, 1);
    }

    // if merge core, we need to do a final sum on the tile in cb_x2 and then write the result to cb_out_final
    if (is_merge_core) {
        constexpr uint32_t cb_x2_merge = tt::CBIndex::c_15;
        constexpr uint32_t cb_out_final = tt::CBIndex::c_14;
        constexpr int dst0 = 0;

        // Wait for all num_cores_y tiles
        cb_wait_front(cb_x2_merge, num_cores_y);
        cb_wait_front(cb_zero, 1);

        // Reserve output space
        cb_reserve_back(cb_out_final, onetile);

        // Initialize accumulation
        binary_op_init_common(cb_x2_merge, cb_zero, cb_out_final);
        reconfig_data_format(cb_x2_merge, cb_zero);
        pack_reconfig_data_format(cb_out_final);
        add_tiles_init(cb_x2_merge, cb_zero, true);

        // Acquire registers
        ACQ();

        // Add all 8 tiles together
        for (uint32_t i = 0; i < num_cores_y; i++) {
            add_tiles(cb_x2_merge, cb_zero, i, 0, dst0);
        }

        // Pack result
        pack_tile(dst0, cb_out_final);
        REL();

        // Push output and pop input
        cb_push_back(cb_out_final, onetile);
        cb_pop_front(cb_x2_merge, num_cores_y);
    }
}
