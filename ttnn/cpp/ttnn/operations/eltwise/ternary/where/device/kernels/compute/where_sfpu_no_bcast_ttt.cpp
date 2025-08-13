// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/bcast.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;  // predicate
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;  // true tensor
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;  // false tensor
    constexpr auto cb_out = tt::CBIndex::c_3;

    // Additional CBs for broadcast operations
    constexpr auto cb_bcast_pred = tt::CBIndex::c_4;   // for predicate broadcast
    constexpr auto cb_bcast_true = tt::CBIndex::c_5;   // for true tensor broadcast
    constexpr auto cb_bcast_false = tt::CBIndex::c_6;  // for false tensor broadcast

    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for input tensors
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);  // predicate
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);  // true tensor
        cb_wait_front(cb_pre_in3, num_tiles_per_cycle);  // false tensor

        // Handle predicate broadcast if needed
#ifdef SRC_BCAST
        cb_reserve_back(cb_bcast_pred, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_pre_in1, cb_bcast_pred);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_pre_in1, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_bcast_pred);
        cb_push_back(cb_bcast_pred, num_tiles_per_cycle);
        tile_regs_release();
        constexpr auto cb_final_pred = cb_bcast_pred;
#else
        constexpr auto cb_final_pred = cb_pre_in1;
#endif

        // Handle true tensor broadcast if needed
#ifdef SRC_BCAST_B
        cb_reserve_back(cb_bcast_true, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_pre_in2, cb_bcast_true);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_pre_in2, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_bcast_true);
        cb_push_back(cb_bcast_true, num_tiles_per_cycle);
        tile_regs_release();
        constexpr auto cb_final_true = cb_bcast_true;
#else
        constexpr auto cb_final_true = cb_pre_in2;
#endif

        // Handle false tensor broadcast if needed
#ifdef SRC_BCAST_C
        cb_reserve_back(cb_bcast_false, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(cb_pre_in3, cb_bcast_false);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_pre_in3, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_bcast_false);
        cb_push_back(cb_bcast_false, num_tiles_per_cycle);
        tile_regs_release();
        constexpr auto cb_final_false = cb_bcast_false;
#else
        constexpr auto cb_final_false = cb_pre_in3;
#endif

        // Now wait for the final (possibly broadcast) tensors
        cb_wait_front(cb_final_pred, num_tiles_per_cycle);
        cb_wait_front(cb_final_true, num_tiles_per_cycle);
        cb_wait_front(cb_final_false, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy final tensors to destination registers
        copy_tile_to_dst_init_short(cb_final_pred);
        copy_tile(cb_final_pred, 0, 0);  // predicate to dst reg 0

        copy_tile_to_dst_init_short(cb_final_true);
        copy_tile(cb_final_true, 0, 1);  // true tensor to dst reg 1

        copy_tile_to_dst_init_short(cb_final_false);
        copy_tile(cb_final_false, 0, 2);  // false tensor to dst reg 2

        // Perform where operation
        where_tile_init();
        WHERE_LLK(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);

        // Pop from input CBs
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in3, num_tiles_per_cycle);

        // Pop from broadcast CBs if used
#ifdef SRC_BCAST
        cb_pop_front(cb_bcast_pred, num_tiles_per_cycle);
#endif
#ifdef SRC_BCAST_B
        cb_pop_front(cb_bcast_true, num_tiles_per_cycle);
#endif
#ifdef SRC_BCAST_C
        cb_pop_front(cb_bcast_false, num_tiles_per_cycle);
#endif
    }
}
}  // namespace NAMESPACE
