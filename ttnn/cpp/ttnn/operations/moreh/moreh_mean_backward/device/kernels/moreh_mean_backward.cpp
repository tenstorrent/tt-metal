// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // zero tile
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    cb_wait_front(cb_in1, onetile);

    // PARTIAL migration: per-iter loop body.
    //   migrated:
    //     stage 1: BinaryFpu(Add, Cols/Rows/Scalar) when bcast OR CopyTile when no bcast,
    //              + PackTile(cb_intermed0).
    //     stage 2: BinaryFpu(Mul, Scalar) cb_intermed0 * cb_scalar -> cb_out0.
    //   skipped : nothing — full main loop migrated.
    {
        using namespace compute_kernel_lib;
        for (uint32_t i = 0; i < num_output_tiles; i++) {
            // Stage 1 — accumulate (or pure copy) into cb_intermed0.
            if constexpr (ht_need_bcast || wt_need_bcast) {
                constexpr BroadcastDim BCAST_DIM =
                    (ht_need_bcast && wt_need_bcast) ? BroadcastDim::Scalar :
                    ht_need_bcast                    ? BroadcastDim::Row :
                                                       BroadcastDim::Col;
                using AddBcast = BinaryFpu<
                    cb_in1,
                    cb_in0,
                    BinaryFpuOp::Add,
                    BCAST_DIM,
                    BinaryFpuOutputPolicy::PerTile,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::NoWaitNoPop,
                    CopyTilePolicy::WaitAndPop,
                    CbIndexMode::FirstTile,
                    CbIndexMode::FirstTile,
                    Dst::D0,
                    cb_intermed0>;
                eltwise_chain(
                    onetile,
                    AddBcast{},
                    PackTile<cb_intermed0, Dst::D0, PackTilePolicy::PerTileReserveAndPush,
                             PackTileIndexMode::FirstTile, PackTileReconfig::Output>{});
            } else {
                eltwise_chain(
                    onetile,
                    CopyTile<cb_in0, Dst::D0, CopyTilePolicy::WaitAndPop,
                             CbIndexMode::FirstTile, CopyTileReconfig::None>{},
                    PackTile<cb_intermed0, Dst::D0, PackTilePolicy::PerTileReserveAndPush,
                             PackTileIndexMode::FirstTile, PackTileReconfig::Output>{});
            }

            // Stage 2 — output * (1 / number_of_elements).
            using MulScalar = BinaryFpu<
                cb_intermed0,
                cb_scalar,
                BinaryFpuOp::Mul,
                BroadcastDim::Scalar,
                BinaryFpuOutputPolicy::PerTile,
                BinaryDataFormatReconfig::Input,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::NoWaitNoPop,
                CbIndexMode::FirstTile,
                CbIndexMode::FirstTile,
                Dst::D0,
                cb_out0>;
            eltwise_chain(
                onetile,
                MulScalar{},
                PackTile<cb_out0, Dst::D0, PackTilePolicy::PerTileReserveAndPush,
                         PackTileIndexMode::FirstTile, PackTileReconfig::Output>{});
        }
    }

    cb_pop_front(cb_in1, onetile);
}
