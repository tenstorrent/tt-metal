// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // zero tile
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_in1, cb_in0, cb_out0);
    cb_wait_front(cb_in1, onetile);

    // PARTIAL migration: full main loop migrated as a single chain.
    //   migrated: BinaryFpu(Add, Cols/Rows/Scalar) when bcast, or CopyTile when no bcast,
    //             both followed by PackTile.
    //   skipped : nothing — full main loop migrated.
    {
        using namespace compute_kernel_lib;
        if constexpr (ht_need_bcast || wt_need_bcast) {
            constexpr BroadcastDim BCAST_DIM =
                (ht_need_bcast && wt_need_bcast) ? BroadcastDim::Scalar :
                ht_need_bcast                    ? BroadcastDim::Row :
                                                   BroadcastDim::Col;
            using AddBcast = BinaryFpu<
                cb_in1,
                cb_in0,
                cb_out0,
                BinaryFpuOp::Add,
                BCAST_DIM,
                BinaryDataFormatReconfig::None,
                CopyTilePolicy::NoWaitNoPop,
                CopyTilePolicy::WaitAndPop,
                CbIndexMode::FirstTile,
                Dst::D0>;
            eltwise_chain(
                num_output_tiles,
                AddBcast{},
                PackTile<cb_out0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
        } else {
            // No-bcast path: just copy cb_in0 -> cb_out0 (cb_in1 unused per iteration).
            eltwise_chain(
                num_output_tiles,
                CopyTile<cb_in0, Dst::D0, CopyTilePolicy::WaitAndPop>{},
                PackTile<cb_out0, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
        }
    }

    cb_pop_front(cb_in1, onetile);
}
