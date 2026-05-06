// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint8_t cb_x = 0;                   // input
    constexpr uint8_t cb_clip_coef_clamped = 1;   // clip_coef_clamped (held scalar)
    constexpr uint8_t cb_y = 16;                  // output

    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_x, cb_clip_coef_clamped, cb_y);

    cb_wait_front(cb_clip_coef_clamped, onetile);  // comes from the reader

    // PARTIAL migration: y = x * clip_coef_clamped (scalar broadcast).
    //   migrated: BinaryFpu(Mul, BroadcastDim::Scalar) + PackTile chain.
    //   skipped : nothing — full main loop migrated.
    {
        using namespace compute_kernel_lib;
        using MulBcast = BinaryFpu<
            cb_x, cb_clip_coef_clamped, BinaryFpuOp::Mul, BroadcastDim::Scalar,
            BinaryFpuOutputPolicy::PerTile, BinaryDataFormatReconfig::None,
            CopyTilePolicy::WaitAndPop, CopyTilePolicy::NoWaitNoPop,
            CbIndexMode::FirstTile, CbIndexMode::FirstTile, Dst::D0,
            0, 0, 0, cb_y>;
        eltwise_chain(
            num_tiles,
            MulBcast{},
            PackTile<cb_y, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
    }

    cb_pop_front(cb_clip_coef_clamped, onetile);
}
