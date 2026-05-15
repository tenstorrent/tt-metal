// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_lib::eltwise_chain_with_init<compute_kernel_lib::DEST_AUTO_LIMIT>(
        per_core_tile_cnt,
        compute_kernel_lib::CopyTile<
            cb_in,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::CopyTilePolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::CbIndexMode::BlockIter,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::PackTilePolicy::UpfrontReservePushAtEnd,
            compute_kernel_lib::PackTileIndexMode::BlockIter,
            compute_kernel_lib::PackTileReconfig::None>{});
}
