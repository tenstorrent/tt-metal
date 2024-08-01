// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    ArgFetcher arg_fetcher;
    const auto num_tiles{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto start_id{arg_fetcher.get_next_arg_val<uint32_t>()};

    constexpr auto cb_in0{tt::CB::c_in0};    // input
    constexpr auto cb_out0{tt::CB::c_out0};  // output
    constexpr auto cb_intermed0{tt::CB::c_intermed0};    // temp (bfloat16)
    constexpr auto cb_intermed1{tt::CB::c_intermed1};    // temp (uint32)

    constexpr uint32_t onetile{1};

    unary_op_init_common(cb_in0, cb_out0); // uint8, uint8
    // unary_op_init_common(cb_intermed0, cb_intermed0); // bfloat16, bfloat16
    // unary_op_init_common(cb_intermed1, cb_intermed1); // uint32, uint32

    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        // read input tile
        cb_wait_front(cb_in0, onetile);
        unpack_reconfig_data_format_srca(cb_in0);
        copy_tile_to_dst_init_short(cb_in0);
        copy_tile(cb_in0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_out0);
        cb_reserve_back(cb_out0, onetile);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, onetile);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
