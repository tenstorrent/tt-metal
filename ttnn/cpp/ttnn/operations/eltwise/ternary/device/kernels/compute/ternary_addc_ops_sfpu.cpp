// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t scalar_arg = get_arg(args::scalar_arg);
    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);  // set to 1

    DataflowBuffer dfb_in0(dfb::in_a);  // input_a
    DataflowBuffer dfb_in1(dfb::in_b);  // input_b
    DataflowBuffer dfb_in2(dfb::in_c);  // input_c
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in_a, dfb::out);
    copy_init(dfb::in_a);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        dfb_in0.wait_front(num_tiles_per_cycle);
        dfb_in1.wait_front(num_tiles_per_cycle);
        dfb_in2.wait_front(num_tiles_per_cycle);

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        copy_init(dfb::in_a);
        copy_tile(dfb::in_a, 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        reconfig_data_format_srca(dfb::in_a, dfb::in_b);
        copy_init(dfb::in_b);
        copy_tile(dfb::in_b, 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        reconfig_data_format_srca(dfb::in_b, dfb::in_c);
        copy_init(dfb::in_c);
        copy_tile(dfb::in_c, 0 /*in_tile_index*/, 2 /*dst_tile_index*/);
        reconfig_data_format_srca(dfb::in_c, dfb::in_a);

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb::out);

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_in0.pop_front(num_tiles_per_cycle);
        dfb_in1.pop_front(num_tiles_per_cycle);
        dfb_in2.pop_front(num_tiles_per_cycle);
    }
}
