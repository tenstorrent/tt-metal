// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    DataflowBuffer dfb_pre_in1(tt::CBIndex::c_0);
    DataflowBuffer dfb_pre_in2(tt::CBIndex::c_1);
    DataflowBuffer dfb_pre_in3(tt::CBIndex::c_2);
    DataflowBuffer dfb_out(tt::CBIndex::c_3);

    compute_kernel_hw_startup(dfb_pre_in1.get_id(), dfb_out.get_id());
    copy_init(dfb_pre_in1.get_id());

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        dfb_pre_in1.wait_front(num_tiles_per_cycle);
        dfb_pre_in2.wait_front(num_tiles_per_cycle);
        dfb_pre_in3.wait_front(num_tiles_per_cycle);

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        copy_init(dfb_pre_in1.get_id());
        copy_tile(dfb_pre_in1.get_id(), 0, 0);  // Copy to dst reg 0

        copy_init(dfb_pre_in2.get_id());
        copy_tile(dfb_pre_in2.get_id(), 0, 1);  // Copy to dst reg 1

        copy_init(dfb_pre_in3.get_id());
        copy_tile(dfb_pre_in3.get_id(), 0, 2);  // Copy to dst reg 2

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb_out.get_id());

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_pre_in1.pop_front(num_tiles_per_cycle);
        dfb_pre_in2.pop_front(num_tiles_per_cycle);
        dfb_pre_in3.pop_front(num_tiles_per_cycle);
    }
}
