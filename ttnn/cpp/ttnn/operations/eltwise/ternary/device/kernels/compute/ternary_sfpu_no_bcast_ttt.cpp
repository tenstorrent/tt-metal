// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/mac.h"
#include "api/compute/eltwise_unary/snake_beta.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);  // set to 1

    DataflowBuffer dfb_pre_in1(dfb::in_a);
    DataflowBuffer dfb_pre_in2(dfb::in_b);
    DataflowBuffer dfb_pre_in3(dfb::in_c);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in_a, dfb::out);
    copy_init(dfb::in_a);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        dfb_pre_in1.wait_front(num_tiles_per_cycle);
        dfb_pre_in2.wait_front(num_tiles_per_cycle);
        dfb_pre_in3.wait_front(num_tiles_per_cycle);

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // copy_init does not reconfigure the unpacker data format, and the operands may differ in dtype.
        reconfig_data_format_srca(dfb::in_c, dfb::in_a);
        copy_init(dfb::in_a);
        copy_tile(dfb::in_a, 0, 0);  // Copy to dst reg 0

        reconfig_data_format_srca(dfb::in_a, dfb::in_b);
        copy_init(dfb::in_b);
        copy_tile(dfb::in_b, 0, 1);  // Copy to dst reg 1

        reconfig_data_format_srca(dfb::in_b, dfb::in_c);
        copy_init(dfb::in_c);
        copy_tile(dfb::in_c, 0, 2);  // Copy to dst reg 2

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb::out);

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_pre_in1.pop_front(num_tiles_per_cycle);
        dfb_pre_in2.pop_front(num_tiles_per_cycle);
        dfb_pre_in3.pop_front(num_tiles_per_cycle);
    }
}
