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
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t scalar_value = get_arg(args::scalar_arg);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);  // set to 1
    constexpr bool scalar_is_true = get_arg(args::scalar_is_true);  // 1=TST (scalar=true), 0=TTS (scalar=false)

    DataflowBuffer dfb_pre_in1(dfb::in_a);
    DataflowBuffer dfb_pre_in2(dfb::in_b);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in_a, dfb::out);
    copy_init(dfb::in_a);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        dfb_pre_in1.wait_front(num_tiles_per_cycle);
        dfb_pre_in2.wait_front(num_tiles_per_cycle);
        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // Always copy condition to dst reg 0
        copy_init(dfb::in_a);
        copy_tile(dfb::in_a, 0, 0);  // Copy condition to dst reg 0

        // Copy tensor to appropriate dst register based on variant
        reconfig_data_format_srca(dfb::in_a, dfb::in_b);
        copy_init(dfb::in_b);
        if constexpr (scalar_is_true) {
            // TST: tensor is false value, goes to dst reg 2
            copy_tile(dfb::in_b, 0, 2);  // Copy false tensor to dst reg 2
        } else {
            // TTS: tensor is true value, goes to dst reg 1
            copy_tile(dfb::in_b, 0, 1);  // Copy true tensor to dst reg 1
        }
        reconfig_data_format_srca(dfb::in_b, dfb::in_a);

        // Fill scalar value to appropriate dst register
        fill_tile_init();
        const auto scalar_val = reinterpret_cast<const float*>(&scalar_value);
        if constexpr (scalar_is_true) {
            // TST: scalar is true value, goes to dst reg 1
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(1, scalar_value);
#endif
        } else {
            // TTS: scalar is false value, goes to dst reg 2
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(2, scalar_value);
#endif
        }

        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb::out);

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_pre_in1.pop_front(num_tiles_per_cycle);
        dfb_pre_in2.pop_front(num_tiles_per_cycle);
    }
}
