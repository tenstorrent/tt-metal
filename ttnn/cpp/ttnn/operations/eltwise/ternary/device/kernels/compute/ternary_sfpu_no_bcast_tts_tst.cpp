// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);           // 1=TST (scalar=true), 0=TTS (scalar=false)

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        // Always copy condition to dst reg 0
        copy_tile_to_dst_init_short(cb_pre_in1);
        copy_tile(cb_pre_in1, 0, 0);  // Copy condition to dst reg 0

        // Copy tensor to appropriate dst register based on variant
        copy_tile_to_dst_init_short(cb_pre_in2);
        if constexpr (scalar_is_true) {
            // TST: tensor is false value, goes to dst reg 2
            copy_tile(cb_pre_in2, 0, 2);  // Copy false tensor to dst reg 2
        } else {
            // TTS: tensor is true value, goes to dst reg 1
            copy_tile(cb_pre_in2, 0, 1);  // Copy true tensor to dst reg 1
        }

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

        pack_tile(0, cb_out);

        tile_regs_release();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
