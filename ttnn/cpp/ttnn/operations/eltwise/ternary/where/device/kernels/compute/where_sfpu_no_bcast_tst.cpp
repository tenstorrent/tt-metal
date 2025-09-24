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
    const uint32_t true_scalar = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    const auto true_value = reinterpret_cast<const float*>(&true_scalar);

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_pre_in1);
        copy_tile(cb_pre_in1, 0, 0);  // Copy to dst reg 0

        // Fill scalar value to dst reg 1
        fill_tile_init();
#ifdef FILL_WITH_VALUE_FLOAT
        const auto true_value = reinterpret_cast<const float*>(&true_scalar);
        FILL_LLK(1, *true_value);
#endif
#ifdef FILL_WITH_VALUE_INT
        FILL_LLK(1, true_scalar);
#endif

        copy_tile_to_dst_init_short(cb_pre_in2);
        copy_tile(cb_pre_in2, 0, 2);  // Copy to dst reg 2

        where_tile_init();
        WHERE_LLK(0, 1, 2, 0);

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
