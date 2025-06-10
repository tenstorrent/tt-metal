// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/where.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in3, num_tiles_per_cycle);

        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // DPRINT << "cb_condition" << ENDL();
        // DPRINT << TSLICE(tt::CBIndex::c_0, 0, SliceRange::h0_w0_32()) << ENDL();

        // DPRINT << "cb_true" << ENDL();
        // DPRINT << TSLICE(tt::CBIndex::c_1, 0, SliceRange::h0_w0_32()) << ENDL();

        // DPRINT << "cb_false" << ENDL();
        // DPRINT << TSLICE(tt::CBIndex::c_2, 0, SliceRange::h0_w0_32()) << ENDL();

        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_pre_in1);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_pre_in1, i, i * 2);  // Copy to dst reg 0
        }
        copy_tile_to_dst_init_short(cb_pre_in2);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_pre_in2, i, i * 2 + 1);  // Copy to dst reg 1
        }
        copy_tile_to_dst_init_short(cb_pre_in3);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_pre_in3, i, i * 2 + 2);  // Copy to dst reg 2
            // TODO: Use the where op LLK API here
            where_tile_init();
            WHERE_LLK(i * 2, i * 2 + 1, i * 2 + 2);
        }
        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(0, cb_out);
        }

        tile_regs_release();

        DPRINT << "cb_out" << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_3, 0, SliceRange::h0_w0_32()) << ENDL();

        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in3, num_tiles_per_cycle);
    }
}
}  // namespace NAMESPACE
