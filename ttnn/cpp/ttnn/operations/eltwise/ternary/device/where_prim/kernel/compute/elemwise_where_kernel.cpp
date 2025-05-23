// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// #include <coroutine>

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

template <typename unary_op_a_t, typename unary_op_b_t>
inline __attribute__((always_inline)) void sfpu_unary_2op(
    tt::CBIndex cb_in,
    tt::CBIndex cb_out_a,
    tt::CBIndex cb_out_b,
    uint32_t per_core_block_cnt,
    uint32_t per_core_block_size,
    unary_op_a_t&& unary_op_a,
    unary_op_b_t&& unary_op_b) {
    init_sfpu(cb_in, cb_out_a);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // DPRINT << "sfpu_unary_op, block_index = " << block_index << ENDL();

        cb_reserve_back(cb_out_a, per_core_block_size);
        cb_reserve_back(cb_out_b, per_core_block_size);
        // DPRINT << "exec unary op " << ENDL();
        for (uint32_t tile_index = 0; tile_index < per_core_block_size; ++tile_index) {
            // Pop tile after tile, copy to DST and pack
            cb_wait_front(cb_in, 1);

            tile_regs_acquire();
            copy_tile(cb_in, 0, 0);
            unary_op_a();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out_a);
            // tile_regs_release();

            // tile_regs_acquire();
            copy_tile(cb_in, 0, 0);
            unary_op_b();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out_b);
            tile_regs_release();

            cb_pop_front(cb_in, 1);
        }
        cb_push_back(cb_out_a, per_core_block_size);
        cb_push_back(cb_out_b, per_core_block_size);
    }
}

template <typename binary_op_t>
inline __attribute__((always_inline)) void sfpu_binary_op(
    tt::CBIndex cb_in0,
    tt::CBIndex cb_in1,
    tt::CBIndex cb_out0,
    uint32_t per_core_block_cnt,
    uint32_t per_core_block_size,
    binary_op_t&& binary_op) {
    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_in0, per_core_block_size);
        // DPRINT << " wait cb0" << ENDL();
        cb_wait_front(cb_in1, per_core_block_size);
        // DPRINT << "get cb_in0, wait cb1" << ENDL();
        cb_reserve_back(cb_out0, per_core_block_size);
        // DPRINT << "get cb_in0, wait cb1 done" << ENDL();

        tile_regs_acquire();
        tile_regs_wait();

        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_in0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_in0, cb_in1);
        DPRINT << "binary_op..." << ENDL();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in1, i, i * 2 + 1);  // (uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)

            binary_op(i);
            // DPRINT << "binary_op exec done" << ENDL();
            pack_tile(i * 2, cb_out0);
            DPRINT << "pack_tile exec done" << ENDL();
        }
        // DPRINT << "binary_op done" << ENDL();
        tile_regs_commit();
        tile_regs_release();

        // DPRINT << "tile_regs_release done..." << ENDL();
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
        // DPRINT << "cb_pop_front done" << ENDL();
        cb_push_back(cb_out0, per_core_block_size);
    }
}

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_3 = tt::CBIndex::c_3;
    constexpr auto cb_4 = tt::CBIndex::c_4;
    constexpr auto cb_5 = tt::CBIndex::c_5;
    constexpr auto cb_6 = tt::CBIndex::c_6;
    // TODO: I take cb_16 from tt-metal examples.  Check the perf impact of selected CB
    // It also affect writer kernel
    constexpr auto cb_out0 = tt::CBIndex::c_7;

    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    // DPRINT << "per_core_block_cnt = " << per_core_block_cnt << ENDL();
    // DPRINT << "per_core_block_size = " << per_core_block_size << ENDL();

    // DPRINT << "gtz + ltz" << ENDL();
    sfpu_unary_2op(
        cb_in0,
        cb_3,
        cb_4,
        per_core_block_cnt,
        per_core_block_size,
        []() {
            ckernel::gtz_tile_init();
            ckernel::gtz_tile(0);
        },
        []() {
            ckernel::ltz_tile_init();
            ckernel::ltz_tile(0);
        });

    // DPRINT << "mul1" << ENDL();
    sfpu_binary_op(cb_in1, cb_3, cb_5, per_core_block_cnt, per_core_block_size, [](uint32_t i) {
        ckernel::mul_binary_tile_init();
        ckernel::mul_binary_tile(i * 2, i * 2 + 1);
    });

    // DPRINT << "mul2" << ENDL();
    sfpu_binary_op(cb_in2, cb_4, cb_6, per_core_block_cnt, per_core_block_size, [](uint32_t i) {
        ckernel::mul_binary_tile_init();
        ckernel::mul_binary_tile(i * 2, i * 2 + 1);
    });

    DPRINT << "add" << ENDL();
    sfpu_binary_op(cb_5, cb_6, cb_out0, per_core_block_cnt, per_core_block_size, [](uint32_t i) {
        ckernel::add_binary_tile_init();
        ckernel::add_binary_tile(i * 2, i * 2 + 1);
    });

    // DPRINT << "push res" << ENDL();
}
}  // namespace NAMESPACE
