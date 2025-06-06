// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "elemwise_where_kernel_args.hpp"
#include "cpp/kernel/kernel_utils.hpp"

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

template <typename binary_op_t>
inline __attribute__((always_inline)) void sfpu_binary_op(
    tt::CBIndex cb_in0,
    tt::CBIndex cb_in1,
    tt::CBIndex cb_out0,
    uint32_t per_core_block_size,
    binary_op_t&& binary_op) {
    cb_wait_front(cb_in0, per_core_block_size);
    cb_wait_front(cb_in1, per_core_block_size);
    cb_reserve_back(cb_out0, per_core_block_size);

    tile_regs_acquire();
    tile_regs_wait();

    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in0, i, i * 2);
    }

    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in1, i, i * 2 + 1);
        binary_op(i);
        pack_tile(i, cb_out0);
    }

    tile_regs_commit();
    tile_regs_release();

    cb_pop_front(cb_in0, per_core_block_size);
    cb_pop_front(cb_in1, per_core_block_size);
    cb_push_back(cb_out0, per_core_block_size);
}

template <typename unary_op_t>
inline __attribute__((always_inline)) void sfpu_unary_op(tt::CBIndex cb_in, tt::CBIndex cb_out, unary_op_t&& unary_op) {
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);

    unary_op();

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

constexpr auto cb_cond = tt::CBIndex::c_0;
constexpr auto cb_true_values = tt::CBIndex::c_1;
constexpr auto cb_false_values = tt::CBIndex::c_2;
constexpr auto cb_3 = tt::CBIndex::c_3;
constexpr auto cb_4 = tt::CBIndex::c_4;
constexpr auto cb_5 = tt::CBIndex::c_5;
constexpr auto cb_6 = tt::CBIndex::c_6;
constexpr auto cb_out0 = tt::CBIndex::c_7;

namespace NAMESPACE {
void MAIN {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::where_args;
    auto args = make_runtime_struct_from_args<ElemwiseComputeKernelArgs>();

    unary_op_init_common(cb_true_values, cb_out0);
    copy_tile_to_dst_init_short(cb_true_values);

    for (uint32_t block_index = 0; block_index < args.per_core_block_cnt; block_index++) {
        cb_wait_front(cb_cond, 1);

        sfpu_unary_op(cb_cond, cb_3, []() {
            ckernel::gtz_tile_init();
            ckernel::gtz_tile(0);
        });

        sfpu_unary_op(cb_cond, cb_4, []() {
            ckernel::lez_tile_init();
            ckernel::lez_tile(0);
        });

        cb_pop_front(cb_cond, 1);

        sfpu_binary_op(cb_true_values, cb_3, cb_5, args.per_core_block_size, [](uint32_t i) {
            ckernel::mul_binary_tile_init();
            ckernel::mul_binary_tile(i * 2, i * 2 + 1);
        });

        sfpu_binary_op(cb_false_values, cb_4, cb_6, args.per_core_block_size, [](uint32_t i) {
            ckernel::mul_binary_tile_init();
            ckernel::mul_binary_tile(i * 2, i * 2 + 1);
        });

        sfpu_binary_op(cb_5, cb_6, cb_out0, args.per_core_block_size, [](uint32_t i) {
            ckernel::add_binary_tile_init();
            ckernel::add_binary_tile(i * 2, i * 2 + 1);
        });
    }
}
}  // namespace NAMESPACE
