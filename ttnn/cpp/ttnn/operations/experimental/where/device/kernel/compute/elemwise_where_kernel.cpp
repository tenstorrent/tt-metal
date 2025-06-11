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
inline __attribute__((always_inline)) void fpu_binary_op(
    tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex cb_out, uint32_t per_core_block_size, binary_op_t&& binary_op) {
    cb_wait_front(cb_in0, per_core_block_size);
    cb_wait_front(cb_in1, per_core_block_size);
    cb_reserve_back(cb_out, per_core_block_size);

    tile_regs_acquire();
    tile_regs_wait();

    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        binary_op(cb_in0, cb_in1, i);
        pack_tile(i, cb_out);
    }

    tile_regs_commit();
    tile_regs_release();

    cb_pop_front(cb_in0, per_core_block_size);
    cb_pop_front(cb_in1, per_core_block_size);
    cb_push_back(cb_out, per_core_block_size);
}

template <typename binary_op_t>
inline __attribute__((always_inline)) void sfpu_binary_op(
    tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex cb_out, uint32_t per_core_block_size, binary_op_t&& binary_op) {
    cb_wait_front(cb_in0, per_core_block_size);
    cb_wait_front(cb_in1, per_core_block_size);
    cb_reserve_back(cb_out, per_core_block_size);

    tile_regs_acquire();
    tile_regs_wait();

    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in0, i, i * 2);
    }

    for (uint32_t i = 0; i < per_core_block_size; ++i) {
        copy_tile(cb_in1, i, i * 2 + 1);
        binary_op(i);
        pack_tile(i, cb_out);
    }

    tile_regs_commit();
    tile_regs_release();

    cb_pop_front(cb_in0, per_core_block_size);
    cb_pop_front(cb_in1, per_core_block_size);
    cb_push_back(cb_out, per_core_block_size);
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

constexpr auto cb_condition = tt::CBIndex::c_0;
constexpr auto cb_true_values = tt::CBIndex::c_1;
constexpr auto cb_false_values = tt::CBIndex::c_2;
constexpr auto cb_positive_mask = tt::CBIndex::c_3;
constexpr auto cb_negative_mask = tt::CBIndex::c_4;
constexpr auto cb_true_values_out = tt::CBIndex::c_5;
constexpr auto cb_false_values_out = tt::CBIndex::c_6;
constexpr auto cb_out = tt::CBIndex::c_7;

namespace NAMESPACE {
void MAIN {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::where_args;
    auto args = make_runtime_struct_from_args<ElemwiseComputeKernelArgs>();

    unary_op_init_common(cb_true_values, cb_out);
    copy_tile_to_dst_init_short(cb_true_values);

    for (uint32_t block_index = 0; block_index < args.per_core_block_cnt; block_index++) {
        cb_wait_front(cb_condition, 1);

        sfpu_unary_op(cb_condition, cb_positive_mask, []() {
            ckernel::gtz_tile_init();
            ckernel::gtz_tile(0);
        });

        sfpu_unary_op(cb_condition, cb_negative_mask, []() {
            ckernel::lez_tile_init();
            ckernel::lez_tile(0);
        });

        cb_pop_front(cb_condition, 1);

        fpu_binary_op(
            cb_true_values,
            cb_positive_mask,
            cb_true_values_out,
            args.per_core_block_size,
            [](uint32_t icb0, uint32_t icb1, uint32_t i) {
                ckernel::mul_tiles_init(icb0, icb1);
                ckernel::mul_tiles(icb0, icb1, i, i, i);
            });

        fpu_binary_op(
            cb_false_values,
            cb_negative_mask,
            cb_false_values_out,
            args.per_core_block_size,
            [](uint32_t icb0, uint32_t icb1, uint32_t i) {
                ckernel::mul_tiles_init(icb0, icb1);
                ckernel::mul_tiles(icb0, icb1, i, i, i);
            });

        fpu_binary_op(
            cb_true_values_out,
            cb_false_values_out,
            cb_out,
            args.per_core_block_size,
            [](uint32_t icb0, uint32_t icb1, uint32_t i) {
                ckernel::add_tiles_init(icb0, icb1, false);
                ckernel::add_tiles(icb0, icb1, i, i, i);
            });
        // SFPU IMPLEMENTATION
        // sfpu_binary_op(cb_true_values, cb_positive_mask, cb_true_values_out, args.per_core_block_size, [](uint32_t i)
        // {
        //     ckernel::mul_binary_tile_init();
        //     ckernel::mul_binary_tile(i * 2, i * 2 + 1);
        // });

        // sfpu_binary_op(
        //     cb_false_values, cb_negative_mask, cb_false_values_out, args.per_core_block_size, [](uint32_t i) {
        //         ckernel::mul_binary_tile_init();
        //         ckernel::mul_binary_tile(i * 2, i * 2 + 1);
        //     });

        // sfpu_binary_op(cb_true_values_out, cb_false_values_out, cb_out, args.per_core_block_size, [](uint32_t i) {
        //     ckernel::add_binary_tile_init();
        //     ckernel::add_binary_tile(i * 2, i * 2 + 1);
        // });
    }
}

}  // namespace NAMESPACE
