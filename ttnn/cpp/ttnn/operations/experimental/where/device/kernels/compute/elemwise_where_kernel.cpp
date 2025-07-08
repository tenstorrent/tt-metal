// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "elemwise_where_kernel_args.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_binary.h"

constexpr auto cb_condition = tt::CBIndex::c_0;
constexpr auto cb_true_values = tt::CBIndex::c_1;
constexpr auto cb_false_values = tt::CBIndex::c_2;
constexpr auto cb_true_values_out = tt::CBIndex::c_3;
constexpr auto cb_out = tt::CBIndex::c_4;

namespace NAMESPACE {
void MAIN {
    using namespace ttnn::kernel_utils;
    using namespace ttnn::kernel::eltwise::where_args;
    auto args = make_runtime_struct_from_args<ElemwiseComputeKernelArgs>();

    unary_op_init_common(cb_true_values, cb_out);
    copy_tile_to_dst_init_short(cb_true_values);

    for (uint32_t block_index = 0; block_index < args.per_core_block_cnt; block_index++) {
        cb_wait_front(cb_condition, args.per_core_block_size);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_condition);
        copy_tile(cb_condition, 0, 0);

        ckernel::gtz_tile_init();
        ckernel::gtz_tile(0);

        cb_wait_front(cb_true_values, args.per_core_block_size);

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_true_values);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_true_values, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_true_values_out, args.per_core_block_size);
        pack_tile(0, cb_true_values_out);
        tile_regs_release();
        cb_push_back(cb_true_values_out, args.per_core_block_size);

        cb_wait_front(cb_false_values, args.per_core_block_size);
        cb_wait_front(cb_true_values_out, args.per_core_block_size);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_condition);
        copy_tile(cb_condition, 0, 0);

        ckernel::lez_tile_init();
        ckernel::lez_tile(0);

        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_false_values);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_false_values, 0, 0);

        cb_reserve_back(cb_out, args.per_core_block_size);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_true_values_out);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_true_values_out, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_pop_front(cb_true_values_out, args.per_core_block_size);
        cb_pop_front(cb_condition, args.per_core_block_size);
        cb_pop_front(cb_true_values, args.per_core_block_size);
        cb_pop_front(cb_false_values, args.per_core_block_size);

        cb_push_back(cb_out, args.per_core_block_size);
    }
}

}  // namespace NAMESPACE
