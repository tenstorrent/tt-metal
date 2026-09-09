// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// apply_twiddles_compute.cpp — TRISC compute kernel for the apply_twiddles
// op.  Pure SFPU complex multiply on a stream of (A, T) input tiles:
//
//     (B_R, B_I) = (A_R * T_R - A_I * T_I,
//                   A_R * T_I + A_I * T_R)
//
// This shared compute kernel serves table twiddles, XL twiddles, and generic
// complex multiplication through the same named dataflow-buffer layout.
// We keep them as separate translation units to avoid coupling the legacy
// pass2 path to apply_twiddles' lifecycle.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

enum : uint32_t { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2 };

template <uint32_t OP, typename DfbA, typename DfbB, typename DfbOut>
FORCE_INLINE void sfpu_binop_push(DfbA a, DfbB b, DfbOut out) {
    tile_regs_acquire();

    copy_init(a);
    copy_tile(a, 0, 0);
    reconfig_data_format_srca(a, b);
    copy_init(b);
    copy_tile(b, 0, 1);

    if constexpr (OP == OP_ADD) {
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    } else if constexpr (OP == OP_SUB) {
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
    } else if constexpr (OP == OP_MUL) {
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    }

    tile_regs_commit();

    DataflowBuffer cb_out(out);
    cb_out.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_out.push_back(1);
}

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    compute_kernel_hw_startup(dfb::a_r, dfb::b_r);
    copy_init(dfb::a_r);

    DataflowBuffer cb_a_r(dfb::a_r);
    DataflowBuffer cb_a_i(dfb::a_i);
    DataflowBuffer cb_t_r(dfb::t_r);
    DataflowBuffer cb_t_i(dfb::t_i);
    DataflowBuffer cb_tmp_r(dfb::tmp_r);
    DataflowBuffer cb_tmp_i(dfb::tmp_i);

    for (uint32_t k = 0; k < num_tiles; ++k) {
        cb_a_r.wait_front(1);
        cb_a_i.wait_front(1);
        cb_t_r.wait_front(1);
        cb_t_i.wait_front(1);

        // B_R = A_R * T_R - A_I * T_I
        sfpu_binop_push<OP_MUL>(dfb::a_r, dfb::t_r, dfb::tmp_r);
        sfpu_binop_push<OP_MUL>(dfb::a_i, dfb::t_i, dfb::tmp_i);
        cb_tmp_r.wait_front(1);
        cb_tmp_i.wait_front(1);
        sfpu_binop_push<OP_SUB>(dfb::tmp_r, dfb::tmp_i, dfb::b_r);
        cb_tmp_r.pop_front(1);
        cb_tmp_i.pop_front(1);

        // B_I = A_R * T_I + A_I * T_R
        sfpu_binop_push<OP_MUL>(dfb::a_r, dfb::t_i, dfb::tmp_r);
        sfpu_binop_push<OP_MUL>(dfb::a_i, dfb::t_r, dfb::tmp_i);
        cb_tmp_r.wait_front(1);
        cb_tmp_i.wait_front(1);
        sfpu_binop_push<OP_ADD>(dfb::tmp_r, dfb::tmp_i, dfb::b_i);
        cb_tmp_r.pop_front(1);
        cb_tmp_i.pop_front(1);

        cb_a_r.pop_front(1);
        cb_a_i.pop_front(1);
        cb_t_r.pop_front(1);
        cb_t_i.pop_front(1);
    }
}
