// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// batch_fft_compute.cpp — TRISC compute for device-side BATCH FFT.
//

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

constexpr uint32_t LOG2_SUB_N = get_arg(args::log2_sub_n);

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

template <typename AR, typename AI, typename BR, typename BI, typename OutR, typename OutI>
FORCE_INLINE void cmul(AR ar, AI ai, BR br, BI bi, OutR outr, OutI outi) {
    DataflowBuffer cb_tmp_r(dfb::tmp_r);
    DataflowBuffer cb_tmp_i(dfb::tmp_i);
    sfpu_binop_push<OP_MUL>(ar, br, dfb::tmp_r);
    sfpu_binop_push<OP_MUL>(ai, bi, dfb::tmp_i);
    cb_tmp_r.wait_front(1);
    cb_tmp_i.wait_front(1);
    sfpu_binop_push<OP_SUB>(dfb::tmp_r, dfb::tmp_i, outr);
    cb_tmp_r.pop_front(1);
    cb_tmp_i.pop_front(1);

    sfpu_binop_push<OP_MUL>(ar, bi, dfb::tmp_r);
    sfpu_binop_push<OP_MUL>(ai, br, dfb::tmp_i);
    cb_tmp_r.wait_front(1);
    cb_tmp_i.wait_front(1);
    sfpu_binop_push<OP_ADD>(dfb::tmp_r, dfb::tmp_i, outi);
    cb_tmp_r.pop_front(1);
    cb_tmp_i.pop_front(1);
}

void kernel_main() {
    const uint32_t batch_per_core = get_arg(args::batch_per_core);

    compute_kernel_hw_startup(dfb::even_r, dfb::out0_r);
    copy_init(dfb::even_r);

    DataflowBuffer cb_even_r(dfb::even_r);
    DataflowBuffer cb_even_i(dfb::even_i);
    DataflowBuffer cb_odd_r(dfb::odd_r);
    DataflowBuffer cb_odd_i(dfb::odd_i);
    DataflowBuffer cb_tw_r(dfb::twiddle_r);
    DataflowBuffer cb_tw_i(dfb::twiddle_i);
    DataflowBuffer cb_tw_odd_r(dfb::tw_odd_r);
    DataflowBuffer cb_tw_odd_i(dfb::tw_odd_i);

    for (uint32_t k = 0; k < batch_per_core; ++k) {
        for (uint32_t s = 0; s < LOG2_SUB_N; ++s) {
            cb_even_r.wait_front(1);
            cb_even_i.wait_front(1);
            cb_odd_r.wait_front(1);
            cb_odd_i.wait_front(1);
            cb_tw_r.wait_front(1);
            cb_tw_i.wait_front(1);

            cmul(dfb::odd_r, dfb::odd_i, dfb::twiddle_r, dfb::twiddle_i, dfb::tw_odd_r, dfb::tw_odd_i);

            cb_odd_r.pop_front(1);
            cb_odd_i.pop_front(1);
            cb_tw_r.pop_front(1);
            cb_tw_i.pop_front(1);

            cb_tw_odd_r.wait_front(1);
            cb_tw_odd_i.wait_front(1);

            sfpu_binop_push<OP_ADD>(dfb::even_r, dfb::tw_odd_r, dfb::out0_r);
            sfpu_binop_push<OP_ADD>(dfb::even_i, dfb::tw_odd_i, dfb::out0_i);
            sfpu_binop_push<OP_SUB>(dfb::even_r, dfb::tw_odd_r, dfb::out1_r);
            sfpu_binop_push<OP_SUB>(dfb::even_i, dfb::tw_odd_i, dfb::out1_i);

            cb_even_r.pop_front(1);
            cb_even_i.pop_front(1);
            cb_tw_odd_r.pop_front(1);
            cb_tw_odd_i.pop_front(1);
        }
    }
}
