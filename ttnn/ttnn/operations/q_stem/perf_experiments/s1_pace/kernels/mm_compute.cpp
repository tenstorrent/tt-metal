// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// s1_pace stage-1 matmul per block, isolated. q block (M x cols) = x round (M x K, bf16) @ resident q_b slice
// (K x cols, bfp8_b), HiFi2, 16-bit DEST (the PERF FOCUS contract; set by the host). Variants (MM_V):
//   0 helper        the op today: kernel_lib matmul_block, InitMode::Short + reconfig per block, QbKBlockWait
//   1 helper_hoist  the same helper call with InitMode::None + DataFormatReconfig::NONE, one init before the loop
//   2 raw           raw LLK (bypasses kernel_lib matmul_block): one matmul_block_init before the loop, per block one
//                   DEST acquire, K x ckernel::matmul_block(ct=cols, rt=M), pack cols*M tiles. Same LLK calls and
//                   K order as the helper -> bit-identical; drops the helper's per-block init/reconfig, its
//                   per-K-block callbacks and its subblock bookkeeping.
//   3 raw_ct1       raw, ct_dim = 1 per column (in0 re-unpacked per column): unpacker-cost probe
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

using namespace compute_kernel_lib;

namespace {
struct QbKBlockWait {
    uint32_t cb, tiles_per_kblock;
    ALWI void operator()(uint32_t kb, uint32_t, bool) const { cb_wait_front(cb, (kb + 1) * tiles_per_kblock); }
};
struct QbKBlockOffset {
    uint32_t tiles_per_kblock;
    ALWI uint32_t operator()(uint32_t block) const { return block * tiles_per_kblock; }
};
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_qb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t M = get_compile_time_arg_val(5);
    constexpr uint32_t NB = get_compile_time_arg_val(6);
    constexpr uint32_t sbw = get_compile_time_arg_val(7);

    CircularBuffer x_buf(cb_x), qb_buf(cb_qb), q_buf(cb_out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_qb, cb_out);

#if MM_V == 0
    const auto shape = MatmulBlockShape::of(M, cols / sbw, 1, sbw, K, 1);
    for (uint32_t j = 0; j < NB; ++j) {
        matmul_block<
            false,
            false,
            LastBlockTarget::Out,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndPopPerKBlock,
            InputPolicy::NoWaitNoPop,
            NoPostCompute,
            QbKBlockWait,
            NoPostKBlock,
            0,
            NoKBlockInnerDimFn,
            NoIn0Source,
            QbKBlockOffset>(
            x_buf,
            qb_buf,
            q_buf,
            q_buf,
            shape,
            {},
            QbKBlockWait{cb_qb, K * cols},
            cols,
            0,
            {},
            {},
            {},
            QbKBlockOffset{K * cols});
    }
#elif MM_V == 1
    const auto shape = MatmulBlockShape::of(M, cols / sbw, 1, sbw, K, 1);
    cb_wait_front(cb_qb, K * cols);
    reconfig_data_format(cb_qb, cb_x);
    PACK((pack_reconfig_data_format(cb_out)));
    matmul_block_init(cb_x, cb_qb, false, sbw, 1, K);
    for (uint32_t j = 0; j < NB; ++j) {
        matmul_block<
            false,
            false,
            LastBlockTarget::Out,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::None,
            InputPolicy::WaitAndPopPerKBlock,
            InputPolicy::NoWaitNoPop,
            NoPostCompute,
            NoPreKBlock,
            NoPostKBlock,
            0,
            NoKBlockInnerDimFn,
            NoIn0Source,
            NoIn1BaseOffset,
            false,
            NoneActivation,
            matmul_config::DataFormatReconfig::NONE>(x_buf, qb_buf, q_buf, q_buf, shape, {}, {}, cols, 0);
    }
#elif MM_V == 2 || MM_V == 3
    constexpr uint32_t ct = MM_V == 2 ? cols : 1;
    cb_wait_front(cb_qb, K * cols);
    matmul_block_init(cb_x, cb_qb, false, ct, M, K);
    for (uint32_t j = 0; j < NB; ++j) {
        cb_wait_front(cb_x, M * K);
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; ++k) {
            if constexpr (MM_V == 2) {
                matmul_block(cb_x, cb_qb, k, k * cols, 0, false, cols, M, K);
            } else {
                for (uint32_t c = 0; c < cols; ++c) {
                    matmul_block(cb_x, cb_qb, k, k * cols + c, c, false, 1, M, K);  // M == 1
                }
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, M * cols);
        tile_regs_wait();
        for (uint32_t t = 0; t < M * cols; ++t) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, M * cols);
        cb_pop_front(cb_x, M * K);
    }
#endif
    cb_pop_front(cb_qb, K * cols);
}
