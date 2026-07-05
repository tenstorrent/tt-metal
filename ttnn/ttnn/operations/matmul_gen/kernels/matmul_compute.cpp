// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// matmul compute (TRISC). One matmul_block per output block; the helper loops
// the K-blocks internally (waits/pops one K-block of in0/in1 per iteration,
// spills/reloads via cb_interm, packs the final SubblockMajor block to cb_out).
// All output blocks are identical work, so they are just called in lock-step
// with the reader/writer.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

using namespace compute_kernel_lib;

constexpr uint32_t CB_IN0_ACT = 0;
constexpr uint32_t CB_IN1_WEIGHT = 1;
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t CB_INTERM = 24;

void kernel_main() {
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(0);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(3);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t total_blocks = get_compile_time_arg_val(6);
    // Refinements 1 (Lever B) + 1b: when 1, use hardware packer L1 accumulation so the
    // last-K-block pack-to-out data-format reconfig (gated on packer_l1_acc ||
    // fp32_dest_acc_en) fires even with fp32_dest_acc_en=False. That lets cb_interm
    // carry a format FINER than a low-precision cb_out, so the running K-sum
    // accumulates in L1 in that finer format across all K-blocks instead of
    // re-quantizing to the output format each spill: bf8b out -> bf16 interm
    // (Lever B), bf16 out -> fp32 interm (Refinement 1b). The 16-bit DEST is still
    // honored per K-block; only the cross-K-block L1 accumulation is finer.
    constexpr uint32_t packer_l1_acc = get_compile_time_arg_val(7);

    CircularBuffer in0_buf(CB_IN0_ACT);
    CircularBuffer in1_buf(CB_IN1_WEIGHT);
    CircularBuffer out_buf(CB_OUT);
    CircularBuffer interm_buf(CB_INTERM);

    // Boot once (hw_configure-bearing init is the caller's one-time job).
    compute_kernel_hw_startup(CB_IN0_ACT, CB_IN1_WEIGHT, CB_OUT);
    mm_block_init(CB_IN0_ACT, CB_IN1_WEIGHT, CB_OUT, 0 /*transpose*/, out_subblock_w, out_subblock_h, in0_block_w);

    // All per-core output blocks are identical work (same subblock shape, same
    // K-blocking) with NO per-block compute phase in between, so pass the block
    // count as the helper's `batch`: matmul_block inits once and loops the full
    // K-pipeline `total_blocks` times internally (one output block pushed per
    // iteration, in lock-step with the reader/writer). This is the helper's
    // documented matmul-only pattern — strictly fewer mm_block_init_short MMIO
    // re-issues than an external per-block loop, identical CB push/pop counts.
    const auto shape = MatmulBlockShape::of(
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w,
        in0_block_w,
        num_k_blocks,
        total_blocks /*batch — every per-core block shares shape + init*/);

    // Template param 2 is packer_l1_acc. Branch at compile time so the default
    // (software spill/reload) path is unchanged for every cell except the bf8b
    // acc=False deep-K corner the descriptor opts into L1-acc for.
    if constexpr (packer_l1_acc != 0) {
        matmul_block<false /*transpose*/, true /*packer_l1_acc*/>(in0_buf, in1_buf, out_buf, interm_buf, shape);
    } else {
        matmul_block<>(in0_buf, in1_buf, out_buf, interm_buf, shape);
    }
}
