// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for scaled_dot_product_attention (Flash Attention).
//
// Stage 0 (init): Boot the Tensix compute pipeline. The reader initializes
// running-state CBs (cb_max_old, cb_sum_old, cb_o) via raw constant fills.
//
// Stage 1 (qkt_matmul): QK^T score matmul via matmul_block (transpose=true).
//   S = Q @ K^T → cb_scores (16 tiles)
//   - in0 = cb_q (B_q_t * D_t tiles, retained via WaitAndRetainOnLastBlock)
//   - in1 = cb_k (B_kv_t * D_t tiles, consumed via WaitAndPopPerKBlock)
//   - out = cb_scores (B_q_t * B_kv_t = 16 tiles)
//   - MatmulBlockShape::of(2, 2, 2, 2, D_t, 1)
//     in0_num_subblocks=2, in1_num_subblocks=2 (cover all 4 M-rows and 4 N-cols)
//     out_subblock_h=2, out_subblock_w=2 (4 tiles per subblock, DEST-safe)
//     in0_block_k=D_t, num_k_blocks=1 (full K-dim in one block)
//
// DPRINT: cb_scores tile 0, first 4×4 elements — Q @ K^T values (pre-scale).
// Printed from the unpacker (TRISC0) via DPRINT_UNPACK, front of CB
// (between cb_wait_front and cb_pop_front).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    // CB indices (match op_design.md CB layout).
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_scores = 24;

    // Compile-time args: [B_q_t, B_kv_t, D_t]
    constexpr uint32_t B_q_t = get_compile_time_arg_val(0);   // 4
    constexpr uint32_t B_kv_t = get_compile_time_arg_val(1);  // 4
    constexpr uint32_t D_t = get_compile_time_arg_val(2);     // D/32

    // Boot the matmul pipeline once (hw_configure-bearing — top of kernel_main).
    // transpose=1 (K^T), ct_dim=2 (out_subblock_w), rt_dim=2 (out_subblock_h),
    // kt_dim=D_t (in0_block_k).
    mm_block_init(cb_q, cb_k, cb_scores, /*transpose=*/1, /*ct_dim=*/2, /*rt_dim=*/2, /*kt_dim=*/D_t);

    // --- Stage 1: QK^T score matmul ---
    CircularBuffer q_buf(cb_q);
    CircularBuffer k_buf(cb_k);
    CircularBuffer scores_buf(cb_scores);

    // MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
    //                       out_subblock_h, out_subblock_w, in0_block_k, num_k_blocks)
    // M=B_q_t=4, N=B_kv_t=4, K=D_t.
    // in0_num_subblocks = M / out_subblock_h = 4/2 = 2
    // in1_num_subblocks = N / out_subblock_w = 4/2 = 2
    // out_subblock_h=2, out_subblock_w=2 → 4 tiles per subblock (DEST-safe with fp32 acc)
    // in0_block_k = D_t (full K-dim in one K-block)
    // num_k_blocks = 1
    constexpr auto qkt_shape = MatmulBlockShape::of(2, 2, 2, 2, D_t, 1);

    // transpose=true (K^T), packer_l1_acc=false, init_mode=Short (default),
    // in0_policy=WaitAndRetainOnLastBlock (retain Q; NoWaitNoPop is in1-only),
    // in1_policy=WaitAndPopPerKBlock (consume K per KV-block).
    matmul_block<
        /*transpose=*/true,
        /*packer_l1_acc=*/false,
        /*last_block_target=*/LastBlockTarget::Out,
        /*tile_order=*/OutputCBLayout::SubblockMajor,
        /*init_mode=*/matmul_config::InitMode::Short,
        /*in0_policy=*/InputPolicy::WaitAndRetainOnLastBlock,
        /*in1_policy=*/InputPolicy::WaitAndPopPerKBlock>(q_buf, k_buf, scores_buf, scores_buf, qkt_shape);

    // --- Stage 1 DPRINT: cb_scores tile 0, first 4×4 ---
    // Printed from the unpacker (TRISC0) — the only TRISC that can
    // cb_wait_front/cb_pop_front in a compute kernel. The matmul_block
    // helper has completed and pushed all 16 score tiles to cb_scores.
    // We wait_front tile 0, print it, then pop_front to keep the CB
    // balanced for later stages.
#ifdef TRISC_UNPACK
    {
        cb_wait_front(cb_scores, 1);
        SliceRange sr = SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1};
        DPRINT("stage_1 qkt_matmul:\n{}\n", TileSlice(cb_scores, 0, sr, true, true));
        cb_pop_front(cb_scores, 1);
    }
#endif
}
