// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker compute: one kernel_lib matmul_block over num_k_blocks K-steps. Every K-step (the
// first plain, the rest with packer L1 accumulation) packs straight into cb_out_block, which compute
// reserves once and pushes once at the end (caller_owns_pack_target, TileRowMajor, LastBlockTarget::Interm
// with the pack target = the output CB).
// Perf 1: the helper's LastBlockTarget::Out + L1-acc path spilled each K-step into cb_interm_acc and drained
// it on the UNPACK thread before the next K-step (and reloaded it on the last), so unpack of K-step k+1 waited
// for the packer to finish K-step k — 12 exposed pack latencies. Compute alone 15.34 -> 12.17 us (matmul zone).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Zones (permanent): compute_wait_x0 = unpack waiting for x round 0 + W block 0 (hoisted out of the helper so
// pipeline fill is not charged to the matmul); compute_matmul = the whole matmul_block (its later per-K-step waits
// included: occupancy on TRISC_0/2, wait+work on TRISC_1). Ablation: SKIP_COMPUTE (helper switch).

// ---- tail_store2 variants (defines via pd.TAIL) ----
// TAIL_LAST=1|2: the helper runs K-steps 0..n-2 (unchanged: caller_owns, Interm, L1-acc into cb_out_block,
//   TileRowMajor); the LAST K-step is raw LLK in groups of 1x1 (=1) or 1x2 (=2, one output row) tiles, in
//   TileRowMajor order, each packed with L1-acc onto its partial and PUSHED at once, so the writer stores
//   group g while compute does group g+1. Bypasses compute_kernel_lib::matmul_block for the last K-step only.
//   Why (capability): with caller_owns_pack_target the helper never pushes (one caller push at the end), and
//   without it TileRowMajor pushes per M-row-group only through the interm-drain path Perf 1 removed; its
//   subblock (5x1) is a column, which is not a contiguous prefix of the TileRowMajor CB, so no per-subblock push
//   is expressible either. A separate helper call for the last K-step cannot L1-accumulate (block==0 packs
//   with L1-acc off).
// TAIL_RAWCOL: fully raw compute (Perf 1 compute_llk_inner candidate): 5x1 subblocks, column-major output
//   (tile (m,n) at n*5+m), L1-acc every K-step, each column pushed as soon as its last-K-step pack lands.

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_block = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_block = get_compile_time_arg_val(2);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(5);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t out_block_tiles = in0_num_subblocks * out_subblock_h * in1_num_subblocks * out_subblock_w;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_block, cb_out_block);

    CircularBuffer x_buf(cb_x_resident), w_buf(cb_w_block), out_buf(cb_out_block);
    {
        MaybeDeviceZoneScope("compute_wait_x0");
        cb_wait_front(cb_x_resident, in0_num_subblocks * out_subblock_h * block_k_tiles);  // x round 0
        cb_wait_front(cb_w_block, block_k_tiles * in1_num_subblocks * out_subblock_w);     // W block 0
    }
    MaybeDeviceZoneScope("compute_matmul");
#ifdef TAIL_RAWCOL
    constexpr uint32_t block_m = in0_num_subblocks * out_subblock_h;
    constexpr uint32_t block_n = in1_num_subblocks * out_subblock_w;
    constexpr uint32_t x_round = block_m * block_k_tiles;
    constexpr uint32_t w_block = block_k_tiles * block_n;
    static_assert(out_subblock_h == block_m && out_subblock_w == 1, "TAIL_RAWCOL: 5x1-style column subblocks");
    matmul_block_init(cb_x_resident, cb_w_block, 0, 1, block_m, block_k_tiles);
    cb_reserve_back(cb_out_block, out_block_tiles);
    PACK((llk_pack_reconfig_l1_acc(0)));
    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        const bool last = kb == num_k_blocks - 1;
        if (kb > 0) {
            cb_wait_front(cb_x_resident, x_round);
            cb_wait_front(cb_w_block, w_block);
        }
        if (kb == 1) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        uint32_t pushed = 0;
        for (uint32_t n = 0; n < block_n; ++n) {
            tile_regs_acquire();
            for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
                matmul_block(cb_x_resident, cb_w_block, kk, kk * block_n + n, 0, 0, 1, block_m, block_k_tiles);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < block_m; ++r) {
                pack_tile<true>(r, cb_out_block, n * block_m + r - pushed);
            }
            tile_regs_release();
            if (last) {
                cb_push_back(cb_out_block, block_m);
                pushed += block_m;
            }
        }
        cb_pop_front(cb_x_resident, x_round);
        cb_pop_front(cb_w_block, w_block);
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
    return;
#endif
#ifdef TAIL_LAST
    constexpr uint32_t helper_k_blocks = num_k_blocks - 1;
    static_assert(num_k_blocks >= 3, "TAIL_LAST: the helper must leave L1-acc on");
#else
    constexpr uint32_t helper_k_blocks = num_k_blocks;
#endif
    out_buf.reserve_back(out_block_tiles);
    compute_kernel_lib::matmul_block<
        false /*transpose*/,
        true /*packer_l1_acc*/,
        compute_kernel_lib::LastBlockTarget::Interm,
        compute_kernel_lib::OutputCBLayout::TileRowMajor,
        compute_kernel_lib::matmul_config::InitMode::Short,
        compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
        compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
        compute_kernel_lib::NoPostCompute,
        compute_kernel_lib::NoPreKBlock,
        compute_kernel_lib::NoPostKBlock,
        0,
        compute_kernel_lib::NoKBlockInnerDimFn,
        compute_kernel_lib::NoIn0Source,
        compute_kernel_lib::NoIn1BaseOffset,
        true /*caller_owns_pack_target*/>(
        x_buf,
        w_buf,
        out_buf,
        out_buf /* pack target = the output CB itself */,
        compute_kernel_lib::MatmulBlockShape::of(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, block_k_tiles, helper_k_blocks));
#ifdef TAIL_LAST
    {
        // raw last K-step: groups of 1 x TAIL_LAST tiles in TileRowMajor order, L1-acc onto the partials, push each.
        constexpr uint32_t gw = TAIL_LAST;
        constexpr uint32_t block_m = in0_num_subblocks * out_subblock_h;
        constexpr uint32_t block_n = in1_num_subblocks * out_subblock_w;
        static_assert(block_n % gw == 0, "TAIL_LAST group width must divide block_n");
        constexpr uint32_t x_round = block_m * block_k_tiles;
        constexpr uint32_t w_block = block_k_tiles * block_n;
        cb_wait_front(cb_x_resident, x_round);
        cb_wait_front(cb_w_block, w_block);
        matmul_block_init(cb_x_resident, cb_w_block, 0, gw, 1, block_k_tiles);
        PACK((pack_reconfig_data_format(cb_out_block)));
        PACK((llk_pack_reconfig_l1_acc(1)));
        for (uint32_t m = 0; m < block_m; ++m) {
            for (uint32_t n0 = 0; n0 < block_n; n0 += gw) {
                tile_regs_acquire();
                for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
#ifndef SKIP_COMPUTE
                    matmul_block(
                        cb_x_resident,
                        cb_w_block,
                        m * block_k_tiles + kk,
                        kk * block_n + n0,
                        0,
                        0,
                        gw,
                        1,
                        block_k_tiles);
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < gw; ++c) {
                    pack_tile<true>(c, cb_out_block, c);  // wr_ptr advances with each push
                }
                tile_regs_release();
                cb_push_back(cb_out_block, gw);
            }
        }
        cb_pop_front(cb_x_resident, x_round);
        cb_pop_front(cb_w_block, w_block);
    }
#else
    out_buf.push_back(out_block_tiles);
#endif
    PACK((llk_pack_reconfig_l1_acc(0)));
}
