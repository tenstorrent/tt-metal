// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — compute-core TRISC: one matmul_block per scatter block (in compute order).
//
// Per block: core_m_tiles x core_n_tiles output tiles over Kt, in num_k_blocks K-blocks of k_block_tiles, packer L1
// accumulation in cb_partial_accum, last K-block packed bf16 into the block's cb_partial_handoff slot in
// TileRowMajor order (tile (r, c) at page r * core_n_tiles + c) so the transport cores gather a segment piece with
// one contiguous NoC read.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_act_operand = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_operand = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial_accum = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partial_handoff = get_compile_time_arg_val(3);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t k_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(10);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_operand, cb_weight_operand, cb_partial_handoff);

    CircularBuffer in0_buf(cb_act_operand), in1_buf(cb_weight_operand);
    CircularBuffer out_buf(cb_partial_handoff), interm_buf(cb_partial_accum);
    constexpr auto shape = ckl::MatmulBlockShape::of(
        in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, k_block_tiles, num_k_blocks);

    // Stage zone (permanent; opt-in via KERNEL_PERF_ZONES): one per scatter block. The helper waits on the operand
    // CBs per K-block and reserves the hand-off slot internally, so this is OCCUPANCY (math + operand starvation +
    // hand-off back-pressure), not payload; MMRS_ABLATE_MATMUL (perf ablation only, wrong results) keeps the CB
    // handshakes and drops the math.
    for (uint32_t b = 0; b < num_blocks; ++b) {
        MaybeDeviceZoneScope("compute_matmul_block");
#ifdef MMRS_ABLATE_MATMUL
        constexpr uint32_t cm = in0_num_subblocks * out_subblock_h, cn = in1_num_subblocks * out_subblock_w;
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            cb_wait_front(cb_act_operand, cm * k_block_tiles);
            cb_wait_front(cb_weight_operand, k_block_tiles * cn);
            cb_pop_front(cb_act_operand, cm * k_block_tiles);
            cb_pop_front(cb_weight_operand, k_block_tiles * cn);
        }
        cb_reserve_back(cb_partial_handoff, cm * cn);
        cb_push_back(cb_partial_handoff, cm * cn);
        (void)interm_buf;
        (void)shape;
#else
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/true,
            ckl::LastBlockTarget::Out,
            ckl::OutputCBLayout::TileRowMajor>(in0_buf, in1_buf, out_buf, interm_buf, shape);
#endif
    }
}
