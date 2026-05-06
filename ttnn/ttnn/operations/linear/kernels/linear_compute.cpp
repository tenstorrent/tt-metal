// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// linear compute (Phase 0): TRISC unpack/math/pack.
//
// Pipeline:
//
//   no-bias:
//     compute_kernel_hw_startup(cb_input, cb_weight, cb_output)
//     matmul_block<LastBlockTarget::Out>(input, weight, output, output, ...)
//
//   with bias:
//     compute_kernel_hw_startup(cb_input, cb_weight, cb_partials)
//     matmul_block<LastBlockTarget::Interm>(input, weight, output, partials, ...)
//     cb_wait_front(cb_bias_tiles, Nt)
//     add_bias_bcast_rows<RowBroadcast, SubblockMajor>(partials, bias, output, ...)
//     cb_pop_front(cb_bias_tiles, Nt)
//
// Subblock geometry: in0_num_subblocks = Mt, in1_num_subblocks = Nt,
// out_subblock_h = out_subblock_w = 1, in0_block_w = Kt, num_k_blocks = 1.
// Each subblock holds 1 tile in DEST — far below the 8-tile bf16 limit.
// SubblockMajor pack with 1×1 subblocks emits tiles in tile-row-major
// (m, n) order, which matches the writer's DRAM-interleaved page indexing.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"

void kernel_main() {
    // ── Compile-time args ───────────────────────────────────────────────
    constexpr uint32_t has_bias = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);

    // ── CB indices (semantic) ───────────────────────────────────────────
    constexpr uint32_t cb_input_tiles = 0;
    constexpr uint32_t cb_weight_tiles = 1;
    constexpr uint32_t cb_bias_tiles = 2;
    constexpr uint32_t cb_output_tiles = 16;
    constexpr uint32_t cb_partials = 24;

    // ── Buffer-object wrappers required by the helpers ─────────────────
    experimental::CircularBuffer input_buf(cb_input_tiles);
    experimental::CircularBuffer weight_buf(cb_weight_tiles);
    experimental::CircularBuffer output_buf(cb_output_tiles);

    // Subblock geometry: 1-tile subblocks, single K-block.
    // in0_num_subblocks=Mt, in1_num_subblocks=Nt, in0_block_w=Kt, num_k_blocks=1.
    constexpr auto matmul_shape = compute_kernel_lib::MatmulBlockShape::of(
        Mt,  // in0_num_subblocks
        Nt,  // in1_num_subblocks
        1,   // out_subblock_h
        1,   // out_subblock_w
        Kt,  // in0_block_w (entire K dim)
        1,   // num_k_blocks
        1    // batch
    );

    if constexpr (has_bias) {
        // Bias path: matmul packs to cb_partials, then bias adds onto cb_output.
        experimental::CircularBuffer partials_buf(cb_partials);
        experimental::CircularBuffer bias_buf(cb_bias_tiles);

        // Hardware startup names the matmul's pack target (= cb_partials in
        // bias mode). All three CBs that the matmul uses (input/weight/partials)
        // are referenced; the bias and output CBs come up later through
        // add_bias_bcast_rows's reconfig.
        compute_kernel_hw_startup(cb_input_tiles, cb_weight_tiles, cb_partials);

        // Matmul → cb_partials. With num_k_blocks=1 the last/only block is
        // packed via LastBlockTarget::Interm into partials_buf; output_buf is
        // unread on this path but the helper signature still requires a valid
        // buffer reference for the out_buf slot.
        compute_kernel_lib::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            compute_kernel_lib::LastBlockTarget::Interm,
            compute_kernel_lib::OutputLayout::SubblockMajor>(
            input_buf, weight_buf, output_buf, partials_buf, matmul_shape);

        // Bias CB lifecycle is caller-owned (helper does NOT touch wait/pop on
        // bias_cb — see bias_add_helpers.hpp:93-97). Wait once across the whole
        // bias add, pop once after.
        cb_wait_front(cb_bias_tiles, Nt);

        // Bias add → cb_output. Layout MUST match upstream matmul (both
        // SubblockMajor) so cb_partials is consumed in the same order it was
        // produced. out_row_width=0 is fine — only consulted under RowMajor.
        compute_kernel_lib::add_bias_bcast_rows<
            compute_kernel_lib::BiasBroadcast::RowBroadcast,
            compute_kernel_lib::OutputLayout::SubblockMajor>(
            partials_buf, bias_buf, output_buf, compute_kernel_lib::BiasAddShape::of(Mt, Nt, 1, 1));

        cb_pop_front(cb_bias_tiles, Nt);
    } else {
        // No-bias path: matmul packs directly to cb_output_tiles.
        compute_kernel_hw_startup(cb_input_tiles, cb_weight_tiles, cb_output_tiles);

        // interm_buf is unused when num_k_blocks==1 but the signature requires
        // a valid buffer reference — pass output_buf (same buffer type, never
        // read on this path; documented gotcha #4).
        compute_kernel_lib::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            compute_kernel_lib::LastBlockTarget::Out,
            compute_kernel_lib::OutputLayout::SubblockMajor>(
            input_buf, weight_buf, output_buf, output_buf, matmul_shape);
    }
}
