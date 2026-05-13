// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo_ring_common.h"
#include "tt-metalium/constants.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_kloop_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t reduce_semaphore_id = get_named_compile_time_arg_val("reduce_semaphore_id");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_out = tt::CBIndex::c_2;

    // Buffer wrappers consumed by matmul_kloop_pack: in0/in1 feed the K-loop's
    // FMA stream; out_buf receives the final pack_tile_block.
    experimental::CircularBuffer in0_buf(cb_s2c_in);
    experimental::CircularBuffer in1_buf(cb_r2c_w);
    experimental::CircularBuffer out_buf(cb_c2w_out);

    // Constants for the kernel
    constexpr uint32_t num_w_tiles_w = matmul_wo_ring::NUM_W_TILES_W;
    constexpr uint32_t num_n_tiles_per_iter = matmul_wo_ring::N_TILES_PER_ITER;
    constexpr uint32_t num_iters = num_w_tiles_w / num_n_tiles_per_iter;
    const uint32_t num_tiles_h = matmul_wo_ring::K_TILES_PER_CORE_A[dram_bank_id];

    //-------------------------------------------------------------------------
    // W reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w_txns_per_block = matmul_wo_ring::W_TXNS_PER_BLOCK;
    constexpr uint32_t w_tiles_per_txn = matmul_wo_ring::W_TILES_PER_TXN;
    constexpr uint32_t w_tiles_per_block = w_tiles_per_txn * w_txns_per_block;
    const uint32_t num_blocks_per_iter =
        (num_tiles_h * num_n_tiles_per_iter + w_tiles_per_block - 1) / w_tiles_per_block;

    //-------------------------------------------------------------------------
    // Compute
    //-------------------------------------------------------------------------
    // Pack is always configured to Float16_b
    pack_reconfig_data_format(cb_c2w_out);

    // Unpacker B is for input/activation and eltiwse inputs, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in);

    // Unpacker A is for W, so Bf8_b
    reconfig_data_format_srca(cb_r2c_w);

    // Initialize matmul
    mm_block_init(cb_s2c_in, cb_r2c_w, cb_c2w_out, /*transpose=*/false, /*ct_dim=*/7, /*rt_dim=*/1, /*kt_dim=*/1);

    //---------------------------------------------------------------------
    // Compute in @ W
    //---------------------------------------------------------------------
    uint32_t in0_index_base = 0;

    // We should read in0 at an offset that corresponds to the range of K that this core uses.
    for (uint32_t core_id = 0; core_id < dram_bank_id; ++core_id) {
        in0_index_base += matmul_wo_ring::K_TILES_PER_CORE_A[core_id];
    }

    // matmul_kloop_pack absorbs the DST scope and the segmented K-loop
    // (cb_wait / FMA stride / cb_pop on cb_r2c_w). KStepDefault advances
    // in0_index from in0_index_base by 1 per FMA; SimplePack handles the
    // pack_tile_block of num_n_tiles_per_iter tiles to cb_c2w_out.
    const SegmentedKLoopShape iter_shape =
        SegmentedKLoopShape::of(num_blocks_per_iter, w_tiles_per_block, /*ct_dim=*/num_n_tiles_per_iter);
    for (uint32_t iter_id = 0; iter_id < num_iters; ++iter_id) {
        KStepDefault<experimental::CircularBuffer> k_step{
            in0_buf, in1_buf, /*in0_index=*/in0_index_base, /*transpose=*/false};
        matmul_kloop_pack(
            in1_buf, iter_shape, k_step, SimplePack<experimental::CircularBuffer>{out_buf, num_n_tiles_per_iter});
    }

    // Drain the pipeline - the last dummy push
    cb_wait_front(cb_r2c_w, w_tiles_per_block);
    cb_pop_front(cb_r2c_w, w_tiles_per_block);
}
