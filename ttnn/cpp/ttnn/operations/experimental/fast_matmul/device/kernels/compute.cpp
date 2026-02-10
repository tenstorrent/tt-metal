// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;

    // DPRINT <<"M_block_tiles: " << M_block_tiles << ", K_block_tiles: " << K_block_tiles << ", N_block_tiles: " <<
    // N_block_tiles << ENDL(); DPRINT <<"M_blocks_per_core: " << M_blocks_per_core << ", N_blocks_per_core: " <<
    // N_blocks_per_core << ENDL(); DPRINT <<"K_num_blocks: " << K_num_blocks << ENDL();

    // DPRINT <<"subblock_h: " << subblock_h << ", subblock_w: " << subblock_w << ENDL();

    ckernel::mm_init(in0_cb, in1_cb, out_cb);
    ckernel::mm_block_init_short(
        in0_cb, in1_cb, false /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, K_block_tiles /*kt_dim*/);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    cb_wait_front(in0_cb, in0_block_num_tiles);
    cb_wait_front(in1_cb, in1_block_num_tiles);
    cb_reserve_back(out_cb, out_block_num_tiles);

    for (uint32_t m_block_index = 0; m_block_index < M_blocks_per_core * M_block_tiles; m_block_index += subblock_h) {
        uint32_t m_index = m_block_index * K_block_tiles * K_block_tiles;
        for (uint32_t n_block_index = 0; n_block_index < N_blocks_per_core * N_block_tiles;
             n_block_index += subblock_w) {
            uint32_t n_index = n_block_index;
            // Accumulation buffer
            ckernel::tile_regs_acquire();

            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                // DPRINT <<"Got in0 and in1 blocks for M block iter "<<m_block_iter<<" N block iter "<<n_block_iter<<"
                // K block iter "<<k_block<<ENDL(); UNPACK(
                //     DPRINT <<"in0 index: " << m_index <<"in1 index : "<< n_index << ENDL();
                // )

                ckernel::matmul_block(
                    in0_cb, in1_cb, m_index, n_index, 0, false, subblock_w, subblock_h, K_block_tiles);
                m_index += K_block_tiles;
                n_index += K_block_tiles * N_block_tiles * N_blocks_per_core;
            }
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();

            for (uint32_t i = 0; i < subblock_h * subblock_w; ++i) {
                ckernel::pack_tile(i, out_cb);
            }
            ckernel::tile_regs_release();
            cb_push_back(out_cb, subblock_h * subblock_w);

            // cb_push_back(intermediate_cb, out_block_num_tiles);
            // PACK((llk_pack_reconfig_l1_acc(0)));
            // cb_wait_front(intermediate_cb, out_block_num_tiles);
            // cb_reserve_back(out_cb, out_block_num_tiles);
            // #ifndef FUSE_BIAS
            //             copy_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            // #else
            //             cb_wait_front(in2_cb, N_block_tiles);
            //             add_bias_block(intermediate_cb, in2_cb, out_cb, M_block_tiles, N_block_tiles);
            //             cb_pop_front(in2_cb, N_block_tiles);
            // #endif
            // cb_pop_front(intermediate_cb, out_block_num_tiles);
        }
    }
}
