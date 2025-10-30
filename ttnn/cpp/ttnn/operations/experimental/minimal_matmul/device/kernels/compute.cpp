// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include <tt-metalium/constants.hpp>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "tools/profiler/kernel_profiler.hpp"

// FIXME: this shouldn't be statically allocated
constexpr uint32_t PERF_INPUT_A = 0x1A000;
constexpr uint32_t PERF_INPUT_B = PERF_INPUT_A + 16 * 4096;
constexpr uint32_t PERF_INPUT_C = PERF_INPUT_B + 16 * 4096;
constexpr uint32_t PERF_OUTPUT = PERF_INPUT_C + 16 * 4096;

constexpr uint32_t PERF_ADDRESS(uint32_t buffer, uint32_t tile) {
    uint32_t address = buffer + (tile % 16) * 4096;  // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                         // Correct the L1 Address for Tensix
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    copy_tile_to_dst_init_short(in_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst();
        copy_tile(in_cb, i, 0 /*dst*/);
        pack_tile<true>(0, out_cb, i);
        release_dst();
    }
}

// Slightly modified from compute_common.hpp
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t M_num_subblocks,
    const uint32_t N_num_subblocks,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const bool accumulate_intermediate) {
#if 0
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    mm_block_init_short(
        in0_cb, in1_cb, false /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, K_block_tiles /*kt_dim*/);

    uint32_t in0_index_offset = 0;

    reconfig_data_format(in1_cb, in0_cb);
    pack_reconfig_data_format(out_cb);

    // asm volatile("ebreak");
    // if (accumulate_intermediate) {
    //     PACK((llk_pack_reconfig_l1_acc(1)));
    // } else {
    //     PACK((llk_pack_reconfig_l1_acc(0)));
    // }

    DPRINT << "M_num_subblocks: " << M_num_subblocks
           << ", N_num_subblocks: " << N_num_subblocks
           << ", K_block_tiles: " << K_block_tiles
           << ENDL();

    for (uint32_t M_subblock = 0; M_subblock < M_num_subblocks; ++M_subblock) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_subblock = 0; N_subblock < N_num_subblocks; ++N_subblock) {
            // DeviceZoneScopedN("MATMUL_BLOCK");
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                DPRINT << "sublock_w: " << subblock_w << ", subblock_h: " << subblock_h
                       << ", K_block_tiles: " << K_block_tiles
                       << ENDL();
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, false, subblock_w, subblock_h, K_block_tiles);
                in0_index++;
                in1_index += N_block_tiles;
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_subblock * subblock_h + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_subblock * subblock_w + w;
                    uint32_t out_tile_id = h_tile_id * N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                    dst_index++;
                }
            }
            tile_regs_release();

            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }

    // PACK((llk_pack_reconfig_l1_acc(0)));
#endif
    for (uint32_t loop = 0; loop < 16; loop++) {
        for (uint32_t tile = 0; tile < 4; tile++) {
            // _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, tile));
            pack_tile<true>(loop, out_cb, tile);
        }
    }
}

void safe_print_full_tile(uint32_t cb_id) {
#if defined(DEBUG_PRINT_ENABLED)
    UNPACK(tt::compute::common::print_full_tile(cb_id));
#endif
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    mm_init(in0_cb, in1_cb, intermediate_cb);

    constexpr uint32_t M_num_blocks = M_tiles / M_block_tiles;
    constexpr uint32_t K_num_blocks = K_tiles / K_block_tiles;
    constexpr uint32_t N_num_blocks = N_tiles / N_block_tiles;

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t M_num_subblocks = M_block_tiles / subblock_h;
    constexpr uint32_t N_num_subblocks = N_block_tiles / subblock_w;

    // mm_block_init_short(
    //     in0_cb, in1_cb, false /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, K_block_tiles /*kt_dim*/);

    // reconfig_data_format(in1_cb, in0_cb);
    // pack_reconfig_data_format(intermediate_cb);

    PACK((_llk_pack_hw_configure_<true>(0, 0, 4)));
    PACK((_llk_pack_init_<
          /* untilize */ false,
          /* zero_output */ false,
          DstTileFaceLayout::RowMajor,
          /* write_tile_header */ false>(0)));
    PACK((_llk_pack_dest_init_<DstSync::SyncHalf, true>()));
    ckernel::tensix_sync();

    // asm volatile("ebreak");
    // for(int i = 0; i < 1; i++)
    {
        DeviceZoneScopedN("MATMUL_BLOCKS");
        for (uint32_t loop = 0; loop < 256; loop++) {
            for (uint32_t tile = 0; tile < 4; tile++) {
                // Testing Packer API
                // pack_tile<false>(tile, out_cb, tile);

                // Testing Packer llk
                PACK((_llk_pack_<DstSync::SyncHalf, true>(tile, PERF_ADDRESS(PERF_OUTPUT, tile))));

                // Testing llk initialization in loop
                // PACK((_llk_pack_hw_configure_<true>(0, 0, 4)));
                // PACK((_llk_pack_init_</* untilize */ false,/* zero_output */ false,DstTileFaceLayout::RowMajor,/*
                // write_tile_header */ false>(0))); PACK((_llk_pack_dest_init_<DstSync::SyncHalf, true>()));

                // Testing RISC counter
                // volatile uint32_t counter = 0;
                // for (uint32_t i = 0; i < 1000; i++)
                // {
                //     counter++;
                // }
            }
        }
        ckernel::tensix_sync();
    }

    // for (uint32_t m_block = 0; m_block < M_num_blocks; m_block++) {
    //     for (uint32_t n_block = 0; n_block < N_num_blocks; n_block++) {
    //         // Accumulation buffer
    //         // cb_reserve_back(intermediate_cb, out_block_num_tiles);
    //         for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
    //             //cb_wait_front(in0_cb, in0_block_num_tiles);
    //             //cb_wait_front(in1_cb, in1_block_num_tiles);
    //             {
    //                 DeviceZoneScopedN("MATMUL_BLOCKS");
    //                 // // safe_print_full_tile(in0_cb);
    //                 // // safe_print_full_tile(in1_cb)
    //                 // DPRINT << "M_num_blocks: " << M_num_blocks
    //                 //     << ", N_num_blocks: " << N_num_blocks
    //                 //     << ", K_num_blocks: " << K_num_blocks
    //                 //     << ENDL();
    //                 // DPRINT << "matmul on m_block: " << m_block << ", n_block: " << n_block << ", k_block: " <<
    //                 k_block
    //                 //     << ENDL();
    //                 // matmul_blocks(
    //                 //     in0_cb,
    //                 //     in1_cb,
    //                 //     intermediate_cb,
    //                 //     M_block_tiles,
    //                 //     N_block_tiles,
    //                 //     K_block_tiles,
    //                 //     M_num_subblocks,
    //                 //     N_num_subblocks,
    //                 //     subblock_h,
    //                 //     subblock_w,
    //                 //     k_block > 0);

    //             }
    //             //cb_pop_front(in0_cb, in0_block_num_tiles);
    //             //cb_pop_front(in1_cb, in1_block_num_tiles);
    //         }
    //         // cb_push_back(intermediate_cb, out_block_num_tiles);
    //         // cb_wait_front(intermediate_cb, out_block_num_tiles);
    //         // safe_print_full_tile(intermediate_cb);
    //         //cb_reserve_back(out_cb, out_block_num_tiles);
    //         //copy_block(intermediate_cb, out_cb, out_block_num_tiles);
    //         //cb_push_back(out_cb, out_block_num_tiles);
    //         // cb_pop_front(intermediate_cb, out_block_num_tiles);
    //     }
    // }
}
}  // namespace NAMESPACE
