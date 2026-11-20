// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.grouped_matmul compute: out [MT x PN] = sum over the K blocks of in0 [MT x BW] @ in1 [BW x PN].
// Per K block and output row tile m: DST (fp32) accumulates the BW products of the block (matmul_block, 1 x PN
// tiles), then the packer adds them to the Float32 partials in L1 (block 0 overwrites; the packer writes Float32, so
// the sum over blocks is an fp32 sum). After the last block each partial tile is copied (unpacked to DEST in fp32)
// and packed once in the output format.
// RT (compile-time, default 1): row tiles a DST block (RT x PN tiles; RT > 1 with full-sync DEST): fewer unpacks per
// product (RT + PN unpacks for RT PN products); the last block of a K block takes the remainder MT % RT.
// in1 blocks are BW x PNX tiles (row stride PNX >= PN, the reader's layout).
// compile-time args: MT, BW, PN, NBLK, CB_IN0, CB_IN1, CB_PART, CB_OUT, RT, PNX

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t BW = get_compile_time_arg_val(1);
    constexpr uint32_t PN = get_compile_time_arg_val(2);
    constexpr uint32_t NBLK = get_compile_time_arg_val(3);
    constexpr uint32_t CB_IN0 = get_compile_time_arg_val(4);
    constexpr uint32_t CB_IN1 = get_compile_time_arg_val(5);
    constexpr uint32_t CB_PART = get_compile_time_arg_val(6);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(7);
    constexpr uint32_t RT = get_compile_time_arg_val(8);
    constexpr uint32_t PNX = get_compile_time_arg_val(9);
    constexpr uint32_t OUT_TILES = MT * PN;
    constexpr uint32_t RT_LAST = MT % RT;

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_IN1, CB_PART);  // (matmul: in1 -> SrcA, in0 -> SrcB)
    matmul_block_init(CB_IN0, CB_IN1, false, PN, RT, BW);
    cb_reserve_back(CB_PART, OUT_TILES);
    for (uint32_t kb = 0; kb < NBLK; ++kb) {
        cb_wait_front(CB_IN0, MT * BW);
        cb_wait_front(CB_IN1, BW * PNX);
        pack_reconfig_l1_acc(kb > 0 ? 1 : 0);
        for (uint32_t m = 0; m + RT <= MT; m += RT) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < BW; ++k) {
                matmul_block(CB_IN0, CB_IN1, m * BW + k, k * PNX, 0, false, PN, RT, BW);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < RT; ++r) {
                for (uint32_t j = 0; j < PN; ++j) {
                    pack_tile<true>(r * PN + j, CB_PART, (m + r) * PN + j);
                }
            }
            tile_regs_release();
        }
        if constexpr (RT_LAST > 0) {
            constexpr uint32_t m = MT - RT_LAST;
            matmul_block_init(CB_IN0, CB_IN1, false, PN, RT_LAST, BW);
            tile_regs_acquire();
            for (uint32_t k = 0; k < BW; ++k) {
                matmul_block(CB_IN0, CB_IN1, m * BW + k, k * PNX, 0, false, PN, RT_LAST, BW);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < RT_LAST; ++r) {
                for (uint32_t j = 0; j < PN; ++j) {
                    pack_tile<true>(r * PN + j, CB_PART, (m + r) * PN + j);
                }
            }
            tile_regs_release();
            matmul_block_init(CB_IN0, CB_IN1, false, PN, RT, BW);
        }
        cb_pop_front(CB_IN0, MT * BW);
        cb_pop_front(CB_IN1, BW * PNX);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(CB_PART, OUT_TILES);

    // fp32 partials -> the output format
    cb_wait_front(CB_PART, OUT_TILES);
    cb_reserve_back(CB_OUT, OUT_TILES);
    reconfig_data_format_srca(CB_PART);
    pack_reconfig_data_format(CB_OUT);
    copy_init(CB_PART);
    for (uint32_t t = 0; t < OUT_TILES; ++t) {
        tile_regs_acquire();
        copy_tile(CB_PART, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
    }
    cb_push_back(CB_OUT, OUT_TILES);
    cb_pop_front(CB_PART, OUT_TILES);
}
