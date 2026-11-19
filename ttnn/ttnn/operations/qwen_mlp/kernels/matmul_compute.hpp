// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "named_args.hpp"
#include "swiglu_sfpu.hpp"

ALWI void matmul_subblock(const MatmulShape& s, uint32_t in0_offset, uint32_t in1_offset) {
    uint32_t in0_index = in0_offset;
    uint32_t in1_index = in1_offset;
    for (uint32_t k = 0; k < s.K_block; k++) {
        matmul_block(s.cb_in0, s.cb_in1, in0_index, in1_index, 0, false, s.sub_w, s.sub_h, s.K_block);
        in0_index++;
        in1_index += s.N_block;
    }
}

ALWI void matmul_block_to_partial(const MatmulShape& s) {
    for (uint32_t m = 0; m < s.M_block; m += s.sub_h) {
        for (uint32_t n = 0; n < s.N_block; n += s.sub_w) {
            tile_regs_acquire();
            matmul_subblock(s, m * s.K_block, n);
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst = 0;
            for (uint32_t h = 0; h < s.sub_h; h++) {
                for (uint32_t w = 0; w < s.sub_w; w++) {
                    pack_tile<true>(dst++, s.cb_partial, (m + h) * s.N_block + n + w);
                }
            }
            tile_regs_release();
        }
    }
}

ALWI void add_partials_to_dst(const MatmulShape& s, uint32_t m, uint32_t n) {
    reconfig_data_format_srca(s.cb_in1, s.cb_partial);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(s.cb_partial);
    uint32_t dst = 0;
    for (uint32_t h = 0; h < s.sub_h; h++) {
        for (uint32_t w = 0; w < s.sub_w; w++) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                s.cb_partial, (m + h) * s.N_block + n + w, dst++);
        }
    }
    reconfig_data_format_srca(s.cb_partial, s.cb_in1);
    matmul_block_init(s.cb_in0, s.cb_in1, false, s.sub_w, s.sub_h, s.K_block);
}

ALWI void matmul_block_to_out(const MatmulShape& s, bool add_partials) {
    const uint32_t out_row_block = s.swiglu ? s.N_block / 2 : s.N_block;
    for (uint32_t m = 0; m < s.M_block; m += s.sub_h) {
        for (uint32_t n = 0; n < s.N_block; n += s.sub_w) {
            tile_regs_acquire();
            matmul_subblock(s, m * s.K_block, n);
            if (add_partials) {
                add_partials_to_dst(s, m, n);
            }
            tile_regs_commit();
            if (s.swiglu) {
                PACK(TTI_SEMWAIT(
                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                    semaphore::t6_sem(semaphore::MATH_PACK),
                    p_stall::STALL_ON_ZERO));
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                for (uint32_t h = 0; h < s.sub_h; h++) {
                    for (uint32_t w = 0; w < s.sub_w; w += 2) {
                        const uint32_t gate = h * s.sub_w + w;
                        PACK((swiglu_tile(gate, gate + 1, gate)));
                    }
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                for (uint32_t h = 0; h < s.sub_h; h++) {
                    for (uint32_t w = 0; w < s.sub_w; w += 2) {
                        pack_tile<true>(h * s.sub_w + w, s.cb_out, (m + h) * out_row_block + ((n + w) >> 1));
                    }
                }
            } else {
                tile_regs_wait();
                uint32_t dst = 0;
                for (uint32_t h = 0; h < s.sub_h; h++) {
                    for (uint32_t w = 0; w < s.sub_w; w++) {
                        pack_tile<true>(dst++, s.cb_out, (m + h) * out_row_block + n + w);
                    }
                }
            }
            tile_regs_release();
        }
    }
}

// hook(step) runs between the K blocks and returns whether it did work that needs the matmul state restored.
template <typename Hook>
ALWI void matmul_phase(const MatmulShape& s, Hook hook) {
    CircularBuffer in0(s.cb_in0);
    CircularBuffer in1(s.cb_in1);
    CircularBuffer out(s.cb_out);
    CircularBuffer partial(s.cb_partial);
    const uint32_t in0_block_tiles = s.M_block * s.K_block;
    const uint32_t in1_block_tiles = s.in1_slot_tiles;
    const uint32_t partial_tiles = s.partial_slot_tiles;
    const uint32_t out_tiles = s.M_block * s.N_block / (s.swiglu ? 2 : 1);

    if (s.swiglu) {
        PACK((swiglu_tile_init()));
    }
    matmul_init(s.cb_in0, s.cb_in1);
    uint32_t step = 0;

    for (uint32_t m_block = 0; m_block < s.M_blocks; m_block++) {
        bool reuse_in0 = false;
        for (uint32_t n_block = 0; n_block < s.N_blocks; n_block++) {
            reconfig_data_format(s.cb_in1, s.cb_in0);
            pack_reconfig_data_format(s.cb_partial);
            matmul_block_init(s.cb_in0, s.cb_in1, false, s.sub_w, s.sub_h, s.K_block);
            partial.reserve_back(partial_tiles);
            for (uint32_t k_block = 0; k_block < s.K_blocks; k_block++) {
                in0.wait_front(in0_block_tiles);
                in1.wait_front(in1_block_tiles);
                if (k_block == s.K_blocks - 1) {
                    if (s.K_blocks > 1) {
                        partial.push_back(partial_tiles);
                        partial.wait_front(partial_tiles);
                    }
                    pack_reconfig_l1_acc(0);
                    pack_reconfig_data_format(s.cb_out);
                    out.reserve_back(out_tiles);
                    matmul_block_to_out(s, s.K_blocks > 1);
                    out.push_back(out_tiles);
                    if (s.K_blocks > 1) {
                        partial.pop_front(partial_tiles);
                    }
                } else {
                    matmul_block_to_partial(s);
                }
                reuse_in0 = k_block == s.K_blocks - 1 && n_block < s.N_blocks - 1;
                if (!reuse_in0) {
                    in0.pop_front(in0_block_tiles);
                }
                in1.pop_front(in1_block_tiles);
                if (k_block == 0) {
                    pack_reconfig_l1_acc(1);
                }
                if (hook(step++) && k_block < s.K_blocks - 1) {
                    reconfig_data_format(s.cb_in1, s.cb_in0);
                    pack_reconfig_data_format(s.cb_partial);
                    matmul_block_init(s.cb_in0, s.cb_in1, false, s.sub_w, s.sub_h, s.K_block);
                    pack_reconfig_l1_acc(1);
                }
            }
            pack_reconfig_l1_acc(0);
        }
    }
}
