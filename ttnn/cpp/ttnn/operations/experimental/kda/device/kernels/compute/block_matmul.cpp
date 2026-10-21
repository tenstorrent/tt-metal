// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// Block matmul compute: K blocks accumulate in an FP32 buffer with packer L1 accumulation; the last K block reloads
// that buffer into DST (unpacked straight to DST, exact) and packs the output dtype subblock by subblock.
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t m_block = get_compile_time_arg_val(4);
    constexpr uint32_t n_block = get_compile_time_arg_val(5);
    constexpr uint32_t k_block = get_compile_time_arg_val(6);
    constexpr uint32_t k_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t sub_w = get_compile_time_arg_val(8);  // output subblock: sub_h x sub_w tiles
    constexpr uint32_t sub_h = get_compile_time_arg_val(9);
    constexpr uint32_t out_tiles = m_block * n_block;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_acc);
    // Subblocks one tile high land in the output in order, so the writer can take them as they come.
    constexpr bool stream = sub_h == 1;
    cb_reserve_back(cb_acc, out_tiles);
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
        const bool last = kb + 1 == k_blocks;
        cb_wait_front(cb_a, m_block * k_block);
        cb_wait_front(cb_w, k_block * n_block);
        reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_w);
        matmul_block_init(cb_a, cb_w, false, sub_w, sub_h, k_block);
        if (!last) {
            // The first K block overwrites the accumulator; later ones add to it in L1.
            pack_reconfig_l1_acc(kb == 0 ? 0 : 1);
        } else {
            // The last K block starts from the FP32 partial (unpacked straight to DST, exact) and packs BF16.
            pack_reconfig_l1_acc(0);
            if (k_blocks > 1) {
                cb_push_back(cb_acc, out_tiles);
                cb_wait_front(cb_acc, out_tiles);
            }
            pack_reconfig_data_format(cb_out);
            if constexpr (!stream) {
                cb_reserve_back(cb_out, out_tiles);
            }
        }
        for (uint32_t r0 = 0; r0 < m_block; r0 += sub_h) {
            for (uint32_t c0 = 0; c0 < n_block; c0 += sub_w) {
                tile_regs_acquire();
                if (last && k_blocks > 1) {
                    reconfig_data_format_srca(cb_w, cb_acc);
                    copy_init(cb_acc);
                    for (uint32_t r = 0; r < sub_h; ++r) {
                        for (uint32_t t = 0; t < sub_w; ++t) {
                            copy_tile(cb_acc, (r0 + r) * n_block + c0 + t, r * sub_w + t);
                        }
                    }
                    reconfig_data_format_srca(cb_acc, cb_w);
                    matmul_block_init(cb_a, cb_w, false, sub_w, sub_h, k_block);
                }
                for (uint32_t k = 0; k < k_block; ++k) {
                    matmul_block(cb_a, cb_w, r0 * k_block + k, k * n_block + c0, 0, false, sub_w, sub_h, k_block);
                }
                tile_regs_commit();
                tile_regs_wait();
                if (!last) {
                    for (uint32_t r = 0; r < sub_h; ++r) {
                        for (uint32_t t = 0; t < sub_w; ++t) {
                            pack_tile<true>(r * sub_w + t, cb_acc, (r0 + r) * n_block + c0 + t);
                        }
                    }
                } else if constexpr (stream) {
                    cb_reserve_back(cb_out, sub_w);
                    for (uint32_t t = 0; t < sub_w; ++t) {
                        pack_tile(t, cb_out);
                    }
                    cb_push_back(cb_out, sub_w);
                } else {
                    for (uint32_t r = 0; r < sub_h; ++r) {
                        for (uint32_t t = 0; t < sub_w; ++t) {
                            pack_tile<true>(r * sub_w + t, cb_out, (r0 + r) * n_block + c0 + t);
                        }
                    }
                }
                tile_regs_release();
            }
        }
        cb_pop_front(cb_a, m_block * k_block);
        cb_pop_front(cb_w, k_block * n_block);
    }
    if constexpr (!stream) {
        cb_push_back(cb_out, out_tiles);
    }
    if (k_blocks > 1) {
        cb_pop_front(cb_acc, out_tiles);
    }
}
