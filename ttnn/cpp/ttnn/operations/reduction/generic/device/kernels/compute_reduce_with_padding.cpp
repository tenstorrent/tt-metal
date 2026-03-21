// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
    uint32_t reduce_op = get_arg_val<uint32_t>(2);
    uint32_t reduce_dim = get_arg_val<uint32_t>(3);
    uint32_t scaler = get_arg_val<uint32_t>(4);
    uint32_t orig_h = get_arg_val<uint32_t>(5);
    uint32_t orig_w = get_arg_val<uint32_t>(6);
    uint32_t padded_h = get_arg_val<uint32_t>(7);
    uint32_t padded_w = get_arg_val<uint32_t>(8);
    uint32_t pad_value = get_arg_val<uint32_t>(9);

    constexpr uint32_t onetile = 1;
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = get_arg_val<uint32_t>(10),
        .page_size = get_arg_val<uint32_t>(11),
        .data_format = get_arg_val<uint32_t>(12)
    };

    const InterleavedAddrGenFast<false> d = {
        .bank_base_address = get_arg_val<uint32_t>(13),
        .page_size = get_arg_val<uint32_t>(14),
        .data_format = get_arg_val<uint32_t>(15)
    };

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    
    bool needs_padding = (orig_h != padded_h) || (orig_w != padded_w);
    bool width_padding = (orig_w != padded_w);
    bool height_padding = (orig_h != padded_h);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            uint32_t tile_id = block * per_core_block_size + i;
            
            cb_wait_front(tt::CB::c_in0, onetile);
            cb_reserve_back(tt::CB::c_out0, onetile);

            if (needs_padding) {
                uint32_t h_idx = tile_id / (padded_w / 32);
                uint32_t w_idx = tile_id % (padded_w / 32);
                uint32_t h_tile = h_idx / 32;
                uint32_t w_tile = w_idx / 32;
                uint32_t h_offset = h_idx % 32;
                uint32_t w_offset = w_idx % 32;

                bool needs_height_pad = height_padding && (h_tile * 32 + h_offset >= orig_h);
                bool needs_width_pad = width_padding && (w_tile * 32 + w_offset >= orig_w);

                if (needs_height_pad || needs_width_pad) {
                    acquire_dst(tt::DstMode::Half);
                    
                    copy_tile_to_dst_init_short();
                    copy_tile(tt::CB::c_in0, 0, dst0);
                    
                    if (needs_height_pad && needs_width_pad) {
                        fill_tile_with_val_init_short();
                        fill_tile_with_val(dst0, pad_value);
                    } else if (needs_height_pad) {
                        for (uint32_t row = orig_h % 32; row < 32; ++row) {
                            fill_tile_row_with_val(dst0, row, pad_value);
                        }
                    } else if (needs_width_pad) {
                        for (uint32_t col = orig_w % 32; col < 32; ++col) {
                            fill_tile_col_with_val(dst0, col, pad_value);
                        }
                    }
                    
                    pack_tile(dst0, tt::CB::c_intermed0);
                    release_dst(tt::DstMode::Half);
                    
                    cb_push_back(tt::CB::c_intermed0, onetile);
                    cb_wait_front(tt::CB::c_intermed0, onetile);
                    
                    acquire_dst(tt::DstMode::Half);
                    copy_tile_to_dst_init_short();
                    copy_tile(tt::CB::c_intermed0, 0, dst0);
                    
                } else {
                    acquire_dst(tt::DstMode::Half);
                    copy_tile_to_dst_init_short();
                    copy_tile(tt::CB::c_in0, 0, dst0);
                }
            } else {
                acquire_dst(tt::DstMode::Half);
                copy_tile_to_dst_init_short();
                copy_tile(tt::CB::c_in0, 0, dst0);
            }

            switch (reduce_op) {
                case 0: // sum
                    if (reduce_dim == 0) { // reduce H
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::SUM, REDUCE_DIM::H, dst0, dst1, scaler);
                    } else { // reduce W
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::SUM, REDUCE_DIM::W, dst0, dst1, scaler);
                    }
                    break;
                case 1: // mean
                    if (reduce_dim == 0) {
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::SUM, REDUCE_DIM::H, dst0, dst1, scaler);
                    } else {
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::SUM, REDUCE_DIM::W, dst0, dst1, scaler);
                    }
                    break;
                case 2: // max
                    if (reduce_dim == 0) {
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::MAX, REDUCE_DIM::H, dst0, dst1, scaler);
                    } else {
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::MAX, REDUCE_DIM::W, dst0, dst1, scaler);
                    }
                    break;
                case 3: // min
                    if (reduce_dim == 0) {
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::MIN, REDUCE_DIM::H, dst0, dst1, scaler);
                    } else {
                        reduce_init_delta<false>();
                        reduce_tile(REDUCE_OP::MIN, REDUCE_DIM::W, dst0, dst1, scaler);
                    }
                    break;
            }

            pack_tile(dst1, tt::CB::c_out0);
            release_dst(tt::DstMode::Half);

            cb_pop_front(tt::CB::c_in0, onetile);
            cb_push_back(tt::CB::c_out0, onetile);
            
            if (needs_padding) {
                cb_pop_front(tt::CB::c_intermed0, onetile);
            }
        }
    }
}
}