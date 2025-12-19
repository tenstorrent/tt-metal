// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t row_chunk = compute_kernel_lib::DEST_AUTO_LIMIT;

    // Circular buffers:
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_acc = tt::CBIndex::c_4;
    constexpr uint32_t cb_ineg = tt::CBIndex::c_5;

    experimental::CircularBuffer cb_input_obj(cb_input);
    experimental::CircularBuffer cb_scaler_obj(cb_scaler);
    experimental::CircularBuffer cb_output_obj(cb_output);
    experimental::CircularBuffer cb_acc_obj(cb_acc);
    experimental::CircularBuffer cb_ineg_obj(cb_ineg);

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);
    cb_scaler_obj.wait_front(1);  // scaler tile from the reader

    constexpr int onetile = 1;

    // tiles are expected to come in the N C W_skip H W_chunk order
    // W_skip(chunk size) represents the number of tile columns whose reduction will be intertwined
    // H W_chunk represent tiles of the chunk in row major order
    // each column in the chunk will have its intermediate result in a separate tile of DST
    // chunk size is calculated based on the number of available tiles in DST
    // exmpl. Ht = 3; Wt = 4; row_chunk = 2;
    //        tile order (H, W):
    //        1. chunk: (0, 0); (0, 1); (1, 0); (1, 1); (2, 0); (2, 1);
    //        2. chunk: (0, 2); (0, 3); (1, 2); (1, 3); (2, 2); (2, 3);
    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; wt += row_chunk) {
            uint32_t chunk_end = std::min(wt + row_chunk, Wt);
            int reduce_dst_idx = 0;
            uint32_t ntiles = chunk_end - wt;

            // reduction for one chunk
            // accumulation of Ht results in separate DST indexes
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                reduce_dst_idx = 0;
                tile_regs_acquire();
                cb_input_obj.wait_front(ntiles);

                reconfig_data_format_srca(cb_input);
                copy_tile_init(cb_input);
                negative_tile_init();
                for (uint32_t i = 0; i < ntiles; ++i) {
                    copy_tile(cb_input, i, i);
                    negative_tile(i);
                }

                tile_regs_commit();
                cb_input_obj.pop_front(chunk_end - wt);
                cb_ineg_obj.reserve_back(ntiles);
                tile_regs_wait();
                pack_reconfig_data_format(cb_ineg);
                for (uint32_t i = 0; i < ntiles; ++i) {
                    pack_tile(i, cb_ineg);
                }
                tile_regs_release();
                cb_ineg_obj.push_back(ntiles);

                tile_regs_acquire();

                if (ht > 0) {
                    cb_acc_obj.wait_front(ntiles);
                }

                cb_ineg_obj.wait_front(ntiles);

                if (ht > 0) {
                    reconfig_data_format_srca(cb_acc);
                    copy_tile_init(cb_acc);
                    for (uint32_t i = 0; i < ntiles; ++i) {
                        copy_tile(cb_acc, i, i);
                    }
                }
                reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc);
                pack_reconfig_data_format(cb_acc);
                for (uint32_t i = 0; i < ntiles; ++i) {
                    reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, i, 0, i);
                }
                reduce_uninit(cb_ineg);
                tile_regs_commit();
                cb_ineg_obj.pop_front(ntiles);

                if (ht > 0) {
                    cb_acc_obj.pop_front(ntiles);
                }
                cb_acc_obj.reserve_back(ntiles);
                tile_regs_wait();
                for (uint32_t i = 0; i < ntiles; ++i) {
                    pack_tile(i, cb_acc);
                }
                tile_regs_release();
                cb_acc_obj.push_back(ntiles);
            }

            tile_regs_acquire();

            cb_acc_obj.wait_front(ntiles);

            reconfig_data_format_srca(cb_acc);
            copy_tile_init(cb_acc);
            for (uint32_t i = 0; i < ntiles; ++i) {
                copy_tile(cb_acc, i, i);
            }
            negative_tile_init();
            for (uint32_t i = 0; i < ntiles; ++i) {
                negative_tile(i);
            }

            tile_regs_commit();
            cb_acc_obj.pop_front(ntiles);
            cb_output_obj.reserve_back(ntiles);
            tile_regs_wait();
            pack_reconfig_data_format(cb_output);
            for (uint32_t i = 0; i < ntiles; ++i) {
                pack_tile(i, cb_output);
            }
            tile_regs_release();
            cb_output_obj.push_back(ntiles);
        }
    }
}
