// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "api/debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    uint32_t row_chunk = get_compile_time_arg_val(3);

    // Circular buffers:
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_acc = tt::CBIndex::c_4;
    constexpr uint32_t cb_ineg = tt::CBIndex::c_5;

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);
    reduce_init(cb_input, cb_scaler, cb_output);

    DPRINT << "Starting reduce_h_neg kernel" << ENDL();

    cb_wait_front(cb_scaler, 1);  // scaler tile from the reader

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
            DPRINT << "wt=" << wt << ", chunk_end=" << chunk_end << ", ntiles = " << ntiles << ENDL();

            // reduction for one chunk
            // accumulation of Ht results in separate DST indexes
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                reduce_dst_idx = 0;
                DPRINT << "ht=" << ht << ENDL();
                DPRINT << "cb_wait_front(cb_input, ntiles)" << ENDL();
                acquire_dst();
                cb_wait_front(cb_input, ntiles);

                unary_op_init_common(cb_input, cb_ineg);

                DPRINT << "copy_tile_init(cb_input)" << ENDL();
                copy_tile_init(cb_input);
                DPRINT << "negative_tile_init()" << ENDL();
                negative_tile_init();
                for (uint32_t i = 0; i < ntiles; ++i) {
                    DPRINT << "f1: i=" << i << ", ntiles=" << ntiles << ENDL();
                    DPRINT << "copy_tile(cb_input, i, i)" << ENDL();
                    copy_tile(cb_input, i, i);
                    DPRINT << "negative_tile(i)" << ENDL();
                    negative_tile(i);
                }
                DPRINT << "tile_regs_wait()" << ENDL();

                DPRINT << "cb_pop_front(cb_input, chunk_end - wt)" << ENDL();
                cb_pop_front(cb_input, chunk_end - wt);
                DPRINT << "cb_reserve_back(cb_ineg, ntiles)" << ENDL();
                cb_reserve_back(cb_ineg, ntiles);

                for (uint32_t i = 0; i < ntiles; ++i) {
                    DPRINT << "pack_tile(i, cb_ineg): i=" << i << ", ntiles=" << ntiles << ENDL();
                    pack_tile(i, cb_ineg);
                }
                DPRINT << "cb_push_back(cb_ineg, ntiles)" << ENDL();
                cb_push_back(cb_ineg, ntiles);
                release_dst();

                acquire_dst();

                if (ht > 0) {
                    DPRINT << "cb_wait_front(cb_acc, ntiles)" << ENDL();
                    cb_wait_front(cb_acc, ntiles);
                }
                DPRINT << "cb_wait_front(cb_ineg, ntiles)" << ENDL();

                cb_wait_front(cb_ineg, ntiles);

                if (ht > 0) {
                    DPRINT << "copy_tile_init(cb_acc)" << ENDL();
                    copy_tile_init(cb_acc);
                    for (uint32_t i = 0; i < ntiles; ++i) {
                        copy_tile(cb_acc, i, i);
                    }
                }
                DPRINT << "reduce_init(cb_ineg, cb_scaler, cb_acc)" << ENDL();
                reduce_init(cb_ineg, cb_scaler, cb_acc);
                for (uint32_t i = 0; i < ntiles; ++i) {
                    DPRINT << "reduce_tile(cb_ineg, cb_scalar: i=" << i << ", ntiles=" << ntiles << ENDL();
                    reduce_tile(cb_ineg, cb_scaler, i, 0, i);
                }
                reduce_uninit();
                cb_pop_front(cb_ineg, ntiles);

                if (ht > 0) {
                    cb_pop_front(cb_acc, ntiles);
                }
                cb_reserve_back(cb_acc, ntiles);

                for (uint32_t i = 0; i < ntiles; ++i) {
                    DPRINT << "pack_tile(i, cb_acc): i=" << i << ", ntiles=" << ntiles << ENDL();
                    pack_tile(i, cb_acc);
                }
                DPRINT << "cb_push_back(cb_acc, ntiles)" << ENDL();
                cb_push_back(cb_acc, ntiles);
                release_dst();
            }

            acquire_dst();

            DPRINT << "cb_wait_front(cb_acc, ntiles)" << ENDL();
            cb_wait_front(cb_acc, ntiles);

            unary_op_init_common(cb_acc, cb_output);

            copy_tile_init(cb_acc);
            for (uint32_t i = 0; i < ntiles; ++i) {
                copy_tile(cb_acc, i, i);
            }
            negative_tile_init();
            for (uint32_t i = 0; i < ntiles; ++i) {
                negative_tile(i);
            }

            DPRINT << "cb_pop_front(cb_acc, ntiles)" << ENDL();
            cb_pop_front(cb_acc, ntiles);
            cb_reserve_back(cb_output, ntiles);

            for (uint32_t i = 0; i < ntiles; ++i) {
                DPRINT << "pack_tile(i, cb_output): i=" << i << ", ntiles=" << ntiles << ENDL();
                pack_tile(i, cb_output);
            }
            DPRINT << "cb_push_back(cb_output, ntiles)" << ENDL();
            cb_push_back(cb_output, ntiles);
            release_dst();
            DPRINT << "cb_push_back(cb_output, ntiles)" << ENDL();
        }
    }
}
}  // namespace NAMESPACE
