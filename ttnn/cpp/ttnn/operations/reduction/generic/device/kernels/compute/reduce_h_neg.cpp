// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reduce H with negation, ported to Metal 2.0.

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif
    constexpr uint32_t row_chunk = compute_kernel_lib::DEST_AUTO_LIMIT;

    DataflowBuffer cb_input_obj(dfb::input);
    DataflowBuffer cb_scaler_obj(dfb::scaler);
    DataflowBuffer cb_output_obj(dfb::output);
    DataflowBuffer cb_acc_obj(dfb::acc);
    DataflowBuffer cb_ineg_obj(dfb::ineg);

    compute_kernel_hw_startup(dfb::input, dfb::scaler, dfb::output);
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

                reconfig_data_format_srca(dfb::input);
                copy_tile_init(dfb::input);
                negative_tile_init();
                // Partial chunk (ntiles < row_chunk): the input DFB depth matches row_chunk, but only consume ntiles
                // tiles. Indexed reads plus a bulk pop of ntiles do not advance the DFB head during reads, leaving
                // trailing slots effectively stale; the next pass can index into those offsets and read stale L1 data.
                for (uint32_t i = 0; i < ntiles; ++i) {
                    // Read from index 0 and pop_front(1) per tile to keep the DFB head in sync and avoid stale data.
                    copy_tile(dfb::input, 0, i);
                    cb_input_obj.pop_front(1);
                    negative_tile(i);
                }

                tile_regs_commit();
                cb_ineg_obj.reserve_back(ntiles);
                tile_regs_wait();
                pack_reconfig_data_format(dfb::ineg);
                for (uint32_t i = 0; i < ntiles; ++i) {
                    pack_tile(i, dfb::ineg);
                }
                tile_regs_release();
                cb_ineg_obj.push_back(ntiles);

                tile_regs_acquire();

                if (ht > 0) {
                    cb_acc_obj.wait_front(ntiles);
                }

                cb_ineg_obj.wait_front(ntiles);

                if (ht > 0) {
                    reconfig_data_format_srca(dfb::acc);
                    copy_tile_init(dfb::acc);
                    for (uint32_t i = 0; i < ntiles; ++i) {
                        copy_tile(dfb::acc, i, i);
                    }
                }
                reduce_init<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, dfb::acc);
                pack_reconfig_data_format(dfb::acc);
                for (uint32_t i = 0; i < ntiles; ++i) {
                    reduce_tile<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, i, 0, i);
                }
                reduce_uninit(dfb::ineg);
                tile_regs_commit();
                cb_ineg_obj.pop_front(ntiles);

                if (ht > 0) {
                    cb_acc_obj.pop_front(ntiles);
                }
                cb_acc_obj.reserve_back(ntiles);
                tile_regs_wait();
                for (uint32_t i = 0; i < ntiles; ++i) {
                    pack_tile(i, dfb::acc);
                }
                tile_regs_release();
                cb_acc_obj.push_back(ntiles);
            }

            tile_regs_acquire();

            cb_acc_obj.wait_front(ntiles);

            reconfig_data_format_srca(dfb::acc);
            copy_tile_init(dfb::acc);
            for (uint32_t i = 0; i < ntiles; ++i) {
                copy_tile(dfb::acc, i, i);
            }
            negative_tile_init();
            for (uint32_t i = 0; i < ntiles; ++i) {
                negative_tile(i);
            }

#ifdef REDUCE_POST_MUL
            // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
            // with scaler=1.0 and then applies the user scalar via mul_unary_tile (SFPU) on each
            // output DEST register.
            binop_with_scalar_tile_init();
            for (uint32_t i = 0; i < ntiles; ++i) {
                mul_unary_tile(i, post_mul_scaler_bits);
            }
#endif

            tile_regs_commit();
            cb_acc_obj.pop_front(ntiles);
            cb_output_obj.reserve_back(ntiles);
            tile_regs_wait();
            pack_reconfig_data_format(dfb::output);
            for (uint32_t i = 0; i < ntiles; ++i) {
                pack_tile(i, dfb::output);
            }
            tile_regs_release();
            cb_output_obj.push_back(ntiles);
        }
    }
}
