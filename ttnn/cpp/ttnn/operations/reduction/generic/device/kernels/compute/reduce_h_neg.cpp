// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 compute kernel for the multi-core H reduction primitive *with negation*.
//
// Migration notes (mirrors reduce_h.cpp):
//   - Wt is bound as a per-node runtime arg because the H factory's split_work_to_cores
//     produces two core groups with different per-core column counts.
//   - Ht, NC and post_mul_scaler_bits are compile-time.
//   - Local DataflowBuffers are bound by name (dfb::input, dfb::scaler, dfb::output,
//     dfb::acc_w/dfb::acc_r, dfb::ineg_w/dfb::ineg_r). The accumulator and
//     intermediate-negation buffers are produced AND consumed by this same kernel,
//     so each has two host-side bindings (*_w PRODUCER, *_r CONSUMER) with distinct
//     local_accessor_names; on Gen1 they resolve to the same underlying CB.

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    const uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
    constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif
    constexpr uint32_t row_chunk = compute_kernel_lib::DEST_AUTO_LIMIT;

    experimental::DataflowBuffer input_buf(dfb::input);
    experimental::DataflowBuffer scaler_buf(dfb::scaler);
    experimental::DataflowBuffer output_buf(dfb::output);
    experimental::DataflowBuffer acc_writer(dfb::acc_w);
    experimental::DataflowBuffer acc_reader(dfb::acc_r);
    experimental::DataflowBuffer ineg_writer(dfb::ineg_w);
    experimental::DataflowBuffer ineg_reader(dfb::ineg_r);

    // LLK calls take raw buffer ids.
    const uint32_t input_id = input_buf.get_id();
    const uint32_t scaler_id = scaler_buf.get_id();
    const uint32_t output_id = output_buf.get_id();
    const uint32_t acc_id = acc_writer.get_id();    // == acc_reader.get_id()
    const uint32_t ineg_id = ineg_writer.get_id();  // == ineg_reader.get_id()

    compute_kernel_hw_startup(input_id, scaler_id, output_id);
    scaler_buf.wait_front(1);  // scaler tile from the reader

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
                input_buf.wait_front(ntiles);

                reconfig_data_format_srca(input_id);
                copy_tile_init(input_id);
                negative_tile_init();
                // Partial chunk (ntiles < row_chunk): the input CB depth matches row_chunk, but only consume ntiles
                // tiles. Indexed reads plus a bulk pop of ntiles do not advance the CB head during reads, leaving
                // trailing slots effectively stale; the next pass can index into those offsets and read stale L1 data.
                for (uint32_t i = 0; i < ntiles; ++i) {
                    // Read from index 0 and pop_front(1) per tile to keep the CB head in sync and avoid stale data.
                    copy_tile(input_id, 0, i);
                    input_buf.pop_front(1);
                    negative_tile(i);
                }

                tile_regs_commit();
                ineg_writer.reserve_back(ntiles);
                tile_regs_wait();
                pack_reconfig_data_format(ineg_id);
                for (uint32_t i = 0; i < ntiles; ++i) {
                    pack_tile(i, ineg_id);
                }
                tile_regs_release();
                ineg_writer.push_back(ntiles);

                tile_regs_acquire();

                if (ht > 0) {
                    acc_reader.wait_front(ntiles);
                }

                ineg_reader.wait_front(ntiles);

                if (ht > 0) {
                    reconfig_data_format_srca(acc_id);
                    copy_tile_init(acc_id);
                    for (uint32_t i = 0; i < ntiles; ++i) {
                        copy_tile(acc_id, i, i);
                    }
                }
                reduce_init<REDUCE_OP, REDUCE_DIM>(ineg_id, scaler_id, acc_id);
                pack_reconfig_data_format(acc_id);
                for (uint32_t i = 0; i < ntiles; ++i) {
                    reduce_tile<REDUCE_OP, REDUCE_DIM>(ineg_id, scaler_id, i, 0, i);
                }
                reduce_uninit(ineg_id);
                tile_regs_commit();
                ineg_reader.pop_front(ntiles);

                if (ht > 0) {
                    acc_reader.pop_front(ntiles);
                }
                acc_writer.reserve_back(ntiles);
                tile_regs_wait();
                for (uint32_t i = 0; i < ntiles; ++i) {
                    pack_tile(i, acc_id);
                }
                tile_regs_release();
                acc_writer.push_back(ntiles);
            }

            tile_regs_acquire();

            acc_reader.wait_front(ntiles);

            reconfig_data_format_srca(acc_id);
            copy_tile_init(acc_id);
            for (uint32_t i = 0; i < ntiles; ++i) {
                copy_tile(acc_id, i, i);
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
            acc_reader.pop_front(ntiles);
            output_buf.reserve_back(ntiles);
            tile_regs_wait();
            pack_reconfig_data_format(output_id);
            for (uint32_t i = 0; i < ntiles; ++i) {
                pack_tile(i, output_id);
            }
            tile_regs_release();
            output_buf.push_back(ntiles);
        }
    }
}
