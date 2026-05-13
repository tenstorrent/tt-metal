// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 compute kernel for the single-core HW reduction primitive *with negation*.
//
// Migration notes (mirrors reduce_w_neg.cpp):
//   - Compile-time arguments are bound by name (args::Ht, args::Wt, args::NC,
//     args::post_mul_scaler_bits).
//   - DataflowBuffers are bound by name. The accumulator and intermediate-negation
//     buffers are produced AND consumed by this same kernel, so each has two host-side
//     bindings (*_w PRODUCER, *_r CONSUMER) with distinct local_accessor_names; on
//     Gen1 they resolve to the same underlying CB.

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/dataflow_buffer.h"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t NC = get_arg(args::NC);
#ifdef REDUCE_POST_MUL
    // Packed fp32 user scalar applied via mul_unary_tile after the reduce+negate finishes.
    constexpr uint32_t post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
#endif

    experimental::DataflowBuffer dfb_input(dfb::input);
    experimental::DataflowBuffer dfb_scaler(dfb::scaler);
    experimental::DataflowBuffer dfb_output(dfb::output);
    experimental::DataflowBuffer dfb_acc_writer(dfb::acc_w);
    experimental::DataflowBuffer dfb_acc_reader(dfb::acc_r);
    experimental::DataflowBuffer dfb_ineg_writer(dfb::ineg_w);
    experimental::DataflowBuffer dfb_ineg_reader(dfb::ineg_r);

    // LLK calls still take raw buffer ids.
    const uint32_t input_id = dfb_input.get_id();
    const uint32_t scaler_id = dfb_scaler.get_id();
    const uint32_t output_id = dfb_output.get_id();
    const uint32_t acc_id = dfb_acc_writer.get_id();    // == dfb_acc_reader.get_id()
    const uint32_t ineg_id = dfb_ineg_writer.get_id();  // == dfb_ineg_reader.get_id()

    compute_kernel_hw_startup(input_id, scaler_id, output_id);
    dfb_scaler.wait_front(1);  // scaler tile from the reader

    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                acquire_dst();
                dfb_input.wait_front(onetile);
                copy_tile_init(input_id);
                copy_tile(input_id, 0, reduce_dst_idx);
                negative_tile_init();
                negative_tile(reduce_dst_idx);
                dfb_input.pop_front(onetile);
                dfb_ineg_writer.reserve_back(onetile);
                pack_tile(reduce_dst_idx, ineg_id);
                dfb_ineg_writer.push_back(onetile);
                release_dst();

                acquire_dst();
                if (wt > 0 || ht > 0) {
                    dfb_acc_reader.wait_front(onetile);
                    copy_tile_init(acc_id);
                    copy_tile(acc_id, 0, reduce_dst_idx);
                }

                dfb_ineg_reader.wait_front(onetile);
                reduce_init<REDUCE_OP, REDUCE_DIM>(ineg_id, scaler_id, acc_id);
                reduce_tile<REDUCE_OP, REDUCE_DIM>(ineg_id, scaler_id, 0, 0, reduce_dst_idx);
                reduce_uninit();
                dfb_ineg_reader.pop_front(onetile);
                if (wt > 0 || ht > 0) {
                    dfb_acc_reader.pop_front(onetile);
                }
                dfb_acc_writer.reserve_back(onetile);
                pack_tile(reduce_dst_idx, acc_id);
                dfb_acc_writer.push_back(onetile);
                release_dst();
            }  // wt
        }  // ht

        acquire_dst();
        dfb_acc_reader.wait_front(onetile);
        copy_tile_init(acc_id);
        copy_tile(acc_id, 0, reduce_dst_idx);
        negative_tile_init();
        negative_tile(reduce_dst_idx);
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN, so the host requests reduction
        // with scaler=1.0 and then applies the user scalar via mul_unary_tile (SFPU) on each
        // output DEST register.
        binop_with_scalar_tile_init();
        mul_unary_tile(reduce_dst_idx, post_mul_scaler_bits);
#endif
        dfb_acc_reader.pop_front(onetile);
        dfb_output.reserve_back(onetile);
        pack_tile(reduce_dst_idx, output_id);
        dfb_output.push_back(onetile);
        release_dst();
    }  // nc
}
