// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

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
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                acquire_dst();
                cb_input_obj.wait_front(onetile);
                copy_tile_init(cb_input);
                copy_tile(cb_input, 0, reduce_dst_idx);
                negative_tile_init();
                negative_tile(reduce_dst_idx);
                cb_input_obj.pop_front(onetile);
                cb_ineg_obj.reserve_back(onetile);
                pack_tile(reduce_dst_idx, cb_ineg);
                cb_ineg_obj.push_back(onetile);
                release_dst();

                acquire_dst();
                if (wt > 0 || ht > 0) {
                    cb_acc_obj.wait_front(onetile);
                    copy_tile_init(cb_acc);
                    copy_tile(cb_acc, 0, reduce_dst_idx);
                }

                cb_ineg_obj.wait_front(onetile);
                reduce_init(cb_ineg, cb_scaler, cb_acc);
                reduce_tile(cb_ineg, cb_scaler, 0, 0, reduce_dst_idx);
                reduce_uninit();
                cb_ineg_obj.pop_front(onetile);
                if (wt > 0 || ht > 0) {
                    cb_acc_obj.pop_front(onetile);
                }
                cb_acc_obj.reserve_back(onetile);
                pack_tile(reduce_dst_idx, cb_acc);
                cb_acc_obj.push_back(onetile);
                release_dst();
            }  // wt
        }  // ht

        acquire_dst();
        cb_acc_obj.wait_front(onetile);
        copy_tile_init(cb_acc);
        copy_tile(cb_acc, 0, reduce_dst_idx);
        negative_tile_init();
        negative_tile(reduce_dst_idx);
        cb_acc_obj.pop_front(onetile);
        cb_output_obj.reserve_back(onetile);
        pack_tile(reduce_dst_idx, cb_output);
        cb_output_obj.push_back(onetile);
        release_dst();
    }  // nc
}
