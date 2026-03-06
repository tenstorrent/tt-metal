// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for rotary_embedding_llama in HC-transpose decode mode.
//
// For each (head, batch_tile) pair assigned to this core, performs:
//   rotated  = x @ trans_mat          (matmul, tile-wise over head_dim)
//   sin_term = rotated * sin           (element-wise)
//   cos_term = x * cos                 (element-wise)
//   out      = cos_term + sin_term     (element-wise)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    uint32_t argrt = 0;
    uint32_t head_start = get_arg_val<uint32_t>(argrt++);
    uint32_t head_end = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);  // head_dim_t

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, out_cb);

    // Wait for transformation matrix once — it is reused for every (head, batch_tile).
    cb_wait_front(trans_mat_cb, onetile);

    for (uint32_t h = head_start; h < head_end; ++h) {
        for (uint32_t bt = batch_t_start; bt < batch_t_end; ++bt) {
            // Wait for input, cos, sin tiles pushed by the reader.
            cb_wait_front(in_cb, Wt);
            cb_wait_front(cos_cb, Wt);
            cb_wait_front(sin_cb, Wt);

            cb_reserve_back(rotated_in_interm_cb, Wt);
            cb_reserve_back(sin_interm_cb, Wt);
            cb_reserve_back(cos_interm_cb, Wt);
            cb_reserve_back(out_cb, Wt);

            // rotated = x @ trans_mat  (each of Wt tiles independently)
            mm_init_short(in_cb, trans_mat_cb);
            ACQ();
            for (uint32_t j = 0; j < Wt; ++j) {
                matmul_tiles(in_cb, trans_mat_cb, j, 0, j);
                pack_tile(j, rotated_in_interm_cb, j);
            }
            REL();
            cb_push_back(rotated_in_interm_cb, Wt);
            cb_wait_front(rotated_in_interm_cb, Wt);

            // sin_term = rotated * sin
            mul_tiles_init(rotated_in_interm_cb, sin_cb);
            ACQ();
            for (uint32_t j = 0; j < Wt; ++j) {
                mul_tiles(rotated_in_interm_cb, sin_cb, j, j, j);
                pack_tile(j, sin_interm_cb, j);
            }
            REL();
            cb_push_back(sin_interm_cb, Wt);
            cb_pop_front(rotated_in_interm_cb, Wt);

            // cos_term = x * cos
            ACQ();
            for (uint32_t j = 0; j < Wt; ++j) {
                mul_tiles(in_cb, cos_cb, j, j, j);
                pack_tile(j, cos_interm_cb, j);
            }
            REL();
            cb_push_back(cos_interm_cb, Wt);
            cb_pop_front(in_cb, Wt);
            cb_pop_front(cos_cb, Wt);
            cb_pop_front(sin_cb, Wt);

            // out = cos_term + sin_term
            cb_wait_front(sin_interm_cb, Wt);
            cb_wait_front(cos_interm_cb, Wt);
            add_tiles_init(cos_interm_cb, sin_interm_cb);
            ACQ();
            for (uint32_t j = 0; j < Wt; ++j) {
                add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
                pack_tile(j, out_cb, j);
            }
            REL();
            cb_push_back(out_cb, Wt);
            cb_pop_front(sin_interm_cb, Wt);
            cb_pop_front(cos_interm_cb, Wt);
        }
    }

    cb_pop_front(trans_mat_cb, onetile);
}
