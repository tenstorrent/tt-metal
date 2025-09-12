// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/tile_move_copy.h"

#pragma once

/*

Kahan Summation Algorithm

sum = 0
c = 0
for i in range:
  y = input[i]   # copy_tile
  y -= c         # sub_binary_tile  y < c
  t = sum        # copy_dest_values
  t += y         # add_binary_tile  t < y
  #c = (t - sum) - y
  c = t
  c -= sum       # sub_binary_tile c < sum
  c -= y         # sub_binary_tile c < y - conflict with above, needs llk update
  sum = t        # copy_test_values
return sum

The general reduction code in the kernels has a scaler applied at each step. That

Following function is the portion in the loop.
Needs to be called
*/

void kahan_iterative_sum(uint32_t in_cb_id, uint32_t in_cb_tile_index) {
    uint32_t sum_index = 0;
    uint32_t c_index = 1;
    uint32_t y_index = 2;
    uint32_t t_index = 3;

    copy_tile_init(in_cb_id);
    copy_tile(in_cb_id, in_cb_tile_index, y_index);  // y = input[i]
    sub_binary_tile_init();
    sub_binary_tile(y_index, c_index, y_index);  // y -= c
    copy_dest_values_init();
    copy_dest_values(y_index, t_index);  // t = sum
    add_binary_tile_init();
    add_binary_tile(t_index, y_index, t_index);  // t += y
    copy_dest_values_init();
    copy_dest_values(t_index, c_index);  // c = t
    sub_binary_tile_init();
    sub_binary_tile(c_index, sum_index, c_index);  // c -= sum
    sub_binary_tile(c_index, y_index, c_index);    // c -= y
    copy_dest_values_init();
    copy_dest_values(t_index, sum_index);  // sum  = t
}

/*
Final reduction
// Need to apply scaler separately
void kahan_final_multiply(uint32_t in_cb_id, uint32_t in_cb_tile_index) {
    uint32_t sum_index = 0;
    uint32_t t_index = 3;
    copy_tile_init(in_cb_id);
    copy_tile(in_cb_id, in_cb_tile_index, t_index); // dst0 = cb
    mul_binary_tile_init();
    mul_binary_tile(sum_index, t_index, sum_index); // sum /= dst0
}
*/
