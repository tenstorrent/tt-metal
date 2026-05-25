// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sharded writer kernel for iterative_topk.
// Input and output data are both in L1 via globally-allocated CBs.
// For each row in the local shard, finds the top-k values by repeatedly
// scanning for the maximum, masking each found max with -inf.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_values = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_indices = get_compile_time_arg_val(2);
    constexpr uint32_t width = get_compile_time_arg_val(3);
    constexpr uint32_t k = get_compile_time_arg_val(4);
    constexpr uint32_t num_rows = get_compile_time_arg_val(5);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t output_values_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t output_indices_page_size = get_compile_time_arg_val(8);

    constexpr uint32_t NEG_INF_U32 = 0xFF800000u;

    // Wait for reader to activate input pages
    cb_wait_front(cb_input, num_rows);

    uint32_t input_base = get_read_ptr(cb_input);
    uint32_t values_base = get_write_ptr(cb_out_values);
    uint32_t indices_base = get_write_ptr(cb_out_indices);

    for (uint32_t row = 0; row < num_rows; row++) {
        volatile tt_l1_ptr uint32_t* data_u32 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_base + row * input_page_size);
        volatile tt_l1_ptr float* data =
            reinterpret_cast<volatile tt_l1_ptr float*>(input_base + row * input_page_size);

        volatile tt_l1_ptr uint32_t* out_vals_u32 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(values_base + row * output_values_page_size);
        volatile tt_l1_ptr uint32_t* out_idxs =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(indices_base + row * output_indices_page_size);

        for (uint32_t ki = 0; ki < k; ki++) {
            float max_val = data[0];
            uint32_t max_idx = 0;

            for (uint32_t j = 1; j < width; j++) {
                float val = data[j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }

            out_vals_u32[ki] = data_u32[max_idx];
            out_idxs[ki] = max_idx;

            data_u32[max_idx] = NEG_INF_U32;
        }
    }

    // Mark output pages as ready
    cb_push_back(cb_out_values, num_rows);
    cb_push_back(cb_out_indices, num_rows);
}
