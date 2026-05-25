// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for iterative_topk.
// For each row, finds the top-k values by repeatedly scanning for the maximum,
// recording it, then masking that position with -inf before the next iteration.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_values_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t end_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_values = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_indices = get_compile_time_arg_val(2);
    constexpr uint32_t width = get_compile_time_arg_val(3);
    constexpr uint32_t k = get_compile_time_arg_val(4);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t output_values_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t output_indices_page_size = get_compile_time_arg_val(7);

    constexpr uint32_t values_accessor_offset = 8;
    constexpr auto values_args = TensorAccessorArgs<values_accessor_offset>();
    const auto values_accessor = TensorAccessor(values_args, output_values_addr);

    constexpr uint32_t indices_accessor_offset = values_args.next_compile_time_args_offset();
    constexpr auto indices_args = TensorAccessorArgs<indices_accessor_offset>();
    const auto indices_accessor = TensorAccessor(indices_args, output_indices_addr);

    constexpr uint32_t NEG_INF_U32 = 0xFF800000u;

    // Reserve output CB space once — used as scratch for assembling output pages.
    cb_reserve_back(cb_out_values, 1);
    cb_reserve_back(cb_out_indices, 1);
    uint32_t out_values_l1 = get_write_ptr(cb_out_values);
    uint32_t out_indices_l1 = get_write_ptr(cb_out_indices);

    volatile tt_l1_ptr uint32_t* out_vals_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_values_l1);
    volatile tt_l1_ptr uint32_t* out_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_indices_l1);

    for (uint32_t row = start_row; row < end_row; row++) {
        cb_wait_front(cb_input, 1);
        uint32_t input_l1_addr = get_read_ptr(cb_input);

        volatile tt_l1_ptr uint32_t* data_u32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
        volatile tt_l1_ptr float* data = reinterpret_cast<volatile tt_l1_ptr float*>(input_l1_addr);

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

            // Mask found max so next iteration skips it
            data_u32[max_idx] = NEG_INF_U32;
        }

        noc_async_write_page(row, values_accessor, out_values_l1);
        noc_async_write_page(row, indices_accessor, out_indices_l1);
        noc_async_write_barrier();

        cb_pop_front(cb_input, 1);
    }
}
