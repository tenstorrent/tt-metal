// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "utils/bfloat16.h"

#define NEG_INF_BFLOAT16 0xFF80  // Representation of negative infinity in bfloat16

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr uint32_t in_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t out_stick_size = get_compile_time_arg_val(5);

    // This is the number of elements in the output. i.e. for a tensor of shape (B, C, H, W), this is B * C * H
    // Similar computation works for tensors of all ranks (product of all dims except the reduction dim)
    constexpr uint32_t num_outputs = get_compile_time_arg_val(6);

    // This is the number of elements in the input tensor along the reduction dim
    constexpr uint32_t red_dim_length = get_compile_time_arg_val(7);

    // Boolean to indicate if we reduce across _all_ dimensions or just on the reduction dim (last dim)
    constexpr bool reduce_all = (bool)get_compile_time_arg_val(8);

    const InterleavedAddrGen<src_is_dram> s_in = {.bank_base_address = src_addr, .page_size = in_stick_size};
    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_size};

    // Use cb as L1 scratch memory
    uint32_t in_addr = get_write_ptr(cb_id_in);
    volatile tt_l1_ptr uint16_t* in_vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_addr);

    // Use cb as L1 scratch memory
    uint32_t out_addr = get_write_ptr(cb_id_out);
    volatile tt_l1_ptr uint32_t* out_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr);

    uint32_t max_index;
    uint32_t max_val = NEG_INF_BFLOAT16;

    for (uint32_t j = 0; j < num_outputs; j++) {
        noc_async_read_page(j, s_in, in_addr);
        noc_async_read_barrier();

        // Reset max_val for each new output
        if constexpr (not reduce_all) {
            max_val = NEG_INF_BFLOAT16;
        }

        for (uint32_t i = 0; i < red_dim_length; ++i) {
            uint16_t val = in_vals[i];
            if (bfloat16_greater(val, max_val)) {
                max_index = reduce_all ? (j * red_dim_length + i) : i;
                max_val = val;
            }
        }
        if constexpr (not reduce_all) {
            out_vals[j] = max_index;
        }
    }
    // TODO: Generalize write for argmax for other dims
    if constexpr (reduce_all) {
        out_vals[0] = max_index;
    }
    uint64_t dst_noc_addr = get_noc_addr(0, s_out);

    noc_async_write(out_addr, dst_noc_addr, out_stick_size);
    noc_async_write_barrier();
}
