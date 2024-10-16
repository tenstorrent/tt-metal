// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr_0 = get_arg_val<uint32_t>(1);
    uint32_t output_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t aligned_elements = get_arg_val<uint32_t>(3);
    uint32_t actual_elements = get_arg_val<uint32_t>(4);
    uint32_t element_size = get_arg_val<uint32_t>(5);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_index_0 = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_index_1 = get_compile_time_arg_val(2);
    constexpr bool src0_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool dst0_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool dst1_is_dram = get_compile_time_arg_val(5) == 1;

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = input_addr,
                                                 .page_size = aligned_elements * element_size};

    const InterleavedAddrGen<dst0_is_dram> out0 = {.bank_base_address = output_addr_0, .page_size = 32};

    const InterleavedAddrGen<dst1_is_dram> out1 = {.bank_base_address = output_addr_1,
                                                   .page_size = aligned_elements * element_size};

    uint64_t src_noc_addr = get_noc_addr(0, s0);
    uint32_t input_l1_addr = get_write_ptr(input_cb_index);
    noc_async_read(src_noc_addr, input_l1_addr, aligned_elements * element_size);
    noc_async_read_barrier();
    uint32_t indices_cb_id_index_addr = get_write_ptr(output_cb_index_1);
    volatile tt_l1_ptr int* indices_cb_id_index_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr int*>(indices_cb_id_index_addr);

#if NUM_BYTES == 4
    volatile tt_l1_ptr int* input_addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(input_l1_addr);
#elif NUM_BYTES == 2
    volatile tt_l1_ptr uint16_t* input_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);
#elif NUM_BYTES == 1
    volatile tt_l1_ptr char* input_addr_ptr = reinterpret_cast<volatile tt_l1_ptr char*>(input_l1_addr);
#endif

    uint32_t num_non_zero_indices = 0;
    uint32_t local_index = 0;
    for (uint32_t i = 0; i < actual_elements; i++) {
        if (input_addr_ptr[i] != 0) {
            indices_cb_id_index_addr_ptr[num_non_zero_indices] = i;
            num_non_zero_indices++;
        }
        local_index++;
    }

    uint32_t output_l1_addr = get_write_ptr(output_cb_index_0);
    volatile tt_l1_ptr uint32_t* output_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_l1_addr);
    output_addr_ptr[0] = num_non_zero_indices;
    uint64_t dst_noc_addr = get_noc_addr(0, out0);
    noc_async_write(output_l1_addr, dst_noc_addr, 32);
    noc_async_write_barrier();

    dst_noc_addr = get_noc_addr(0, out1);
    noc_async_write(indices_cb_id_index_addr, dst_noc_addr, aligned_elements * 4);
    noc_async_write_barrier();
}
