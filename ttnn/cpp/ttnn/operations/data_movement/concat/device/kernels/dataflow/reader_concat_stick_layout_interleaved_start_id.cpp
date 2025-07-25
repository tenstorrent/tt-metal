// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Make n reads defined by num_reads
// Writes to Specified Circular Buffers in L1
// Expects n provided src_addr, src_noc_x, src_noc_y, and cb_id_in
void kernel_main() {
    const uint32_t num_pages = get_arg_val<uint32_t>(0);
    const uint32_t start_tensor = get_arg_val<uint32_t>(1);
    const uint32_t start_tensor_id = get_arg_val<uint32_t>(2);

    constexpr auto tensor_args_0 = TensorAccessorArgs<0>();
    constexpr auto tensor_args_1 = TensorAccessorArgs<tensor_args_0.compile_time_args_skip()>();
    constexpr auto tensor_args_2 = TensorAccessorArgs<tensor_args_1.compile_time_args_skip()>();
    constexpr auto tensor_args_3 = TensorAccessorArgs<tensor_args_2.compile_time_args_skip()>();
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(tensor_args_3.compile_time_args_skip());
    constexpr uint32_t num_tensors = get_compile_time_arg_val(tensor_args_3.compile_time_args_skip() + 1);

    // ublocks size defined in pages
    constexpr uint32_t ublock_size_pages = 1;

    uint8_t tensor_accessor_memblk_0[sizeof(TensorAccessor)];
    uint8_t tensor_accessor_memblk_1[sizeof(TensorAccessor)];
    uint8_t tensor_accessor_memblk_2[sizeof(TensorAccessor)];
    uint8_t tensor_accessor_memblk_3[sizeof(TensorAccessor)];

    TensorAccessor* tensor_accessors[num_tensors];

    uint32_t num_pages_per_block[num_tensors];
    uint32_t page_id_per_tensor[num_tensors];
    constexpr uint32_t src_addr_base_idx = 3;
    constexpr uint32_t num_pages_per_block_base_offset = num_tensors;
    constexpr uint32_t page_size_per_tensor_offset = num_pages_per_block_base_offset + num_tensors;
    constexpr uint32_t page_id_per_tensor_offset = page_size_per_tensor_offset + num_tensors;
    tt_l1_ptr uint32_t* arg_ptr = (tt_l1_ptr uint32_t*)get_arg_addr(src_addr_base_idx);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t src_addr = arg_ptr[i];
        num_pages_per_block[i] = arg_ptr[num_pages_per_block_base_offset + i];
        page_id_per_tensor[i] = arg_ptr[page_id_per_tensor_offset + i];
        uint32_t page_size = arg_ptr[page_size_per_tensor_offset + i];

        if (i == 0) {
            tensor_accessors[i] = new (tensor_accessor_memblk_0) TensorAccessor(tensor_args_0, src_addr, page_size);
        } else if (i == 1) {
            tensor_accessors[i] = new (tensor_accessor_memblk_1) TensorAccessor(tensor_args_1, src_addr, page_size);
        } else if (i == 2) {
            tensor_accessors[i] = new (tensor_accessor_memblk_2) TensorAccessor(tensor_args_2, src_addr, page_size);
        } else if (i == 3) {
            tensor_accessors[i] = new (tensor_accessor_memblk_3) TensorAccessor(tensor_args_3, src_addr, page_size);
        }
    }

    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    // FIX RM CONCAT WIDTH
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id_in, ublock_size_pages);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);
#ifdef WIDTH_CONCAT
        // For width concat we know we start at curr_tensor=0
        // num_pages_per_block[curr_tensor] is always one for width concat
        for (uint32_t j = 0; j < num_tensors; ++j) {
            noc_async_read_page(page_id_per_tensor[curr_tensor], *tensor_accessors[curr_tensor], l1_write_addr);
            l1_write_addr += tensor_accessors[curr_tensor]->page_size;
            page_id_per_tensor[curr_tensor]++;
            curr_tensor++;
        }
        curr_tensor = 0;
#else
        noc_async_read_page(page_id_per_tensor[curr_tensor], *tensor_accessors[curr_tensor], l1_write_addr);

        page_id_per_tensor[curr_tensor]++;
        curr_tensor_id++;

        if (curr_tensor_id == num_pages_per_block[curr_tensor]) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
            }
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_id_in, ublock_size_pages);
    }
}
