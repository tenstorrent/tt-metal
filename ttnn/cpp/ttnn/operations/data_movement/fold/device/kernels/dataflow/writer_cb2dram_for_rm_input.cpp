// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

#define dump(a)                                               \
    do {                                                      \
        DPRINT << "Activations: " << #a " = " << a << ENDL(); \
    } while (false)

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_stick_nbytes_dram = get_compile_time_arg_val(2);
    constexpr uint32_t stride_h = get_compile_time_arg_val(3);
    constexpr uint32_t stride_w = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t work_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(7);
    constexpr bool is_l1_aligned = get_compile_time_arg_val(8);
    constexpr auto dst_args = TensorAccessorArgs<9>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t patch_size = stride_h * stride_w;
    const auto s_out = TensorAccessor(dst_args, dst_addr, stick_nbytes * patch_size);
    uint32_t dst_index = get_arg_val<uint32_t>(1);
    uint32_t intermed_l1_scratch = get_write_ptr(cb_id_in1);
    volatile tt_l1_ptr uint8_t* patch_data = (volatile uint8_t*)intermed_l1_scratch;
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        uint32_t idx = 0;
        cb_wait_front(cb_id_in0, 1);
        uint32_t l1_addr = get_read_ptr(cb_id_in0);
        if constexpr (!is_l1_aligned) {
            for (uint32_t i = 0; i < patch_size; i++) {
                for (uint32_t j = 0; j < stick_nbytes; j++) {
                    patch_data[idx++] = *(volatile uint8_t*)(l1_addr + j);
                }
                l1_addr += aligned_stick_nbytes_dram;
            }
        }
        uint64_t dst_noc_addr = get_noc_addr(dst_index, s_out);
        if constexpr (!is_l1_aligned) {
            noc_async_write((uint32_t)patch_data, dst_noc_addr, stick_nbytes * patch_size);
        } else {
            // If L1 aligned, write directly from the circular buffer
            noc_async_write(l1_addr, dst_noc_addr, stick_nbytes * patch_size);
        }
        dump(dst_index);
        dump(patch_size);
        print_bf16_pages((uint32_t)patch_data, stick_nbytes / 2, patch_size);
        noc_async_write_barrier();
        dst_index++;
        cb_pop_front(cb_id_in0, 1);
    }
}
