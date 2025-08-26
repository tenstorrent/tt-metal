// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;
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
    // Datatypes will be multiple of 2 bytes only so it is safe to use uint16_t pointer
    volatile tt_l1_ptr uint16_t* patch_data = (volatile uint16_t*)intermed_l1_scratch;
    for (uint32_t input_idx = 0; input_idx < work_per_core; input_idx++) {
        uint32_t idx = 0;
        cb_wait_front(cb_id_in0, 1);
        uint32_t l1_addr = get_read_ptr(cb_id_in0);
        if constexpr (!is_l1_aligned) {
            for (uint32_t i = 0; i < patch_size; i++) {
                for (uint32_t j = 0; j < (stick_nbytes / 2); j++) {
                    patch_data[idx++] = *(volatile uint16_t*)(l1_addr + j * 2);
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
        noc_async_write_barrier();
        cb_pop_front(cb_id_in0, 1);
        dst_index++;
    }
}
