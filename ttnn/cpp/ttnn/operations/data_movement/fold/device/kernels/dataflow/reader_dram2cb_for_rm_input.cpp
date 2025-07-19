// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(2);
    constexpr bool src_stick_size_is_power_of_two = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t src_log2_stick_size = get_compile_time_arg_val(4);
    constexpr uint32_t aligned_stick_nbytes_dram = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);
    constexpr uint32_t work_per_core = get_compile_time_arg_val(7);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const auto s0 =
        get_interleaved_addr_gen<true, src_stick_size_is_power_of_two>(src_addr, stick_nbytes, src_log2_stick_size);

    uint32_t config_l1_addr = get_read_ptr(config_cb_id);
    volatile tt_l1_ptr uint32_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_l1_addr);

    constexpr uint32_t config_buffer_entries = work_per_core * 2;
    for (uint32_t input_idx = 0; input_idx < config_buffer_entries; input_idx += 2) {
        // The fold mapping table(config_data) stores (src_index, dst_index) pairs that define how to extract
        // spatial patches(in_channels * stride_w) from the input tensor and reorganize them into the output tensor.
        // READER uses src_index (first entry at mapping_offset) to locate the input patch position,
        // Each table entry uses 2 uint32_t values, so we step by 2 for each transformation.
        uint32_t src_index = config_data[input_idx];

        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t i = 0; i < stride_w; i++) {
            uint64_t src_noc_addr = get_noc_addr(src_index, s0);
            noc_async_read(src_noc_addr, l1_write_addr, stick_nbytes);
            src_index++;
            l1_write_addr += aligned_stick_nbytes_dram;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    }
}
