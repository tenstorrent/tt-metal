// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t padded_stick_size = get_arg_val<uint32_t>(1);
    const uint32_t unpadded_stick_size = get_arg_val<uint32_t>(2);
    const uint32_t stick_size_offset = get_arg_val<uint32_t>(3);
    const uint32_t num_dims = get_arg_val<uint32_t>(4);
    const uint32_t start_id = get_arg_val<uint32_t>(5);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(7);
    const uint32_t num_read_per_barrier = get_arg_val<uint32_t>(8);
    const uint32_t num_unpadded_sticks_addr = get_arg_addr(9);

    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(num_unpadded_sticks_addr);
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;

    constexpr uint32_t is_non_aligned = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_non_aligned = get_compile_time_arg_val(1);
    constexpr uint32_t src_buffer_alignment = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    uint32_t read_size = unpadded_stick_size;

    constexpr uint32_t cb_id_in0 = 0;

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    uint32_t misalignment = 0;
    if constexpr (is_non_aligned) {
        misalignment = (src_addr % src_buffer_alignment);
    }
    const auto s0 = TensorAccessor(src_args, src_addr - misalignment, padded_stick_size);

#ifdef DEBUG
    DPRINT << "src_addr: " << src_addr << ", padded_stick_size: " << padded_stick_size
           << ", unpadded_stick_size: " << unpadded_stick_size << ", stick_size_offset: " << stick_size_offset
           << ", num_dims: " << num_dims << ", start_id: " << start_id
           << ", num_sticks_per_core: " << num_sticks_per_core
           << ", num_sticks_per_core_read: " << num_sticks_per_core_read
           << ", num_read_per_barrier: " << num_read_per_barrier << ENDL();

    DPRINT << "non_aligned: " << is_non_aligned << "cb = " << cb_id_non_aligned << ", read_size: " << read_size
           << ", src_stick_id: " << src_stick_id << ", sticks_read: " << sticks_read << ENDL();

    DPRINT << "num_unpadded_sticks: " << num_unpadded_sticks[0] << " " << num_unpadded_sticks[1] << " "
           << num_unpadded_sticks[2] << " " << num_unpadded_sticks[3] << " " << ENDL();
    DPRINT << "num_padded_sticks: " << num_padded_sticks[0] << " " << num_padded_sticks[1] << " "
           << num_padded_sticks[2] << " " << num_padded_sticks[3] << " " << ENDL();
    DPRINT << "Out CB Page size: " << get_local_cb_interface(cb_id_in0).fifo_page_size << ENDL();
#endif
    const uint32_t base_src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    const uint64_t base_noc_addr = get_noc_addr(0, s0);
    uint32_t non_aligned_temp_addr = 0;
    if constexpr (is_non_aligned) {
        cb_reserve_back(cb_id_non_aligned, 1);
        non_aligned_temp_addr = get_write_ptr(cb_id_non_aligned);
    }
    for (uint32_t iter = 0; iter < num_sticks_per_core_read and sticks_read < num_sticks_per_core; ++iter) {
        cb_reserve_back(cb_id_in0, num_read_per_barrier);
        uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);

        for (uint32_t i = 0; i < num_read_per_barrier and sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            uint64_t src_noc_addr = get_noc_addr(src_stick_id, s0);
            if constexpr (is_non_aligned) {
                noc_async_read(src_noc_addr, non_aligned_temp_addr, padded_stick_size + misalignment);
                noc_async_read_barrier();
                noc_async_read(
                    get_noc_addr(non_aligned_temp_addr + misalignment), src_buffer_l1_addr, unpadded_stick_size);
                noc_async_read_barrier();
            } else {
                noc_async_read(src_noc_addr, src_buffer_l1_addr, unpadded_stick_size);
            }
            src_buffer_l1_addr += stick_size_offset;
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_read_per_barrier);
    }
}
