// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Function reads from RM and writes to RM repeating the last dimension
*/
#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    // We are guranteed to be in 2D going to 2D

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    // Which set of pages to deal with
    const uint32_t page_start = get_arg_val<uint32_t>(2);
    const uint32_t page_end = get_arg_val<uint32_t>(3);
    // If work is not divided up nicely between the cores/ tensor too small we can use this to not run this core.
    const uint32_t nop = get_arg_val<uint32_t>(4);

    constexpr bool tensor_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t original_page_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_repeats = get_compile_time_arg_val(2);
    // cb_id_in0 and cb_id_in1 is each 1 page of size:
    // if original_page_size_bytes is a multiple of 16, equal to original_page_size_bytes + 128
    // else if original_page_size_bytes is a multiple of 8, equal to original_page_size_bytes * 2 + 128
    // else if original_page_size_bytes is a multiple of 4, equal to original_page_size_bytes * 4 + 128
    // else if original_page_size_bytes is a multiple of 2, equal to original_page_size_bytes * 8 + 128
    // if it is an odd number equal to original_page_size_bytes * 16 + 128
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(4);
    constexpr bool source_page_is_pow_2 = (get_compile_time_arg_val(5) == 1);
    constexpr uint32_t source_page_pow_2 = get_compile_time_arg_val(6);
    constexpr bool dest_page_is_pow_2 = (get_compile_time_arg_val(7) == 1);
    constexpr uint32_t dest_page_pow_2 = get_compile_time_arg_val(8);
    constexpr uint32_t dest_page_size_bytes = original_page_size_bytes * num_repeats;
    // Number of times we must double the input page to make it write aligned
    constexpr uint32_t num_doublings = ((original_page_size_bytes % 16) == 0)  ? 0
                                       : ((original_page_size_bytes % 8) == 0) ? 1
                                       : ((original_page_size_bytes % 4) == 0) ? 2
                                       : ((original_page_size_bytes % 2) == 0) ? 3
                                                                               : 4;

    // Since we need to operate on a grid of cores but sometimes pages don't split properly, if nop then don't use this
    // core
    if (nop == 1) {
        return;
    }

    const auto s = get_interleaved_addr_gen<tensor_is_dram, source_page_is_pow_2>(
        src_addr, original_page_size_bytes, source_page_pow_2);
    const auto d =
        get_interleaved_addr_gen<tensor_is_dram, dest_page_is_pow_2>(dst_addr, dest_page_size_bytes, dest_page_pow_2);

    // Get scratchpads guaranteed to be allocated until the function terminates
    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_buffer = get_write_ptr(cb_id_in0);
    uint32_t alignment_buffer = get_write_ptr(cb_id_in1);
    cb_push_back(cb_id_in1, 1);
    cb_push_back(cb_id_in0, 1);

    constexpr uint64_t r_mask_to_use = tensor_is_dram ? MASK_64 : MASK_16;
    constexpr uint64_t r_offset_to_use = tensor_is_dram ? OFFSET_64 : OFFSET_16;
    constexpr uint32_t r_alignment_requirement = tensor_is_dram ? 64 : 16;
    constexpr uint32_t w_alignment_requirement = 16;
    constexpr uint64_t w_mask_to_use = MASK_16;
    constexpr uint64_t w_offset_to_use = OFFSET_16;

    alignment_buffer =
        align_address<w_alignment_requirement>(alignment_buffer, w_mask_to_use);         // Guaranteed aligned for write
    input_buffer = align_address<r_alignment_requirement>(input_buffer, r_mask_to_use);  // Guaranteed aligned for reads

    uint32_t cur_page_size = original_page_size_bytes;
    for (uint32_t i = page_start; i < page_end; i++) {
        // Read from source
        uint64_t src_noc_addr = s.get_noc_addr(i, 0);
        uint64_t dst_noc_addr = d.get_noc_addr(i, 0);
        uint32_t data_location =
            input_buffer + (src_noc_addr & r_offset_to_use);  // Guaranteed to be aligned for our read
        enhanced_noc_async_read<original_page_size_bytes, false>(src_noc_addr, data_location, original_page_size_bytes);
        cur_page_size = original_page_size_bytes;
        noc_async_read_barrier();
        if constexpr (num_doublings != 0) {
            // The if is not needed but it is just for performance as the vast majority of times num_doublings will be 0
            // and we don't want target offset to be allocated and the for loop bounds computed
            uint32_t target_offset = original_page_size_bytes;
            for (uint32_t j = 0; j < num_doublings; j++) {
                // This ensures the cur_page_size will be alligned to 16B so future walk retains allignment
                tt_memmove<false, false, false, 16 * original_page_size_bytes>(
                    data_location + target_offset, data_location, cur_page_size);
                target_offset += cur_page_size;
                cur_page_size *= 2;
            }
        }
        // Write to destination
        // data is at data_location and there is cur_page_size bytes worth of data there
        if ((data_location & w_offset_to_use) != (dst_noc_addr & w_offset_to_use)) {
            // Can't directly copy due to alignment
            tt_memmove<false, false, false, 16 * original_page_size_bytes>(
                alignment_buffer + (dst_noc_addr & w_offset_to_use), data_location, cur_page_size);
            data_location = alignment_buffer + (dst_noc_addr & w_offset_to_use);
        }

        uint64_t num_written = 0;
        while (num_written < dest_page_size_bytes) {
            // Either write out the whole input buffer or however much is left
            uint32_t to_write = (dest_page_size_bytes - num_written) > cur_page_size
                                    ? cur_page_size
                                    : (dest_page_size_bytes - num_written);
            enhanced_noc_async_write<dest_page_size_bytes, false>(data_location, dst_noc_addr + num_written, to_write);
            num_written += to_write;
        }
        noc_async_write_barrier();
    }
    return;
}
