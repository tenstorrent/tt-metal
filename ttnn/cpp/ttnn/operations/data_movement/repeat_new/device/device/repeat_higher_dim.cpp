// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Function reads from RM and writes to RM repeating the last dimension
*/
#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // We are guranteed to be in 2D going to 2D

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    // Program factory can control the start and end of each of the 3 dims
    const uint32_t higher_dim_start = get_arg_val<uint32_t>(2);
    const uint32_t higher_dim_end = get_arg_val<uint32_t>(3);
    const uint32_t lower_dim_start = get_arg_val<uint32_t>(4);
    const uint32_t lower_dim_end = get_arg_val<uint32_t>(5);
    const uint32_t rep_dim_start = get_arg_val<uint32_t>(6);
    const uint32_t rep_dim_end = get_arg_val<uint32_t>(7);
    const uint32_t repetitions = get_arg_val<uint32_t>(8);
    // nop lets you intentionally not use this core if the dims don't divide nicely
    const uint32_t nop = get_arg_val<uint32_t>(9);

    constexpr bool tensor_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t original_page_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);
#define page_is_pow_2 (get_compile_time_arg_val(3) == 1)
    constexpr uint32_t page_pow_2 = get_compile_time_arg_val(4);
    //(higher_dim,rep_dim,lower_dim,page_size)
    // cb_id_in0 and cb_id_in1 is each 1 page of size:
    // if multiple of 16, equal to original_page_size_bytes + 128
    // else if multiple of 8, equal to original_page_size_bytes * 2 + 128
    // else if multiple of 4, equal to original_page_size_bytes * 4 + 128
    // else if multiple of 2, equal to original_page_size_bytes * 8 + 128
    // else equal to original_page_size_bytes * 16
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(6);
    constexpr uint32_t LOWER_DIMS = get_compile_time_arg_val(7);
    constexpr uint32_t HIGHER_DIMS = get_compile_time_arg_val(8);
    constexpr uint32_t REP_DIM = get_compile_time_arg_val(9);

    // Since we need to operate on a grid of cores but sometimes pages don't split properly, if nop then don't use this
    // core
    if (nop == 1) {
        return;
    }
#if page_is_pow_2
    // TODO: add CCL sharded native support
    const InterleavedPow2AddrGen<tensor_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = page_pow_2};
    const InterleavedPow2AddrGen<tensor_is_dram> d = {
        .bank_base_address = dst_addr, .log_base_2_of_page_size = page_pow_2};
#else
    const InterleavedAddrGen<tensor_is_dram> s = {.bank_base_address = src_addr, .page_size = original_page_size_bytes};
    const InterleavedAddrGen<tensor_is_dram> d = {.bank_base_address = dst_addr, .page_size = original_page_size_bytes};
#endif

    // alignments pre-calculations
    constexpr uint64_t r_mask_to_use = tensor_is_dram ? MASK_64 : MASK_16;
    constexpr uint64_t r_offset_to_use = tensor_is_dram ? OFFSET_64 : OFFSET_16;
    constexpr uint32_t r_alignment_requirement = tensor_is_dram ? 64 : 16;
    const uint32_t w_alignment_requirement = 16;
    const uint64_t w_mask_to_use = MASK_16;
    const uint64_t w_offset_to_use = OFFSET_16;

    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_buffer = get_write_ptr(cb_id_in0);
    uint32_t alignment_buffer = get_write_ptr(cb_id_in1);
    cb_push_back(cb_id_in1, 1);
    cb_push_back(cb_id_in0, 1);

    alignment_buffer = (alignment_buffer & w_mask_to_use) + w_alignment_requirement;  // alligned for writes
    input_buffer = (input_buffer & r_mask_to_use) + r_alignment_requirement;          // alligned for reads
    // The jump for the upper dims
    constexpr uint32_t higher_jump = LOWER_DIMS * REP_DIM;
    uint32_t src_noc_addr = 0;
    uint32_t data_location = 0;
    // lower dimensions stride
    uint32_t l_offset = lower_dim_start;
    // target dimension stride
    uint32_t r_offset = rep_dim_start * LOWER_DIMS;
    // upper dimension stride in read and in write. Note in write we need to consider the repetitions of the r dimension
    uint32_t u_r_offset = higher_dim_start * higher_jump;
    uint32_t u_w_offset = higher_dim_start * higher_jump * repetitions;
    for (int l = lower_dim_start; l < lower_dim_end; l++) {
        for (int r = rep_dim_start; r < rep_dim_end; r++) {
            uint32_t combined_offset = l_offset + r_offset;
            for (int h = higher_dim_start; h < higher_dim_end; h++) {
                // Perform the read
                src_noc_addr = s.get_noc_addr(combined_offset + u_r_offset, 0);
                data_location = input_buffer + (src_noc_addr & offset_to_use);  // Guaranteed aligned to src_noc_addr
                tt::data_movement::common::enhanced_noc_async_read<original_page_size_bytes, false>(
                    src_noc_addr, data_location, original_page_size_bytes);
                combined_offset += u_w_offset;  // offset of the higher dims
                noc_async_read_barrier();
                for (int n = 0; n < repetitions; n++) {
                    // Perform the writes
                    const uint32_t dst_noc_addr = d.get_noc_addr(combined_offset, 0);
                    combined_offset += higher_jump;
                    if ((data_location & w_offset_to_use) != (dst_noc_addr & w_offset_to_use)) {
                        // Can't directly copy
                        const uint32_t target_align_buffer =
                            alignment_buffer +
                            (dst_noc_addr & w_offset_to_use);  // Guaranteed aligned to target page addr
                        tt::data_movement::common::tt_memmove<false, false, false, original_page_size_bytes>(
                            target_align_buffer,
                            data_location,
                            original_page_size_bytes);  // Data is copied to align buffer
                        data_location = alignment_buffer +
                                        (dst_noc_addr & w_offset_to_use);  // Update data location to use write buffer
                    }
                    // Now we are ensured the data is at write_buffer and it is aligned for the write
                    // Orchestrate the write
                    tt::data_movement::common::enhanced_noc_async_write<original_page_size_bytes, false>(
                        data_location, dst_noc_addr, original_page_size_bytes);
                }
                // Go to the next upper dim offset
                u_r_offset += higher_jump;
                u_w_offset += (higher_jump * repetitions);
                // We invoked all the writes, now we let them complete
                noc_async_write_barrier();
            }
            // Go to the next repetition dim value
            r_offset += LOWER_DIMS;
        }
        // Go to the next page
        l_offset++;
    }
    return;
}
