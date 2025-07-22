// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    // We are guranteed to be in 4D going to 4D
    //<higher_dim,rep_dim,lower_dim,page_size>

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    // Program factory can control the start and end of each of the 3 dims
    const uint32_t higher_dim_start = get_arg_val<uint32_t>(2);
    const uint32_t higher_dim_end = get_arg_val<uint32_t>(3);
    const uint32_t lower_dim_start = get_arg_val<uint32_t>(4);
    const uint32_t lower_dim_end = get_arg_val<uint32_t>(5);
    const uint32_t repetitions = get_arg_val<uint32_t>(6);
    // nop lets you intentionally not use this core if the dims don't divide nicely
    const uint32_t nop = get_arg_val<uint32_t>(7);

    constexpr bool tensor_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t original_page_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(3);
    constexpr bool page_is_pow_2 = (get_compile_time_arg_val(4) == 1);
    constexpr uint32_t page_pow_2 = get_compile_time_arg_val(5);
    //(higher_dim,rep_dim,lower_dim,page_size)
    // cb_id_in0 and cb_id_in1 is each 1 page of size:
    // 128 + page size in bytes
    constexpr uint32_t LOWER_DIMS = get_compile_time_arg_val(6);
    constexpr uint32_t REP_DIM = get_compile_time_arg_val(7);

    constexpr uint32_t LOWER_DIMS_TIMES_REP_DIM = LOWER_DIMS * REP_DIM;

    // Since we need to operate on a grid of cores but sometimes pages don't split properly, if nop then don't use this
    // core
    if (nop == 1) {
        return;
    }

    const auto s =
        get_interleaved_addr_gen<tensor_is_dram, page_is_pow_2>(src_addr, original_page_size_bytes, page_pow_2);
    const auto d =
        get_interleaved_addr_gen<tensor_is_dram, page_is_pow_2>(dst_addr, original_page_size_bytes, page_pow_2);

    // alignments pre-calculations
    constexpr uint64_t r_mask_to_use = tensor_is_dram ? MASK_64 : MASK_16;
    constexpr uint64_t r_offset_to_use = tensor_is_dram ? OFFSET_64 : OFFSET_16;

    constexpr uint32_t r_alignment_requirement = tensor_is_dram ? 64 : 16;
    constexpr uint32_t w_alignment_requirement = 16;
    const uint64_t w_mask_to_use = MASK_16;
    const uint64_t w_offset_to_use = OFFSET_16;

    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_buffer = get_write_ptr(cb_id_in0);
    uint32_t alignment_buffer = get_write_ptr(cb_id_in1);
    cb_push_back(cb_id_in1, 1);
    cb_push_back(cb_id_in0, 1);

    alignment_buffer = align_address<w_alignment_requirement>(alignment_buffer, w_mask_to_use);  // aligned for writes
    input_buffer = align_address<r_alignment_requirement>(input_buffer, r_mask_to_use);          // aligned for reads

    uint64_t src_noc_addr = 0;
    uint32_t data_location = 0;

    for (uint32_t h = higher_dim_start; h < higher_dim_end; h++) {
        uint32_t h_offset = h * LOWER_DIMS_TIMES_REP_DIM;
        uint32_t h_offset_rep = h_offset * repetitions;
        for (uint32_t r = 0; r < REP_DIM; r++) {
            uint32_t r_offset = r * LOWER_DIMS;
            for (uint32_t l = lower_dim_start; l < lower_dim_end; l++) {
                uint32_t read_offset = h_offset + r_offset + l;
                src_noc_addr = s.get_noc_addr(read_offset, 0);
                data_location = input_buffer + (src_noc_addr & r_offset_to_use);  // Guaranteed aligned to src_noc_addr
                enhanced_noc_async_read<original_page_size_bytes, false>(
                    src_noc_addr, data_location, original_page_size_bytes);
                noc_async_read_barrier();

                for (uint32_t n = 0; n < repetitions; n++) {
                    // Perform the writes
                    uint32_t write_offset = h_offset_rep + n * LOWER_DIMS_TIMES_REP_DIM + r_offset + l;
                    const uint64_t dst_noc_addr = d.get_noc_addr(write_offset, 0);
                    if ((data_location & w_offset_to_use) != (dst_noc_addr & w_offset_to_use)) {
                        // Can't directly copy
                        const uint32_t target_align_buffer =
                            alignment_buffer +
                            (dst_noc_addr & w_offset_to_use);  // Guaranteed aligned to target page addr
                        tt_memmove<false, false, false, original_page_size_bytes>(
                            target_align_buffer,
                            data_location,
                            original_page_size_bytes);  // Data is copied to align buffer
                        data_location = alignment_buffer +
                                        (dst_noc_addr & w_offset_to_use);  // Update data location to use write buffer
                    }
                    // Now we are ensured the data is at write_buffer and it is aligned for the write
                    // Orchestrate the write
                    enhanced_noc_async_write<original_page_size_bytes, false>(
                        data_location, dst_noc_addr, original_page_size_bytes);
                }
                noc_async_write_barrier();
            }
        }
    }
    return;
}
