// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "cpp/ttnn/operations/experimental/ccl/split_scatter/device/kernels/dataflow/scatter_utils.hpp"

void kernel_main() {
    DPRINT << "send reader start" << ENDL();
    uint32_t arg_idx = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_start_addr_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t col_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t col_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cols = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_output_page_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_output_addr_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_addr_offset = get_arg_val<uint32_t>(arg_idx++);
    const bool is_clockwise_direction = get_arg_val<uint32_t>(arg_idx++) == 1;

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(4);
    uint32_t sem_addr = get_semaphore(get_compile_time_arg_val(5));
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(6);
    constexpr uint32_t ring_size = get_compile_time_arg_val(7);
    constexpr uint32_t input_tile_size = get_compile_time_arg_val(8);
    constexpr uint32_t output_tile_size = get_compile_time_arg_val(9);

    ASSERT(half_cb_n_pages > rem_num_pages);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

#ifdef ROW_MAJOR_LAYOUT
#ifdef INTERLEAVED_MEM_LAYOUT
    const InterleavedAddrGen<src_is_dram> s = {.bank_base_address = src_addr, .page_size = page_size};
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
#endif
#elif defined TILED_LAYOUT
#ifdef INTERLEAVED_MEM_LAYOUT
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram, input_tile_size> s = {
        .bank_base_address = src_addr, .page_size = page_size, .data_format = in0_df};

    InterleavedAddrGenFast<dst_is_dram, output_tile_size> d = {
        .bank_base_address = dst_addr, .page_size = output_page_size, .data_format = in0_df};
#endif
#endif
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t input_page_idx = input_start_idx;
    uint32_t output_base_page_idx = output_start_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;
    DPRINT << "send reader " << ENDL();
    DPRINT << "src_addr " << src_addr << ENDL();
    DPRINT << "dst_addr " << dst_addr << ENDL();
    DPRINT << "input_ring_idx " << input_ring_idx << ENDL();
    DPRINT << "input_page_idx " << input_page_idx << ENDL();
    DPRINT << "output_base_page_idx " << output_base_page_idx << ENDL();
    DPRINT << "output_page_idx " << output_page_idx << ENDL();
    DPRINT << "col_idx " << col_idx << ENDL();
    DPRINT << "row_idx " << row_idx << ENDL();
    DPRINT << "row_start_idx " << row_start_idx << ENDL();
    DPRINT << "col_start_idx " << col_start_idx << ENDL();
    DPRINT << "num_rows " << num_rows << ENDL();
    DPRINT << "num_rows " << num_cols << ENDL();
    DPRINT << "num_full_chunks " << num_full_chunks << ENDL();
    if (num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            read_chunk_from_input_tensor(input_page_idx, cb_id_in0, s, num_pages, page_size);
        }
    }
    DPRINT << "rem_num_pages " << rem_num_pages << ENDL();
    if (rem_num_pages > 0) {
        read_chunk_from_input_tensor(input_page_idx, cb_id_in0, s, rem_num_pages, page_size);
        ASSERT(num_pages == 0 || num_pages > rem_num_pages);
        ASSERT(half_cb_n_pages > rem_num_pages);
        push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
    }

    uint32_t sem_idx = 1;
    DPRINT << "num_transfersssss " << num_transfers << ENDL();
    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
        DPRINT << "num_transfers " << num_transfers << ENDL();
        if (is_clockwise_direction) {
            if (input_ring_idx == 0) {
                input_ring_idx = ring_size - 1;
                if (output_addr_offset != 0) {
                    d.bank_base_address += last_output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx += last_output_page_offset;
                }
            } else {
                input_ring_idx--;
                if (output_addr_offset != 0) {
                    d.bank_base_address -= output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx -= output_page_offset;
                }
            }
        } else {
            if (input_ring_idx == ring_size - 1) {  // 0) {
                input_ring_idx = 0;
                if (output_addr_offset != 0) {
                    d.bank_base_address -= last_output_addr_offset;
                    // d.bank_base_address = last_output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx -= last_output_page_offset;
                    // output_base_page_idx = last_output_page_offset;
                }
            } else {
                input_ring_idx++;
                if (output_addr_offset != 0) {
                    d.bank_base_address += output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx += output_page_offset;
                }
            }
        }
        output_page_idx = output_base_page_idx;
        col_idx = col_start_idx;
        row_idx = row_start_idx;
        DPRINT << "num_full_chunks num_full_chunks " << num_full_chunks << ENDL();
        DPRINT << "rem_num_pages rem_num_pages " << rem_num_pages << ENDL();
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
                sem_idx++;
                read_chunk_from_output_tensor(
                    output_page_idx,
                    col_idx,
                    row_idx,
                    cb_id_in0,
                    d,
                    num_cols,
                    num_rows,
                    col_offset,
                    row_offset,
                    num_pages,
                    page_size);
            }
        }
        // if (rem_num_pages > 0) {
        //     noc_semaphore_wait_min(sender_semaphore_addr_ptr, sem_idx);
        //     sem_idx++;
        //     read_chunk_from_output_tensor(
        //         output_page_idx,
        //         col_idx,
        //         row_idx,
        //         cb_id_in0,
        //         d,
        //         num_cols,
        //         num_rows,
        //         col_offset,
        //         row_offset,
        //         rem_num_pages,
        //         page_size);
        //     ASSERT(num_pages == 0 || num_pages > rem_num_pages);
        //     ASSERT(half_cb_n_pages > rem_num_pages);
        //     push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        // }
    }
    DPRINT << "send reader end" << ENDL();
}
