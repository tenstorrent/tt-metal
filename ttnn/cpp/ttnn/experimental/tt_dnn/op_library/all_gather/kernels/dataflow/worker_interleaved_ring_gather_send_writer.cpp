// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t eth_sender_l1_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t eth_sender_l1_sem_addr = get_arg_val<uint32_t>(2);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(6);
    constexpr uint32_t input_start_idx = get_compile_time_arg_val(7);
    constexpr uint32_t output_start_idx = get_compile_time_arg_val(8);
    constexpr uint32_t output_start_addr_offset = get_compile_time_arg_val(9);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(10);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(11);
    constexpr uint32_t row_offset = get_compile_time_arg_val(12);
    constexpr uint32_t col_offset = get_compile_time_arg_val(13);
    constexpr uint32_t num_rows = get_compile_time_arg_val(14);
    constexpr uint32_t num_cols = get_compile_time_arg_val(15);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(16);
    constexpr uint32_t writer_send_sem_addr = get_compile_time_arg_val(17);
    constexpr uint32_t eth_sender_noc_x = get_compile_time_arg_val(18);
    constexpr uint32_t eth_sender_noc_y = get_compile_time_arg_val(19);
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(20);
    static_assert(half_cb_n_pages > rem_num_pages, "half_cb_n_pages must be greater than or equal to rem_num_pages");

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    #ifdef RM_INTERLEAVED
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
    #elif defined TILE_INTERLEAVED
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = output_page_size,
        .data_format = in0_df
    };
    #endif

    // Used to wait until eth sender has space available
    volatile tt_l1_ptr uint32_t* writer_send_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_send_sem_addr);

    uint32_t output_page_idx = output_start_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;
    // This is different per writer core
    const uint64_t eth_l1_sender_base_noc_addr = get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_base_addr);
    // Used to signal eth sender that data is available. This is different per writer core
    const uint64_t eth_l1_sender_semaphore_addr = get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_sem_addr);

    uint32_t ID = (my_y[0] << 16) | my_x[0];

    if constexpr(num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
            noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
            // TODO: Might be better to split this?
            write_and_send_chunk(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, num_pages, page_size, eth_l1_sender_base_noc_addr, eth_l1_sender_semaphore_addr);
        }
    }

    if constexpr(rem_num_pages > 0) {
        noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
        noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
        write_and_send_chunk(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size, eth_l1_sender_base_noc_addr,eth_l1_sender_semaphore_addr);
        ASSERT(num_pages == 0 || num_pages > rem_num_pages);
        ASSERT(half_cb_n_pages > rem_num_pages);
        pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
    }

    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
        if constexpr(num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
                noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
                send_chunk(cb_id_in0, num_pages, page_size, eth_l1_sender_base_noc_addr);
                noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
            }
        }
        if constexpr(rem_num_pages > 0) {
            noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
            noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
            send_chunk(cb_id_in0, rem_num_pages, page_size, eth_l1_sender_base_noc_addr);
            noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }
    }

}
