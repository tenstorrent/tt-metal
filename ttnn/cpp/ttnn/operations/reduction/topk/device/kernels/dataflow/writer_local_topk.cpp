// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t receiver_semaphore = get_semaphore(get_compile_time_arg_val(0));
    uint32_t sender_semaphore = get_semaphore(get_compile_time_arg_val(1));
    constexpr uint32_t noc_final_x = get_compile_time_arg_val(2);
    constexpr uint32_t noc_final_y = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t Kt = get_compile_time_arg_val(6);

    uint32_t start_wt = get_arg_val<uint32_t>(0);

    constexpr uint32_t values_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t output_ind_cb_index = tt::CBIndex::c_17;

    constexpr uint32_t topk_local_values_cb_index = tt::CBIndex::c_24;
    constexpr uint32_t topk_local_indices_cb_index = tt::CBIndex::c_25;

    constexpr uint32_t final_values_cb_index = tt::CBIndex::c_26;
    constexpr uint32_t final_indices_cb_index = tt::CBIndex::c_27;

    // can amortize the noc reads by doing them side by side for the two tensors
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes_values = get_tile_size(values_cb_index);
    const uint32_t tile_bytes_ind = get_tile_size(output_ind_cb_index);

    uint32_t final_values_cb_addr = get_write_ptr(final_values_cb_index);
    uint32_t final_indices_cb_addr = get_write_ptr(final_indices_cb_index);

    uint64_t noc_final_addr_values =
        get_noc_addr(noc_final_x, noc_final_y, final_values_cb_addr) + start_wt * tile_bytes_values * Kt;
    uint64_t noc_value_addr_values =
        get_noc_addr(noc_final_x, noc_final_y, final_indices_cb_addr) + start_wt * tile_bytes_ind * Kt;

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore);
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore);

    uint64_t noc_remote_sender_semaphore_addr = get_noc_addr(noc_final_x, noc_final_y, (uint32_t)sender_semaphore_addr);

    // Get Kt rows of values and then Kt rows of indices from compute kernel
    for (uint32_t j = 0; j < Ht; ++j) {
        // wait until we can write to the cb of the final topk core
        noc_semaphore_wait(receiver_semaphore_addr, VALID);
        // write out topk values of the local chunk
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(values_cb_index, onetile);
            uint32_t l1_read_addr_val = get_read_ptr(values_cb_index);
            noc_async_write(l1_read_addr_val, noc_final_addr_values + i * tile_bytes_values, tile_bytes_values);
            cb_pop_front(values_cb_index, onetile);
        }

        // write out topk indices of the local chunk
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(output_ind_cb_index, onetile);
            uint32_t l1_read_addr_ind = get_read_ptr(output_ind_cb_index);
            noc_async_write(l1_read_addr_ind, noc_value_addr_values + i * tile_bytes_ind, tile_bytes_ind);
            cb_pop_front(output_ind_cb_index, onetile);
        }
        // since we're writing to a precise location we don't need the barrier until later
        noc_async_write_barrier();
        // signal the receiver that this local chunk has sent its Kt tiles
        noc_semaphore_inc(noc_remote_sender_semaphore_addr, Kt);
        // set the receiver ready semaphore to invalid until the receiver is ready to receive data
        noc_semaphore_set(receiver_semaphore_addr, INVALID);
    }
    noc_async_atomic_barrier();
}
