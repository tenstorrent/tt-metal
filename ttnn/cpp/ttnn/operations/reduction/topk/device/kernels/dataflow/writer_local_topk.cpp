// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

/**
 * TopK Multicore Writer Kernel - Local Core Data Transmission
 *
 * This kernel runs on each local processing core and is responsible for sending
 * locally computed TopK results to the final aggregation core. It implements
 * the sender side of the semaphore-based synchronization protocol.
 *
 * Responsibilities:
 * 1. Read local TopK results from compute kernel output buffers
 * 2. Coordinate with final core using semaphore-based flow control
 * 3. Transmit data via NoC to final core's aggregation buffers
 * 4. Handle proper sequencing of values and indices transmission
 *
 * Data Flow:
 * Local Compute → Local Buffers → NoC Transmission → Final Core Buffers
 */
void kernel_main() {
    // Inter-core synchronization semaphores
    uint32_t receiver_semaphore = get_semaphore(get_compile_time_arg_val(0));  // Final core readiness signal
    uint32_t sender_semaphore = get_semaphore(get_compile_time_arg_val(1));    // Local core completion signal

    // Target final core coordinates for NoC communication
    constexpr uint32_t noc_final_x = get_compile_time_arg_val(2);  // Final core X coordinate
    constexpr uint32_t noc_final_y = get_compile_time_arg_val(3);  // Final core Y coordinate

    // Processing dimensions
    constexpr uint32_t Ht = get_compile_time_arg_val(4);  // Height tiles to process
    constexpr uint32_t K = get_compile_time_arg_val(5);   // TopK value
    constexpr uint32_t Kt = get_compile_time_arg_val(6);  // TopK in tile units (ceil(K/32))

    // Runtime parameter: width offset for this core's data in final aggregation
    uint32_t start_wt = get_arg_val<uint32_t>(0);

    // Local circular buffer indices (receive from compute kernel)
    constexpr uint32_t values_cb_index = tt::CBIndex::c_16;      // Local TopK values output
    constexpr uint32_t output_ind_cb_index = tt::CBIndex::c_17;  // Local TopK indices output

    // Final core circular buffer indices (destination for transmission)
    constexpr uint32_t final_values_cb_index = tt::CBIndex::c_26;   // Final aggregation values buffer
    constexpr uint32_t final_indices_cb_index = tt::CBIndex::c_27;  // Final aggregation indices buffer

    // Memory transfer configuration
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes_values = get_tile_size(values_cb_index);
    const uint32_t tile_bytes_ind = get_tile_size(output_ind_cb_index);

    // Calculate target addresses in final core's L1 memory
    uint32_t final_values_cb_addr = get_write_ptr(final_values_cb_index);
    uint32_t final_indices_cb_addr = get_write_ptr(final_indices_cb_index);

    // NoC addresses with offset for this core's contribution
    // Each core writes to a specific offset based on its position in the aggregation
    uint64_t noc_final_addr_values =
        get_noc_addr(noc_final_x, noc_final_y, final_values_cb_addr) + start_wt * tile_bytes_values * Kt;
    uint64_t noc_final_addr_indices =
        get_noc_addr(noc_final_x, noc_final_y, final_indices_cb_addr) + start_wt * tile_bytes_ind * Kt;

    // Semaphore address mapping for NoC operations
    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore);
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore);

    // NoC address for signaling completion to final core
    uint64_t noc_remote_sender_semaphore_addr = get_noc_addr(noc_final_x, noc_final_y, (uint32_t)sender_semaphore_addr);

    // MAIN TRANSMISSION LOOP: Send local TopK results to final core
    for (uint32_t j = 0; j < Ht; ++j) {  // For each height row
        // STEP 1: WAIT FOR PERMISSION TO SEND
        // Block until the final core signals readiness to receive data
        noc_semaphore_wait(receiver_semaphore_addr, VALID);

        // STEP 2: TRANSMIT LOCAL TopK VALUES
        // Send Kt tiles of locally computed TopK values to final core
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(values_cb_index, onetile);  // Wait for compute kernel output
            uint32_t l1_read_addr_val = get_read_ptr(values_cb_index);

            // Direct NoC write to final core's aggregation buffer
            noc_async_write(l1_read_addr_val, noc_final_addr_values + i * tile_bytes_values, tile_bytes_values);
            cb_pop_front(values_cb_index, onetile);
        }

        // STEP 3: TRANSMIT LOCAL TopK INDICES
        // Send Kt tiles of corresponding TopK indices to final core
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(output_ind_cb_index, onetile);  // Wait for compute kernel output
            uint32_t l1_read_addr_ind = get_read_ptr(output_ind_cb_index);

            // Direct NoC write to final core's aggregation buffer
            noc_async_write(l1_read_addr_ind, noc_final_addr_indices + i * tile_bytes_ind, tile_bytes_ind);
            cb_pop_front(output_ind_cb_index, onetile);
        }

        // STEP 4: COMPLETE TRANSMISSION PROTOCOL
        noc_async_write_barrier();  // Ensure all data is transmitted before signaling

        // Signal completion: increment sender semaphore by Kt (number of tiles sent)
        noc_semaphore_inc(noc_remote_sender_semaphore_addr, Kt);

        // Reset receiver semaphore to prepare for next round
        noc_semaphore_set(receiver_semaphore_addr, INVALID);
    }

    // Ensure all atomic operations complete before kernel termination
    noc_async_atomic_barrier();
}
