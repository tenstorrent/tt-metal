// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * TopK Multicore Reader Kernel - Final Core Data Aggregation
 *
 * This kernel runs exclusively on the final aggregation core and coordinates the
 * collection of local TopK results from all worker cores. It implements a
 * semaphore-based synchronization protocol to aggregate results efficiently.
 *
 * Responsibilities:
 * 1. Coordinate data reception from multiple local processing cores
 * 2. Manage semaphore-based flow control to prevent buffer overflow
 * 3. Aggregate received data for final compute kernel processing
 * 4. Handle multicast synchronization across all contributing cores
 *
 * Synchronization Protocol:
 * - Uses paired semaphores for bidirectional flow control
 * - Receiver semaphore: Signals readiness to receive data
 * - Sender semaphore: Tracks completion of data transmission
 * - Multicast operations coordinate with multiple sender cores simultaneously
 */
void kernel_main() {
    // Compiletime args
    const uint32_t receiver_semaphore = get_semaphore(get_compile_time_arg_val(0));  // Ready-to-receive signal
    const uint32_t sender_semaphore = get_semaphore(get_compile_time_arg_val(1));    // Data-sent confirmation
    constexpr uint32_t noc_start_x = get_compile_time_arg_val(2);              // Starting X coordinate of core range
    constexpr uint32_t noc_start_y = get_compile_time_arg_val(3);              // Starting Y coordinate of core range
    constexpr uint32_t noc_end_x = get_compile_time_arg_val(4);                // Ending X coordinate of core range
    constexpr uint32_t noc_end_y = get_compile_time_arg_val(5);                // Ending Y coordinate of core range
    constexpr uint32_t Ht = get_compile_time_arg_val(6);                       // Height tiles to process
    constexpr uint32_t Wt_final = get_compile_time_arg_val(7);                 // Total width tiles from all cores
    constexpr uint32_t num_dests = get_compile_time_arg_val(8);                // Number of sending cores
    constexpr uint32_t final_values_cb_index = get_compile_time_arg_val(9);    // Aggregated TopK values
    constexpr uint32_t final_indices_cb_index = get_compile_time_arg_val(10);  // Aggregated TopK indices

    // Semaphore address mapping for direct NoC operations
    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_semaphore);
    volatile tt_l1_ptr uint32_t* sender_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_semaphore);

    // Multicast address for coordinating with all sender cores simultaneously
    const uint64_t mcast_receiver_semaphore_noc_addr =
        get_noc_multicast_addr(noc_start_x, noc_start_y, noc_end_x, noc_end_y, receiver_semaphore);

    // Collect local TopK results from all cores
    for (uint32_t i = 0; i < Ht; ++i) {  // Process each height row
        // STEP 1: PREPARE RECEPTION BUFFERS
        // Reserve space for incoming data from all local cores
        cb_reserve_back(final_values_cb_index, Wt_final);   // Space for all TopK values
        cb_reserve_back(final_indices_cb_index, Wt_final);  // Space for all TopK indices

        // STEP 2: INITIALIZE SYNCHRONIZATION STATE
        // Reset synchronization state for this height row
        noc_semaphore_set(sender_semaphore_addr, INVALID);  // Mark data as not yet sent
        noc_semaphore_set(receiver_semaphore_addr, VALID);  // Signal readiness to receive

        // STEP 3: COORDINATE MULTICAST RECEPTION
        // Enable all local cores to send their data simultaneously by broadcasting
        // the receiver semaphore state. This allows for efficient parallel transmission.
        noc_semaphore_set_multicast(receiver_semaphore, mcast_receiver_semaphore_noc_addr, num_dests);
        noc_async_write_barrier();

        // STEP 4: WAIT FOR DATA COMPLETION
        // Block until all expected data (Wt_final tiles) has been received from
        // the local cores. The sender semaphore is incremented by each sending core.
        noc_semaphore_wait(sender_semaphore_addr, Wt_final);

        // STEP 5: COMMIT RECEIVED DATA
        // Mark the received data as available to the final compute kernel
        cb_push_back(final_values_cb_index, Wt_final);
        cb_push_back(final_indices_cb_index, Wt_final);
    }

    // Ensure all NoC operations complete before kernel termination
    noc_async_write_barrier();
}
