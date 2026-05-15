// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    // Compile time args
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(0);          // Ready-to-receive signal
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(1);            // Data-sent confirmation
    constexpr uint32_t noc_start_x = get_compile_time_arg_val(2);              // Starting X coordinate of core range
    constexpr uint32_t noc_start_y = get_compile_time_arg_val(3);              // Starting Y coordinate of core range
    constexpr uint32_t noc_end_x = get_compile_time_arg_val(4);                // Ending X coordinate of core range
    constexpr uint32_t noc_end_y = get_compile_time_arg_val(5);                // Ending Y coordinate of core range
    constexpr uint32_t Ht = get_compile_time_arg_val(6);                       // Height tiles to process
    constexpr uint32_t Wt_final = get_compile_time_arg_val(7);                 // Total width tiles from all cores
    constexpr uint32_t num_dests = get_compile_time_arg_val(8);                // Number of sending cores
    constexpr uint32_t final_values_cb_index = get_compile_time_arg_val(9);    // Aggregated TopK values
    constexpr uint32_t final_indices_cb_index = get_compile_time_arg_val(10);  // Aggregated TopK indices

    Noc noc;
    Semaphore<> receiver_sem(receiver_sem_id);
    Semaphore<> sender_sem(sender_sem_id);
    CircularBuffer final_values_cb(final_values_cb_index);
    CircularBuffer final_indices_cb(final_indices_cb_index);

    // Collect local TopK results from all cores
    for (uint32_t i = 0; i < Ht; ++i) {  // Process each height row
        // Reserve space for incoming data from all local cores
        final_values_cb.reserve_back(Wt_final);   // Space for all TopK values
        final_indices_cb.reserve_back(Wt_final);  // Space for all TopK indices

        // Initialize semaphores for this height row
        // Reset synchronization state for this height row
        sender_sem.set(INVALID);  // Mark data as not yet sent
        receiver_sem.set(VALID);  // Signal readiness to receive

        // Coordinate multicast reception
        // Enable all local cores to send their data simultaneously by broadcasting
        // the receiver semaphore state. This allows for efficient parallel transmission.
        receiver_sem.set_multicast<Noc::McastMode::EXCLUDE_SRC>(
            noc, noc_start_x, noc_start_y, noc_end_x, noc_end_y, num_dests);
        noc.async_write_barrier();

        // Wait for all data to arrive
        // Block until all expected data (Wt_final tiles) has been received from
        // the local cores. The sender semaphore is incremented by each sending core.
        sender_sem.wait(Wt_final);

        // Commit received data
        // Mark the received data as available to the final compute kernel
        final_values_cb.push_back(Wt_final);
        final_indices_cb.push_back(Wt_final);
    }  // i loop

    // Ensure all NoC operations complete before kernel termination
    noc.async_write_barrier();
}
