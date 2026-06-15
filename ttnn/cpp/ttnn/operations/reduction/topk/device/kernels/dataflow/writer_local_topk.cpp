// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    // Runtime args
    const uint32_t start_wt = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(0);                // Final core readiness signal
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(1);                  // Local core completion signal
    constexpr uint32_t noc_final_x = get_compile_time_arg_val(2);                    // Final core X coordinate
    constexpr uint32_t noc_final_y = get_compile_time_arg_val(3);                    // Final core Y coordinate
    constexpr uint32_t Ht = get_compile_time_arg_val(4);                             // Height tiles to process
    constexpr uint32_t K = get_compile_time_arg_val(5);                              // TopK value
    constexpr uint32_t Kt = get_compile_time_arg_val(6);                             // TopK in tile units (ceil(K/32))
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(7);                // Local TopK values output
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(8);            // Local TopK indices output
    constexpr uint32_t final_values_cb_index = get_compile_time_arg_val(9);          // Final aggregation values buffer
    constexpr uint32_t final_indices_cb_index = get_compile_time_arg_val(10);        // Final aggregation indices buffer

    // Constants
    constexpr uint32_t onetile = 1;

    Noc noc;
    Semaphore<> receiver_sem(receiver_sem_id);
    Semaphore<> sender_sem(sender_sem_id);
    UnicastEndpoint remote;
    CircularBuffer values_cb(values_cb_index);
    CircularBuffer indices_cb(output_ind_cb_index);
    CircularBuffer final_values_cb(final_values_cb_index);
    CircularBuffer final_indices_cb(final_indices_cb_index);

    // Memory transfer configuration
    const uint32_t tile_bytes_values = values_cb.get_tile_size();
    const uint32_t tile_bytes_ind = indices_cb.get_tile_size();

    // Calculate target addresses in final core's L1 memory
    const uint32_t final_values_cb_addr = final_values_cb.get_write_ptr();
    const uint32_t final_indices_cb_addr = final_indices_cb.get_write_ptr();

    // Base addresses in final core's L1 memory with offset for this core's contribution
    const uint32_t final_values_base = final_values_cb_addr + start_wt * tile_bytes_values * Kt;
    const uint32_t final_indices_base = final_indices_cb_addr + start_wt * tile_bytes_ind * Kt;

    // Send local TopK results to final core
    for (uint32_t j = 0; j < Ht; ++j) {  // For each height row
        // Wait for permission to send
        // Block until the final core signals readiness to receive data
        receiver_sem.wait(VALID);

        // Transfer local TopK results
        // Send Kt tiles of locally computed TopK values to final core
        for (uint32_t i = 0; i < Kt; ++i) {
            values_cb.wait_front(onetile);  // Wait for compute kernel output

            // Direct NoC write to final core's aggregation buffer
            noc.async_write(
                values_cb,
                remote,
                tile_bytes_values,
                {.offset_bytes = 0},
                {.noc_x = noc_final_x, .noc_y = noc_final_y, .addr = final_values_base + i * tile_bytes_values});
            values_cb.pop_front(onetile);
        }  // i loop

        // Transfer local TopK indices
        // Send Kt tiles of corresponding TopK indices to final core
        for (uint32_t i = 0; i < Kt; ++i) {
            indices_cb.wait_front(onetile);  // Wait for compute kernel output

            // Direct NoC write to final core's aggregation buffer
            noc.async_write(
                indices_cb,
                remote,
                tile_bytes_ind,
                {.offset_bytes = 0},
                {.noc_x = noc_final_x, .noc_y = noc_final_y, .addr = final_indices_base + i * tile_bytes_ind});
            indices_cb.pop_front(onetile);
        }  // i loop

        // Complete all pending NoC writes
        noc.async_write_barrier();  // Ensure all data is transmitted before signaling

        // Signal completion: increment sender semaphore by Kt (number of tiles sent)
        sender_sem.up(noc, noc_final_x, noc_final_y, Kt);
        noc.async_atomic_barrier();

        // Reset receiver semaphore to prepare for next round
        receiver_sem.set(INVALID);
    }  // j loop

    // Ensure all atomic operations complete before kernel termination
    noc.async_atomic_barrier();
}
