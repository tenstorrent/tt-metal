// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

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

    // mcast_pipe: this aggregator broadcasts a readiness flag to the rectangle of local-topk sender
    // cores (a flag-only control signal, R2), then waits a fan-in counter (sender_sem) for all Wt_final
    // tiles to land. Only the readiness broadcast is a Pipe op (send_signal); the fan-in counter wait
    // is a separate multi-producer channel the (single-sender) Pipe does not own (INV9), kept raw.
    // data_ready=receiver_sem (the flag we broadcast); consumed unused on this control path.
    dataflow_kernel_lib::Pipe<> ready_pipe(
        noc,
        dataflow_kernel_lib::McastRect{noc_start_x, noc_start_y, noc_end_x, noc_end_y},  // area() = num_dests
        num_dests,  // active-core count (send_signal does not consult it; kept meaningful)
        receiver_sem,
        sender_sem);

    // Collect local TopK results from all cores
    for (uint32_t i = 0; i < Ht; ++i) {  // Process each height row
        // Reserve space for incoming data from all local cores
        final_values_cb.reserve_back(Wt_final);   // Space for all TopK values
        final_indices_cb.reserve_back(Wt_final);  // Space for all TopK indices

        // Initialize semaphores for this height row
        // Reset synchronization state for this height row
        sender_sem.set(INVALID);  // Mark data as not yet sent

        // Coordinate multicast reception
        // mcast_pipe: broadcast the readiness flag (VALID) to all local-topk sender cores. send_signal
        // sets the local cell + mcasts it to the rect + fences (flush). This replaces the explicit
        // set(VALID) + set_multicast + async_write_barrier readiness broadcast.
        ready_pipe.send_signal(VALID);

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
