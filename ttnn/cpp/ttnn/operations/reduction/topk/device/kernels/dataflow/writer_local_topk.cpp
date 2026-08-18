// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    // Runtime args
    const std::uint32_t start_wt = get_arg_val<std::uint32_t>(0);

    // Compile time args
    constexpr std::uint32_t receiver_sem_id = get_compile_time_arg_val(0);           // Final core readiness signal
    constexpr std::uint32_t sender_sem_id = get_compile_time_arg_val(1);             // Local core completion signal
    constexpr std::uint32_t noc_final_x = get_compile_time_arg_val(2);               // Final core X coordinate
    constexpr std::uint32_t noc_final_y = get_compile_time_arg_val(3);               // Final core Y coordinate
    constexpr std::uint32_t Ht = get_compile_time_arg_val(4);                        // Height tiles to process
    constexpr std::uint32_t K = get_compile_time_arg_val(5);                         // TopK value
    constexpr std::uint32_t Kt = get_compile_time_arg_val(6);                        // TopK in tile units (ceil(K/32))
    constexpr std::uint32_t values_dfb_index = get_compile_time_arg_val(7);          // Local TopK values output
    constexpr std::uint32_t output_ind_dfb_index = get_compile_time_arg_val(8);      // Local TopK indices output
    constexpr std::uint32_t final_values_dfb_index = get_compile_time_arg_val(9);    // Final aggregation values buffer
    constexpr std::uint32_t final_indices_dfb_index = get_compile_time_arg_val(10);  // Final aggregation indices buffer

    // Constants
    constexpr std::uint32_t onetile = 1;

    Noc noc;
    Semaphore<> receiver_sem(receiver_sem_id);
    Semaphore<> sender_sem(sender_sem_id);
    UnicastEndpoint remote;
    DataflowBuffer values_dfb(values_dfb_index);
    DataflowBuffer final_values_dfb(final_values_dfb_index);

    // Memory transfer configuration
    const std::uint32_t tile_bytes_values = values_dfb.get_entry_size();

    // Calculate target addresses in final core's L1 memory
    const std::uint32_t final_values_dfb_addr = final_values_dfb.get_write_ptr();

    // Base addresses in final core's L1 memory with offset for this core's contribution
    const std::uint32_t final_values_base = final_values_dfb_addr + start_wt * tile_bytes_values * Kt;

#if !defined(TOPK_FUSED_STABLE_KEYS)
    // Fused-key mode has no separate index stream (and the index CBs may not exist on this core).
    DataflowBuffer indices_dfb(output_ind_dfb_index);
    DataflowBuffer final_indices_dfb(final_indices_dfb_index);
    const std::uint32_t tile_bytes_ind = indices_dfb.get_entry_size();
    const std::uint32_t final_indices_dfb_addr = final_indices_dfb.get_write_ptr();
    const std::uint32_t final_indices_base = final_indices_dfb_addr + start_wt * tile_bytes_ind * Kt;
#endif

    // Send local TopK results to final core
    for (std::uint32_t j = 0; j < Ht; ++j) {  // For each height row
        // Wait for permission to send
        // Block until the final core signals readiness to receive data
        receiver_sem.wait(VALID);

        // Transfer local TopK results
        // Send Kt tiles of locally computed TopK values to final core
        for (std::uint32_t i = 0; i < Kt; ++i) {
            values_dfb.wait_front(onetile);  // Wait for compute kernel output

            // Direct NoC write to final core's aggregation buffer
            noc.async_write(
                values_dfb,
                remote,
                tile_bytes_values,
                {.offset_bytes = 0},
                {.noc_x = noc_final_x, .noc_y = noc_final_y, .addr = final_values_base + i * tile_bytes_values});
            // Drain the write's source-read before releasing the slot for reuse by the compute
            // producer: cb_pop_front only advances the read pointer, so without this barrier the
            // producer's next pack_tile could overwrite this slot while the NoC write is still
            // reading it (WAR), corrupting the data landed at the final core.
            noc.async_write_barrier();
            values_dfb.pop_front(onetile);
        }  // i loop

#if !defined(TOPK_FUSED_STABLE_KEYS)
        // Transfer local TopK indices
        // Send Kt tiles of corresponding TopK indices to final core
        // (Fused-key mode: the indices ride inside the packed value tiles sent above — there is no
        // separate index stream, and half the NoC transactions disappear. The semaphore count is
        // unchanged: it has always counted only the Kt value tiles.)
        for (std::uint32_t i = 0; i < Kt; ++i) {
            indices_dfb.wait_front(onetile);  // Wait for compute kernel output

            // Direct NoC write to final core's aggregation buffer
            noc.async_write(
                indices_dfb,
                remote,
                tile_bytes_ind,
                {.offset_bytes = 0},
                {.noc_x = noc_final_x, .noc_y = noc_final_y, .addr = final_indices_base + i * tile_bytes_ind});
            noc.async_write_barrier();  // drain before releasing the slot for producer reuse (WAR)
            indices_dfb.pop_front(onetile);
        }  // i loop
#endif

        // All per-tile writes were drained before their slots were popped above.

        // Signal completion: increment sender semaphore by Kt (number of tiles sent)
        sender_sem.up(noc, noc_final_x, noc_final_y, Kt);
        noc.async_atomic_barrier();

        // Reset receiver semaphore to prepare for next round
        receiver_sem.set(INVALID);
    }  // j loop

    // Ensure all atomic operations complete before kernel termination
    noc.async_atomic_barrier();
}
