// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// Each write is drained before its slot is popped: the compute's next pack could otherwise overwrite the slot
// while the NoC is still reading it.
void send_tiles(
    const Noc& noc,
    DataflowBuffer& src_dfb,
    std::uint32_t Kt,
    std::uint32_t tile_bytes,
    std::uint32_t noc_x,
    std::uint32_t noc_y,
    std::uint32_t dst_base) {
    const UnicastEndpoint remote;
    for (std::uint32_t i = 0; i < Kt; ++i) {
        src_dfb.wait_front(1);
        noc.async_write(
            src_dfb,
            remote,
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = noc_x, .noc_y = noc_y, .addr = dst_base + (i * tile_bytes)});
        noc.async_write_barrier();
        src_dfb.pop_front(1);
    }
}

void kernel_main() {
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
    constexpr std::uint32_t landing_values_dfb_index = get_compile_time_arg_val(11);   // Tree-merge landing (values)
    constexpr std::uint32_t landing_indices_dfb_index = get_compile_time_arg_val(12);  // Tree-merge landing (indices)
    constexpr std::uint32_t credit_sem_id = get_compile_time_arg_val(13);              // Parent freed its landing slot
    constexpr std::uint32_t data_sem_id = get_compile_time_arg_val(14);                // Child landed its tiles
    constexpr std::uint32_t tree_rounds = get_compile_time_arg_val(15);                // Tree merge rounds

    // Runtime args
    const std::uint32_t final_slot = get_arg_val<std::uint32_t>(0);       // Slot in the final core's gather buffer
    const std::uint32_t num_recv_rounds = get_arg_val<std::uint32_t>(1);  // Rounds in which this core receives
    const bool sends_to_final = get_arg_val<std::uint32_t>(2) == 1;       // Survivor of the tree
    const std::uint32_t dest_noc_x = get_arg_val<std::uint32_t>(3);       // Parent (or final core) coordinates
    const std::uint32_t dest_noc_y = get_arg_val<std::uint32_t>(4);
    const std::uint32_t self_noc_x = get_arg_val<std::uint32_t>(5);  // Own coordinates for the landing-slot copy
    const std::uint32_t self_noc_y = get_arg_val<std::uint32_t>(6);
    constexpr std::uint32_t child_coords_arg_base = 7;  // (x, y) of the round r child at args 7 + 2r, 8 + 2r

    const Noc noc;
    Semaphore<> receiver_sem(receiver_sem_id);
    Semaphore<> sender_sem(sender_sem_id);
    Semaphore<> credit_sem(credit_sem_id);
    Semaphore<> data_sem(data_sem_id);
    DataflowBuffer values_dfb(values_dfb_index);
    DataflowBuffer landing_values_dfb(landing_values_dfb_index);
    const DataflowBuffer final_values_dfb(final_values_dfb_index);

    // Memory transfer configuration
    const std::uint32_t tile_bytes_values = values_dfb.get_entry_size();

    // The landing CB is one 2*Kt slot cycled whole, so its base is the same address on every local core.
    const std::uint32_t landing_values_base = landing_values_dfb.get_write_ptr();

    // Base address in the final core's L1 with the offset for this core's contribution
    const std::uint32_t final_values_base = final_values_dfb.get_write_ptr() + final_slot * tile_bytes_values * Kt;

#if !defined(TOPK_FUSED_STABLE_KEYS)
    // Fused-key mode has no separate index stream (and the index CBs may not exist on this core).
    DataflowBuffer indices_dfb(output_ind_dfb_index);
    DataflowBuffer landing_indices_dfb(landing_indices_dfb_index);
    const DataflowBuffer final_indices_dfb(final_indices_dfb_index);
    const std::uint32_t tile_bytes_ind = indices_dfb.get_entry_size();
    const std::uint32_t landing_indices_base = landing_indices_dfb.get_write_ptr();
    const std::uint32_t final_indices_base = final_indices_dfb.get_write_ptr() + final_slot * tile_bytes_ind * Kt;
#endif

    std::uint32_t landed = 0;  // Partner deliveries so far; data_sem counts them monotonically across rows
    for (std::uint32_t j = 0; j < Ht; ++j) {  // For each height row
        // Tree merge rounds this core receives in: land own tiles and the round r child's tiles in one slot.
        for (std::uint32_t r = 0; r < num_recv_rounds; ++r) {
            const std::uint32_t child_noc_x = get_arg_val<std::uint32_t>(child_coords_arg_base + 2 * r);
            const std::uint32_t child_noc_y = get_arg_val<std::uint32_t>(child_coords_arg_base + 2 * r + 1);

            landing_values_dfb.reserve_back(2 * Kt);
#if !defined(TOPK_FUSED_STABLE_KEYS)
            landing_indices_dfb.reserve_back(2 * Kt);
#endif
            // The slot is free: let the child write its half while our own half is copied in.
            credit_sem.up(noc, child_noc_x, child_noc_y, 1);

            send_tiles(noc, values_dfb, Kt, tile_bytes_values, self_noc_x, self_noc_y, landing_values_base);
#if !defined(TOPK_FUSED_STABLE_KEYS)
            send_tiles(noc, indices_dfb, Kt, tile_bytes_ind, self_noc_x, self_noc_y, landing_indices_base);
#endif

            // The child increments data_sem only after its writes are drained, so the slot is complete here.
            ++landed;
            data_sem.wait_min(landed);
            landing_values_dfb.push_back(2 * Kt);
#if !defined(TOPK_FUSED_STABLE_KEYS)
            landing_indices_dfb.push_back(2 * Kt);
#endif
        }

        // Survivors go to the final core, losers into the second half of the parent's landing slot.
        std::uint32_t dest_values_base = landing_values_base + Kt * tile_bytes_values;
        if (sends_to_final) {
            receiver_sem.wait(VALID);
            dest_values_base = final_values_base;
        } else {
            credit_sem.wait_min(j + 1);
        }
        send_tiles(noc, values_dfb, Kt, tile_bytes_values, dest_noc_x, dest_noc_y, dest_values_base);
#if !defined(TOPK_FUSED_STABLE_KEYS)
        const std::uint32_t dest_indices_base =
            sends_to_final ? final_indices_base : landing_indices_base + Kt * tile_bytes_ind;
        send_tiles(noc, indices_dfb, Kt, tile_bytes_ind, dest_noc_x, dest_noc_y, dest_indices_base);
#endif

        // All per-tile writes were drained before their slots were popped above.
        if (sends_to_final) {
            // Signal completion: increment sender semaphore by Kt (number of tiles sent)
            sender_sem.up(noc, dest_noc_x, dest_noc_y, Kt);
            noc.async_atomic_barrier();

            // Reset receiver semaphore to prepare for next round
            receiver_sem.set(INVALID);
        } else {
            data_sem.up(noc, dest_noc_x, dest_noc_y, 1);
            noc.async_atomic_barrier();
        }
    }  // j loop

    // Ensure all atomic operations complete before kernel termination
    noc.async_atomic_barrier();
}
