// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "topk_local_writer_common.hpp"

void kernel_main() {
    using namespace topk_local_writer;

    // Runtime args
    const std::uint32_t start_wt = get_arg_val<std::uint32_t>(0);

    // Compile time args
    constexpr std::uint32_t noc_final_x = get_compile_time_arg_val(2);  // Final core X coordinate
    constexpr std::uint32_t noc_final_y = get_compile_time_arg_val(3);  // Final core Y coordinate

    const Noc noc;
    Semaphore<> receiver_sem(receiver_sem_id);
    Semaphore<> sender_sem(sender_sem_id);
    DataflowBuffer values_dfb(values_dfb_index);
    const DataflowBuffer final_values_dfb(final_values_dfb_index);

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
        // Block until the final core signals readiness to receive data
        receiver_sem.wait(VALID);

        send_tiles(noc, values_dfb, Kt, tile_bytes_values, noc_final_x, noc_final_y, final_values_base);
#if !defined(TOPK_FUSED_STABLE_KEYS)
        // In fused-key mode the indices ride inside the packed value tiles sent above; the semaphore
        // count is the same since it only counts the Kt value tiles.
        send_tiles(noc, indices_dfb, Kt, tile_bytes_ind, noc_final_x, noc_final_y, final_indices_base);
#endif

        // Signal completion: increment sender semaphore by Kt (number of tiles sent)
        sender_sem.up(noc, noc_final_x, noc_final_y, Kt);
        noc.async_atomic_barrier();

        // Reset receiver semaphore to prepare for next round
        receiver_sem.set(INVALID);
    }  // j loop

    // Ensure all atomic operations complete before kernel termination
    noc.async_atomic_barrier();
}
