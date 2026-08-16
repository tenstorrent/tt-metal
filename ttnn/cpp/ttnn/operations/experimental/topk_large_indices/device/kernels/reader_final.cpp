// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Column-parallel final-core gather coordinator (modeled on the reduction
// topk multi-core reader_final): per row, reserve space for every local
// core's sequence in the gathered CBs, then multicast the receiver semaphore
// to the local-core rectangle so all slices send in parallel, wait for all
// num_slices sender increments, and publish the gathered tiles to the final
// compute kernel.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(1);
    constexpr uint32_t mcast_start_x = get_compile_time_arg_val(2);
    constexpr uint32_t mcast_start_y = get_compile_time_arg_val(3);
    constexpr uint32_t mcast_end_x = get_compile_time_arg_val(4);
    constexpr uint32_t mcast_end_y = get_compile_time_arg_val(5);
    constexpr uint32_t num_slices = get_compile_time_arg_val(6);
    constexpr uint32_t tiles_per_sequence = get_compile_time_arg_val(7);
    constexpr uint32_t gathered_values_cb = get_compile_time_arg_val(8);
    constexpr uint32_t gathered_indices_cb = get_compile_time_arg_val(9);

    constexpr uint32_t gathered_tiles = num_slices * tiles_per_sequence;

    Noc noc;
    Semaphore<> receiver_sem(receiver_sem_id);
    Semaphore<> sender_sem(sender_sem_id);
    CircularBuffer gathered_values_obj(gathered_values_cb);
    CircularBuffer gathered_indices_obj(gathered_indices_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Reserve space for every slice's sequence before inviting senders.
        gathered_values_obj.reserve_back(gathered_tiles);
        gathered_indices_obj.reserve_back(gathered_tiles);

        sender_sem.set(INVALID);
        receiver_sem.set(VALID);
        receiver_sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_slices);
        noc.async_write_barrier();

        // Each local writer bumps the sender semaphore once per row.
        sender_sem.wait(num_slices);

        gathered_values_obj.push_back(gathered_tiles);
        gathered_indices_obj.push_back(gathered_tiles);
    }

    noc.async_write_barrier();
}
