// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

void kernel_main() {
    // Compile time args
    constexpr dataflow_kernel_lib::McastArgs<0, 0> readiness_mcast_args;
    constexpr uint32_t post_mcast_ct_offset = readiness_mcast_args.next_compile_time_args_offset();
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(post_mcast_ct_offset);  // Arrival counter
    constexpr uint32_t Ht = get_compile_time_arg_val(post_mcast_ct_offset + 1);
    constexpr uint32_t Wt_final = get_compile_time_arg_val(post_mcast_ct_offset + 2);
    constexpr uint32_t final_values_dfb_index = get_compile_time_arg_val(post_mcast_ct_offset + 3);
    constexpr uint32_t final_indices_dfb_index = get_compile_time_arg_val(post_mcast_ct_offset + 4);

    Noc noc;
    auto readiness_pipe = readiness_mcast_args.sender(noc);
    Semaphore<> sender_sem(sender_sem_id);
    DataflowBuffer final_values_dfb(final_values_dfb_index);
    DataflowBuffer final_indices_dfb(final_indices_dfb_index);

    // Collect local TopK results from all cores
    for (uint32_t i = 0; i < Ht; ++i) {  // Process each height row
        // Reserve space for incoming data from all local cores
        final_values_dfb.reserve_back(Wt_final);   // Space for all TopK values
        final_indices_dfb.reserve_back(Wt_final);  // Space for all TopK indices

        // The arrival counter remains operation-owned and is reset only after the prior round's
        // exact wait completed. Readiness is a helper-owned monotone Counter, so it is never reset.
        sender_sem.set(INVALID);
        readiness_pipe.send_signal();

        // Wait for all data to arrive
        // Block until all expected data (Wt_final tiles) has been received from
        // the local cores. The sender semaphore is incremented by each sending core.
        sender_sem.wait(Wt_final);

        // Commit received data
        // Mark the received data as available to the final compute kernel
        final_values_dfb.push_back(Wt_final);
        final_indices_dfb.push_back(Wt_final);
    }  // i loop

    // Ensure all NoC operations complete before kernel termination
    noc.async_write_barrier();
}
