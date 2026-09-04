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
    constexpr uint32_t arrival_counter_sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt_final = get_compile_time_arg_val(2);
    constexpr uint32_t final_values_dfb_index = get_compile_time_arg_val(3);
    constexpr uint32_t final_indices_dfb_index = get_compile_time_arg_val(4);
    constexpr dataflow_kernel_lib::McastArgs<5, 0> readiness_mcast_args;

    Noc noc;
    auto readiness_pipe = readiness_mcast_args.sender(noc);
    Semaphore<> arrival_counter_sem(arrival_counter_sem_id);
    DataflowBuffer final_values_dfb(final_values_dfb_index);
    DataflowBuffer final_indices_dfb(final_indices_dfb_index);

    // Collect local TopK results from all cores
    for (uint32_t i = 0; i < Ht; ++i) {  // Process each height row
        // Reserve space for incoming data from all local cores
        final_values_dfb.reserve_back(Wt_final);   // Space for all TopK values
        final_indices_dfb.reserve_back(Wt_final);  // Space for all TopK indices

        // The arrival counter remains operation-owned and is reset only after the prior round's
        // exact wait completed. Readiness is a helper-owned monotone Counter, so it is never reset.
        arrival_counter_sem.set(INVALID);
        readiness_pipe.send_signal();

        // Wait for all data to arrive
        // Block until all expected data (Wt_final tiles) has been received from
        // the local cores. The arrival counter is incremented by each sending core.
        arrival_counter_sem.wait(Wt_final);

        // Commit received data
        // Mark the received data as available to the final compute kernel
        final_values_dfb.push_back(Wt_final);
        final_indices_dfb.push_back(Wt_final);
    }  // i loop

    // Ensure all NoC operations complete before kernel termination
    noc.async_write_barrier();
}
