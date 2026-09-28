// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

void kernel_main() {
    // Compile time args
    constexpr std::uint32_t arrival_counter_sem_id = get_compile_time_arg_val(0);
    constexpr std::uint32_t Ht = get_compile_time_arg_val(1);
    constexpr std::uint32_t Wt_final = get_compile_time_arg_val(2);
    constexpr std::uint32_t final_values_dfb_index = get_compile_time_arg_val(3);
    constexpr std::uint32_t final_indices_dfb_index = get_compile_time_arg_val(4);
    constexpr dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("readiness_mcast_ct_offset"),
        get_named_compile_time_arg_val("readiness_mcast_rt_offset")>
        readiness_mcast_args;

    const Noc noc;
    auto readiness_pipe = readiness_mcast_args.sender(noc);
    Semaphore<> arrival_counter_sem(arrival_counter_sem_id);
    DataflowBuffer final_values_dfb(final_values_dfb_index);
#if !defined(TOPK_FUSED_STABLE_KEYS)
    // Fused-key mode: the indices ride inside the packed value tiles; no index gather CB.
    DataflowBuffer final_indices_dfb(final_indices_dfb_index);
#endif

    // Collect local TopK results from all cores
    for (std::uint32_t i = 0; i < Ht; ++i) {  // Process each height row
        // Reserve space for incoming data from all local cores
        final_values_dfb.reserve_back(Wt_final);  // Space for all TopK values (packed keys when fused)
#if !defined(TOPK_FUSED_STABLE_KEYS)
        final_indices_dfb.reserve_back(Wt_final);  // Space for all TopK indices
#endif

        // The arrival counter remains operation-owned and is reset only after the prior round's
        // exact wait completed. The multicast helper signals that the destination buffers are ready.
        arrival_counter_sem.set(INVALID);
        readiness_pipe.send_signal();

        // Wait for all data to arrive
        // Block until all expected data (Wt_final tiles) has been received from
        // the local cores. The arrival counter is incremented by each sending core.
        arrival_counter_sem.wait(Wt_final);

        // Commit received data
        // Mark the received data as available to the final compute kernel
        final_values_dfb.push_back(Wt_final);
#if !defined(TOPK_FUSED_STABLE_KEYS)
        final_indices_dfb.push_back(Wt_final);
#endif
    }  // i loop

    // Ensure all NoC operations complete before kernel termination
    noc.async_write_barrier();
}
