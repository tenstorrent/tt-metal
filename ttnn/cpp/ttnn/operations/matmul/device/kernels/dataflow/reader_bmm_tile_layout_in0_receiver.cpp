// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
void kernel_main() {
    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(3);
    // batch args
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    // sparsity args
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(5);

    constexpr auto in0_mcast_args = dataflow_kernel_lib::McastArgs<6, 0>();

    constexpr uint32_t dfb_id_in0 = get_named_compile_time_arg_val("cb_in0");

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    auto in0_pipe = in0_mcast_args.receiver(noc);

    for (uint32_t b = 0; b < batch; ++b) {
        if constexpr (get_batch_from_reader) {
            // This means we have unstructured sparsity.
            // The compute kernel needs to be made aware whether this batch is valid or not.
            // We do this by passing the value to the compute kernel via mailbox.
            const auto is_batch_valid = in0_pipe.receive_signal() == VALID;

            // We need to pass the value to compute cores regardless of the value of is_batch_valid
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, is_batch_valid);
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, is_batch_valid);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, is_batch_valid);

            // Skip sending the input tensor for this batch as it is not valid.
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
                    // Operand 0
                    dfb_in0.reserve_back(in0_block_num_tiles);

                    in0_pipe.receive();

                    dfb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }

    // Drain the mcast-ready atomics (sender_sem.up) before returning, so no non-posted atomic is
    // in flight at kernel exit. Matches the dram_sharded / ring_all_gather receivers.
    noc.async_atomic_barrier();
}
