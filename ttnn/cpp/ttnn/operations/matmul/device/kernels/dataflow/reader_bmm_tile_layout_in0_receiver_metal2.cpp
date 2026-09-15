// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "experimental/kernel_args.h"

// This is the Metal 2.0 fork of reader_bmm_tile_layout_in0_receiver.cpp, which still sits beside it
// and still serves the matmul factories that have not been ported. Changes to either copy should be
// evaluated for the other until the last legacy consumer migrates and the legacy copy is retired.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.

void kernel_main() {
    // in0 mcast args
    const uint32_t in0_mcast_sender_noc_x = get_arg(args::in0_mcast_sender_noc_x);
    const uint32_t in0_mcast_sender_noc_y = get_arg(args::in0_mcast_sender_noc_y);

    // COMPILE TIME ARGS
    // in0 block args
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    // in0/in1 common args
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    // batch args
    constexpr auto batch = get_arg(args::batch);
    // sparsity args
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = static_cast<bool>(get_arg(args::get_batch_from_reader));

    const Noc noc;
    // dfb::in0 is the multicast destination: the sender writes each in0 block straight into this
    // buffer's write pointer on every receiver, so this kernel only drives the FIFO around it.
    DataflowBuffer dfb_in0(dfb::in0);
    Semaphore sender_sem(sem::in0_mcast_sender);
    Semaphore receiver_sem(sem::in0_mcast_receiver);

    for (uint32_t b = 0; b < batch; ++b) {
        if constexpr (get_batch_from_reader) {
            // This means we have unstructured sparsity.
            // The compute kernel needs to be made aware whether this batch is valid or not.
            // We do this by passing the value to the compute kernel via mailbox.
            // But first, lets wait for the sparsity data to be multicast to us.
            // Set in0 semaphore value to INVALID
            receiver_sem.set(INVALID);
            // Atomic increment source core counter
            sender_sem.up(noc, in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, 1);
            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            receiver_sem.wait_min(VALID);

            // The sender multicasts VALID for a live batch and IGNORE_BATCH (a higher value, which is
            // why the wait above is wait_min) for a skipped one, so the exact value distinguishes them.
            const auto is_batch_valid = receiver_sem.value() == VALID;

            // We need to pass the value to compute cores regardless of the value of is_batch_valid
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, static_cast<uint32_t>(is_batch_valid));
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(is_batch_valid));
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, static_cast<uint32_t>(is_batch_valid));

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

                    // Set in0 semaphore value to INVALID
                    receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    sender_sem.up(noc, in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, 1);

                    // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    receiver_sem.wait(VALID);

                    dfb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }

    // Drain the mcast-ready atomics (sender_sem.up) before returning, so no non-posted atomic is
    // in flight at kernel exit. Matches the dram_sharded / ring_all_gather receivers.
    noc.async_atomic_barrier();
}
