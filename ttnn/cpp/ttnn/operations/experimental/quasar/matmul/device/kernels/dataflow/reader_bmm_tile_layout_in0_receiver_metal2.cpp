// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in0_receiver.cpp.
//
// Algorithm body is byte-for-byte identical to the legacy kernel; only the host-binding surface is
// converted to the Metal 2.0 form:
//   - positional get_compile_time_arg_val(N) -> named get_arg(args::name)
//   - positional get_arg_val<uint32_t>(i)    -> named get_arg(args::name)
//   - named CB-index CTA ("cb_in0")          -> dfb::cb_in0 token
//   - Semaphore<>(get_compile_time_arg_val(id)) -> Semaphore(sem::name) bound via SemaphoreBinding
//
// See METAL2_MCAST_PLAN.md and the merged reuse_optimized reader fork for the conventions.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // in0 mcast args
    const uint32_t in0_mcast_sender_noc_x = get_arg(args::in0_mcast_sender_noc_x);
    const uint32_t in0_mcast_sender_noc_y = get_arg(args::in0_mcast_sender_noc_y);

    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    // batch args
    constexpr uint32_t batch = get_arg(args::batch);
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_arg(args::get_batch_from_reader);

    constexpr uint32_t cb_id_in0 = dfb::cb_in0;

    Noc noc;
    DataflowBuffer cb_in0(dfb::cb_in0);
    Semaphore sender_sem(sem::in0_sender);
    Semaphore receiver_sem(sem::in0_receiver);

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_sem.get_l1_addr());

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

            const auto is_batch_valid = *in0_mcast_receiver_semaphore_addr_ptr == VALID;

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
                    cb_in0.reserve_back(in0_block_num_tiles);

                    // Set in0 semaphore value to INVALID
                    receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    sender_sem.up(noc, in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, 1);

                    // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    receiver_sem.wait(VALID);

                    cb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }
}
