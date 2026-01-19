// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    // in0 mcast args
    const uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // in0 block args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    // in0/in1 common args
    constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(3);
    // in0 mcast args
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(4));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    // batch args
    constexpr uint32_t batch = get_compile_time_arg_val(6);
    // sparsity args
    // This boolean is set when the number of batches is only known at runtime, typically based on a sparsity tensor.
    constexpr bool get_batch_from_reader = (bool)get_compile_time_arg_val(7);

    constexpr uint32_t cb_id_in0 = 0;

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);

    const uint64_t in0_mcast_sender_semaphore_noc_addr =
        get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);

    for (uint32_t b = 0; b < batch; ++b) {
        if constexpr (get_batch_from_reader) {
            // This means we have unstructured sparsity.
            // The compute kernel needs to be made aware whether this batch is valid or not.
            // We do this by passing the value to the compute kernel via mailbox.
            // But first, lets wait for the sparsity data to be multicast to us.
            // Set in0 semaphore value to INVALID
            noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);
            // Atomic increment source core counter
            noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);
            // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
            noc_semaphore_wait_min(in0_mcast_receiver_semaphore_addr_ptr, VALID);

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
                    cb_reserve_back(cb_id_in0, in0_block_num_tiles);

                    // Set in0 semaphore value to INVALID
                    noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

                    // Atomic increment source core counter
                    noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);

                    // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
                    noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);

                    cb_push_back(cb_id_in0, in0_block_num_tiles);
                }
            }
        }
    }
}
