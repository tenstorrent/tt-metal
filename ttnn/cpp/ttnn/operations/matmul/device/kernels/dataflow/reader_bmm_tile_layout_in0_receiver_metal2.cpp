// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_bmm_tile_layout_in0_receiver.cpp, which lives beside it. Factories
// ported to Metal 2.0 bind this fork; the original serves the consumers still on the legacy
// ProgramDescriptor API. Until the last of them migrates and the original is retired, changes to
// either copy likely belong in the other too.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// BATCH_VALID_DFB (Quasar only) gates the batch_valid dataflow buffer, through which this kernel tells
// the compute kernel on its node which batches to skip; WH/BH use the BRISC mailbox for that. It is a
// define rather than a compile-time arg because a binding that the host does not declare produces no
// dfb:: token at all. A factory that sets get_batch_from_reader on Quasar must bind the buffer and set
// the define for each kernel that uses the buffer; this kernel rejects a Quasar build that sets
// get_batch_from_reader without the define.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#ifndef ARCH_QUASAR
// Compute(TRISC)-only headers (pull ckernel_addrmod.h -> ckernel_trisc_id.h, which #errors without
// COMPILE_FOR_TRISC on a DM build); needed only for the batch-valid mailbox handoff. Quasar has no such
// mailbox and uses the batch_valid dataflow buffer instead. Guard out on Quasar.
#include "ckernel.h"
#include "ckernel_defs.h"
#endif
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_metal2.hpp"

void kernel_main() {
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
    // in0 is filled here from the multicast and drained by the compute kernel.
    DataflowBuffer dfb_in0(dfb::in0);
    constexpr auto in0_mcast = MCAST_ARGS(in0_mcast);
    auto receiver = in0_mcast.receiver(noc);
#ifdef BATCH_VALID_DFB
    // One entry per batch to the compute kernel on this node: nonzero when the batch is valid.
    DataflowBuffer dfb_batch_valid(dfb::batch_valid);
#elif defined(ARCH_QUASAR)
    static_assert(!get_batch_from_reader, "get_batch_from_reader on Quasar requires BATCH_VALID_DFB");
#endif

    for (uint32_t b = 0; b < batch; ++b) {
        if constexpr (get_batch_from_reader) {
            // This means we have unstructured sparsity.
            // The compute kernel needs to be made aware whether this batch is valid or not.
            // We do this by passing the value to the compute kernel via mailbox (on Quasar, through the
            // batch_valid dataflow buffer).
            // But first, lets wait for the sparsity data to be multicast to us.
            const auto is_batch_valid = receiver.receive_signal() == VALID;

            // We need to pass the value to compute cores regardless of the value of is_batch_valid
#ifndef ARCH_QUASAR
            // No BRISC->compute mailbox on Quasar; there the value goes through dfb::batch_valid.
            ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, static_cast<uint32_t>(is_batch_valid));
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(is_batch_valid));
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, static_cast<uint32_t>(is_batch_valid));
#elif defined(BATCH_VALID_DFB)
            dfb_batch_valid.reserve_back(1);
            {
                // Releasing the lock makes the store visible in SRAM before push_back hands the entry
                // to the compute kernel.
                const auto batch_valid_lock = dfb_batch_valid.scoped_write_lock(1);
                batch_valid_lock.get_ptr<volatile uint16_t>()[0] = static_cast<uint16_t>(is_batch_valid);
            }
            dfb_batch_valid.push_back(1);
#endif

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

                    receiver.receive();

                    dfb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }

    // Drain multicast ready atomics before returning.
    noc.async_atomic_barrier();
}
