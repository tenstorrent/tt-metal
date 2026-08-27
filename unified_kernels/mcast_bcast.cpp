// SPDX-License-Identifier: Apache-2.0
//
// Multicast broadcast: ONE source, compiled once per baby RISC-V thread, run on
// a row of cores.
//
// The first core of the row reads a block from DRAM and multicasts it to every
// core in the row, including itself. Each core then computes exp(x) over its
// copy and writes the result to its own slice of the output. So every core's
// output slice should be identical -- which is what makes a mis-addressed
// semaphore or a skipped handshake visible.
//
// Compile-time args, all named: tiles_per_block, and a cb_<name> per buffer that TT_U_CB
// reads.
//
// Runtime args, identical on all three kernels: out_block, this core's output slice, read by
// NAME. That is hazard D17 closed -- a missing or misspelled name is an error from metal
// rather than a garbage loop bound, which is the failure that hung this device three times.
//
// Defines:
//   MC_ROW_W        cores in the row
//   MC_DM_THREAD    which data-movement thread broadcasts (default 0)
//   MC_BARRIER      if set, barrier twice between the broadcast and the store
//
// The handshake semaphores are not named here: the harness reserves two per data-movement
// thread and passes their base in as a define, and tt/unified/api.h checks that base and the
// run's contiguity against the ids the host actually assigned.
//
// The broadcast runs on DM thread 0, i.e. NOC 0, and that is a SIMULATOR
// accommodation rather than a property of the model. Metal multicasts in virtual
// coordinates -- NOC_0_X is the identity, mirroring only happens in the
// *_PHYS_COORD variants -- so an ascending rectangle is correct on either NOC on
// hardware. ttsim does not implement coordinate virtualization (see its comment
// in tile.cpp) and compensates by demanding NOC 1 multicast coordinates already
// be mirrored into descending order, which ascending virtual coords cannot
// satisfy. NOC 0 has no such constraint. Set MC_DM_THREAD=1 on real hardware.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t tiles_per_block = get_named_compile_time_arg_val("tiles_per_block");

    constexpr uint32_t kCbIn = TT_U_CB(in);
    constexpr uint32_t kCbOut = TT_U_CB(out);
    const uint32_t out_block = get_arg(args::out_block);

    u::compute_init(kCbIn, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in_storage(kCbIn);
    u::Storage<Block1D> out_storage(kCbOut);

    const auto in = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);

    // The whole row, expressed logically. Core (0,0) of the row is the sender.
    const u::LogicalMcast row{u::LogicalCoord::yx(0, 0), u::Extent::hw(1, MC_ROW_W)};

    u::ComputeBlock x = u::noc_load<MC_DM_THREAD>(in_storage, row, in, 0).wait();
#if defined(MC_BARRIER)
    // Twice, deliberately: back-to-back is the case that fails if the barrier
    // clears its arrival count after releasing rather than before -- a core let go
    // by the first barrier immediately re-arrives for the second, and a late reset
    // erases that arrival and hangs. It also checks the pair is left clean by the
    // multicast that just ran on the same semaphores.
    u::synchronize_cores<MC_DM_THREAD>();
    u::synchronize_cores<MC_DM_THREAD>();
#endif

    u::Block result = out_storage.store(u::exp_(x));
    u::noc_store<0>(std::move(result), out, out_block);
}
