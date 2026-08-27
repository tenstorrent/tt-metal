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
// Compile-time args (named):
//   tiles_per_block
//   cb_in, cb_out            -- Metal 2.0 only; the legacy path hardcodes them below
//
// Runtime args, identical on all three kernels. Positionally on the legacy path -- the in
// and out base addresses, then this core's output block index -- and by NAME on Metal 2.0,
// where only out_block survives because the addresses ride along with the tensor bindings.
// The named form is hazard D17 closed: a missing or misspelled name is an error from metal,
// not a garbage read, so the runtime-arg sentinel has nothing left to guard.
//
// Defines:
//   MC_ROW_W        cores in the row
//   MC_DM_THREAD    which data-movement thread broadcasts (default 0)
//   MC_BARRIER      if set, barrier twice between the broadcast and the store
//
// The handshake semaphores are not named here: unified_program() reserves two per
// data-movement thread and passes their base in as a define.
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
#if defined(TT_UNIFIED_METAL2)
#include "experimental/kernel_args.h"
#endif

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t tiles_per_block = get_named_compile_time_arg_val("tiles_per_block");

#if defined(TT_UNIFIED_METAL2)
    constexpr uint32_t kCbIn = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t kCbOut = get_named_compile_time_arg_val("cb_out");
    const uint32_t out_block = get_arg(args::out_block);
#else
    constexpr uint32_t kCbIn = 0;
    constexpr uint32_t kCbOut = 16;
    const uint32_t out_block = get_arg_val<uint32_t>(2);
    u::check_runtime_args<3>();
#endif

    u::compute_init(kCbIn, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in_storage(kCbIn);
    u::Storage<Block1D> out_storage(kCbOut);

#if defined(TT_UNIFIED_METAL2) && defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    // The compute projection binds every buffer, so it is where the harness's predicted
    // slots can all be checked; see unified_kernels/unary.cpp for the argument.
    static_assert(
        kCbIn == static_cast<uint32_t>(dfb::in) && kCbOut == static_cast<uint32_t>(dfb::out),
        "the harness's predicted dataflow buffer slots do not match the ones the host assigned");
#endif

#if defined(TT_UNIFIED_METAL2)
    const auto in = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);
#else
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(1));
#endif

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
