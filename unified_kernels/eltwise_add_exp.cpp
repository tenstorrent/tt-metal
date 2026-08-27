// SPDX-License-Identifier: Apache-2.0
//
// A unified kernel: ONE source, compiled once per baby RISC-V thread.
//
// Computes   out = exp(in0 + in1)   over `num_blocks` blocks of
// `tiles_per_block` tiles each.
//
// The host points three KernelDescriptors at this file -- a reader
// (RISCV_1/NCRISC), a writer (RISCV_0/BRISC), and a compute kernel -- with
// identical compile-time and runtime args. No per-thread defines are needed:
// <tt/unified/core> pulls in tt/unified/adaptor_v1.hpp, which derives the projection
// from the defines metal already emits for each build.
//
// What each thread ends up executing:
//
//   NCRISC   cb_reserve(in0) -> noc_read x N -> cb_push(in0)   (and in1)
//   TRISC    cb_wait(in0), cb_wait(in1), cb_reserve(out)
//              per tile: acquire -> copy,copy,add,exp -> commit/wait -> pack
//            cb_push(out), cb_pop(in1), cb_pop(in0)
//   BRISC    cb_wait(out) -> noc_write x N -> cb_pop(out)
//
// Compile-time args, all named, plus a cb_<name> per buffer:
//   num_blocks
//   tiles_per_block
//
// No runtime args: the tensors are bound, so their addresses ride with the accessors.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);

    constexpr uint32_t kCbIn0 = get_arg(args::cb_in0);
    constexpr uint32_t kCbIn1 = get_arg(args::cb_in1);
    constexpr uint32_t kCbOut = get_arg(args::cb_out);

    u::compute_init(kCbIn0, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in0_storage(kCbIn0);
    u::Storage<Block1D> in1_storage(kCbIn1);
    u::Storage<Block1D> out_storage(kCbOut);

    const auto in0 = TensorAccessor(tensor::in0);
    const auto in1 = TensorAccessor(tensor::in1);
    const auto out = TensorAccessor(tensor::out);

    for (uint32_t b = 0; b < num_blocks; ++b) {
#if defined(EA_CUSTOM_LOAD)
        // Same fill, spelled as a custom routine. The lambda's body is compiled
        // on all five projections, so it doubles as the check that the compute
        // projection can see the data-movement intrinsics.
        u::ComputeBlock a =
            u::noc_load<0>(in0_storage, [&](u::L1Pages pages) {
                for (uint32_t p = 0; p < pages.count; ++p) {
                    noc_async_read(in0.get_noc_addr(b * tiles_per_block + p), pages.addr(p), pages.page_bytes);
                }
            }).wait();
        u::ComputeBlock c =
            u::noc_load<0>(in1_storage, [&](u::L1Pages pages) {
                for (uint32_t p = 0; p < pages.count; ++p) {
                    noc_async_read(in1.get_noc_addr(b * tiles_per_block + p), pages.addr(p), pages.page_bytes);
                }
            }).wait();
#else
        // Reader (DM thread 1) fills these; compute waits on them.
        u::ComputeBlock a = u::noc_load<0>(in0_storage, in0, b).wait();
        u::ComputeBlock c = u::noc_load<0>(in1_storage, in1, b).wait();
#endif

        // Compute evaluates the expression; the allocator picks DST slots.
        //   copy a -> dst0, copy c -> dst1, add(dst0,dst1) -> dst0, exp(dst0)
        u::Block result = out_storage.store(u::exp_(a + c));

        // Writer (DM thread 0) drains it.
        u::noc_store<1>(std::move(result), out, b);
    }
}
