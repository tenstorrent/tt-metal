// SPDX-License-Identifier: Apache-2.0
//
// The custom_compute escape hatch: a pass written against the raw compute API, on
// buffers the unified model still owns.
//
// What it computes is a - b, which is chosen for what it CHECKS rather than for being
// interesting. Subtraction is not commutative, so the routine receiving the two circular
// buffer ids in the wrong order is a wrong answer rather than the same answer -- which is
// the one property of the hatch that a commutative op could not test.
//
// Everything the routine does here is deliberately hand-rolled: the reserve, the DST
// bracketing, the pack, the push. That is the point. The harness waits the two input
// blocks and pops them at the end of the scope, and nothing else.
//
// Compile-time args, by name:
//   tiles      how many tiles the block holds
//
// Runtime args, named and identical on all three kernels:
//   the sentinel

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t tiles = get_arg(args::tiles);

    constexpr uint32_t kDfbA = get_arg(args::dfb_a);
    constexpr uint32_t kDfbB = get_arg(args::dfb_b);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);

    const auto a_acc = TensorAccessor(tensor::a);
    const auto b_acc = TensorAccessor(tensor::b);
    const auto out = TensorAccessor(tensor::out);

    using Blk = u::Shape<1, tiles>;

    u::Storage<Blk> a_storage(kDfbA);
    u::Storage<Blk> b_storage(kDfbB);
    u::Storage<Blk> out_storage(kDfbOut);

    u::compute_init(kDfbA, kDfbOut);

    u::ComputeBlock a = u::noc_load<0>(a_storage, a_acc, 0).wait();
    u::ComputeBlock b = u::noc_load<0>(b_storage, b_acc, 0).wait();

    u::custom_compute(a, b, [&](uint32_t a_dfb, uint32_t b_dfb) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_reserve_back(kDfbOut, tiles);
        ckernel::sub_init(a_dfb, b_dfb);
        for (uint32_t t = 0; t < tiles; ++t) {
            ckernel::tile_regs_acquire();
            ckernel::sub_tiles(a_dfb, b_dfb, t, t, 0);
            ckernel::tile_regs_commit();
            ckernel::tile_regs_wait();
            ckernel::pack_tile(0, kDfbOut);
            ckernel::tile_regs_release();
        }
        cb_push_back(kDfbOut, tiles);
#endif
    });

    // The routine pushed the pages, so this only names them. A data-movement thread
    // drains it exactly as it would a Block from Storage::store.
    u::noc_store<1>(u::Block<Blk>{out_storage}, out, 0);
}
