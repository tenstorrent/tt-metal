// SPDX-License-Identifier: Apache-2.0
//
// Does sharing a handshake pair break when the two collectives have the SAME rectangle?
//
// Hazard 13b is proven for DIFFERENT rectangles: the ready counter counts without
// identifying, so a sender gets released by a core outside its rectangle. The open
// question is whether rectangle equality is what makes sharing safe, or whether two
// transactions on one semaphore is unsafe on its own.
//
// This does the cleanest version of the question. One operand, broadcast TWICE over the
// same rectangle into two different buffers, on the same pair. Same sender, same
// receivers, same extent -- only the buffer differs. Both outputs must equal the input,
// so a protocol failure shows up as a hang or as one of them being wrong, and neither
// can be confused with bad arithmetic: there is no arithmetic.
//
// A single round proves nothing, which is the lesson from hazard 13b: twelve clean trials
// there were all taken without the skew in the place that could corrupt. So this loops, and
// MS_SKEW holds a subset of receivers' buffers LIVE while the rest race into the next
// round -- the same three conditions that exposed 13b, minus the differing rectangle. If
// rectangle equality is what makes sharing safe, this stays clean under skew; if two
// transactions on one semaphore is unsafe on its own, this breaks.
//
// Compile-time args, by name:
//   tiles      tiles per block
//   rounds     how many times to repeat both broadcasts
//   grid_h     the rectangle, which is the whole core grid
//   grid_w
//
// Runtime args, named and identical on all three kernels:
//   the sentinel
//
// Defines:
//   MS_SHARE_PAIR   put both collectives on pair 0. Without it they take 0 and 1, which
//                   is the control.
//   MS_SKEW         busy-wait this many iterations on odd-numbered cores AFTER both loads,
//                   so they hold live buffers while even cores run ahead.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t tiles = get_arg(args::tiles);
    constexpr uint32_t rounds = get_arg(args::rounds);
    constexpr uint32_t grid_h = get_arg(args::grid_h);
    constexpr uint32_t grid_w = get_arg(args::grid_w);

    constexpr uint32_t kDfbA = get_arg(args::dfb_a);
    constexpr uint32_t kDfbB = get_arg(args::dfb_b);
    constexpr uint32_t kDfbOut0 = get_arg(args::dfb_out0);
    constexpr uint32_t kDfbOut1 = get_arg(args::dfb_out1);

    const auto in = TensorAccessor(tensor::in);
    const auto out0 = TensorAccessor(tensor::out0);
    const auto out1 = TensorAccessor(tensor::out1);

    using Blk = u::Shape<1, tiles>;

    u::Storage<Blk> a_storage(kDfbA);
    u::Storage<Blk> b_storage(kDfbB);
    u::Storage<Blk> out0_storage(kDfbOut0);
    u::Storage<Blk> out1_storage(kDfbOut1);

    u::compute_init(kDfbA, kDfbOut0);

    const u::LogicalCoord me = u::LogicalCoord::this_core();

    // ONE rectangle: the whole grid, so both collectives have the identical sender at
    // (0, 0) and the identical receiver set. Every core builds the same one.
    const u::LogicalMcast whole{u::LogicalCoord::yx(0, 0), u::Extent::hw(grid_h, grid_w)};

#if defined(MS_SHARE_PAIR)
    constexpr int kPairA = 0;
    constexpr int kPairB = 0;
#else
    constexpr int kPairA = 0;
    constexpr int kPairB = 1;
#endif

    const uint32_t core = me.y * grid_w + me.x;

    for (uint32_t r = 0; r < rounds; ++r) {
        // Round r broadcasts block r, so a corrupted round is visible rather than masked
        // by every round carrying the same bytes.
        u::ComputeBlock a = u::noc_load<0, kPairA>(a_storage, whole, in, r).wait();
        u::ComputeBlock b = u::noc_load<0, kPairB>(b_storage, whole, in, r).wait();

#if defined(MS_SKEW)
        // AFTER the loads, so `a` and `b` are still live -- their pops are at the end of
        // this iteration. Half the receivers therefore hold buffers they have not freed
        // while the other half run into round r+1. This is the placement that mattered for
        // 13b; before the loads it lands after the previous pop and cannot corrupt.
        if ((core & 1u) != 0u) {
            for (volatile uint32_t d = 0; d < MS_SKEW; ++d) {
            }
        }
#endif

        u::noc_store<1>(out0_storage, u::copy(a), out0, core * rounds + r);
        u::noc_store<1>(out1_storage, u::copy(b), out1, core * rounds + r);
    }
}
