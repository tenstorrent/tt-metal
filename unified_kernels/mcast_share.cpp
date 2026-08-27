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
// Runtime args:
//   0..2       in, out0, out1 base addresses
//   3          the sentinel
//
// Defines:
//   MS_SHARE_PAIR   put both collectives on pair 0. Without it they take 0 and 1, which
//                   is the control.
//   MS_SKEW         busy-wait this many iterations on odd-numbered cores AFTER both loads,
//                   so they hold live buffers while even cores run ahead.

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbA = 0;
constexpr uint32_t kCbB = 1;
constexpr uint32_t kCbOut0 = 16;
constexpr uint32_t kCbOut1 = 17;

void kernel_main() {
    constexpr uint32_t tiles = get_named_compile_time_arg_val("tiles");
    constexpr uint32_t rounds = get_named_compile_time_arg_val("rounds");
    constexpr uint32_t grid_h = get_named_compile_time_arg_val("grid_h");
    constexpr uint32_t grid_w = get_named_compile_time_arg_val("grid_w");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out0_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto out1_args = TensorAccessorArgs<out0_args.next_compile_time_args_offset()>();

    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const auto out0 = TensorAccessor(out0_args, get_arg_val<uint32_t>(1));
    const auto out1 = TensorAccessor(out1_args, get_arg_val<uint32_t>(2));
    u::check_runtime_args<3>();

    using Blk = u::Shape<1, tiles>;

    u::Storage<Blk> a_storage(kCbA);
    u::Storage<Blk> b_storage(kCbB);
    u::Storage<Blk> out0_storage(kCbOut0);
    u::Storage<Blk> out1_storage(kCbOut1);

    u::compute_init(kCbA, kCbOut0);

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

        u::noc_store<1>(out0_storage.store(u::copy(a)), out0, core * rounds + r);
        u::noc_store<1>(out1_storage.store(u::copy(b)), out1, core * rounds + r);
    }
}
