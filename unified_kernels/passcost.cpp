// SPDX-License-Identifier: Apache-2.0
//
// A measurement kernel, not a real op: out = in, through PASSES identity passes.
//
// Every pass is `copy`, the cheapest possible expression, and each lands in its own
// scratch CB.  So the slope of runtime in PASSES is the cost of one L1 round trip of
// `tiles_per_block` tiles plus its CB handshake, with the math held at zero.  That is
// the number that says what fusing two passes into one is worth.
//
// Compile-time args:
//   0            tiles_per_block
//   1..          TensorAccessorArgs for in, then out
//
// Runtime args: 0 = in base address, 1 = out base address.
//
// Define PASSES (1..8).

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;

#ifndef PASSES
#define PASSES 1
#endif

void kernel_main() {
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(0);

    constexpr auto in_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    u::compute_init(kCbIn, kCbOut);

    using S = u::Shape<1, tiles_per_block>;
    u::Storage<S> in_storage(kCbIn);
    u::Storage<S> out_storage(kCbOut);

    // One scratch CB per intermediate pass, so no pass ever reuses a buffer that a
    // live block still occupies.
    u::Storage<S> s1(1), s2(2), s3(3), s4(4), s5(5), s6(6), s7(7);

    const auto in = TensorAccessor(in_args, in_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    u::ComputeBlock c0 = u::noc_load<1>(in_storage, in, 0).wait();
#if PASSES >= 2
    u::ComputeBlock c1 = s1.store(u::copy(c0));
#endif
#if PASSES >= 3
    u::ComputeBlock c2 = s2.store(u::copy(c1));
#endif
#if PASSES >= 4
    u::ComputeBlock c3 = s3.store(u::copy(c2));
#endif
#if PASSES >= 5
    u::ComputeBlock c4 = s4.store(u::copy(c3));
#endif
#if PASSES >= 6
    u::ComputeBlock c5 = s5.store(u::copy(c4));
#endif
#if PASSES >= 7
    u::ComputeBlock c6 = s6.store(u::copy(c5));
#endif
#if PASSES >= 8
    u::ComputeBlock c7 = s7.store(u::copy(c6));
#endif

    // The last pass writes the output CB, so PASSES passes total.
#if PASSES == 1
    u::noc_store<0>(out_storage.store(u::copy(c0)), out, 0);
#elif PASSES == 2
    u::noc_store<0>(out_storage.store(u::copy(c1)), out, 0);
#elif PASSES == 3
    u::noc_store<0>(out_storage.store(u::copy(c2)), out, 0);
#elif PASSES == 4
    u::noc_store<0>(out_storage.store(u::copy(c3)), out, 0);
#elif PASSES == 5
    u::noc_store<0>(out_storage.store(u::copy(c4)), out, 0);
#elif PASSES == 6
    u::noc_store<0>(out_storage.store(u::copy(c5)), out, 0);
#elif PASSES == 7
    u::noc_store<0>(out_storage.store(u::copy(c6)), out, 0);
#else
    u::noc_store<0>(out_storage.store(u::copy(c7)), out, 0);
#endif
}
