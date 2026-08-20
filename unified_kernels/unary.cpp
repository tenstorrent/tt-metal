// SPDX-License-Identifier: Apache-2.0
//
// A unified kernel: ONE source, compiled once per baby RISC-V thread.
//
// Computes   out = <unary>(in)   over `num_blocks` blocks of `tiles_per_block`
// tiles each. Which unary is a define, so one source covers the whole set.
//
// UN_CHAIN is the interesting case: recip(sqrt(x)) is rsqrt(x) by identity, so
// it checks a two-op SFPU chain against a single-op one rather than only against
// torch -- a chain that silently dropped a link would still match torch on the
// op that survived.
//
// Compile-time args:
//   0            num_blocks
//   1            tiles_per_block
//   2..          TensorAccessorArgs for in, then out
//
// Runtime args (identical on all three kernels):
//   0            in base address
//   1            out base address
//
// Defines: one of UN_SQRT, UN_RSQRT, UN_EXP, UN_CHAIN; recip is the default.

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;

#if defined(UN_SQRT)
#define UN_APPLY(x) u::sqrt_(x)
#elif defined(UN_RSQRT)
#define UN_APPLY(x) u::rsqrt(x)
#elif defined(UN_EXP)
#define UN_APPLY(x) u::exp_(x)
#elif defined(UN_CHAIN)
#define UN_APPLY(x) u::recip(u::sqrt_(x))
#else
#define UN_APPLY(x) u::recip(x)
#endif

void kernel_main() {
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(1);

    constexpr auto in_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    u::compute_init(kCbIn, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in_storage(kCbIn);
    u::Storage<Block1D> out_storage(kCbOut);

    const auto in = TensorAccessor(in_args, in_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        u::ComputeBlock a = u::noc_load<1>(in_storage, in, b).wait();
        u::Block result = out_storage.store(UN_APPLY(a));
        u::noc_store<0>(std::move(result), out, b);
    }
}
