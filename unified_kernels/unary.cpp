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
// Compile-time args, all named: num_blocks, tiles_per_block, and a dfb_<name> per buffer.
// No runtime args at all -- the tensors are BOUND, so their addresses
// ride along with the accessors rather than being passed and counted.
//
// Defines: one of UN_SQRT, UN_RSQRT, UN_EXP, UN_CHAIN; recip is the default.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

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
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);

    constexpr uint32_t kDfbIn = get_arg(args::dfb_in);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);

    u::compute_init(kDfbIn, kDfbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in_storage(kDfbIn);
    u::Storage<Block1D> out_storage(kDfbOut);

    const auto in = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        u::ComputeBlock a = u::noc_load<0>(in_storage, in, b).wait();
        u::Block result = out_storage.store(UN_APPLY(a));
        u::noc_store<1>(std::move(result), out, b);
    }
}
