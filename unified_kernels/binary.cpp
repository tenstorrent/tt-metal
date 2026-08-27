// SPDX-License-Identifier: Apache-2.0
//
// A unified kernel: ONE source, compiled once per baby RISC-V thread.
//
// Computes   out = in0 <binop> in1   over `num_blocks` blocks of
// `tiles_per_block` tiles each. Which binop is a define, so one source covers
// the whole set.
//
// BN_CHAIN is the case that matters beyond arithmetic: ((a + b) - a) * b / a is
// b*b/a by identity, left-associated, and mixes commutative with
// non-commutative ops. It costs two DST slots however long it gets -- each
// intermediate is consumed immediately -- so it also checks the allocator.
//
// Compile-time args, BY NAME:
//   num_blocks, tiles_per_block
//
// Named rather than positional because a position is a contract with nothing behind it.
// A name that does not exist fails the BUILD -- get_named_compile_time_arg_val walks a
// generated map and falls off the end into __builtin_unreachable(), which in a constexpr
// context is a compile error naming the line and the bad name. (The header's note that
// this "fails with a segfault" is stale for our toolchain: both clang-20 and the riscv
// g++ that builds kernels give a proper diagnostic.)
//
// The accessors stay POSITIONAL and now start at 0, which is the larger half of the win:
// they were at 2, and every scalar added ahead of them used to shift all three.
//
// Runtime args, named and identical on all three kernels:
//   block_begin    first block this core owns
//   block_count    how many it owns
//
// Blocks are the unit of partitioning and need no coordination to split: block b reads
// pages [b*tiles_per_block, +tiles_per_block) of each input and writes the same range of
// the output, so two cores on different blocks never touch the same page. num_blocks stays
// a compile-time arg because it sizes nothing the core walks -- the range does that -- but
// it is what the host divides up.
//
// Defines: one of BN_SUB, BN_MUL, BN_DIV, BN_MAX, BN_CHAIN; add is the default.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

#if defined(BN_SUB)
#define BN_APPLY(a, b) ((a) - (b))
#elif defined(BN_MUL)
#define BN_APPLY(a, b) ((a) * (b))
#elif defined(BN_DIV)
#define BN_APPLY(a, b) ((a) / (b))
#elif defined(BN_MAX)
#define BN_APPLY(a, b) (u::max_((a), (b)))
#elif defined(BN_SILU_MUL)
// SwiGLU's core: silu(gate) * up, one expression so the activation rides in DST with the
// multiply rather than going out to L1 and back.
#define BN_APPLY(a, b) (u::silu(a) * (b))
#elif defined(BN_CHAIN)
#define BN_APPLY(a, b) ((((a) + (b)) - (a)) * (b) / (a))
#else
#define BN_APPLY(a, b) ((a) + (b))
#endif

void kernel_main() {
    constexpr uint32_t kCbIn0 = get_arg(args::cb_in0);
    constexpr uint32_t kCbIn1 = get_arg(args::cb_in1);
    constexpr uint32_t kCbOut = get_arg(args::cb_out);
    [[maybe_unused]] constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);
    const uint32_t block_begin = get_arg(args::block_begin);
    const uint32_t block_count = get_arg(args::block_count);

    u::compute_init(kCbIn0, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in0_storage(kCbIn0);
    u::Storage<Block1D> in1_storage(kCbIn1);
    u::Storage<Block1D> out_storage(kCbOut);

    const auto in0 = TensorAccessor(tensor::in0);
    const auto in1 = TensorAccessor(tensor::in1);
    const auto out = TensorAccessor(tensor::out);

    for (uint32_t n = 0; n < block_count; ++n) {
        const uint32_t b = block_begin + n;
        u::ComputeBlock a = u::noc_load<0>(in0_storage, in0, b).wait();
        u::ComputeBlock c = u::noc_load<0>(in1_storage, in1, b).wait();
        u::Block result = out_storage.store(BN_APPLY(a, c));
        u::noc_store<1>(std::move(result), out, b);
    }
}
