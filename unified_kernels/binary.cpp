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
// Compile-time args:
//   0            num_blocks
//   1            tiles_per_block
//   2..          TensorAccessorArgs for in0, then in1, then out
//
// Runtime args (identical on all three kernels):
//   0            in0 base address
//   1            in1 base address
//   2            out base address
//
// Defines: one of BN_SUB, BN_MUL, BN_DIV, BN_MAX, BN_CHAIN; add is the default.

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbIn0 = 0;
constexpr uint32_t kCbIn1 = 1;
constexpr uint32_t kCbOut = 16;

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
    constexpr uint32_t num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(1);

    constexpr auto in0_args = TensorAccessorArgs<2>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_arg_val<uint32_t>(0);
    const uint32_t in1_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    u::compute_init(kCbIn0, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in0_storage(kCbIn0);
    u::Storage<Block1D> in1_storage(kCbIn1);
    u::Storage<Block1D> out_storage(kCbOut);

    const auto in0 = TensorAccessor(in0_args, in0_addr);
    const auto in1 = TensorAccessor(in1_args, in1_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        u::ComputeBlock a = u::noc_load<1>(in0_storage, in0, b).wait();
        u::ComputeBlock c = u::noc_load<1>(in1_storage, in1, b).wait();
        u::Block result = out_storage.store(BN_APPLY(a, c));
        u::noc_store<0>(std::move(result), out, b);
    }
}
