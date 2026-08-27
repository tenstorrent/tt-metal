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
// Compile-time args (named):
//   num_blocks, tiles_per_block
//   cb_in, cb_out   -- Metal 2.0 only; the legacy path hardcodes them below
//
// On the LEGACY path, positionally: TensorAccessorArgs for in, then out; and two runtime
// args, the in and out base addresses, identical on all three kernels.
//
// On the METAL 2.0 path there are neither. The accessors come from binding tokens and the
// base addresses ride along with them, which is hazard D18 gone rather than mitigated. The
// two spellings differ in four lines, marked below.
//
// Defines: one of UN_SQRT, UN_RSQRT, UN_EXP, UN_CHAIN; recip is the default.
//   TT_UNIFIED_METAL2  selects the 2.0 spelling; set by unified_program_spec().

#include <tt/unified/core>

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
    constexpr uint32_t num_blocks = get_named_compile_time_arg_val("num_blocks");
    constexpr uint32_t tiles_per_block = get_named_compile_time_arg_val("tiles_per_block");

#if defined(TT_UNIFIED_METAL2)
    // The buffer slots come from the host as VALUES, not as dfb:: tokens: a token exists
    // only in the kernels that bind that buffer, and the three statements below are
    // compiled on every projection. See unified_metal2_spec.md 7.1.
    constexpr uint32_t kCbIn = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t kCbOut = get_named_compile_time_arg_val("cb_out");
#else
    constexpr uint32_t kCbIn = 0;
    constexpr uint32_t kCbOut = 16;
#endif

    u::compute_init(kCbIn, kCbOut);

    using Block1D = u::Shape<1, tiles_per_block>;
    u::Storage<Block1D> in_storage(kCbIn);
    u::Storage<Block1D> out_storage(kCbOut);

#if defined(TT_UNIFIED_METAL2) && defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    // The slots above are PREDICTED by the harness from metal's allocator rule. Compute is
    // the one projection that binds every buffer -- inputs as consumer, outputs as producer
    // -- so it is the only place all of them can be checked against the ids the host really
    // assigned. That is what the dfb:: tokens are good for here: not naming, which they
    // cannot do for a shared source, but verifying.
    static_assert(
        kCbIn == static_cast<uint32_t>(dfb::in) && kCbOut == static_cast<uint32_t>(dfb::out),
        "the harness's predicted dataflow buffer slots do not match the ones the host assigned; "
        "unified_program_spec() derives them from declaration order, which no longer holds");
#endif

#if defined(TT_UNIFIED_METAL2)
    const auto in = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);
#else
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    u::check_runtime_args<2>();
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(1));
#endif

    for (uint32_t b = 0; b < num_blocks; ++b) {
        u::ComputeBlock a = u::noc_load<0>(in_storage, in, b).wait();
        u::Block result = out_storage.store(UN_APPLY(a));
        u::noc_store<1>(std::move(result), out, b);
    }
}
