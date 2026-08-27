// SPDX-License-Identifier: Apache-2.0
//
// A measurement kernel, not a real op: PASSES passes over one block, each pass the
// op under test, so the SLOPE in PASSES is what one pass of that op costs.
//
// The slope is the point. It cancels every fixed cost -- program launch, the initial
// load, the final store -- because those are paid once regardless of PASSES, so
// nothing has to be modelled or guessed at. Each pass lands in its own scratch CB, so
// no pass reuses a buffer a live block still occupies.
//
//   default    out = copy(in)                      the zero-math control
//   PC_BCAST   out = prev + bcast<Cols>(vec)       shape-preserving, so chainable
//   PC_MATMUL  out = matmul(prev, w), w square     shape-preserving, so chainable
//   PC_BIN     out = prev <op> rhs, both whole blocks of the same shape. With PC_FPU
//              the op runs on the FPU, reading both operands out of L1 and needing no
//              copy_tile at all; without it, on the SFPU, which costs two copy_tiles
//              per output tile. PC_OP_SUB / PC_OP_MUL pick the op, add is the default.
//              This is the pair the FPU-vs-SFPU question turns on.
//   PC_ALT     alternates bcast and copy per pass  same tiles, mixed KINDS
//   PC_REDUCE  out = reduce_max<Cols>(in, one)     collapses, so PASSES==1 only
//
// PC_MATMUL takes w as the IDENTITY, which makes the chain an identity chain and the
// reference exact -- every product but one is a zero, so nothing rounds -- while the
// FPU still does the full inner product it would do for any other operand. The
// hardware does not shortcut a zero.
//
// Subtracting the copy slope from the bcast slope leaves the broadcast's own math:
// both move the same tiles in and out per pass, so the plumbing cancels exactly.
//
// PC_ALT is the control for the homogeneous chains above. A chain of identical passes
// is the best case for the hardware: nothing about the unpacker or the DST format
// changes between them, so the slope excludes reconfiguration by construction. A real
// kernel alternates KINDS on nearly every pass, and a broadcast in particular leaves
// the unpacker in a broadcast mode that the next SFPU op has to put back. Comparing
// PC_ALT's slope against the mean of the copy and bcast slopes prices that switch.
//
// A reduction cannot be chained -- its output is narrower than its input, so pass two
// would measure a different shape than pass one. It is measured instead by sweeping
// the shape at PASSES=1: widening `cols` at fixed `rows` prices one more INPUT tile
// (an unpack plus an accumulate, with no per-tile pack), and raising `rows` at fixed
// `cols` prices one more OUTPUT tile, which is what puts a number on the one
// tile_regs_acquire per output tile that Strategy<ReduceFusion> still does.
//
// Compile-time args, all named, plus a cb_<name> per buffer that TT_U_CB reads:
//   block height in tiles
//   block width in tiles
//
// Runtime args: in address, then vec (PC_BCAST only), then out address.
//
// Define PASSES (1..8).

#include <tt/unified/core>

namespace u = tt::unified;

#ifndef PASSES
#define PASSES 1
#endif

#if defined(PC_REDUCE)
static_assert(PASSES == 1, "a reduction is not shape-preserving, so it cannot be chained");
#endif

// Every pass is one expression, so the mode is the expression and the chain below is
// the same code in all three cases.
#if defined(PC_BIN)
#if defined(PC_OP_SUB)
#define PC_PASS(x) PC_BIN_APPLY(sub, -, x)
#elif defined(PC_OP_MUL)
#define PC_PASS(x) PC_BIN_APPLY(mul, *, x)
#else
#define PC_PASS(x) PC_BIN_APPLY(add, +, x)
#endif
#if defined(PC_FPU)
#define PC_BIN_APPLY(fpu, sym, x) u::fpu_##fpu((x), rhs)
#else
#define PC_BIN_APPLY(fpu, sym, x) ((x)sym rhs)
#endif
#elif defined(PC_BCAST)
#define PC_PASS(x) ((x) + u::bcast<u::Axis::Cols>(vec))
#elif defined(PC_MATMUL)
#define PC_PASS(x) u::matmul((x), vec)
#else
#define PC_PASS(x) u::copy(x)
#endif

// Alternating needs the kind to differ per pass, which a single macro cannot express
// because the two expressions have different types. Hence a second name and an
// explicit ladder below.
#if defined(PC_ALT)
#define PC_ODD(x) ((x) + u::bcast<u::Axis::Cols>(vec))
#define PC_EVEN(x) u::copy(x)
#endif

void kernel_main() {
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cols = get_named_compile_time_arg_val("cols");

    constexpr uint32_t kCbIn = TT_U_CB(in);
    constexpr uint32_t kCbVec = TT_U_CB(vec);
    constexpr uint32_t kCbOut = TT_U_CB(out);

#if defined(PC_MATMUL)
    static_assert(rows == cols, "a chained matmul has to be square to preserve the shape");
    u::matmul_init<u::Shape<rows, cols>, u::Shape<rows, cols>>(kCbIn, kCbVec, kCbOut);
#else
    u::compute_init(kCbIn, kCbOut);
#endif

    using S = u::Shape<rows, cols>;
#if defined(PC_BCAST) || defined(PC_REDUCE) || defined(PC_ALT)
    // Whatever a fold along that axis produces is exactly what a broadcast back along
    // it demands, so one alias serves both.
    using Vec = u::reduce_shape<S, u::Axis::Cols>;  // rows x 1
#endif

    u::Storage<S> in_storage(kCbIn);
#if defined(PC_REDUCE)
    u::Storage<Vec> out_storage(kCbOut);
#else
    u::Storage<S> out_storage(kCbOut);
#endif

    // One scratch CB per intermediate pass.
    u::Storage<S> s1(1), s2(2), s3(3), s4(4), s5(5), s6(6), s7(7);

    const auto in_acc = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);

    u::ComputeBlock c0 = u::noc_load<1>(in_storage, in_acc, 0).wait();

#if defined(PC_BIN)
    u::Storage<S> rhs_storage(kCbVec);
    const auto rhs_acc = TensorAccessor(tensor::vec);
    u::ComputeBlock rhs = u::noc_load<1>(rhs_storage, rhs_acc, 0).wait();
#endif
#if defined(PC_BCAST) || defined(PC_MATMUL) || defined(PC_ALT)
    // Read once and used by every pass: a ComputeBlock is not consumed by being read,
    // which is what lets one operand feed the whole chain.
#if defined(PC_MATMUL)
    u::Storage<S> vec_storage(kCbVec);
#else
    u::Storage<Vec> vec_storage(kCbVec);
#endif
    const auto vec_acc = TensorAccessor(tensor::vec);
    u::ComputeBlock vec = u::noc_load<1>(vec_storage, vec_acc, 0).wait();
#endif

#if defined(PC_REDUCE)
    u::Storage<u::Shape<1, 1>> one_storage(kCbVec);
    u::ComputeBlock one = u::fill_reduce_scaler<1>(one_storage, u::kReduceScalerOne);
    u::noc_store<0>(out_storage.store(u::reduce_max<u::Axis::Cols>(c0, one)), out, 0);
#else
#if defined(PC_ALT)
    // Odd passes broadcast, even passes copy, so every pass changes kind.
#if PASSES >= 2
    u::ComputeBlock c1 = s1.store(PC_ODD(c0));
#endif
#if PASSES >= 3
    u::ComputeBlock c2 = s2.store(PC_EVEN(c1));
#endif
#if PASSES >= 4
    u::ComputeBlock c3 = s3.store(PC_ODD(c2));
#endif
#if PASSES >= 5
    u::ComputeBlock c4 = s4.store(PC_EVEN(c3));
#endif
#if PASSES >= 6
    u::ComputeBlock c5 = s5.store(PC_ODD(c4));
#endif
#if PASSES >= 7
    u::ComputeBlock c6 = s6.store(PC_EVEN(c5));
#endif
#if PASSES >= 8
    u::ComputeBlock c7 = s7.store(PC_ODD(c6));
#endif
#if PASSES == 1
    u::noc_store<0>(out_storage.store(PC_ODD(c0)), out, 0);
#elif PASSES == 2
    u::noc_store<0>(out_storage.store(PC_EVEN(c1)), out, 0);
#elif PASSES == 3
    u::noc_store<0>(out_storage.store(PC_ODD(c2)), out, 0);
#elif PASSES == 4
    u::noc_store<0>(out_storage.store(PC_EVEN(c3)), out, 0);
#elif PASSES == 5
    u::noc_store<0>(out_storage.store(PC_ODD(c4)), out, 0);
#elif PASSES == 6
    u::noc_store<0>(out_storage.store(PC_EVEN(c5)), out, 0);
#elif PASSES == 7
    u::noc_store<0>(out_storage.store(PC_ODD(c6)), out, 0);
#else
    u::noc_store<0>(out_storage.store(PC_EVEN(c7)), out, 0);
#endif
#else
#if PASSES >= 2
    u::ComputeBlock c1 = s1.store(PC_PASS(c0));
#endif
#if PASSES >= 3
    u::ComputeBlock c2 = s2.store(PC_PASS(c1));
#endif
#if PASSES >= 4
    u::ComputeBlock c3 = s3.store(PC_PASS(c2));
#endif
#if PASSES >= 5
    u::ComputeBlock c4 = s4.store(PC_PASS(c3));
#endif
#if PASSES >= 6
    u::ComputeBlock c5 = s5.store(PC_PASS(c4));
#endif
#if PASSES >= 7
    u::ComputeBlock c6 = s6.store(PC_PASS(c5));
#endif
#if PASSES >= 8
    u::ComputeBlock c7 = s7.store(PC_PASS(c6));
#endif

    // The last pass writes the output CB, so PASSES passes in total.
#if PASSES == 1
    u::noc_store<0>(out_storage.store(PC_PASS(c0)), out, 0);
#elif PASSES == 2
    u::noc_store<0>(out_storage.store(PC_PASS(c1)), out, 0);
#elif PASSES == 3
    u::noc_store<0>(out_storage.store(PC_PASS(c2)), out, 0);
#elif PASSES == 4
    u::noc_store<0>(out_storage.store(PC_PASS(c3)), out, 0);
#elif PASSES == 5
    u::noc_store<0>(out_storage.store(PC_PASS(c4)), out, 0);
#elif PASSES == 6
    u::noc_store<0>(out_storage.store(PC_PASS(c5)), out, 0);
#elif PASSES == 7
    u::noc_store<0>(out_storage.store(PC_PASS(c6)), out, 0);
#else
    u::noc_store<0>(out_storage.store(PC_PASS(c7)), out, 0);
#endif
#endif
#endif
}
