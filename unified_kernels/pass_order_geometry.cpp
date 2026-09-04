// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TILE GEOMETRY ACROSS A SEQUENCE OF PASSES. mixed_geometry.cpp varies WHICH GEOMETRY THE
// INIT NAMES for a body of one or two passes; this varies WHICH PASSES RUN, AND IN WHAT
// ORDER, with the init held fixed at the shapes the pass under test names for itself.
//
// It exists because blaze's u_flash_kda is still wrong at the row form after
// `f1d762a3707` and `eb17b41d7fd` -- s_out PCC 0.675, o 0.830 -- and its body is
//
//     bcast -> matmul -> SFPU -> matmul -> matmul
//
// which is longer than anything in this directory. Bisecting it through blaze pointed at two
// things neither of those commits covers, and both reproduce here with no blaze in the
// picture. See unified_blaze_integration_spec.md A3.
//
// WHAT THE BODIES ARE FOR. Two claims, one each:
//
//   PO_BCAST      the BROADCAST path never programmed operand geometry at all -- FIXED.
//                 Its operands are ALL 32x32 in this body, so its result must not depend on
//                 what the init named, and it did: with PO_ROW_INIT the same all-square
//                 bcast came back wrong. Strategy<BcastFusion> now programs its operands'
//                 descriptors beside their formats, and both bodies are exact.
//
//   PO_SFPU_MM    an SFPU pass over 32x32 operands, followed by a row-form matmul that
//                 named its own shapes, broke the MATMUL -- FIXED. PO_MM_ONLY is the same
//                 matmul alone and was always exact; PO_MM_SFPU runs the two in the other
//                 order and was always exact both ways.
//
//                 PO_BCAST_MM used to be the control here -- a bcast in front instead of
//                 the SFPU pass, and exact. Fixing the bcast BROKE it, because the bcast
//                 was harmless only by virtue of programming nothing, and that is what
//                 settled the shape of this: not "the SFPU path reprograms and the FPU path
//                 does not notice", but any pass that programs operand geometry ahead of a
//                 matmul. Strategy<FPUFusion>::run and ::run_banded now program their
//                 operands' descriptors at entry, beside the block dimensions they already
//                 programmed there, and every body is exact at both tile counts.
//
//                 That also retires PO_REINIT below as a workaround: there is nothing left
//                 for a mid-body re-init to repair.
//
// This is DISTINCT from the matmul limitation A3 already records. That one is two matmul
// SHAPES in one body; PO_SFPU_MM has a single matmul, and what precedes it is not a matmul.
//
// IT WAS NOT FIXABLE FROM A KERNEL, which is why the fix is in the library. `u::compute_init`
// naming the matmul's own operands, inserted between the two passes, made PO_SFPU_MM WORSE
// rather than better (measured through blaze: 0.599 -> 0.396), and matmul_init cannot be run
// twice -- compute_kernel_hw_startup is MMIO plus a pack-sync init and the second call hangs.
// PO_REINIT is that attempt, kept so the negative result is in tree rather than in a commit
// message.
//
// Compile-time args, all named: a dfb_<name> per buffer. No runtime args -- the tensors are
// bound, so their addresses ride along with the accessors.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    // The block is ht x wt TILES. kt_dim of the matmul under test is Row::cols == wt, so
    // wt > 1 is what makes it walk SEVERAL k-steps -- and a k-step re-reads srcB. A
    // single-tile matmul never exposes srcB's descriptor twice, which is why the whole
    // kt=1 story about the mid-body re-init does not carry: see PO_REINIT.
    constexpr uint32_t ht = get_arg(args::ht);
    constexpr uint32_t wt = get_arg(args::wt);
    static_assert(ht == wt, "matmul(Row, Blk) needs Row::cols == Blk::rows");

    constexpr uint32_t kDfbIn = get_arg(args::dfb_in);
    constexpr uint32_t kDfbVec = get_arg(args::dfb_vec);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);
    constexpr uint32_t kDfbInRow = get_arg(args::dfb_in_row);
    constexpr uint32_t kDfbOutRow = get_arg(args::dfb_out_row);

    using Blk = u::Shape<ht, wt>;
    using Row = u::Tiled<u::Tile<1, 32>, u::Shape<1, wt>>;
    // The broadcast vector is whatever Axis::Cols requires of the block: one column of
    // tiles, at the block's own 32x32 geometry.
    using Vec = u::reduce_shape<Blk, u::Axis::Cols>;

    u::Input<0, kDfbIn, Blk> in_storage;
    u::Input<0, kDfbVec, Vec> vec_storage;
    u::Output<1, kDfbOut, Blk> out_storage;
    u::Input<0, kDfbInRow, Row> in_row_storage;
    u::Output<1, kDfbOutRow, Row> out_row_storage;

    const auto in = TensorAccessor(tensor::in);
    const auto vec = TensorAccessor(tensor::vec);
    const auto out = TensorAccessor(tensor::out);
    const auto in_row = TensorAccessor(tensor::in_row);
    const auto out_row = TensorAccessor(tensor::out_row);

#if defined(PO_BCAST)
    // FINDING 1. One broadcast pass, every operand 32x32, nothing else in the body.
    //
    // PO_ROW_INIT points the init at the ROW pair. Nothing in this body touches a row-form
    // buffer, so the result cannot legitimately depend on that -- if it does, the bcast pass
    // is reading whatever geometry the init left rather than programming its operands'.
    //
    // Axis::Cols with a single tile: the vector is Shape<1,1> too, so the broadcast is
    // column 0 of `vec` spread across the block's columns.
#if defined(PO_ROW_INIT)
    u::compute_init(kDfbInRow, kDfbOutRow);
#else
    u::compute_init(kDfbIn, kDfbOut);
#endif
    u::ComputeBlock b = u::noc_load(in_storage, in, 0).wait();
    u::ComputeBlock v = u::noc_load(vec_storage, vec, 0).wait();
    u::noc_store(out_storage, b * u::bcast<u::Axis::Cols>(v), out, 0);
    return;
#else

    // FINDING 2's four bodies. The init names the MATMUL's own shapes in all of them --
    // row-form LHS against a 32x32 RHS, row-form output -- so the matmul is never the pass
    // whose geometry the init got wrong. Whatever breaks it is the pass beside it.
    u::matmul_init<Row, Blk>(kDfbInRow, kDfbIn, kDfbOutRow);

#if defined(PO_SFPU_FIRST)
    // The 32x32 SFPU pass, BEFORE the matmul. This is the body that fails.
    u::ComputeBlock s = u::noc_load(in_storage, in, 0).wait();
    u::noc_store(out_storage, u::recip(s), out, 0);
#elif defined(PO_BCAST_FIRST)
    // A bcast in front instead. This was the control -- exact, because the bcast path
    // disturbed nothing by programming nothing -- and once it programs its own descriptors
    // it breaks the matmul behind it exactly as the SFPU pass does. It is now the second
    // witness for finding 2 rather than a control for it.
    u::ComputeBlock b = u::noc_load(in_storage, in, 0).wait();
    u::ComputeBlock v = u::noc_load(vec_storage, vec, 0).wait();
    u::noc_store(out_storage, b * u::bcast<u::Axis::Cols>(v), out, 0);
#endif

    // MID-BODY RE-INIT, which is a TRAP rather than a workaround -- and only says so at
    // wt > 1. It makes the matmul exact at wt=1 and leaves it half wrong at wt=4:
    //
    //     wt=1  face0=16/16 face1=16/16      wt=4  face0=29/64 face1=35/64
    //
    // `init_sfpu(in, out)` forwards to `unpack_hw_configure(in, in)`, programming srcA AND
    // srcB from ONE buffer. That is right for a copy or an SFPU pass and wrong for a matmul
    // whose operands differ in geometry -- and it only shows once srcB is read more than
    // once, which is what a second k-step does. Hence the two tile counts.
    //
    // Both bodies below behave identically, so the pack_to_forget() that mixed_geometry's
    // MG_REINIT pairs this with is not load-bearing either way.
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
#if defined(PO_REINIT)
    u::pack_to_forget();
    ckernel::init_sfpu(kDfbInRow, kDfbOutRow);
#elif defined(PO_REINIT_NO_FORGET)
    ckernel::init_sfpu(kDfbInRow, kDfbOutRow);
#endif
#endif

    // The matmul under test: B3's `1x32 @ 32x32`, output inheriting the LHS's row tile.
    u::ComputeBlock r = u::noc_load(in_row_storage, in_row, 0).wait();
    u::ComputeBlock m = u::noc_load(in_storage, in, 0).wait();
    u::noc_store(out_row_storage, u::matmul(r, m), out_row, 0);

#if defined(PO_SFPU_LAST)
    // The mirror of PO_SFPU_FIRST: the same two passes, matmul first. Both come back exact,
    // which is what makes the staleness one-directional rather than symmetric.
    u::ComputeBlock s2 = u::noc_load(in_storage, in, 0).wait();
    u::noc_store(out_storage, u::recip(s2), out, 0);
#endif
#endif
}
