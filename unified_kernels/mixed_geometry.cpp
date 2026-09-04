// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TWO STORES OF DIFFERENT TILE GEOMETRY IN ONE BODY, which is the one thing no other kernel
// here does -- and the reason the defect in unified_blaze_integration_spec.md A3 survived
// unseen. Every other kernel in this directory is homogeneous: one geometry throughout, so
// `pack_to`'s reconfiguration had nothing to get wrong.
//
// The order matters. The 32x32 store goes FIRST, so a 4-face 16x16 configuration is what the
// packer holds when the row-form store arrives. A 1x32 tile is TWO faces of 1x16, and packed
// through a four-face configuration it loses face 1 entirely -- elements 16..31 of the row --
// which is why the launcher checks the two faces separately rather than only a PCC. Reversing
// the order would test the same transition in the direction that happens to be harmless,
// because the row form's configuration is the narrower of the two.
//
// This is a REGRESSION test for a fix, so what it must do is fail without it. Verified by
// reverting `pack_to`'s geometry branch: face 1 comes back 0/16 correct, exactly the
// signature A3 measured on craq-sim through blaze's `u_flash_kda`.
//
// Compile-time args, all named: a dfb_<name> per buffer. No runtime args -- the tensors are
// bound, so their addresses ride along with the accessors.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t kDfbIn = get_arg(args::dfb_in);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);
    constexpr uint32_t kDfbInRow = get_arg(args::dfb_in_row);
    constexpr uint32_t kDfbOutRow = get_arg(args::dfb_out_row);

    // One full tile, and one ROW-FORM tile. `Tiled<>` is what says the second pair's pages
    // are 1x32 rather than 32x32; the launcher says the same thing to the host through
    // dfb(..., tile=), and Storage static_asserts the two against the JIT's own tables.
    using Blk = u::Shape<1, 1>;
    using Row = u::Tiled<u::Tile<1, 32>, u::Shape<1, 1>>;

    // MG_ROW_INIT points the kernel's init at the ROW pair instead of the 32x32 one. A3's own
    // table row 1 is this case -- init naming the row buffer makes the row store exact -- so
    // it separates "the packer was left configured for the other geometry" from "the init is
    // the only thing that ever configures geometry, on either side".
#if defined(MG_ROW_INIT)
    u::compute_init(kDfbInRow, kDfbOutRow);
#elif defined(MG_MATMUL)
    // The FPU MATMUL path, which configures its operands in a third place again -- the
    // "put back what matmul_block needs" restore beside matmul_block_init -- and in reversed
    // operand order. matmul_init is not interchangeable with compute_init: it programs the
    // ALU for FPU work, so this body is its own shape rather than a variant of the others.
    // The row-form matmul, with the init naming its own shapes -- B3's `1x32 @ 32x32`, in
    // tree at last and in the honest Tiled<> spelling rather than the plain Shape<1,1> its
    // probe used (which B3a explains no longer compiles).
    //
    // ONE matmul shape, deliberately. A second matmul of a DIFFERENT shape in the same body
    // comes back entirely wrong -- 0/16 on both faces, measured -- because matmul_init
    // programs the MOP once from one MatmulGeometry and nothing reprograms kt_dim/rt_dim/
    // ct_dim per pass. That is a structural limit of the matmul path, not the descriptor
    // defect this kernel is about, so it is recorded in A3 rather than tested here.
    u::matmul_init<Row, Blk>(kDfbInRow, kDfbIn, kDfbOutRow);
#else
    u::compute_init(kDfbIn, kDfbOut);
#endif

    u::Input<0, kDfbIn, Blk> in_storage;
    u::Output<1, kDfbOut, Blk> out_storage;
    u::Input<0, kDfbInRow, Row> in_row_storage;
    u::Output<1, kDfbOutRow, Row> out_row_storage;

    const auto in = TensorAccessor(tensor::in);
    const auto out = TensorAccessor(tensor::out);
    const auto in_row = TensorAccessor(tensor::in_row);
    const auto out_row = TensorAccessor(tensor::out_row);

#if defined(MG_OUTER)
    // B4's RANK-1 UPDATE: k [K,1] (x) delta [1,V], which every gated-delta-net and
    // linear-attention decode step is built from.
    //
    // The LHS wants a 32x1 tile and there is no such tile -- metal's TILE_FACE_HW_CHOICES has
    // nothing one column wide -- so k lives in a 32x32 tile with columns 1..31 ZERO, and that
    // zero padding is what makes the product right: the hardware takes a full 32-element
    // inner step and the surplus terms contribute nothing. No type can see whether those
    // columns are actually zero, so that much stays the author's contract; what the asserts
    // do reach is that A is at least as WIDE as B is tall, which is what makes the surplus
    // A's to zero rather than B's real values multiplying whatever SrcA holds.
    //
    // Under the old ELEMENTS check this had no spelling at all: logical 32 columns against
    // logical 1 row, refused, with no third form satisfying both the check and the buffers'
    // declared geometry. See B4.
    u::matmul_init<Blk, Row>(kDfbIn, kDfbInRow, kDfbOut);

    u::ComputeBlock k = u::noc_load(in_storage, in, 0).wait();
    u::ComputeBlock d = u::noc_load(in_row_storage, in_row, 0).wait();
    u::noc_store(out_storage, u::matmul(k, d), out, 0);
    return;
#endif

#if defined(MG_MATMUL)
    // Pass 1: a 32x32 product, which leaves every descriptor at the 32x32 geometry.
    // Pass 2: a ROW-FORM LHS, which is B3's `1x32 @ 32x32` -- the inner dimension still
    // agrees in elements (1 tile x 32 wide against 1 tile x 32 high), and the output
    // inherits the LHS tile, so it is a row-form store as well. Reverse order means the
    // row-form operand lands in srcB and the 32x32 one in srcA: a genuinely MIXED pair,
    // which is the case the one-argument descriptor form cannot express.
    u::ComputeBlock ma = u::noc_load(in_storage, in, 0).wait();
    u::ComputeBlock mr = u::noc_load(in_row_storage, in_row, 0).wait();
    u::noc_store(out_row_storage, u::matmul(mr, ma), out_row, 0);
    return;
#endif

    // 32x32 first: this is what leaves the packer configured for four faces.
    //
    // MG_ROW_ONLY skips it, which is how the row store is tested in ISOLATION: if the row
    // pass fails alone then the transition is not the mechanism and the row-form pack path
    // is broken on its own terms.
#if !defined(MG_ROW_ONLY)
    u::ComputeBlock a = u::noc_load(in_storage, in, 0).wait();
#if defined(MG_FPU)
    u::noc_store(out_storage, a + a, out, 0);
#else
    u::noc_store(out_storage, u::recip(a), out, 0);
#endif
#endif

    // MG_REINIT re-inits the SFPU for the ROW pair right here, mid-body, through the raw
    // API. This is A3's table row 5 in-tree, and it identifies the repairing call exactly:
    // compute_init is ckernel::init_sfpu(in, out), which programs the geometry on BOTH sides
    // -- unpacker and packer. pack_to_forget() goes with it because the memo would otherwise
    // skip the format reconfig that the re-init's own programming has to be followed by.
#if defined(MG_REINIT) && defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    u::pack_to_forget();
    ckernel::init_sfpu(kDfbInRow, kDfbOutRow);
#endif

    // Then the row form, whose store is the one that needs the packer reprogrammed and not
    // merely reformatted. Same op both times, so a difference between the two outputs is the
    // geometry transition and nothing else.
    u::ComputeBlock r = u::noc_load(in_row_storage, in_row, 0).wait();
#if defined(MG_FPU)
    // The FPU path instead of the SFPU one, which configures its operands somewhere else
    // entirely: fpu_seed_init points srcA and srcB at two buffers, so it is the two-operand
    // descriptor form that has to be right. An elementwise binary cannot MIX geometries --
    // the shapes would not match -- so the geometry changes BETWEEN the two passes, which is
    // the real shape of it anyway: blaze's flash_kda alternates 32x32 ops and row ops.
    u::noc_store(out_row_storage, r + r, out_row, 0);
#else
    u::noc_store(out_row_storage, u::recip(r), out_row, 0);
#endif
}
