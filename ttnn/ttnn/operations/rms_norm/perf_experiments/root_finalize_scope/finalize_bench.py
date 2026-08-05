# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for rms_norm's FINALIZE stage:  1/rms = rsqrt(sum(x^2) * (1/W) + eps).

Two levers, measured on their own:
  (a) SFPU WORK-SCOPING of the whole finalize chain.  cb_row_stat is a REDUCE_ROW
      result -- only COLUMN 0 is ever read back (pass B consumes it as
      mul<BroadcastDim::Col>).  The op today scopes only the rsqrt (VectorMode::C,
      Lamp L6b); `mul_unary_tile` / `add_unary_tile` still run VectorMode::RC (all
      32 SFPU vector ops).  The ladder pushes further to the axis-optimal
      even-parity `c_skip` stride (8 vector ops -> column 0 exactly).
  (b) DELETING THE HANDOFF COPY.  The root finalizes IN PLACE into cb_row_stat and
      then copies the whole fp32 tile block into cb_stat_handoff.  A transform that
      READS cb_row_stat and WRITES cb_stat_handoff in one pass does the same work
      with one pack instead of two and no re-unpack.

PRECISION CONTRACT (fixed, never a lever):  fp32 stat CB, bf16 activations,
math_fidelity = HiFi2, fp32_dest_acc_en = False, math_approx_mode = False.  Every
variant runs under the identical ComputeConfigDescriptor.

------------------------------------------------------------------------------
Three measurement modes
------------------------------------------------------------------------------
MODE_SFPU  ("isolated")  -- the sfpu_tile_scope technique.  One tile is copied into
    DEST once and packed out once, both OUTSIDE a DeviceZoneScopedN; inside the zone
    a MATH-thread-only loop applies the finalize chain `reps` times.  The zone's
    TRISC_1 (math) duration / reps is the PURE SFPU cost of the chain, with no
    unpack, pack or CB handshake in it.

MODE_STAGE ("structural") -- mirrors the op.  `rows` fp32 stat tiles live in a
    sharded CB; the kernel runs the op's exact stage sequence and two PERMANENT
    zones bracket it:
        bench_finalize   the finalize itself  == the op's compute_finalize /
                         compute_root_finalize (this is ALSO the number for the
                         LOCAL, non-combine path, which has no handoff at all)
        bench_handoff    the cb_row_stat -> cb_stat_handoff copy == the op's
                         compute_stat_handoff
    so this mode reports the number that predicts the whole-op delta (copy + pack
    + CB handshake included), per tile.

MODE_CONSUME ("consumer") -- the CORRECTNESS experiment that decides the domain.
    Finalize the stat, then run pass B's real consumer,
    BinaryFpu<x, stat, Mul, BroadcastDim::Col>, and check x * (1/rms) against
    torch.  The stat tile is seeded with DELIBERATELY WRONG values in every column
    but column 0, so if the broadcast consumer read any lane a scope left stale the
    PCC craters.  This is what turns "column 0 is all that's read" from an
    assumption into a measurement.

------------------------------------------------------------------------------
Finalize variants (the (a) ladder)
------------------------------------------------------------------------------
MEASURED on bh-qb-13 (blackhole p150b) / BH / 1350 MHz / 2026-08-05, at the pinned
config.  "SFPU ns" = isolated MATH-thread ns per finalize call (reps=2000, N=3 median);
"stage ns/tile" = the structural finalize+handoff cost per tile at rows=32 (warm).

    id  name           mul        add        rsqrt      pass vec  SFPU ns  stage ns/tile
    0   rc_all         RC (32)    RC (32)    RC (32)      3   96    970.4      1170.6
    1   base           RC (32)    RC (32)    C  (16)      3   80    600.7       762.1  <- THE OP TODAY
    2   scope_c        C  (16)    C  (16)    C  (16)      3   48    498.5       649.3
    3   cskip3         c_skip(8)  c_skip(8)  c_skip(8)    3   24    267.5       402.5
    4   cskip2         --- c_skip(8) ---     c_skip(8)    2   16    244.5       372.8
    5   fused_c        ---- one sfpi body, VectorMode::C ----      1   16    417.2       571.9
    6   cskip_fused    ---- one sfpi body, C + even parity ---     1    8    215.7       347.4

The isolated cost decomposes exactly: rsqrt ~23.1 ns per 32-lane vector op,
mul_unary / add_unary ~3.6 ns each.  So the baseline's 600.7 ns is 370 ns of rsqrt (16
vectors) plus 231 ns of mul+add running the OTHER 32 vectors each -- 38% of the stage
spent scaling lanes nobody reads.  That 38% is what lever (a) recovers.

FIRST-CALL COST OF THE RAW-SFPI BODIES (measured, and the reason the win depends on how
many tiles a call sequence covers).  The stock SFPU calls are driven by SFPLOADMACRO
hardware sequencing; a hand-written sfpi body is a straight instruction stream, so its
FIRST execution in a program pays an instruction-fetch cost the reps loop amortizes away:
    cskip_fused   first call ~595 ns, warm ~266 ns
    cskip3        first call ~497 ns, warm ~330 ns
    base          first call ~640 ns, warm ~656 ns   (no first-call penalty at all)
So at rows == 1 the (a) win on the finalize zone shrinks to 1.08x-1.37x, and at
rows >= 10 it is the full 2.0x-2.4x.  Flat-ish, never a regression.

"Passes" matters as much as vector count: each separate SFPU op re-reads and
re-writes DEST.  cskip2 folds *(1/W) and +eps into one body; fused_c / cskip_fused
fold all three into one, so the two intermediates stay in an LREG at fp32 instead of
round-tripping through a 16-bit DEST word (fp32_dest_acc_en == False) -- FEWER
roundings than the baseline, so their accuracy is >= baseline, not <=.

LANES LEFT STALE (measured, see test):
    rc_all         none -- whole tile finalized.
    base           cols 16-31: `*(1/W) + eps` APPLIED but NOT rsqrt'd (touched-stale).
    scope_c        cols 16-31: raw sum(x^2), nothing applied (untouched-stale).
    cskip3/2/fused cols 1,3,..,15 AND cols 16-31: raw sum(x^2) (untouched-stale).
    fused_c        cols 16-31: raw sum(x^2) (untouched-stale).
None of the stale lanes can be inf/NaN: they carry a finite reduce result, or a
finite value times 1/W plus eps.  Column 0 is bit-identical across variants 0-3
(the scope only removes lanes; it never changes the math on a lane it runs).

------------------------------------------------------------------------------
Handoff structures (the (b) lever)
------------------------------------------------------------------------------
    id  name             what it does
    0   inplace_copy     THE OP TODAY: `rows` x transform_in_place(cb_row_stat)
                         then ckl::copy<cb_row_stat -> cb_stat_handoff>(rows).
                         2 unpacks + 2 packs per tile.
    1   xfer_raw         `rows` x transform_to(cb_row_stat -> cb_stat_handoff):
                         hand-rolled A->B twin of transform_in_place.
                         1 unpack + 1 pack per tile.
    2   xfer_chain       the SAME A->B pass expressed through the helper family:
                         eltwise_chain(tiles(rows), CopyTile<input(A)>,
                                       <finalize element>, PackTile<output(B)>).
                         See LIBRARY GAP below.

LIBRARY GAP (an inexpressible-in-helpers finding, reported as such):
  * `transform_in_place` (streaming_reduce_helpers.hpp) is the ONLY user-lambda
    transform in the family and it is single-CB by construction: it pops from `cb`
    and packs back into `cb`.  There is NO `transform(cb_in, cb_out, lambda)`.
  * The A->B pass IS expressible, but only by going through `eltwise_chain` with a
    chain ELEMENT rather than a lambda.  With stock elements
    (MulUnary + AddUnary + Rsqrt) that costs the op its existing L6b rsqrt scope,
    because NONE of the stock elements exposes a VectorMode seam:
    `mul_unary_tile` / `add_unary_tile` hardcode VectorMode::RC
    (binop_with_scalar.h) and `rsqrt_tile` hardcodes VectorMode::RC (rsqrt.h).
    So a scoped A->B pass needs a user-defined `UnaryOp<>` element carrying the
    raw SFPU_UNARY_CALL -- which is exactly the extension surface eltwise_chain.inl
    documents for the CRTP bases, so it is IN-pattern, not a new licence.
  * variant `xfer_chain` here is that user-defined element, so `xfer_chain` and
    `xfer_raw` differ only in who owns the CB lifecycle -- and MEASURED, the helper wins:
    one `eltwise_chain` over `tiles(rows)` hoists its init + reconfig out of the per-tile
    loop, which the `rows`-trip `transform_in_place` / `transform_to` loop re-emits every
    tile.  25 ns/tile at rows=32; 20110 vs 20910 ns for the whole stage.

MEASURED handoff structures (rows=32, baseline finalize, ns for the whole stage):
    inplace_copy   20970 finalize + 3419 handoff = 24389   <- THE OP TODAY
    xfer_raw       20910 + 0                     = 20910   1.17x
    xfer_chain     20110 + 0                     = 20110   1.21x
and with the best (a) on top: xfer_chain + cskip_fused = 8076 ns => 3.02x vs baseline.
"""

import ttnn

TILE = 32

CB_STAT = 0  # fp32, `rows` pages -- the op's cb_row_stat
CB_X = 1  # bf16, `rows` pages -- pass B's x (MODE_CONSUME only)
CB_OUT = 16  # the op's cb_stat_handoff (fp32) / pass B's output (bf16)

MODE_SFPU, MODE_STAGE, MODE_CONSUME = 0, 1, 2
MODES = ("isolated", "structural", "consumer")
_MODE_ID = {"isolated": MODE_SFPU, "structural": MODE_STAGE, "consumer": MODE_CONSUME}

# finalize variant -> id
VARIANTS = ("rc_all", "base", "scope_c", "cskip3", "cskip2", "fused_c", "cskip_fused")
_VAR_ID = {n: i for i, n in enumerate(VARIANTS)}
BASELINE = "base"  # the op's current approach == the honest baseline

HANDOFFS = ("inplace_copy", "xfer_raw", "xfer_chain")
_HOFF_ID = {n: i for i, n in enumerate(HANDOFFS)}
BASELINE_HANDOFF = "inplace_copy"

# 32-lane SFPU vector ops each variant runs, and how many DEST round-trips (passes).
VEC_OPS = {"rc_all": 96, "base": 80, "scope_c": 48, "cskip3": 24, "cskip2": 16, "fused_c": 16, "cskip_fused": 8}
PASSES = {"rc_all": 3, "base": 3, "scope_c": 3, "cskip3": 3, "cskip2": 2, "fused_c": 1, "cskip_fused": 1}
# The fully-fused bodies need W to be a POWER OF FOUR (sqrt(W) as an exponent add).
NEEDS_POW4_W = ("fused_c", "cskip_fused")

# Columns of the 32x32 stat tile each variant leaves FULLY FINALIZED (the rest is
# stale -- see the module docstring).  Used by the test to slice output vs golden.
VALID_COLS = {
    "rc_all": list(range(32)),
    "base": list(range(16)),
    "scope_c": list(range(16)),
    "cskip3": list(range(0, 16, 2)),
    "cskip2": list(range(0, 16, 2)),
    "fused_c": list(range(16)),
    "cskip_fused": list(range(0, 16, 2)),
}

ZONE_SFPU = "RFS_SFPU"
ZONE_FINALIZE = "bench_finalize"
ZONE_HANDOFF = "bench_handoff"


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(shape):
    """Height-shard `shape` onto a single core (the whole tensor is one shard)."""
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# =============================================================================
# The compute kernel.
#
# CT args: [MODE, VAR, HOFF, INV_W_BITS, EPS_BITS, ROWS]
# RT args: [reps]   (MODE_SFPU only; ignored elsewhere)
# =============================================================================
_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_binop_with_unary.h"
#endif

namespace ckl = compute_kernel_lib;
using ckernel::VectorMode;

constexpr uint32_t cb_stat = 0;
constexpr uint32_t cb_x = 1;
constexpr uint32_t cb_out = 16;

// =====================================================================================
// RAW-LLK SUBSTITUTIONS -- one comment, four functions, all for the same reason.
//
// The finalize chain's three SFPU ops each hardcode `VectorMode::RC` and expose NO
// VectorMode seam -- not a template parameter, not a runtime argument:
//     mul_unary_tile / add_unary_tile   api/compute/eltwise_unary/binop_with_scalar.h
//     rsqrt_tile                        api/compute/eltwise_unary/rsqrt.h:45
// and the SFPU walks a face as [rg0-even, rg0-odd, rg1-even, ...], so COLUMN PARITY is
// the INNER walk axis -- unreachable by `ITERATIONS`, which truncates contiguously.
// Scoping to column 0 therefore needs (i) VectorMode::C and (ii) a hand-addressed sfpi
// body that strides the DEST address by 2.  These are the same macro the stock calls
// expand to with one argument changed, plus the even-parity bodies.  The op already
// carries exactly this substitution for the rsqrt (`rsqrt_tile_col`, Lamp L6b), with
// the in-tree precedent sdpa/.../compute_common.hpp `recip_tile_first_column`.
// =====================================================================================

#ifdef TRISC_MATH

// --- VectorMode-scoped twins of the stock calls (coarse: 2 faces instead of 4) ------
ALWI void mul_unary_tile_col(uint32_t idst, uint32_t param1) {
    SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, ckernel::sfpu::MUL, 8 /*ITERATIONS*/), idst, VectorMode::C, param1);
}
ALWI void add_unary_tile_col(uint32_t idst, uint32_t param1) {
    SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, ckernel::sfpu::ADD, 8 /*ITERATIONS*/), idst, VectorMode::C, param1);
}
ALWI void rsqrt_tile_col(uint32_t idst) {
    SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
        (APPROX, 8 /*ITERATIONS*/, DST_ACCUM_MODE, false /*FAST_APPROX*/, false /*legacy_compat*/),
        idst, VectorMode::C);
}

// --- Axis-optimal COLUMN-0 bodies (even-parity stride) -----------------------------
// Column 0 is EVEN, and an even-parity vector covers columns 0,2,..,14; the odd
// vectors touch only 1,3,..,15 and can never hold column 0.  Visiting offsets
// 0,2,4,6 (`dst_reg += 2`) runs 4 vectors per face instead of 8.  The NET dst_reg
// advance is +8 == the stock ITERATIONS=8, so VectorMode::C's face-0 -> face-2
// stepping (_llk_math_eltwise_sfpu_apply_vector_mode_) composes unchanged.
template <int BINOP>
sfpi_inline void cskip_binop_body(uint32_t param) {
    const sfpi::vFloat p = ckernel::sfpu::Converter::as_float(param);
    for (int rg = 0; rg < 4; ++rg) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        if constexpr (BINOP == ckernel::sfpu::MUL) {
            sfpi::dst_reg[0] = v * p;
        } else {
            sfpi::dst_reg[0] = v + p;
        }
        sfpi::dst_reg += 2;  // skip the odd-parity vector (cols 1,3,..,15)
    }
}
sfpi_inline void cskip_rsqrt_body() {
    for (int rg = 0; rg < 4; ++rg) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}

// --- SCALE body: *(1/W) and +eps in ONE pass (two cheap ops, one DEST round-trip) ---
template <int STRIDE, int ITERS>
sfpi_inline void scale_body(uint32_t inv_w, uint32_t eps) {
    const sfpi::vFloat iw = ckernel::sfpu::Converter::as_float(inv_w);
    const sfpi::vFloat ep = ckernel::sfpu::Converter::as_float(eps);
    for (int i = 0; i < ITERS; ++i) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * iw + ep;
        sfpi::dst_reg += STRIDE;
    }
}

// --- FULLY FUSED body: *(1/W), +eps AND rsqrt in ONE pass over DEST ----------------
//
// MEASURED SFPI-COMPILER CONSTRAINT (this is why the body is written this way, not the
// obvious way).  The obvious fusion --
//     v = dst_reg[0] * iw + ep;  t = _calculate_sqrt_body_(v);
// -- holds TWO loop-invariant vFloat constants (1/W and eps) live across the heavy
// rsqrt body and the sfpi register allocator HARD-ERRORS:
//     sfpi_funcs.h:481: internal compiler error: cannot store sfpu register
//                       (register spill)
// It fails at ITERS=8 and at ITERS=4, with and without `#pragma GCC unroll 1`.  The
// stock `_calculate_sqrt_internal_` can afford `unroll 8` only because it holds no
// constants of its own.
//
// The way out is algebra, not a pragma:
//     rsqrt(v/W + eps)  ==  sqrt(W) * rsqrt(v + eps*W)
// which needs ONE live constant (eps*W) plus a final multiply by sqrt(W).  When W is a
// POWER OF FOUR, sqrt(W) = 2^(log2(W)/2) exactly, so that multiply is an EXPONENT ADD
// (`sfpi::addexp`) -- no constant register at all, one instruction.  ONE live constant
// fits, and the fusion compiles.
//
// Fusing also removes TWO intermediate roundings: at fp32_dest_acc_en == False a DEST
// word is 16-bit, so the 3-call chain rounds the `*1/W` result and the `+eps` result to
// bf16 on the way through DEST; the fused body keeps both in an fp32 LREG.  Accuracy is
// therefore >= the baseline, not a trade.
template <int STRIDE, int ITERS, int HALF_LOG2W>
sfpi_inline void fused_body(uint32_t eps_times_w) {
    const sfpi::vFloat epw = ckernel::sfpu::Converter::as_float(eps_times_w);
    for (int i = 0; i < ITERS; ++i) {
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0] + epw);
        t = sfpi::addexp(t, HALF_LOG2W);  // x sqrt(W), exactly, as an exponent add
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}

ALWI void cskip_mul(uint32_t idst, uint32_t p) {
    _llk_math_eltwise_unary_sfpu_params_(cskip_binop_body<ckernel::sfpu::MUL>, idst, VectorMode::C, p);
}
ALWI void cskip_add(uint32_t idst, uint32_t p) {
    _llk_math_eltwise_unary_sfpu_params_(cskip_binop_body<ckernel::sfpu::ADD>, idst, VectorMode::C, p);
}
ALWI void cskip_rsqrt(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(cskip_rsqrt_body, idst, VectorMode::C);
}
// cskip2's first pass: *(1/W) and +eps together, 4 even-parity vectors per face.
ALWI void cskip_scale(uint32_t idst, uint32_t iw, uint32_t eps) {
    _llk_math_eltwise_unary_sfpu_params_(scale_body<2, 4>, idst, VectorMode::C, iw, eps);
}
// cskip_fused: the whole finalize, 4 even-parity vectors per face -> column 0 exactly.
template <int HALF_LOG2W>
ALWI void cskip_fused(uint32_t idst, uint32_t eps_w) {
    _llk_math_eltwise_unary_sfpu_params_(fused_body<2, 4, HALF_LOG2W>, idst, VectorMode::C, eps_w);
}
// fused_c: the whole finalize at plain VectorMode::C -- 8 contiguous vectors per face.
template <int HALF_LOG2W>
ALWI void fused_c(uint32_t idst, uint32_t eps_w) {
    _llk_math_eltwise_unary_sfpu_params_(fused_body<1, 8, HALF_LOG2W>, idst, VectorMode::C, eps_w);
}

#endif  // TRISC_MATH

// =====================================================================================
// The finalize chain, one definition per variant.  `finalize<V, RFS_IW, RFS_EPS>(dst)` is what
// every measurement mode and every handoff structure calls, so the levers are
// orthogonal by construction.
// =====================================================================================
#define RFS_FINALIZE_TPARAMS uint32_t V, uint32_t RFS_IW, uint32_t RFS_EPS, uint32_t RFS_EPSW, int RFS_HALFLOG2W
#define RFS_FINALIZE_TARGS V, RFS_IW, RFS_EPS, RFS_EPSW, RFS_HALFLOG2W

template <RFS_FINALIZE_TPARAMS>
ALWI void finalize_init() {
    if constexpr (V <= 3) {
        binop_with_scalar_tile_init();
    }
    rsqrt_tile_init();
}

// The SFPU payload only (no init), so MODE_SFPU can hoist every init out of the timed
// zone and the zone measures the SFPU work and nothing else.
template <RFS_FINALIZE_TPARAMS>
ALWI void finalize_payload(uint32_t dst) {
    if constexpr (V == 0) {  // rc_all -- the pre-L6b whole-tile chain
        mul_unary_tile(dst, RFS_IW);
        add_unary_tile(dst, RFS_EPS);
        rsqrt_tile(dst);
    } else if constexpr (V == 1) {  // base -- THE OP TODAY (L6b: rsqrt only, at C)
        mul_unary_tile(dst, RFS_IW);
        add_unary_tile(dst, RFS_EPS);
        MATH((rsqrt_tile_col(dst)));
    } else if constexpr (V == 2) {  // scope_c -- all three at VectorMode::C
        MATH((mul_unary_tile_col(dst, RFS_IW)));
        MATH((add_unary_tile_col(dst, RFS_EPS)));
        MATH((rsqrt_tile_col(dst)));
    } else if constexpr (V == 3) {  // cskip3 -- all three at C + even-parity stride
        MATH((cskip_mul(dst, RFS_IW)));
        MATH((cskip_add(dst, RFS_EPS)));
        MATH((cskip_rsqrt(dst)));
    } else if constexpr (V == 4) {  // cskip2 -- (mul,add) fused, then rsqrt; both c_skip
        MATH((cskip_scale(dst, RFS_IW, RFS_EPS)));
        MATH((cskip_rsqrt(dst)));
    } else if constexpr (V == 5) {  // fused_c -- one body, VectorMode::C
        MATH((fused_c<RFS_HALFLOG2W>(dst, RFS_EPSW)));
    } else {  // cskip_fused -- one body, C + even-parity stride
        MATH((cskip_fused<RFS_HALFLOG2W>(dst, RFS_EPSW)));
    }
}

// The op's finalize lambda: init + payload, exactly as rms_norm_compute.cpp spells it
// (the inits are inside the per-tile transform there, so they are inside here too).
template <RFS_FINALIZE_TPARAMS>
ALWI void finalize(uint32_t dst) {
    finalize_init<RFS_FINALIZE_TARGS>();
    finalize_payload<RFS_FINALIZE_TARGS>(dst);
}

// =====================================================================================
// (b) The A->B transform.
//
// `transform_to` is the two-CB twin of `ckl::transform_in_place`
// (streaming_reduce_helpers.inl:75): identical body, except the pack targets a
// DIFFERENT CB, so the finalize and the handoff become ONE pass -- one unpack and
// one pack per tile instead of two of each.  The library has no such helper; see the
// LIBRARY GAP note in the module docstring.
// =====================================================================================
template <typename Transform>
ALWI void transform_to(uint32_t cb_in, uint32_t cb_o, Transform t) {
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_o);
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    t(0);
    tile_regs_commit();
    cb_pop_front(cb_in, 1);
    cb_reserve_back(cb_o, 1);
    tile_regs_wait();
    pack_tile(0, cb_o);
    tile_regs_release();
    cb_push_back(cb_o, 1);
}

// The SAME A->B pass through the helper family: a user-defined chain element on the
// documented `UnaryOp<Derived, Slot>` CRTP surface (eltwise_chain.inl:627-660).  Stock
// elements (MulUnary/AddUnary/Rsqrt) cannot express the scope, so the element carries
// the scoped chain itself.
template <RFS_FINALIZE_TPARAMS>
struct FinalizeElem : ckl::UnaryOp<FinalizeElem<RFS_FINALIZE_TARGS>, ckl::Dst::D0> {
    static ALWI void init() { finalize_init<RFS_FINALIZE_TARGS>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { finalize_payload<RFS_FINALIZE_TARGS>(slot_offset); }
};

void kernel_main() {
    constexpr uint32_t RFS_MODE = get_compile_time_arg_val(0);
    constexpr uint32_t RFS_VAR = get_compile_time_arg_val(1);
    constexpr uint32_t RFS_HOFF = get_compile_time_arg_val(2);
    constexpr uint32_t RFS_IW = get_compile_time_arg_val(3);
    constexpr uint32_t RFS_EPS = get_compile_time_arg_val(4);
    constexpr uint32_t RFS_ROWS = get_compile_time_arg_val(5);
    constexpr uint32_t RFS_EPSW = get_compile_time_arg_val(6);       // bits of eps * W
    constexpr int RFS_HALFLOG2W = (int)get_compile_time_arg_val(7);  // log2(W)/2 (W a power of 4)
    constexpr uint32_t V = RFS_VAR;  // RFS_FINALIZE_TARGS names the variant `V`

    if constexpr (RFS_MODE == 0) {
        // ---------------- ISOLATED: pure SFPU cost on the MATH thread ----------------
        const uint32_t reps = get_arg_val<uint32_t>(0);
        compute_kernel_hw_startup(cb_stat, cb_stat, cb_out);
        copy_tile_init(cb_stat);
        finalize_init<RFS_FINALIZE_TARGS>();  // every init OUTSIDE the timed zone

        cb_reserve_back(cb_stat, 1);
        cb_push_back(cb_stat, 1);  // sharded input already resident
        cb_wait_front(cb_stat, 1);

        tile_regs_acquire();
        copy_tile(cb_stat, 0, 0);  // seed DEST[0] once -- outside the zone
        {
            DeviceZoneScopedN("RFS_SFPU");
            for (uint32_t r = 0; r < reps; ++r) {
                finalize_payload<RFS_FINALIZE_TARGS>(0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_stat, 1);
    } else if constexpr (RFS_MODE == 1) {
        // ---------------- STRUCTURAL: the op's stage sequence, per-stage zones -------
        auto fin = [](uint32_t dst) { finalize<RFS_FINALIZE_TARGS>(dst); };
        compute_kernel_hw_startup(cb_stat, cb_stat, cb_out);
        cb_reserve_back(cb_stat, RFS_ROWS);
        cb_push_back(cb_stat, RFS_ROWS);  // the stat block is already resident in L1

        if constexpr (RFS_HOFF == 0) {
            // THE OP TODAY: finalize in place, then copy the block to the handoff CB.
            {
                MaybeDeviceZoneScope("bench_finalize");
                for (uint32_t i = 0; i < RFS_ROWS; ++i) {
                    ckl::transform_in_place(cb_stat, fin);
                }
            }
            {
                MaybeDeviceZoneScope("bench_handoff");
                ckl::copy<ckl::input(cb_stat), ckl::output(cb_out)>(ckl::EltwiseShape::tiles(RFS_ROWS));
            }
        } else if constexpr (RFS_HOFF == 1) {
            // (b): ONE pass, A -> B.  There is no separate handoff stage to time.
            MaybeDeviceZoneScope("bench_finalize");
            for (uint32_t i = 0; i < RFS_ROWS; ++i) {
                transform_to(cb_stat, cb_out, fin);
            }
        } else {
            // (b) through the helper family: one eltwise_chain call, A -> B.
            MaybeDeviceZoneScope("bench_finalize");
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(RFS_ROWS),
                ckl::CopyTile<ckl::input(cb_stat)>{},
                FinalizeElem<RFS_FINALIZE_TARGS>{},
                ckl::PackTile<ckl::output(cb_out)>{});
        }
    } else {
        // ---------------- CONSUMER: does pass B ever read a stale lane? --------------
        // RFS_HOFF doubles as "run the finalize first" here:
        //   0  finalize the stat, then consume it   (the op's real sequence)
        //   1  consume the RAW stat, no finalize    (the PURE lane test: which lanes of a
        //      tile does BroadcastDim::Col read?  The raw stat's column 0 is ~4096 and its
        //      other columns are 2e4..3.2e5, so a leak is unmissable.)
        auto fin = [](uint32_t dst) { finalize<RFS_FINALIZE_TARGS>(dst); };
        compute_kernel_hw_startup(cb_x, cb_stat, cb_out);
        cb_reserve_back(cb_stat, RFS_ROWS);
        cb_push_back(cb_stat, RFS_ROWS);
        cb_reserve_back(cb_x, RFS_ROWS);
        cb_push_back(cb_x, RFS_ROWS);

        if constexpr (RFS_HOFF == 0) {
            for (uint32_t i = 0; i < RFS_ROWS; ++i) {
                ckl::transform_in_place(cb_stat, fin);
            }
            // Second stage, second pack-output CB (bf16 out vs the fp32 stat) -> one boot
            // per stage, which is eltwise_chain.hpp's documented placement rule.
            compute_kernel_hw_startup(cb_x, cb_stat, cb_out);
        }
        // Pass B's REAL consumer, verbatim from rms_norm_compute.cpp: the stat is a
        // REDUCE_ROW result, so it broadcasts back ACROSS columns (BroadcastDim::Col)
        // and is operand B; OperandKind::Col indexes it by row only.
        ckl::eltwise_chain(
            ckl::EltwiseShape::grid(RFS_ROWS, 1),
            ckl::BinaryFpu<
                ckl::input(cb_x, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(cb_stat, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Col>{},
            ckl::PackTile<ckl::output(cb_out)>{});
        cb_pop_front(cb_stat, RFS_ROWS);
    }
}
"""


def _compute_config():
    """The PINNED precision contract -- identical for every variant."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def create_program_descriptor(
    stat_tensor,
    out_tensor,
    *,
    mode,
    variant,
    handoff=BASELINE_HANDOFF,
    inv_w_bits,
    eps_bits,
    eps_w_bits,
    half_log2_w,
    rows,
    reps=1,
    x_tensor=None,
):
    if mode not in _MODE_ID:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if variant not in _VAR_ID:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if handoff not in _HOFF_ID:
        raise ValueError(f"handoff must be one of {HANDOFFS}, got {handoff!r}")
    if rows < 1 or reps < 1:
        raise ValueError("rows and reps must be positive")
    if stat_tensor.dtype != ttnn.float32 or stat_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("the stat tensor must be float32 TILE_LAYOUT (the op's cb_row_stat format)")
    if mode == "consumer" and x_tensor is None:
        raise ValueError("consumer mode needs an x tensor")
    if variant in NEEDS_POW4_W and half_log2_w < 0:
        raise ValueError(
            f"{variant} fuses the whole chain as sqrt(W) * rsqrt(v + eps*W) and applies the sqrt(W) as an "
            f"EXPONENT ADD, so it needs W to be a power of FOUR (see the fused_body comment)"
        )

    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            _MODE_ID[mode],
            _VAR_ID[variant],
            _HOFF_ID[handoff],
            inv_w_bits,
            eps_bits,
            rows,
            eps_w_bits,
            max(half_log2_w, 0),
        ],
        runtime_args=[(ttnn.CoreCoord(0, 0), [reps])],
        config=_compute_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT, stat_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out_tensor),
    ]
    # generic_op takes io_tensors.back() as THE OUTPUT
    # (generic_op_device_operation.cpp:133), so out_tensor MUST be last.  Getting this
    # wrong returns the wrong tensor and silently looks like "the compute did nothing".
    tensors = [stat_tensor]
    if x_tensor is not None:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X, x_tensor))
        tensors.append(x_tensor)
    tensors.append(out_tensor)
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs), tensors


def run_op(stat_tensor, out_tensor, **kwargs):
    descriptor, tensors = create_program_descriptor(stat_tensor, out_tensor, **kwargs)
    return ttnn.generic_op(tensors, descriptor)
