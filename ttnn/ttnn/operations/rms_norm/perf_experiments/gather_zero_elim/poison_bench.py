# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED SAFETY MEASUREMENT: is the gather CB's boot-zeroing needed AT ALL?

THE STAGE, and the claim under test
-----------------------------------
On a width/block-sharded rms_norm the group ROOT boot-zeroes, once, every face of
`cb_partials_gathered` the gather never ships -- faces 1 and 3 at GATHER_FACES == 2,
plus any odd-GROUP_SIZE pad slot (`writer_gather_zero`, rms_norm_writer.cpp).  The
recorded justification for it is REASONED, not measured:

    "faces 1/3 garbage cannot reach column 0 through the FPU fold or the
     column-scoped finalize"

This bench converts that into a MEASUREMENT, the way D17 and D23 measured theirs:
SEED THE UNSHIPPED LANES CATASTROPHICALLY WRONG AND LOOK AT PCC / rel-RMS.

Why faces 1/3 are the question.  A cross-core partial is a REDUCE_ROW result -- a
COLUMN vector, meaningful in column 0 only.  A 32x32 tile is 2x2 faces of 16x16
(1024 B each at offsets 0 / 1024 / 2048 / 3072); columns 0..15 live in faces 0 and 2
and columns 16..31 in faces 1 and 3.  GATHER_FACES == 2 ships faces 0 and 2 only, so
columns 16..31 of every landed page hold WHATEVER WAS IN THE ROOT'S L1.

The chain that then consumes the page (post-D22) is:
  1  `add_tiles(cb_partials_gathered, cb_partials_gathered, i, j, 0)` with
     acc_to_dest -- an FPU elementwise add.  The FPU has NO lane scope, so the
     garbage DOES enter DEST.  (Control below: we CHECK that it did.)
  2  `stat_finalize_payload<INV_W,EPS>(0)` -- raw sfpi, <STRIDE=2, ITERS=4> at
     VectorMode::C, i.e. an even-parity walk that reaches faces 0 and 2 only.
  3  `pack_tile(0, cb_stat_handoff)` -- packs the WHOLE tile, so the garbage
     DOES reach L1 (and the multicast then copies it to every member verbatim).
  4  pass B: `BinaryFpu<x, stat, Mul, BroadcastDim::Col>` with `OperandKind::Col`,
     which reads COLUMN 0 only.
So the claim is "carried but never read".  The op has twice been bitten by errors
that held pcc >= 0.9997 and showed ONLY in rel-RMS, so BOTH are reported.

What this bench runs
--------------------
ONE Tensix core, compute only, no NoC in the datapath.  The gathered partials are a
resident L1 shard laid out exactly as the real gather leaves them (row-major,
`page = r * GATHER_SLOTS + slot`, D16), with columns 0..15 of every LIVE slot holding
the member's real partial (column 0) and the SEED in columns 16..31.  Then, verbatim
from rms_norm_compute.cpp:

  * the D22 fused root chain -- the MANDATORY `reconfig_data_format` /
    `pack_reconfig_data_format` pair, `add_tiles_init(acc_to_dest=true)`, the pairwise
    GATHER_SLOTS/2 walk, `stat_finalize_payload`, one `pack_tile` per tile-row into an
    fp32 stat CB;
  * pass B's EXACT consumer -- `eltwise_chain(grid(rows, WT, PASS_B_BLK),
    BinaryFpu<x, stat(OperandKind::Col), Mul, BroadcastDim::Col>, PackTile<>)`.

The multicast between them is a whole-tile byte copy (writer `noc_async_write` of
`rows * stat_bytes`), so it preserves the garbage exactly and is not reconstructed;
pass B reads the stat CB the fold packed.  The gamma multiply after pass B is a
second elementwise that never touches the stat, so it is not reconstructed either.

FIXED, never a lever: bf16 activations / fp32 partials + fp32 stat / TILE layout /
MathFidelity.HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False -- the focus
case's pinned contract, identical for every seed.

THE SEEDS (a single benign seed proves nothing)
-----------------------------------------------
`face_seed` fills columns 16..31 (faces 1 and 3) of every live slot:
    zero      0.0                       -- THE REFERENCE.  What the boot achieves.
    big_pos   +1e30                     -- 30+ orders above a real partial
    big_neg   -1e30
    nan       NaN
    inf       +Inf
    ninf      -Inf
    denorm    1e-42 (fp32 subnormal)
    mixed     per-lane cycle of {NaN, +Inf, -Inf, 1e30, -1e30, 1e-42, -3.4e38, 1e-30}
              -- so the pairwise fold actually evaluates Inf + (-Inf) = NaN
    l1_like   +-1e20 * randn, a plausible stale-L1 pattern
`pad_seed` fills the WHOLE pad page (odd GROUP_SIZE only -- a pad page is folded
whole, so it is a DIFFERENT question from faces 1/3 and is reported separately):
    zero / big_pos / nan / mixed        -- whole page
    faces13                             -- pad page poisoned in columns 16..31 ONLY
                                           (faces 0/2 exactly zero), i.e. the probe
                                           for "does the pad need faces 1/3 zeroed
                                           too, or only faces 0/2?"

CONTROL.  Every run also reports what fraction of the PACKED STAT TILE's columns
16..31 came back non-finite (`stat_hi_nonfinite`) and the max |value| there
(`stat_hi_absmax`).  A poisoned run whose stat tile is clean would mean the poison
never entered the datapath and the correctness result would be vacuous.  It is the
evidence that the garbage IS carried.

------------------------------------------------------------------------------
MEASURED  (blackhole p150b, 1350 MHz; bf16 act / fp32 partials+stat / TILE /
MathFidelity.HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False -- UNCHANGED for
every seed).  Full table: measurements.txt section A.
------------------------------------------------------------------------------
FACES 1/3 (the unshipped faces): SAFE, measured, everywhere run.
  All 9 seeds at the FOCUS geometry (GROUP_SIZE 8, rows 8, WT 4) and the worst seed
  (`mixed`) at 13 geometries (GROUP_SIZE 4/8/9/28/32 x rows 1/8/32, plus WT 8) produced
  output BIT-IDENTICAL to the boot-zeroed run (`torch.equal`), and a BIT-IDENTICAL stat
  column 0.  pcc_out 0.999990-0.999995 / rel-RMS 3.3e-3-5.1e-3, matching the zeroed
  reference to every digit printed.
  THE CONTROL FIRED: the packed stat tile's columns 16..31 came back 100% non-finite for
  the NaN / +-Inf seeds, 50% non-finite for `mixed`, and |max| 7.96e30 for the 1e30
  seeds.  So the garbage DID enter DEST through the acc_to_dest FPU fold and DID get
  packed to L1 -- it is carried and never read, exactly as claimed.  Denormals flush;
  Inf + (-Inf) = NaN is evaluated in those lanes and stays in them.

ODD-GROUP_SIZE PAD PAGE: LOAD-BEARING.  Poisoning the pad page whole is catastrophic at
GROUP_SIZE 9, rows 1 and 8:
    pad=big_pos  pcc_out 0.999672 / 0.999757   rel-RMS 1.00   <- pcc BARELY moves
    pad=nan      pcc_out nan                   rel-RMS 1.00
    pad=mixed    pcc_out nan                   rel-RMS 1.00
  `big_pos` is the exact failure shape this op has been bitten by twice: a uniform scale
  error keeps pcc at 0.9997 and shows ONLY in rel-RMS.
  BUT the pad needs only its SHIPPED faces defined: `pad=faces13` (pad page's columns
  16..31 poisoned with `mixed`, columns 0..15 exactly zero) is BIT-IDENTICAL at both
  geometries.  So the pad's requirement is faces 0 and 2, not the whole page.
"""

import ttnn

TILE = 32

CB_X = 0  # bf16 activations, resident shard        == cb_input_tiles
CB_PART = 1  # fp32 gathered partials, resident shard  == cb_partials_gathered
CB_STAT = 2  # fp32 finalized stat, resident shard     == cb_stat_handoff / cb_row_final
CB_OUT = 3  # bf16 normalized output, resident shard  == cb_normalized

FACE_SEEDS = ("zero", "big_pos", "big_neg", "nan", "inf", "ninf", "denorm", "mixed", "l1_like")
PAD_SEEDS = ("zero", "big_pos", "nan", "mixed", "faces13")

REFERENCE = ("zero", "zero")


_KERNEL = r"""
// =============================================================================
// rms_norm perf experiment: gather_zero_elim  (ISOLATED SAFETY BENCH KERNEL)
// =============================================================================
// The op's D22 fused root chain followed by the op's EXACT pass-B consumer, on a
// resident gather shard whose UNSHIPPED faces the host seeded catastrophically wrong.
// Nothing here is a perf variant: every seed runs the SAME kernel, so the only thing
// that differs between runs is the bytes in columns 16..31 of the gather pages (and,
// at odd GROUP_SIZE, the pad page).  That is what makes the pcc / rel-RMS delta
// attributable to the unshipped lanes alone.
//
// RAW LLK: the fold and the finalize are copied VERBATIM from
// rms_norm_compute.cpp (D22 + D17) rather than re-derived, so the SFPU spelling,
// the VectorMode::C lane walk and the acc_to_dest FPU accumulation are
// bit-identical to the op's.  Re-deriving them would measure a different kernel.
// =============================================================================
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"              // ckernel::sfpu::_calculate_sqrt_body_
#include "ckernel_sfpu_binop_with_unary.h"  // ckernel::sfpu::Converter::as_float
#endif

namespace ckl = compute_kernel_lib;
using ckernel::VectorMode;

// ---- the op's finalize, VERBATIM (Perf 1 / D17 `cskip2`) ---------------------
// INVARIANT (carried over): STRIDE/ITERS are <2,4> in BOTH bodies and their product is
// 8, so `*(1/W)+eps` and `rsqrt` visit EXACTLY the same lane set -- an all-zero row can
// never become rsqrt(0) = inf.
#ifdef TRISC_MATH
template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_scale_body(uint32_t inv_w_bits, uint32_t eps_bits) {
    const sfpi::vFloat iw = ckernel::sfpu::Converter::as_float(inv_w_bits);
    const sfpi::vFloat ep = ckernel::sfpu::Converter::as_float(eps_bits);
    for (int i = 0; i < ITERS; ++i) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * iw + ep;
        sfpi::dst_reg += STRIDE;
    }
}

template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_rsqrt_body() {
    for (int i = 0; i < ITERS; ++i) {
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}

ALWI void stat_scale_col_skip(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2, 4>, idst, VectorMode::C, inv_w_bits, eps_bits);
}
ALWI void rsqrt_tile_col_skip(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<2, 4>, idst, VectorMode::C);
}
#endif  // TRISC_MATH

template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
ALWI void stat_finalize_payload(uint32_t dst) {
    MATH((stat_scale_col_skip(dst, RMS_INV_W, RMS_EPS)));
    MATH((rsqrt_tile_col_skip(dst)));
}

// The op's pass-B DEST-lane block size (D21), same definition.
constexpr uint32_t pass_b_blk(uint32_t wt, uint32_t cap) {
    uint32_t b = (cap < wt) ? cap : wt;
    while (b > 1 && (wt % b) != 0) {
        --b;
    }
    return b;
}

void kernel_main() {
    constexpr uint32_t ROWS = get_compile_time_arg_val(0);   // tile-rows in this row-block
    constexpr uint32_t GP = get_compile_time_arg_val(1);     // GATHER_SLOTS landed per tile-row
    constexpr uint32_t WT = get_compile_time_arg_val(2);     // width tiles of this core's slice
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);

    constexpr uint32_t cb_x = 0, cb_part = 1, cb_stat = 2, cb_out = 3;
    constexpr uint32_t GATHER_HALF = GP / 2;
    constexpr uint32_t PASS_B_BLK = pass_b_blk(WT, ckl::DEST_AUTO_LIMIT);

    // Start the hardware in the state PASS A leaves it: unpacker on the bf16
    // activations, packer on a bf16 target.  That is what makes the fold's
    // reconfig pair MANDATORY here, exactly as it is in the op.
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);

    // Expose the resident shards as this round's CB fronts.
    cb_reserve_back(cb_x, ROWS * WT);
    cb_push_back(cb_x, ROWS * WT);
    cb_reserve_back(cb_part, GP * ROWS);
    cb_push_back(cb_part, GP * ROWS);

    // ================= the op's D22 FUSED ROOT CHAIN, verbatim =================
    {
        MaybeDeviceZoneScope("bench_root_fused");
        cb_wait_front(cb_part, GP * ROWS);
        // NOT optional: pass A left the unpacker on bf16 and the packer on bf16, while
        // the gather and the handoff are fp32.  Without these the fold unpacks fp32 L1
        // through a bf16 srcA/srcB and the sum reads as ~0 -- an error that keeps pcc at
        // 0.9997 and shows only in rel-RMS (the op's own integration bug).
        reconfig_data_format(cb_part, cb_part);
        pack_reconfig_data_format(cb_stat);
        add_tiles_init(cb_part, cb_part, /*acc_to_dest=*/true);
        rsqrt_tile_init();
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t base = r * GP;
            tile_regs_acquire();
            for (uint32_t p = 0; p < GATHER_HALF; ++p) {
                add_tiles(cb_part, cb_part, base + p, base + GATHER_HALF + p, 0);
            }
            stat_finalize_payload<INV_W_BITS, EPS_BITS>(0);
            tile_regs_commit();
            cb_reserve_back(cb_stat, 1);
            tile_regs_wait();
            pack_tile(0, cb_stat);
            tile_regs_release();
            cb_push_back(cb_stat, 1);
        }
        cb_pop_front(cb_part, GP * ROWS);
    }

    // ================= pass B's EXACT consumer =================================
    // x * (1/rms).  The stat is a REDUCE_ROW result: column-shaped, so it broadcasts
    // back ACROSS columns (BroadcastDim::Col) and must be operand B; OperandKind::Col
    // indexes it by row only.  This is the ONLY reader of the stat tile in the op.
    {
        MaybeDeviceZoneScope("bench_pass_b");
        ckl::eltwise_chain(
            ckl::EltwiseShape::grid(ROWS, WT, PASS_B_BLK),
            ckl::BinaryFpu<
                ckl::input(cb_x, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block),
                ckl::input(cb_stat, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Col>{},
            ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk)>{});
    }
    cb_pop_front(cb_stat, ROWS);
    cb_pop_front(cb_x, ROWS * WT);
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded(h_tiles, w_tiles=1):
    """The whole [h_tiles x w_tiles] tile matrix as one shard on one core (tiles row-major)."""
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def perf_case_config():
    """The focus case's pinned compute config -- FIXED, identical for every seed."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def gather_slots(group_size):
    """The op's derived layout: GROUP_SIZE rounded UP TO EVEN (D22)."""
    return group_size + group_size % 2


def l1_fp32_pages(group_size, rows, wt):
    """fp32-tile-page equivalents this geometry pins in L1 (bf16 pages count 1/2)."""
    gp = gather_slots(group_size)
    return gp * rows + rows + rows * wt  # part(fp32) + stat(fp32) + x/out (bf16, 2 x 1/2)


def _f32_bits(x):
    import struct

    return int(struct.unpack("<I", struct.pack("<f", float(x)))[0])


# ---------------------------------------------------------------------------
# the seeds
# ---------------------------------------------------------------------------

_MIXED_CYCLE = (float("nan"), float("inf"), float("-inf"), 1e30, -1e30, 1e-42, -3.4e38, 1e-30)


def _seed_block(name, shape, gen):
    """A [*shape] fp32 block of the named catastrophic pattern."""
    import torch

    if name == "zero":
        return torch.zeros(shape, dtype=torch.float32)
    if name == "big_pos":
        return torch.full(shape, 1e30, dtype=torch.float32)
    if name == "big_neg":
        return torch.full(shape, -1e30, dtype=torch.float32)
    if name == "nan":
        return torch.full(shape, float("nan"), dtype=torch.float32)
    if name == "inf":
        return torch.full(shape, float("inf"), dtype=torch.float32)
    if name == "ninf":
        return torch.full(shape, float("-inf"), dtype=torch.float32)
    if name == "denorm":
        return torch.full(shape, 1e-42, dtype=torch.float32)
    if name == "mixed":
        flat = torch.tensor(_MIXED_CYCLE, dtype=torch.float32)
        n = 1
        for s in shape:
            n *= s
        return flat.repeat((n + len(_MIXED_CYCLE) - 1) // len(_MIXED_CYCLE))[:n].reshape(shape).clone()
    if name == "l1_like":
        return (torch.randn(shape, generator=gen) * 1e20).to(torch.float32)
    raise ValueError(f"gather_zero_elim: unknown seed {name!r}")


# ---------------------------------------------------------------------------
# host-side reference (float64 -- carries no error of its own)
# ---------------------------------------------------------------------------


def reference(group_size, rows, wt, seed, eps):
    """Realistic partials plus the exact group sum and the exact finalized stat.

    Each partial is what a member core's REDUCE_ROW actually produces: `sum(x^2)` over
    that core's `wt * 32` columns of a bf16 activation row, held in COLUMN 0 of an fp32
    tile.  `x` is this (root) core's own bf16 slice, which pass B scales.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    w_per_core = wt * TILE
    W = group_size * w_per_core
    all_x = torch.randn(rows * TILE, W, generator=gen).to(torch.bfloat16).to(torch.float64)
    partials = torch.empty(group_size, rows, TILE, dtype=torch.float64)
    for g in range(group_size):
        s = (all_x[:, g * w_per_core : (g + 1) * w_per_core] ** 2).sum(dim=1)
        partials[g] = s.reshape(rows, TILE)
    exact_sum = partials.sum(dim=0)
    exact_stat = torch.rsqrt(exact_sum / W + eps)
    x = all_x[:, :w_per_core]  # the root's own slice == what pass B scales here
    return x, partials, exact_sum, exact_stat


def _pages(partials, group_size, rows, gp, face_seed, pad_seed, gen):
    """Lay the partials out in the gather CB's ROW-MAJOR page order (`page = r*gp + g`),
    then apply the seeds to exactly the bytes the gather never writes.

    Columns 0..15 == faces 0 and 2 == what GATHER_FACES == 2 SHIPS: column 0 holds the
    member's real partial and columns 1..15 hold zero (held constant across every seed,
    so they can never be what a delta is attributable to).
    Columns 16..31 == faces 1 and 3 == UNSHIPPED -> `face_seed`.
    Slots >= group_size are PAD pages (odd GROUP_SIZE only) -> `pad_seed`, whole page,
    except `faces13` which poisons only their columns 16..31.
    """
    import torch

    pages = torch.zeros(gp * rows, TILE, TILE, dtype=torch.float32)
    hi = _seed_block(face_seed, (TILE, TILE - 16), gen)
    for r in range(rows):
        for g in range(group_size):
            p = r * gp + g
            pages[p, :, 0] = partials[g, r].to(torch.float32)
            pages[p, :, 16:] = hi
        for g in range(group_size, gp):  # pad slots
            p = r * gp + g
            if pad_seed == "faces13":
                pages[p, :, 16:] = _seed_block("mixed", (TILE, TILE - 16), gen)
            else:
                pages[p] = _seed_block(pad_seed, (TILE, TILE), gen)
    return pages.reshape(gp * rows * TILE, TILE)


def _pcc(a, b):
    import torch

    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(a.norm() * b.norm())
    if den == 0.0:
        return float("nan")
    return float((a @ b) / den)


def _rel_rms(got, ref):
    import torch

    if not torch.isfinite(got).all():
        return float("inf")
    return float((got - ref).norm() / ref.norm())


def run_seed(device, face_seed, pad_seed, group_size, rows, *, wt=4, seed=1234, eps=1e-5):
    """Run ONE seed on device and return its metrics plus the raw device tensors.

    Correctness is the only pass/fail; nothing here asserts a perf direction.
    """
    import torch

    if face_seed not in FACE_SEEDS:
        raise ValueError(f"face_seed must be one of {FACE_SEEDS}, got {face_seed!r}")
    if pad_seed not in PAD_SEEDS:
        raise ValueError(f"pad_seed must be one of {PAD_SEEDS}, got {pad_seed!r}")

    gp = gather_slots(group_size)
    gen = torch.Generator().manual_seed(seed + 7)
    W = group_size * wt * TILE
    x, partials, exact_sum, exact_stat = reference(group_size, rows, wt, seed, eps)
    pages = _pages(partials, group_size, rows, gp, face_seed, pad_seed, gen)

    x_dev = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(rows, wt),
    )
    part_dev = ttnn.from_torch(
        pages, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_sharded(gp * rows)
    )
    stat_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows * TILE, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, _sharded(rows)
    )
    out_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows * TILE, wt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, _sharded(rows, wt)
    )

    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[rows, gp, wt, _f32_bits(1.0 / float(W)), _f32_bits(eps)],
        config=perf_case_config(),
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[compute],
        semaphores=[],
        cbs=[
            ttnn.cb_descriptor_from_sharded_tensor(CB_X, x_dev),
            ttnn.cb_descriptor_from_sharded_tensor(CB_PART, part_dev),
            ttnn.cb_descriptor_from_sharded_tensor(CB_STAT, stat_dev),
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out_dev),
        ],
    )
    ttnn.generic_op([x_dev, part_dev, stat_dev, out_dev], descriptor)

    stat_raw = ttnn.to_torch(stat_dev).to(torch.float32)  # [rows*32, 32] fp32, WHOLE tile
    out_raw = ttnn.to_torch(out_dev).to(torch.float32)  # [rows*32, wt*32] bf16-valued

    # ---- the CONTROL: did the poison actually enter the datapath? -------------
    hi = stat_raw[:, 16:]
    lo0 = stat_raw[:, 0]
    res = {
        "face_seed": face_seed,
        "pad_seed": pad_seed,
        "group_size": group_size,
        "gather_slots": gp,
        "rows": rows,
        "wt": wt,
        "has_pad": gp != group_size,
        "stat_hi_nonfinite": float((~torch.isfinite(hi)).to(torch.float64).mean()),
        "stat_hi_absmax": float(hi[torch.isfinite(hi)].abs().max()) if torch.isfinite(hi).any() else float("inf"),
        "stat_col0_nonfinite": float((~torch.isfinite(lo0)).to(torch.float64).mean()),
        "out_nonfinite": float((~torch.isfinite(out_raw)).to(torch.float64).mean()),
    }

    # ---- accuracy of the quantity the op actually produces --------------------
    dev_stat = torch.stack([lo0[r * TILE : (r + 1) * TILE] for r in range(rows)]).to(torch.float64)  # [rows,32]
    res["rel_rms_stat"] = _rel_rms(dev_stat, exact_stat)
    res["pcc_stat"] = _pcc(dev_stat, exact_stat)
    exact_out = x * exact_stat.reshape(-1).unsqueeze(1)
    got_out = out_raw.to(torch.float64)
    res["rel_rms_out"] = _rel_rms(got_out, exact_out)
    res["pcc_out"] = _pcc(got_out, exact_out)
    return res, stat_raw, out_raw
