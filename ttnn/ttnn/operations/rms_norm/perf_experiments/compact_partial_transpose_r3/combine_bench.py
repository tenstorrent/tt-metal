# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""BENCH A (Perf 3): rms_norm's group-root COMBINE **compute** chain, PER-ROW vs COMPACT partials.

REBASE, not a re-derivation.  perf_experiments/compact_partial_transpose_r2 measured this idea
against the PRE-D22 root chain (D16 packer-accumulating fold + D17 column finalize fused with the
handoff by D19) and got 6398 -> 1732 ns/block.  D22 then replaced that baseline entirely, so the
old number is against code that no longer exists.  This file re-measures both sides against the
tree as it stands, in the SAME spelling:

  BASELINE `base_d22` -- kernels/rms_norm_compute.cpp, the `is_root != 0` branch, VERBATIM:
      cb_wait_front(cb_partials_gathered, GATHER_SLOTS * rows)   [once per block]
      reconfig_data_format(...) + pack_reconfig_data_format(...)  [MANDATORY -- see the kernel]
      add_tiles_init(acc_to_dest = true); rsqrt_tile_init();
      for r in rows:  acquire; GATHER_HALF x add_tiles(base+p, base+GATHER_HALF+p, 0);
                      stat_finalize_payload<2,4>(0); commit; reserve 1; pack_tile; push 1
      cb_pop_front(...)

  CANDIDATE `cand_recv` -- the SAME code with ONE window instead of `rows` windows, because a
      sender has already permuted its `rows` partials into `rows` COLUMNS of ONE tile.  So the
      root's whole block costs GATHER_HALF add_tiles, ONE finalize and ONE pack, independent of
      BLOCK_ROWS.  The finalize must WIDEN (see FIN) because the live data is now spread across
      columns 0..rows-1 instead of sitting in column 0.

  `base_d19` is kept as MODE 6 purely as a CROSS-ROUND CALIBRATION point: if it reproduces r2's
      number on this box, the two rounds' tables are comparable.

THE IDEA.  A partial sum(x^2) for one tile-row is a REDUCE_ROW result: 32 useful fp32 numbers in
COLUMN 0 of a tile.  If a sender permutes its BLOCK_ROWS partials into BLOCK_ROWS distinct COLUMNS
of ONE tile, the root folds GATHER_SLOTS tiles per BLOCK instead of per ROW, and finalizes ONE tile
instead of BLOCK_ROWS.  The permutation is ONE `matmul_tiles` against a one-hot tile (see the
RAW-LLK justification at the kernel head) -- NOT a transpose_wh.

Variants (MODE):
  0 base_d22     the CURRENT (D22) fused root chain.  `rows` DEST windows, `rows` output pages.
  1 cand_recv    compact: ONE DEST window over GATHER_SLOTS compact tiles + ONE widened finalize
                 -> ONE output page.  The un-permute is moved to the MULTICAST RECEIVERS, which
                 pay it in parallel (mode 5 measures that).
  2 cand_root    compact fold + ONE finalize, then the un-permute ON THE ROOT (serial, on the
                 critical path) -> `rows` output pages, so the mcast is unchanged.
  3 member_pack  what a sender ADDS to produce a compact partial: `rows` matmuls into one DEST
                 tile + one pack.  (Its baseline is ZERO: D18 has pass A's reduce pack straight
                 into cb_sum_handoff, so today a sender pays nothing at all for the handoff.)
  5 recv_unpack  what every multicast receiver ADDS under `cand_recv`: `rows` matmuls + `rows`
                 packs to get each row's stat back into column 0 of its own tile, which is what
                 pass B's mul<BroadcastDim::Col> reads.
  6 base_d19     r2's (pre-D22) baseline -- calibration only, never a graduation candidate.

Knobs that are OPTIONS, not correctness dials:
  FIN     0 = D17's <2,4> even-parity VectorMode::C  (columns 0,2,..,14 -- correct for a
              column-shaped stat, and for a compact tile ONLY at BLOCK_ROWS == 1)
          1 = <1,8> VectorMode::C   (columns 0..15  -- correct for a compact tile up to BR = 16)
          2 = <1,8> VectorMode::RC  (all 32 columns -- correct for any BR)
          The candidate MUST widen (1 or 2).  r2 MEASURED that D17's shipped <2,4> scope is
          silently wrong on a compact tile from BLOCK_ROWS = 2 (pcc 0.9974): it reaches only the
          EVEN columns, so the odd rows' stats are never scaled or rsqrt-ed.  FIN 0 is kept and
          measured at rows > 1 so that hazard stays visible as a NUMBER, not a caveat.
  BANK    0 = EF  : 2*rows+1 one-hot pages (E_r for the pack, F_r for the un-pack, + a zero page)
          1 = E_T : rows+1 pages -- the un-pack reuses E_r with matmul's srcB `transpose` flag,
              since E_r^T == F_r.  Halves the one-hot bank's L1.
  BANKDT  the bank tensor's dtype, host-side: float32 or bfloat16.  A one-hot tile is exactly
          representable in bf16, so this is a pure L1 lever (another 2x) -- measured, not assumed.
  SEED    0 = rely on the packer leaving DEST zeroed (llk_pack_dest_section_done's ZEROACC)
          1 = copy an explicit zero tile into each DEST slot first (portable, +1 op per tile)

Precision contract (FIXED, identical for EVERY variant): fp32 partial/stat CBs, bf16-equivalent
DEST (fp32_dest_acc_en = False), math_fidelity = HiFi2, math_approx_mode = False.  Nothing here
tunes those.

Everything is resident in one core's L1 (no DRAM, no NoC, no gather) so the measured delta is the
root chain's COMPUTE alone.  The gather WAIT and the transport are measured SEPARATELY by bench B
(transport_bench.py), which is where the GATHER_FACES byte-amplification lives.  The resident input
CB holds exactly the pages ONE call consumes and the kernel re-exposes them per call, so the torch
reference walks the same logical stream the kernel does even where the modelled stream is larger
than L1; L1 is uniform-latency and cacheless, so re-reading the same GATHER_SLOTS pages every
window costs exactly what walking `rows` distinct windows costs.
"""

import ttnn

TILE = 32
FP32_TILE_BYTES = TILE * TILE * 4

CB_PART = 0  # resident partials (fp32): cb_partials_gathered analogue
CB_BANK = 1  # resident one-hot bank
CB_OUT = 2  # result (fp32): cb_stat_handoff / cb_row_final analogue
CB_MID = 3  # cand_root only: the finalized compact tile before the root's un-permute
CB_ACC = 4  # base_d19 only: the pre-D22 fp32 L1 accumulator (cb_row_stat analogue)

MODES = {
    "base_d22": 0,
    "cand_recv": 1,
    "cand_root": 2,
    "member_pack": 3,
    "recv_unpack": 5,
    "base_d19": 6,
}

FIN_SKIP, FIN_C, FIN_RC = 0, 1, 2
BANK_EF, BANK_ET = 0, 1
SEED_NONE, SEED_COPY = 0, 1


def gather_slots(group_size):
    return group_size + group_size % 2


_KERNEL = r"""
// ISOLATED perf experiment for rms_norm's cross-core combine -- NOT the op's kernel.
//
// RAW-LLK / RAW-API JUSTIFICATION #1 (the fold): the baseline itself is raw LLK in the tree
//   today (D22's fused root chain), for a reason recorded there -- eltwise_chain runs every
//   element on every inner iteration, so a finalize placed after an accumulating BinaryFpu
//   would rsqrt a PARTIAL sum GATHER_HALF times.  Re-spelling the baseline with helpers would
//   make it slower than the op and inflate the candidate's speedup, so it is carried verbatim,
//   and the candidate is the SAME raw code with one window instead of `rows`.
//
// RAW-LLK / RAW-API JUSTIFICATION #2 (a column permutation by ONE matmul against a one-hot tile).
//   A core's partial sum(x^2) for one tile-row is a REDUCE_ROW result: 32 useful fp32 numbers in
//   COLUMN 0 of a tile.  Moving tile-row r's column into COLUMN r of one compact tile is a
//   HORIZONTAL data move, and the FPU's only horizontal-mixing primitive is the matmul:
//       pack   : C = partial_r x E_r ,  E_r[0][r] = 1  ->  C[i][r] = partial_r[i][0]
//       un-pack: C = compact  x F_r ,  F_r[r][0] = 1  ->  C[i][0] = compact[i][r]
//   and F_r == E_r^T, so with matmul's srcB `transpose` flag ONE bank of `rows` one-hot tiles
//   serves both directions.  Each move is ONE matmul_tiles.  No kernel_lib helper expresses a
//   column permutation (the eltwise / bcast / reduce families all preserve or collapse the column
//   axis; transpose_wh transposes the WHOLE tile, which is not this), so these two lines are the
//   raw compute API by necessity, not by preference.
//   SAFETY NOTE any later refactor must preserve: the matmul sums 32 products, so EVERY column of
//   BOTH operands must be FINITE -- an inf/NaN in an unused column becomes inf*0 = NaN and poisons
//   the result.  `pack_tile` always writes a WHOLE tile from a fully-defined DEST, so a partial
//   page holds only sums of squares (finite unless the input itself overflowed, in which case the
//   true sum is inf too) and DEST-zeroed lanes; the reader's pad-lane invariant (pad lanes are
//   multiplied by an exact 0 before the reduce) is what keeps the padding out of it.  A compact
//   tile's columns >= rows are EXACTLY 0 by construction.  This is ALSO why bench B's compact
//   gather ships WHOLE tiles: a partially-shipped compact page would leave un-written fp32 L1 in
//   the columns the receiver's un-pack matmul multiplies by zero.
//
// The finalize bodies below are VERBATIM from rms_norm_compute.cpp (D17) except that
// STRIDE / ITERS / VectorMode are template parameters, so the baseline runs the tree's exact
// spelling and the candidate can widen to the lanes a compact tile actually occupies.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_binop_with_unary.h"

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

// D17 as it stands in the tree: even-parity <2,4> over faces 0/2 -> columns 0,2,..,14.
ALWI void fin_skip(uint32_t idst, uint32_t iw, uint32_t eps) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2, 4>, idst, VectorMode::C, iw, eps);
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<2, 4>, idst, VectorMode::C);
}
// Widened to every column of faces 0/2 -> columns 0..15 (a compact tile up to rows = 16).
ALWI void fin_cfull(uint32_t idst, uint32_t iw, uint32_t eps) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<1, 8>, idst, VectorMode::C, iw, eps);
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<1, 8>, idst, VectorMode::C);
}
// The whole tile -> columns 0..31 (a compact tile above rows = 16 has live data in faces 1/3).
ALWI void fin_rc(uint32_t idst, uint32_t iw, uint32_t eps) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<1, 8>, idst, VectorMode::RC, iw, eps);
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<1, 8>, idst, VectorMode::RC);
}
#endif  // TRISC_MATH

// NB: `EPS` is a MACRO in llk_math_common_api.h, hence the RMS_ prefixes (the op does the same).
template <uint32_t RMS_IW, uint32_t RMS_EPS, uint32_t RMS_FIN>
ALWI void fin_payload(uint32_t dst) {
    if constexpr (RMS_FIN == 0) {
        MATH((fin_skip(dst, RMS_IW, RMS_EPS)));
    } else if constexpr (RMS_FIN == 1) {
        MATH((fin_cfull(dst, RMS_IW, RMS_EPS)));
    } else {
        MATH((fin_rc(dst, RMS_IW, RMS_EPS)));
    }
}

// The op's PRE-D22 (D19) chain element, kept for the base_d19 calibration variant only.
template <uint32_t RMS_IW, uint32_t RMS_EPS, uint32_t RMS_FIN>
struct StatFinalize : ckl::UnaryOp<StatFinalize<RMS_IW, RMS_EPS, RMS_FIN>, ckl::Dst::D0> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { fin_payload<RMS_IW, RMS_EPS, RMS_FIN>(slot_offset); }
};

namespace {
constexpr uint32_t cb_part = 0;
constexpr uint32_t cb_bank = 1;
constexpr uint32_t cb_out = 2;
constexpr uint32_t cb_mid = 3;
constexpr uint32_t cb_acc = 4;
}  // namespace

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t FIN = get_compile_time_arg_val(5);
    constexpr uint32_t BANK = get_compile_time_arg_val(6);   // 0 = EF, 1 = E + transpose
    constexpr uint32_t SEED = get_compile_time_arg_val(7);   // 0 = packer ZEROACC, 1 = zero copy
    constexpr uint32_t DEST_BATCH = get_compile_time_arg_val(8);
    // Runtime (not compile-time) so the ITERS=1 and ITERS=N launches share ONE compiled kernel:
    // the amortisation pair (t_N - t_1) / (N - 1) then cancels the fixed launch floor exactly.
    const uint32_t ITERS = get_arg_val<uint32_t>(0);

    // The op's D22 landing stride and the half-stride its pairwise DEST walk uses.  DERIVED from
    // GROUP_SIZE exactly as the op derives it, so no CT arg can drift.
    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;
    constexpr uint32_t GATHER_HALF = GATHER_SLOTS / 2;

    constexpr bool USES_MM = (MODE == 2 || MODE == 3 || MODE == 5);
    // Pages the resident input CB holds == the pages ONE call consumes.
    constexpr uint32_t CHUNK = (MODE == 3) ? ROWS : ((MODE == 5) ? 1 : ((MODE == 6) ? GROUP_SIZE : GATHER_SLOTS));
    // DEST windows the fused fold opens: `rows` for a per-row baseline, ONE for a compact block.
    constexpr uint32_t NWIN = (MODE == 0) ? ROWS : 1;
    constexpr uint32_t NOUT = (MODE == 1 || MODE == 3) ? 1 : ROWS;  // output pages per call
    constexpr uint32_t BANK_PAGES = (BANK == 0) ? (2 * ROWS + 1) : (ROWS + 1);
    constexpr uint32_t ZERO_IDX = BANK_PAGES - 1;
    constexpr uint32_t UNPACK_TRANSPOSE = (BANK == 0) ? 0 : 1;
    constexpr uint32_t UNPACK_BASE = (BANK == 0) ? ROWS : 0;

    if constexpr (USES_MM) {
        compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_part, cb_bank, cb_out);
    } else {
        compute_kernel_hw_startup(cb_part, cb_part, cb_out);
    }

    // The one-hot bank is resident for the whole kernel; expose it once.
    cb_reserve_back(cb_bank, BANK_PAGES);
    cb_push_back(cb_bank, BANK_PAGES);
    cb_wait_front(cb_bank, BANK_PAGES);

    // base_d19 only (the pre-D22 fold target): ONE output tile pinned for the whole call,
    // packer-accumulated in the fp32 CB.
    constexpr auto FOLD_OUT = ckl::output(
        cb_acc,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::SeedFirst);

    // PACK: `rows` matmuls ACCUMULATE into ONE DEST tile (matmul_tiles is DST += A*B), so the
    // whole compact tile costs ONE pack.  Columns >= rows stay EXACTLY 0.
    auto do_pack = [](uint32_t cb_in, uint32_t cb_o) {
        cb_reserve_back(cb_o, 1);
        tile_regs_acquire();
        if constexpr (SEED == 1) {
            reconfig_data_format_srca(cb_bank);
            copy_tile_to_dst_init_short(cb_bank);
            copy_tile(cb_bank, ZERO_IDX, 0);
        }
        matmul_init(cb_in, cb_bank, 0);
        for (uint32_t r = 0; r < ROWS; ++r) {
            matmul_tiles(cb_in, cb_bank, r, r, 0);
        }
        tile_regs_commit();
        pack_reconfig_data_format(cb_o);
        tile_regs_wait();
        pack_tile(0, cb_o, 0);
        tile_regs_release();
        cb_push_back(cb_o, 1);
    };
    // UN-PACK: `rows` matmuls, DEST-batched, each writing one column-shaped tile.
    auto do_unpack = [](uint32_t cb_in, uint32_t cb_o) {
        cb_reserve_back(cb_o, ROWS);
        matmul_init(cb_in, cb_bank, UNPACK_TRANSPOSE);
        for (uint32_t r0 = 0; r0 < ROWS; r0 += DEST_BATCH) {
            const uint32_t n = (ROWS - r0 < DEST_BATCH) ? (ROWS - r0) : DEST_BATCH;
            tile_regs_acquire();
            for (uint32_t d = 0; d < n; ++d) {
                if constexpr (SEED == 1) {
                    reconfig_data_format_srca(cb_bank);
                    copy_tile_to_dst_init_short(cb_bank);
                    copy_tile(cb_bank, ZERO_IDX, d);
                    matmul_init(cb_in, cb_bank, UNPACK_TRANSPOSE);
                }
                matmul_tiles(cb_in, cb_bank, 0, UNPACK_BASE + r0 + d, d);
            }
            tile_regs_commit();
            pack_reconfig_data_format(cb_o);
            tile_regs_wait();
            for (uint32_t d = 0; d < n; ++d) {
                pack_tile(d, cb_o, r0 + d);
            }
            tile_regs_release();
        }
        cb_push_back(cb_o, ROWS);
    };

    // THE FUSED ROOT CHAIN, carried verbatim from kernels/rms_norm_compute.cpp (D22).  `windows`
    // DEST windows over a GATHER_SLOTS-page landing stride; ONE `pack_tile` per window.
    auto do_fused_fold = [](uint32_t cb_o, uint32_t windows) {
        // The DATA-FORMAT RECONFIG is NOT optional in the op (pass A leaves the unpacker on the
        // bf16 input and the packer on cb_sum_handoff, while the gather is fp32); omitting it
        // gives a uniform ~1000x scale error that HOLDS pcc at 0.9997 and shows up only in
        // rel-RMS.  Every variant here pays it identically.
        reconfig_data_format(cb_part, cb_part);
        pack_reconfig_data_format(cb_o);
        add_tiles_init(cb_part, cb_part, /*acc_to_dest=*/true);
        rsqrt_tile_init();
        for (uint32_t w = 0; w < windows; ++w) {
            tile_regs_acquire();
            for (uint32_t p = 0; p < GATHER_HALF; ++p) {
                add_tiles(cb_part, cb_part, p, GATHER_HALF + p, 0);
            }
            fin_payload<INV_W_BITS, EPS_BITS, FIN>(0);
            tile_regs_commit();
            // Reserve/push PER WINDOW, exactly as the op does (so the writer's multicast can
            // start on the first finalized row); `pack_tile(0, cb_o)` therefore always lands on
            // the page this reserve just handed out.
            cb_reserve_back(cb_o, 1);
            tile_regs_wait();
            pack_tile(0, cb_o);
            tile_regs_release();
            cb_push_back(cb_o, 1);
        }
    };

    for (uint32_t iter = 0; iter < ITERS; ++iter) {
        if constexpr (MODE == 3) {
            // ---- sender: PACK `rows` column-shaped partials into ONE compact tile ------------
            MaybeDeviceZoneScope("member_pack");
            cb_reserve_back(cb_part, CHUNK);
            cb_push_back(cb_part, CHUNK);
            cb_wait_front(cb_part, CHUNK);
            do_pack(cb_part, cb_out);
            cb_pop_front(cb_part, CHUNK);
        } else if constexpr (MODE == 5) {
            // ---- receiver: UN-PACK the multicast compact stat into `rows` column tiles -------
            MaybeDeviceZoneScope("recv_unpack");
            cb_reserve_back(cb_part, 1);
            cb_push_back(cb_part, 1);
            cb_wait_front(cb_part, 1);
            do_unpack(cb_part, cb_out);
            cb_pop_front(cb_part, 1);
        } else if constexpr (MODE == 6) {
            // ---- base_d19: r2's pre-D22 baseline (calibration only) --------------------------
            {
                MaybeDeviceZoneScope("root_sum");
                for (uint32_t r = 0; r < ROWS; ++r) {
                    cb_reserve_back(cb_part, CHUNK);
                    cb_push_back(cb_part, CHUNK);
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(GROUP_SIZE),
                        ckl::CopyTile<ckl::input(cb_part)>{},
                        ckl::PackTile<FOLD_OUT>{});
                }
            }
            {
                MaybeDeviceZoneScope("root_finalize");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ROWS),
                    ckl::CopyTile<ckl::input(cb_acc)>{},
                    StatFinalize<INV_W_BITS, EPS_BITS, FIN>{},
                    ckl::PackTile<ckl::output(cb_out)>{});
            }
        } else {
            // ---- the D22 fused root chain: fold + finalize + pack in ONE DEST window ---------
            {
                MaybeDeviceZoneScope("root_fused");
                cb_reserve_back(cb_part, CHUNK);
                cb_push_back(cb_part, CHUNK);
                cb_wait_front(cb_part, CHUNK);
                do_fused_fold((MODE == 2) ? cb_mid : cb_out, NWIN);
                cb_pop_front(cb_part, CHUNK);
            }
            if constexpr (MODE == 2) {
                // ---- un-permute ON THE ROOT: serial, on the critical path -------------------
                MaybeDeviceZoneScope("root_unpack");
                cb_wait_front(cb_mid, 1);
                do_unpack(cb_mid, cb_out);
                cb_pop_front(cb_mid, 1);
            }
        }

        // Drain between in-kernel iterations so the next pass starts clean; the LAST iteration's
        // output stays in L1 for the host to read back.
        if (iter + 1 < ITERS) {
            cb_wait_front(cb_out, NOUT);
            cb_pop_front(cb_out, NOUT);
        }
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def single_core_shard(n_tiles):
    """The whole [n_tiles*32 x 32] tile column as ONE shard on core (0,0)."""
    return ttnn.create_sharded_memory_config(
        shape=(n_tiles * TILE, TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def bank_pages(rows, bank_mode):
    return (2 * rows + 1) if bank_mode == BANK_EF else (rows + 1)


def geometry(variant, group_size, rows, bank_mode):
    """(resident input pages, bank pages, output pages) for a variant."""
    mode = MODES[variant]
    if mode == 3:
        chunk = rows
    elif mode == 5:
        chunk = 1
    elif mode == 6:
        chunk = group_size
    else:
        chunk = gather_slots(group_size)
    out_pages = 1 if mode in (1, 3) else rows
    return chunk, bank_pages(rows, bank_mode), out_pages


def create_program_descriptor(
    part,
    bank,
    out,
    *,
    variant,
    group_size,
    rows,
    inv_w_bits,
    eps_bits,
    iters,
    fin,
    bank_mode,
    seed,
    dest_batch,
):
    mode = MODES[variant]
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [iters]
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[mode, group_size, rows, inv_w_bits, eps_bits, fin, bank_mode, seed, dest_batch],
        runtime_args=rt,
        # THE USER'S PRECISION CONTRACT, identical for every variant.  Never a lever.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )
    _, _, out_pages = geometry(variant, group_size, rows, bank_mode)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_PART, part),
        ttnn.cb_descriptor_from_sharded_tensor(CB_BANK, bank),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        ttnn.CBDescriptor(
            total_size=2 * FP32_TILE_BYTES,
            core_ranges=_single_core(),
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MID, data_format=ttnn.float32, page_size=FP32_TILE_BYTES)
            ],
        ),
        # cb_acc (base_d19 only) mirrors the op's pre-D22 cb_row_stat: depth 2 * pages.
        ttnn.CBDescriptor(
            total_size=2 * max(out_pages, 1) * FP32_TILE_BYTES,
            core_ranges=_single_core(),
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_ACC, data_format=ttnn.float32, page_size=FP32_TILE_BYTES)
            ],
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run(part, bank, out, **kw):
    return ttnn.generic_op([part, bank, out], create_program_descriptor(part, bank, out, **kw))
