// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF: rms_norm's SCALE PASS only.  out = (x * 1/rms) * gamma
//
// Everything except the scale pass is held constant / trivial: every operand is
// already resident in L1 (the CBs are backed by single-core sharded tensors, so
// there is NO reader, NO writer and NO NoC traffic in this kernel's span), and
// the whole span is repeated KERNEL_ITERS times so the per-tile compute cost is
// measured instead of a DRAM bound.  The only thing that varies between arms is
// HOW the two multiplies are emitted.
//
//   1/rms  is a COLUMN broadcast (one scalar per tile-row,  cb_rms  col 0)
//   gamma  is a ROW    broadcast (one scalar per tile-col,  cb_gamma row 0)
//
// VARIANTS
//   0 baseline           the op's current form: mul<Col>(x, rms) -> cb_normed,
//                        then mul<Row>(normed, gamma) -> out.  Full L1 round
//                        trip on cb_normed (BLOCK_HT * WT pages).
//   1 baseline_reversed  same two helper calls, operands swapped: the ROW
//                        broadcast runs first.  Tests whether one broadcast
//                        direction is cheaper to produce than the other.
//   2 fused_rmsfull      ONE dst-sync window, cb_normed DELETED.
//                        BinaryFpu<Mul, x, gamma(Row bcast)> -> D0, then
//                        DestReuseBinary<Mul, rms_full> -> D0, then one pack.
//                        DestReuseBinary carries no BroadcastDim, so the reuse
//                        operand must be a full 32x32 tile -- and the CHEAP one
//                        to materialise is 1/rms (BLOCK_HT tiles per row-block),
//                        NOT gamma (Wt_core tiles).  UnaryBcast<Col> does the
//                        materialise straight out of the col-broadcast tile, so
//                        no cb_ones is needed.
//   3 fused_inchain      ONE window, no BinaryFpu at all:
//                        UnaryBcast<Row>(gamma) -> D0 (gamma materialised in
//                        DEST, per tile, never in L1), then two DestReuseBinary
//                        muls (x, then rms_full).  3 FPU ops, 1 unpack each.
//   4 fused_gammafull    the op_design L4 sketch: pre-broadcast GAMMA into full
//                        tiles (WT tiles per chunk), then
//                        BinaryFpu<Mul, x, rms(Col bcast)> + DestReuseBinary.
//                        Prices the expensive pre-pass per row-block.
//   5 fused_gammafull_am same, but the gamma pre-broadcast is hoisted out of the
//                        iteration loop = amortised over many row-blocks
//                        (Regime A with num_row_blocks >> 1).
//   8 baseline_subchunk  the two muls kept as-is, but the W-chunk walked in
//                        SUB_CHUNK-tile groups so cb_normed shrinks to
//                        SUB_CHUNK pages -- the SAME L1 the fusion frees, bought
//                        with extra per-call overhead instead of a slower op.
//   7 fused_sfpu       ONE window, NO dest-reuse: x*rms on the FPU into D0,
//                        gamma broadcast-copied into D1, SFPU multiply D0*=D1.
//   6 raw_llk            variant 2's sequence hand-written on raw LLK, no chain:
//                        the ceiling if the chain's per-block init/transition
//                        bookkeeping is what costs.
//
// HELPER BYPASS (variant 6 only, and only inside this bench): the raw arm
// bypasses ckl::eltwise_chain / ckl::mul.  It exists to price the chain's
// per-block-iteration init emission (a chain mixing BinaryFpu with
// DestReuseBinary is NOT math-MOP-uniform, so chain_hoist_math_mop is false and
// both inits are re-emitted per DEST block).  Everything it does is otherwise
// identical to variant 2.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_rms = 2;
constexpr uint32_t cb_normed = 3;
constexpr uint32_t cb_rms_full = 4;
constexpr uint32_t cb_gamma_full = 5;
constexpr uint32_t cb_out = 16;

constexpr uint32_t WT = get_compile_time_arg_val(0);
constexpr uint32_t BLOCK_HT = get_compile_time_arg_val(1);
constexpr uint32_t DEST_BLOCK_CT = get_compile_time_arg_val(2);
constexpr uint32_t KERNEL_ITERS = get_compile_time_arg_val(3);
constexpr uint32_t VARIANT = get_compile_time_arg_val(4);
constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(5);
// variant 8 only: tiles per sub-group (0 = the whole W-chunk, i.e. the baseline).
constexpr uint32_t SUB_CHUNK_CT = get_compile_time_arg_val(6);
constexpr uint32_t SUB_CHUNK = SUB_CHUNK_CT ? SUB_CHUNK_CT : WT;

constexpr uint32_t DEST_BLOCK = (DEST_BLOCK_CT < ckl::DEST_AUTO_LIMIT) ? DEST_BLOCK_CT : ckl::DEST_AUTO_LIMIT;

constexpr uint32_t N_TILES = BLOCK_HT * WT;

// Held-resident operand policies: every input tile is already in L1 for the
// whole launch, so nothing is popped until the very end.  Identical for every
// arm, so the arm-to-arm delta is the math, not the CB bookkeeping.
constexpr auto x_in = ckl::input(cb_x, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block);
constexpr auto rms_col_bcast =
    ckl::input(cb_rms, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col);
constexpr auto gamma_row_bcast = ckl::input(
    cb_gamma, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row);
constexpr auto out_blk = ckl::output(cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);

ALWI void mark_resident(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
}

// --- the no-gamma control: one mul straight to the output, nothing to fuse ----
ALWI void scale_only() {
    MaybeDeviceZoneScope("sc_scale");
    ckl::mul<x_in, rms_col_bcast, out_blk>(ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK));
}

// --- variant 0: the op's current two-call form through cb_normed -------------
ALWI void baseline() {
    {
        MaybeDeviceZoneScope("sc_scale");
        ckl::mul<
            x_in,
            rms_col_bcast,
            ckl::output(cb_normed, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK));
    }
    {
        MaybeDeviceZoneScope("sc_gamma");
        ckl::mul<
            ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            gamma_row_bcast,
            out_blk>(ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK));
    }
}

// --- variant 1: same two calls, ROW broadcast first --------------------------
ALWI void baseline_reversed() {
    {
        MaybeDeviceZoneScope("sc_gamma");
        ckl::mul<
            x_in,
            gamma_row_bcast,
            ckl::output(cb_normed, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK));
    }
    {
        MaybeDeviceZoneScope("sc_scale");
        ckl::mul<
            ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            rms_col_bcast,
            out_blk>(ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK));
    }
}

// --- materialise the col-broadcast 1/rms into BLOCK_HT full tiles ------------
// BLOCK_HT tiles per row-block against BLOCK_HT*WT tiles of payload: 1/WT of the
// work (0.45% at the focus shape's WT=112).
ALWI void prebroadcast_rms() {
    MaybeDeviceZoneScope("sc_pre_rms");
    ckl::unary_bcast<
        ckl::BroadcastDim::Col,
        ckl::input(cb_rms, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block),
        ckl::output(cb_rms_full, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
        ckl::IterationShape::tiles(BLOCK_HT).block_size(DEST_BLOCK));
}

ALWI void prebroadcast_gamma() {
    MaybeDeviceZoneScope("sc_pre_gamma");
    ckl::unary_bcast<
        ckl::BroadcastDim::Row,
        ckl::input(cb_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block),
        ckl::output(cb_gamma_full, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
        ckl::IterationShape::tiles(WT).block_size(DEST_BLOCK));
}

// --- variant 2: ONE window, gamma stays a hardware ROW broadcast -------------
ALWI void fused_rmsfull() {
    MaybeDeviceZoneScope("sc_fused");
    ckl::eltwise_chain(
        ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, gamma_row_bcast>{},
        ckl::DestReuseBinary<
            ckl::input(cb_rms_full, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Col),
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCA>{},
        ckl::PackTile<out_blk>{});
}

// --- variant 3: ONE window, gamma materialised in DEST per tile --------------
ALWI void fused_inchain() {
    MaybeDeviceZoneScope("sc_fused");
    ckl::eltwise_chain(
        ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK),
        ckl::UnaryBcast<
            ckl::BroadcastDim::Row,
            ckl::input(cb_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row)>{},
        ckl::DestReuseBinary<x_in, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCB>{},
        ckl::DestReuseBinary<
            ckl::input(cb_rms_full, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Col),
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCB>{},
        ckl::PackTile<out_blk>{});
}

// --- variants 4/5: the op_design L4 sketch (gamma pre-broadcast) -------------
// GammaPop = AtEnd on the per-row-block arm (the pre-pass re-pushes every
// iteration), None on the amortised arm (materialised once per launch).
template <ckl::PopPolicy GammaPop>
ALWI void fused_gammafull() {
    MaybeDeviceZoneScope("sc_fused");
    ckl::eltwise_chain(
        ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, rms_col_bcast>{},
        ckl::DestReuseBinary<
            ckl::input(cb_gamma_full, ckl::WaitPolicy::Upfront, GammaPop, ckl::OperandKind::Row),
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCA>{},
        ckl::PackTile<out_blk>{});
}

// --- variant 8: the two muls kept, but cb_normed made TINY -------------------
// The fusion's real prize is L1, not cycles (`cb_normed` is BLOCK_HT * WT pages
// of a shared budget).  This arm buys the same L1 WITHOUT touching the datapath:
// it walks the W-chunk in SUB_CHUNK-tile groups and lets both muls run inside
// each group, so cb_normed only ever needs SUB_CHUNK * BLOCK_HT pages.  What it
// costs instead is one extra pair of chain calls per group (per-call init,
// reconfig, upfront wait), which is what this arm prices.
//
// The resident operands do not advance (PopPolicy::None), so each group indexes
// them with a runtime TileOffset::Set base; only cb_out advances by itself.
template <uint32_t SUB>
ALWI void baseline_subchunk() {
    for (uint32_t w0 = 0; w0 < WT; w0 += SUB) {
        const uint32_t n = (WT - w0 < SUB) ? (WT - w0) : SUB;
        {
            MaybeDeviceZoneScope("sc_scale");
            ckl::eltwise_chain(
                ckl::IterationShape::grid(BLOCK_HT, n).block_size(DEST_BLOCK),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        cb_x,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Block,
                        ckl::TileOffset::Set),
                    rms_col_bcast>{w0 * BLOCK_HT, 0},
                ckl::PackTile<
                    ckl::output(cb_normed, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
        }
        {
            MaybeDeviceZoneScope("sc_gamma");
            ckl::eltwise_chain(
                ckl::IterationShape::grid(BLOCK_HT, n).block_size(DEST_BLOCK),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::input(
                        cb_gamma,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Row,
                        ckl::TileOffset::Set)>{0, w0},
                ckl::PackTile<out_blk>{});
        }
    }
}

// --- variant 7: ONE window, no DEST-REUSE at all -----------------------------
// The other route out of the L1 round trip: keep both operands in DEST and
// combine them with an SFPU multiply.  gamma is materialised into D1 by the
// FPU's broadcast-copy (UnaryBcast), so no CB and no pre-pass is needed and the
// slow DEST->src reuse path is avoided.  Costs a DEST lane (lane_width 2 -> the
// chain clamps block_size to DEST_AUTO_LIMIT/2).
ALWI void fused_sfpu() {
    MaybeDeviceZoneScope("sc_fused");
    ckl::eltwise_chain(
        ckl::IterationShape::grid(BLOCK_HT, WT).block_size(DEST_BLOCK),
        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, rms_col_bcast>{},
        ckl::UnaryBcast<
            ckl::BroadcastDim::Row,
            ckl::input(cb_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row),
            ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<out_blk>{});
}

// --- variant 6: variant 2's sequence, raw LLK, no chain ---------------------
ALWI void raw_llk() {
    MaybeDeviceZoneScope("sc_fused");
    cb_wait_front(cb_x, N_TILES);
    cb_wait_front(cb_gamma, WT);
    cb_wait_front(cb_rms_full, BLOCK_HT);
    for (uint32_t h = 0; h < BLOCK_HT; ++h) {
        for (uint32_t w0 = 0; w0 < WT; w0 += DEST_BLOCK) {
            const uint32_t n = (WT - w0 < DEST_BLOCK) ? (WT - w0) : DEST_BLOCK;
            cb_reserve_back(cb_out, n);
            tile_regs_acquire();
            mul_bcast_rows_init(cb_x, cb_gamma);
            for (uint32_t j = 0; j < n; ++j) {
                mul_tiles_bcast_rows(cb_x, cb_gamma, h * WT + w0 + j, w0 + j, j);
            }
            mul_reuse_dest_init<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_rms_full);
            for (uint32_t j = 0; j < n; ++j) {
                mul_reuse_dest_tiles<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_rms_full, h, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, n);
        }
    }
    cb_pop_front(cb_rms_full, BLOCK_HT);
}

}  // namespace

void kernel_main() {
    // Every input is already resident in L1 (tensor-backed CB): publish it once.
    mark_resident(cb_x, N_TILES);
    if constexpr (HAS_GAMMA) {
        mark_resident(cb_gamma, WT);
    }
    mark_resident(cb_rms, BLOCK_HT);

    compute_kernel_hw_startup(cb_x, cb_gamma, cb_out);

    if constexpr (VARIANT == 5 && HAS_GAMMA) {
        prebroadcast_gamma();  // amortised arm: once per launch, not per row-block
    }

    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {
        if constexpr (!HAS_GAMMA) {
            scale_only();
        } else if constexpr (VARIANT == 0) {
            baseline();
        } else if constexpr (VARIANT == 1) {
            baseline_reversed();
        } else if constexpr (VARIANT == 2) {
            prebroadcast_rms();
            fused_rmsfull();
        } else if constexpr (VARIANT == 3) {
            prebroadcast_rms();
            fused_inchain();
        } else if constexpr (VARIANT == 4) {
            prebroadcast_gamma();
            fused_gammafull<ckl::PopPolicy::AtEnd>();
        } else if constexpr (VARIANT == 5) {
            fused_gammafull<ckl::PopPolicy::None>();
        } else if constexpr (VARIANT == 7) {
            fused_sfpu();
        } else if constexpr (VARIANT == 8) {
            baseline_subchunk<SUB_CHUNK>();
        } else {
            prebroadcast_rms();
            raw_llk();
        }

        if (iter + 1 < KERNEL_ITERS) {
            cb_wait_front(cb_out, N_TILES);
            cb_pop_front(cb_out, N_TILES);
        }
    }

    cb_pop_front(cb_x, N_TILES);
    cb_pop_front(cb_rms, BLOCK_HT);
    if constexpr (HAS_GAMMA) {
        cb_pop_front(cb_gamma, WT);
    }
}
