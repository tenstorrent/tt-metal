# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated micro-bench: rms_norm cross-core PASS 2 (x*rstd*gamma) — BATCH x FUSE composition.

Perf idea under test (owner: pass2_batch_and_fuse): the deliberate COMBINATION of two sibling
levers, measured together to see whether they compose or one supersedes the other:
  (A) BATCH pass-2 across the C tile-rows of a cross-core round into ONE 2D grid walk (instead of
      per-tile-row), amortizing the per-chain init/reconfig over C rows.
  (B) FUSE away the cb_norm round-trip: keep x*rstd resident in DEST and apply gamma IN PLACE via
      an FPU DEST-reuse op (DestReuseBinary DEST_TO_SRCA) against a PRE-REPLICATED gamma (full
      [32,32] tiles, so ·gamma is a plain NO-broadcast mul — the only expressible dest-reuse form,
      since DestReuseBinary carries no BroadcastDim). This HALVES the pack count (one pack/tile
      instead of pack-to-cb_norm + pack-to-cb_out) and eliminates cb_norm entirely, at the cost of
      a DEST->srca reload each tile.

Everything but the pass-2 chain structure is held constant (concept-isolation): rstd (1/RMS) tiles
are PRE-SUPPLIED resident in cb_stat_global (no pass-1, no cross-core gather/fold), x is a resident
bf16 sharded W-slice in cb_x_in, gamma a resident bf16 (pre-replicated) slice in cb_gamma, cb_out is
the resident bf16 output shard. The ONLY timed thing is the pass-2 compute pipeline, so the delta
between variants is attributable to chain STRUCTURE + fusion alone.

Precision contract (FROZEN — identical for every variant, never tuned for speed): bf16 x + bf16
TILE gamma, fp32 rstd, HiFi2, fp32_dest_acc_en=False, math_approx_mode=False. PCC gate 0.9995.

The 4-way menu (reveals composition):
  baseline    : the op's CURRENT graduated Perf-1 pass2 — per tile-row, ONE x*rstd chain over
                PER_W_T tiles (Block x, Col rstd) -> pack cb_norm; ONE norm*gamma chain over PER_W_T
                (Scalar norm streamed, Row gamma) -> pack cb_out. reconfig-skip (SrcB + None).
                2*C chains/round. cb_norm depth = 2*PER_W_T.
  batch_only  : (A) alone — ONE x*rstd chain over grid(C, PER_W_T) (Block x, Col rstd = ht) -> deep
                cb_norm; ONE norm*gamma chain over grid(C, PER_W_T) (Block norm Bulk, Row gamma) ->
                cb_out. 2 chains/round. cb_norm depth = C*PER_W_T. (== round-1 batch_both, reconfirmed
                as the batch-only reference point.)
  fuse_only   : (B) alone — per tile-row, ONE fused chain: BinaryFpu x*rstd (Block x, Col rstd) ->
                DestReuseBinary ·gamma_full (DEST_TO_SRCA, NO bcast, Row-indexed gamma) -> PackTile
                cb_out. NO cb_norm. C chains/round.
  batch_fuse  : (A)+(B) — ONE fused chain over grid(C, PER_W_T): BinaryFpu x*rstd (Block x, Col rstd
                = ht) -> DestReuseBinary ·gamma_full (Row-indexed gamma = wt) -> PackTile cb_out.
                NO cb_norm. 1 chain/round.

All four compute IDENTICAL math (same Mul ops, same broadcast semantics baked into the data), so they
are numerically equivalent; PCC is reported vs a torch fp32 reference of x*rstd*gamma.

MEASURED (blackhole_p150b, BH, N=7 median, kernel_iters=50, DEVICE KERNEL DURATION [ns]) —
FOCUS PER_W_T=4 HT_LOCAL=32 C_ROWS=8:
    baseline    14077.6 ns  1.000x  pcc 0.99999  (cb_norm 8 tiles)
    batch_only  13884.8 ns  1.014x  pcc 0.99999  (cb_norm 32 tiles)  <- small win
    fuse_only   15172.9 ns  0.928x  pcc 0.99999  (cb_norm 0)         <- REGRESSION
    batch_fuse  14919.8 ns  0.944x  pcc 0.99999  (cb_norm 0)         <- REGRESSION (my idea)

VERDICT for the batch+fuse COMBINATION: REGRESSION. The FUSE half is a Blackhole dead-end — the
DEST->srca reload the dest-reuse pays costs MORE than the cb_norm pack+unpack it removes (confirms
R6g's 0.94-1.00x standalone dstreuse, now measured in the full batched pass-2). Crucially, BATCHING
does NOT change the fusion economics: the fusion penalty is ~+7.5% WITH batching (batch_fuse /
batch_only) vs ~+7.8% WITHOUT (fuse_only / baseline) — stable, because the per-tile DEST reload is
intrinsic and chain-level batching only amortizes chain INIT. So batch+fuse does NOT supersede the
individual levers: batch_only (1.014x) strictly dominates batch_fuse (0.944x). Precision-neutral
(all 0.99999; DEST is bf16 == cb_norm bf16 intermediate). batch_fuse's ONLY upside is L1 (cb_norm
eliminated), which is not the binding constraint on the focus core.

Predicate (sweep, HT=16): batch_only wins big at PER_W_T=2 (1.12-1.13x), flat at PER_W_T>=4; the
fused variants REGRESS across the whole sweep and worsen as PER_W_T grows (pwt=8: 0.889x) since each
added W-tile is another DEST reload. Batched variants require C | HT_LOCAL (the grid-across-C chain
has no partial-C-block form; a short tail corrupts — per-row baseline/fuse_only handle it fine).
"""

import ttnn

TILE = 32
BF16_TILE = ttnn.tile_size(ttnn.bfloat16)  # 2048 bytes
FP32_TILE = ttnn.tile_size(ttnn.float32)  # 4096 bytes

# CB assignment (mirrors the real xcore compute kernel's namespace).
CB_X_IN = 1  # resident bf16 sharded W-slice (zero-copy) — x
CB_GAMMA = 3  # resident bf16 gamma W-slice (zero-copy), PRE-REPLICATED across all 32 rows
CB_STAT_GLOBAL = 7  # resident fp32 1/RMS tiles, one per tile-row (zero-copy)
CB_OUT = 16  # resident bf16 output W-slice (zero-copy)
CB_NORM = 26  # pass-2 intermediate x*rstd (scratch) — ONLY the unfused variants allocate it

VARIANTS = ("baseline", "batch_only", "fuse_only", "batch_fuse")
BASELINE = "baseline"

_VARIANT_ID = {"baseline": 0, "batch_only": 1, "fuse_only": 2, "batch_fuse": 3}
_FUSED = {"fuse_only", "batch_fuse"}


def cb_norm_depth_for(variant, per_w_t, c_rows):
    """L1 depth of the pass-2 intermediate cb_norm (in tiles). Fused variants use NONE."""
    if variant in _FUSED:
        return 0
    if variant == "baseline":
        return 2 * per_w_t
    return c_rows * per_w_t  # batch_only stages the whole round's x*rstd


def variant_is_valid(variant, per_w_t, c_rows, ht_local):
    return variant in _VARIANT_ID and per_w_t >= 1 and c_rows >= 1 and ht_local >= 1


# =============================================================================
# Compute kernel — one source; the variant selector + geometry are compile-time args.
# CT args: [variant_id, PER_W_T, HT_LOCAL, C_ROWS, HAS_GAMMA, KERNEL_ITERS]
#
# reconfig-skip discipline (matches the graduated Perf-1 pass2): boot programs srcA=cb_x_in (bf16),
# srcB=cb_stat_global (fp32), pack=cb_out (bf16). srcA is ALWAYS bf16 across pass 2 (cb_x_in / cb_norm)
# and the pack target is ALWAYS bf16 (cb_norm / cb_out), so their reconfig is wasted MMIO -> dropped
# (BinaryDataFormatReconfig::SrcB + PackTileReconfig::None). srcB genuinely alternates fp32(rstd) <->
# bf16(gamma) every chain, so it is the only side kept. Numerically byte-identical (same ops).
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_x_in = 1;
constexpr uint32_t cb_gamma = 3;
constexpr uint32_t cb_stat_global = 7;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_norm = 26;
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(1);
    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(2);
    constexpr uint32_t C_ROWS = get_compile_time_arg_val(3);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t KERNEL_ITERS = get_compile_time_arg_val(5);

    constexpr uint32_t shard_tiles = HT_LOCAL * PER_W_T;
    constexpr uint32_t num_rounds = (HT_LOCAL + C_ROWS - 1) / C_ROWS;

    constexpr bool FUSED = (VARIANT == 2 || VARIANT == 3);
    constexpr bool BATCHED = (VARIANT == 1 || VARIANT == 3);

    // srcA <- cb_x_in (bf16), srcB <- cb_stat_global (fp32), packer <- cb_out (bf16).
    compute_kernel_hw_startup(cb_x_in, cb_stat_global, cb_out);

    // Arm the resident zero-copy inputs ONCE (held for the whole kernel, never popped).
    cb_reserve_back(cb_x_in, shard_tiles);      cb_push_back(cb_x_in, shard_tiles);
    cb_reserve_back(cb_stat_global, HT_LOCAL);  cb_push_back(cb_stat_global, HT_LOCAL);
    if constexpr (HAS_GAMMA) {
        cb_reserve_back(cb_gamma, PER_W_T);     cb_push_back(cb_gamma, PER_W_T);
    }
    cb_wait_front(cb_x_in, shard_tiles);
    cb_wait_front(cb_stat_global, HT_LOCAL);
    if constexpr (HAS_GAMMA) {
        cb_wait_front(cb_gamma, PER_W_T);
    }

    // reconfig-skip ESTABLISH (once): srcA=cb_x_in (bf16), pack=cb_out (bf16). Boot already set these,
    // but pin them explicitly so the SrcB/None chains below have the invariant they assert.
    reconfig_data_format(cb_x_in, cb_x_in);
    pack_reconfig_data_format(cb_out);

    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {
        for (uint32_t r = 0; r < num_rounds; ++r) {
            const uint32_t base_t = r * C_ROWS;
            uint32_t C_this = HT_LOCAL - base_t;
            if (C_this > C_ROWS) {
                C_this = C_ROWS;
            }

            if constexpr (VARIANT == 0) {
                // ---- baseline: per-tile-row, 2 W-batched chains through cb_norm (reconfig-skip) ----
                for (uint32_t cc = 0; cc < C_this; ++cc) {
                    const uint32_t t = base_t + cc;
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::of(1, PER_W_T),
                        ckl::BinaryFpu<
                            cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Col,
                            ckl::TileOffset::Set, ckl::TileOffset::Set>{t * PER_W_T, t},
                        ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                    if constexpr (HAS_GAMMA) {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::of(1, PER_W_T),
                            ckl::BinaryFpu<
                                cb_norm, cb_gamma, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row,
                                ckl::InputLifecycle::Streaming, ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                                ckl::OperandKind::Scalar, ckl::OperandKind::Row,
                                ckl::TileOffset::Unset, ckl::TileOffset::Set>{0, 0},
                            ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                    } else {
                        ckl::copy<cb_norm, cb_out>(ckl::EltwiseShape::of(1, PER_W_T));
                    }
                }
            } else if constexpr (VARIANT == 1) {
                // ---- batch_only: ONE x*rstd + ONE gamma chain per round (2D grid), reconfig-skip ----
                ckl::eltwise_chain(
                    ckl::EltwiseShape::of(C_this, PER_W_T),
                    ckl::BinaryFpu<
                        cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                        ckl::OperandKind::Block, ckl::OperandKind::Col,
                        ckl::TileOffset::Set, ckl::TileOffset::Set>{base_t * PER_W_T, base_t},
                    ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                if constexpr (HAS_GAMMA) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::of(C_this, PER_W_T),
                        ckl::BinaryFpu<
                            cb_norm, cb_gamma, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Row,
                            ckl::TileOffset::Unset, ckl::TileOffset::Unset>{0, 0},
                        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                } else {
                    ckl::copy<cb_norm, cb_out>(ckl::EltwiseShape::of(1, C_this * PER_W_T));
                }
            } else if constexpr (VARIANT == 2) {
                // ---- fuse_only: per tile-row, ONE fused chain — NO cb_norm round-trip ----
                // BinaryFpu x*rstd (Block x, Col rstd) -> DEST; DestReuseBinary DEST_TO_SRCA multiplies
                // the RUNNING DEST by pre-replicated gamma_full (NO broadcast, Row-indexed tile = wt) ->
                // DEST; PackTile packs DEST -> cb_out. One pack/tile (vs two) and cb_norm eliminated.
                for (uint32_t cc = 0; cc < C_this; ++cc) {
                    const uint32_t t = base_t + cc;
                    if constexpr (HAS_GAMMA) {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::of(1, PER_W_T),
                            ckl::BinaryFpu<
                                cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                                ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                                ckl::OperandKind::Block, ckl::OperandKind::Col,
                                ckl::TileOffset::Set, ckl::TileOffset::Set>{t * PER_W_T, t},
                            ckl::DestReuseBinary<
                                cb_gamma, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA,
                                ckl::InputLifecycle::CallerManaged, ckl::DestReuseReconfig::Input,
                                ckl::Dst::D0, ckl::Dst::D0, ckl::OperandKind::Row, ckl::TileOffset::Unset>{},
                            ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                    } else {
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::of(1, PER_W_T),
                            ckl::BinaryFpu<
                                cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                                ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                                ckl::OperandKind::Block, ckl::OperandKind::Col,
                                ckl::TileOffset::Set, ckl::TileOffset::Set>{t * PER_W_T, t},
                            ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                    }
                }
            } else {
                // ---- batch_fuse: ONE fused chain over grid(C, PER_W_T) — NO cb_norm round-trip ----
                if constexpr (HAS_GAMMA) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::of(C_this, PER_W_T),
                        ckl::BinaryFpu<
                            cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Col,
                            ckl::TileOffset::Set, ckl::TileOffset::Set>{base_t * PER_W_T, base_t},
                        ckl::DestReuseBinary<
                            cb_gamma, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA,
                            ckl::InputLifecycle::CallerManaged, ckl::DestReuseReconfig::Input,
                            ckl::Dst::D0, ckl::Dst::D0, ckl::OperandKind::Row, ckl::TileOffset::Unset>{},
                        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                } else {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::of(C_this, PER_W_T),
                        ckl::BinaryFpu<
                            cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::SrcB, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Col,
                            ckl::TileOffset::Set, ckl::TileOffset::Set>{base_t * PER_W_T, base_t},
                        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
                }
            }
        }

        // Drain the resident output between steady-state iterations; leave the last pass in L1.
        if (iter + 1 < KERNEL_ITERS) {
            cb_wait_front(cb_out, shard_tiles);
            cb_pop_front(cb_out, shard_tiles);
        }
    }
}
"""


# =============================================================================
# Host-side sharded-L1 layout + program descriptor
# =============================================================================


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(shape):
    """Whole `shape` as a single-core height shard (row-major orientation)."""
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, num_tiles):
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat16, page_size=BF16_TILE)
    return ttnn.CBDescriptor(total_size=BF16_TILE * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])


def create_program_descriptor(x, rstd, gamma, out, *, variant, per_w_t, ht_local, c_rows, has_gamma, kernel_iters=1):
    if variant not in _VARIANT_ID:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError("x must be bfloat16 TILE_LAYOUT")
    if rstd.dtype != ttnn.float32 or rstd.layout != ttnn.TILE_LAYOUT:
        raise ValueError("rstd must be float32 TILE_LAYOUT")
    if has_gamma and (gamma.dtype != ttnn.bfloat16 or gamma.layout != ttnn.TILE_LAYOUT):
        raise ValueError("gamma must be bfloat16 TILE_LAYOUT")

    compile_time_args = [
        _VARIANT_ID[variant],
        per_w_t,
        ht_local,
        c_rows,
        int(has_gamma),
        kernel_iters,
    ]

    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        # FROZEN precision contract: bf16 in, HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT_GLOBAL, rstd),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    norm_depth = cb_norm_depth_for(variant, per_w_t, c_rows)
    if norm_depth > 0:
        cbs.append(_scratch_cb(CB_NORM, norm_depth))
    tensors = [x, rstd]
    if has_gamma:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, gamma))
        tensors.append(gamma)
    tensors.append(out)

    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs), tensors


def run_op(x, rstd, gamma, *, variant, per_w_t, ht_local, c_rows, has_gamma, kernel_iters=1):
    """Allocate the resident output shard and run one pass-2 variant."""
    m = ht_local * TILE
    n = per_w_t * TILE
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([m, n]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        x.device(),
        create_sharded_memory_config((m, n)),
    )
    descriptor, tensors = create_program_descriptor(
        x,
        rstd,
        gamma,
        out,
        variant=variant,
        per_w_t=per_w_t,
        ht_local=ht_local,
        c_rows=c_rows,
        has_gamma=has_gamma,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op(tensors, descriptor)
