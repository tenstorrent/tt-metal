# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated micro-bench: rms_norm cross-core PASS 2 (x*rstd * gamma), one core, resident L1.

Perf idea under test (owner: pass2_batch_rows): BATCH pass-2 across the C tile-rows of a
cross-core round instead of issuing it per-tile-row. Pass 2 dominates the cross-core compute
kernel on the perf-flagged BLOCK_SHARDED (1,1,8192,1024) 8x8 case (~63% of the critical core).

Everything but the pass-2 chain structure is held constant (see the concept-isolation table):
the rstd (1/RMS) tiles are PRE-SUPPLIED resident in cb_stat_global (no pass-1, no cross-core
gather/fold), x is a resident bf16 sharded W-slice in cb_x_in, gamma a resident bf16 slice in
cb_gamma, and cb_out is the resident bf16 output shard. So the ONLY thing timed is the pass-2
compute pipeline, and the delta between variants is attributable to the chain structure alone.

Precision contract (FIXED — identical for every variant): bf16 x + bf16 TILE gamma, fp32 rstd,
HiFi2, fp32_dest_acc_en=False, math_approx_mode=False. Never tuned for speed.

Variants (the menu):
  baseline    : per-tile-row PASS2_BATCH (the op's CURRENT approach) — for each of the C_this
                tile-rows: ONE x*rstd chain over PER_W_T tiles (Block x, Col rstd) then ONE
                gamma chain over PER_W_T tiles (Scalar norm streamed, Row gamma). = 2*C_this
                chains/round. cb_norm depth = 2*PER_W_T.
  batch_gamma : per-row x*rstd (C_this chains, all landing in a deep cb_norm) then ONE gamma
                chain over the whole round's C_this*PER_W_T tiles (Block norm, Row gamma) —
                amortizes only gamma's init across the C rows. cb_norm depth = C_ROWS*PER_W_T.
  batch_both  : ONE x*rstd chain over the whole round as a 2D grid(C_this, PER_W_T) with the
                per-row rstd as a Col operand (index = ht) + ONE gamma chain grid(C_this,
                PER_W_T) (Block norm, Row gamma). = 2 chains/round. cb_norm = C_ROWS*PER_W_T.
                This is the interleaved resident block-form (rms_norm_compute.cpp) extended
                from a 1D per-row walk to a 2D walk over the batched tile-rows.
  batch_both_blk : batch_both with EltwiseShape block_size = PER_W_T (widen the DST batch: each
                outer iter processes a whole tile-row's W across PER_W_T DEST lanes). A/B on
                whether widening the DST batch helps or hits the DEST budget (bf16 DEST here).

All variants compute IDENTICAL math (same Mul ops, same Col/Row broadcast semantics), so they
should be numerically equivalent; PCC is reported vs a torch reference for x*rstd*gamma.
"""

import ttnn

TILE = 32
BF16_TILE = ttnn.tile_size(ttnn.bfloat16)  # 2048 bytes
FP32_TILE = ttnn.tile_size(ttnn.float32)  # 4096 bytes

# CB assignment (mirrors the real xcore compute kernel's namespace for clarity).
CB_X_IN = 1  # resident bf16 sharded W-slice (zero-copy) — x
CB_GAMMA = 3  # resident bf16 gamma W-slice (zero-copy)
CB_STAT_GLOBAL = 7  # resident fp32 1/RMS tiles, one per tile-row (zero-copy)
CB_OUT = 16  # resident bf16 output W-slice (zero-copy)
CB_NORM = 26  # pass-2 intermediate x*rstd (scratch)

# batch_both_blk (DST-batch widening via EltwiseShape block_size on the broadcast+Bulk pass-2
# chain) DEADLOCKS on device — the block_size lever is not expressible on these chains, so it is
# excluded from the run set. The kernel branch (variant 3) is kept for the record; the batching
# win therefore comes purely from reducing chain COUNT, not from widening the DST-lane batch.
VARIANTS = ("baseline", "batch_gamma", "batch_both")
BASELINE = "baseline"

_VARIANT_ID = {"baseline": 0, "batch_gamma": 1, "batch_both": 2, "batch_both_blk": 3}


def cb_norm_depth_for(variant, per_w_t, c_rows):
    """L1 depth of the pass-2 intermediate cb_norm (in tiles) for a variant."""
    if variant == "baseline":
        return 2 * per_w_t
    # batched variants stage the whole round's x*rstd in cb_norm before the gamma pass.
    return c_rows * per_w_t


def variant_is_valid(variant, per_w_t, c_rows, ht_local):
    if variant not in _VARIANT_ID:
        return False
    if per_w_t < 1 or c_rows < 1 or ht_local < 1:
        return False
    # batch_both_blk widens the DST batch to a whole tile-row (PER_W_T lanes); only meaningful
    # when PER_W_T > 1 (else it degenerates to batch_both).
    return True


# =============================================================================
# Compute kernel — one source for every variant. The variant selector + geometry are
# compile-time args; only the pass-2 chain structure changes between variants.
# CT args: [variant_id, PER_W_T, HT_LOCAL, C_ROWS, HAS_GAMMA, kernel_iters]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
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

    // srcA <- cb_x_in (bf16), srcB <- cb_stat_global (fp32), packer <- cb_out (bf16).
    compute_kernel_hw_startup(cb_x_in, cb_stat_global, cb_out);

    // Arm the resident zero-copy inputs ONCE (held for the whole kernel, never popped).
    cb_reserve_back(cb_x_in, shard_tiles);
    cb_push_back(cb_x_in, shard_tiles);
    cb_reserve_back(cb_stat_global, HT_LOCAL);
    cb_push_back(cb_stat_global, HT_LOCAL);
    if constexpr (HAS_GAMMA) {
        cb_reserve_back(cb_gamma, PER_W_T);
        cb_push_back(cb_gamma, PER_W_T);
    }
    cb_wait_front(cb_x_in, shard_tiles);
    cb_wait_front(cb_stat_global, HT_LOCAL);
    if constexpr (HAS_GAMMA) {
        cb_wait_front(cb_gamma, PER_W_T);
    }

    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {
        for (uint32_t r = 0; r < num_rounds; ++r) {
            const uint32_t base_t = r * C_ROWS;
            uint32_t C_this = HT_LOCAL - base_t;
            if (C_this > C_ROWS) {
                C_this = C_ROWS;
            }

            if constexpr (VARIANT == 0) {
                // ---- baseline: per-tile-row (2 chains per tile-row) ----
                for (uint32_t cc = 0; cc < C_this; ++cc) {
                    const uint32_t t = base_t + cc;
                    // x*rstd over the tile-row's PER_W_T tiles: Block x @ t*PER_W_T, Col rstd @ t.
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::of(1, PER_W_T),
                        ckl::BinaryFpu<
                            cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Col,
                            ckl::TileOffset::Set, ckl::TileOffset::Set>{t * PER_W_T, t},
                        ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                    if constexpr (HAS_GAMMA) {
                        // norm*gamma: Scalar norm (streamed, front-walked by pops), Row gamma.
                        ckl::eltwise_chain(
                            ckl::EltwiseShape::of(1, PER_W_T),
                            ckl::BinaryFpu<
                                cb_norm, cb_gamma, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row,
                                ckl::InputLifecycle::Streaming, ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                                ckl::OperandKind::Scalar, ckl::OperandKind::Row,
                                ckl::TileOffset::Unset, ckl::TileOffset::Set>{0, 0},
                            ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                    } else {
                        ckl::copy<cb_norm, cb_out>(ckl::EltwiseShape::of(1, PER_W_T));
                    }
                }
            } else if constexpr (VARIANT == 1) {
                // ---- batch_gamma: per-row x*rstd into a deep cb_norm, then ONE gamma chain ----
                for (uint32_t cc = 0; cc < C_this; ++cc) {
                    const uint32_t t = base_t + cc;
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::of(1, PER_W_T),
                        ckl::BinaryFpu<
                            cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Col,
                            ckl::TileOffset::Set, ckl::TileOffset::Set>{t * PER_W_T, t},
                        ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                }
                if constexpr (HAS_GAMMA) {
                    // ONE gamma chain over the whole round: Block norm (Bulk: wait all, pop at end),
                    // Row gamma (index = wt). gamma tile wt is identical for every tile-row.
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::of(C_this, PER_W_T),
                        ckl::BinaryFpu<
                            cb_norm, cb_gamma, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Row,
                            ckl::TileOffset::Unset, ckl::TileOffset::Unset>{0, 0},
                        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                } else {
                    ckl::copy<cb_norm, cb_out>(ckl::EltwiseShape::of(1, C_this * PER_W_T));
                }
            } else {
                // ---- batch_both / batch_both_blk: ONE x*rstd + ONE gamma chain per round ----
                // block_size = PER_W_T for batch_both_blk (widen the DST batch to a tile-row),
                // else 1. The chain clamps block_size to fit DEST.
                constexpr uint32_t BLK = (VARIANT == 3) ? PER_W_T : 1;
                // x*rstd over the whole round as a 2D grid: Block x @ base_t*PER_W_T (index
                // ht*PER_W_T + wt), Col rstd @ base_t (index = ht -> the per-row 1/RMS).
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(C_this, PER_W_T, BLK),
                    ckl::BinaryFpu<
                        cb_x_in, cb_stat_global, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                        ckl::OperandKind::Block, ckl::OperandKind::Col,
                        ckl::TileOffset::Set, ckl::TileOffset::Set>{base_t * PER_W_T, base_t},
                    ckl::PackTile<cb_norm, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                if constexpr (HAS_GAMMA) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(C_this, PER_W_T, BLK),
                        ckl::BinaryFpu<
                            cb_norm, cb_gamma, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk, ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                            ckl::OperandKind::Block, ckl::OperandKind::Row,
                            ckl::TileOffset::Unset, ckl::TileOffset::Unset>{0, 0},
                        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                } else {
                    ckl::copy<cb_norm, cb_out>(ckl::EltwiseShape::of(1, C_this * PER_W_T));
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
        # FIXED precision contract: bf16 in, HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT_GLOBAL, rstd),
        _scratch_cb(CB_NORM, cb_norm_depth_for(variant, per_w_t, c_rows)),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
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
