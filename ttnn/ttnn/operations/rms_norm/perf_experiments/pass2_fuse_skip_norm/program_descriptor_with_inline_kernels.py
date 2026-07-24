# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated single-core micro-bench for the rms_norm PASS 2 (out = x*rstd*gamma), studying ONE idea:

    ELIMINATE the cb_norm round-trip by FUSING the two pass-2 muls in one DEST-sync window.

Pass 2, per tile-row, computes:  out[r,c] = x[r,c] * rstd[r] * gamma[c]
  - rstd is a REDUCE_ROW result -> column-shaped ([Ht,1]) -> BroadcastDim::Col
  - gamma is [1, W]             -> row-shaped    ([1,Wt]) -> BroadcastDim::Row

CURRENT BASELINE (already-graduated Perf-1 pass2 = the shipped op): per tile-row, TWO FPU chains
with an intermediate cb_norm that round-trips L1:
  chain1: x * rstd  (Col bcast)  -> cb_norm  (PER_W_T packs to L1)
  chain2: norm*gamma (Row bcast)  -> cb_out  (PER_W_T unpacks from L1)
The cb_norm round-trip is PER_W_T packs + PER_W_T unpacks of pure intermediate overhead / tile-row.
Baseline runs with the shipped PASS2_RECONFIG_SKIP (only srcB reconfig, the genuinely-changing one).

THE NEW ANGLE (round 1 measured FPU dest-reuse of a SINGLE binary at 0.94-1.02x and blocked gamma
fusion because gamma needs a Row broadcast and DestReuseBinary has no BroadcastDim):
  Pre-REPLICATE gamma into full [32,32] tiles ONCE per kernel invocation (broadcast the [1,32] gamma
  row down all 32 rows; only PER_W_T tiles/core). Then `*gamma` becomes a PLAIN elementwise mul with
  NO broadcast, which DestReuseBinary (or an SFPU DEST-DEST mul) CAN express — so x*rstd can stay
  resident in DEST and be combined with gamma WITHOUT the cb_norm pack/unpack.
  The one-time replication cost is amortized over HT_LOCAL tile-rows (it runs once, then every
  tile-row's pass 2 reuses cb_gamma_full), so the bench replicates once per steady-state iter (an
  iter == one real launch == replicate + HT_LOCAL rows) to charge the amortization honestly.

Everything lives in sharded L1 on ONE Tensix core (zero-copy resident cb_x_in / cb_stat_global /
cb_gamma / cb_out) — no DRAM movement, so the measured delta is pure pass-2 compute.

FIXED precision contract (identical across every variant — NEVER tuned for speed):
  bf16 x, bf16 TILE gamma, fp32 rstd (cb_stat_global), HiFi2, fp32_dest_acc_en=False, approx=False.
Because the fused path keeps x*rstd in bf16-DEST (vs the baseline's bf16-L1 cb_norm), PCC may shift;
it is measured vs an fp32 torch reference for EVERY variant and reported per option.

Variant menu (baseline first):
  baseline               per-tile-row 2-chain through cb_norm, reconfig-skip (the SHIPPED pass 2).
  baseline_rowbatch      same 2-chain, but BOTH chains batched over a block of C tile-rows in one
                         grid(C, PER_W_T) walk each (cb_norm grows to C*PER_W_T). ± row-batching arm.
  fused_dstreuse         per-tile-row FUSED chain, replicated gamma: BinaryFpu(x*rstd, Col) -> DEST,
                         DestReuseBinary(DEST * gamma_full, plain Mul, Row-indexed) -> DEST, one pack
                         to cb_out. No cb_norm. (candidate b)
  fused_dstreuse_rowbatch  fused_dstreuse batched over C tile-rows (grid(C, PER_W_T)).
  fused_sfpu             per-tile-row FUSED chain, replicated gamma: BinaryFpu(x*rstd, Col) -> D0,
                         CopyTile(gamma_full -> D1), MulBinary(D0*D1 -> D0) [SFPU DEST-DEST], one pack
                         to cb_out. (candidate c — SFPU consumer of the second mul)
  fused_sfpu_rowbatch    fused_sfpu batched over C tile-rows.

Raw LLK justification: the one-time gamma replication uses raw `unary_bcast<ROW>(cb_gamma, wt, dst)`
because eltwise_chain's UnaryBcast element hard-codes in_tile_index=0 (it always reads tile 0 and has
no TileOffset), so it cannot walk the PER_W_T resident gamma tiles without popping them — and gamma
is held resident zero-copy (never popped). The raw call takes an explicit source tile index, so it
replicates each resident gamma tile in place. This is the sanctioned bcast-datacopy pattern.
"""

import ttnn

TILE = 32
TILE_BF16 = ttnn.tile_size(ttnn.bfloat16)  # 2048 bytes
TILE_FP32 = ttnn.tile_size(ttnn.float32)  # 4096 bytes

# CB assignment — mirrors the real xcore compute kernel's indices.
CB_X_IN = 1  # resident sharded W-slice x (bf16)
CB_GAMMA = 3  # resident gamma W-slice (bf16 TILE, row-broadcast form)
CB_OUT = 16  # resident sharded output (bf16)
CB_STAT_GLOBAL = 7  # resident 1/RMS (fp32), one tile per tile-row
CB_NORM = 26  # pass-2 intermediate x*rstd (bf16) — baseline variants only
CB_GAMMA_FULL = 27  # replicated [32,32] gamma tiles (bf16) — fused variants only

# variant -> method id (the `if constexpr` selector in the kernel)
_VARIANT_METHOD = {
    "baseline": 0,
    "baseline_rowbatch": 1,
    "fused_dstreuse": 2,
    "fused_dstreuse_rowbatch": 3,
    "fused_sfpu": 4,
    "fused_sfpu_rowbatch": 5,
    "fused_dstreuse_norepl": 6,  # ablation: gamma replicated ONCE outside the steady-state loop
}
VARIANTS = tuple(_VARIANT_METHOD)
BASELINE = "baseline"

_ROWBATCH = {"baseline_rowbatch", "fused_dstreuse_rowbatch", "fused_sfpu_rowbatch"}
_FUSED = {
    "fused_dstreuse",
    "fused_dstreuse_rowbatch",
    "fused_sfpu",
    "fused_sfpu_rowbatch",
    "fused_dstreuse_norepl",
}


def _cb_norm_tiles(variant, per_w_t, c_block):
    """cb_norm depth (baseline only): per-row double-buffers 2*PER_W_T; rowbatch holds C*PER_W_T."""
    if variant in _ROWBATCH:
        return c_block * per_w_t
    return 2 * per_w_t


# =============================================================================
# Compute kernel — one source, `method` (CT arg 0) selects the variant.
# CT args: [method, PER_W_T, HT_LOCAL, C_BLOCK, num_blocks, kernel_iters]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"

// out[r,c] = x[r,c] * rstd[r] * gamma[c].
void kernel_main() {
    constexpr uint32_t cb_x_in = 1, cb_gamma = 3, cb_out = 16, cb_stat_global = 7,
                       cb_norm = 26, cb_gamma_full = 27;

    constexpr uint32_t method     = get_compile_time_arg_val(0);
    constexpr uint32_t PER_W_T    = get_compile_time_arg_val(1);
    constexpr uint32_t HT_LOCAL   = get_compile_time_arg_val(2);
    constexpr uint32_t C_BLOCK    = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);

    using namespace compute_kernel_lib;

    constexpr bool rowbatch = (method == 1 || method == 3 || method == 5);
    constexpr bool fused    = (method >= 2);
    constexpr bool dstreuse = (method == 2 || method == 3 || method == 6);
    constexpr bool sfpu     = (method == 4 || method == 5);
    // method 6 (fused_dstreuse_norepl): identical to method 2, but gamma is replicated ONCE outside
    // the steady-state loop instead of once per iter — an ABLATION isolating the pure fusion cost
    // (per-iter excludes replication) from the amortized gamma-replication cost.
    constexpr bool repl_each_iter = fused && (method != 6);

    constexpr uint32_t shard_tiles = HT_LOCAL * PER_W_T;

    // Boot: srca = cb_x_in (bf16), srcb = cb_stat_global (fp32), pack = cb_out (bf16).
    compute_kernel_hw_startup(cb_x_in, cb_stat_global, cb_out);

    // Arm the resident inputs once (zero-copy sharded; never popped -> stay available every iter).
    cb_reserve_back(cb_x_in, shard_tiles);        cb_push_back(cb_x_in, shard_tiles);
    cb_reserve_back(cb_stat_global, HT_LOCAL);    cb_push_back(cb_stat_global, HT_LOCAL);
    cb_reserve_back(cb_gamma, PER_W_T);           cb_push_back(cb_gamma, PER_W_T);

    // One-time gamma replication (amortized over HT_LOCAL rows). RAW LLK: eltwise_chain's UnaryBcast
    // hard-codes in_tile_index=0, so it cannot index the PER_W_T resident gamma tiles without popping
    // them; unary_bcast<ROW>(cb, wt, dst) takes an explicit source tile index and replicates gamma
    // tile wt (row 0 broadcast down 32 rows). Then restore the pass-2 formats ONCE (the replication
    // left srca=cb_gamma / pack=cb_gamma_full) so the fused chains reconfig only the changing srcB.
    auto do_replicate = [&]() {
        if constexpr (fused) {
            cb_reserve_back(cb_gamma_full, PER_W_T);
            unary_bcast_init<ckernel::BroadcastType::ROW>(cb_gamma, cb_gamma_full);
            for (uint32_t wt = 0; wt < PER_W_T; ++wt) {
                tile_regs_acquire();
                unary_bcast<ckernel::BroadcastType::ROW>(cb_gamma, wt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_gamma_full);
                tile_regs_release();
            }
            cb_push_back(cb_gamma_full, PER_W_T);
            cb_wait_front(cb_gamma_full, PER_W_T);
            reconfig_data_format(cb_x_in, cb_x_in);
            pack_reconfig_data_format(cb_out);
        }
    };

    if constexpr (fused && !repl_each_iter) {
        do_replicate();  // ablation: replicate once, hold across every steady-state iter
    }

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (!fused) {
            // ================= BASELINE: 2-chain through cb_norm, reconfig-skip =================
            // reconfig-skip: srcA (cb_x_in / cb_norm) is always bf16 and pack (cb_norm / cb_out) is
            // always bf16 across the loop -> establish ONCE, then srcB-only folds (srcB genuinely
            // alternates fp32 rstd <-> bf16 gamma). Matches the shipped PASS2_RECONFIG_SKIP.
            reconfig_data_format(cb_x_in, cb_x_in);
            pack_reconfig_data_format(cb_out);
            if constexpr (!rowbatch) {
                for (uint32_t t = 0; t < HT_LOCAL; ++t) {
                    eltwise_chain(
                        EltwiseShape::of(1, PER_W_T),
                        BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                                  InputLifecycle::CallerManaged, InputLifecycle::CallerManaged,
                                  BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Block,
                                  OperandKind::Col, TileOffset::Set, TileOffset::Set>{t * PER_W_T, t},
                        PackTile<cb_norm, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                    eltwise_chain(
                        EltwiseShape::of(1, PER_W_T),
                        BinaryFpu<cb_norm, cb_gamma, BinaryFpuOp::Mul, BroadcastDim::Row,
                                  InputLifecycle::Streaming, InputLifecycle::CallerManaged,
                                  BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Scalar,
                                  OperandKind::Row, TileOffset::Unset, TileOffset::Set>{0, 0},
                        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                }
            } else {
                for (uint32_t b = 0; b < num_blocks; ++b) {
                    const uint32_t base_t = b * C_BLOCK;
                    eltwise_chain(
                        EltwiseShape::grid(C_BLOCK, PER_W_T),
                        BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                                  InputLifecycle::CallerManaged, InputLifecycle::CallerManaged,
                                  BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Block,
                                  OperandKind::Col, TileOffset::Set, TileOffset::Set>{base_t * PER_W_T, base_t},
                        PackTile<cb_norm, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                    eltwise_chain(
                        EltwiseShape::grid(C_BLOCK, PER_W_T),
                        BinaryFpu<cb_norm, cb_gamma, BinaryFpuOp::Mul, BroadcastDim::Row,
                                  InputLifecycle::Streaming, InputLifecycle::CallerManaged,
                                  BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Scalar,
                                  OperandKind::Row, TileOffset::Unset, TileOffset::Set>{0, 0},
                        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                }
            }
        } else {
            // ================= FUSED: replicate gamma, then no cb_norm round-trip =================
            if constexpr (repl_each_iter) {
                do_replicate();  // once per iter == once per real launch (amortized over HT_LOCAL)
            }

            if constexpr (dstreuse) {
                // ---- candidate (b): FPU DEST-reuse of the second mul ----
                // BinaryFpu writes x*rstd to DEST; DestReuseBinary reuses DEST as srcA and reads
                // gamma_full into srcB (plain Mul, no broadcast), packs once to cb_out.
                if constexpr (!rowbatch) {
                    for (uint32_t t = 0; t < HT_LOCAL; ++t) {
                        eltwise_chain(
                            EltwiseShape::of(1, PER_W_T),
                            BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                                      InputLifecycle::CallerManaged, InputLifecycle::CallerManaged,
                                      BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Block,
                                      OperandKind::Col, TileOffset::Set, TileOffset::Set>{t * PER_W_T, t},
                            DestReuseBinary<cb_gamma_full, BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA,
                                            InputLifecycle::HeldBulk, DestReuseReconfig::SrcB,
                                            Dst::D0, Dst::D0, OperandKind::Row, TileOffset::Unset>{},
                            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                    }
                } else {
                    for (uint32_t b = 0; b < num_blocks; ++b) {
                        const uint32_t base_t = b * C_BLOCK;
                        eltwise_chain(
                            EltwiseShape::grid(C_BLOCK, PER_W_T),
                            BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                                      InputLifecycle::CallerManaged, InputLifecycle::CallerManaged,
                                      BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Block,
                                      OperandKind::Col, TileOffset::Set, TileOffset::Set>{base_t * PER_W_T, base_t},
                            DestReuseBinary<cb_gamma_full, BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA,
                                            InputLifecycle::HeldBulk, DestReuseReconfig::SrcB,
                                            Dst::D0, Dst::D0, OperandKind::Row, TileOffset::Unset>{},
                            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                    }
                }
            } else {
                // ---- candidate (c): SFPU DEST-DEST mul of the second mul ----
                // BinaryFpu writes x*rstd to D0; CopyTile loads gamma_full into D1; MulBinary
                // (SFPU) multiplies D0*D1 -> D0, packs once to cb_out. The replication block left
                // srcB bound to cb_gamma, so BinaryFpu MUST reconfig srcB back to cb_stat_global
                // (fp32) each chain (SrcB); srcA stays bf16 across BinaryFpu(cb_x_in)/CopyTile(
                // cb_gamma_full) so no srcA reconfig is needed.
                if constexpr (!rowbatch) {
                    for (uint32_t t = 0; t < HT_LOCAL; ++t) {
                        eltwise_chain(
                            EltwiseShape::of(1, PER_W_T),
                            BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                                      InputLifecycle::CallerManaged, InputLifecycle::CallerManaged,
                                      BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Block,
                                      OperandKind::Col, TileOffset::Set, TileOffset::Set>{t * PER_W_T, t},
                            CopyTile<cb_gamma_full, Dst::D1, InputLifecycle::HeldBulk,
                                     CopyTileReconfig::None, OperandKind::Row, TileOffset::Unset>{},
                            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                    }
                } else {
                    for (uint32_t b = 0; b < num_blocks; ++b) {
                        const uint32_t base_t = b * C_BLOCK;
                        eltwise_chain(
                            EltwiseShape::grid(C_BLOCK, PER_W_T),
                            BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                                      InputLifecycle::CallerManaged, InputLifecycle::CallerManaged,
                                      BinaryDataFormatReconfig::SrcB, Dst::D0, OperandKind::Block,
                                      OperandKind::Col, TileOffset::Set, TileOffset::Set>{base_t * PER_W_T, base_t},
                            CopyTile<cb_gamma_full, Dst::D1, InputLifecycle::HeldBulk,
                                     CopyTileReconfig::None, OperandKind::Row, TileOffset::Unset>{},
                            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                            PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::None>{});
                    }
                }
            }
            if constexpr (repl_each_iter) {
                cb_pop_front(cb_gamma_full, PER_W_T);  // reproduced next iter
            }
        }

        // Drain the output between steady-state iterations; leave the last pass resident for readback.
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, shard_tiles);
            cb_pop_front(cb_out, shard_tiles);
        }
    }

    if constexpr (fused && !repl_each_iter) {
        cb_pop_front(cb_gamma_full, PER_W_T);  // ablation held gamma_full across the whole loop
    }
    cb_pop_front(cb_x_in, shard_tiles);
    cb_pop_front(cb_stat_global, HT_LOCAL);
    cb_pop_front(cb_gamma, PER_W_T);
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
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat16, page_size=TILE_BF16)
    return ttnn.CBDescriptor(total_size=TILE_BF16 * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])


def create_program_descriptor(input_tensors, output_tensor, *, variant, per_w_t, ht_local, c_block, kernel_iters=1):
    if variant not in _VARIANT_METHOD:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if per_w_t < 1 or ht_local < 1 or c_block < 1 or kernel_iters < 1:
        raise ValueError("per_w_t, ht_local, c_block, kernel_iters must be positive")
    if variant in _ROWBATCH and ht_local % c_block:
        raise ValueError(f"c_block={c_block} must divide ht_local={ht_local} for {variant}")

    x, gamma, stat = input_tensors
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError("x must be bfloat16 TILE_LAYOUT")
    if gamma.dtype != ttnn.bfloat16 or gamma.layout != ttnn.TILE_LAYOUT:
        raise ValueError("gamma must be bfloat16 TILE_LAYOUT")
    if stat.dtype != ttnn.float32 or stat.layout != ttnn.TILE_LAYOUT:
        raise ValueError("stat (1/RMS) must be float32 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.bfloat16 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be bfloat16 TILE_LAYOUT")

    method = _VARIANT_METHOD[variant]
    num_blocks = (ht_local // c_block) if variant in _ROWBATCH else 0

    compile_time_args = [method, per_w_t, ht_local, c_block, num_blocks, kernel_iters]

    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        # FIXED precision contract for the focus case — identical for every variant.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, gamma),
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT_GLOBAL, stat),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    if variant in _FUSED:
        cbs.append(_scratch_cb(CB_GAMMA_FULL, per_w_t))
    else:
        cbs.append(_scratch_cb(CB_NORM, _cb_norm_tiles(variant, per_w_t, c_block)))

    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_pass2(input_tensors, *, variant, per_w_t, ht_local, c_block, kernel_iters=1):
    """Allocate the sharded output and run one variant. Output = [HT_LOCAL*32, PER_W_T*32] bf16."""
    x = input_tensors[0]
    h = ht_local * TILE
    w = per_w_t * TILE
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h, w]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        x.device(),
        create_sharded_memory_config((h, w)),
    )
    descriptor = create_program_descriptor(
        input_tensors,
        output,
        variant=variant,
        per_w_t=per_w_t,
        ht_local=ht_local,
        c_block=c_block,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([*input_tensors, output], descriptor)
