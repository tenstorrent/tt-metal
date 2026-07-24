# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated single-core micro-bench for the block-sharded rms_norm PASS 2 (x*rstd*gamma).

Pass 2, per tile-row, computes:  out[r,c] = x[r,c] * rstd[r] * gamma[c]
  - rstd is a REDUCE_ROW result -> column-shaped ([Ht,1]) -> BroadcastDim::Col
  - gamma is [1, W]             -> row-shaped    ([1,Wt]) -> BroadcastDim::Row

The current op computes this as TWO FPU chains per tile-row, with an intermediate cb_norm that
round-trips L1 (pack after x*rstd, unpack before *gamma):
  chain1: x * rstd  (Col bcast)  -> cb_norm  (packed to L1)
  chain2: norm * gamma (Row bcast) -> cb_out

Everything lives in sharded L1 on one Tensix core (zero-copy resident cb_x_in / cb_stat_global /
cb_gamma / cb_out) — no DRAM movement, so the measured delta is pure pass-2 compute: the per-call
eltwise-chain init / data-format reconfig, and the cb_norm pack+unpack.

FIXED precision contract (identical across every variant — never tuned for speed):
  bf16 x, bf16 TILE gamma, fp32 rstd (cb_stat_global), HiFi2, fp32_dest_acc_en=False, approx=False.

Variants (baseline first):
  baseline        per-tile-row 2-chain, every chain re-issues BinaryDataFormatReconfig::Input +
                  PackTileReconfig::Output (the op's current PASS2_BATCH form).
  reconfig_skip   same 2-chain structure, but drop the data-format reconfigs that are provably
                  constant across the steady-state loop and keep only the ones that genuinely
                  change format. srcA path is always bf16 (cb_x_in / cb_norm) -> droppable; the
                  pack path is always bf16 (cb_norm / cb_out) -> droppable; srcB alternates
                  fp32 (rstd) <-> bf16 (gamma) every chain -> REQUIRED (kept as SrcB).
  rowblock        batch BOTH chains over a block of C tile-rows in one grid(C, PER_W_T) walk each,
                  instead of looping the 2 chains per tile-row — amortizing the per-chain init /
                  reconfig / pipeline fill-drain over C rows (the compute_block_size lever, applied
                  across the C tile-rows a cross-core round already has in hand). Costs L1: cb_norm
                  grows 2*PER_W_T -> C*PER_W_T.
  rowblock_skip   rowblock + reconfig_skip (the two amortize-overhead levers compounded).

NOT benched here (measured dead-ends, documented in the report):
  * FPU DEST-reuse fusion of (x*rstd)*gamma. Both factors are broadcasts, so the second mul is
    intrinsically a broadcast on the reuse side; DestReuseBinary has no BroadcastDim and the raw
    mul_tiles_bcast_{rows,cols}/any_tiles_bcast primitives read TWO CBs (never DEST) -> the fusion
    is not expressible without a new broadcast-capable dest-reuse LLK primitive. Refinement 6g
    already measured the (non-broadcast) FPU dest-reuse at 0.94-1.00x (never beats pack-to-L1).
  * SFPU-resident strip-cb_norm: would need gamma row-broadcast into DEST + an SFPU mul;
    compute_fusion measured a plain SFPU mul at 0.58x vs the FPU -> strictly a regression.
"""

import ttnn

TILE = 32
TILE_BF16 = ttnn.tile_size(ttnn.bfloat16)  # 2048 bytes
TILE_FP32 = ttnn.tile_size(ttnn.float32)  # 4096 bytes

# CB assignment — mirrors the real xcore compute kernel's indices.
CB_X_IN = 1  # resident sharded W-slice x (bf16)
CB_GAMMA = 3  # resident gamma W-slice (bf16 TILE)
CB_OUT = 16  # resident sharded output (bf16)
CB_NORM = 26  # pass-2 intermediate x*rstd (bf16)
CB_STAT_GLOBAL = 7  # resident 1/RMS (fp32), one tile per tile-row

# variant -> method id (the `if constexpr` selector in the kernel)
_VARIANT_METHOD = {
    "baseline": 0,
    "reconfig_skip": 1,
    "rowblock": 2,
    "rowblock_skip": 3,
}
VARIANTS = tuple(_VARIANT_METHOD)
BASELINE = "baseline"

_IS_ROWBLOCK = {"rowblock", "rowblock_skip"}


def _cb_norm_tiles(variant, per_w_t, c_block):
    """cb_norm depth: per-row variants double-buffer 2*PER_W_T; rowblock holds C_BLOCK*PER_W_T."""
    if variant in _IS_ROWBLOCK:
        return c_block * per_w_t
    return 2 * per_w_t


# =============================================================================
# Compute kernel — one source, `method` (CT arg 0) selects the variant.
# CT args: [method, PER_W_T, HT_LOCAL, C_BLOCK, num_blocks, kernel_iters]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

// out[r,c] = x[r,c] * rstd[r] * gamma[c]   (two FPU chains through cb_norm).
void kernel_main() {
    constexpr uint32_t cb_x_in = 1, cb_gamma = 3, cb_out = 16, cb_norm = 26, cb_stat_global = 7;

    constexpr uint32_t method     = get_compile_time_arg_val(0);
    constexpr uint32_t PER_W_T    = get_compile_time_arg_val(1);
    constexpr uint32_t HT_LOCAL   = get_compile_time_arg_val(2);
    constexpr uint32_t C_BLOCK    = get_compile_time_arg_val(3);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);

    using namespace compute_kernel_lib;

    constexpr bool skip      = (method == 1 || method == 3);
    constexpr bool rowblock  = (method == 2 || method == 3);

    // reconfig-skip: srcA (x_in / norm) is always bf16 and the pack target (norm / out) is always
    // bf16 across the steady-state loop, so their reconfig is wasted MMIO -> drop it (SrcB keeps only
    // the srcB fold, PackTileReconfig::None keeps the boot-programmed pack format). srcB genuinely
    // alternates fp32(rstd) <-> bf16(gamma) every chain, so SrcB is REQUIRED either way.
    constexpr auto RC  = skip ? BinaryDataFormatReconfig::SrcB : BinaryDataFormatReconfig::Input;
    constexpr auto PRC = skip ? PackTileReconfig::None : PackTileReconfig::Output;

    constexpr uint32_t shard_tiles = HT_LOCAL * PER_W_T;

    // Boot programs srca = cb_x_in (bf16), srcb = cb_stat_global (fp32), pack = cb_out (bf16) — the
    // exact formats reconfig_skip relies on being live before the loop.
    compute_kernel_hw_startup(cb_x_in, cb_stat_global, cb_out);

    // Arm the resident inputs once (zero-copy sharded; never popped -> stay available every iter).
    cb_reserve_back(cb_x_in, shard_tiles);        cb_push_back(cb_x_in, shard_tiles);
    cb_reserve_back(cb_stat_global, HT_LOCAL);    cb_push_back(cb_stat_global, HT_LOCAL);
    cb_reserve_back(cb_gamma, PER_W_T);           cb_push_back(cb_gamma, PER_W_T);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (!rowblock) {
            // ---- per-tile-row 2-chain (baseline / reconfig_skip) ----
            for (uint32_t t = 0; t < HT_LOCAL; ++t) {
                // chain1: x * rstd  (Col bcast, rstd tile = t) -> cb_norm
                eltwise_chain(
                    EltwiseShape::of(1, PER_W_T),
                    BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                              InputLifecycle::CallerManaged, InputLifecycle::CallerManaged, RC,
                              Dst::D0, OperandKind::Block, OperandKind::Col,
                              TileOffset::Set, TileOffset::Set>{t * PER_W_T, t},
                    PackTile<cb_norm, OutputLifecycle::Streaming, PRC>{});
                // chain2: norm * gamma (Row bcast, gamma tile = wt) -> cb_out
                eltwise_chain(
                    EltwiseShape::of(1, PER_W_T),
                    BinaryFpu<cb_norm, cb_gamma, BinaryFpuOp::Mul, BroadcastDim::Row,
                              InputLifecycle::Streaming, InputLifecycle::CallerManaged, RC,
                              Dst::D0, OperandKind::Scalar, OperandKind::Row,
                              TileOffset::Unset, TileOffset::Set>{0, 0},
                    PackTile<cb_out, OutputLifecycle::Streaming, PRC>{});
            }
        } else {
            // ---- batched over a block of C_BLOCK tile-rows (rowblock / rowblock_skip) ----
            for (uint32_t b = 0; b < num_blocks; ++b) {
                const uint32_t base_t = b * C_BLOCK;
                // chain1: x * rstd over grid(C_BLOCK, PER_W_T). Block x from base_t*PER_W_T,
                // Col rstd = base_t + ht. Streams C_BLOCK*PER_W_T tiles into cb_norm.
                eltwise_chain(
                    EltwiseShape::grid(C_BLOCK, PER_W_T),
                    BinaryFpu<cb_x_in, cb_stat_global, BinaryFpuOp::Mul, BroadcastDim::Col,
                              InputLifecycle::CallerManaged, InputLifecycle::CallerManaged, RC,
                              Dst::D0, OperandKind::Block, OperandKind::Col,
                              TileOffset::Set, TileOffset::Set>{base_t * PER_W_T, base_t},
                    PackTile<cb_norm, OutputLifecycle::Streaming, PRC>{});
                // chain2: norm * gamma over grid(C_BLOCK, PER_W_T). cb_norm front-walked (Scalar),
                // gamma Row = wt (same gamma every row).
                eltwise_chain(
                    EltwiseShape::grid(C_BLOCK, PER_W_T),
                    BinaryFpu<cb_norm, cb_gamma, BinaryFpuOp::Mul, BroadcastDim::Row,
                              InputLifecycle::Streaming, InputLifecycle::CallerManaged, RC,
                              Dst::D0, OperandKind::Scalar, OperandKind::Row,
                              TileOffset::Unset, TileOffset::Set>{0, 0},
                    PackTile<cb_out, OutputLifecycle::Streaming, PRC>{});
            }
        }

        // Drain the output between steady-state iterations; leave the last pass resident for readback.
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, shard_tiles);
            cb_pop_front(cb_out, shard_tiles);
        }
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
    if variant in _IS_ROWBLOCK and ht_local % c_block:
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
    num_blocks = (ht_local // c_block) if variant in _IS_ROWBLOCK else 0

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
        _scratch_cb(CB_NORM, _cb_norm_tiles(variant, per_w_t, c_block)),
    ]

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
