# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED micro-benchmark: rms_norm's PASS B (`out = x * stat_col * gamma_row`).

ONE idea under test, with a menu of spellings:

    baseline    TWO `eltwise_chain` calls with a full bf16 intermediate CB between
                them -- exactly rms_norm_compute.cpp's `compute_scale` +
                `compute_gamma_mul`:
                    x * bcast_col(stat)  -> cb_normalized   (L1)
                    cb_normalized * bcast_row(gamma) -> cb_output_tiles

    fused       ONE chain / ONE DEST window: the scale's DEST result is multiplied
                by gamma IN PLACE (`DestReuseBinary`, DEST -> srcB, gamma -> srcA)
                and packed once.  cb_normalized disappears entirely.

                `DestReuseBinary` carries NO BroadcastDim (the LLK
                `binary_dest_reuse_tiles` has no broadcast form), so the fused
                spelling needs gamma as a FULL tile.  It is materialised ONCE per
                core by a `UnaryBcast<BroadcastDim::Row>` pass over the WT gamma
                tiles (that pass is inside the measured kernel, not hidden).

    outer       the third framing: precompute the outer product
                `sg = gamma_full * bcast_col(stat)` (rows x WT tiles) and then a
                plain non-broadcast `x * sg`.  Same tile count as the baseline's two
                passes, no broadcast on the hot pass.

Two orthogonal LEVERS are knobs on both families, because `master.md`
(`compute_block_size`, its `report_reconfig_ablation.md`) says they are where the
per-call cost actually lives:

    blk         the chain's DEST-lane block size.  The chain walk is ELEMENT-MAJOR
                inside a block (eltwise_chain.inl `elem_apply_compute`): each
                element's `init()` + format reconfig is emitted ONCE per outer iter
                and then its `exec` runs over all `blk` lanes.  For the fused chain
                that is the whole ballgame -- `BinaryFpu` and `DestReuseBinary` are
                different math-MOP types, so the chain is NOT init-hoistable
                (`chain_math_mop_uniform`) and pays both MOP inits per outer iter.
                blk amortises them over blk tiles.

    norecfg     drop the data-format reconfigs that are provably no-ops, and ONLY
                those.  In the real op the per-block format state machine is
                    square(pack bf16) -> reduce(srcA bf16, pack fp32)
                      -> scale(srcA x bf16, srcB stat fp32, pack bf16)
                      -> gamma(srcA norm bf16, srcB gamma bf16, pack bf16)
                so exactly three of pass B's six reconfigs are inert:
                    scale.srcA   (bf16 -> bf16, after the reduce's cb_x_squared)
                    gamma.srcA   (bf16 -> bf16, after the scale's x)
                    gamma.pack   (bf16 -> bf16, after the scale's output)
                and three are REQUIRED and stay on in every variant:
                    scale.srcB   (bf16 -> fp32 stat)
                    gamma.srcB   (fp32 -> bf16 gamma)
                    scale.pack   (fp32 reduce output -> bf16)
                The bench keeps that split so the number transfers to the op.

Everything else is held constant / trivial (per /perf-lab concept isolation):
  * pass A, the finalize and the cross-core combine are NOT in the bench.  The stat
    arrives as a RESIDENT fp32 tensor-backed CB the compute kernel publishes itself,
    so no variant can hide a `cb_wait_front` on the multicast inside its number.
  * x and out are ZERO-COPY sharded tensors (`cb_descriptor_from_sharded_tensor`) --
    exactly the op's NATIVE_IN / NATIVE_OUT focus configuration, where the reader
    only publishes the resident shard and the writer only takes a completion barrier.
  * one core.  Pass B is embarrassingly parallel and identical on all 64.
  * NO device zones: every variant is measured by DEVICE KERNEL DURATION of a kernel
    that contains pass B and nothing else, so zone-count asymmetry between a
    two-call and a one-call variant cannot bias the comparison.

Precision contract (FIXED, identical in every variant -- never a lever):
  bf16 x / gamma / normalized / out, fp32 stat, TILE layout,
  math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.

Geometry knobs mirror the op's:
  rows   == BLOCK_ROWS          tile-rows per row-block (the chain's grid H)
  wt     == WT_CHUNK            width tiles per chunk   (the chain's grid W)
  blocks == number of row-blocks the core walks
  Focus (per core of the (1,1,8192,1024) BLOCK_SHARDED 64c case):
  rows=8, wt=4, blocks=4  ->  32 tile-rows, 128 tiles per pass, 4+4 chain calls.
"""

import ttnn

TILE = 32

# --- CB assignment.  0/1/2/4 are tensor-backed (zero-copy), 3/5/6 are scratch. ---
CB_X = 0  # bf16  x            ZERO-COPY on the input shard (the op's NATIVE_IN)
CB_STAT = 1  # fp32  1/rms      column-shaped, the op's cb_row_final
CB_GAMMA = 2  # bf16  gamma      row-shaped (valid in row 0), the op's cb_gamma_tiles
CB_NORM = 3  # bf16  scratch    the op's cb_normalized -- the CB the fusion deletes
CB_OUT = 4  # bf16  out        ZERO-COPY on the output shard (the op's NATIVE_OUT)
CB_GFULL = 5  # bf16  scratch    gamma replicated down all 32 rows (fused / outer)
CB_SG = 6  # bf16  scratch    stat (x) gamma outer product (outer only)

# variant name -> compile-time id.  Ordering is the report ordering.
VARIANTS = {
    "baseline": 0,  # the op today: two chains, blk=1, all reconfig on
    "baseline_blk": 1,  # + DEST-lane blocking
    "baseline_norecfg": 2,  # + the 3 provably-inert reconfigs dropped
    "baseline_blk_norecfg": 3,  # both levers, still two chains
    "fused": 4,  # ONE chain via DestReuseBinary, blk=1
    "fused_blk": 5,  # + DEST-lane blocking
    "fused_blk_norecfg": 6,  # both levers
    "outer": 7,  # precompute stat (x) gamma, then a plain mul
    "baseline_gfull_blk": 8,  # two chains, 2nd one NON-broadcast on a full gamma tile
    "baseline_up": 9,  # blk=1, but ONE reserve+push per chain call (Upfront/AtEnd)
    "baseline_blk_up": 10,  # blk + Upfront/AtEnd: the fewest CB ops reachable
}
BASELINE = "baseline"


def blk_for(wt):
    """DEST-lane block size: the largest divisor of `wt` that fits DEST.

    DEST holds 8 bf16 tiles at fp32_dest_acc_en=False / half sync
    (`DEST_AUTO_LIMIT`).  `EltwiseShape::grid` blocks along W, so a block may not
    straddle a tile-row; keeping it a DIVISOR of wt also keeps every outer iter a
    full block (`BlockTailSync::FullBlock`).
    """
    for b in (8, 4, 2, 1):
        if b <= wt and wt % b == 0:
            return b
    return 1


# =============================================================================
# Compute kernel — pass B, every variant in ONE source (CT-arg selected).
#
# CT args: [VARIANT, ROWS, WT, BLOCKS, HAS_GAMMA, BLK]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_bcast.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_stat = 1;
constexpr uint32_t cb_gamma = 2;
constexpr uint32_t cb_norm = 3;
constexpr uint32_t cb_out = 4;
constexpr uint32_t cb_gfull = 5;
constexpr uint32_t cb_sg = 6;
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);
    constexpr uint32_t WT = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCKS = get_compile_time_arg_val(3);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(4);
    constexpr uint32_t BLK = get_compile_time_arg_val(5);

    constexpr uint32_t TOTAL = ROWS * WT * BLOCKS;
    constexpr uint32_t ROWS_TOTAL = ROWS * BLOCKS;
    constexpr bool HAS_G = (HAS_GAMMA != 0);

    // ---- variant decode ---------------------------------------------------
    constexpr bool FUSED = (VARIANT == 4 || VARIANT == 5 || VARIANT == 6);
    constexpr bool OUTER = (VARIANT == 7);
    constexpr bool GFULL_BASE = (VARIANT == 8);
    constexpr bool NEED_GFULL = HAS_G && (FUSED || OUTER || GFULL_BASE);
    constexpr bool NORECFG = (VARIANT == 2 || VARIANT == 3 || VARIANT == 6);
    // ONE reserve + ONE push per chain CALL instead of per block -- isolates "is the
    // block win just fewer CB ops?" from "is it the amortised DEST handshake + init?".
    constexpr bool UPFRONT_OUT = (VARIANT == 9 || VARIANT == 10);
    // blk == 1 for the "as the op spells it" variants, BLK for the blocked ones.
    constexpr uint32_t B = (VARIANT == 0 || VARIANT == 2 || VARIANT == 4 || VARIANT == 9) ? 1u : BLK;
    // A PACK lifecycle is emitted ONCE PER OUTER ITER, and an outer iter covers B
    // tiles (eltwise_chain.inl `elem_apply_pack`: the reserve/push are outside the
    // lane loop).  So PerTile reserve/push is only correct at B == 1 -- at B > 1 it
    // reserves 1 page and packs B, which corrupts the ring and hangs the consumer.
    // PerChunk is the blocked spelling (count == the block's valid tiles).
    constexpr auto RSV = UPFRONT_OUT ? ckl::ReservePolicy::Upfront
                                     : ((B == 1) ? ckl::ReservePolicy::PerTile : ckl::ReservePolicy::PerChunk);
    constexpr auto PSH =
        UPFRONT_OUT ? ckl::PushPolicy::AtEnd : ((B == 1) ? ckl::PushPolicy::PerTile : ckl::PushPolicy::PerChunk);

    // ---- reconfig policy (see the module docstring for the legality argument) --
    constexpr auto RC_ON = ckl::DataFormatReconfig::Enabled;
    constexpr auto RC_OFF = ckl::DataFormatReconfig::Disabled;
    constexpr auto RC_A = NORECFG ? RC_OFF : RC_ON;         // srcA: bf16 throughout
    constexpr auto RC_GPACK = NORECFG ? RC_OFF : RC_ON;     // gamma chain's pack: bf16 -> bf16
    // scale.srcB (fp32 stat), gamma.srcB (bf16 gamma) and scale.pack (fp32 -> bf16)
    // are REQUIRED in the op and stay RC_ON in every variant.

    // Boot: srcA = bf16 x, srcB = fp32 stat, pack = bf16 out.  Identical in every
    // variant, so no measured delta can come from a different hw_configure.
    compute_kernel_hw_startup(cb_x, cb_stat, cb_out);

    // ---- publish the RESIDENT operands (stand-ins for the isolated-away stages) --
    // x  : the op's `reader_native_publish` -- the whole shard, once, no NoC.
    // out: nothing to publish (the writer only takes a completion barrier).
    // stat: pass A + finalize + combine, replaced by a resident fp32 tensor so no
    //       variant can absorb a multicast wait.
    cb_reserve_back(cb_x, TOTAL);
    cb_push_back(cb_x, TOTAL);
    cb_reserve_back(cb_stat, ROWS_TOTAL);
    cb_push_back(cb_stat, ROWS_TOTAL);
    if constexpr (HAS_G) {
        cb_reserve_back(cb_gamma, WT);
        cb_push_back(cb_gamma, WT);
    }

    // ---- ONE-TIME: gamma replicated down all 32 rows ----------------------
    // `DestReuseBinary` has no BroadcastDim, so the fused (and outer) spellings need
    // a full gamma tile.  WT tiles of work per CORE (not per row-block, not per
    // tile-row), inside the measured kernel.
    if constexpr (NEED_GFULL) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(WT),
            ckl::UnaryBcast<
                ckl::BroadcastDim::Row,
                ckl::input(cb_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block)>{},
            ckl::PackTile<ckl::output(cb_gfull)>{});
    }

    // The op's pass-B operand specs, verbatim (focus regime: NUM_W_CHUNKS == 1, so
    // TileOffset is Unset and x is popped AtEnd by pass B).
    constexpr auto X_IN_B =
        ckl::input(cb_x, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block, RC_A);
    constexpr auto STAT_IN =
        ckl::input(cb_stat, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col, RC_ON);
    constexpr auto G_IN =
        ckl::input(cb_gamma, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row, RC_ON);
    constexpr auto GFULL_IN_A =
        ckl::input(cb_gfull, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row, RC_A);
    constexpr auto GFULL_IN_B =
        ckl::input(cb_gfull, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row, RC_ON);
    constexpr uint32_t NORM_OUT = HAS_G ? cb_norm : cb_out;

    for (uint32_t blk = 0; blk < BLOCKS; ++blk) {
        if constexpr (FUSED && HAS_G) {
            // THE IDEA: x * bcast_col(stat) stays in DEST and is multiplied by the
            // full gamma tile IN PLACE (DEST -> srcB, gamma -> srcA), then packed
            // once.  cb_normalized never exists.
            //
            // DEST_TO_SRCB (not SRCA) on purpose: it routes the CB operand to srcA,
            // which is bf16 in BOTH elements, so the chain's srcB programming
            // (fp32 stat) is untouched inside the chain.  DEST_TO_SRCA would flip
            // srcB between fp32 and bf16 on every tile.
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(ROWS, WT, B),
                ckl::BinaryFpu<X_IN_B, STAT_IN, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{},
                ckl::DestReuseBinary<GFULL_IN_A, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCB>{},
                ckl::PackTile<ckl::output(cb_out, RSV, PSH, RC_ON)>{});
        } else if constexpr (OUTER && HAS_G) {
            // sg = gamma_full * bcast_col(stat): a FULL-tile outer product, rows x WT
            // tiles of extra FPU work per block.
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(ROWS, WT, B),
                ckl::BinaryFpu<GFULL_IN_A, STAT_IN, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{},
                ckl::PackTile<ckl::output(cb_sg, RSV, PSH, RC_ON)>{});
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(ROWS, WT, B),
                ckl::BinaryFpu<
                    X_IN_B,
                    ckl::input(cb_sg, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block, RC_ON),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None>{},
                ckl::PackTile<ckl::output(cb_out, RSV, PSH, RC_ON)>{});
        } else {
            // ---- the op's current spelling: two chains through cb_normalized ----
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(ROWS, WT, B),
                ckl::BinaryFpu<X_IN_B, STAT_IN, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{},
                ckl::PackTile<ckl::output(NORM_OUT, RSV, PSH, RC_ON)>{});
            if constexpr (HAS_G) {
                constexpr auto NORM_IN = ckl::input(
                    cb_norm, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block, RC_A);
                constexpr auto G_OUT = ckl::output(cb_out, RSV, PSH, RC_GPACK);
                if constexpr (GFULL_BASE) {
                    // Same two chains, but the second multiply is NON-broadcast on the
                    // replicated gamma tile -- isolates "is the FPU row-broadcast
                    // itself costing anything?" from the fusion question.
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(ROWS, WT, B),
                        ckl::BinaryFpu<NORM_IN, GFULL_IN_B, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::None>{},
                        ckl::PackTile<G_OUT>{});
                } else {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::grid(ROWS, WT, B),
                        ckl::BinaryFpu<NORM_IN, G_IN, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{},
                        ckl::PackTile<G_OUT>{});
                }
            }
        }
        cb_pop_front(cb_stat, ROWS);
    }

    if constexpr (HAS_G) {
        cb_pop_front(cb_gamma, WT);
    }
}
"""


# =============================================================================
# Reader / writer — the op's NATIVE_IN / NATIVE_OUT halves, which is nothing at
# all on the read side and a completion barrier on the write side.  The writer is
# present because it is cb_out's single consumer.
#
# writer CT args: [TOTAL]
# =============================================================================
_READER_KERNEL = r"""
#include <cstdint>
void kernel_main() {}
"""

_WRITER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = 4;
    constexpr uint32_t TOTAL = get_compile_time_arg_val(0);
    // NATIVE_OUT: compute packed into the shard itself.  Take the completion
    // barrier and leave the pages pushed -- they ARE the tensor.
    cb_wait_front(cb_out, TOTAL);
}
"""


# =============================================================================
# Host side
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_memory_config(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, data_format, num_pages):
    page = ttnn.tile_size(data_format)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=data_format, page_size=page)
    return ttnn.CBDescriptor(total_size=page * num_pages, core_ranges=_single_core(), format_descriptors=[fmt])


def compute_config():
    """The op's PINNED focus-case precision contract.  Identical for every variant."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def create_program_descriptor(x, stat, gamma, out, *, variant, rows, wt, blocks, has_gamma, blk=None):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {tuple(VARIANTS)}, got {variant!r}")
    if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
        raise ValueError("x must be bfloat16 TILE_LAYOUT")
    if stat.dtype != ttnn.float32 or stat.layout != ttnn.TILE_LAYOUT:
        raise ValueError("the stat tensor must be float32 TILE_LAYOUT (the op's fp32 stat CB)")
    if out.dtype != ttnn.bfloat16 or out.layout != ttnn.TILE_LAYOUT:
        raise ValueError("out must be bfloat16 TILE_LAYOUT")

    vid = VARIANTS[variant]
    total = rows * wt * blocks
    if blk is None:
        blk = blk_for(wt)
    if wt % blk != 0 or blk > 8:
        raise ValueError(f"blk={blk} must divide wt={wt} and fit DEST (<= 8 bf16 tiles)")
    cores = _single_core()
    tensors = [x, stat, out]

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT, stat),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    if has_gamma:
        if gamma is None:
            raise ValueError("has_gamma=True needs a gamma tensor")
        tensors.insert(2, gamma)
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, gamma))
        # Only allocate what the variant actually references, so no variant is
        # silently handed a different L1 footprint than it needs.
        fused = vid in (4, 5, 6)
        outer = vid == 7
        if not (fused or outer):
            cbs.append(_scratch_cb(CB_NORM, ttnn.bfloat16, rows * wt))  # the op's cb_normalized
        if fused or outer or vid == 8:
            cbs.append(_scratch_cb(CB_GFULL, ttnn.bfloat16, wt))
        if outer:
            cbs.append(_scratch_cb(CB_SG, ttnn.bfloat16, rows * wt))

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader_rt[0][0] = []
    writer_rt[0][0] = []
    compute_rt[0][0] = []

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[],
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[total],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[vid, rows, wt, blocks, 1 if has_gamma else 0, blk],
        runtime_args=compute_rt,
        config=compute_config(),
    )
    return tensors, ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def run(x, stat, gamma, out, *, variant, rows, wt, blocks, has_gamma=True, blk=None):
    tensors, descriptor = create_program_descriptor(
        x, stat, gamma, out, variant=variant, rows=rows, wt=wt, blocks=blocks, has_gamma=has_gamma, blk=blk
    )
    return ttnn.generic_op(tensors, descriptor)
