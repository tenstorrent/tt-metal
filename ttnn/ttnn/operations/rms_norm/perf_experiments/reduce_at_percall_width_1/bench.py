# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED micro-benchmark: which reduce DATAPATH belongs at which PER-CALL width?

THE IDEA (one, and only one): rms_norm picks between the two `reduce` datapaths
(`ReduceAlgorithm::AccumulateViaAdd` vs `ReduceTile`) with a host-side predicate that
is evaluated against `wt_chunk` -- but Refinement 4's L6d DEST fold (D12) collapses the
reduce's ACTUAL per-call reduce-dim width to `X_SQUARED_WT`, which is 1 whenever the
fold is on.  So the predicate's `REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT` floor, which exists
precisely to keep AccumulateViaAdd off narrow reduces, is reading the wrong number and
the op runs AccumulateViaAdd at a per-call width of 1.

This bench measures the crossover directly: BOTH datapaths at every
(per-call width) x (tile-rows per call) point, ns AND precision, at the op's FROZEN
precision contract.

Three benchable stages, selected by MODE (each holds everything else constant):

  MODE_REDUCE_ONLY (0)  the reduce alone.  cb_in is a ZERO-COPY L1-sharded tensor the
                        host fills with x^2-shaped data, so both datapaths see
                        byte-identical input at zero producer cost.  This is the
                        crossover measurement -- `width` here IS the reduce's per-call
                        reduce-dim width (the op's X_SQUARED_WT).

  MODE_SQ_REDUCE (1)    pass A as the op spells it today: `square` (eltwise_chain,
                        DestAccumulation::PerRow when X_SQ_WT == 1 -- the D12 fold) ->
                        cb_x_squared, then accumulate_reduce_block.  Measures the idea
                        in the op's real pass-A geometry, both datapaths.

  MODE_FUSED (2)        the SECOND option in this slot.  With the fold on, the square
                        has ALREADY summed the chunk's width tiles inside one DEST
                        register; all the reduce does is the within-tile 32-column sum.
                        So do that sum ON THE SQUARE'S OWN DEST WINDOW with
                        `sfpu_reduce<SUM, ..., REDUCE_ROW>` and pack straight to the
                        fp32 stat CB -- deleting cb_x_squared, its L1 round-trip, and
                        the reduce call entirely.  Two spellings:
                          FUSED_PER_ROW  one DEST acquire per tile-row
                          FUSED_BATCH    one DEST acquire per BATCH tile-rows, one
                                         sfpu_reduce call with rt_dim = BATCH

Precision contract (FROZEN -- identical in every variant, never a lever):
  bf16 in, fp32 stat CB, TILE layout, math_fidelity=HiFi2, fp32_dest_acc_en=False,
  math_approx_mode=False.  The two datapaths differ in accumulation DEPTH, so every
  point reports pcc + rel-RMS as well as ns.

Non-tile-aligned width is a first-class case here, because the two datapaths take
DIFFERENT partial mechanisms and the op's reader must emit different cb_scaler
contents for each:
  ReduceTile        cb_scaler = [full 1.0 scaler, PARTIAL scaler]  (SCALER_TILES 2)
  AccumulateViaAdd  cb_scaler = [0/1 MASK tile]                    (SCALER_TILES 1)
The bench drives both from the same CT args the op uses, and the host POISONS the pad
lanes so a leak is catastrophic rather than marginal.
"""

import ttnn

TILE = 32

# CB assignment.  0 and 3 are tensor-backed (zero-copy sharded), 1/2 are scratch.
CB_IN = 0  # bf16 -- x^2 tiles (MODE 0) or x tiles (MODE 1/2), ZERO-COPY on the input tensor
CB_SCALER = 1  # bf16 -- the op's boot scaler / partial scaler pair / 0-1 mask
CB_X_SQUARED = 2  # bf16 -- MODE 1 only (the CB the fused option deletes)
CB_OUT = 3  # fp32 -- the stat CB, ZERO-COPY on the output tensor

MODE_REDUCE_ONLY = 0
MODE_SQ_REDUCE = 1
MODE_FUSED = 2

# variant name -> (mode, algo_id, fused_batch)
#   algo_id 1 = ReduceAlgorithm::AccumulateViaAdd, 0 = Auto (resolves to ReduceTile)
VARIANT_SPEC = {
    # --- MODE 0: the crossover measurement, reduce alone -------------------
    "acc_add": (MODE_REDUCE_ONLY, 1, 0),
    "reduce_tile": (MODE_REDUCE_ONLY, 0, 0),
    # --- MODE 1: the op's real pass A (square + reduce) --------------------
    "sq_acc_add": (MODE_SQ_REDUCE, 1, 0),  # == the op TODAY on the focus shape
    "sq_reduce_tile": (MODE_SQ_REDUCE, 0, 0),  # == the predicate fix
    # --- MODE 2: option 2 -- the reduce rides the square's DEST window -----
    "fused_per_row": (MODE_FUSED, 0, 0),
    "fused_batch": (MODE_FUSED, 0, 1),
}
VARIANTS = tuple(VARIANT_SPEC)

# Datapath-only variant pair, for the width x rows crossover sweep.
DATAPATHS = ("acc_add", "reduce_tile")
# Pass-A variant menu, for the focus geometry.
PASS_A = ("sq_acc_add", "sq_reduce_tile", "fused_per_row", "fused_batch")


# =============================================================================
# Compute kernel — every variant in ONE source, CT-arg selected.
#
# CT args: [MODE, WIDTH, X_SQ_WT, NUM_BLOCKS, ALGO, PARTIAL_W, SCALER_TILES, BATCH]
# RT args: [rows]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_x_squared = 2;
constexpr uint32_t cb_out = 3;
}  // namespace

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t WIDTH = get_compile_time_arg_val(1);         // WT_CHUNK (== per-call width in MODE 0)
    constexpr uint32_t X_SQ_WT = get_compile_time_arg_val(2);       // 1 => the D12 DEST fold
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr uint32_t ACC_VIA_ADD = get_compile_time_arg_val(4);
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(5);
    constexpr uint32_t SCALER_TILES = get_compile_time_arg_val(6);
    constexpr uint32_t BATCH = get_compile_time_arg_val(7);         // MODE 2: tile-rows per DEST acquire

    const uint32_t rows = get_arg_val<uint32_t>(0);

    // ---- the FROZEN policy knobs, identical in every variant --------------
    constexpr auto REDUCE_POLICY = ckl::ReduceInputPolicy::BulkWaitBulkPop;   // the op's REDUCE_BULK == 1
    constexpr auto REDUCE_ALGO =
        (ACC_VIA_ADD != 0) ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;
    static_assert(
        ACC_VIA_ADD == 0 || REDUCE_POLICY == ckl::ReduceInputPolicy::BulkWaitBulkPop,
        "AccumulateViaAdd is BulkWaitBulkPop-only (reduce_helpers_compute.inl static_assert)");
    // DEST format at the frozen fp32_dest_acc_en == False.  Drives the fused SFPU reduce.
    constexpr auto DST_FMT = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;

    // The reduce input CB per mode: MODE 0 reduces the resident tensor directly,
    // MODE 1 reduces the square's output.
    constexpr uint32_t REDUCE_IN = (MODE == 0) ? cb_in : cb_x_squared;
    // Per-CALL reduce-dim width -- THE quantity this whole bench is about.
    constexpr uint32_t PER_CALL_W = (MODE == 0) ? WIDTH : X_SQ_WT;

    // Non-tile-aligned width: the two datapaths take DIFFERENT partial mechanisms,
    // exactly as rms_norm_compute.cpp spells them.
    const auto PARTIAL_SCALER = (PARTIAL_W == 0)
                                    ? ckl::ReducePartialScaler::none()
                                    : (ACC_VIA_ADD != 0
                                           ? ckl::ReducePartialScaler::partial_mask(PARTIAL_W, /*mask_idx=*/0)
                                           : ckl::ReducePartialScaler::last_tile_at(1));

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // ---- MODE 1's square, spelled exactly as the op spells it -------------
    // D12 fold ON  (X_SQ_WT == 1): DestAccumulation::PerRow, one packed tile per row.
    // D12 fold OFF (X_SQ_WT == WIDTH): a plain per-tile square.
    constexpr bool SQ_FOLD = (X_SQ_WT == 1) && (WIDTH > 1);
    constexpr auto SQ_OUT_FOLDED = ckl::output(
        cb_x_squared,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::Disabled,
        ckl::DestAccumulation::PerRow);
    constexpr auto SQ_OUT = SQ_FOLD ? SQ_OUT_FOLDED : ckl::output(cb_x_squared);
    // x is RESIDENT for the whole bench (one publish, never popped) -- the op's
    // X_RESIDENT pass-A operand.
    constexpr auto X_IN = ckl::input(
        cb_in, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled);

    if constexpr (MODE != 0) {
        // Publish the resident x block ONCE.  Stand-in for the (isolated-away) reader.
        cb_reserve_back(cb_in, rows * WIDTH);
        cb_push_back(cb_in, rows * WIDTH);
        cb_wait_front(cb_in, rows * WIDTH);
    }

    if constexpr (MODE == 2) {
        // ------------------------------------------------------------------
        // OPTION 2: the within-tile 32-column sum rides the square's own DEST
        // window.  No cb_x_squared, no reduce call.
        //
        // Hand-written rather than helper-composed: eltwise_chain has no SFPU
        // stage, so there is no seam to insert the column sum between the last
        // accumulating mul and the pack.  Every primitive used here is a PUBLIC
        // compute API (mul_tiles_init(.., acc_to_dest) / mul_tiles / sfpu_reduce /
        // pack_tile) and the DEST lifecycle is byte-for-byte eltwise_chain's
        // DestAccumulation::PerRow lifecycle (acquire per outer, one pack per
        // outer, release) -- see eltwise_chain.inl's per_row_dest_accumulation
        // branch.
        // ------------------------------------------------------------------
        reconfig_data_format(cb_in, cb_in);
        pack_reconfig_data_format(cb_out);
        // acc_to_dest exactly as eltwise_chain's DestAccumulation::PerRow sets it
        // (eltwise_chain.inl: mul_tiles_init(CbA, CbB, acc_to_dest)); at WIDTH == 1
        // there is nothing to accumulate, which is also when the op turns SQ_FOLD off.
        mul_tiles_init(cb_in, cb_in, /*acc_to_dest=*/(WIDTH > 1) ? 1u : 0u, __builtin_LINE());
        sfpu_reduce_init<PoolType::SUM, DST_FMT>();
        for (uint32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
            cb_reserve_back(cb_out, rows);
            for (uint32_t r0 = 0; r0 < rows; r0 += BATCH) {
                const uint32_t n = (r0 + BATCH <= rows) ? BATCH : (rows - r0);
                MaybeDeviceZoneScope("bench_fused");
                tile_regs_acquire();
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t base = (r0 + j) * WIDTH;
                    for (uint32_t w = 0; w < WIDTH; ++w) {
                        mul_tiles(cb_in, cb_in, base + w, base + w, j);
                    }
                }
                // ONE call folds n independent tile-rows' 32 columns (rt_dim = n).
                sfpu_reduce<PoolType::SUM, DST_FMT, ReduceDim::REDUCE_ROW>(0, /*ct_dim=*/1, /*rt_dim=*/n);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < n; ++j) {
                    pack_tile(j, cb_out, r0 + j);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, rows);
        }
    } else {
        for (uint32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
            if constexpr (MODE == 0) {
                // Stand-in for the (isolated-away) producer: publish the resident
                // x^2 block.  Zero payload, exact CB contract.
                cb_reserve_back(cb_in, rows * WIDTH);
                cb_push_back(cb_in, rows * WIDTH);
            } else {
                MaybeDeviceZoneScope("bench_square");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows, WIDTH),
                    ckl::BinaryFpu<
                        X_IN,
                        X_IN,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::Dst::D0,
                        SQ_OUT.dest_accumulation>{},
                    ckl::PackTile<SQ_OUT>{});
            }
            {
                MaybeDeviceZoneScope("bench_reduce");
                ckl::accumulate_reduce_block<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    REDUCE_IN,
                    cb_scaler,
                    cb_out,
                    REDUCE_POLICY,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    REDUCE_ALGO>(
                    ckl::ReduceInputBlockShape::of(rows, PER_CALL_W), 0, 1, PARTIAL_SCALER);
            }
        }
    }

    cb_pop_front(cb_scaler, SCALER_TILES);
}
"""


# =============================================================================
# Reader — the op's boot scaler, per datapath.  Nothing else.
# =============================================================================
_READER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t ACC_VIA_ADD = get_compile_time_arg_val(0);
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(1);

    // Byte-for-byte rms_norm_reader.cpp's `reader_scaler_boot` branch.
    if constexpr (ACC_VIA_ADD != 0) {
        if constexpr (PARTIAL_W != 0) {
            dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ckernel::ReduceDim::REDUCE_ROW>(PARTIAL_W);
        } else {
            dataflow_kernel_lib::
                prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
        }
    } else if constexpr (PARTIAL_W != 0) {
        dataflow_kernel_lib::prepare_partial_reduce_scalers<
            cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            PARTIAL_W>(1.0f);
    } else {
        dataflow_kernel_lib::
            prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(1.0f);
    }
}
"""


# =============================================================================
# Writer — the stat CB's SINGLE consumer.  Pure CB flow control (no NoC), so it
# costs the same in every variant and cannot mask the delta.  The LAST block is
# left un-popped so the host can read the final stat tiles out of the tensor.
#
# CT args: [NUM_BLOCKS]
# RT args: [rows]
# =============================================================================
_WRITER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = 3;
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(0);
    const uint32_t rows = get_arg_val<uint32_t>(0);

    for (uint32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        cb_wait_front(cb_out, rows);
        if (blk + 1 < NUM_BLOCKS) {
            cb_pop_front(cb_out, rows);
        }
    }
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
    """The op's PINNED `_perf_case` precision contract.  Identical for every variant."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


# Tile-rows folded into one DEST acquire by `fused_batch`.  fp32_dest_acc_en=False +
# half DEST sync => 8 usable DEST tiles on this arch; the batch is capped there.
DEST_TILE_CAP = 8


def create_program_descriptor(x_in, stat_out, *, variant, rows, width, blocks, partial_w=0, fold=True):
    if variant not in VARIANT_SPEC:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if x_in.dtype != ttnn.bfloat16 or x_in.layout != ttnn.TILE_LAYOUT:
        raise ValueError("the input tensor must be bfloat16 TILE_LAYOUT (the op's cb_input_tiles/cb_x_squared format)")
    if stat_out.dtype != ttnn.float32 or stat_out.layout != ttnn.TILE_LAYOUT:
        raise ValueError("the stat tensor must be float32 TILE_LAYOUT (the op's fp32 stat CB)")

    mode, algo_id, batched = VARIANT_SPEC[variant]
    # The D12 fold makes X_SQUARED_WT 1; without it the reduce's per-call width is WT_CHUNK.
    # The fold is illegal at PARTIAL_W != 0 (the op static_asserts it), so partial-W points
    # are necessarily unfolded.
    x_sq_wt = 1 if (fold and partial_w == 0) else width
    if mode == MODE_FUSED and (partial_w != 0 or not fold):
        raise ValueError("the fused option inherits the D12 fold's PARTIAL_W == 0 precondition")
    # SCALER_TILES: the descriptor's single source of truth.  AccumulateViaAdd takes ONE
    # (the 0/1 mask, or an unused 1.0 scaler); ReduceTile takes the [full, partial] pair.
    scaler_tiles = 1 if (algo_id != 0 or partial_w == 0) else 2
    batch = min(rows, DEST_TILE_CAP) if batched else 1

    cores = _single_core()
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, x_in),
        _scratch_cb(CB_SCALER, ttnn.bfloat16, 2),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, stat_out),
    ]
    # cb_x_squared exists ONLY where the kernel references it -- the fused option
    # DELETES it, and giving the variants different L1 pressure would be dishonest.
    if mode == MODE_SQ_REDUCE:
        cbs.append(_scratch_cb(CB_X_SQUARED, ttnn.bfloat16, rows * x_sq_wt))

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader_rt[0][0] = []
    writer_rt[0][0] = [rows]
    compute_rt[0][0] = [rows]

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[algo_id, partial_w],
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[blocks],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[mode, width, x_sq_wt, blocks, algo_id, partial_w, scaler_tiles, batch],
        runtime_args=compute_rt,
        config=compute_config(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def run(x_in, stat_out, *, variant, rows, width, blocks, partial_w=0, fold=True):
    descriptor = create_program_descriptor(
        x_in, stat_out, variant=variant, rows=rows, width=width, blocks=blocks, partial_w=partial_w, fold=fold
    )
    return ttnn.generic_op([x_in, stat_out], descriptor)
