# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-core, compute-only reconstruction of groupnorm_sc_N_1_HW_C's pass-1 chunk work (`colsum_chunk`).

Per (rows x cols) tile chunk of x (bf16, resident in L1 -- the reads are not what is measured) the op produces the
column sums S_j = sum_r x_rj and Q_j = sum_r (x_rj)^2 in cb_colsum = [S_0..S_{c-1} ; Q_0..Q_{c-1}] (Float32,
row-0-valid tiles), accumulated across the row chunks rc with `Accumulate::at(cb_colsum, rc)`.

Variants (compile-time `method`):
  0 baseline      the op's current three helper calls: ckl::square (FPU Mul(x,x)) -> cb_xsq (Float16_b pages),
                  REDUCE_COL SUM over x -> S, REDUCE_COL SUM over cb_xsq -> Q.
  1 fused_copy    ONE eltwise_chain per chunk: CopyTile x -> D0, pack D0 to cb_xx column j (stride 2c);
                  BinaryFpu Mul(x,x) -> D1, pack D1 to cb_xx column c+j. Then ONE REDUCE_COL over the
                  (rows x 2c) block of cb_xx -> exactly [S..; Q..].
  2 fused_mulone  as 1 but the x copy is BinaryFpu Mul(x, ones) -> D0 (uniform math MOP with the square, no per-tile
                  copy<->mul init switch; costs one extra srcB unpack per tile). Exact for bf16 x (fp32 DEST).
  3 interleaved_ub COPY-FREE UPPER BOUND, not integrable as-is: x is pre-laid by the host at columns 0..c-1 of a
                  2c-wide block and the square is packed straight into columns c..2c-1 of the SAME CB, then one
                  REDUCE_COL over (rows x 2c). In the op this needs the reader and the compute to both write one
                  CB (single-producer invariant) -- measured only to bound how much of fused_copy's cost is the copy.
  4 dest_acc_xsq  option (b): Q via DEST accumulation. Per column j one chain accumulates Mul(x_rj, x_rj) over the
                  chunk's rows in DEST (fp32) and packs ONE full tile sum_r x_rj^2 to cb_qf (Float32); the S reduce
                  is the baseline's; the Q reduce then runs over a (1 x cols) block of cb_qf. NOT bit-identical to
                  the baseline (x^2 is not rounded to Float16_b per tile; row order differs). The chunk holding the
                  image's partial last tile-row (hw_tail != 0) keeps the baseline path at runtime: the partial scaler
                  cannot mask rows that were already summed in DEST.
  5 dest_acc_xsq_sfpu  as 4, but the (1 x cols) Q reduce over the Float32 qf tiles uses ReduceAlgorithm::AccumulateViaAdd
                  (copy to fp32 DEST + SFPU column reduce) instead of the FPU reduce_tile, whose SrcA path truncates the
                  Float32 partial sums.

Precision contract, FIXED for every variant: fp32_dest_acc_en=True, HiFi4, math_approx_mode=False,
dst_full_sync_en=True; x bf16; xsq pages Float16_b; colsum Float32; SUM scaler from
dataflow_kernel_lib::calculate_and_prepare_reduce_scaler (+ the partial pair when hw_tail != 0).
"""

import ttnn

TILE = 32

CB_X = 0
CB_SCALER = 1
CB_XSQ = 2
CB_XX = 3
CB_ONES = 4
CB_QF = 5  # dest_acc_xsq: one full tile sum_r x_rj^2 per column (Float32)
CB_COLSUM = 16

METHODS = {
    "baseline": 0,
    "fused_copy": 1,
    "fused_mulone": 2,
    "interleaved_ub": 3,
    "dest_acc_xsq": 4,
    "dest_acc_xsq_sfpu": 5,
}
EXACT_VARIANTS = (
    "baseline",
    "fused_copy",
    "fused_mulone",
    "interleaved_ub",
)  # bit-identical to baseline by construction
VARIANTS = tuple(METHODS)

FOCUS = dict(rows=2, cols=2, num_rc=2)  # the op's chunk geometry on the focus shape (1,1,1024,640) G=32


def x_tile_width(variant, cols):
    """Tiles per tile-row of the x tensor handed to the kernel (interleaved_ub carries the xsq half)."""
    return 2 * cols if variant == "interleaved_ub" else cols


_READER_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// The op's reader prepares the REDUCE_COL SUM scaler once per kernel: a [full, partial] pair when the image's
// HW is not tile-aligned (hw_tail = HW % 32 valid rows of the last tile-row), one full tile otherwise.
void kernel_main() {
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(0);
    constexpr uint32_t hw_tail = get_compile_time_arg_val(1);
    if constexpr (hw_tail != 0) {
        dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<
            cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL, hw_tail>();
    } else {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();
    }
}
"""

_COMPUTE_KERNEL = r"""
#include <stdint.h>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/generators/fill.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace ckl = compute_kernel_lib;

// Pass-1 chunk work of groupnorm_sc_N_1_HW_C (compute kernel, lambda colsum_chunk), one variant per `method`.
// Every variant leaves cb_colsum = [S_0..S_{c-1} ; Q_0..Q_{c-1}] after the last row chunk (Accumulate reload).
void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_xsq = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xx = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(4);
    constexpr uint32_t cb_colsum = get_compile_time_arg_val(5);
    constexpr uint32_t method = get_compile_time_arg_val(6);
    constexpr bool has_partial = get_compile_time_arg_val(7) != 0;  // hw_tail != 0: last row tile of the last rc
    constexpr uint32_t cb_qf = get_compile_time_arg_val(8);
    using ckl::DestAccumulation;

    const uint32_t rows = get_arg_val<uint32_t>(0);  // chunk_rows
    const uint32_t cols = get_arg_val<uint32_t>(1);  // cols_per_group
    const uint32_t num_rc = get_arg_val<uint32_t>(2);
    const uint32_t chunk = rows * cols;

    using ckl::BinaryFpuOp;
    using ckl::DataFormatReconfig;
    using ckl::Dst;
    using ckl::input;
    using ckl::InputTileMapping;
    using ckl::IterationShape;
    using ckl::output;
    using ckl::PopPolicy;
    using ckl::PushPolicy;
    using ckl::ReservePolicy;
    using ckl::StridedTileRange;
    using ckl::TileAddressing;
    using ckl::WaitPolicy;

    compute_kernel_hw_startup(cb_x, cb_scaler, cb_colsum);

    if constexpr (method != 3) {
        // x is a resident shard: expose it as `num_rc` chunk-quanta the way the op's cb_x_pass1 ring does.
        cb_reserve_back(cb_x, chunk * num_rc);
        cb_push_back(cb_x, chunk * num_rc);
    }
    if constexpr (method == 2) {
        // ones tile for the exact x copy Mul(x, 1.0)
        ckl::eltwise_chain(
            IterationShape::tiles(1),
            ckl::FillScalar<Dst::D0>{1.0f},
            ckl::PackTile<output(cb_ones, ReservePolicy::Upfront, PushPolicy::AtEnd)>{});
    }

    // ---------------- baseline: the op's current colsum_chunk (also the hw_tail fallback of methods 4/5) ----------------
    auto baseline_chunk = [&](uint32_t rc, ckl::ReducePartialScaler partial_scaler) {
        {
            cb_wait_front(cb_x, chunk);
            {
                MaybeDeviceZoneScope("c_square");
                cb_reserve_back(cb_xsq, chunk);
                ckl::square<
                    input(cb_x, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block),
                    output(cb_xsq, ReservePolicy::None, PushPolicy::None)>(IterationShape::tiles(chunk));
                cb_push_back(cb_xsq, chunk);
            }
            {
                MaybeDeviceZoneScope("c_reduce_x");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_COL,
                    cb_x,
                    cb_scaler,
                    cb_colsum,
                    ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                    ckl::ReduceInputBlockShape::of(rows, cols),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_colsum, rc),
                    ckl::NoOp{},
                    partial_scaler);
                cb_pop_front(cb_x, chunk);
            }
            {
                MaybeDeviceZoneScope("c_reduce_xsq");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_COL,
                    cb_xsq,
                    cb_scaler,
                    cb_colsum,
                    ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                    ckl::ReduceInputBlockShape::of(rows, cols),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_colsum, rc),
                    ckl::NoOp{},
                    partial_scaler);
                cb_pop_front(cb_xsq, chunk);
            }
        }
    };

    for (uint32_t rc = 0; rc < num_rc; ++rc) {
        const bool partial_last_row = has_partial && (rc + 1 == num_rc);
        const auto partial_scaler =
            partial_last_row ? ckl::ReducePartialScaler::with_partial() : ckl::ReducePartialScaler::none();

        if constexpr (method == 0) {
            baseline_chunk(rc, partial_scaler);
        } else if constexpr (method == 1 || method == 2) {
            // ---------------- fused: one chain builds [x | x^2], one REDUCE_COL over 2c columns ----------------
            cb_wait_front(cb_x, chunk);
            {
                MaybeDeviceZoneScope("c_chain_xx");
                cb_reserve_back(cb_xx, 2 * chunk);
                // Caller-managed x window (the cb_wait_front above): two chain ELEMENTS may not both stage
                // (WaitPolicy::Upfront) the same CB front (chain rule `chain_has_duplicate_upfront_cbs_v`), so
                // the copy and the square both read the already-waited block with (None, None) + Block mapping.
                constexpr auto x_in = input(cb_x, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
                constexpr auto xx_out = output(
                    cb_xx, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Strided);
                if constexpr (method == 1) {
                    ckl::eltwise_chain(
                        IterationShape::grid(rows, cols),
                        ckl::CopyTile<x_in, Dst::D0>{},
                        ckl::PackTile<xx_out, Dst::D0>{StridedTileRange{0, 2 * cols}},
                        ckl::BinaryFpu<BinaryFpuOp::Mul, x_in, x_in, Dst::D1>{},
                        ckl::PackTile<xx_out, Dst::D1>{StridedTileRange{cols, 2 * cols}});
                } else {
                    constexpr auto ones_in =
                        input(cb_ones, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Scalar);
                    ckl::eltwise_chain(
                        IterationShape::grid(rows, cols),
                        ckl::BinaryFpu<BinaryFpuOp::Mul, x_in, ones_in, Dst::D0>{},
                        ckl::PackTile<xx_out, Dst::D0>{StridedTileRange{0, 2 * cols}},
                        ckl::BinaryFpu<BinaryFpuOp::Mul, x_in, x_in, Dst::D1>{},
                        ckl::PackTile<xx_out, Dst::D1>{StridedTileRange{cols, 2 * cols}});
                }
                cb_push_back(cb_xx, 2 * chunk);
                cb_pop_front(cb_x, chunk);
            }
            {
                MaybeDeviceZoneScope("c_reduce_xx");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_COL,
                    cb_xx,
                    cb_scaler,
                    cb_colsum,
                    ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                    ckl::ReduceInputBlockShape::of(rows, 2 * cols),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_colsum, rc),
                    ckl::NoOp{},
                    partial_scaler);
                cb_pop_front(cb_xx, 2 * chunk);
            }
        } else if constexpr (method == 4 || method == 5) {
            // ---------------- dest_acc_xsq: Q from DEST-accumulated x*x full tiles, S as the baseline ----------------
            if (partial_last_row) {
                // rows summed in DEST cannot be masked by the partial scaler afterwards: this one chunk keeps the
                // op's current path (runtime branch on the same flag the op already computes).
                baseline_chunk(rc, partial_scaler);
                continue;
            }
            cb_wait_front(cb_x, chunk);
            {
                MaybeDeviceZoneScope("c_xsq_dest_acc");
                cb_reserve_back(cb_qf, cols);
                constexpr auto x_in = input(
                    cb_x, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled,
                    TileAddressing::Strided);
                constexpr auto qf_out = output(
                    cb_qf, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Offset,
                    DestAccumulation::WholeShape);
                for (uint32_t j = 0; j < cols; ++j) {
                    ckl::eltwise_chain(
                        IterationShape::grid(rows, 1),
                        ckl::BinaryFpu<BinaryFpuOp::Mul, x_in, x_in, Dst::D0, DestAccumulation::WholeShape>{
                            StridedTileRange{j, cols}, StridedTileRange{j, cols}},
                        ckl::PackTile<qf_out, Dst::D0>{j});
                }
                cb_push_back(cb_qf, cols);
            }
            {
                MaybeDeviceZoneScope("c_reduce_x");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_COL,
                    cb_x,
                    cb_scaler,
                    cb_colsum,
                    ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                    ckl::ReduceInputBlockShape::of(rows, cols),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_colsum, rc),
                    ckl::NoOp{},
                    partial_scaler);
                cb_pop_front(cb_x, chunk);
            }
            {
                MaybeDeviceZoneScope("c_reduce_qf");
                // AccumulateViaAdd keeps the RAW (unreduced) partial-sum tile in the accumulator CB between chunks
                // and finalizes on at_last(); the hw_tail fallback chunk (ReduceTile, expects REDUCED accumulator
                // tiles) is incompatible with that, so with a partial last row method 5 degrades to method 4's FPU
                // Q reduce. (A native AccumulateViaAdd partial would need the 0/1 mask tile the helper documents.)
                constexpr auto q_algo = (method == 5 && !has_partial) ? ckl::ReduceAlgorithm::AccumulateViaAdd
                                                                       : ckl::ReduceAlgorithm::ReduceTile;
                const auto q_acc = (q_algo == ckl::ReduceAlgorithm::AccumulateViaAdd && rc + 1 == num_rc)
                                       ? ckl::Accumulate::at_last(cb_colsum, rc)
                                       : ckl::Accumulate::at(cb_colsum, rc);
                // AccumulateViaAdd + cross-call Accumulate is BulkWaitBulkPop-only (helper static_assert): the
                // helper then waits the whole (1 x cols) block and pops it itself.
                constexpr auto q_policy = (q_algo == ckl::ReduceAlgorithm::AccumulateViaAdd)
                                              ? ckl::ReduceInputPolicy::BulkWaitBulkPop
                                              : ckl::ReduceInputPolicy::WaitUpfrontNoPop;
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_COL,
                    cb_qf,
                    cb_scaler,
                    cb_colsum,
                    q_policy,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,  // global-namespace enum (reduce_helpers_common.hpp)
                    q_algo>(
                    ckl::ReduceInputBlockShape::of(1, cols),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    q_acc,
                    ckl::NoOp{},
                    ckl::ReducePartialScaler::none());  // rows already summed (the hw_tail chunk took the fallback)
                if constexpr (q_policy == ckl::ReduceInputPolicy::WaitUpfrontNoPop) {
                    cb_pop_front(cb_qf, cols);
                }
            }
        } else {
            // ---------------- interleaved_ub: x already sits at columns 0..c-1 of a 2c-wide block ----------------
            // The chunk's window is reserved (write side) before it is pushed/waited (read side): rd and wr
            // pointers advance in lockstep so both address the same 2*chunk pages the host pre-filled.
            {
                MaybeDeviceZoneScope("c_square_strided");
                cb_reserve_back(cb_x, 2 * chunk);
                constexpr auto x_in = input(
                    cb_x, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled,
                    TileAddressing::Strided);
                constexpr auto xx_out = output(
                    cb_x, ReservePolicy::None, PushPolicy::None, DataFormatReconfig::Enabled, TileAddressing::Strided);
                ckl::eltwise_chain(
                    IterationShape::grid(rows, cols),
                    ckl::BinaryFpu<BinaryFpuOp::Mul, x_in, x_in, Dst::D0>{
                        StridedTileRange{0, 2 * cols}, StridedTileRange{0, 2 * cols}},
                    ckl::PackTile<xx_out, Dst::D0>{StridedTileRange{cols, 2 * cols}});
                cb_push_back(cb_x, 2 * chunk);
            }
            {
                MaybeDeviceZoneScope("c_reduce_xx");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_COL,
                    cb_x,
                    cb_scaler,
                    cb_colsum,
                    ckl::ReduceInputPolicy::WaitUpfrontNoPop>(
                    ckl::ReduceInputBlockShape::of(rows, 2 * cols),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_colsum, rc),
                    ckl::NoOp{},
                    partial_scaler);
                cb_pop_front(cb_x, 2 * chunk);
            }
        }
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_config(tile_rows, tile_cols):
    """A (tile_rows x tile_cols)-tile block resident on core (0,0), row-major tile order."""
    return ttnn.create_sharded_memory_config(
        shape=(tile_rows * TILE, tile_cols * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, num_pages, dtype):
    page = 4096 if dtype == ttnn.float32 else 2048
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=page * num_pages, core_ranges=_single_core(), format_descriptors=[fmt])


def compute_config():
    """The op's precision contract -- identical for every variant, never tuned."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,
    )


def create_program_descriptor(x_tensor, colsum_tensor, *, variant, rows, cols, num_rc, hw_tail=0, zones=False):
    method = METHODS[variant]
    chunk = rows * cols
    scaler_pages = 2 if hw_tail else 1

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_COLSUM, colsum_tensor),
        _scratch_cb(CB_SCALER, scaler_pages, ttnn.bfloat16),
    ]
    if method in (0, 4, 5):
        cbs.append(_scratch_cb(CB_XSQ, chunk, ttnn.bfloat16))  # Float16_b xsq pages (XSQ_16BIT_FOR_16BIT_INPUT)
    if method in (1, 2):
        cbs.append(_scratch_cb(CB_XX, 2 * chunk, ttnn.bfloat16))  # replaces cb_xsq: [x | xsq] per tile-row
    if method == 2:
        cbs.append(_scratch_cb(CB_ONES, 1, ttnn.bfloat16))
    if method in (4, 5):
        cbs.append(_scratch_cb(CB_QF, cols, ttnn.float32))

    defines = [] if zones else [("KERNEL_LIB_PERF_ZONES_OFF", "1")]
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = []
    compute_rt = ttnn.RuntimeArgs()
    compute_rt[0][0] = [rows, cols, num_rc]
    reader = ttnn.KernelDescriptor(
        kernel_source=_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[CB_SCALER, hw_tail],
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[CB_X, CB_SCALER, CB_XSQ, CB_XX, CB_ONES, CB_COLSUM, method, 1 if hw_tail else 0, CB_QF],
        runtime_args=compute_rt,
        defines=defines,
        config=compute_config(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)


def run_colsum(x_tensor, *, variant, rows, cols, num_rc, hw_tail=0, zones=False):
    """x_tensor: bf16 TILE tensor (rows*num_rc tile-rows x x_tile_width(variant, cols) tile-cols) sharded on core
    (0,0). Returns the Float32 [S_0..S_{c-1}, Q_0..Q_{c-1}] tile row (32 x 2c*32), column sums in row 0."""
    colsum = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, 2 * cols * TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        x_tensor.device(),
        sharded_config(1, 2 * cols),
    )
    desc = create_program_descriptor(
        x_tensor, colsum, variant=variant, rows=rows, cols=cols, num_rc=num_rc, hw_tail=hw_tail, zones=zones
    )
    return ttnn.generic_op([x_tensor, colsum], desc)
