# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED micro-benchmark: pass A's TAIL on the cross-core combine path.

Concept under test (ONE idea, nothing else):
    On the width-combine path a core's pass A reduces cb_x_squared into an fp32
    stat tile and then hands that tile to the WRITER, which ships it to the group
    root.  Two ways to spell that hand-off:

      baseline   reduce -> cb_row_stat  (the compute-private accumulator),
                 then a PURE fp32 tile copy  cb_row_stat -> cb_sum_handoff.
                 This is rms_norm_compute.cpp's `compute_partial_handoff` stage.

      candidate  reduce -> cb_sum_handoff DIRECTLY.  No copy at all.
                 Legal because the combine path is statically ONE width chunk
                 (rms_norm_writer.cpp:  static_assert(!CROSS_CORE ||
                 NUM_W_CHUNKS == 1)), so accumulate_reduce_block's cross-chunk
                 reload -- the only reason its output CB has to be a re-readable
                 accumulator -- never runs.

Everything else is held constant / trivial (per /perf-lab's concept isolation):
  * the SQUARE stage is not in the bench at all -- cb_x_squared is a ZERO-COPY
    L1-sharded tensor the host fills with x^2-shaped data, so both variants see
    byte-identical reduce input with zero producer cost.
  * the WRITER is a pure CB consumer (wait/pop, no NoC).  It is present because
    it is the second half of the CB-ownership question: with the candidate,
    cb_sum_handoff has compute as its single producer and the writer as its
    single consumer.
  * the reader only fills cb_scaler, exactly as rms_norm_reader.cpp does at boot.

Precision contract (FIXED, identical in both variants -- never a lever):
  bf16 cb_x_squared, fp32 stat CBs, TILE layout, math_fidelity=HiFi2,
  fp32_dest_acc_en=False, math_approx_mode=False.

Geometry knobs mirror the op's:
  rows   == BLOCK_ROWS      (tile-rows per row-block)
  width  == X_SQUARED_WT    (the reduce's PER-CALL reduce-dim width; the op's
                             Refinement-4 L6d DEST fold sets this to 1)
  blocks == number of row-blocks the core walks
  algo   == the reduce datapath the op picked (AccumulateViaAdd / ReduceTile)
"""

import ttnn

TILE = 32

# CB assignment.  0 and 3 are tensor-backed (zero-copy sharded), 1/2 are scratch.
CB_X_SQUARED = 0  # bf16, rows*width pages, ZERO-COPY on the input tensor
CB_SCALER = 1  # bf16, 1 page (the op's aligned-W boot scaler)
CB_ROW_STAT = 2  # fp32, CB_ROW_STAT_DEPTH * rows pages (baseline only path)
CB_SUM_HANDOFF = 3  # fp32, rows pages, ZERO-COPY on the output tensor

# The op's cb_row_stat ring depth (rms_norm_program_descriptor.CB_ROW_STAT_DEPTH).
CB_ROW_STAT_DEPTH = 2

VARIANTS = ("baseline", "candidate")
BASELINE = "baseline"

_ALGO_ID = {"acc_add": 1, "reduce_tile": 0}
ALGOS = tuple(_ALGO_ID)


# =============================================================================
# Compute kernel — pass A's tail, both variants in ONE source (CT-arg selected).
#
# CT args: [VARIANT, WIDTH, NUM_BLOCKS, ACC_VIA_ADD]
# RT args: [rows]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_x_squared = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_row_stat = 2;
constexpr uint32_t cb_sum_handoff = 3;
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);  // 0 baseline, 1 candidate
    constexpr uint32_t WIDTH = get_compile_time_arg_val(1);    // X_SQUARED_WT
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(2);
    constexpr uint32_t ACC_VIA_ADD = get_compile_time_arg_val(3);

    const uint32_t rows = get_arg_val<uint32_t>(0);

    // THE IDEA, in one line: where the reduce packs its result.
    //   baseline  -> cb_row_stat, then a separate fp32 tile copy to the handoff
    //   candidate -> the handoff CB itself
    constexpr uint32_t REDUCE_OUT = (VARIANT == 0) ? cb_row_stat : cb_sum_handoff;

    // Identical boot in BOTH variants (same srcA, same scaler, same packer CB), so
    // the measured delta cannot come from a different hw_configure.
    compute_kernel_hw_startup(cb_x_squared, cb_scaler, cb_sum_handoff);

    constexpr auto REDUCE_POLICY = ckl::ReduceInputPolicy::BulkWaitBulkPop;  // the op's REDUCE_BULK == 1
    constexpr auto REDUCE_ALGO =
        (ACC_VIA_ADD != 0) ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;

    for (uint32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        // Stand-in for the (isolated-away) square stage: publish the resident
        // x^2 block.  Zero payload, exact CB contract.
        cb_reserve_back(cb_x_squared, rows * WIDTH);
        cb_push_back(cb_x_squared, rows * WIDTH);

        {
            MaybeDeviceZoneScope("bench_reduce");
            ckl::accumulate_reduce_block<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_x_squared,
                cb_scaler,
                REDUCE_OUT,
                REDUCE_POLICY,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                REDUCE_ALGO>(ckl::ReduceInputBlockShape::of(rows, WIDTH), 0, 1, ckl::ReducePartialScaler::none());
        }

        if constexpr (VARIANT == 0) {
            MaybeDeviceZoneScope("bench_handoff");
            ckl::copy<ckl::input(cb_row_stat), ckl::output(cb_sum_handoff)>(ckl::EltwiseShape::tiles(rows));
        }
    }

    cb_pop_front(cb_scaler, 1);  // the op's SCALER_TILES == 1 on an aligned W
}
"""


# =============================================================================
# Reader — the op's boot scaler, nothing else.
# =============================================================================
_READER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        1.0f);
}
"""


# =============================================================================
# Writer — the handoff CB's SINGLE consumer.  Pure CB flow control (no NoC), so
# it costs the same in both variants and cannot mask the delta.  The LAST block
# is left un-popped so the host can read the final stat tiles out of the tensor.
#
# CT args: [NUM_BLOCKS]
# RT args: [rows]
# =============================================================================
_WRITER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_sum_handoff = 3;
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(0);
    const uint32_t rows = get_arg_val<uint32_t>(0);

    for (uint32_t blk = 0; blk < NUM_BLOCKS; ++blk) {
        cb_wait_front(cb_sum_handoff, rows);
        if (blk + 1 < NUM_BLOCKS) {
            cb_pop_front(cb_sum_handoff, rows);
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


def create_program_descriptor(x_squared, stat_out, *, variant, rows, width, blocks, algo):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if algo not in _ALGO_ID:
        raise ValueError(f"algo must be one of {ALGOS}, got {algo!r}")
    if x_squared.dtype != ttnn.bfloat16 or x_squared.layout != ttnn.TILE_LAYOUT:
        raise ValueError("cb_x_squared tensor must be bfloat16 TILE_LAYOUT (the op's cb_x_squared format)")
    if stat_out.dtype != ttnn.float32 or stat_out.layout != ttnn.TILE_LAYOUT:
        raise ValueError("the stat tensor must be float32 TILE_LAYOUT (the op's fp32 handoff CB)")

    variant_id = VARIANTS.index(variant)
    cores = _single_core()

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X_SQUARED, x_squared),
        _scratch_cb(CB_SCALER, ttnn.bfloat16, 1),
        ttnn.cb_descriptor_from_sharded_tensor(CB_SUM_HANDOFF, stat_out),
    ]
    # cb_row_stat exists ONLY on the baseline path.  Its disappearance from a
    # non-root combine core is a (reported, not measured) L1 side-benefit of the
    # candidate; the bench allocates it only where the kernel references it so the
    # two variants are not silently given different L1 pressure.
    if variant == "baseline":
        cbs.append(_scratch_cb(CB_ROW_STAT, ttnn.float32, CB_ROW_STAT_DEPTH * rows))

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader_rt[0][0] = []
    writer_rt[0][0] = [rows]
    compute_rt[0][0] = [rows]

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
        compile_time_args=[blocks],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[variant_id, width, blocks, _ALGO_ID[algo]],
        runtime_args=compute_rt,
        config=compute_config(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def run(x_squared, stat_out, *, variant, rows, width, blocks, algo="acc_add"):
    descriptor = create_program_descriptor(
        x_squared, stat_out, variant=variant, rows=rows, width=width, blocks=blocks, algo=algo
    )
    return ttnn.generic_op([x_squared, stat_out], descriptor)
