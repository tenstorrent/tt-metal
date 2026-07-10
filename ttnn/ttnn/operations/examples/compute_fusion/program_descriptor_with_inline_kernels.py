# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only benchmark: fusing an eltwise expression / a reduce epilogue into one
pass through DEST versus computing it as separate helper calls that round-trip intermediates
through an L1 circular buffer.

Everything lives in sharded L1 on one Tensix core — there is no DRAM movement, so the measured
delta is pure compute: the cost of the intermediate pack-to-L1 + CB handshake + unpack-from-L1
(and the extra per-op init/reconfig) that fusion removes.

Three scenarios, each an A/B (fused vs unfused) with the same math:

  sfpu_chain    out = exp(sqrt(x) + y)     (SFPU sqrt, SFPU add-in-DEST, SFPU exp)
  fpu_sfpu      out = sqrt(x) * b          (SFPU sqrt, then combine with b)
  reduce_recip  out = 1 / rowsum(x)        (SUM reduce, then reciprocal)

`fpu_sfpu` carries a third variant so the *combine* step can be compared directly:
  - dstreuse : FPU multiply that reuses the DEST result of sqrt as one operand (no second copy)
  - sfpu     : copy b into a second DEST slot, SFPU multiply
  - unfused  : sqrt -> L1, FPU multiply reads it back

Micro-benchmark mode: when built with the `CF_MICROBENCH` define, each phase is wrapped in a
`DeviceZoneScopedN` zone (named CF_*). A compute-kernel zone records on all three TRISCs
(unpack/math/pack), so the per-zone durations show where each phase spends its time and which
engine dominates. The clean perf path leaves the define unset, so the zones compile out entirely.
"""

import ttnn

TILE = 32

# CB assignment (shared across scenarios; unused ones are simply not declared per variant).
CB_IN0 = 0  # x
CB_IN1 = 1  # y (sfpu_chain) / b (fpu_sfpu)
CB_SCALER = 2  # reduce scaler (reduce_recip)
CB_S1 = 3  # scratch intermediate (unfused)
CB_S2 = 4  # scratch intermediate (unfused sfpu_chain only)
CB_OUT = 16

# variant name -> (scenario, method_id). method_id is the `if constexpr` selector in the kernel.
_VARIANT_SPEC = {
    "sfpu_chain.fused": ("sfpu_chain", 0),
    "sfpu_chain.unfused": ("sfpu_chain", 1),
    "fpu_sfpu.dstreuse": ("fpu_sfpu", 0),
    "fpu_sfpu.sfpu": ("fpu_sfpu", 1),
    "fpu_sfpu.unfused": ("fpu_sfpu", 2),
    "reduce_recip.fused": ("reduce_recip", 0),
    "reduce_recip.unfused": ("reduce_recip", 1),
}

VARIANTS = tuple(_VARIANT_SPEC)

SCENARIOS = ("sfpu_chain", "fpu_sfpu", "reduce_recip")

# Phase-zone names each variant emits under CF_MICROBENCH (in kernel-emission order).
PHASE_ZONES = {
    "sfpu_chain.fused": ("CF_FUSED",),
    "sfpu_chain.unfused": ("CF_SQRT", "CF_ADD", "CF_EXP"),
    "fpu_sfpu.dstreuse": ("CF_FUSED",),
    "fpu_sfpu.sfpu": ("CF_FUSED",),
    "fpu_sfpu.unfused": ("CF_SQRT", "CF_MUL"),
    "reduce_recip.fused": ("CF_FUSED",),
    "reduce_recip.unfused": ("CF_REDUCE", "CF_RECIP"),
}

# variants that round-trip through scratch CBs (host must declare them)
_USES_S1 = {"sfpu_chain.unfused", "fpu_sfpu.unfused", "reduce_recip.unfused"}
_USES_S2 = {"sfpu_chain.unfused"}
_NUM_INPUTS = {"sfpu_chain": 2, "fpu_sfpu": 2, "reduce_recip": 1}

# scenarios whose fused chains take the DEST-lane block path
_BLOCK_SCENARIOS = {"sfpu_chain", "fpu_sfpu"}


def variants_for(scenario):
    return tuple(v for v in VARIANTS if _VARIANT_SPEC[v][0] == scenario)


# =============================================================================
# Compute kernels — one source per scenario, method selected by CT arg 3.
# CT args (all kernels): [num_tiles, block_size, kernel_iters, method]
# CF_PHASE(name) is a DeviceZoneScopedN zone under CF_MICROBENCH, else nothing.
# =============================================================================

_ZONE_MACRO = r"""
#ifdef CF_MICROBENCH
#include "tools/profiler/kernel_profiler.hpp"
#define CF_PHASE(name) DeviceZoneScopedN(name)
#else
#define CF_PHASE(name)
#endif
"""

_SFPU_CHAIN_KERNEL = (
    r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
"""
    + _ZONE_MACRO
    + r"""
// out = exp(sqrt(x) + y).  FUSED: one chain, sqrt/add/exp all in DEST, no L1 round trip.
// UNFUSED: sqrt->L1(s1), s1+y->L1(s2), exp(s2)->out  (two intermediate L1 round trips).
void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_y = 1, cb_s1 = 3, cb_s2 = 4, cb_out = 16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t method = get_compile_time_arg_val(3);

    using namespace compute_kernel_lib;

    // Sharded inputs are already resident — mark them available once.
    cb_reserve_back(cb_x, n); cb_push_back(cb_x, n);
    cb_reserve_back(cb_y, n); cb_push_back(cb_y, n);

    compute_kernel_hw_startup(cb_x, cb_y, cb_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (method == 0) {
            // FUSED: sqrt(x) in D0, y in D1, SFPU add -> D0, exp -> D0, pack once.
            CF_PHASE("CF_FUSED");
            eltwise_chain(
                EltwiseShape::tiles(n, blk),
                CopyTile<cb_x, Dst::D0, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                Sqrt<>{},
                CopyTile<cb_y, Dst::D1, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                Exp<>{},
                PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
        } else {
            // UNFUSED: three chains, each packing to / reading from L1.
            { CF_PHASE("CF_SQRT");  // s1 = sqrt(x)
              eltwise_chain(
                EltwiseShape::tiles(n, blk),
                CopyTile<cb_x, Dst::D0, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                Sqrt<>{},
                PackTile<cb_s1, OutputLifecycle::Bulk, PackTileReconfig::Output>{}); }
            { CF_PHASE("CF_ADD");  // s2 = s1 + y
              eltwise_chain(
                EltwiseShape::tiles(n, blk),
                CopyTile<cb_s1, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
                CopyTile<cb_y, Dst::D1, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<cb_s2, OutputLifecycle::Bulk, PackTileReconfig::Output>{}); }
            { CF_PHASE("CF_EXP");  // out = exp(s2)
              eltwise_chain(
                EltwiseShape::tiles(n, blk),
                CopyTile<cb_s2, Dst::D0, InputLifecycle::Bulk, CopyTileReconfig::Input, OperandKind::Block>{},
                Exp<>{},
                PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{}); }
        }
        if (iter + 1 < kernel_iters) { cb_wait_front(cb_out, n); cb_pop_front(cb_out, n); }
    }
    cb_pop_front(cb_x, n);
    cb_pop_front(cb_y, n);
}
"""
)


_FPU_SFPU_KERNEL = (
    r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
"""
    + _ZONE_MACRO
    + r"""
// out = sqrt(x) * b.  Three ways to combine sqrt(x) (already in DEST) with b:
//   method 0 dstreuse : FPU mul reuses DEST(sqrt) as srca, b -> srcb. One DEST slot.
//   method 1 sfpu     : copy b into D1, SFPU mul D0*D1 -> D0. Two DEST slots.
//   method 2 unfused  : sqrt -> L1(s1), FPU mul reads s1 back.  One L1 round trip.
void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_b = 1, cb_s1 = 3, cb_out = 16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t method = get_compile_time_arg_val(3);

    using namespace compute_kernel_lib;

    cb_reserve_back(cb_x, n); cb_push_back(cb_x, n);
    cb_reserve_back(cb_b, n); cb_push_back(cb_b, n);

    compute_kernel_hw_startup(cb_x, cb_b, cb_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (method == 0) {
            CF_PHASE("CF_FUSED");
            eltwise_chain(
                EltwiseShape::tiles(n, blk),
                CopyTile<cb_x, Dst::D0, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                Sqrt<>{},
                DestReuseBinary<cb_b, BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA,
                                InputLifecycle::HeldBulk, DestReuseReconfig::Input, Dst::D0, Dst::D0,
                                OperandKind::Block>{},
                PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
        } else if constexpr (method == 1) {
            CF_PHASE("CF_FUSED");
            eltwise_chain(
                EltwiseShape::tiles(n, blk),
                CopyTile<cb_x, Dst::D0, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                Sqrt<>{},
                CopyTile<cb_b, Dst::D1, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
        } else {
            { CF_PHASE("CF_SQRT");  // s1 = sqrt(x)
              eltwise_chain(
                EltwiseShape::tiles(n, blk),
                CopyTile<cb_x, Dst::D0, InputLifecycle::HeldBulk, CopyTileReconfig::Input, OperandKind::Block>{},
                Sqrt<>{},
                PackTile<cb_s1, OutputLifecycle::Bulk, PackTileReconfig::Output>{}); }
            { CF_PHASE("CF_MUL");  // out = s1 * b (FPU)
              eltwise_chain(
                EltwiseShape::tiles(n, blk),
                BinaryFpu<cb_s1, cb_b, BinaryFpuOp::Mul, BroadcastDim::None,
                          InputLifecycle::Bulk, InputLifecycle::HeldBulk, BinaryDataFormatReconfig::Input,
                          Dst::D0, OperandKind::Block, OperandKind::Block>{},
                PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{}); }
        }
        if (iter + 1 < kernel_iters) { cb_wait_front(cb_out, n); cb_pop_front(cb_out, n); }
    }
    cb_pop_front(cb_x, n);
    cb_pop_front(cb_b, n);
}
"""
)


_REDUCE_RECIP_KERNEL = (
    r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
"""
    + _ZONE_MACRO
    + r"""
// out = 1 / rowsum(x), one row of Wt tiles -> one output tile.
//   method 0 fused   : SUM reduce with a post-reduce reciprocal in DEST (no L1 round trip).
//   method 1 unfused : SUM reduce -> L1(s1), then a reciprocal chain reads s1 back.
void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_scaler = 2, cb_s1 = 3, cb_out = 16;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t method = get_compile_time_arg_val(3);

    using namespace compute_kernel_lib;

    cb_reserve_back(cb_x, Wt); cb_push_back(cb_x, Wt);

    compute_kernel_hw_startup(cb_x, cb_scaler, cb_out);

    constexpr auto shape = ReduceInputBlockShape::of(1, Wt, 1);
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (method == 0) {
            CF_PHASE("CF_FUSED");
            reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_x, cb_scaler, cb_out,
                   ReduceInputPolicy::WaitUpfrontNoPop>(
                shape,
                ReduceInputMemoryLayout::contiguous(),
                NoAccumulation{},
                [](uint32_t dst) { recip_tile_init(); recip_tile(dst); });
        } else {
            { CF_PHASE("CF_REDUCE");
              reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, cb_x, cb_scaler, cb_s1,
                     ReduceInputPolicy::WaitUpfrontNoPop>(shape); }
            { CF_PHASE("CF_RECIP");
              eltwise_chain(
                EltwiseShape::tiles(1),
                CopyTile<cb_s1, Dst::D0, InputLifecycle::Bulk>{},
                Recip<>{},
                PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{}); }
        }
        if (iter + 1 < kernel_iters) { cb_wait_front(cb_out, 1); cb_pop_front(cb_out, 1); }
    }
    cb_pop_front(cb_x, Wt);
}
"""
)


# Dataflow kernel: fill the reduce scaler tile once per launch (SUM -> 1.0). Never popped, so a
# single push covers every kernel_iters reduce in the launch.
_SCALER_KERNEL = r"""
#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 2;
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
}
"""

_KERNEL_SOURCE = {
    "sfpu_chain": _SFPU_CHAIN_KERNEL,
    "fpu_sfpu": _FPU_SFPU_KERNEL,
    "reduce_recip": _REDUCE_RECIP_KERNEL,
}


# =============================================================================
# Host-side sharded-L1 layout + program descriptor
# =============================================================================


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(num_tiles):
    """One row of `num_tiles` tiles, height-sharded onto a single core."""
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    return ttnn.create_sharded_memory_config(
        shape=(TILE, num_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, num_tiles):
    page = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat16, page_size=2048)
    return ttnn.CBDescriptor(
        total_size=2048 * num_tiles,
        core_ranges=_single_core(),
        format_descriptors=[page],
    )


def create_program_descriptor(
    input_tensors, output_tensor, *, variant, num_tiles, block_size=1, kernel_iters=1, microbench=False
):
    if variant not in _VARIANT_SPEC:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    scenario, method = _VARIANT_SPEC[variant]
    if num_tiles < 1 or kernel_iters < 1 or block_size < 1:
        raise ValueError("num_tiles, kernel_iters, block_size must be positive")
    if len(input_tensors) != _NUM_INPUTS[scenario]:
        raise ValueError(f"{scenario} needs {_NUM_INPUTS[scenario]} input tensor(s)")
    for t in (*input_tensors, output_tensor):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError("compute_fusion uses bfloat16 TILE_LAYOUT tensors")

    blk = block_size if scenario in _BLOCK_SCENARIOS else 1
    compile_time_args = [num_tiles, blk, kernel_iters, method]
    defines = [("CF_MICROBENCH", "1")] if microbench else []

    kernels = []
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL_SOURCE[scenario],
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=compile_time_args,
        defines=defines,
        config=ttnn.ComputeConfigDescriptor(),
    )

    # CBs: tensor-backed inputs + output, plus any scratch / scaler this variant needs.
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_IN0, input_tensors[0])]
    if len(input_tensors) > 1:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_IN1, input_tensors[1]))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor))

    if scenario == "reduce_recip":
        cbs.append(_scratch_cb(CB_SCALER, 2))
        scaler = ttnn.KernelDescriptor(
            kernel_source=_SCALER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_single_core(),
            compile_time_args=[],
            runtime_args=[],
            config=ttnn.ReaderConfigDescriptor(),
        )
        kernels.append(scaler)
    if variant in _USES_S1:
        cbs.append(_scratch_cb(CB_S1, 1 if scenario == "reduce_recip" else num_tiles))
    if variant in _USES_S2:
        cbs.append(_scratch_cb(CB_S2, num_tiles))

    kernels.append(compute)
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def run_fusion(input_tensors, *, variant, num_tiles, block_size=1, kernel_iters=1, microbench=False):
    """Allocate the sharded output and run one variant. Output shape = input row width, except
    reduce_recip which collapses the row to a single tile."""
    scenario, _ = _VARIANT_SPEC[variant]
    out_tiles = 1 if scenario == "reduce_recip" else num_tiles
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, out_tiles * TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensors[0].device(),
        create_sharded_memory_config(out_tiles),
    )
    descriptor = create_program_descriptor(
        input_tensors,
        output,
        variant=variant,
        num_tiles=num_tiles,
        block_size=block_size,
        kernel_iters=kernel_iters,
        microbench=microbench,
    )
    return ttnn.generic_op([*input_tensors, output], descriptor)
