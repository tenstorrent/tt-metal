# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only benchmark: a SUM/mean reduce built as elementwise ACCUMULATE + SFPU FINALIZE,
compared against the standard reduce library, across all three reduce dimensions.

The reduce is split into two stages:
  1. cross-tile ACCUMULATE — sum the N input tiles elementwise into ONE DEST tile with pairwise
     `add_tiles(acc_to_dest)` (parity resolved at the seed: copy_tile one tile when N is odd, add the
     first pair when even — no phantom zero CB). This stage is IDENTICAL for every reduce dimension.
  2. within-tile FINALIZE on the SFPU — collapse that one tile in DEST (`sfpu_reduce`, which reads DEST
     natively) and apply the 1/N mean scaler with an SFPU scalar-multiply. Only this stage is
     dimension-specific:
        row    (reduce width  / REDUCE_ROW): sfpu_reduce<REDUCE_ROW>            -> per-row means in col 0
        col    (reduce height / REDUCE_COL): sfpu_reduce<REDUCE_COL>            -> per-col means in row 0
        scalar (reduce both   / REDUCE_SCALAR): sfpu_reduce<ROW> then <COL>     -> single mean at [0,0]

Variants:
  helper   — the standard reduce library (FPU matmul-with-ones datapath), the thing the fast path would
             replace. AVG pool type so the 1/N scaler is handled by the library per dimension.
  fast     — the accumulate + SFPU-finalize path above.
  dispatch — pick per width: `fast` when N >= DISPATCH_MIN_TILES, else `helper`. This is the "universally
             good" routing a shared reduce helper would use — the fast path wins only once there are
             enough tiles to amortize its finalize over (below that the single matmul-reduce is cheaper).

Everything is sharded in L1 on one Tensix core (no DRAM in the fast path). Correctness is the only
pass/fail; perf (device kernel ns) and accuracy (vs the fp64 mean) are measured and reported.
"""

import struct

import ttnn

TILE = 32

# CB assignment (semantic names).
CB_IN = 0  # input tiles, sharded L1 (resident): N tiles indexed 0..N-1
CB_SCALER = 1  # reduce scaler tile (helper only)
CB_OUT = 16  # output: 1 tile, fp32, tensor-backed

VARIANTS = ("helper", "fast", "dispatch")
BASELINE = "helper"

# reduce dimension -> compile-time id (the finalize selector in the fast kernel + the reduce<dim> in helper).
_DIM_ID = {"row": 0, "col": 1, "scalar": 2}
DIMS = tuple(_DIM_ID)

DTYPES = ("fp32", "bf16")  # accumulation (DEST / SFPU) dtype; input is always bf16

# The fast path amortizes its within-tile finalize over the cross-tile accumulation, so it only wins
# once there are enough tiles — and the crossover is DIMENSION-DEPENDENT (measured): the FPU
# REDUCE_COL datapath is cheaper than REDUCE_ROW, so col needs more tiles before the fast path pulls
# ahead. `dispatch` falls back to the single matmul-reduce below the per-dim threshold. A flat
# threshold would mis-dispatch col/scalar at ~4 tiles (fast still slower there).
_DISPATCH_MIN = {"row": 4, "col": 8, "scalar": 8}


def dispatch_min(dim):
    """Fewest reduced tiles at which the fast path beats the reduce library, per reduce dim (measured)."""
    return _DISPATCH_MIN[dim]


def elements_reduced(dim, num_tiles):
    """How many input elements collapse into one output value (for the 1/N mean scaler)."""
    if dim in ("row", "col"):
        return num_tiles * TILE  # a length-(num_tiles*32) row/column per output
    if dim == "scalar":
        return num_tiles * TILE * TILE  # the whole [32, 32*num_tiles] (or [32*N,32]) block
    raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")


# =============================================================================
# Fast kernel — cross-tile accumulate (shared) + SFPU finalize (dim-specific).
# CT args: [num_tiles, dim, kernel_iters, dst_fp32, scaler_bits]
# =============================================================================
_FAST_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t dim = get_compile_time_arg_val(1);          // 0 row, 1 col, 2 scalar
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t dst_fp32 = get_compile_time_arg_val(3);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);  // float bits of 1/N (mean scaler)

    using ckernel::PoolType;
    using ckernel::ReduceDim;
    constexpr DataFormat dst_fmt = (dst_fp32 != 0) ? DataFormat::Float32 : DataFormat::Float16_b;

    cb_reserve_back(cb_in, num_tiles);
    cb_push_back(cb_in, num_tiles);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // Re-establish the binary hw config each iteration (the previous iter's finalize reprogrammed
        // math/pack). pack -> cb_out (the finished mean is packed straight out; no intermediate CB).
        binary_op_init_common(cb_in, cb_in, cb_out);
        cb_wait_front(cb_in, num_tiles);

        tile_regs_acquire();

        // ---- Stage 1: cross-tile accumulate into DEST[0]. Parity at the seed (no phantom zero). ----
        uint32_t k;
        if constexpr (num_tiles & 1u) {
            copy_tile_init(cb_in);
            copy_tile(cb_in, 0, 0);                 // odd: DEST = cb_in[0]
            k = 1;
        } else {
            add_tiles_init(cb_in, cb_in, false);
            add_tiles(cb_in, cb_in, 0, 1, 0);       // even: DEST = cb_in[0] + cb_in[1]
            k = 2;
        }
        add_tiles_init(cb_in, cb_in, true);
        for (; k < num_tiles; k += 2) {
            add_tiles(cb_in, cb_in, k, k + 1, 0);   // DEST += cb_in[k] + cb_in[k+1]
        }

        // ---- Stage 2: within-tile finalize on the SFPU (reads DEST in place), then the 1/N scaler. ----
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        if constexpr (dim == 0) {
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);   // -> per-row in col 0
        } else if constexpr (dim == 1) {
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(0, 1, 1);   // -> per-col in row 0
        } else {
            // scalar: reduce width then height; the total lands at [0,0].
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(0, 1, 1);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scaler_bits);             // DEST *= 1/N  (mean)

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out, 0);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, 1);
            cb_pop_front(cb_out, 1);
        }
    }
    cb_pop_front(cb_in, num_tiles);
}
"""


# =============================================================================
# Helper kernel — the standard reduce library (FPU), AVG so the 1/N scaler is applied per dim.
# CT args: [num_tiles, dim, kernel_iters]
# =============================================================================
_HELPER_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_out = 16;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t dim = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);

    using namespace compute_kernel_lib;
    using ckernel::PoolType;
    using ckernel::ReduceDim;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_reserve_back(cb_in, num_tiles);
        cb_push_back(cb_in, num_tiles);
        if constexpr (dim == 0) {
            reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, cb_in, cb_scaler, cb_out,
                   ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(1, num_tiles));
        } else if constexpr (dim == 1) {
            reduce<PoolType::AVG, ReduceDim::REDUCE_COL, cb_in, cb_scaler, cb_out,
                   ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(num_tiles, 1));
        } else {
            reduce<PoolType::AVG, ReduceDim::REDUCE_SCALAR, cb_in, cb_scaler, cb_out,
                   ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(1, num_tiles));
        }
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, 1);
            cb_pop_front(cb_out, 1);
        }
    }
}
"""


# =============================================================================
# Scaler dataflow kernel (helper only) — fills the AVG scaler tile for the right dim + reduce factor.
# CT args: [dim, reduce_factor]
# =============================================================================
_SCALER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t dim = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_factor = get_compile_time_arg_val(1);
    using ckernel::PoolType;
    using ckernel::ReduceDim;
    if constexpr (dim == 0) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::AVG, ReduceDim::REDUCE_ROW,
                                                                 reduce_factor>();
    } else if constexpr (dim == 1) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::AVG, ReduceDim::REDUCE_COL,
                                                                 reduce_factor>();
    } else {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::AVG, ReduceDim::REDUCE_SCALAR,
                                                                 reduce_factor>();
    }
}
"""


# =============================================================================
# Host-side layout + program descriptor
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def input_shape(dim, num_tiles):
    """The [H, W] element shape of the input block for a reduce dim + tile count."""
    if dim == "col":
        return (num_tiles * TILE, TILE)  # [N, 1] tiles stacked vertically (reduce height)
    return (TILE, num_tiles * TILE)  # [1, N] tiles (reduce width / whole block for scalar)


def create_sharded_memory_config(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dtype_of(name):
    return ttnn.float32 if name == "fp32" else ttnn.bfloat16


def _scratch_cb(cb_id, data_format, num=1):
    ts = ttnn.tile_size(data_format)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=data_format, page_size=ts)
    return ttnn.CBDescriptor(total_size=ts * num, core_ranges=_single_core(), format_descriptors=[fmt])


def _resolve(variant, dim, num_tiles):
    """dispatch picks the physical path by (dim, width); helper/fast are themselves."""
    if variant == "dispatch":
        return "fast" if num_tiles >= dispatch_min(dim) else "helper"
    return variant


def create_program_descriptor(
    input_tensor, output_tensor, *, variant, dim, num_tiles, accum="fp32", kernel_iters=1, math_fidelity=None
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if dim not in _DIM_ID:
        raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")
    if num_tiles < 1 or kernel_iters < 1:
        raise ValueError("num_tiles and kernel_iters must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")

    path = _resolve(variant, dim, num_tiles)
    dim_id = _DIM_ID[dim]
    fp32_dest = accum == "fp32"
    fidelity = math_fidelity or ttnn.MathFidelity.HiFi4  # default HiFi4; pass to sweep the fidelity axis

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]

    if path == "fast":
        scaler_bits = struct.unpack("<I", struct.pack("<f", 1.0 / elements_reduced(dim, num_tiles)))[0]
        compute = ttnn.KernelDescriptor(
            kernel_source=_FAST_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_single_core(),
            compile_time_args=[num_tiles, dim_id, kernel_iters, int(fp32_dest), scaler_bits],
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest),
        )
        return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)

    # helper path: needs the AVG scaler CB + the dataflow kernel that fills it.
    cbs.append(_scratch_cb(CB_SCALER, _dtype_of(accum)))
    compute = ttnn.KernelDescriptor(
        kernel_source=_HELPER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[num_tiles, dim_id, kernel_iters],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest),
    )
    scaler = ttnn.KernelDescriptor(
        kernel_source=_SCALER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[dim_id, elements_reduced(dim, num_tiles)],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[scaler, compute], semaphores=[], cbs=cbs)


def run_op(input_tensor, *, variant, dim, num_tiles, accum="fp32", kernel_iters=1, math_fidelity=None):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config((TILE, TILE)),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        variant=variant,
        dim=dim,
        num_tiles=num_tiles,
        accum=accum,
        kernel_iters=kernel_iters,
        math_fidelity=math_fidelity,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
