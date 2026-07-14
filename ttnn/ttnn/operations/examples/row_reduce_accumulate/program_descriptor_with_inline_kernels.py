# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only benchmark: four ways to take the MEAN across a row of tiles.

The op reduces one row of `width_tiles` tiles (a [32, 32*width_tiles] region, one tile tall) down
to a single output tile whose column 0 holds the per-row mean over the whole width — a plain
`REDUCE_ROW` mean. Every variant computes the identical result; they differ only in HOW the
cross-tile accumulation (summing the width_tiles tiles together) is carried out:

  reduce_fold        one reduce call folds the cross-tile sum AND the within-tile column-sum
                     together, accumulating the running row-sum in DEST across the width.
  l1_accum           the packer's L1 accumulator sums the width_tiles tiles into ONE L1 tile
                     (copy each to DEST, pack-accumulate to the same L1 slot); then ONE reduce
                     collapses that single tile to the mean.
  dest_accum         `add_tiles(acc_to_dest)` sums each input tile into DEST against a zero tile
                     (one input tile per add); pack the sum to L1; then ONE reduce -> mean.
  dest_accum_pairs   same DEST accumulation but two input tiles per add (DEST += cb[k] + cb[k+1]),
                     resolving odd/even parity at the seed (copy_tile one tile when odd, add-pair
                     when even) so NO phantom zero tile is needed; pack; then ONE reduce -> mean.

So `reduce_fold` does the cross-tile accumulation inside the reduce datapath; the other three do it
in a separate accumulation pass (packer L1 adder, or the FPU add-into-DEST datapath) and only use
the reduce library for the final within-tile collapse.

Knobs, held constant within a run:
  * width_tiles  — the accumulation length (1..N tiles). The sweep runs 1,2,4,8,16,32 tiles
                   (= 32..1024 elements wide).
  * precision    — two independent fidelity axes as "<input>-<accum>": the INPUT tensor dtype and the
                   ACCUMULATION (DEST / intermediate-CB) dtype. Three configs: fp32-fp32 (both
                   precise), bf16-fp32 (lossy input, accurate accumulation — the input-quantization
                   floor), bf16-bf16 (+ accumulation loss). fp32-bf16 is omitted. The OUTPUT CB is
                   always fp32, so a measured accuracy gap reflects input + accumulation precision,
                   not final-output rounding. fp32-fp32 vs bf16-fp32 isolates the input effect;
                   bf16-fp32 vs bf16-bf16 isolates the accumulation effect.

Everything lives in sharded L1 on one Tensix core — there is no DRAM movement, so nothing but the
on-core compute pipeline is timed. Correctness is the only pass/fail; perf and accuracy are
measured and reported, never asserted.
"""

import struct

import ttnn

TILE = 32
TILE_W = 32  # elements along the reduce (width) dimension per tile

# CB assignment (semantic names, fixed indices).
CB_IN = 0  # input row: width_tiles tiles, tensor-backed (format = input dtype)
CB_SCALER = 1  # reduce scaler tile (1/(width_tiles*32)); format tracks the reduce input
CB_ZERO = 2  # a tile of zeros (second add operand for the 1-tile-per-add dest_accum); format = input dtype
CB_INTERM = 3  # single accumulator tile fed to the finalize reduce; format = accumulation dtype
CB_OUT = 16  # output: 1 tile, fp32, tensor-backed

# variant name -> compile-time method id (the `if constexpr` selector in the kernel). Methods 4/5 are
# the SFPU-finalize twins of dest_accum / dest_accum_pairs: identical DEST accumulation, but the final
# within-tile collapse runs on the SFPU (in DEST, no L1 round-trip) instead of the FPU reduce library.
_METHOD_ID = {
    "reduce_fold": 0,
    "l1_accum": 1,
    "dest_accum": 2,
    "dest_accum_pairs": 3,
    "dest_accum_sfpu": 4,
    "dest_accum_pairs_sfpu": 5,
}
METHODS = tuple(_METHOD_ID)
BASELINE = "reduce_fold"

# Per-method CB needs. SFPU-finalize methods pack the finished mean straight to cb_out (no interm) and
# scale via an SFPU scalar-multiply (no reduce scaler CB); the FPU-reduce methods need the scaler CB.
# Only the 1-tile-per-add methods need the zero tile — the pairs methods resolve odd/even parity at
# the seed (copy_tile for odd, add-pair for even), so they need no phantom zero CB.
_SFPU_FINALIZE = {"dest_accum_sfpu", "dest_accum_pairs_sfpu"}
_NEEDS_ZERO = {"dest_accum", "dest_accum_sfpu"}
_NEEDS_INTERM = {"l1_accum", "dest_accum", "dest_accum_pairs"}
_NEEDS_SCALER_CB = {"reduce_fold", "l1_accum", "dest_accum", "dest_accum_pairs"}

# Precision configs, "<input>-<accum>": the two independent knobs are the INPUT tensor dtype and the
# ACCUMULATION (DEST / intermediate-CB) dtype. Only three of the four combinations are meaningful:
#   fp32-fp32 : fp32 input, fp32 accumulation                       (baseline — both precise)
#   bf16-fp32 : bf16 input, fp32 accumulation                       (input-quantization floor)
#   bf16-bf16 : bf16 input, bf16 accumulation                       (+ accumulation loss)
# fp32-bf16 (fp32 input into a bf16 accumulator) is omitted: a bf16 accumulator below an fp32 input
# is never what you'd build. l1_accum is special — the packer L1-accumulate datapath is fp32-DEST-only,
# so it always uses fp32 DEST and the accum knob selects the L1-accumulator CB format instead.
PRECISIONS = ("fp32-fp32", "bf16-fp32", "bf16-bf16")
_DTYPE_NAMES = ("fp32", "bf16")


def split_precision(precision):
    """ "bf16-fp32" -> ("bf16", "fp32") = (input dtype, accumulation dtype)."""
    parts = precision.split("-")
    if len(parts) != 2 or parts[0] not in _DTYPE_NAMES or parts[1] not in _DTYPE_NAMES:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return parts[0], parts[1]


# =============================================================================
# Compute kernel — ONE source for all four methods. `method` is a compile-time arg, so each
# variant compiles to exactly the accumulation path it names and nothing else.
# CT args: [width_tiles, method, kernel_iters]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_zero = 2, cb_interm = 3, cb_out = 16;

    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t method = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t dst_fp32 = get_compile_time_arg_val(3);       // 1 -> DEST is fp32, else bf16
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(4);    // float bits of 1/(width*32) (SFPU finalize)

    using namespace compute_kernel_lib;
    using ckernel::PoolType;
    using ckernel::ReduceDim;

    // method ids: 0 reduce_fold | 1 l1_accum | 2 dest_accum(FPU finalize) | 3 dest_accum_pairs(FPU) |
    //             4 dest_accum(SFPU finalize) | 5 dest_accum_pairs(SFPU finalize)
    constexpr bool is_dest = (method >= 2);          // DST add-accumulation methods
    constexpr bool sfpu_final = (method >= 4);       // finalize the accumulated tile on the SFPU
    constexpr bool pairs = (method == 3 || method == 5);
    constexpr bool uses_zero = is_dest && !pairs;    // only the 1-tile-per-add methods need the zero tile
    constexpr DataFormat dst_fmt = (dst_fp32 != 0) ? DataFormat::Float32 : DataFormat::Float16_b;

    // Boot hw configure for reduce_fold (once, while the engines are idle). The accumulation methods
    // re-init INSIDE the loop instead — see the per-iteration comment below.
    if constexpr (method == 0) {
        compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    }

    // reduce_fold consumes (pops) the whole row every reduce call, so it re-marks the resident
    // shard each iteration. The accumulation methods use indexed access, so they mark the row
    // available ONCE and never pop it until the end.
    if constexpr (method != 0) {
        cb_reserve_back(cb_in, width_tiles);
        cb_push_back(cb_in, width_tiles);
        if constexpr (uses_zero) {  // only the 1-tile-per-add methods add against the zero tile
            cb_wait_front(cb_zero, 1);
        }
    }

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (method == 0) {
            // ---- reduce_fold: the reduce library folds the cross-tile accumulation (running sum
            //      across the width in DEST) together with the within-tile column reduce.
            cb_reserve_back(cb_in, width_tiles);
            cb_push_back(cb_in, width_tiles);
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_in, cb_scaler, cb_out,
                   ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(1, width_tiles));
        } else {
            // Re-establish the full accumulate hw config EVERY iteration. The previous iteration's
            // finalize reprogrammed unpack/math/pack; the short inits below (copy_tile_init /
            // add_tiles_init) do NOT restore that. `ocb` is where the compute-phase pack lands: the
            // L1 accumulator/interm for the FPU-finalize methods, or cb_out directly for the SFPU-
            // finalize methods (they pack the finished mean straight out — no interm round-trip).
            if constexpr (method == 1) {
                binary_op_init_common(cb_in, cb_in, cb_interm);
            } else {
                // dest methods: pairs add (cb_in, cb_in), 1-per-add adds (cb_in, cb_zero). SFPU
                // finalize packs straight to cb_out; FPU finalize packs the running sum to cb_interm.
                binary_op_init_common(cb_in, pairs ? cb_in : cb_zero, sfpu_final ? cb_out : cb_interm);
            }

            cb_wait_front(cb_in, width_tiles);

            if constexpr (method == 1) {
                // ---- l1_accum: pack each tile onto ONE L1 tile via the packer's L1 accumulator.
                //      pack_tile<true> keeps the out-of-order write pointer fixed so L1-acc lands on
                //      the running tile; tile 0 initialises (L1-acc off), tiles 1.. accumulate.
                cb_reserve_back(cb_interm, 1);
                copy_tile_init(cb_in);
                tile_regs_acquire();
                copy_tile(cb_in, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_interm, 0);
                tile_regs_release();
                for (uint32_t k = 1; k < width_tiles; ++k) {
                    tile_regs_acquire();
                    copy_tile(cb_in, k, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_l1_acc(1);
                    pack_tile<true>(0, cb_interm, 0);
                    pack_reconfig_l1_acc(0);
                    tile_regs_release();
                }
                cb_push_back(cb_interm, 1);
                reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_interm, cb_scaler, cb_out,
                       ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(1, 1));
            } else {
                // ---- dest_accum (2/4) / dest_accum_pairs (3/5): accumulate the row into DEST[0] via
                //      add_tiles(acc_to_dest), then finalize on the FPU or the SFPU.
                tile_regs_acquire();
                if constexpr (!pairs) {
                    add_tiles_init(cb_in, cb_zero, false);   // seed: DEST = cb_in[0] + 0
                    add_tiles(cb_in, cb_zero, 0, 0, 0);
                    add_tiles_init(cb_in, cb_zero, true);     // DEST += cb_in[k] + 0
                    for (uint32_t k = 1; k < width_tiles; ++k) {
                        add_tiles(cb_in, cb_zero, k, 0, 0);
                    }
                } else {
                    // two input tiles per add: DEST += cb_in[k] + cb_in[k+1]. Resolve odd/even parity
                    // AT THE SEED so the pair loop is uniform and needs NO phantom zero tile: an odd
                    // width seeds ONE tile via copy_tile (unary — no partner needed), an even width
                    // seeds the first pair; the remainder is then always even. More general than a
                    // zero-CB leftover — no extra CB, no dataflow zero-fill, and W==1 falls out for free.
                    uint32_t k;
                    if constexpr (width_tiles & 1u) {
                        copy_tile_init(cb_in);
                        copy_tile(cb_in, 0, 0);               // odd: DEST = cb_in[0]
                        k = 1;
                    } else {
                        add_tiles_init(cb_in, cb_in, false);
                        add_tiles(cb_in, cb_in, 0, 1, 0);     // even: DEST = cb_in[0] + cb_in[1]
                        k = 2;
                    }
                    add_tiles_init(cb_in, cb_in, true);
                    for (; k < width_tiles; k += 2) {         // even remainder -> exact pairs, no leftover
                        add_tiles(cb_in, cb_in, k, k + 1, 0);
                    }
                }

                if constexpr (sfpu_final) {
                    // ---- SFPU finalize: collapse DEST[0]'s 32 columns IN PLACE (the SFPU reads DEST
                    //      natively — no pack->L1->unpack round-trip that the FPU reduce needs), then
                    //      apply the 1/(W*32) mean scaler with an SFPU scalar-multiply. REDUCE_ROW on
                    //      the SFPU supports SUM (not AVG), hence the explicit post-scale.
                    sfpu_reduce_init<PoolType::SUM, dst_fmt>();
                    sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, /*ct_dim=*/1, /*rt_dim=*/1);
                    binop_with_scalar_tile_init();
                    mul_unary_tile(0, scaler_bits);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_out, 1);
                    pack_tile(0, cb_out, 0);
                    cb_push_back(cb_out, 1);
                    tile_regs_release();
                } else {
                    // ---- FPU finalize: pack the sum to L1, then the reduce library collapses it. ----
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_interm, 1);
                    pack_tile(0, cb_interm, 0);
                    cb_push_back(cb_interm, 1);
                    tile_regs_release();
                    reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_interm, cb_scaler, cb_out,
                           ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(1, 1));
                }
            }
        }

        // Drain the output between steady-state iters; leave the last pass resident for readback.
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, 1);
            cb_pop_front(cb_out, 1);
        }
    }

    if constexpr (method != 0) {
        cb_pop_front(cb_in, width_tiles);
    }
}
"""


# =============================================================================
# Dataflow kernel — fills the reduce scaler tile (1/(width_tiles*32)) for the FPU-reduce methods, and a
# zero tile for the dest_accum methods. Both are pushed once per launch and never popped. The SFPU-
# finalize methods use neither (they scale via an SFPU scalar-multiply), so needs_scaler is 0 there.
# CT args: [width_tiles, needs_scaler, needs_zero]
# =============================================================================
_SCALER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1, cb_zero = 2;
    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t needs_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t needs_zero = get_compile_time_arg_val(2);

    // Mean over the full width = sum over (width_tiles * 32) elements, scaled by 1/N. For the
    // power-of-two widths in the sweep, 1/N is exact in both bf16 and fp32, so the scaler adds no
    // error and the accuracy comparison is purely the accumulation path.
    if constexpr (needs_scaler) {
        const float scaler = 1.0f / static_cast<float>(width_tiles * 32);
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scaler);
    }
    if constexpr (needs_zero) {
        dataflow_kernel_lib::prepare_zero_tile<cb_zero>();
    }
}
"""


# =============================================================================
# Host-side sharded-L1 layout + program descriptor
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(num_tiles):
    """A [32, 32*num_tiles] region as a single-core height shard (tiled)."""
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    return ttnn.create_sharded_memory_config(
        shape=(TILE, num_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dtype_of(dtype_name):
    return ttnn.float32 if dtype_name == "fp32" else ttnn.bfloat16


def _scratch_cb(cb_id, data_format, num_tiles):
    tile_size = ttnn.tile_size(data_format)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=data_format, page_size=tile_size)
    return ttnn.CBDescriptor(total_size=tile_size * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])


def create_program_descriptor(
    input_tensor, output_tensor, *, method, precision, width_tiles, kernel_iters=1, math_fidelity=None
):
    if method not in _METHOD_ID:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    input_dtype, accum_dtype = split_precision(precision)
    if width_tiles < 1:
        raise ValueError(f"width_tiles must be positive, got {width_tiles}")
    if kernel_iters < 1:
        raise ValueError("kernel_iters must be positive")
    input_format = _dtype_of(input_dtype)
    if input_tensor.dtype != input_format or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"input must be {input_dtype} TILE_LAYOUT for precision {precision}")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")

    expected_in = [TILE, width_tiles * TILE]
    expected_out = [TILE, TILE]
    if list(input_tensor.shape) != expected_in or list(output_tensor.shape) != expected_out:
        raise ValueError(f"input shape must be {expected_in} and output shape must be {expected_out}")

    method_id = _METHOD_ID[method]
    accum_format = _dtype_of(accum_dtype)
    needs_zero = method in _NEEDS_ZERO
    needs_interm = method in _NEEDS_INTERM
    needs_scaler_cb = method in _NEEDS_SCALER_CB

    # The scaler CB format tracks the reduce INPUT: reduce_fold reduces the input row directly (input
    # dtype); the other FPU-reduce methods reduce the accumulation-dtype intermediate.
    scaler_format = input_format if method == "reduce_fold" else accum_format

    # fp32 DEST accumulation whenever the accum dtype is fp32. l1_accum is special: the packer's
    # L1-accumulate datapath is fp32-DEST-only (a bf16 DEST corrupts the accumulate), so l1_accum
    # ALWAYS uses fp32 DEST — its accum knob controls the bf16-vs-fp32 L1 accumulator CB instead
    # (each pack-accumulate rounds the running sum to that CB's format).
    fp32_dest = (accum_dtype == "fp32") or (method == "l1_accum")

    # SFPU finalize applies the 1/(width*32) mean scaler with an SFPU scalar-multiply, so pass the
    # float as raw bits (mul_unary_tile takes a bit-pattern). Unused by the FPU-reduce methods.
    scaler_bits = struct.unpack("<I", struct.pack("<f", 1.0 / (width_tiles * TILE)))[0]

    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[width_tiles, method_id, kernel_iters, int(fp32_dest), scaler_bits],
        # HiFi4 (the default) fixes the reduce's scaler-multiply fidelity across every variant, so the
        # measured accuracy gap is the input/accumulation dtype, not a fidelity difference. Pass an
        # explicit `math_fidelity` to sweep that axis (LoFi/HiFi2/HiFi3/HiFi4) instead.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity or ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest
        ),
    )
    scaler = ttnn.KernelDescriptor(
        kernel_source=_SCALER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[width_tiles, int(needs_scaler_cb), int(needs_zero)],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    if needs_scaler_cb:
        cbs.append(_scratch_cb(CB_SCALER, scaler_format, 1))
    if needs_zero:
        cbs.append(_scratch_cb(CB_ZERO, input_format, 1))  # add operand must match the input format
    if needs_interm:
        cbs.append(_scratch_cb(CB_INTERM, accum_format, 1))

    return ttnn.ProgramDescriptor(kernels=[scaler, compute], semaphores=[], cbs=cbs)


def run_op(input_tensor, *, method, precision, width_tiles, kernel_iters=1, math_fidelity=None):
    """Allocate the fp32 single-tile output and run one (method, precision) over the resident row."""
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(1),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        method=method,
        precision=precision,
        width_tiles=width_tiles,
        kernel_iters=kernel_iters,
        math_fidelity=math_fidelity,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
