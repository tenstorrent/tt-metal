# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated single-core micro-bench: rms_norm pass-1 (per-tile-row Sigma_slice x^2 * (1/W)),
reconstructed faithfully from kernels/rms_norm_xcore_compute.cpp `do_pass1`, and A/B'd against
three restructures of the Sigma x^2 reduce. Pure compute, zero-copy sharded L1 (no DRAM) — exactly
the regime that made the op COMPUTE-BOUND, which is where a compute-only reduce restructure can show.

Per core (focus BLOCK_SHARDED (1,1,8192,1024), 8x8 grid): HT_LOCAL tile-rows resident, each PER_W_T
W-tiles wide; vwt (<= PER_W_T) valid W-tiles reduced. For each tile-row the kernel produces one
partial = (Sum over the vwt*32 slice columns of x^2) * (1/factor), written to cb_stat_local col 0.

Variants (the MENU) — all run under the SAME user precision contract (fp32_dest_acc_en, math_fidelity
fixed by the host; never tuned for speed):
  baseline    - op-faithful: square the vwt tiles in ONE eltwise_chain (BinaryFpu Mul, Block-walk) ->
                cb_xsq (streaming L1), then ckl::reduce<SUM,REDUCE_ROW> (Algorithm Auto == ReduceTile,
                the FPU matmul-with-ones datapath) with a 1/factor scaler. This is the op's do_pass1.
  accviaadd   - IDENTICAL square -> cb_xsq, but finalize with ckl::reduce_mean<REDUCE_ROW,
                AccumulateViaAdd> (pairwise add_tiles(acc_to_dest) cross-tile accumulate + SFPU
                sfpu_reduce finalize + 1/factor). Isolates JUST the reduce-restructure (no fusion),
                helper-level (only the reduce algorithm template param changes vs baseline).
  fused_fpu   - fuse the square INTO the accumulate: BinaryFpu<Mul, DestAccumulation::Enabled>
                accumulates Sum_w x_w^2 into one sticky DEST tile (no cb_xsq round-trip), packs it,
                then ckl::reduce<SUM,REDUCE_ROW> over that ONE tile (within-tile collapse) * 1/factor.
                Helper-level fusion; FPU finalize.
  fused_sfpu  - fuse the square into the accumulate AND finalize on the SFPU in the SAME DEST window:
                raw-LLK mul_tiles(acc_to_dest) square-accumulate -> sfpu_reduce<SUM,REDUCE_ROW> ->
                mul_unary_tile(1/factor) -> pack. No cb_xsq, no intermediate pack. (Raw LLK — see the
                kernel-head comment: no helper does square-acc-then-SFPU-finalize in one DEST acquire.)

Correctness is the only pass/fail: PCC of the per-row partials vs a torch reference. Perf (DEVICE
KERNEL DURATION [ns]) is measured, never asserted.
"""

import struct

import ttnn

TILE = 32

# CB assignment — recognizable op indices (kernels/rms_norm_xcore_compute.cpp namespace).
CB_X_IN = 1  # resident sharded W-slice (zero-copy): HT_LOCAL*PER_W_T bf16 tiles
CB_SCALER = 2  # 1/factor reduce scaler (bf16), filled by the reader kernel
CB_SQSUM = 3  # fused_fpu: packed Sum_w x_w^2 (one tile) before the within-tile reduce
CB_STAT_LOCAL = 25  # output: HT_LOCAL partial tiles (fp32), tensor-backed
CB_XSQ = 24  # baseline/accviaadd: x^2 streaming intermediate (bf16), depth 2*PER_W_T

VARIANTS = ("baseline", "accviaadd", "fused_fpu", "fused_sfpu")
BASELINE = "baseline"
_VARIANT_ID = {name: i for i, name in enumerate(VARIANTS)}


# =============================================================================
# Compute kernel — all four variants behind an `if constexpr (VARIANT == ...)`.
# CT args: [HT_LOCAL, PER_W_T, vwt, VARIANT, kernel_iters, fp32_dest, scaler_bits, factor]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;
using ckernel::PoolType;
using ckernel::ReduceDim;

namespace {
constexpr uint32_t cb_x_in = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_sqsum = 3;
constexpr uint32_t cb_xsq = 24;
constexpr uint32_t cb_stat_local = 25;
}  // namespace

void kernel_main() {
    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(0);
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(1);
    constexpr uint32_t vwt = get_compile_time_arg_val(2);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(3);  // 0 baseline,1 accviaadd,2 fused_fpu,3 fused_sfpu
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t fp32_dest = get_compile_time_arg_val(5);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(6);  // float bits of 1/factor
    constexpr uint32_t factor = get_compile_time_arg_val(7);       // elements reduced per output row

    constexpr DataFormat dst_fmt = (fp32_dest != 0) ? DataFormat::Float32 : DataFormat::Float16_b;
    constexpr uint32_t shard_tiles = HT_LOCAL * PER_W_T;
    constexpr auto one_tile = ckl::EltwiseShape::of(1, 1);

    compute_kernel_hw_startup(cb_x_in, cb_scaler, cb_stat_local);
    cb_wait_front(cb_scaler, 1);  // scaler resident (reader filled it once); wait, never pop

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // Re-expose the resident zero-copy shard for this pass (self-armed; no external producer).
        cb_reserve_back(cb_x_in, shard_tiles);
        cb_push_back(cb_x_in, shard_tiles);
        cb_wait_front(cb_x_in, shard_tiles);

        for (uint32_t t = 0; t < HT_LOCAL; ++t) {
            const uint32_t base = t * PER_W_T;

            if constexpr (VARIANT == 0) {
                // -------- baseline: square block -> cb_xsq (streaming) + matmul-reduce (ReduceTile) --------
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(vwt),
                    ckl::BinaryFpu<
                        cb_x_in, cb_x_in, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                        ckl::OperandKind::Block, ckl::OperandKind::Block,
                        ckl::TileOffset::Set, ckl::TileOffset::Set>{base, base},
                    ckl::PackTile<cb_xsq, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_stat_local>(
                    ckl::ReduceInputBlockShape::of(1, vwt, 1));

            } else if constexpr (VARIANT == 1) {
                // -------- accviaadd: same square -> cb_xsq, but AccumulateViaAdd + SFPU finalize --------
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(vwt),
                    ckl::BinaryFpu<
                        cb_x_in, cb_x_in, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                        ckl::OperandKind::Block, ckl::OperandKind::Block,
                        ckl::TileOffset::Set, ckl::TileOffset::Set>{base, base},
                    ckl::PackTile<cb_xsq, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
                ckl::reduce_mean<
                    ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_stat_local,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    ckl::ReduceInputBlockShape::of(1, vwt, 1), factor);

            } else if constexpr (VARIANT == 2) {
                // -------- fused_fpu: DestAccumulation Mul square-acc -> pack -> matmul-reduce(1 tile) --------
                // BinaryFpu<Mul, DestAccumulation::Enabled> => mul_tiles_init(acc_to_dest=1): DEST += x_w^2
                // over the vwt-tile Block-walk (A==B==cb_x_in tile w). One pack of the summed tile at exit.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(vwt),
                    ckl::BinaryFpu<
                        cb_x_in, cb_x_in, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::None,
                        ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                        ckl::OperandKind::Block, ckl::OperandKind::Block,
                        ckl::TileOffset::Set, ckl::TileOffset::Set,
                        ckl::DestAccumulation::Enabled>{base, base},
                    ckl::PackTile<cb_sqsum, ckl::OutputLifecycle::DestAccumulation,
                                  ckl::PackTileReconfig::Output, ckl::Dst::D0>{});
                // within-tile collapse of the summed tile (1 tile) * 1/factor scaler.
                ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sqsum, cb_scaler, cb_stat_local>(
                    ckl::ReduceInputBlockShape::of(1, 1, 1));

            } else {
                // -------- fused_sfpu (RAW LLK): mul_tiles(acc_to_dest) square-acc + sfpu_reduce finalize --------
                // Raw LLK is used here because NO kernel_lib helper does square-accumulate-then-SFPU-finalize
                // inside ONE DEST acquire: the DestAccumulation PackTile helper (fused_fpu) packs the summed
                // tile to L1 at chain exit, forcing an L1 round-trip before any within-tile reduce; here the
                // sfpu_reduce collapses the accumulated tile IN DEST (reads DST natively) with no pack. The
                // helpers bypassed: ckl::eltwise_chain(BinaryFpu Mul,DestAccumulation) + ckl::reduce. Mechanism:
                // FPU dest-accumulate (mul_tiles_init acc_to_dest, seeded non-acc on tile 0) keeps Sum_w x_w^2
                // in one sticky DST tile, then the SFPU within-tile reduce + 1/factor mul run on that same tile.
                tile_regs_acquire();
                // 4-arg form (explicit call_line) selects the acc_to_dest overload unambiguously.
                mul_tiles_init(cb_x_in, cb_x_in, /*acc_to_dest=*/0u, __builtin_LINE());  // seed: DST = x_0^2
                mul_tiles(cb_x_in, cb_x_in, base + 0, base + 0, 0);
                if constexpr (vwt > 1) {
                    mul_tiles_init(cb_x_in, cb_x_in, /*acc_to_dest=*/1u, __builtin_LINE());  // DST += x_w^2
                    for (uint32_t w = 1; w < vwt; ++w) {
                        mul_tiles(cb_x_in, cb_x_in, base + w, base + w, 0);
                    }
                }
                sfpu_reduce_init<PoolType::SUM, dst_fmt>();
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);  // per-row Sum in col 0
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scaler_bits);  // * 1/factor
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_stat_local, 1);
                pack_reconfig_data_format(cb_stat_local);
                pack_tile(0, cb_stat_local, 0);
                cb_push_back(cb_stat_local, 1);
                tile_regs_release();
            }
        }

        // Drain the HT_LOCAL partials + release the resident input so the next pass starts clean.
        cb_pop_front(cb_x_in, shard_tiles);
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_stat_local, HT_LOCAL);
            cb_pop_front(cb_stat_local, HT_LOCAL);
        }
    }
}
"""


# =============================================================================
# Reader (scaler prep) kernel — fills cb_scaler with 1/factor in the reduce row-0 layout.
# Uniform across ALL variants so the program structure is identical and the measured delta is
# attributable only to the compute-kernel restructure. CT args: [scaler_bits, factor]
# =============================================================================
_SCALER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);  // float bits of 1/factor
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        scaler_f);
}
"""


# =============================================================================
# Host-side layout + program descriptor
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def input_shape(ht_local, per_w_t):
    """[H, W] element shape of one core's resident W-slice: HT_LOCAL tile-rows x PER_W_T W-tiles."""
    return (ht_local * TILE, per_w_t * TILE)


def output_shape(ht_local):
    """[H, 32] fp32 output: one tile per tile-row (partial lives in column 0)."""
    return (ht_local * TILE, TILE)


def create_sharded_memory_config(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, data_format, num):
    ts = ttnn.tile_size(data_format)
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=data_format, page_size=ts)
    return ttnn.CBDescriptor(total_size=ts * num, core_ranges=_single_core(), format_descriptors=[fmt])


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    ht_local,
    per_w_t,
    vwt,
    factor,
    fp32_dest=False,
    kernel_iters=1,
    math_fidelity=None,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if not (1 <= vwt <= per_w_t):
        raise ValueError(f"vwt must be in [1, per_w_t]={per_w_t}, got {vwt}")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")

    fidelity = math_fidelity or ttnn.MathFidelity.HiFi2  # focus contract: HiFi2
    scaler_bits = _f32_bits(1.0 / factor)

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_STAT_LOCAL, output_tensor),
        _scratch_cb(CB_SCALER, ttnn.bfloat16, num=1),
    ]
    if variant in ("baseline", "accviaadd"):
        cbs.append(_scratch_cb(CB_XSQ, ttnn.bfloat16, num=2 * per_w_t))
    if variant == "fused_fpu":
        cbs.append(_scratch_cb(CB_SQSUM, ttnn.float32 if fp32_dest else ttnn.bfloat16, num=2))

    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            ht_local,
            per_w_t,
            vwt,
            _VARIANT_ID[variant],
            kernel_iters,
            int(fp32_dest),
            scaler_bits,
            factor,
        ],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=bool(fp32_dest)),
    )
    scaler = ttnn.KernelDescriptor(
        kernel_source=_SCALER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[scaler_bits, factor],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[scaler, compute], semaphores=[], cbs=cbs)


def allocate_output(device, ht_local):
    """Allocate ONE fp32 output tensor for a config; reuse it across variants/trials (generic_op
    overwrites it each launch). Avoids per-call L1 allocations piling up across a sweep."""
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(output_shape(ht_local))),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(output_shape(ht_local)),
    )


def run_op(
    input_tensor,
    output=None,
    *,
    variant,
    ht_local,
    per_w_t,
    vwt,
    factor,
    fp32_dest=False,
    kernel_iters=1,
    math_fidelity=None,
):
    if output is None:
        output = allocate_output(input_tensor.device(), ht_local)
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        variant=variant,
        ht_local=ht_local,
        per_w_t=per_w_t,
        vwt=vwt,
        factor=factor,
        fp32_dest=fp32_dest,
        kernel_iters=kernel_iters,
        math_fidelity=math_fidelity,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
