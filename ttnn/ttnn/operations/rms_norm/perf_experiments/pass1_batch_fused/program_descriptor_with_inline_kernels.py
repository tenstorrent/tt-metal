# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated single-core micro-bench: rms_norm pass-1, COMPOSING the graduated fused-FPU
square-DEST-accumulate with a C-row BATCHED reduce.

Reconstructed faithfully from kernels/rms_norm_xcore_compute.cpp `do_pass1` (the PASS1_FUSED path,
~line 356). Round 1 tested "fused" (square into a DEST-accumulate) and "batch-rows" (batch the
reduce over C tile-rows) as ALTERNATIVES; the fused path won and graduated. This bench COMPOSES
them: keep the fused square-DEST-accumulate that collapses vwt x-tiles into ONE summed x^2 tile per
tile-row, but instead of running a separate 1-tile REDUCE_ROW per tile-row (C reduce datapath
invocations / round), buffer the C summed x^2 tiles and run ONE batched REDUCE_ROW of(C, 1) that
amortizes the reduce init + data-format reconfig + scaler-wait + pipeline fill/drain over the whole
round.

Per core (focus BLOCK_SHARDED (1,1,8192,1024), 8x8 grid): HT_LOCAL=32 tile-rows resident, each
PER_W_T=4 W-tiles wide; vwt(<=PER_W_T)=4 valid W-tiles reduced; C_ROWS=8 tile-rows batched per
cross-core round; num_rounds=ceil(HT_LOCAL/C_ROWS)=4. Pure compute, zero-copy sharded L1 (no DRAM,
no NoC) — exactly the regime that made pass-1 COMPUTE-BOUND, where a reduce-setup amortization can
show. Each tile-row produces one partial = (Sum over the vwt*32 slice cols of x^2) * (1/factor) in
cb_stat_local col 0.

Variants (the MENU) — all run under the SAME user precision contract, never tuned for speed:
  baseline_fused          - op-faithful PASS1_FUSED: per tile-row, BinaryFpu<Mul,
                            DestAccumulation::Enabled> accumulates Sum_w x_w^2 into ONE DEST tile
                            -> pack once to cb_xsq -> a SINGLE 1-tile REDUCE_ROW * (1/factor) ->
                            one partial. The reduce datapath is invoked C times / round.
  batch_fused             - COMPOSITION: run the C fused square-accumulates first (-> C summed x^2
                            tiles buffered in cb_xsq), THEN ONE batched reduce of(C, 1) producing C
                            independent partials. Same math, same DEST-accumulate order per row
                            (so PCC is identical to baseline_fused); only the reduce SETUP
                            (init/reconfig/scaler-wait/fill-drain) is paid once/round instead of C
                            times. cb_xsq deepens 2 -> C tiles (the L1 cost — the predicate bound).
  batch_fused_noreconfig  - batch_fused + drop the reduce's redundant INPUT data-format reconfig
                            (cb_xsq is bf16, same as the fused chain's srcA format; only the OUTPUT
                            reconfig to fp32 cb_stat_local is a real change and is kept). A second,
                            compounding lever; numerically byte-identical to batch_fused.

Correctness is the only pass/fail: PCC of the per-row partials vs an fp32 torch reference. Perf
(DEVICE KERNEL DURATION [ns]) and PCC are measured, never asserted. The precision contract
(bf16 in / fp32 out / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False) is identical across
every variant.
"""

import struct

import ttnn

TILE = 32

# CB assignment — recognizable op indices (kernels/rms_norm_xcore_compute.cpp namespace).
CB_X_IN = 1  # resident sharded W-slice (zero-copy): HT_LOCAL*PER_W_T bf16 tiles
CB_SCALER = 2  # 1/factor reduce scaler (bf16), filled by the reader kernel
CB_XSQ = 24  # summed x^2 tiles (bf16): 1 tile/row after the fused DEST-accumulate
CB_STAT_LOCAL = 25  # output: HT_LOCAL partial tiles (fp32), tensor-backed

VARIANTS = ("baseline_fused", "batch_fused", "batch_fused_noreconfig")
BASELINE = "baseline_fused"
_VARIANT_ID = {name: i for i, name in enumerate(VARIANTS)}


# =============================================================================
# Compute kernel — all three variants behind an `if constexpr (VARIANT == ...)`.
# CT args: [HT_LOCAL, PER_W_T, vwt, C_ROWS, VARIANT, kernel_iters, fp32_dest, scaler_bits, factor]
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;
using ckernel::PoolType;
using ckernel::ReduceDim;

namespace {
constexpr uint32_t cb_x_in = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_xsq = 24;
constexpr uint32_t cb_stat_local = 25;
}  // namespace

// Isolated rms_norm pass-1, fused square-DEST-accumulate composed with a C-row batched reduce.
//
// The fused square (BinaryFpu<Mul, DestAccumulation::Enabled>) is IDENTICAL across variants — it
// collapses the vwt resident x-tiles of a tile-row into ONE summed x^2 tile in DEST (packed once to
// cb_xsq). The ONLY axis under test is how the per-row summed tiles are reduced into partials:
//   VARIANT 0 (baseline_fused): a SEPARATE 1-tile REDUCE_ROW right after each row's square (the
//              op's current PASS1_FUSED) — the reduce datapath (init + reconfig + scaler-wait +
//              fill/drain) is paid C times per cross-core round.
//   VARIANT 1/2 (batch_fused): buffer the round's C summed x^2 tiles, then ONE batched reduce
//              of(C, 1) producing C partials — reduce setup paid ONCE per round. VARIANT 2 also
//              drops the reduce's redundant INPUT format reconfig (cb_xsq is bf16, constant).
// No raw LLK: every phase is a kernel_lib helper (ckl::eltwise_chain + ckl::reduce), exactly the
// helpers the op uses; only the reduce call granularity (per-row vs batched) + reconfig flag change.
void kernel_main() {
    constexpr uint32_t HT_LOCAL = get_compile_time_arg_val(0);
    constexpr uint32_t PER_W_T = get_compile_time_arg_val(1);
    constexpr uint32_t vwt = get_compile_time_arg_val(2);
    constexpr uint32_t C_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(4);  // 0 baseline_fused, 1 batch_fused, 2 batch_fused_noreconfig
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr uint32_t fp32_dest = get_compile_time_arg_val(6);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(7);  // float bits of 1/factor
    constexpr uint32_t factor = get_compile_time_arg_val(8);       // elements reduced per output row

    constexpr uint32_t shard_tiles = HT_LOCAL * PER_W_T;
    constexpr uint32_t num_rounds = (HT_LOCAL + C_ROWS - 1) / C_ROWS;
    constexpr bool BATCHED = (VARIANT != 0);
    // Reduce INPUT reconfig is redundant on the fused path (cb_xsq is bf16, same format the fused
    // chain left srcA in); only the OUTPUT (fp32 cb_stat_local) is a real change. VARIANT 2 drops
    // the INPUT reconfig; everything else keeps the op's default INPUT_AND_OUTPUT.
    constexpr auto RD_RC = (VARIANT == 2) ? ckl::ReduceDataFormatReconfigMode::OUTPUT
                                          : ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;

    compute_kernel_hw_startup(cb_x_in, cb_scaler, cb_stat_local);
    cb_wait_front(cb_scaler, 1);  // scaler resident (reader filled it once); wait, never pop

    // Fused square-DEST-accumulate of tile-row t: Sum_w x_w^2 -> ONE summed tile packed to cb_xsq.
    // BinaryFpu<Mul, DestAccumulation::Enabled> == mul_tiles_init(acc_to_dest) over the vwt-tile
    // Block-walk from the resident base t*PER_W_T (A==B==cb_x_in). Identical to the op's PASS1_FUSED
    // square; the summed tile is packed via the DestAccumulation PackTile (one push_back to cb_xsq).
    auto fused_square = [&](uint32_t t) {
        const uint32_t base = t * PER_W_T;
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(vwt),
            ckl::BinaryFpu<
                cb_x_in, cb_x_in, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::None,
                ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::Input, ckl::Dst::D0,
                ckl::OperandKind::Block, ckl::OperandKind::Block,
                ckl::TileOffset::Set, ckl::TileOffset::Set,
                ckl::DestAccumulation::Enabled>{base, base},
            ckl::PackTile<cb_xsq, ckl::OutputLifecycle::DestAccumulation,
                          ckl::PackTileReconfig::Output, ckl::Dst::D0>{});
    };

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // Re-expose the resident zero-copy shard for this pass (self-armed; no external producer).
        cb_reserve_back(cb_x_in, shard_tiles);
        cb_push_back(cb_x_in, shard_tiles);
        cb_wait_front(cb_x_in, shard_tiles);

        for (uint32_t r = 0; r < num_rounds; ++r) {
            const uint32_t base_t = r * C_ROWS;
            const uint32_t rem = HT_LOCAL - base_t;
            const uint32_t C_this = (rem > C_ROWS) ? C_ROWS : rem;  // short last round

            if constexpr (!BATCHED) {
                // -------- baseline_fused (op PASS1_FUSED): per tile-row square-acc + 1-tile reduce --------
                for (uint32_t cc = 0; cc < C_this; ++cc) {
                    fused_square(base_t + cc);
                    // within-tile collapse of the summed tile (1 tile) * 1/factor scaler -> one partial.
                    ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_stat_local>(
                        ckl::ReduceInputBlockShape::of(1, 1, 1));
                }
            } else {
                // -------- batch_fused: C fused square-accs buffered, then ONE batched reduce of(C,1) --------
                for (uint32_t cc = 0; cc < C_this; ++cc) {
                    fused_square(base_t + cc);  // pushes 1 summed x^2 tile to cb_xsq (C_this total)
                }
                // ONE reduce over the round's C_this summed tiles: loops rows internally (1 acquire/pack
                // per row), but pays the reduce init + format reconfig + scaler-wait ONCE for the block.
                ckl::reduce<
                    PoolType::SUM, ReduceDim::REDUCE_ROW, cb_xsq, cb_scaler, cb_stat_local,
                    ckl::ReduceInputPolicy::WaitAndPopPerTile, RD_RC>(
                    ckl::ReduceInputBlockShape::of(C_this, 1, 1));
            }
        }

        // Drain the HT_LOCAL partials between steady-state iters; leave the last pass resident for
        // readback (correctness gate). cb_stat_local is sized HT_LOCAL so a whole iter fits.
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
# attributable only to the compute-kernel restructure. CT args: [scaler_bits]
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


def cb_xsq_tiles(variant, c_rows):
    """L1 cost of the fused-square buffer (the predicate bound).

    baseline_fused: double-buffered ONE summed tile (2), identical to the op's per-row PASS1_FUSED.
    batch_fused*  : the whole round's C summed tiles must buffer before the single reduce drains
                    them (else the same-kernel packer/unpacker pair deadlocks) -> C_ROWS tiles.
    """
    return 2 if variant == BASELINE else c_rows


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    ht_local,
    per_w_t,
    vwt,
    c_rows,
    factor,
    fp32_dest=False,
    kernel_iters=1,
    math_fidelity=None,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if not (1 <= vwt <= per_w_t):
        raise ValueError(f"vwt must be in [1, per_w_t]={per_w_t}, got {vwt}")
    if not (1 <= c_rows <= ht_local):
        raise ValueError(f"c_rows must be in [1, ht_local]={ht_local}, got {c_rows}")
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
        _scratch_cb(
            CB_XSQ,
            ttnn.float32 if fp32_dest else ttnn.bfloat16,
            num=cb_xsq_tiles(variant, c_rows),
        ),
    ]

    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            ht_local,
            per_w_t,
            vwt,
            c_rows,
            _VARIANT_ID[variant],
            kernel_iters,
            int(fp32_dest),
            scaler_bits,
            factor,
        ],
        # FIXED precision contract of the BLOCK_SHARDED focus case — identical across every variant,
        # never tuned for speed.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fidelity,
            fp32_dest_acc_en=bool(fp32_dest),
            math_approx_mode=False,
        ),
    )
    scaler = ttnn.KernelDescriptor(
        kernel_source=_SCALER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[scaler_bits],
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
    c_rows,
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
        c_rows=c_rows,
        factor=factor,
        fp32_dest=fp32_dest,
        kernel_iters=kernel_iters,
        math_fidelity=math_fidelity,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
