# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The accumulate + SFPU-finalize fast path, generalized to a 2-D tile block reduced along one dim.

The committed `reduce_accumulate` example only reduces a block whose NON-reduced dimension is a single
tile — `of(1, Wt)` (row) or `of(Ht, 1)` (col) — so it always collapses the whole input into ONE output
tile. The reduce library, though, accepts a full `(Ht, Wt, NC)` block reduced along one dim and emits
MULTIPLE output tiles. This module takes the fast path into exactly those shapes and benchmarks it against
the library there.

The fast path does not collapse a 2-D block into one tile; instead it becomes a LOOP over output tiles,
each an independent "accumulate its input subset into one DEST tile, then SFPU-finalize" — the same kernel
the 1-D example uses, applied per output tile. The subset per output tile is the only per-dim difference:

    REDUCE_ROW  (reduce width):  output (nc,h) = the Wt tiles contiguous from (nc*Ht+h)*Wt   -> Ht*NC tiles
    REDUCE_COL  (reduce height): output (nc,w) = the Ht tiles strided by Wt from nc*Ht*Wt+w  -> Wt*NC tiles
    REDUCE_SCALAR:               output (nc)   = all Ht*Wt tiles of batch nc                 -> NC tiles

Because each output tile uses a single DEST register (accumulate, then finalize in place), the fast path
handles an arbitrary block one DEST at a time — it never hits the DEST/chunk limit the library's REDUCE_COL
path chunks around.

Variants:
  reduce_tile               — the reduce library, default datapath (ReduceAlgorithm::Auto -> ReduceTile,
                              FPU matmul-with-ones), AVG so the 1/N is per dim.
  accumulate_via_add        — the reduce library with the opt-in ReduceAlgorithm::AccumulateViaAdd.
  accumulate_via_add_inline — the per-output-tile accumulate + SFPU-finalize loop as a standalone
                              hand-written kernel (init hoisted out of the kernel_iters loop).
  dispatch                  — accumulate_via_add when the reduced tile count per output (row=Wt, col=Ht,
                              scalar=Ht*Wt) >= the measured per-dim threshold, else reduce_tile.

Input bf16, output fp32, everything sharded in L1 on one Tensix core. Correctness is the gate; perf (device
kernel ns) and accuracy (vs the fp64 mean) are measured.
"""

import struct

import ttnn

TILE = 32

# CB assignment (semantic names).
CB_IN = 0  # input tiles, sharded L1 (resident): Ht*Wt*NC tiles, row-major, batch-major
CB_SCALER = 1  # AVG reduce scaler tile (reduce_tile / library paths only)
CB_ACC = 2  # cross-call accumulate: running RAW partial-sum tile per output (accumulate path only)
CB_OUT = 16  # output tiles, fp32, tensor-backed (count depends on dim)

VARIANTS = ("reduce_tile", "accumulate_via_add", "accumulate_via_add_inline", "dispatch")
BASELINE = "reduce_tile"
# reduce_tile               = the reduce library, default datapath (ReduceAlgorithm::Auto -> ReduceTile:
#                             FPU matmul-with-ones reduce_tile per input tile).
# accumulate_via_add        = the reduce library with the opt-in ReduceAlgorithm::AccumulateViaAdd.
# accumulate_via_add_inline = the same algorithm as a standalone hand-written kernel, with the one-time
#                             init hoisted OUT of the kernel_iters loop (the init-hoisted reference).

_DIM_ID = {"row": 0, "col": 1, "scalar": 2}
DIMS = tuple(_DIM_ID)
DTYPES = ("fp32", "bf16")  # accumulation (DEST / SFPU) dtype; input is always bf16

# ReduceInputPolicy selector (both datapaths). bulk = BulkWaitBulkPop (resident block, per-output pop);
# stream = WaitAndPopPerTile (stream reduce dim thru DST); wait_upfront = WaitUpfrontNoPop, no_wait =
# NoWaitNoPop (both leave input resident + bulk-reserve the outputs). "stream" is the back-compat name for
# the old `stream=True` flag.
_POLICY_ID = {"bulk": 0, "stream": 1, "wait_upfront": 2, "no_wait": 3}
POLICIES = tuple(_POLICY_ID)
_NO_POP = ("wait_upfront", "no_wait")

# AccumulateViaAdd cross-call reload strategy (later-chunk accumulator fold), for the accumulate bake-off.
# fold = FoldViaAdd (acc as add_tiles SRCB operand; Default-acc only); copy_pairs = CopySeedPairs (copy_tile
# reload + pairwise new + DEST-reuse leftover; safe for UnpackToDestFp32 acc); copy_uniform = CopySeedUniform
# (copy_tile reload + DEST-reuse every new tile; safe, 1 tile/op).
_RELOAD_ID = {"fold": 0, "copy_pairs": 1, "copy_uniform": 2, "copy_sfpu": 3, "copy_zero": 4}
RELOADS = tuple(_RELOAD_ID)

# Fewest REDUCED tiles per output at which the fast path beats the library, per dim (from the 1-D sweep:
# the REDUCE_COL datapath is cheaper than REDUCE_ROW, so col needs more tiles before fast pulls ahead).
_DISPATCH_MIN = {"row": 4, "col": 8, "scalar": 8}


def dispatch_min(dim):
    return _DISPATCH_MIN[dim]


def reduced_count(dim, Ht, Wt):
    """Tiles that collapse into ONE output tile (the reduce length, in tiles), per dim."""
    if dim == "row":
        return Wt
    if dim == "col":
        return Ht
    return Ht * Wt  # scalar


# =============================================================================
# Shape helpers — how a (Ht, Wt, NC) block maps to input/output tensors.
# Input is a [NC*Ht*32, Wt*32] tensor: batch nc occupies tile-rows [nc*Ht, (nc+1)*Ht), so the row-major
# tile order (nc*Ht + h)*Wt + w matches the library's batch-major of(Ht, Wt, NC) traversal — and the fast
# path indexes the same order.
# =============================================================================
def input_shape(Ht, Wt, NC=1):
    return (NC * Ht * TILE, Wt * TILE)


def output_shape(dim, Ht, Wt, NC=1):
    """Output tensor [H, W]. Output tiles are stacked so tile order matches both paths."""
    if dim == "row":  # Ht*NC tiles stacked vertically; per-row means in col 0
        return (NC * Ht * TILE, TILE)
    if dim == "col":  # Wt*NC tiles laid horizontally; per-col means in row 0
        return (TILE, NC * Wt * TILE)
    if dim == "scalar":  # NC tiles; one mean per batch at each tile's [0, 0]
        return (NC * TILE, TILE)
    raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")


def out_tile_count(dim, Ht, Wt, NC=1):
    if dim == "row":
        return NC * Ht
    if dim == "col":
        return NC * Wt
    return NC  # scalar


def elements_reduced(dim, Ht, Wt):
    """Input elements that collapse into one output value (for the 1/N mean scaler)."""
    if dim == "row":
        return Wt * TILE
    if dim == "col":
        return Ht * TILE
    return Ht * Wt * TILE * TILE  # scalar


# =============================================================================
# Fast kernel — per-output-tile accumulate (dim-specific subset) + SFPU finalize.
# Partial (non-tile-aligned) reduce dims (ROW/COL only): the last reduce-dim tile is folded in with a
# DEST-ACCUMULATING masked broadcast-mul (mask 0/1 in the scaler CB), so the invalid positions contribute
# 0. The bulk stays pure add_tiles (fidelity-flat, 2 tiles/op); only the one partial tile is a mul.
# CT args: [Ht, Wt, NC, dim, kernel_iters, dst_fp32, scaler_bits, out_tiles, partial_elems]
#   partial_elems = valid reduce-dim elements in the LAST tile (0 = tile-aligned; the scaler CB is unused).
# =============================================================================
_FAST_BLOCK_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_out = 16;
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t dim = get_compile_time_arg_val(3);          // 0 row, 1 col, 2 scalar
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t dst_fp32 = get_compile_time_arg_val(5);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(6);  // float bits of 1/N (mean scaler)
    constexpr uint32_t out_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t partial_elems = get_compile_time_arg_val(8);  // valid elems in last tile (0 = aligned)
    constexpr uint32_t in_tiles = Ht * Wt * NC;
    constexpr bool has_partial = (partial_elems != 0u);

    using ckernel::PoolType;
    using ckernel::ReduceDim;
    using ckernel::BroadcastType;
    using ckernel::EltwiseBinaryType;
    constexpr DataFormat dst_fmt = (dst_fp32 != 0) ? DataFormat::Float32 : DataFormat::Float16_b;

    // per-dim accumulation geometry (compile-time). full_cnt tiles go through the pure add path; the last
    // reduce-dim tile is masked (partial) or just the last add (aligned).
    constexpr uint32_t cnt = (dim == 0) ? Wt : (dim == 1) ? Ht : (Ht * Wt);
    constexpr uint32_t stride = (dim == 1) ? Wt : 1u;
    constexpr uint32_t full_cnt = has_partial ? (cnt - 1u) : cnt;   // tiles summed via add_tiles
    constexpr BroadcastType MASK_BCAST = (dim == 1) ? BroadcastType::COL : BroadcastType::ROW;

    binary_op_init_common(cb_in, cb_in, cb_out);
    sfpu_reduce_init<PoolType::SUM, dst_fmt>();  // SFPU reduce macro persists across FPU ops (replay)
    if constexpr (has_partial) {
        cb_wait_front(cb_scaler, 1);             // 0/1 mask tile (row0 for ROW, col0 for COL)
    }

    cb_reserve_back(cb_in, in_tiles);
    cb_push_back(cb_in, in_tiles);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_wait_front(cb_in, in_tiles);

        for (uint32_t o = 0; o < out_tiles; ++o) {
            uint32_t start;
            if constexpr (dim == 0) {
                start = o * Wt;                              // row: Wt contiguous tiles
            } else if constexpr (dim == 1) {
                start = (o / Wt) * (Ht * Wt) + (o % Wt);     // col: Ht tiles strided by Wt, in batch o/Wt
            } else {
                start = o * (Ht * Wt);                       // scalar: whole HxW block of batch o
            }

            tile_regs_acquire();

            // ---- pure add accumulate of the full_cnt full tiles, parity resolved at the seed. ----
            // acc_to_dest=true throughout: a freshly-acquired DEST reads 0 on its first write, so the first add
            // is the plain sum (no overwrite-seed), and full_cnt==0 (a single partial tile) simply skips the
            // loop and leaves DEST=0 for the masked fold below — matching the library. Odd count: unary-copy seed.
            uint32_t k = 0;
            if constexpr (full_cnt & 1u) {
                copy_tile_init(cb_in);
                copy_tile(cb_in, start, 0);                  // odd: DEST = cb_in[start]
                k = 1;
            }
            add_tiles_init(cb_in, cb_in, true);
            for (; k < full_cnt; k += 2) {
                add_tiles(cb_in, cb_in, start + k * stride, start + (k + 1) * stride, 0);
            }

            // ---- partial: fold the LAST reduce-dim tile in, masked, ACCUMULATING into DEST. ----
            // The mul_tiles_bcast_* shorthands overwrite (clear_fp32_dst_acc=true); to accumulate we call
            // the LLK directly with acc_to_dest=1 at init and clear_fp32_dst_acc=false at the op.
            if constexpr (has_partial) {
                const uint32_t last = start + full_cnt * stride;
                MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, MASK_BCAST, MATH_FIDELITY>(
                    cb_in, cb_scaler, 1 /* acc_to_dest */)));
                UNPACK((llk_unpack_AB_init<MASK_BCAST>(cb_in, cb_scaler)));
                UNPACK((llk_unpack_AB<MASK_BCAST>(cb_in, cb_scaler, last, 0)));
                MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, MASK_BCAST, DST_ACCUM_MODE,
                                              MATH_FIDELITY>(0, false /* clear_fp32_dst_acc */)));
            }

            // ---- within-tile finalize on the SFPU, then the 1/N mean scaler. ----
            if constexpr (dim == 0) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
            } else if constexpr (dim == 1) {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(0, 1, 1);
            } else {
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
                sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_COL>(0, 1, 1);
            }
            mul_unary_tile(0, scaler_bits);                  // DEST *= 1/N_true (mean)

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out, 0);                         // output tile o -> page o (matches tensor)
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }

        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, out_tiles);
            cb_pop_front(cb_out, out_tiles);
        }
    }
    cb_pop_front(cb_in, in_tiles);
}
"""


# =============================================================================
# Helper kernel — the reduce library over the general (Ht, Wt, NC) block, AVG pool. The `algo` CT arg
# selects the library datapath: 0 = ReduceTile (Auto default, FPU matmul-with-ones), 1 = AccumulateViaAdd.
# `policy` selects one of the four ReduceInputPolicy values for BOTH datapaths; `avg_direct` makes the
# AccumulateViaAdd path use reduce<AVG> (geometry 1/N) instead of reduce_mean (explicit N).
#
# The per-iter reduce dispatch is hoisted into a TEMPLATE (do_reduce_block) so `if constexpr (algo)` GENUINELY
# discards the dead branch. kernel_main is not a template, so an `if constexpr (algo)` there would still
# fully-CHECK (instantiate) the dead branch — a reduce_tile build (algo=0) would then instantiate the
# AccumulateViaAdd branch with e.g. POLICY=WaitUpfrontNoPop or dim=col+stream and trip the fast-path
# static_asserts, breaking the reduce_tile kernel compile. In a template, the discarded branch is not
# instantiated, so each build compiles only its own algo's reduce().
# CT args: [Ht, Wt, NC, dim, kernel_iters, out_tiles, algo, partial_elems, policy_id, recfg, row_stride,
#           n_reduced, avg_direct]
# =============================================================================
_HELPER_BLOCK_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_out = 16;

// Templated so `if constexpr (algo)` discards the dead branch (see the header note). Each build instantiates
// exactly one algo's reduce(), so a policy/dim invalid for the OTHER algo never trips its static_asserts.
template <uint32_t algo, uint32_t dim, uint32_t policy_id, uint32_t avg_direct>
ALWI void do_reduce_block(
    [[maybe_unused]] uint32_t iter,
    compute_kernel_lib::ReduceInputBlockShape shape,
    compute_kernel_lib::ReduceInputMemoryLayout ml,
    compute_kernel_lib::ReducePartialScaler ps,
    [[maybe_unused]] uint32_t n_reduced,
    [[maybe_unused]] uint32_t recfg) {
    using namespace compute_kernel_lib;
    using ckernel::PoolType;
    using ckernel::ReduceDim;
    constexpr ReduceAlgorithm ALG = (algo == 1u) ? ReduceAlgorithm::AccumulateViaAdd : ReduceAlgorithm::Auto;
    constexpr ReduceInputPolicy POLICY = (policy_id == 1u)   ? ReduceInputPolicy::WaitAndPopPerTile
                                         : (policy_id == 2u) ? ReduceInputPolicy::WaitUpfrontNoPop
                                         : (policy_id == 3u) ? ReduceInputPolicy::NoWaitNoPop
                                                             : ReduceInputPolicy::BulkWaitBulkPop;
    constexpr ReduceDim RDIM = (dim == 0u)   ? ReduceDim::REDUCE_ROW
                               : (dim == 1u) ? ReduceDim::REDUCE_COL
                                             : ReduceDim::REDUCE_SCALAR;
    using RECFG = ReduceDataFormatReconfigMode;
    if constexpr (algo == 1u) {
        // AccumulateViaAdd. recfg==1 exercises RECFG::NONE after the first call (reusing the first call's
        // config). avg_direct==1: reduce<AVG> derives 1/N from geometry; else reduce_mean applies the explicit
        // caller N (which is cross-chunk / uneven-shard safe).
        const bool none_now = (recfg == 1u) && (iter > 0u);
        if constexpr (avg_direct == 1u) {
            if (none_now)
                reduce<PoolType::AVG, RDIM, cb_in, cb_scaler, cb_out, POLICY, RECFG::NONE, ALG>(
                    shape, ml, NoAccumulation{}, NoOp{}, ps);
            else
                reduce<PoolType::AVG, RDIM, cb_in, cb_scaler, cb_out, POLICY, RECFG::INPUT_AND_OUTPUT, ALG>(
                    shape, ml, NoAccumulation{}, NoOp{}, ps);
        } else {
            if (none_now)
                reduce_mean<RDIM, cb_in, cb_scaler, cb_out, POLICY, RECFG::NONE, ALG>(
                    shape, n_reduced, ml, NoAccumulation{}, ps);
            else
                reduce_mean<RDIM, cb_in, cb_scaler, cb_out, POLICY, RECFG::INPUT_AND_OUTPUT, ALG>(
                    shape, n_reduced, ml, NoAccumulation{}, ps);
        }
    } else {
        // ReduceTile: AVG via the reader-computed 1/N scaler.
        reduce<PoolType::AVG, RDIM, cb_in, cb_scaler, cb_out, POLICY, RECFG::INPUT_AND_OUTPUT, ALG>(
            shape, ml, NoAccumulation{}, NoOp{}, ps);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t dim = get_compile_time_arg_val(3);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t algo = get_compile_time_arg_val(6);  // 0 Auto->ReduceTile, 1 AccumulateViaAdd
    constexpr uint32_t partial_elems = get_compile_time_arg_val(7);  // valid elems in last tile (0=aligned)
    constexpr uint32_t policy_id = get_compile_time_arg_val(8);  // 0 Bulk, 1 WaitAndPop, 2 WaitUpfront, 3 NoWait
    constexpr uint32_t recfg = get_compile_time_arg_val(9);  // 1 = ReduceDataFormatReconfigMode::NONE after 1st call
    constexpr uint32_t row_stride = get_compile_time_arg_val(10);  // tile pitch per row (0=contiguous=Wt)
    constexpr uint32_t n_reduced = get_compile_time_arg_val(11);  // real elems/output = the AccumulateViaAdd mean 1/N
    constexpr uint32_t avg_direct = get_compile_time_arg_val(12);  // 1 = AccumulateViaAdd uses reduce<AVG> direct
    constexpr uint32_t row_pitch = (row_stride > 0u) ? row_stride : Wt;
    constexpr uint32_t in_tiles = Ht * row_pitch * NC;  // resident block incl. padded rows
    constexpr bool pops_input = (policy_id <= 1u);  // Bulk + WaitAndPop pop the input; no-pop policies do not

    using namespace compute_kernel_lib;
    // AccumulateViaAdd partial: 0/1 mask tile at scaler-CB index 0 + valid element count for the mean.
    const ReducePartialScaler PS =
        (partial_elems > 0u) ? ReducePartialScaler::partial_mask(partial_elems, 0) : ReducePartialScaler::none();
    const auto SHAPE = ReduceInputBlockShape::of(Ht, Wt, NC);
    const auto ML = (row_stride > 0u) ? ReduceInputMemoryLayout::with_row_stride(row_stride)
                                      : ReduceInputMemoryLayout::contiguous();

    // Boot init (once, never in the loop). The library reduce() does the light per-call reconfig, so
    // compute_kernel_hw_startup is the same boot for both datapaths (the algo==1 form keeps SrcA=SrcB=cb_in
    // explicit); the heavy hw_configure lives here.
    if constexpr (algo == 1u) {
        compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    } else {
        compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    }

    // No-pop policies (WaitUpfront / NoWait) never pop the input, so arm the resident block ONCE — re-arming
    // it per iter with no matching pop would overflow the in_tiles-deep sharded CB (producer-blocked hang).
    // Pop policies (Bulk / WaitAndPop) re-arm each iter, balanced by reduce()'s pop.
    if constexpr (!pops_input) {
        cb_reserve_back(cb_in, in_tiles);
        cb_push_back(cb_in, in_tiles);
    }
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if constexpr (pops_input) {
            cb_reserve_back(cb_in, in_tiles);
            cb_push_back(cb_in, in_tiles);  // resident sharded input -> re-arm each iter
        }
        do_reduce_block<algo, dim, policy_id, avg_direct>(iter, SHAPE, ML, PS, n_reduced, recfg);
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, out_tiles);
            cb_pop_front(cb_out, out_tiles);
        }
    }
}
"""


# =============================================================================
# Accumulate kernel — the reduce library's cross-call Accumulate over AccumulateViaAdd. The reduce dim is
# split into `num_chunks` chunks; each chunk is one reduce() call that folds the running RAW partial-sum tile
# (in cb_acc) into the pairwise add and, only on the LAST chunk, finalizes (sfpu_reduce) into cb_out. Non-last
# chunks point the output CB at cb_acc so the raw partial is written back for the next reload. SUM pool (the
# accumulate path is SUM-only). To exercise the fold without a chunked reader, each chunk re-arms and reduces
# the SAME resident block, so the result is num_chunks * sum(block, reduce_dim).
# CT args: [Ht, Wt, NC, dim, kernel_iters, out_tiles, num_chunks]
# =============================================================================
_ACCUM_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_acc = 2, cb_out = 16;

// One accumulate chunk (dim-templated to keep the per-dim reduce() calls in one place + thread the partial
// scaler / memory layout cleanly). Non-last chunks SUM the RAW partial into cb_acc; the LAST chunk finalizes
// into cb_out — MEAN via reduce_mean with the GRAND-TOTAL divisor mean_n (spanning all chunks), else SUM.
// A partial (ROW/COL) folds the masked last reduce-dim tile into EACH chunk's sum; row_stride reduces the
// first Wt columns of a wider resident block.
template <uint32_t dim, uint32_t mean_n>
ALWI void do_accum_chunk(
    uint32_t c, bool is_last,
    compute_kernel_lib::ReduceInputBlockShape shape,
    compute_kernel_lib::ReduceInputMemoryLayout ml,
    compute_kernel_lib::ReducePartialScaler ps,
    compute_kernel_lib::AccumulateReloadMode reload) {
    using namespace compute_kernel_lib;
    using ckernel::PoolType;
    using ckernel::ReduceDim;
    constexpr ReduceDim RDIM = (dim == 0u)   ? ReduceDim::REDUCE_ROW
                               : (dim == 1u) ? ReduceDim::REDUCE_COL
                                             : ReduceDim::REDUCE_SCALAR;
    constexpr auto POLICY = ReduceInputPolicy::BulkWaitBulkPop;
    constexpr auto RECFG = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
    constexpr auto ALG = ReduceAlgorithm::AccumulateViaAdd;
    if (is_last) {
        if constexpr (mean_n > 0u)
            reduce_mean<RDIM, cb_in, cb_scaler, cb_out, POLICY, RECFG, ALG>(
                shape, mean_n, ml, Accumulate::at_last(cb_acc, c).with_reload(reload), ps);
        else
            reduce<PoolType::SUM, RDIM, cb_in, cb_scaler, cb_out, POLICY, RECFG, ALG>(
                shape, ml, Accumulate::at_last(cb_acc, c).with_reload(reload), NoOp{}, ps);
    } else {
        reduce<PoolType::SUM, RDIM, cb_in, cb_scaler, cb_acc, POLICY, RECFG, ALG>(
            shape, ml, Accumulate::at(cb_acc, c).with_reload(reload), NoOp{}, ps);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t dim = get_compile_time_arg_val(3);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t mean_n = get_compile_time_arg_val(7);  // 0 = SUM; >0 = MEAN, divide the GRAND total by mean_n
    constexpr uint32_t partial_elems = get_compile_time_arg_val(8);  // valid elems in last reduce-dim tile (0=aligned)
    constexpr uint32_t row_stride = get_compile_time_arg_val(9);  // tile pitch per row (0=contiguous=Wt)
    constexpr uint32_t reload_id = get_compile_time_arg_val(10);  // 0 FoldViaAdd, 1 CopySeedPairs, 2 CopySeedUniform
    constexpr uint32_t row_pitch = (row_stride > 0u) ? row_stride : Wt;
    constexpr uint32_t in_tiles = Ht * row_pitch * NC;

    using namespace compute_kernel_lib;
    constexpr AccumulateReloadMode RELOAD = (reload_id == 0u)   ? AccumulateReloadMode::FoldViaAdd
                                            : (reload_id == 2u) ? AccumulateReloadMode::CopySeedUniform
                                            : (reload_id == 3u) ? AccumulateReloadMode::CopySeedSfpuAdd
                                            : (reload_id == 4u) ? AccumulateReloadMode::CopySeedZeroPair
                                                                : AccumulateReloadMode::CopySeedPairs;
    const ReducePartialScaler PS =
        (partial_elems > 0u) ? ReducePartialScaler::partial_mask(partial_elems, 0) : ReducePartialScaler::none();
    const auto SHAPE = ReduceInputBlockShape::of(Ht, Wt, NC);
    const auto ML = (row_stride > 0u) ? ReduceInputMemoryLayout::with_row_stride(row_stride)
                                      : ReduceInputMemoryLayout::contiguous();

    // AccumulateViaAdd is a binary-add datapath -> boot with SrcA = SrcB = cb_in (once, never in the loop);
    // the reduce() helper only does light per-call (re)config. The packer format is re-adapted per call
    // (RECFG=INPUT_AND_OUTPUT) as the output alternates cb_acc (non-last) / cb_out (last).
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t c = 0; c < num_chunks; ++c) {
            cb_reserve_back(cb_in, in_tiles);
            cb_push_back(cb_in, in_tiles);  // re-arm the resident block for this chunk
            do_accum_chunk<dim, mean_n>(c, (c + 1u == num_chunks), SHAPE, ML, PS, RELOAD);
        }
        if (iter + 1 < kernel_iters) {
            cb_wait_front(cb_out, out_tiles);
            cb_pop_front(cb_out, out_tiles);
        }
    }
}
"""


# =============================================================================
# Scaler dataflow kernel (library paths only) — fills the AVG scaler tile for the dim + reduce factor.
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
# Mask dataflow kernel (accumulate_via_add_inline partial path) — fills a 0/1 mask tile: 1.0 in the first
# `partial_elems` reduce-dim positions, 0 elsewhere. PoolType::SUM makes the scaler value 1.0, so the
# partial-fill helper produces exactly a mask. ROW -> row-0 layout (consumed by mul_tiles_bcast_rows).
# CT args: [dim, partial_elems]
# =============================================================================
_MASK_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t dim = get_compile_time_arg_val(0);
    constexpr uint32_t partial_elems = get_compile_time_arg_val(1);
    using ckernel::ReduceDim;
    {
        DeviceZoneScopedN("mask_fill");  // time the 0/1 mask-tile fill on the DM (reader) core
        if constexpr (dim == 0) {
            dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ReduceDim::REDUCE_ROW>(partial_elems);
        } else {
            dataflow_kernel_lib::prepare_reduce_mask<cb_scaler, ReduceDim::REDUCE_COL>(partial_elems);
        }
    }
}
"""


# =============================================================================
# Zero-tile dataflow kernel (CopySeedZeroPair accumulate) — fills the scaler CB with an all-zero tile that the
# odd-leftover add_tiles pairs with. prepare_reduce_scaler zeroes the whole tile then skips the fill for a 0
# value, so scaler_f=0 yields a genuine all-positions-zero tile.
# =============================================================================
_ZERO_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_scaler = 1;
    using ckernel::PoolType;
    using ckernel::ReduceDim;
    // scaler_f=0 -> prepare_reduce_scaler zeroes the whole tile and skips the fill, yielding an all-zero tile.
    // PoolType / ReduceDim are irrelevant for a zero fill.
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(0.0f, 32);
}
"""


# =============================================================================
# Host-side layout + program descriptor
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


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


def _resolve(variant, dim, Ht, Wt):
    if variant == "dispatch":
        return "accumulate_via_add" if reduced_count(dim, Ht, Wt) >= dispatch_min(dim) else "reduce_tile"
    return variant


def _mean_n(dim, Ht, Wt, partial_elems):
    """N_true = real elements reduced into each output (the mean divisor the CALLER supplies to reduce_mean).
    partial_elems>0 -> the last reduce-dim tile holds that many valid elements, so N_true =
    (reduced_tiles-1)*32 + partial_elems (ROW/COL only); else the tile-aligned count."""
    if partial_elems > 0:
        return (reduced_count(dim, Ht, Wt) - 1) * TILE + partial_elems
    return elements_reduced(dim, Ht, Wt)


def _mean_scaler_bits(dim, Ht, Wt, partial_elems):
    """Float bits of 1/N_true for the mean (used by the inline kernel's own scaler CT arg)."""
    return struct.unpack("<I", struct.pack("<f", 1.0 / _mean_n(dim, Ht, Wt, partial_elems)))[0]


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    dim,
    Ht,
    Wt,
    NC=1,
    accum="fp32",
    kernel_iters=1,
    math_fidelity=None,
    partial_elems=0,
    stream=False,
    policy="bulk",
    avg_direct=False,
    reconfig=None,
    row_stride=0,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if dim not in _DIM_ID:
        raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")
    if min(Ht, Wt, NC) < 1 or kernel_iters < 1:
        raise ValueError("Ht, Wt, NC and kernel_iters must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")
    if partial_elems and (partial_elems < 1 or partial_elems >= TILE):
        raise ValueError(
            f"partial_elems must be in [1, {TILE - 1}] (valid elems in the last tile), got {partial_elems}"
        )
    if stream:  # back-compat alias for the old boolean flag
        policy = "stream"
    if policy not in _POLICY_ID:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    # The library handles all four policies for both datapaths; the example's guards below capture the combos
    # the reduce_block SETUP (one row-major resident shard, single core) can actually feed correctly.
    if policy != "bulk":
        # The inline hand-written kernel is BulkWaitBulkPop-only; policies apply only to the library paths.
        if variant == "accumulate_via_add_inline":
            raise ValueError("policy applies to the library variants only (inline is BulkWaitBulkPop)")
        # A column stream would need tiles delivered in column-major order; the row-major resident shard
        # cannot supply that (both datapaths would read the wrong tiles) — reject for reduce_tile too.
        if policy == "stream" and dim == "col":
            raise ValueError("stream is contiguous-only (row/scalar); col needs a column-ordered reader")
        if policy == "stream" and partial_elems and dim != "row":
            raise ValueError("streaming partial is ROW-only")
        # No-pop policies keep the input resident and bulk-reserve the outputs; exercise them aligned +
        # contiguous + no-accumulate here (the paths compose with partial/row_stride in the library, but the
        # example wiring for those is proven on the bulk policy).
        if policy in _NO_POP and (partial_elems or row_stride):
            raise ValueError("no-pop policies are exercised aligned + contiguous in this example")
    if avg_direct and variant != "accumulate_via_add":
        raise ValueError("avg_direct (reduce<AVG> geometry 1/N) is an accumulate_via_add lever")
    if reconfig not in (None, "none_after_first"):
        raise ValueError(f"reconfig must be None or 'none_after_first', got {reconfig!r}")
    if reconfig and variant != "accumulate_via_add":
        raise ValueError("reconfig (NONE-after-first) needs variant=accumulate_via_add")
    if row_stride:
        # row_stride (a wider resident block, padded rows) is a library AccumulateViaAdd feature: reduce the
        # first Wt columns of each row_stride-wide row. ROW/COL only, indexed (not streaming), no partial.
        if variant != "accumulate_via_add":
            raise ValueError("row_stride needs variant=accumulate_via_add")
        if dim not in ("row", "col"):
            raise ValueError("row_stride is supported for row/col only (scalar walks a 2-D block)")
        if policy == "stream":
            raise ValueError("row_stride is incompatible with stream (a pure contiguous stream)")
        if partial_elems:
            raise ValueError("row_stride + partial not supported")
        if row_stride < Wt:
            raise ValueError(f"row_stride ({row_stride}) must be >= Wt ({Wt})")

    path = _resolve(variant, dim, Ht, Wt)
    dim_id = _DIM_ID[dim]
    fp32_dest = accum == "fp32"
    fidelity = math_fidelity or ttnn.MathFidelity.HiFi4
    out_tiles = out_tile_count(dim, Ht, Wt, NC)

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]

    if path == "accumulate_via_add_inline":
        if partial_elems and dim not in ("row", "col"):
            raise ValueError("partial (non-tile-aligned) reduce is supported for row/col only")
        scaler_bits = _mean_scaler_bits(dim, Ht, Wt, partial_elems)
        compute = ttnn.KernelDescriptor(
            kernel_source=_FAST_BLOCK_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_single_core(),
            compile_time_args=[Ht, Wt, NC, dim_id, kernel_iters, int(fp32_dest), scaler_bits, out_tiles, partial_elems],
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest),
        )
        if not partial_elems:
            return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
        # partial: the last reduce-dim tile is folded in masked -> a 0/1 mask tile in the scaler CB.
        cbs.append(_scratch_cb(CB_SCALER, ttnn.bfloat16))
        mask = ttnn.KernelDescriptor(
            kernel_source=_MASK_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_single_core(),
            compile_time_args=[dim_id, partial_elems],
            runtime_args=[],
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[mask, compute], semaphores=[], cbs=cbs)

    # library paths (reduce_tile = algo 0 -> Auto/ReduceTile matmul-reduce; accumulate_via_add = algo 1 ->
    # the opt-in AccumulateViaAdd). AccumulateViaAdd ignores the AVG scaler for aligned reduces (computes
    # 1/N itself); for a PARTIAL reduce it consumes a 0/1 MASK tile from the scaler CB instead.
    algo = 1 if path == "accumulate_via_add" else 0
    use_mask = bool(partial_elems)
    if partial_elems:
        if path != "accumulate_via_add":
            raise ValueError("partial (non-tile-aligned) reduce needs variant=accumulate_via_add")
        if dim not in ("row", "col"):
            raise ValueError("partial reduce is supported for row/col only")
    cbs.append(_scratch_cb(CB_SCALER, ttnn.bfloat16 if use_mask else _dtype_of(accum)))
    compute = ttnn.KernelDescriptor(
        kernel_source=_HELPER_BLOCK_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            Ht,
            Wt,
            NC,
            dim_id,
            kernel_iters,
            out_tiles,
            algo,
            partial_elems,
            _POLICY_ID[policy],
            int(reconfig == "none_after_first"),
            row_stride,
            _mean_n(dim, Ht, Wt, partial_elems),
            int(avg_direct),
        ],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest),
    )
    reader = ttnn.KernelDescriptor(
        kernel_source=_MASK_KERNEL if use_mask else _SCALER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[dim_id, partial_elems] if use_mask else [dim_id, elements_reduced(dim, Ht, Wt)],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)


def run_op(
    input_tensor,
    *,
    variant,
    dim,
    Ht,
    Wt,
    NC=1,
    accum="fp32",
    kernel_iters=1,
    math_fidelity=None,
    partial_elems=0,
    stream=False,
    policy="bulk",
    avg_direct=False,
    reconfig=None,
    row_stride=0,
):
    out_hw = output_shape(dim, Ht, Wt, NC)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_hw)),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(out_hw),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        variant=variant,
        dim=dim,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        accum=accum,
        kernel_iters=kernel_iters,
        math_fidelity=math_fidelity,
        partial_elems=partial_elems,
        stream=stream,
        policy=policy,
        avg_direct=avg_direct,
        reconfig=reconfig,
        row_stride=row_stride,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)


# =============================================================================
# Cross-call Accumulate (AccumulateViaAdd) — the reduce dim is split into `num_chunks` chunks, each folding a
# running RAW partial-sum tile (cb_acc) into the pairwise add; the last chunk finalizes into cb_out. SUM pool.
# Each chunk re-reduces the SAME resident block, so the result is num_chunks * sum(block, reduce_dim).
# =============================================================================
def create_accumulate_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    dim,
    Ht,
    Wt,
    NC=1,
    accum="fp32",
    kernel_iters=1,
    num_chunks=2,
    mean=False,
    math_fidelity=None,
    partial_elems=0,
    row_stride=0,
    reload="copy_pairs",
    acc_unpack_to_dest=False,
):
    if dim not in _DIM_ID:
        raise ValueError(f"dim must be one of {DIMS}, got {dim!r}")
    if reload not in _RELOAD_ID:
        raise ValueError(f"reload must be one of {RELOADS}, got {reload!r}")
    if acc_unpack_to_dest and accum != "fp32":
        raise ValueError("acc_unpack_to_dest (UnpackToDestFp32 accumulator) requires accum='fp32'")
    if min(Ht, Wt, NC) < 1 or kernel_iters < 1 or num_chunks < 1:
        raise ValueError("Ht, Wt, NC, kernel_iters and num_chunks must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("input must be bfloat16 TILE_LAYOUT")
    if output_tensor.dtype != ttnn.float32 or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("output must be float32 TILE_LAYOUT")
    if partial_elems and (partial_elems < 1 or partial_elems >= TILE):
        raise ValueError(f"partial_elems must be in [1, {TILE - 1}], got {partial_elems}")
    if partial_elems and dim not in ("row", "col"):
        raise ValueError("partial + accumulate is row/col only (scalar corner mask not encodable)")
    if row_stride:
        if dim not in ("row", "col"):
            raise ValueError("row_stride + accumulate is row/col only (scalar walks a 2-D block)")
        if partial_elems:
            raise ValueError("row_stride + partial not wired in the accumulate example")
        if row_stride < Wt:
            raise ValueError(f"row_stride ({row_stride}) must be >= Wt ({Wt})")
    if reload == "copy_zero" and partial_elems:
        raise ValueError("copy_zero uses the scaler CB for the zero tile; incompatible with a partial mask")
    if reload == "fold" and acc_unpack_to_dest:
        raise ValueError(
            "reload='fold' (FoldViaAdd) reads the accumulator via SrcB, disabled for an "
            "UnpackToDestFp32 CB; use a CopySeed* reload (copy_pairs/copy_uniform/copy_sfpu/copy_zero)"
        )

    dim_id = _DIM_ID[dim]
    fp32_dest = accum == "fp32"
    fidelity = math_fidelity or ttnn.MathFidelity.HiFi4
    out_tiles = out_tile_count(dim, Ht, Wt, NC)
    use_mask = bool(partial_elems)
    use_zero = reload == "copy_zero"  # scaler CB holds an all-zero tile (aligned only; guarded above)
    # MEAN over the accumulated chunks: each chunk re-reduces the same block, so the GRAND-total element
    # count is num_chunks * (elements reduced per chunk, incl. the partial count). This is the divisor
    # reduce_mean applies on the last chunk — it must span all chunks, not just the last block's count. 0 => SUM.
    mean_n = num_chunks * _mean_n(dim, Ht, Wt, partial_elems) if mean else 0

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
        # SCALER CB: a 0/1 mask tile (partial) or an all-zero tile (copy_zero), both bf16; else unused.
        _scratch_cb(CB_SCALER, ttnn.bfloat16 if (use_mask or use_zero) else _dtype_of(accum)),
        # Running RAW partial-sum tile per output, at the accumulation (DEST) dtype — fp32 when fp32_dest_acc
        # is on, so the cross-chunk partial keeps full precision. The library reconfigures SRCA/SRCB around
        # each accumulator read (copy_tile / add_tiles do NOT), so the accumulator CB may differ in format
        # from the bf16 input (mirrors the standard reload_accumulator_if_needed).
        _scratch_cb(CB_ACC, _dtype_of(accum), num=out_tiles),
    ]
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest)
    if acc_unpack_to_dest:
        # Tag the accumulator CB UnpackToDestFp32 -> lossless fp32 reload via copy_tile, and SrcA/B access
        # DISABLED for it (so FoldViaAdd, which reads it via SrcB, is incorrect here; the CopySeed* reloads
        # are the sanctioned access). Per-CB vector indexed by CB id; only CB_ACC is tagged.
        modes = [ttnn.UnpackToDestMode.Default] * 64  # one entry per circular buffer (NUM_CIRCULAR_BUFFERS)
        modes[CB_ACC] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source=_ACCUM_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            Ht,
            Wt,
            NC,
            dim_id,
            kernel_iters,
            out_tiles,
            num_chunks,
            mean_n,
            partial_elems,
            row_stride,
            _RELOAD_ID[reload],
        ],
        config=compute_cfg,
    )
    kernels = [compute]
    if use_mask:
        # 0/1 mask tile the partial fold multiplies the last reduce-dim tile by (padding -> 0).
        kernels.insert(
            0,
            ttnn.KernelDescriptor(
                kernel_source=_MASK_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_single_core(),
                compile_time_args=[dim_id, partial_elems],
                runtime_args=[],
                config=ttnn.ReaderConfigDescriptor(),
            ),
        )
    elif use_zero:
        # All-zero scaler tile the CopySeedZeroPair odd-leftover add_tiles pairs with.
        kernels.insert(
            0,
            ttnn.KernelDescriptor(
                kernel_source=_ZERO_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_single_core(),
                compile_time_args=[],
                runtime_args=[],
                config=ttnn.ReaderConfigDescriptor(),
            ),
        )
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def run_accumulate(
    input_tensor,
    *,
    dim,
    Ht,
    Wt,
    NC=1,
    accum="fp32",
    kernel_iters=1,
    num_chunks=2,
    mean=False,
    math_fidelity=None,
    partial_elems=0,
    row_stride=0,
    reload="copy_pairs",
    acc_unpack_to_dest=False,
):
    out_hw = output_shape(dim, Ht, Wt, NC)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(out_hw)),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(out_hw),
    )
    descriptor = create_accumulate_program_descriptor(
        input_tensor,
        output,
        dim=dim,
        Ht=Ht,
        Wt=Wt,
        NC=NC,
        accum=accum,
        kernel_iters=kernel_iters,
        num_chunks=num_chunks,
        mean=mean,
        math_fidelity=math_fidelity,
        partial_elems=partial_elems,
        row_stride=row_stride,
        reload=reload,
        acc_unpack_to_dest=acc_unpack_to_dest,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
