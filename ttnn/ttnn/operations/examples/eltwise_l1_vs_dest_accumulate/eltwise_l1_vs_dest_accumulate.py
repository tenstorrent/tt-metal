# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core compute-only benchmark: where the running accumulator lives.

A running accumulator `acc` collects a SUM over a stream of `B` tiles. Addition can happen at THREE
distinct points in the compute pipeline, and a running sum can be built at any of them:
  - the FPU adder      — `add_tiles(A, B)` computes `A + B` into DEST (in fact any FPU binary op);
  - the DEST accumulator — `acc_to_dest` folds each FPU result into a held DEST tile
                           (`DEST += FPU_op(in1, in2)`), with no repack between adds;
  - the L1 accumulator  — `pack_reconfig_l1_acc` folds the packed DEST onto the resident L1 tile
                           (`L1 += <whatever DEST holds>`) at pack time.
What you accumulate is incidental to the mechanism; this example uses a plain per-tile add to isolate
the one thing being measured: where the running accumulator lives and how much L1 traffic it pays.
Three variants combine these adders differently, each stripping more accumulator traffic (all fp32,
identical correct result):

  rmw (baseline): the L1 -> DEST -> L1 round-trip. One streaming `ckl::add<cb_out, cb_in, cb_out>`
                  over B-1 tiles — each iteration unpacks `acc` AND the next tile into DEST, adds on
                  the math engine, packs the result back to `acc`. `acc` is unpacked AND packed every
                  tile; the per-iteration cb_out push/wait drives the round-trip and syncs PACK->UNPACK.

  pack_l1_acc: read TWO tiles per step. One BinaryFpu puts X[2k]+X[2k+1] in DEST (caller-managed
                  input so one add reads two distinct tiles of the same CB via per-operand TileOffset),
                  then the PACK engine folds that onto `acc` in place (L1-accumulation). `acc` is only
                  packed, never unpacked — B/2 steps. Binary init hoisted once (SetupOwner::Caller).

  dest_acc: keep the running sum in a sticky DEST tile for the whole reduction (DestAccumulation),
                  packing `acc` to L1 exactly once at the end. `acc` never touches L1 mid-reduction.
                  Upper bound — only valid when DEST is free (a real loop usually needs DEST for the
                  per-step work, which is why `acc` lives in L1 to begin with).

Everything is sharded in L1 on a single Tensix core (no DRAM in the fast path) and the resident input
is re-exposed block-by-block, so the measured delta is pure compute: the per-step accumulator handling.
See README.md.
"""

import ttnn

TILE = 32
CB_IN = 0  # input blocks, sharded L1 (resident): B tiles, block b at index b
CB_OUT = 16  # running accumulator, sharded L1 (fp32), a single tile

# Single L1 accumulator tile. The helper's L1Accumulation lifecycle folds the whole B-tile stream
# onto one output tile, so each block is exactly one tile (there is no tiles-per-block dimension).
W_TILES = 1

# Baseline first: rmw does the L1<->DEST round-trip; pack_l1_acc lets the packer fold each block
# onto the L1 buffer in place. Both go through the eltwise_chain helper.
VARIANTS = ("rmw", "pack_l1_acc", "dest_acc")
_METHOD = {"rmw": 0, "pack_l1_acc": 1, "dest_acc": 2}

_ACCUMULATE_KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

// acc = sum over B input tiles, into a single L1 accumulator tile (the accumulator lives in L1, not
// DEST). method 0 = the L1 -> DEST -> L1 round-trip via the eltwise_chain helper (ckl::add reads acc
// back from L1, adds one tile, packs back — one tile per step, B steps, acc unpacked every step).
// method 1 = read TWO tiles per step: one BinaryFpu add puts X[2k]+X[2k+1] in DEST, then the packer
// folds that onto the running acc in L1 (pack_reconfig_l1_acc) — B/2 steps, acc never unpacked.
void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t method = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in = 0, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out, cb_out);
    if constexpr (method == 1) {
        // Hoist the binary add init once (the pairwise chains below run SetupOwner::Caller).
        add_tiles_init(cb_in, cb_in);
    } else if constexpr (method == 2) {
        // Same hoist, acc_to_dest=true: the DEST-accumulating add folds each pair into sticky D0.
        add_tiles_init(cb_in, cb_in, true);
    }

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // (Re-)expose the resident sharded input as a B-tile stream for this pass.
        cb_reserve_back(cb_in, B); cb_push_back(cb_in, B);

        if constexpr (method == 0) {
            // ---- L1 <-> DEST round-trip via the eltwise_chain helper ----
            // A single streaming add folds X[1..B-1] onto acc: each iteration unpacks acc AND the
            // next block into DEST, adds, and packs back to acc. acc round-trips L1 every step (the
            // per-iteration cb_out push/wait both drives the round-trip and syncs PACK->UNPACK); the
            // chain retains nothing in DEST between steps. cb_out is one page (operand A and output).
            ckl::copy<cb_in, cb_out>(ckl::EltwiseShape::single());              // acc = X[0]
            ckl::add<cb_out, cb_in, cb_out>(ckl::EltwiseShape::tiles(B - 1));   // acc += X[1..B-1]
        } else if constexpr (method == 1) {
            // ---- read two tiles per step: DEST = X[2k] + X[2k+1], then L1-accumulate onto acc ----
            // The input is caller-managed (one wait/pop for the whole resident stream) so each add
            // can read two distinct tiles of the same CB: srcA = tile 2k, srcB = tile 2k+1, via
            // per-operand TileOffset + OperandKind::Scalar. The packer folds DEST onto the running
            // acc in L1 in place, so acc is never unpacked. B/2 steps; acc reserved/published once.
            constexpr uint32_t N = B / 2;
            cb_wait_front(cb_in, B);      // caller-managed input: wait for the resident stream once
            cb_reserve_back(cb_out, 1);   // caller-managed acc: reserve the single accumulator tile
            using AddPair = ckl::BinaryFpu<
                cb_in, cb_in, ckl::BinaryFpuOp::Add, ckl::BroadcastDim::None,
                ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                ckl::BinaryDataFormatReconfig::None, ckl::Dst::D0,
                ckl::OperandKind::Scalar, ckl::OperandKind::Scalar,
                ckl::TileOffset::Set, ckl::TileOffset::Set>;
            // step 0: seed acc = X[0] + X[1] (plain pack, l1-acc off).
            ckl::eltwise_chain<ckl::SetupOwner::Caller>(
                ckl::EltwiseShape::single(),
                AddPair{0, 1},
                ckl::PackTile<cb_out, ckl::OutputLifecycle::CallerManaged, ckl::PackTileReconfig::None,
                              ckl::Dst::D0>{});
            // steps 1..N-1: acc += X[2k] + X[2k+1] (packer folds DEST onto acc in place).
            for (uint32_t k = 1; k < N; ++k) {
                ckl::eltwise_chain<ckl::SetupOwner::Caller>(
                    ckl::EltwiseShape::single(),
                    AddPair{2 * k, 2 * k + 1},
                    ckl::PackTile<cb_out, ckl::OutputLifecycle::L1AccumulationCallerManaged,
                                  ckl::PackTileReconfig::None, ckl::Dst::D0, ckl::TileOffset::Unset,
                                  ckl::PackTileL1Accumulation::Enabled>{});
            }
            cb_push_back(cb_out, 1);  // publish acc
            cb_pop_front(cb_in, B);   // release the resident input for this pass
        } else {
            // ---- accumulate the running sum in DEST, save to L1 only at the end ----
            // One DestAccumulation chain folds every pair into a single sticky DEST tile
            // (D0 += X[i] + X[i+N]) and packs the final acc to L1 exactly once at chain exit —
            // no pack l1-acc, and acc never round-trips L1 mid-reduction. Because the sticky DEST
            // must live across all pairs, this is one chain (one DEST acquire), so the two operands
            // are the two halves of the stream (srcA = tile i, srcB = tile i+N) rather than the
            // manual adjacent-pair loop above. Only valid because DEST is free the whole reduction.
            constexpr uint32_t N = B / 2;
            cb_wait_front(cb_in, B);  // caller-managed input: wait for the resident stream once
            ckl::eltwise_chain<ckl::SetupOwner::Caller>(
                ckl::EltwiseShape::tiles(N),
                ckl::BinaryFpu<cb_in, cb_in, ckl::BinaryFpuOp::Add, ckl::BroadcastDim::None,
                               ckl::InputLifecycle::CallerManaged, ckl::InputLifecycle::CallerManaged,
                               ckl::BinaryDataFormatReconfig::None, ckl::Dst::D0,
                               ckl::OperandKind::Block, ckl::OperandKind::Block,
                               ckl::TileOffset::Unset, ckl::TileOffset::Set,
                               ckl::DestAccumulation::Enabled>{0, N},
                ckl::PackTile<cb_out, ckl::OutputLifecycle::DestAccumulation, ckl::PackTileReconfig::None,
                              ckl::Dst::D0>{});
            cb_pop_front(cb_in, B);   // release the resident input for this pass
        }
        // Drain acc between in-kernel iterations so the next pass starts clean.
        if (iter + 1 < kernel_iters) { cb_wait_front(cb_out, 1); cb_pop_front(cb_out, 1); }
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config(h_tiles, w_tiles):
    """The whole [h_tiles x w_tiles] tile matrix as one shard on a single core (tiles row-major)."""
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def validate(x, variant):
    if variant not in VARIANTS:
        raise ValueError(f"eltwise_l1_vs_dest_accumulate: variant must be one of {VARIANTS}, got {variant!r}")
    if x.layout != ttnn.TILE_LAYOUT or x.dtype != ttnn.float32:
        raise ValueError("eltwise_l1_vs_dest_accumulate: input must be float32 TILE_LAYOUT (fp32 accumulation)")


def create_program_descriptor(x, acc, *, variant, num_blocks, kernel_iters):
    validate(x, variant)
    if num_blocks % 2 != 0:
        raise ValueError(
            "eltwise_l1_vs_dest_accumulate: num_blocks must be even (pack_l1_acc reads two tiles per step)"
        )
    compute = ttnn.KernelDescriptor(
        kernel_source=_ACCUMULATE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[num_blocks, _METHOD[variant], kernel_iters],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, acc),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def eltwise_l1_vs_dest_accumulate(x, *, variant="pack_l1_acc", num_blocks, kernel_iters=1):
    """acc = sum of `num_blocks` blocks (each a single tile) of the sharded input `x`
    (shape [num_blocks*32, 32]). Output is the identical, correct sum for both variants;
    only the accumulate mechanism differs."""
    if kernel_iters < 1:
        raise ValueError("eltwise_l1_vs_dest_accumulate: kernel_iters must be >= 1")
    validate(x, variant)
    acc = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, W_TILES * TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        x.device(),
        create_sharded_memory_config(1, W_TILES),
    )
    descriptor = create_program_descriptor(x, acc, variant=variant, num_blocks=num_blocks, kernel_iters=kernel_iters)
    return ttnn.generic_op([x, acc], descriptor)
