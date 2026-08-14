// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute.  Realizes op_design.md's named block operations:
//
//   tilize_block            -> compute_kernel_lib::tilize            (ROW_MAJOR only)
//   mask_tail_block         -> eltwise_chain (BinaryFpu Mul, bcast Row, strided, in place)
//   square_accumulate_block -> compute_kernel_lib::sum_of_squares
//   collapse_partial_block  -> compute_kernel_lib::reduce<SUM, REDUCE_ROW> (s == 1)
//   combine_block (root)    -> compute_kernel_lib::reduce<SUM, REDUCE_ROW> over (B, s)
//   scale_block             -> compute_kernel_lib::mul (bcast Col)
//   apply_gamma_block       -> compute_kernel_lib::mul (bcast Row)
//   untilize_block          -> compute_kernel_lib::untilize          (ROW_MAJOR only)
//
// Raw-LLK notes (deviations from "prefer helpers"):
//  * The finalize (x1/W, +epsilon, rsqrt) is written as raw SFPU calls inside a
//    `post_reduce_op` lambda.  That lambda IS the reduce helper's documented
//    extension point (reduce_helpers_compute.hpp:491-495): running the three ops
//    on the reduce's DEST tile is what keeps `mean+eps` from ever being packed
//    to L1.  Expressing it as a separate eltwise_chain would need a whole extra
//    B-page CB and an L1 round trip.
//  * `mask_tail_block` uses `eltwise_chain` rather than the `mul` convenience
//    wrapper because the convenience wrappers default-construct their elements
//    and so cannot carry a `StridedTileRange` (the strided, in-place window over
//    one tile per tile-row).  Same helper, same chain, explicit element ctors.
//
// x lives in cb_input_tiles across THREE phases and is rewritten in place twice.
// The compute kernel owns exactly one cb_wait_front / cb_pop_front window per
// block; every chain that touches it uses WaitPolicy::None / PopPolicy::None /
// ReservePolicy::None / PushPolicy::None so no helper issues a competing
// handshake.  cb_input_tiles' capacity is exactly BLOCK_ROWS*S, which is what
// makes get_write_ptr() == get_read_ptr() and the in-place pack correct.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;
constexpr uint32_t cb_w_mask = 8;
constexpr uint32_t cb_output_tiles = 9;
constexpr uint32_t cb_rm_stage_in = 10;
constexpr uint32_t cb_rm_stage_out = 11;
constexpr uint32_t cb_thread_sync = 12;

constexpr uint32_t NO_MASK_COL = 0xFFFFFFFFu;

// Largest d <= cap that divides `width`.  See DEST_BLOCK below.
constexpr uint32_t dest_block_divisor(uint32_t width, uint32_t cap) {
    for (uint32_t d = (cap < width ? cap : width); d > 1; --d) {
        if (width % d == 0) {
            return d;
        }
    }
    return 1;
}

// PACK -> UNPACK ordering edge for an in-place handoff.
//
// Two consecutive chains that both address cb_input_tiles with caller-managed
// (None, None) policies exchange NO CB handshake, so nothing orders chain N's
// pack against chain N+1's unpack of the same tile.  The dst-sync window bounds
// the skew to a couple of tiles, which is why the bug only bites when a block is
// 1-2 tiles wide -- exactly the narrow-W / single-tile-row shapes.
//
// cb_reserve_back / cb_push_back compile to PACK-only ops and cb_wait_front /
// cb_pop_front to UNPACK-only ops (api/compute/cb_api.h:44-136), so one push/wait
// round trip on a private 1-page CB is precisely the missing edge -- and it moves
// no data.  This is a synchronization op BETWEEN helper calls, never around one.
ALWI void sync_pack_to_unpack() {
    cb_reserve_back(cb_thread_sync, 1);
    cb_push_back(cb_thread_sync, 1);
    cb_wait_front(cb_thread_sync, 1);
    cb_pop_front(cb_thread_sync, 1);
}

void kernel_main() {
    // ---- block knobs ----
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);   // s
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(3);
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(4);
    constexpr uint32_t MASK_ENABLED = get_compile_time_arg_val(5);
    // Pages held in cb_input_tiles at once.  One block everywhere except the
    // TILE + sharded path, where the CB is bound to the caller's WHOLE resident
    // shard: there the reader keeps it exactly full at every block boundary, so
    // waiting the full window is what makes get_write_ptr() == get_read_ptr()
    // (and therefore the in-place rewrite of x) correct for block_rows < shard_rows.
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(6);
    // cb_input_tiles' whole CAPACITY in pages — what the in-place pack index is
    // taken modulo.  Equal to IN_WAIT_TILES on every path where the CB holds
    // exactly the live window; larger when the CB is a resident shard (Refinement
    // 2) or a double-buffered input (IN_CB_DEPTH > 1).  Always a whole multiple of
    // BLOCK_TILES, so the read window never wraps mid-block.
    constexpr uint32_t IN_CAPACITY_TILES = get_compile_time_arg_val(7);
    // Fuse `scale_block` and `apply_gamma_block` into ONE pass over the block.
    //
    // Unfused, x*rsqrt is packed to L1 and unpacked again purely to meet gamma:
    // two FPU ops but FOUR L1 crossings per tile, plus a PACK->UNPACK sync per
    // block.  Fused, the gamma multiply happens on the DEST tile the scale just
    // produced (`DestReuseBinary`), so the block costs two FPU ops and TWO
    // crossings.  The price is that DEST-reuse has no intra-tile broadcast, so
    // gamma must be a FULL tile: the row-0-only vector is expanded once, in place,
    // by a `UnaryBcast<Row>` pass over its S tiles (paid once per kernel, against
    // a saving of B*S packs + B*S unpacks per BLOCK).  The host therefore turns
    // this on only where a core owns more than one tile-row — see
    // GAMMA_FUSE_MIN_ROW_TILES; the single-tile-row decode regime stays
    // byte-identical to Refinement 3/4.
    constexpr uint32_t GAMMA_FUSED = get_compile_time_arg_val(8);
    // Tiles per DEST window on the streaming eltwise passes (scale / apply_gamma).
    // At 1 every tile pays a full tile_regs acquire/commit/wait/release round trip,
    // which serializes MATH against PACK; batching lets the two pipeline.  The chain
    // clamps this to DEST_AUTO_LIMIT at runtime, so an oversized value only costs
    // extra outer iterations.
    constexpr uint32_t DEST_BLOCK_TILES = get_compile_time_arg_val(9);

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;

    // Which datapath the root's cross-core combine reduce uses — a DISPATCH on
    // the reduce width, not a replacement (examples/master.md, `reduce_accumulate`
    // / `reduce_block`).  Pairwise `add_tiles(acc_to_dest)` + one within-tile
    // finalize beats the matmul-with-ones `reduce_tile` datapath once there are
    // enough tiles to amortize it (REDUCE_ROW crosses over at ~4); BELOW the
    // crossover the single matmul-reduce is faster, so dispatching this way is
    // never slower than the library.
    //
    // Both sides are measured here.  Above the crossover the win is a NULL on
    // this op — s = 32 costs 9770 ns vs 9732 ns for ReduceTile, inside noise —
    // which is master.md's own caveat ("the win is compute-only; a
    // data-movement-bound reduce won't show it") and is itself the useful
    // finding: the root's combine cost is the gather INCAST (s stat tiles
    // converging on one core), not the reduce math.  BELOW the crossover the
    // penalty is real and was measured as a regression (the prefill geometry
    // lands at s = 2 and paid ~2.5% for the accumulate datapath), which is
    // exactly why the crossover is honored instead of hardcoding one algorithm.
    constexpr uint32_t COMBINE_ACCUMULATE_MIN_TILES = 4;
    constexpr auto COMBINE_ALGORITHM = NUM_HIDDEN_SLICES >= COMBINE_ACCUMULATE_MIN_TILES
                                           ? ckl::ReduceAlgorithm::AccumulateViaAdd
                                           : ckl::ReduceAlgorithm::Auto;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t mask_local_col = get_arg_val<uint32_t>(2);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(3);
    const uint32_t eps_bits = get_arg_val<uint32_t>(4);

    if constexpr (IS_ROW_MAJOR) {
        compute_kernel_hw_startup(cb_rm_stage_in, cb_scaler, cb_input_tiles);
    } else {
        compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);
    }

    // Resident constants: waited once, never popped.
    //
    // gamma is deliberately NOT waited here.  It is the only DRAM tensor on the
    // sharded path, so a boot-time wait puts a full DRAM round trip in front of
    // Sum(x^2) and the cross-core combine on every core — for an operand the
    // kernel does not touch until `apply_gamma_block`.  The wait therefore lives
    // at the apply site (the reader defers its barrier to match); on every block
    // after the first the CB is already full, so it costs a few cycles.
    const bool do_mask = (MASK_ENABLED != 0) && (mask_local_col != NO_MASK_COL);
    if (do_mask) {
        cb_wait_front(cb_w_mask, 1);
    }

    // finalize: mean = Sum(x^2) * (1/W) using the TRUE element count W, then
    // + epsilon, then rsqrt — applied exactly ONCE, after the cross-core combine.
    // ---- operand configurations (compile-time; every chain that touches
    //      cb_input_tiles is caller-managed so no helper competes for its
    //      wait/pop/reserve/push window) ----
    constexpr auto x_held =
        ckl::input(cb_input_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto rms_col = ckl::input(
        cb_rms_recip,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::AtEnd,
        ckl::OperandKind::Col,
        ckl::TileOffset::Unset);
    constexpr auto gamma_row = ckl::input(
        cb_gamma_tiles,
        ckl::BroadcastDim::Row,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Row,
        ckl::TileOffset::Unset);
    // The fused path's gamma operand: a FULL tile (no intra-tile broadcast — that
    // is what DEST-reuse cannot do), still indexed per hidden column (Row).
    constexpr auto gamma_full = ckl::input(
        cb_gamma_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row, ckl::TileOffset::Unset);
    // ...and the in-place expansion that produces it, once: read tile i with a Row
    // broadcast, write tile i back full.  Legal in place per tile — the chain's
    // unpack of tile i completes inside the dst-sync window before its pack.
    constexpr auto gamma_expand_in = ckl::input(
        cb_gamma_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block, ckl::TileOffset::Unset);
    constexpr auto gamma_expand_out =
        ckl::output(cb_gamma_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Set);
    // In-place pack window.  TileOffset::Set (base 0) is REQUIRED, not cosmetic:
    // with TileOffset::Unset the chain emits `pack_tile<out_of_order_output=false>`,
    // whose LLK path derives the write address from an internal running counter and
    // ignores the tile index.  That counter is only rewound by a pack reconfig — and
    // the chain's reconfig fold ELIDES the reconfig when two consecutive chains pack
    // to the same CB.  So a second in-place chain (scale, then apply_gamma on the
    // ROW_MAJOR path) would keep counting past the block and silently drop its
    // result.  `Set` selects `pack_tile<true>`, which honours base + i_flat.
    constexpr auto in_place =
        ckl::output(cb_input_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Set);
    // Per-tile reserve/push: the chain reserves (PerOuter, PerOuter) exclusively
    // for DEST-accumulating packs, so the streaming output CB uses the per-tile
    // lifecycle.  The writer still drains a whole tile-row (S pages) at a time,
    // so cb_output_tiles' out_cb_depth window is what buys the overlap.
    constexpr auto to_output = ckl::output(cb_output_tiles);
    // The batched twin: a DEST window of DEST_BLOCK tiles must synchronize the
    // output CB once per GROUP, not once per tile — a per-tile lifecycle inside a
    // grouped walk under-pushes and the writer hangs waiting for the rest of the
    // tile-row (measured).  Mirrors kernel_lib's own block_size example.
    constexpr auto to_output_batched =
        ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    constexpr auto block_shape = ckl::IterationShape::grid(BLOCK_ROWS, SLICE_HIDDEN_TILES);
    // Same walk, batched into DEST windows.  The grid walk blocks ROW-WISE, so the
    // group must divide the row width: a block wider than S leaves the row's tail
    // group short and the per-tile output lifecycle never completes it (measured:
    // the writer hangs in cb_wait_front).  Taking the largest divisor of S that
    // fits the knob keeps every group full and needs no tail contract.
    constexpr uint32_t DEST_BLOCK = dest_block_divisor(SLICE_HIDDEN_TILES, DEST_BLOCK_TILES);
    constexpr auto block_shape_batched =
        ckl::IterationShape::grid(BLOCK_ROWS, SLICE_HIDDEN_TILES).block_size(DEST_BLOCK);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Where an IN-PLACE pack lands, measured from cb_input_tiles' BASE.
        //
        // A pack address is `get_write_ptr(cb) + tile_index * page`, and only
        // cb_reserve_back / cb_push_back move a *consumer's* write pointer —
        // compute never pushes to cb_input_tiles, so its write pointer sits at the
        // CB base for the kernel's life while `cb_pop_front` walks the read
        // pointer forward.  When the CB's capacity is exactly one block
        // (interleaved, and TILE+sharded at the default block_rows == shard_rows)
        // the read pointer wraps back to base every block and this is 0.  When the
        // CB is bound to a whole resident shard, or double-buffered (IN_CB_DEPTH
        // > 1), it is the block's page offset into that capacity — without it
        // every block after the first would rewrite block 0's pages and silently
        // drop its own scale factor.
        const uint32_t pack_base = (block * BLOCK_TILES) % IN_CAPACITY_TILES;
        // ---- tilize_block (ROW_MAJOR only) ----
        if constexpr (IS_ROW_MAJOR) {
            ckl::tilize<SLICE_HIDDEN_TILES, cb_rm_stage_in, cb_input_tiles>(BLOCK_ROWS);
        }

        // ---- the single cb_input_tiles window for this block ----
        cb_wait_front(cb_input_tiles, IN_WAIT_TILES);

        // ---- mask_tail_block: zero the W-pad lanes of the LAST hidden tile of
        //      each tile-row, in place. Only on the core owning the global last
        //      hidden tile, and only when W % 32 != 0 under TILE layout.
        if constexpr (MASK_ENABLED) {
            if (do_mask) {
                // Read window is relative to the READ pointer (which pops walk
                // forward); the pack window is relative to the CB base.
                const ckl::StridedTileRange window{mask_local_col, SLICE_HIDDEN_TILES};
                const ckl::StridedTileRange pack_window{pack_base + mask_local_col, SLICE_HIDDEN_TILES};
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(BLOCK_ROWS, 1),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            cb_input_tiles,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Col,
                            ckl::TileOffset::Strided),
                        ckl::input(cb_w_mask, ckl::BroadcastDim::Row, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{
                        window},
                    ckl::PackTile<ckl::output(
                        cb_input_tiles, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Strided)>{
                        pack_window});
                sync_pack_to_unpack();  // mask packed x in place; Sum(x^2) unpacks it next
            }
        }

        // ---- square_accumulate_block: Sum over the slice of x*x, folded in DEST
        //      per tile-row (no x^2 tiles are ever materialized). x is HELD.
        ckl::sum_of_squares<x_held, ckl::row_output(cb_sq_partials)>(block_shape);

        if constexpr (NUM_HIDDEN_SLICES > 1) {
            // ---- combine_block: `collapse_partial_block` is FUSED INTO the root's
            //      combine rather than run per contributor.
            //
            //      Design deviation, recorded: op_design.md routes each slice
            //      through its own within-tile collapse (-> cb_slice_stat,
            //      column-0-valid) and asks the root to sum those with
            //      ReduceWithinTile::Skip.  That template value is currently
            //      UNREACHABLE through compute_kernel_lib::reduce(): the
            //      "Skip is AccumulateViaAdd-only" static_assert
            //      (reduce_helpers_compute.inl:886-891) sits AFTER the
            //      `if constexpr (AccumulateViaAdd) { ... return; }` block, so it
            //      is not in a discarded statement and fires for the
            //      AccumulateViaAdd instantiation too.
            //
            //      The equivalent that stays on a supported path: ship the RAW
            //      per-column partial tile (cb_sq_partials, all 32 columns carry a
            //      partial sum) and let the root do ONE reduce over the (B, s)
            //      gathered block.  Sum-over-contributors then collapse-within-tile
            //      == collapse then sum, so the arithmetic is identical, the NoC
            //      payload is the same B tiles per contributor, and one whole
            //      compute phase disappears from every non-root core.
            //      cb_slice_stat is therefore not created at all; cb_sq_partials'
            //      single consumer is the writer.
            if (is_root) {
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_gathered_partials,
                    cb_scaler,
                    cb_rms_bcast,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    COMBINE_ALGORITHM,
                    ckl::NoAccumulation,
                    decltype(finalize)>(
                    ckl::ReduceInputBlockShape::of(BLOCK_ROWS, NUM_HIDDEN_SLICES),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    finalize);
            }
        } else {
            // ---- collapse_partial_block with the finalize fused in ----
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sq_partials,
                cb_scaler,
                cb_rms_recip,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                ckl::ReduceAlgorithm::Auto,
                ckl::NoAccumulation,
                decltype(finalize)>(
                ckl::ReduceInputBlockShape::of(BLOCK_ROWS, 1),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                finalize);
        }

        // ---- scale_block (+ apply_gamma_block, fused): x *= rsqrt(mean + eps),
        //      then *= gamma.  A REDUCE_ROW result is column-shaped, so it
        //      broadcasts back across columns (Col); gamma is a 1D [W] operand and
        //      so is indexed per hidden column (Row).
        if constexpr (HAS_GAMMA && GAMMA_FUSED) {
            cb_wait_front(cb_gamma_tiles, SLICE_HIDDEN_TILES);  // deferred; see boot
            if (block == 0) {
                // Expand gamma from its row-0 vector form to full tiles, once.
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(SLICE_HIDDEN_TILES),
                    ckl::UnaryBcast<ckl::BroadcastDim::Row, gamma_expand_in>{},
                    ckl::PackTile<gamma_expand_out>{0});
                sync_pack_to_unpack();  // gamma packed in place; the chain below unpacks it
            }
            if constexpr (IS_ROW_MAJOR) {
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                    ckl::DestReuseBinary<gamma_full, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>{},
                    ckl::PackTile<in_place>{pack_base});
                sync_pack_to_unpack();  // x*r*gamma packed in place; untilize unpacks it
            } else {
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                    ckl::DestReuseBinary<gamma_full, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>{},
                    ckl::PackTile<to_output_batched>{});
            }
        } else if constexpr (HAS_GAMMA || IS_ROW_MAJOR) {
            // Spelled as an explicit chain rather than `ckl::mul<...>` only because
            // the convenience wrappers default-construct their elements and so
            // cannot carry the runtime `pack_base`.  Same helper, same chain.
            ckl::eltwise_chain(
                block_shape_batched,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                ckl::PackTile<in_place>{pack_base});
            sync_pack_to_unpack();  // x*r packed in place; the next stage unpacks it
        } else {
            ckl::mul<x_held, rms_col, to_output_batched>(block_shape_batched);
        }

        // ---- apply_gamma_block: gamma is a 1D [W] operand -> Row broadcast ----
        //      (unfused path only; the fused one did it above in the same DEST
        //      window, so nothing is packed between the two multiplies).
        if constexpr (HAS_GAMMA && !GAMMA_FUSED) {
            // The deferred gamma wait (see the boot comment).  Never popped, so
            // only the first block can actually block here.
            cb_wait_front(cb_gamma_tiles, SLICE_HIDDEN_TILES);
            if constexpr (IS_ROW_MAJOR) {
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, gamma_row>{},
                    ckl::PackTile<in_place>{pack_base});
                sync_pack_to_unpack();  // x*r*gamma packed in place; untilize unpacks it
            } else {
                ckl::mul<x_held, gamma_row, to_output_batched>(block_shape_batched);
            }
        }

        // ---- untilize_block (ROW_MAJOR) / release the window (TILE) ----
        if constexpr (IS_ROW_MAJOR) {
            // NoWait: compute already holds the BLOCK_TILES window; untilize's
            // per-tile-row pop IS the window's release.
            ckl::untilize<
                SLICE_HIDDEN_TILES,
                cb_input_tiles,
                cb_rm_stage_out,
                ckl::untilize_config::InitUninitMode::InitAndUninit,
                ckl::untilize_config::WaitMode::NoWait>(BLOCK_ROWS);
        } else {
            cb_pop_front(cb_input_tiles, BLOCK_TILES);
        }
    }
}
