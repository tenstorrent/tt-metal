// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file accumulate_helpers_compute.hpp
 * @brief Elementwise accumulation of whole tile-blocks ACROSS circular buffers — the compute-side
 *        primitive every reduction collective's compute kernel is built from.
 *
 * @par WHY THIS IS NOT reduce_helpers_compute.hpp.
 *   That header reduces WITHIN a tensor along a dimension (@c reduce_tile over ROW/COL/SCALAR). This
 *   one adds whole tile-blocks TOGETHER across separate CBs — @c out[i] = a[i] + b[i]. Different LLK
 *   op, different shape, no overlap. Nothing here is multi-device: a CCL reduce kernel's arithmetic
 *   is plain single-device eltwise-add, and the collective-ness lives entirely in the schedule
 *   (@c ttnn/cpp/ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp) and the fabric egress
 *   (@c ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp). It lives in the shared kernel_lib rather
 *   than under ccl/ for exactly that reason — the reduction collectives are the driving consumer, not
 *   the only legitimate one.
 *
 * @par WHAT IT REMOVES.
 *   Eight of the nine reduction-collective compute kernels open-code the same ~40 lines: two
 *   @c cb_wait_front, @c tile_regs_acquire, an @c add_tiles loop, @c tile_regs_commit, two
 *   @c cb_pop_front, @c cb_reserve_back, @c tile_regs_wait, a @c pack_tile loop,
 *   @c tile_regs_release, @c cb_push_back — around roughly fifteen lines of actual arithmetic. Along
 *   the way they disagree on four things that are not stylistic:
 *
 *   1. THE GRANULARITY/ACTUAL SPLIT. The CB protocol runs at @c granularity (what the producer
 *      pushes and the consumer expects) while the math covers @c num_tiles, which can be smaller —
 *      the last chunk of a slice is short. Passing one number for both is a deadlock: wait for a
 *      count the producer never pushes, or pop a count that strands tiles. This API therefore takes
 *      @c granularity ONCE at arm time (it is invariant) and @c num_tiles per run, so the two can
 *      never be conflated by accident.
 *   2. DST CAPACITY. @c num_tiles must fit the DEST register, because a run is one
 *      acquire/commit/pack pass. Every shipped CCL reduce relies on the host clamping granularity to
 *      DST capacity and nothing in-kernel checks it (and the host derives that clamp in two places,
 *      inconsistently: @c reduce_scatter_minimal_async uses @c fp32_dest_acc_en ? 4 : 8, while
 *      @c strided_reduce_scatter_async hardcodes 8). @c arm() asserts it against
 *      @c DEST_AUTO_LIMIT — which @c dest_helpers.hpp already derives correctly, kernel-side, from
 *      the JIT sync/accum mode. @c run_chunked() serves callers whose block genuinely exceeds DST.
 *   3. INIT HOISTING. Most of these kernels call @c add_tiles_init on every single chunk. @c arm()
 *      programs it once and re-programs ONLY when the seeded/unseeded mode actually changes, tracked
 *      in @c programmed_seeded_ — the same hoist-but-re-establish-on-change discipline the matmul
 *      helper uses.
 *   4. THE 3-INPUT SHAPE. The terminal ring step adds THREE tensors. The idiom is
 *      @c copy_tile the seed into DST, then @c add_tiles_init(..., acc_to_dest=true) so the add
 *      accumulates onto it. @c run_seeded() owns that sequence.
 *
 * @par The DST-zero invariant (acquire does not zero; RELEASE does).
 *   @c tile_regs_acquire() is @c llk_math_wait_for_dest_available() and nothing more — despite an
 *   in-tree comment claiming it "resets DST to 0". The zero comes from the OTHER end of the
 *   lifecycle: @c tile_regs_release() is the pack-side @c llk_pack_dest_section_done, which
 *   ZEROACCs the just-released DST region (CLR_ALL under SyncFull, CLR_HALF under SyncHalf, with
 *   fp32 variants — same on Wormhole and Blackhole). So in the standard
 *   acquire/commit/wait/release flow, DST IS zero at every acquire: boot state is zero and every
 *   release re-zeroes what it hands back. Unseeded @c acc_to_dest accumulation from a zero start
 *   (@c sum_blocks below, all_reduce's reduction, llama_reduce_scatter) is sound under that
 *   invariant; @c run_seeded() copy_tile-seeds because the terminal step's third addend LIVES IN A
 *   CB — it is data plumbing, not zero-safety. The invariant only breaks for code that bypasses
 *   the standard release path.
 *
 * @par OWNERSHIP SPLIT (same discipline as the other helpers).
 *   Owned here: the CB wait/pop/reserve/push protocol at granularity, the @c tile_regs lifecycle, the
 *   @c add_tiles loop, DST-capacity checking, and the per-op @c add_tiles_init placement. NOT owned:
 *   which CBs to reduce and when (that is the schedule), the fabric, any epilogue fused after the add
 *   (e.g. @c strided_reduce_scatter_async's addcmul — the op keeps that and re-arms afterwards), and
 *   anything multi-device.
 *
 *   Also NOT owned, deliberately: HARDWARE STARTUP. @c compute_kernel_hw_startup /
 *   @c binary_op_init_common configure the whole unpack/math/pack pipeline once per KERNEL, whereas an
 *   accumulator is scoped to one CB triple — a kernel may hold one alongside other compute ops (an
 *   addcmul epilogue, a norm). Folding startup into @c arm() would conflate those two lifetimes and
 *   would silently swap one startup path for the other: the two are NOT interchangeable, since
 *   @c binary_op_init_common additionally issues @c state_configure, @c llk_unpack_AB_init,
 *   @c llk_pack_init and @c llk_pack_dest_init. The kernel keeps its existing startup call verbatim;
 *   @c arm() only issues the op-level @c add_tiles_init.
 *
 * @par USAGE.
 * @code
 *   compute_kernel_hw_startup(cb_a, cb_b, cb_out);   // stays with the kernel, unchanged
 *
 *   // Arm once, outside the loop: programs add_tiles_init + asserts granularity <= DST.
 *   auto acc = compute_kernel_lib::BlockAccumulate::arm(cb_a, cb_b, cb_out, tile_granularity);
 *
 *   while (...) {
 *       if (three_input_step) {
 *           acc.run_seeded(cb_seed, n);   // out = seed + a + b
 *       } else {
 *           acc.run(n);                   // out = a + b
 *       }
 *   }
 * @endcode
 */

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

/**
 * @brief An armed elementwise block-accumulator over a fixed (a, b, out) CB triple.
 *
 * Arm once, run many. Construct via @c arm(); the constructor is private so an unarmed instance —
 * one whose @c binary_op_init_common has not run — cannot be named, mirroring the dataflow helper's
 * "holding the handle is the proof that arming happened" property.
 */
class BlockAccumulate {
public:
    /**
     * @brief Program the op-level init for this CB triple and return the armed accumulator.
     *
     * @pre The kernel has already run its hardware startup (@c compute_kernel_hw_startup or
     *      @c binary_op_init_common) — see the header's ownership note on why this does not.
     *
     * @param cb_a         First addend CB (SrcA).
     * @param cb_b         Second addend CB (SrcB).
     * @param cb_out       Destination CB.
     * @param granularity  Tiles the CB protocol operates on per run — what the producer pushes and
     *                     the consumer expects. Must be <= the DEST register capacity; asserted.
     */
    static ALWI BlockAccumulate arm(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t granularity);

    /**
     * @brief @c out = a + b over @c num_tiles tiles.
     * @param num_tiles Tiles to add, <= the armed granularity. A short final chunk is normal.
     * @note Re-programs @c add_tiles_init only if the previous run was seeded.
     */
    ALWI void run(uint32_t num_tiles);

    /**
     * @brief @c out = seed + a + b over @c num_tiles tiles — the terminal ring step's 3-input add.
     *
     * Seeds DST from @c cb_seed with @c copy_tile, then accumulates @c a + @c b onto it via
     * @c acc_to_dest. Waits and pops @c cb_seed at granularity alongside @c a and @c b.
     * @note Re-programs @c add_tiles_init only if the previous run was unseeded.
     */
    ALWI void run_seeded(uint32_t cb_seed, uint32_t num_tiles);

    /**
     * @brief @c run() for a block LARGER than the DEST register: splits into DST-sized passes.
     *
     * Separate from @c run() on purpose. @c run() emits exactly the instruction sequence the shipped
     * reduction collectives were verified with — including popping @c a and @c b BEFORE reserving
     * @c out, which releases the reader's slots as early as possible. Chunking cannot preserve that
     * order (every pass reads @c a and @c b, so the pops must follow the last one, and the reserve
     * must precede the first pack), so it gets its own entry point instead of silently changing the
     * ordering of the verified path. Use this for blocks whose size is not host-clamped to DST —
     * e.g. retiring @c all_reduce_async's `max_dst_tiles = 8  // TODO: Make general`.
     *
     * @param num_tiles     Total tiles to add; may exceed DEST capacity.
     * @param out_capacity  Tiles to reserve/push on @c cb_out (>= num_tiles).
     */
    ALWI void run_chunked(uint32_t num_tiles, uint32_t out_capacity);

    /**
     * @brief Re-establish this accumulator's init after the CALLER ran other compute ops on the same
     *        core, invalidating the cached mode.
     *
     * The mode tracking in @c programmed_seeded_ assumes nothing else reprogrammed the unpack/math
     * state between runs. A kernel that fuses an epilogue — @c strided_reduce_scatter_async's addcmul
     * issues @c mul_tiles_init / @c add_tiles_init / @c reconfig_data_format of its own — breaks that
     * assumption, and the next @c run() would then skip an init it actually needs. Call this after any
     * such interleaved op.
     *
     * Restores the unpack/pack DATA FORMATS as well as the op init, because @c add_tiles_init does not
     * (its @c state_configure call is the ComputeKernelSentinel tracker, not a hardware reconfigure).
     * That closes a pre-existing hazard: @c strided_reduce_scatter_async's normal ring step never
     * re-issued any init, so after a fused final step it ran with the addcmul operands' formats still
     * programmed — benign only while those CBs happen to share a data format with the reduce operands.
     */
    ALWI void rearm();

private:
    ALWI BlockAccumulate(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t granularity) :
        cb_a_(cb_a), cb_b_(cb_b), cb_out_(cb_out), granularity_(granularity) {}

    /// Re-establish add_tiles_init only when the seeded/unseeded mode actually changes.
    ALWI void ensure_mode(bool seeded);

    uint32_t cb_a_;
    uint32_t cb_b_;
    uint32_t cb_out_;
    uint32_t granularity_;
    /// What add_tiles_init currently has programmed. Unlike the dataflow helper's per-channel packet
    /// headers, unpack/math config is SINGULAR hardware state, so two differently-armed accumulators
    /// cannot coexist — hence tracking the mode here rather than handing out two armed objects.
    bool programmed_seeded_ = false;
};

/**
 * @brief @c out = the sum of @c num_blocks equal-shaped tile blocks RESIDENT in one CB — the
 *        all_reduce pattern, where the gathered per-device partials land as contiguous blocks of
 *        one input CB and are summed into a single output block.
 *
 * Owns the CB protocol this shape actually has: waits the WHOLE input
 * (num_blocks * block_num_tiles) up front, reserves/pushes one output block — and pops the input
 * only when @c pop_input is set: all_reduce's input CB is a shell over the gathered data that the
 * op keeps resident (pop_input = false), while llama_reduce_scatter's fabric-receiver CB is a real
 * producer/consumer CB (pop_input = true, popped before the output push, as it always was).
 * Owns the DST-capacity chunking against @c DEST_AUTO_LIMIT (retiring the hand-rolled
 * `max_dst_tiles = 8` it replaces, which ignored fp32 dest-accum), and the ODD block count: an odd
 * count copy_tile-seeds DST with block 0 and accumulates the remaining PAIRS — replacing an empty
 * "TODO: Future support" branch that paired blocks off the end of the CB for odd counts. An even
 * count accumulates all pairs with @c acc_to_dest from DST's zero start (sound per the banner's
 * DST-zero invariant) — bit-identical to the shipped even-count sequence.
 *
 * @note Distinct from reduce_helpers_compute.hpp's @c Accumulate knob, despite the shared
 *   copy_tile-seed-then-accumulate DST idiom: that knob reloads a running partial and accumulates
 *   @c reduce_tile outputs (dimensional reduce, scaler CB, reduced-shape output) across chunk
 *   iterations, while this sums resident blocks element-wise via @c add_tiles (full-shape output,
 *   no scaler). Different LLK op and output shape — neither can express the other.
 *
 * @pre Hardware startup (@c binary_op_init_common) has run — same ownership note as arm().
 * @post @c add_tiles_init is left in acc_to_dest mode; a BlockAccumulate live in the same kernel
 *       must @c rearm() before its next run().
 *
 * @param cb_in           CB holding the num_blocks gathered blocks.
 * @param cb_out          Destination CB for the single summed block.
 * @param num_blocks      Blocks to sum (>= 1; 1 degenerates to a copy of block 0).
 * @param block_num_tiles Tiles per block.
 * @param pop_input       Pop the whole input after the sum (see the ownership note above).
 */
ALWI void sum_blocks(
    uint32_t cb_in, uint32_t cb_out, uint32_t num_blocks, uint32_t block_num_tiles, bool pop_input = false);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.inl"
