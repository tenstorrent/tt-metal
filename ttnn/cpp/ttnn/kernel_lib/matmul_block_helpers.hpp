// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

namespace compute_kernel_lib {

/**
 * Output pack-layout mode selected at compile time on the last K-block.
 *
 * SubblockMajor (default, legacy): sequential pack_tile_block per subblock;
 *   writer reads tiles in subblock-major order. Required by factories that
 *   emit subblock-major writer kernels.
 * RowMajor: absolute-offset pack_tile<true> into row-major positions within
 *   the M-row-group; reserve/push per row-group. Decouples subblock choice
 *   from output layout so factories can grow subblocks freely, and is the
 *   mode SDPA callers require for absolute-offset partial writes.
 *
 * When matmul_block feeds add_bias_bcast_rows, both must use the same
 * OutputLayout so the intermediate CB layout matches.
 */
enum class OutputLayout { SubblockMajor, RowMajor };

/**
 * Where the last K-block packs and what post-op it gets, picked at compile time.
 *
 * Out              (default) Last block packs to out_buf with no relu.
 * OutWithRelu      Last block packs to out_buf with PACK_RELU enabled.
 * Interm           Last block packs to interm_buf for a downstream phase (bias add /
 *                  untilize) to consume. RELU on this path lives in the downstream
 *                  phase, not the matmul.
 * OutWithUntilize  Last block packs through pack_untilize_dest, untilizing tile-format
 *                  DST data into row-major-byte output. Bracketed by per-subblock
 *                  pack_untilize_dest_init / pack_untilize_uninit so other ops can
 *                  resume their own packer config afterwards. Requires
 *                  layout == SubblockMajor and a non-zero untilize_block_ct_dim
 *                  template parameter equal to the per-call block_ct_dim. The
 *                  RowMajor + untilize combination has no caller and isn't
 *                  expressible via the strided fifo math; route through
 *                  Interm + reblock_and_untilize for that path instead.
 *
 * Replaces the previous (pack_last_to_interm, pack_relu) bool pair: the impossible
 * combination (Interm + Relu) is unrepresentable.
 */
enum class LastBlockTarget : uint8_t { Out, OutWithRelu, Interm, OutWithUntilize };

/**
 * Per-operand input lifecycle policy, picked at compile time. Mirrors the
 * `InputPolicy` convention used by reduce/binary helpers — the helper waits
 * and pops on the bound CB unless the caller opts out for a specific reason.
 *
 * WaitAndPopPerKBlock      (default) Helper issues wait_front(block_num_tiles)
 *                          at the start of every K-block and the matching
 *                          pop_front at the end. Standard for callers whose
 *                          producer pushes one block per K-iteration.
 * WaitAndRetainOnLastBlock Helper waits per K-block but skips the pop on the
 *                          last K-block so the caller's outer loop can reuse
 *                          the data across iterations. Use for SDPA reusing Q
 *                          across K chunks (in0) or conv3d reusing weights
 *                          across multiple matmul invocations within an
 *                          output block (in1, num_k_blocks=1 makes every call
 *                          last, so the helper never pops).
 * NoWaitNoPop              (in1 only — enforced via static_assert on the in0 slot)
 *                          Helper skips both wait and pop entirely; the caller
 *                          manages in1 lifecycle externally. Use for cross-chip
 *                          global-CB receivers (the producer push lands via
 *                          fabric and the receiver advances its own rd_ptr per
 *                          K-block via PostKBlockFn) and for pre-populated
 *                          L1-sharded in1 CBs (no in-program producer). There
 *                          is no analogous external-management pattern for in0.
 */
enum class InputPolicy : uint8_t { WaitAndPopPerKBlock, WaitAndRetainOnLastBlock, NoWaitNoPop };

namespace matmul_config {

/**
 * Init lifecycle for matmul_block.
 *
 * The helper owns matmul-state setup so callers don't have to pair every call site with a
 * matching mm_block_init. Mirrors the InitUninitMode convention used by untilize_helpers
 * for back-to-back invocations.
 *
 * Full          (default) Helper calls mm_block_init at the start — the full re-init that
 *               re-runs unpack/math/pack hw_configure plus pack_init/pack_dest_init in
 *               addition to the matmul-specific init. Required when an intervening op
 *               (tilize, untilize, reduce, eltwise) reconfigured PACK/UNPACK/MATH for its
 *               own needs and the matmul-mode hw config must be re-established. After
 *               compute_kernel_hw_startup alone (no intervening ops), Short is sufficient
 *               and faster, but Full is the safe conservative default.
 * Short         Helper calls mm_block_init_short — only the matmul-specific
 *               llk_unpack_AB_matmul_init + llk_math_matmul_init, no hw_configure. Use
 *               (a) right after compute_kernel_hw_startup with the same in0/in1/out CBs,
 *               or (b) for chains of matmul_block calls where the previous helper already
 *               configured matmul state and only the per-call shape/cb might have changed.
 * None          Helper skips init entirely. Use only when the caller has just executed a
 *               compatible matmul_block in the same configuration and no other op has
 *               touched matmul state.
 */
enum class InitMode : uint8_t { Full, Short, None };

}  // namespace matmul_config

/**
 * Block-shape specification for matmul_block.
 *
 * Groups the dimensional params — subblock counts, subblock size, K-blocking,
 * batch — into one struct so callers pass intent instead of seven positional
 * integers. Optional strides (in1_per_core_w / out_row_width) stay on the
 * function signature because they're advanced layout overrides only a few
 * factories need.
 *
 * Usage:
 *   matmul_block<...>(
 *       in0_buf, in1_buf, out_buf, interm_buf,
 *       MatmulBlockShape::of(in0_sb, in1_sb, h, w, in0_block_w, num_k_blocks),
 *       ...);
 */
struct MatmulBlockShape {
    uint32_t in0_num_subblocks;  // Output subblock count along M.
    uint32_t in1_num_subblocks;  // Output subblock count along N.
    uint32_t out_subblock_h;     // Output subblock height in tiles.
    uint32_t out_subblock_w;     // Output subblock width in tiles.
    uint32_t in0_block_w;        // K per K-block in tiles (= A's "per_core_K_block").
    uint32_t num_k_blocks;       // Number of K-blocks along the K dimension.
    uint32_t batch = 1;          // Independent batch slices. Pass the actual batch count
                                 // when the caller has no per-batch work between matmuls
                                 // (matmul-only kernels): the helper's own batch loop runs
                                 // mm_block_init exactly once across all batches, which is
                                 // both faster and avoids the heterogeneous-tile-shape
                                 // re-init corruption fixed in commit 76e99730d2e. Keep
                                 // batch=1 and run the kernel's own batch loop ONLY when
                                 // per-batch phase work (bias add, untilize, mailbox sync)
                                 // must be interleaved between iterations.

    static constexpr MatmulBlockShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t in0_block_w,
        uint32_t num_k_blocks,
        uint32_t batch = 1) {
        return {in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_w, num_k_blocks, batch};
    }
};

// Default no-op post-compute functor.
// Called per output sub-block on the last K-block, before packing.
// Receives out_subblock_num_tiles. Tiles are in DST[0..num_tiles-1].
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

// Default no-op pre-K-block functor.
// Called at the start of each K-block iteration, before input CB waits.
// Receives (block_index, num_k_blocks, is_last_block).
// Use for per-K-block preprocessing (e.g., in0_transpose, global CB pointer manipulation).
struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Default no-op post-K-block functor.
// Called at the very end of each K-block iteration, after the input pop_front
// and after the L1_ACC partial drain. Symmetric counterpart to PreKBlockFn.
// Receives (block_index, num_k_blocks, is_last_block).
// Use for per-K-block postprocessing (e.g., ring CB rd_ptr advance after the
// in1 block has been consumed).
struct NoPostKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Default per-K-block inner-dim step-count functor.
// Called at the start of each K-block's per-subblock matmul loop to determine
// how many FMA steps to issue along the inner dim. The default returns the
// static block_w so the loop runs the full K-tile span — preserving prior
// helper behavior. Callers that pad in0/in1 K-tiles for some K-blocks override
// this to return the unpadded step count for those blocks.
//
// The inner LLK call's kt_dim argument stays block_w in both cases — that
// parameter is the in1 row stride in L1, NOT the FMA step count. Returning
// a smaller value here only shrinks the loop bound; the LLK still strides
// through in1's tile geometry as if it were full-width.
struct NoKBlockInnerDimFn {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t block_w) const { return block_w; }
};

// Default per-K-block in0 source functor.
// Called at the start of each K-block iteration to determine which CB to read
// in0 from on this iteration. The default returns the in0_cb_id the helper was
// invoked with — preserving prior single-source behavior. Callers that swap
// between two physically-distinct in0 CBs (e.g. ring-aware all-gather where
// block 0 reads from a self-mcast CB and blocks 1..N read from a remote CB)
// override this to return the alternate CB id on the appropriate K-blocks.
//
// IMPORTANT — DATAFORMAT INVARIANT: the alternate in0 CBs MUST share the SAME
// dataformat as the helper's bound in0_cb_id. The kernel-entry mm_block_init
// and the helper's reload-time mm_block_init_short_with_dt both configure the
// unpacker for the bound in0_cb_id and don't re-issue per K-block; if a
// returned CB has a different dataformat, the unpacker config is stale and
// in0 unpacks land at wrong values. Helper does not check this — caller's
// responsibility, asserted via the In0SourceFn documentation here.
struct NoIn0Source {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t in0_cb_id) const { return in0_cb_id; }
};

// Default per-K-block in1 base-offset functor.
// Called at the start of each K-block iteration to determine the in1 starting
// tile offset for that block's matmul reads. Default returns 0 — the helper
// reads in1 from offset 0 + in1_subblock * out_subblock_w within each K-block,
// matching prior behavior.
//
// Callers that share a single in1 CB across multiple ring positions (no rd_ptr
// rotation, no separate alternate CBs — just an offset shift into the same
// fronted region) override this to return the per-K-block base offset in tiles
// (e.g. ring-aware all-gather without ENABLE_GLOBAL_CB:
// `return in1_block_num_tiles * curr_ring_idx;`). The wait/pop lifecycle stays
// on the bound in1_cb_id — this is purely an LLK in1_index shift, not a CB
// swap. Pair with `in1_policy=NoWaitNoPop` when the caller manages in1's
// rd_ptr/lifecycle externally.
struct NoIn1BaseOffset {
    ALWI uint32_t operator()(uint32_t /*block*/) const { return 0; }
};

/**
 * matmul_block: sub-blocked tiled matrix multiplication C = A × B with K-blocking.
 *
 * Required includes:
 *   #include "api/compute/compute_kernel_hw_startup.h"  // for compute_kernel_hw_startup()
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
 *
 * One helper serving both standard matmul (non-multicast bmm) and SDPA. Supports two
 * output-pack strategies selected at compile time via layout:
 *
 *   layout=SubblockMajor (default): sequential pack_tile_block per subblock.
 *     Output lands in subblock order. Required by multicast writers that expect
 *     subblock-order tile stream.
 *   layout=RowMajor: absolute-offset pack_tile<true> at row-major positions,
 *     reserve/push per M-row-group. Decouples subblock choice from output layout so
 *     the factory can pick larger subblocks (the main SDPA perf path). Also the
 *     mode SDPA callers require for absolute-offset partial writes across K chunks.
 *
 * K-accumulation also selectable at compile time:
 *   packer_l1_acc=false: Software spill/reload via interm_cb
 *   packer_l1_acc=true:  Hardware L1 accumulation via packer (no spill/reload)
 *
 * ── Precision (host-side ComputeConfig guidance) ──────────────────────────
 * The K-accumulation performs Kt * 32 multiply-adds per output element. The
 * helper itself does not configure math fidelity or DEST precision — those
 * live in the host-side ComputeConfig and the caller picks them. Pairings
 * that match each input dtype:
 *   - bf16 inputs, Kt == 1                : LoFi or HiFi2 are sufficient.
 *   - bf16 inputs, Kt > 1                 : HiFi2 + fp32_dest_acc_en=True is
 *                                           the safe default. Without the
 *                                           fp32 DEST the K-accumulation
 *                                           rounds back to bf16 every step
 *                                           and max-abs error grows roughly
 *                                           O(sqrt(K)).
 *   - fp32 inputs                         : HiFi4 + fp32_dest_acc_en=True
 *                                           is the only correct combination.
 *   - HiFi4 + fp32_dest_acc_en=True
 *     with bf16 inputs                    : KNOWN BAD on Wormhole B0 (issue
 *                                           #38306) — silent precision
 *                                           corruption on the K-accumulator.
 *                                           Use HiFi2 or HiFi3 instead.
 *
 * Init handling: by default the helper calls mm_block_init() itself (init_mode=Full).
 * The caller's only init responsibility is one compute_kernel_hw_startup() at boot.
 * For back-to-back chains, init_mode=Short uses mm_block_init_short (cheap restore);
 * init_mode=None skips init entirely. See matmul_config::InitMode.
 *
 * SKIP_COMPUTE: When this macro is defined by the calling TU (microbenchmark path),
 * the inner ckernel::matmul_block() call is omitted. All other pipeline work (waits,
 * reloads, packs, L1_ACC toggles) still runs so the harness measures non-compute
 * overhead. Handled inside this helper — caller does nothing special.
 *
 * Uses 4-phase DST management (tile_regs_acquire/commit/wait/release) for correct
 * MATH-PACK pipelining.
 *
 * ── Template Parameters ────────────────────────────────────────────────────
 *
 *   transpose         If true, transpose B tiles before multiplication (default: false).
 *   packer_l1_acc     Enable packer L1 accumulation instead of software spill/reload.
 *   last_block_target LastBlockTarget: Out (default), OutWithRelu, or Interm.
 *                     See LastBlockTarget docstring for the three valid pack/RELU
 *                     combinations.
 *   layout            OutputLayout: SubblockMajor (default) or RowMajor (see above).
 *   init_mode         matmul_config::InitMode: Full (default), Short, or None.
 *                     Controls whether the helper itself calls mm_block_init / _short.
 *   in0_policy        InputPolicy: WaitAndPopPerKBlock (default) or
 *                     WaitAndRetainOnLastBlock (caller reuses in0 across the next
 *                     iteration — SDPA reuses Q across K chunks). NoWaitNoPop is
 *                     rejected on the in0 slot via static_assert. See InputPolicy
 *                     docstring above.
 *   in1_policy        InputPolicy: WaitAndPopPerKBlock (default),
 *                     WaitAndRetainOnLastBlock (conv3d reuses weights across
 *                     multiple matmul invocations within an output block), or
 *                     NoWaitNoPop (caller manages in1 lifecycle externally —
 *                     cross-chip global-CB receivers, pre-populated L1-sharded
 *                     in1). See InputPolicy docstring above.
 *   PostComputeFn     Functor called per output sub-block on the last K-block,
 *                     after matmul but before packing. Receives out_subblock_num_tiles.
 *   PreKBlockFn       Functor called at the start of each K-block iteration, before
 *                     input CB waits. Receives (block, num_k_blocks, is_last).
 *                     Use for per-K-block preprocessing such as in0_transpose.
 *   PostKBlockFn      Functor called at the very end of each K-block iteration, after
 *                     input pop_front and after the L1_ACC partial drain. Receives
 *                     (block, num_k_blocks, is_last). Symmetric counterpart to
 *                     PreKBlockFn — use for per-K-block postprocessing such as
 *                     advancing a ring CB rd_ptr after the consumer has run.
 *   KBlockInnerDimFn  Functor called per K-block to determine the inner-dim FMA step
 *                     count. Receives (block, block_w); returns the number of inner
 *                     iterations to issue. Default returns block_w (full K-tile span).
 *                     Use when some K-blocks have unpadded widths smaller than the
 *                     static block_w (e.g. ring-aware all-gather where each block
 *                     consumes a different chip's shard width). The inner LLK call's
 *                     kt_dim arg stays block_w — only the loop bound shrinks.
 *   In0SourceFn       Functor called per K-block to pick which CB to read in0 from.
 *                     Receives (block, in0_cb_id); returns the active in0 CB id for
 *                     this iteration. Default returns in0_cb_id (single-source,
 *                     prior behavior). Use when some K-blocks read from an alternate
 *                     in0 CB (e.g. block 0 reads from a self-primed local CB while
 *                     blocks 1..N read from a remote CB). The active CB drives
 *                     wait_front, pop_front, and the LLK call's in0_cb_id; the
 *                     kernel-entry mm_block_init and the reload's
 *                     mm_block_init_short_with_dt keep using the bound in0_cb_id,
 *                     so the alternate CBs MUST share the same dataformat.
 *   In1BaseOffsetFn   Functor called per K-block to determine the in1 starting tile
 *                     offset. Receives (block); returns the LLK in1_index base for
 *                     this iteration. Default returns 0 (helper reads in1 from
 *                     offset 0 onwards, matching prior behavior). Use when a single
 *                     fronted in1 region holds multiple ring positions and the
 *                     kernel rotates between them via offset arithmetic instead of
 *                     rd_ptr advance — e.g. ring-aware all-gather without
 *                     ENABLE_GLOBAL_CB returns `in1_block_num_tiles * curr_ring_idx`.
 *   caller_owns_pack_target
 *                     Default false. When true, the helper skips ALL of its own
 *                     reserve_back / push_back / inter-block drain calls on the pack
 *                     target buffer (out_buf or interm_buf depending on
 *                     last_block_target) and the spill interm_buf — the caller is
 *                     responsible for one cb_reserve_back BEFORE the matmul_block
 *                     call and one cb_push_back AFTER. Use for kernels that
 *                     accumulate K-blocks into a single pre-reserved L1 region via
 *                     L1_ACC and pack with absolute-offset row-major (typical of
 *                     ring-aware all-gather variants where the matmul output feeds
 *                     a downstream bias/activation phase). Pair with
 *                     `last_block_target=Interm`, `OutputLayout::RowMajor`,
 *                     `packer_l1_acc=true`, and a `MatmulBlockShape` whose runtime
 *                     fields reflect the (possibly partial) M/N being processed
 *                     this invocation.
 *   untilize_block_ct_dim
 *                     Compile-time block_ct_dim threaded into pack_untilize_dest_init /
 *                     pack_untilize_dest. Required (>0) when last_block_target ==
 *                     OutWithUntilize; ignored otherwise. Set equal to
 *                     out_subblock_h * out_subblock_w (= out_subblock_num_tiles) so
 *                     the per-subblock pack_untilize call covers the full DST sub-block.
 *   pin_interm_to_captured_base
 *                     Default false. When true, the helper captures interm_buf's
 *                     fifo_rd_ptr / fifo_wr_ptr at entry and resets them per K-block
 *                     (and once after the K-loop on the pack_last_to_interm path) to
 *                     keep interm_buf operating at a fixed L1 base across all blocks.
 *                     Required when interm_buf is allocated to alias the output buffer
 *                     in L1 (e.g. conv2d's `partials_cb_uses_output=true` path) — the
 *                     K-loop's natural fifo advance would otherwise wrap and overwrite
 *                     previously packed output. Pin pattern matches conv2d's original
 *                     manual K-loop:
 *                       pack_last_to_interm:        rd+wr reset for block < num-1
 *                       !pack_last_to_interm: rd reset for block < num-1; wr reset for
 *                                             block < num-2 (so the last reload still
 *                                             finds the second-to-last block's data at
 *                                             advanced wr_ptr).
 *   Activation        Default NoneActivation. When the bound activation kind is non-NONE
 *                     the helper fuses SFPU activation onto the PACKER thread (TRISC2)
 *                     at the per-subblock pack stage of the last K-block, instead of
 *                     running it on the math thread before pack. The packer-side path
 *                     overlaps activation with the next math K-block and frees DST
 *                     register pressure on the math side. Replaces (rather than
 *                     augments) the standard tile_regs_wait packer-side sync —
 *                     apply_activation_from_pack does its own math/pack semaphore wait
 *                     + dest-offset flip + SFPU stall. The existing PostComputeFn
 *                     template parameter is unaffected: it still fires on the MATH
 *                     thread before tile_regs_commit and is the right hook for
 *                     non-activation math-thread post-compute (e.g. SDPA mask
 *                     application). Allowed alongside last_block_target == Interm — the
 *                     helper does not assume a downstream bias phase; callers on the
 *                     FUSE_BIAS path keep this NoneActivation and route activation to
 *                     the bias helper's Activation slot, while callers on the
 *                     untilize_out + !FUSE_BIAS path (matmul packs to interm; untilize
 *                     phase reads unchanged) set Activation here so activation lands
 *                     during the matmul pack. ActivationInitHelper::init() is issued by
 *                     the helper on init_mode == Full alongside mm_block_init; callers
 *                     using init_mode == Short or None must call init() at boot. Build
 *                     an Activation from one of the named aliases in
 *                     sfpu_activation_helpers.hpp (HardtanhActivation<low_bits, high_bits>,
 *                     SeluActivation<alpha_bits, lambda_bits>, …) so the per-activation
 *                     parameter meaning is explicit at the call site; for host-driven
 *                     kernels that read activation + params from compile-time args, wrap
 *                     them as ActivationOp<activation_type, p0, p1, p2>.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in0_buf, in1_buf   Input buffers for matrices A and B (CircularBuffer or
 *                      DataflowBuffer — pass an experimental::CircularBuffer
 *                      object on legacy CB-backed kernels, or an
 *                      experimental::DataflowBuffer object on Metal-2.0 / DFB
 *                      kernels).
 *   out_buf            Output buffer for the final result.
 *   interm_buf         Intermediate buffer for K-blocking spill/reload or
 *                      L1-ACC FIFO. When num_k_blocks == 1 the helper never
 *                      accesses it; pass out_buf itself in this slot as the
 *                      canonical zero-cost placeholder (no extra CB to
 *                      allocate). Any other buffer of the same type also
 *                      works — SDPA passes in0_buf because Q is retained
 *                      across K-chunks via in0_policy=WaitAndRetainOnLastBlock
 *                      (see the SDPA @example below).
 *   shape              MatmulBlockShape (see above) — subblock counts, subblock size,
 *                      K-blocking, batch. Build with MatmulBlockShape::of(...).
 *   post_compute       PostComputeFn instance (default: {}).
 *   pre_k_block        PreKBlockFn instance (default: {}).
 *   in1_per_core_w     Actual number of N-tiles in the in1 CB per K-block (= what NCRISC
 *                      pushes per block). Defaults to 0, meaning derive from
 *                      out_subblock_w * in1_num_subblocks. Pass the real value when the
 *                      factory pads per_core_N_compute above the actual in1 shard width
 *                      (e.g. matmul_multicore_reuse_mcast_dram_sharded), otherwise the
 *                      helper will wait/pop wrong tile counts and deadlock.
 *   out_row_width      N-tiles per row in the OUTPUT CB layout (row stride for row_major
 *                      pack). Defaults to 0, meaning reuse in1_per_core_w. For most factories
 *                      in1 read stride and output pack stride coincide. DRAM-sharded is
 *                      the exception: it reads in1 at per_core_N_in1_sender (unpadded shard
 *                      width) but packs output at per_core_N_compute (padded after subblock-
 *                      growth); those factories must pass the larger pack stride here.
 *
 * @example
 *   // Single-core matmul with defaults — no transpose, no L1 accumulation,
 *   // no activation fusion, SubblockMajor pack, init_mode=Full. Valid for any
 *   // (M, K, N) whose K dimension fits in a single K-block (= all Kt tiles
 *   // fit alongside one M and N sub-block in L1). Caller constructs
 *   // experimental::CircularBuffer (or DataflowBuffer) once per CB and
 *   // passes the object; the helper wraps wait_front / pop_front /
 *   // reserve_back / push_back on it and issues mm_block_init internally.
 *   // Passes out_buf itself as the interm placeholder — see the interm_buf
 *   // runtime-param doc above for why that is the canonical pattern when
 *   // num_k_blocks == 1.
 *   experimental::CircularBuffer in0_buf(cb_in0);
 *   experimental::CircularBuffer in1_buf(cb_in1);
 *   experimental::CircularBuffer out_buf(cb_out);
 *   matmul_block<>(
 *       in0_buf, in1_buf, out_buf,
 *       out_buf,  // interm placeholder — unread when num_k_blocks == 1
 *       MatmulBlockShape::of(
 *           Mt,    // in0_num_subblocks
 *           Nt,    // in1_num_subblocks
 *           1,     // out_subblock_h
 *           1,     // out_subblock_w
 *           Kt,    // in0_block_w
 *           1));   // num_k_blocks
 *
 * @example
 *   // Row-major output + packer-L1 accumulation across K, no fused bias.
 *   // Template order: transpose, packer_l1_acc, last_block_target, layout.
 *   // Buf is deduced from the buffer-object arguments.
 *   matmul_block<false, true, LastBlockTarget::Out, OutputLayout::RowMajor>(
 *       in0_buf, in1_buf, out_buf, interm_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_k_blocks));
 *
 * @example
 *   // SDPA-style: row-major pack, retain in0 to reuse Q across K chunks, masked post-compute.
 *   // The SDPA-side wrapper does its own [mm_block_init_short, reconfig_data_format]
 *   // pair externally (for ordering parity with matmul_reduce_inplace.inl and
 *   // OptionalMaskPostCompute), so the helper is invoked with init_mode=None.
 *   // Template slot order: transpose, packer_l1_acc, last_block_target, layout,
 *   // init_mode, in0_policy, in1_policy, PostComputeFn.
 *   matmul_block<transpose, false, LastBlockTarget::Out, OutputLayout::RowMajor,
 *                matmul_config::InitMode::None,
 *                InputPolicy::WaitAndRetainOnLastBlock, InputPolicy::WaitAndPopPerKBlock,
 *                OptionalMaskPostCompute>(
 *       in0_buf, in1_buf, out_buf, in0_buf,  // interm unused when num_k_blocks==1
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             subblock_h, subblock_w, in0_block_w, num_blocks),
 *       OptionalMaskPostCompute{...},
 *       NoPreKBlock{});
 *
 * @example
 *   // FUSE_BIAS path: last K-block packs to interm_buf so add_bias_bcast_rows reads it.
 *   // DRAM-sharded passes explicit in1_per_core_w (shard width) and
 *   // out_row_width (padded pack width).
 *   matmul_block<in1_transpose_tile, l1_acc, LastBlockTarget::Interm,
 *                output_layout, matmul_config::InitMode::Full,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                PostFn, PreFn>(
 *       in0_buf, in1_buf, out_buf, mm_partials_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_blocks_inner_dim),
 *       PostFn{}, PreFn{},
 *       in1_block_w,      // in1_per_core_w  (DRAM-sharded shard width)
 *       out_block_w);     // out_row_width   (DRAM-sharded padded pack width)
 *
 * @example
 *   // Per-K-block unpadded inner-dim: each K-block consumes a different chip's
 *   // shard width (ring-aware all-gather pattern). The kernel pads in0/in1 K-tiles
 *   // up to a uniform block_w so CB sizing stays static, then the helper's
 *   // KBlockInnerDimFn callback shrinks the FMA loop to the unpadded count.
 *   struct RingInnerDimFn {
 *       const uint32_t* unpadded_widths;  // ring_size entries, runtime-arg backed
 *       uint32_t ring_idx;
 *       uint32_t ring_size;
 *       ALWI uint32_t operator()(uint32_t block, uint32_t) const {
 *           return unpadded_widths[(ring_idx + block) % ring_size];
 *       }
 *   };
 *   // untilize_block_ct_dim=0 (template slot 12, no untilize on this path).
 *   matmul_block<transpose, l1_acc, LastBlockTarget::Out, OutputLayout::SubblockMajor,
 *                matmul_config::InitMode::None,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                NoPostCompute, NoPreKBlock, false, NoPostKBlock,
 *                0, RingInnerDimFn>(
 *       in0_buf, in1_buf, out_buf, interm_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_blocks),
 *       NoPostCompute{}, NoPreKBlock{},
 *       in1_per_core_w, out_row_width,
 *       NoPostKBlock{}, RingInnerDimFn{unpadded, ring_idx, ring_size});
 *
 * @example
 *   // Ring-aware all-gather matmul: PreKBlockFn captures the next in1 rd_ptr,
 *   // PostKBlockFn commits it after the K-block consumes in1; OutWithUntilize
 *   // untilizes per-subblock output directly through the LLK pack_untilize.
 *   // Template slot order: transpose, packer_l1_acc, last_block_target, layout,
 *   // init_mode, in0_policy, in1_policy, PostComputeFn, PreKBlockFn,
 *   // pin_interm_to_captured_base, PostKBlockFn, untilize_block_ct_dim.
 *   matmul_block<in1_transpose_tile, l1_acc, LastBlockTarget::OutWithUntilize,
 *                OutputLayout::SubblockMajor, matmul_config::InitMode::None,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                NoPostCompute, RingPreKBlock, false,
 *                RingPostKBlock, out_subblock_num_tiles>(
 *       in0_buf, in1_buf, out_buf, mm_partials_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, num_blocks),
 *       NoPostCompute{}, ring_pre_k_block,
 *       in1_per_core_w, out_block_w,
 *       ring_post_k_block);
 *
 * @example
 *   // conv2d pattern: per-K-block tilize (PreKBlockFn) + interm_buf pinned to a captured
 *   // base across the K-loop because matmul_partials_cb is allocated to alias out_cb in
 *   // L1 (partials_cb_uses_output=true). init_mode=None because the kernel-entry
 *   // mm_block_init covers initial state and ConvTilizePreKBlock issues
 *   // mm_block_init_short_with_both_dt after each tilize. Template slot order: transpose,
 *   // packer_l1_acc, last_block_target, layout, init_mode, in0_policy, in1_policy,
 *   // PostComputeFn, PreKBlockFn, pin_interm_to_captured_base.
 *   matmul_block<false, packer_l1_acc, LastBlockTarget::Interm,
 *                OutputLayout::SubblockMajor, matmul_config::InitMode::None,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                ConvSFPUPostCompute, ConvTilizePreKBlock,
 *                true>(  // pin_interm_to_captured_base
 *       cb_mm_in0, cb_in1, cb_matmul_partials, cb_matmul_partials,  // out==interm
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_w, in0_num_blocks_w),
 *       ConvSFPUPostCompute{}, conv_pre_k_block);
 */
template <
    bool transpose = false,
    bool packer_l1_acc = false,
    LastBlockTarget last_block_target = LastBlockTarget::Out,
    OutputLayout layout = OutputLayout::SubblockMajor,
    matmul_config::InitMode init_mode = matmul_config::InitMode::Full,
    InputPolicy in0_policy = InputPolicy::WaitAndPopPerKBlock,
    InputPolicy in1_policy = InputPolicy::WaitAndPopPerKBlock,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock,
    bool pin_interm_to_captured_base = false,
    typename PostKBlockFn = NoPostKBlock,
    uint32_t untilize_block_ct_dim = 0,
    typename KBlockInnerDimFn = NoKBlockInnerDimFn,
    typename In0SourceFn = NoIn0Source,
    typename In1BaseOffsetFn = NoIn1BaseOffset,
    bool caller_owns_pack_target = false,
    typename Activation = NoneActivation,
    typename Buf = ::experimental::CircularBuffer>
ALWI void matmul_block(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    MatmulBlockShape shape,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    uint32_t in1_per_core_w = 0,
    uint32_t out_row_width = 0,
    PostKBlockFn post_k_block = {},
    KBlockInnerDimFn k_block_inner_dim = {},
    In0SourceFn in0_source_fn = {},
    In1BaseOffsetFn in1_base_offset_fn = {});

/**
 * matmul_reduce_inplace: in-place reduce via matmul using a single-tile column identity.
 *
 * Consumes subblock_h×subblock_w tiles from the front of `in_out_cb`, computes
 *   DST = matmul(in_out_cb[0..subblock_h], in1_cb[0]) × block_kt accumulation
 * and packs back onto `in_out_cb` — repeated num_subblocks times to reduce the CB in
 * place. This pattern breaks the standard in0_cb != out_cb invariant that `matmul_block`
 * enforces, so it lives in a dedicated helper; SDPA uses this to fold partial-sum
 * results along M via a column-identity tile in in1_cb.
 *
 * The helper absorbs mm_block_init_short + reconfig_data_format + wait_front on both
 * CBs — the caller only needs to have produced the requisite tiles.
 *
 * ── Runtime Parameters ─────────────────────────────────────────────────────
 *
 *   in_out_buf      Buffer serving as both input and output (in-place).
 *   in1_buf         Buffer with the single column-identity tile (kept
 *                   fronted, not popped).
 *   num_subblocks   Number of subblock iterations (= rows / subblock_h).
 *   subblock_h      Subblock height in tiles (matmul rt_dim).
 *   subblock_w      Subblock width in tiles (matmul ct_dim; typically 1).
 *   block_kt        K dimension in tiles for each matmul call (typically 1 = subblock_w).
 *
 * @example
 *   // SDPA fold M partial-sums using a column-identity tile in in1_buf.
 *   // Before: out_accum_buf has (STATS_GRANULARITY * Wt) tiles;
 *   // After:  out_accum_buf has Wt tiles (one reduced row).
 *   matmul_reduce_inplace(out_accum_buf, col_identity_buf, Wt, STATS_GRANULARITY);
 */
template <typename Buf = ::experimental::CircularBuffer>
ALWI void matmul_reduce_inplace(
    Buf& in_out_buf,
    Buf& in1_buf,
    uint32_t num_subblocks,
    uint32_t subblock_h,
    uint32_t subblock_w = 1,
    uint32_t block_kt = 1);

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
