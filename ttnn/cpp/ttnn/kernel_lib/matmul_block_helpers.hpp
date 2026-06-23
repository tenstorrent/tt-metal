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
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_constraints.hpp"

namespace compute_kernel_lib {

/**
 * Support matrix (every cell supported, no caveats):
 *   tile_order {SubblockMajor, TileRowMajor}
 *     × packer_l1_acc {on, off}
 *     × last_block_target {Out, OutWithRelu, Interm}
 * OutWithUntilize is SubblockMajor-only (route TileRowMajor untilize through
 * Interm + reblock_and_untilize). The default path is FIFO spill/reload: the helper
 * reserves/pushes/pops the pack target in one-block increments per K-block.
 *
 * caller_owns_pack_target opts out of that bookkeeping — the helper skips its own
 * reserve/push/drain; the caller does one reserve before + one push after, packing
 * K-blocks into a fixed caller-owned region. Pairs with TileRowMajor + packer_l1_acc + Interm.
 */

/**
 * Tile order in the OUTPUT CB (compile-time). NOT the compute order (always
 * subblock-by-subblock). Dictated by the writer's read contract, so the helper
 * cannot infer it from shape.
 *
 * SubblockMajor (default): grouped per subblock — subblock(0,0)'s tiles, then
 *   subblock(0,1)'s, etc. One sequential pack_tile_block per subblock.
 * TileRowMajor: grouped per tile-row — tile(0,0)..tile(0,N-1), then row 1, etc.
 *   Per-tile absolute-offset pack, reserve/push per M-row-group. Decouples subblock
 *   choice from writer-visible layout (lets factories grow subblocks); required for
 *   absolute-offset partial writes across K chunks. "Row" is tile-row granularity —
 *   the CB is still tile-format, NOT ROW_MAJOR_LAYOUT byte layout.
 *
 * A downstream phase consuming the output (e.g. a bias add) must use the same value.
 */
enum class OutputCBLayout { SubblockMajor, TileRowMajor };

/**
 * Where the last K-block packs and its post-op (compile-time).
 *
 * Out              (default) pack to out_buf, no relu.
 * OutWithRelu      pack to out_buf with PACK_RELU.
 * Interm           pack to interm_buf for a downstream phase (bias add / untilize);
 *                  any relu lives in that phase, not the matmul.
 * OutWithUntilize  pack through pack_untilize_dest (tile-format DST -> row-major bytes),
 *                  bracketed by pack_untilize_dest_init / pack_untilize_uninit. Requires
 *                  tile_order == SubblockMajor and untilize_block_ct_dim > 0 (= the
 *                  per-call out_subblock_num_tiles). For TileRowMajor untilize, route
 *                  through Interm + reblock_and_untilize.
 */
enum class LastBlockTarget : uint8_t { Out, OutWithRelu, Interm, OutWithUntilize };

/**
 * Per-operand input lifecycle (compile-time). The helper waits/pops the bound CB
 * unless the caller opts out. Mirrors the reduce/binary InputPolicy convention.
 *
 * WaitAndPopPerKBlock      (default) wait_front + pop_front each K-block.
 * WaitAndRetainOnLastBlock wait each K-block; skip the pop on the last so an outer
 *                          loop can reuse the data (in0 reused across K-chunks, or
 *                          in1 weights reused across invocations where num_k_blocks=1).
 * NoWaitNoPop              (in1 only; static_assert on in0) skip wait and pop — caller
 *                          manages in1 lifecycle externally (cross-chip global-CB
 *                          receiver advancing rd_ptr via PostKBlockFn; pre-populated
 *                          L1-sharded in1 with no in-program producer).
 */
enum class InputPolicy : uint8_t { WaitAndPopPerKBlock, WaitAndRetainOnLastBlock, NoWaitNoPop };

namespace matmul_config {

/**
 * Init lifecycle for matmul_block: whether the helper issues mm_block_init_short.
 * Independent of the reconfig switch (separate compile-time gate, per the
 * tilize/reduce/binary helper pattern).
 *
 * Short  (default) issue mm_block_init_short (matmul unpack/math init, no
 *        hw_configure). Correct for every mid-kernel call: restores matmul-block
 *        unpack/math state after an intervening op without the slow hw_configure MMIO.
 * None   skip the init. Use when the caller already issued mm_block_init_short, or when
 *        chaining helper calls that left matmul state configured. Independent of
 *        reconfig (pass reconfig=NONE separately if reconfig is also external).
 * ShortAfterPreKBlock
 *        like Short but issues reconfig + init INSIDE the K-loop, right after
 *        pre_k_block() each iteration. Use when a state-dirtying PreKBlockFn (tilize /
 *        transpose) would otherwise have to restore matmul state itself — the helper
 *        owns the restore instead (see the PreKBlockFn MATMUL-STATE-RESTORE CONTRACT).
 *        The restore keys on active_in0_cb_id, so a non-default In0SourceFn's alternate
 *        CBs must share in0's dataformat.
 *
 * The caller MUST issue mm_block_init() (or compute_kernel_hw_startup + mm_init for
 * scalar-matmul-first kernels) exactly ONCE at the top of kernel_main; hw_configure-
 * bearing inits are unsafe mid-kernel (slow MMIO can race executing units). Likewise
 * ActivationInitHelper::init() is the caller's boot-time job — the helper never issues it.
 */
enum class InitMode : uint8_t { Short, None, ShortAfterPreKBlock };

/**
 * Data-format reconfig for matmul_block: reconfig_data_format on the unpacker (INPUT)
 * and/or pack_reconfig_data_format on the packer (OUTPUT). Independent of init_mode
 * (separate gate, per the tilize/reduce/binary pattern).
 *
 * NONE              skip both. Use when reconfig is external or formats already match.
 * INPUT             unpacker only — reconfig_data_format(in1, in0).
 * OUTPUT            packer only — pack_reconfig_data_format(interm). Targets interm
 *                   because non-last K-blocks spill there; the in-loop swap to out at
 *                   the last block (gated on l1_acc / fp32 DEST) handles the rest.
 * INPUT_AND_OUTPUT  (default) both. Safe for any first-time or post-non-matmul call.
 *
 * Each reconfig is a few MMIO writes; unnecessary ones cost cycles but never corrupt.
 * Narrower modes are perf opt-ins for callers tracking format usage across ops.
 */
enum class DataFormatReconfig : uint8_t { NONE, INPUT, OUTPUT, INPUT_AND_OUTPUT };

}  // namespace matmul_config

/**
 * Block-shape spec for matmul_block — subblock counts, subblock size, K-blocking, batch
 * — so callers pass one struct instead of positional integers. Optional strides
 * (in1_per_core_w / out_row_width) stay on the function signature as advanced overrides.
 * Build with MatmulBlockShape::of(...).
 *
 * Axis naming uses matmul dims: the K-block tile count is `in0_block_k` (legacy code
 * outside the helper calls it `in0_block_w`), so a `_k` parameter always means the K
 * dimension. Same for the KBlockInnerDimFn callback's `block_k`.
 */
struct MatmulBlockShape {
    uint32_t in0_num_subblocks;  // Output subblock count along M.
    uint32_t in1_num_subblocks;  // Output subblock count along N.
    uint32_t out_subblock_h;     // Output subblock height in tiles.
    uint32_t out_subblock_w;     // Output subblock width in tiles.
    uint32_t in0_block_k;        // K per K-block in tiles (= legacy "in0_block_w").
    uint32_t num_k_blocks;       // Number of K-blocks along K.
    uint32_t batch = 1;          // Independent batch slices. Pass the real batch count for
                                 // matmul-only kernels (no per-batch work between matmuls):
                                 // the helper inits once across all batches — faster, and
                                 // avoids heterogeneous-tile-shape re-init corruption. Keep
                                 // batch=1 and loop in the kernel only when per-batch phase
                                 // work must run between iterations.

    // Optional narrowing of the last in1 subblock's FMA width. 0 = inert (out_subblock_w
    // throughout). Nonzero = on the last in1 subblock only, pass this as the matmul ct_dim
    // so the unpacker touches exactly this many columns — for callers whose reader pushes
    // fewer columns than out_subblock_w for the last subblock. Pack region stays full-width;
    // the writer drops the padded output columns.
    uint32_t last_in1_subblock_w_valid = 0;

    static constexpr MatmulBlockShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t in0_block_k,
        uint32_t num_k_blocks,
        uint32_t batch = 1) {
        return {in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_k, num_k_blocks, batch};
    }
};

// No-op post-compute functor (default). Called per output sub-block on the last
// K-block before packing; receives out_subblock_num_tiles (tiles in DST[0..n-1]).
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

// No-op pre-K-block functor (default). Called at the start of each K-block before the
// input waits; receives (block, num_k_blocks, is_last). Use for per-K-block preprocessing
// (e.g. in0 transpose, global-CB pointer manipulation).
//
// ── MATMUL-STATE-RESTORE CONTRACT ────────────────────────────────────────────
// A PreKBlockFn that runs its own init/reconfig (e.g. tilize or transpose) leaves the
// unpacker/math state wrong for the matmul about to fire. Restore it one of two ways via
// init_mode:
//   (A) init_mode=ShortAfterPreKBlock (preferred): the HELPER restores — it issues the
//       reconfig + mm_block_init_short after pre_k_block() each K-block, so the functor
//       does only its own op (+ any uninit it owes) and must NOT also restore matmul
//       state. Matches the reduce/untilize/reblock "helper inits, caller uninits" rule.
//   (B) init_mode=None (legacy): the functor MUST restore before returning, canonically
//       mm_block_init_short_with_both_dt(...) (reconfigs srca/srcb AND re-inits unpack/
//       math). The helper does not redo this for you in this mode.
struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// No-op post-K-block functor (default). Called at the very end of each K-block, after
// the input pop_front and the L1_ACC drain — symmetric to PreKBlockFn. Receives
// (block, num_k_blocks, is_last). Use for per-K-block postprocessing (e.g. advancing a
// CB rd_ptr once the consumer has read the block).
struct NoPostKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Per-K-block inner-dim FMA step-count functor (default returns the static block_k, so
// the loop runs the full K-tile span). Callers that pad in0/in1 K-tiles override this to
// return the unpadded step count for padded blocks. The LLK call's kt_dim stays block_k
// (in1 row stride in L1, not the FMA step count); this only shrinks the loop bound.
struct NoKBlockInnerDimFn {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t block_k) const { return block_k; }
};

// Per-K-block in0 source functor (default returns the bound in0_cb_id). Callers that swap
// between distinct in0 CBs per K-block override this. DATAFORMAT INVARIANT: alternate in0
// CBs MUST share the bound in0_cb_id's dataformat — the unpacker is configured for the
// bound id at boot and at reload and is not re-issued per block, so a mismatched format
// unpacks wrong values. The helper does not check this.
struct NoIn0Source {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t in0_cb_id) const { return in0_cb_id; }
};

// Per-K-block in1 base-offset functor (default returns 0). Override to read a different
// slice of the same fronted in1 region per K-block (e.g. in1_block_num_tiles * pos_idx) —
// an LLK in1_index shift only; the wait/pop lifecycle stays on the bound in1_cb_id. Pair
// with in1_policy=NoWaitNoPop when the caller manages in1's rd_ptr externally.
struct NoIn1BaseOffset {
    ALWI uint32_t operator()(uint32_t /*block*/) const { return 0; }
};

/**
 * matmul_block: sub-blocked tiled matmul C = A × B with K-blocking. One helper for all
 * matmul callers; behavior is selected by the template parameters below.
 *
 * Required includes: "api/compute/matmul.h" (boot-time mm_block_init) and this header.
 * If the kernel's first matmul is a scalar matmul_tiles rather than matmul_block, also
 * include "api/compute/compute_kernel_hw_startup.h" and boot with
 * compute_kernel_hw_startup + mm_init.
 *
 * ── CB contract ─────────────────────────────────────────────────────────────
 * in0_buf, in1_buf, out_buf must be DISTINCT CBs — aliasing silently corrupts FIFO state
 * (the helper reserves the whole output block upfront and pops inputs at the end / per
 * K-block, so a writer on a still-fronted reader's CB overwrites live tiles). A matmul-as-
 * reduce (row-sum against a column of ones) therefore needs its own output CB.
 *
 * interm_buf:
 *   - num_k_blocks == 1: no spill — only a packer-format placeholder; pass out_buf (see
 *     the interm_buf runtime-param note).
 *   - SubblockMajor spill/reload: MAY share out's L1 region (the factory's "share buffer"
 *     layout) — the first-block full-block reserve keeps spills from clobbering output.
 *   - TileRowMajor: must be its OWN region (per-row-group reserve/push can't share).
 *
 * tile_order picks the output-pack layout; packer_l1_acc picks K-accumulation (false =
 * software spill/reload via interm; true = hardware L1 accumulation). See OutputCBLayout
 * and the parameters below.
 *
 * ── Precision (host-side ComputeConfig; the helper sets no fidelity/DEST) ─────
 *   bf16 inputs, Kt == 1 : LoFi or HiFi2.
 *   bf16 inputs, Kt  > 1 : HiFi2 + fp32_dest_acc_en (else the K-accumulation rounds to
 *                          bf16 each step; max-abs error grows ~O(sqrt(K))).
 *   fp32 inputs          : HiFi4 + fp32_dest_acc_en (the only correct combination).
 *   AVOID HiFi4 + fp32_dest_acc_en with bf16 inputs — silent K-accumulator corruption on
 *   Wormhole B0 (issue #38306); use HiFi2/HiFi3.
 *
 * Init/reconfig: by default (init_mode=Short, reconfig=INPUT_AND_OUTPUT) the helper issues
 * the data-format reconfig + mm_block_init_short itself, so callers only do the one
 * boot-time init at the top of kernel_main. See the InitMode / DataFormatReconfig enums
 * for the narrower modes.
 *
 * SKIP_COMPUTE (microbench TU define): omits the inner ckernel::matmul_block call but keeps
 * all pipeline work (waits, reloads, packs, L1_ACC toggles); handled here, caller does
 * nothing. Uses 4-phase DST management (tile_regs_acquire/commit/wait/release).
 *
 * ── Template parameters ──────────────────────────────────────────────────────
 *
 *   transpose          transpose B tiles before the multiply (default false).
 *   packer_l1_acc      hardware L1 K-accumulation instead of software spill/reload.
 *   last_block_target  Out / OutWithRelu / Interm / OutWithUntilize — see LastBlockTarget.
 *   tile_order         SubblockMajor / TileRowMajor — see OutputCBLayout. The helper cannot
 *                      infer this; it follows the writer's read contract.
 *   init_mode          Short / None / ShortAfterPreKBlock — see InitMode.
 *   in0_policy,        per-operand input lifecycle — see InputPolicy (NoWaitNoPop is in1-only).
 *   in1_policy
 *   PostComputeFn      per-subblock hook on the MATH thread, last K-block, before pack.
 *   PreKBlockFn        per-K-block hook before the input waits (see the restore contract).
 *   PostKBlockFn       per-K-block hook after the input pops and the L1_ACC drain.
 *   untilize_block_ct_dim  block_ct_dim for OutWithUntilize (= out_subblock_num_tiles);
 *                          required > 0 for that target, ignored otherwise.
 *   KBlockInnerDimFn   per-K-block FMA step count (for unpadded/partial K-blocks).
 *   In0SourceFn        per-K-block in0 CB selector (alternates must share in0's dataformat).
 *   In1BaseOffsetFn    per-K-block in1 base-offset shift within the fronted region.
 *   caller_owns_pack_target  caller does one reserve before + one push after; the helper
 *                            skips its own reserve/push/drain. Pairs with TileRowMajor +
 *                            packer_l1_acc + Interm.
 *   Activation         fuse an SFPU activation on the PACKER thread at the last-block pack
 *                      (default none); independent of PostComputeFn (MATH thread) and
 *                      allowed with Interm. Build from the sfpu_activation_helpers.hpp
 *                      aliases, or ActivationOp<type,p0,p1,p2> for host-driven kinds. The
 *                      helper does not issue ActivationInitHelper::init() — boot-time, caller's.
 *   reconfig           which data-format reconfigs to issue — see DataFormatReconfig.
 *
 * ── Runtime parameters ───────────────────────────────────────────────────────
 *
 *   in0_buf, in1_buf   A and B (CircularBuffer or DataflowBuffer; Buf is deduced).
 *   out_buf            final result.
 *   interm_buf         spill/reload (software K-accum) or L1-ACC FIFO region. When
 *                      num_k_blocks == 1 there is no spill, but under reconfig
 *                      OUTPUT / INPUT_AND_OUTPUT the helper still issues
 *                      pack_reconfig_data_format to interm's DATA FORMAT before the K-loop,
 *                      so pass out_buf as the zero-cost placeholder (it already carries the
 *                      output format). Any other buffer is safe ONLY if its data format
 *                      matches out_buf — a mismatch silently mis-configures the packer
 *                      width. (Exempt under reconfig=NONE, which issues no pack reconfig.)
 *   shape              MatmulBlockShape — build with MatmulBlockShape::of(...).
 *   post_compute, ...  functor instances for the template hooks above (default {}).
 *   in1_per_core_w     actual N-tiles the producer pushes per K-block. 0 = derive from
 *                      out_subblock_w * in1_num_subblocks. Pass the real value when the
 *                      factory pads the in1 width above the pushed shard width, else the
 *                      wait/pop counts mismatch and deadlock.
 *   out_row_width      N-tiles per row in the OUTPUT CB (TileRowMajor row stride). 0 =
 *                      reuse in1_per_core_w. Pass the larger pack width when the output is
 *                      padded above the in1 read width.
 *
 * @example  // Single K-block, all defaults. Boot once with mm_block_init(cb_in0, cb_in1,
 *           // cb_out, transpose=0, ct_dim=1, rt_dim=1, kt_dim=Kt). Pass out_buf as the
 *           // interm placeholder (unused when num_k_blocks == 1).
 *   CircularBuffer in0_buf(cb_in0), in1_buf(cb_in1), out_buf(cb_out);
 *   matmul_block<>(in0_buf, in1_buf, out_buf, out_buf,
 *       MatmulBlockShape::of(Mt, Nt, 1, 1, Kt, 1));  // in0_sb, in1_sb, sb_h, sb_w, k, num_k
 *
 * @example  // K-blocked, packer-L1 accumulation, TileRowMajor output.
 *   // Template order: transpose, packer_l1_acc, last_block_target, tile_order.
 *   matmul_block<false, true, LastBlockTarget::Out, OutputCBLayout::TileRowMajor>(
 *       in0_buf, in1_buf, out_buf, interm_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                            out_subblock_h, out_subblock_w, in0_block_k, num_k_blocks));
 */
template <
    bool transpose = false,
    bool packer_l1_acc = false,
    LastBlockTarget last_block_target = LastBlockTarget::Out,
    OutputCBLayout tile_order = OutputCBLayout::SubblockMajor,
    matmul_config::InitMode init_mode = matmul_config::InitMode::Short,
    InputPolicy in0_policy = InputPolicy::WaitAndPopPerKBlock,
    InputPolicy in1_policy = InputPolicy::WaitAndPopPerKBlock,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock,
    typename PostKBlockFn = NoPostKBlock,
    uint32_t untilize_block_ct_dim = 0,
    typename KBlockInnerDimFn = NoKBlockInnerDimFn,
    typename In0SourceFn = NoIn0Source,
    typename In1BaseOffsetFn = NoIn1BaseOffset,
    bool caller_owns_pack_target = false,
    typename Activation = NoneActivation,
    matmul_config::DataFormatReconfig reconfig = matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT,
    typename Buf = ::CircularBuffer>
ALWI void matmul_block(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    const MatmulBlockShape& shape,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    uint32_t in1_per_core_w = 0,
    uint32_t out_row_width = 0,
    PostKBlockFn post_k_block = {},
    KBlockInnerDimFn k_block_inner_dim = {},
    In0SourceFn in0_source_fn = {},
    In1BaseOffsetFn in1_base_offset_fn = {});

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
