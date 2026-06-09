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
 * Output-CB tile order selected at compile time.
 *
 * Describes the order tiles land in the OUTPUT CB after the helper finishes — NOT
 * the compute execution order (which is always subblock-by-subblock regardless of
 * this choice). Pick the variant that matches your writer kernel's tile read order;
 * the helper cannot infer this from shape alone because the choice is dictated by
 * the writer's contract, not by in0_num_subblocks / in1_num_subblocks / subblock
 * dimensions.
 *
 * SubblockMajor (default): tiles in the OUTPUT CB are grouped per
 *   subblock — subblock(0,0)'s tiles, then subblock(0,1)'s tiles, ..., then the
 *   next M-row-group's subblocks. Compute issues one sequential pack_tile_block per
 *   subblock at the natural fifo_wr_ptr position. Required by writer kernels that
 *   expect a subblock-ordered tile stream (multicast bmm writers, conv2d, conv3d).
 *
 * TileRowMajor: tiles in the OUTPUT CB are grouped per tile-row — tile(0,0),
 *   tile(0,1), ..., tile(0, N-1), then tile(1,0), tile(1,1), ..., tile(1, N-1),
 *   etc. Compute issues per-tile absolute-offset pack_tile<true> calls into the
 *   caller's row-group reserve, and reserves/pushes per M-row-group. Decouples
 *   subblock choice from the writer-visible layout so factories can grow
 *   subblocks freely, and is the mode SDPA callers require for absolute-offset
 *   partial writes across K chunks.
 *
 *   NOTE: do NOT read TileRowMajor as "ROW_MAJOR_LAYOUT" — the OUTPUT CB is still
 *   tile-format. The "Tile" prefix in the name qualifies the "RowMajor" part:
 *   tiles are grouped in tile-granularity rows in the CB, NOT bytes in a
 *   row-major tensor layout. Read as "tile-row-major", not "tile + row-major
 *   layout".
 *
 * When matmul_block feeds add_bias_bcast_rows, both must use the same
 * OutputCBLayout so the intermediate CB layout matches.
 */
enum class OutputCBLayout { SubblockMajor, TileRowMajor };

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
 *                  TileRowMajor + untilize combination has no caller and isn't
 *                  expressible via the strided fifo math; route through
 *                  Interm + reblock_and_untilize for that path instead.
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
 * Controls whether the helper issues mm_block_init_short. INDEPENDENT of the
 * data-format reconfig switch (see DataFormatReconfig below) — mirrors the
 * tilize_helpers / reduce_helpers / binary_op_helpers pattern where init and
 * reconfig fire on separate compile-time gates.
 *
 * Short  (default) Helper calls mm_block_init_short — only the matmul-specific
 *        llk_unpack_AB_matmul_init + llk_math_matmul_init, no hw_configure. This is
 *        the right mode for every mid-kernel call: it puts the unpacker and math
 *        back into matmul-block mode after an intervening op (tilize, untilize,
 *        reduce, eltwise) reconfigured them, without redoing the slow MMIO writes
 *        in unpack/math/pack hw_configure.
 *
 * None   Helper skips the mm_block_init_short. Use when the caller has already
 *        issued an explicit mm_block_init_short before the helper call, or when
 *        chaining helper invocations whose previous call already configured matmul
 *        state. Independent of `reconfig` — pass reconfig=NONE separately if the
 *        caller also handles dataformat reconfig externally.
 *
 * ShortAfterPreKBlock
 *        Like Short, but the helper issues the reconfig + mm_block_init_short INSIDE
 *        the K-loop, right after pre_k_block() returns each iteration, instead of once
 *        before the loop. Use when the PreKBlockFn dirties matmul state every K-block
 *        (TILIZE / TRANSPOSE) — the helper then OWNS the matmul-state restore, so the
 *        functor does only its own op plus any uninit it owes (see the PreKBlockFn
 *        MATMUL-STATE-RESTORE CONTRACT). The `reconfig` param selects which reconfigs
 *        the per-K-block restore issues, exactly as for Short. Do not combine with a
 *        non-default In0SourceFn whose alternate CBs differ in dataformat from the
 *        operand pre_k_block prepared — the restore keys on active_in0_cb_id and
 *        assumes the per-block in0 carries its declared format.
 *
 * Callers MUST issue mm_block_init() (or compute_kernel_hw_startup + mm_init for
 * scalar-matmul-first kernels) exactly ONCE at the very top of kernel_main, and
 * leave the helper's Short default to handle every subsequent matmul_block call.
 * hw_configure-bearing inits are unsafe to call mid-kernel — they issue slow MMIO
 * writes that can race with executing units (see compute_kernel_hw_startup.h:26-30
 * for the idle-units requirement). Activation init (ActivationInitHelper::init()) is
 * similarly the caller's boot-time responsibility — the helper does not issue it.
 */
enum class InitMode : uint8_t { Short, None, ShortAfterPreKBlock };

/**
 * Data-format reconfiguration mode for the matmul_block helper.
 *
 * Controls whether the helper issues reconfig_data_format on the unpacker (INPUT side)
 * and pack_reconfig_data_format on the packer (OUTPUT side). INDEPENDENT of init_mode
 * — fires on its own gate, mirroring the tilize_helpers / reduce_helpers /
 * binary_op_helpers pattern (NONE/INPUT/OUTPUT/INPUT_AND_OUTPUT).
 *
 * NONE              Skip all reconfiguration. Use when reconfig is handled externally
 *                   (e.g., SDPA wrappers, conv2d's per-K-block tilize PreKBlockFn,
 *                   or callers that paired init_mode=None with their own explicit
 *                   reconfig before invoking the helper) or when formats are known
 *                   to match the previous op.
 * INPUT             Reconfigure unpacker only — reconfig_data_format(in1, in0). Use
 *                   when the unpacker srca/srcb config from the previous op differs
 *                   from in0/in1's format but the packer is already configured for
 *                   the helper's first write target.
 * OUTPUT            Reconfigure packer only — pack_reconfig_data_format(interm).
 *                   Targets interm to match the OLD mm_block_init's 3rd-arg
 *                   behavior (the first non-last K-block spills to interm; the
 *                   in-loop reconfig at the last block — gated on l1_acc / fp32 DEST
 *                   — handles the final swap to out). Use when unpacker is already
 *                   configured (e.g., back-to-back matmul_block calls with the same
 *                   in0/in1) but the previous op's packer target had a different
 *                   format than interm.
 * INPUT_AND_OUTPUT  (default) Reconfigure both. The safe default for any first-time
 *                   or post-non-matmul-op call site.
 *
 * Perf note: each reconfig is a small number of MMIO writes per fired condition.
 * Unnecessary reconfigs cost cycles but never produce wrong results. INPUT_AND_OUTPUT
 * is the always-correct default; narrower modes are perf-tuning opt-ins for callers
 * that track dataformat usage across consecutive ops.
 */
enum class DataFormatReconfig : uint8_t { NONE, INPUT, OUTPUT, INPUT_AND_OUTPUT };

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
 *       MatmulBlockShape::of(in0_sb, in1_sb, h, w, in0_block_k, num_k_blocks),
 *       ...);
 *
 * Naming note: the K-block tile count is `in0_block_k`, not `in0_block_w`. The
 * legacy convention (still present in kernel locals, host-side program config
 * fields, and LLK calls outside the helper) uses operand-local axes — A is M×K
 * so A's "width" is K. The helper's public API uses matmul-dim names instead so
 * a parameter named `_k` always refers to the K dimension, regardless of which
 * operand. Same applies to the KBlockInnerDimFn callback's second arg (`block_k`).
 */
struct MatmulBlockShape {
    uint32_t in0_num_subblocks;  // Output subblock count along M.
    uint32_t in1_num_subblocks;  // Output subblock count along N.
    uint32_t out_subblock_h;     // Output subblock height in tiles.
    uint32_t out_subblock_w;     // Output subblock width in tiles.
    uint32_t in0_block_k;        // K per K-block in tiles (= legacy "in0_block_w" / A's per-core K-block size).
    uint32_t num_k_blocks;       // Number of K-blocks along the K dimension.
    uint32_t batch = 1;          // Independent batch slices. Pass the actual batch count
                                 // when the caller has no per-batch work between matmuls
                                 // (matmul-only kernels): the helper's own batch loop runs
                                 // its init (mm_block_init_short + the requested reconfig)
                                 // exactly once across all batches, which is both faster
                                 // and avoids the heterogeneous-tile-shape re-init
                                 // corruption fixed in commit 76e99730d2e. Keep batch=1
                                 // and run the kernel's own batch loop ONLY when per-batch
                                 // phase work (bias add, untilize, mailbox sync) must be
                                 // interleaved between iterations.

    // Optional narrowing of the last in1 subblock's matmul FMA width. 0 = inert (use
    // out_subblock_w throughout). Nonzero = on the last in1_subblock iteration only,
    // pass this value as the matmul ct_dim instead of out_subblock_w so the unpacker
    // touches exactly `last_in1_subblock_w_valid` columns. Use case: DRAM-sharded matmul
    // that pads per_core_N_compute beyond per_core_N_in1_sender so out_subblock_w can be
    // larger than the reader actually pushes for the last in1 subblock — without this
    // override the unpacker over-reads padded (unpushed) cb_in1 tiles. The pack lifecycle
    // and output region stay full-width; the writer drops the padded output columns.
    // Mirrors the kernel-side fix from tt-metal #44872.
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
//
// ── MATMUL-STATE-RESTORE CONTRACT ────────────────────────────────────────────
// The two real-world PreKBlockFns — TILIZE (conv2d / conv3d) and TRANSPOSE (matmul
// transposing the next K-block of in1) — both have their own llk init paths and
// often their own hw_configure-style reconfigs, which leaves the unpacker / math
// state pointed at the WRONG operands for matmul. Once your PreKBlockFn touches
// any of those subsystems, the helper's matmul state from the previous K-block
// is invalid for the matmul calls about to fire this K-block.
//
// There are two ways to handle the restore; pick via init_mode:
//
//   (A) init_mode == ShortAfterPreKBlock (preferred): the HELPER restores matmul
//       state — it issues the reconfig + mm_block_init_short itself, right after
//       pre_k_block() returns each K-block. Your PreKBlockFn then does ONLY its own
//       op (tilize / transpose) plus any uninit it owes for special modes it
//       engaged; it must NOT issue the matmul restore (doing so would double-init).
//       This keeps the "helper owns short init + data-format reconfig; caller owns
//       uninit" paradigm consistent with the reduce / untilize / reblock helpers.
//
//   (B) init_mode == None (legacy): the CALLER's PreKBlockFn MUST restore matmul
//       state before returning. The canonical restore is
//       mm_block_init_short_with_both_dt(in0_cb, in1_cb, old_in0_cb, old_in1_cb,
//       transpose, ct_dim, rt_dim, kt_dim) — it both reconfigs srca/srcb formats AND
//       re-issues llk_unpack_AB_matmul_init + llk_math_matmul_init. The helper does
//       NOT redo this restore for you in this mode; the per-K-block reload path
//       inside the helper covers reload-time state but does not run on every block.
//       Still used by conv3d and the gathered / ring CCL kernels.
//
// conv2d (ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp,
// ConvTilizePreKBlock) is the canonical mode-(A) caller; conv3d / gathered are the
// remaining mode-(B) callers and are candidates to migrate to (A).
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
// static block_k so the loop runs the full K-tile span — preserving prior
// helper behavior. Callers that pad in0/in1 K-tiles for some K-blocks override
// this to return the unpadded step count for those blocks.
//
// The inner LLK call's kt_dim argument stays block_k in both cases — that
// parameter is the in1 row stride in L1, NOT the FMA step count. Returning
// a smaller value here only shrinks the loop bound; the LLK still strides
// through in1's tile geometry as if it were full-width.
struct NoKBlockInnerDimFn {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t block_k) const { return block_k; }
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
 *   #include "api/compute/matmul.h"                              // for mm_block_init() at boot
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp" // for the helper itself
 *   // If the kernel's first matmul passes through scalar matmul_tiles instead of
 *   // matmul_block, replace the matmul.h include with:
 *   //   #include "api/compute/compute_kernel_hw_startup.h"  // for compute_kernel_hw_startup()
 *   //   #include "api/compute/matmul.h"                     // for mm_init()
 *
 * ── CB Contract ────────────────────────────────────────────────────────────
 * matmul_block requires in0_buf, in1_buf, and out_buf to be DISTINCT circular
 * buffers — in-place aliasing (in0 == out or in1 == out) is NOT supported and
 * will silently corrupt FIFO state. The helper reserves the entire output
 * block upfront (out_block_num_tiles per call) and only pops in0 / in1 at the
 * end (or per-K-block under InputPolicy::WaitAndPopPerKBlock), so a writer
 * landing on the same CB as a still-fronted reader will overwrite live tiles.
 *
 * In particular, matmul-as-reduce patterns (e.g. SDPA's row-sum fold against
 * a column-identity tile) MUST allocate a separate output CB. Callers needing
 * the reduce-via-matmul pattern should call matmul_block with num_k_blocks=1
 * + an in1 column of ones + a separate output CB sized identically to the
 * input — see SDPA's `matmul_reduce` wrapper in
 * `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
 * for the canonical pattern.
 *
 * The one supported aliasing is interm_buf overlaying out_buf in L1 (conv2d's
 * `partials_cb_uses_output` path); opt in via `pin_interm_to_captured_base=true`,
 * which adds explicit base-reset code per K-block to keep the writer pinned
 * to a fixed L1 base. See pin_interm_to_captured_base docstring below.
 *
 * One helper serving both standard matmul (non-multicast bmm) and SDPA. Supports two
 * output-pack strategies selected at compile time via tile_order:
 *
 *   tile_order=SubblockMajor (default): sequential pack_tile_block per subblock.
 *     Output lands in subblock order. Required by multicast writers that expect
 *     subblock-order tile stream.
 *   tile_order=TileRowMajor: absolute-offset pack_tile<true> at tile-row positions,
 *     reserve/push per M-row-group. Decouples subblock choice from the output-CB
 *     tile order so the factory can pick larger subblocks (the main SDPA perf path).
 *     Also the mode SDPA callers require for absolute-offset partial writes across
 *     K chunks. The output CB is still tile-format; "Row" here is tile-row
 *     granularity, NOT the TT-Metal ROW_MAJOR_LAYOUT byte layout.
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
 * Init handling: by default the helper calls reconfig_data_format(in1, in0) +
 * pack_reconfig_data_format(interm) + mm_block_init_short() (init_mode=Short,
 * reconfig=INPUT_AND_OUTPUT — the safe default). This restores matmul-mode
 * unpack/math state AND brings the unpacker/packer dataformats back to match in0/in1/interm
 * without redoing hw_configure. Callers are responsible for one boot-time init at the very
 * top of kernel_main — typically mm_block_init() (matmul-first kernels) or
 * compute_kernel_hw_startup() + mm_init() (kernels whose first matmul passes through scalar
 * matmul_tiles). The boot-time init is the ONLY hw_configure-bearing call needed; every
 * subsequent matmul_block invocation should leave init_mode at its Short default.
 *
 * Narrow the reconfig via the `reconfig` template parameter (NONE / INPUT / OUTPUT)
 * only if you are perf-tuning a back-to-back call sequence and can prove the previous
 * op left the unpacker / packer in a matching state. init_mode=None skips both the init
 * AND the reconfig — use when the caller has already issued an explicit mm_block_init_short
 * + reconfig pair before the helper call (the SDPA wrapper pattern). See
 * matmul_config::InitMode and matmul_config::DataFormatReconfig.
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
 *   tile_order        OutputCBLayout: SubblockMajor (default) or TileRowMajor (see above).
 *                     Pick to match your writer kernel's expected tile read order —
 *                     SubblockMajor if the writer reads tiles subblock-by-subblock
 *                     (multicast bmm, conv2d/3d), TileRowMajor if the writer reads tiles
 *                     in tile-row order (SDPA absolute-offset partial writes, matmul
 *                     factories that grow subblocks without breaking writer contracts).
 *                     The helper cannot infer this from MatmulBlockShape alone — the
 *                     choice is dictated by the writer's contract, not by subblock counts
 *                     or sizes. Note: TileRowMajor's output CB is still tile-format;
 *                     do not confuse with TT-Metal's ROW_MAJOR_LAYOUT byte layout.
 *   init_mode         matmul_config::InitMode: Short (default), None, or
 *                     ShortAfterPreKBlock. Short fires reconfig + mm_block_init_short
 *                     once before the K-loop so most callers never have to pair the call
 *                     site with manual init/reconfig. None skips both — used by callers
 *                     (e.g. SDPA wrappers) that hand-pair reconfig + init before invoking
 *                     the helper. ShortAfterPreKBlock relocates the reconfig + short init
 *                     to inside the K-loop, right after pre_k_block() — use it when a
 *                     state-dirtying PreKBlockFn (tilize / transpose) would otherwise have
 *                     to restore matmul state itself (see the PreKBlockFn MATMUL-STATE-
 *                     RESTORE CONTRACT; conv2d is the canonical caller). `Full` has been
 *                     removed; issue mm_block_init() exactly once at kernel boot instead.
 *   reconfig          matmul_config::DataFormatReconfig: INPUT_AND_OUTPUT (default), INPUT,
 *                     OUTPUT, or NONE. Selects which data-format reconfigs the helper issues.
 *                     Independent of init_mode — pass NONE if the caller already handles
 *                     reconfig externally (typically paired with init_mode=None).
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
 *                     count. Receives (block, block_k); returns the number of inner
 *                     iterations to issue. Default returns block_k (full K-tile span).
 *                     Use when some K-blocks have unpadded widths smaller than the
 *                     static block_k (e.g. ring-aware all-gather where each block
 *                     consumes a different chip's shard width). The inner LLK call's
 *                     kt_dim arg stays block_k — only the loop bound shrinks.
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
 *                     `last_block_target=Interm`, `OutputCBLayout::TileRowMajor`,
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
 *                     Default false. When true, the helper reserves interm_buf
 *                     once at entry (out_block_num_tiles), packs each K-block's
 *                     subblocks to fixed tile offsets within that reservation via
 *                     pack_tile<true>, and reloads via copy_block_matmul_partials
 *                     with start_in_tile_index set to the subblock's tile offset.
 *                     No per-K-block reserve/push/pop on interm and no direct
 *                     fifo_rd_ptr / fifo_wr_ptr access — the CB pointers never
 *                     advance off the captured base because the helper never
 *                     push_backs during the K-loop. On the pack_last_to_interm
 *                     path the helper push_backs out_block_num_tiles once at
 *                     exit so the downstream consumer (bias-add, untilize) sees
 *                     the accumulated block; on the !pack_last_to_interm path
 *                     interm holds only K-loop scratch and no end push is needed.
 *                     Required when interm_buf is allocated to alias the output
 *                     buffer in L1 (e.g. conv2d's `partials_cb_uses_output=true`
 *                     path) — without pin, the K-loop's natural push/pop would
 *                     advance the fifo ptrs past the captured base and wrap into
 *                     already-packed output. Constraints: tile_order must be
 *                     SubblockMajor (offset arithmetic is subblock-aligned),
 *                     last_block_target must not be OutWithUntilize
 *                     (pack_untilize_dest doesn't compose with absolute-offset
 *                     packs), and shape.batch must be 1 (the one-shot reservation
 *                     is outside the batch loop).
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
 *                     during the matmul pack. ActivationInitHelper::init() is the
 *                     caller's boot-time responsibility — the helper does not issue it
 *                     for either init_mode (Short or None). Build
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
 *                      DataflowBuffer — pass an CircularBuffer
 *                      object on legacy CB-backed kernels, or an
 *                      DataflowBuffer object on Metal-2.0 / DFB
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
 *   out_row_width      N-tiles per row in the OUTPUT CB layout (row stride for the
 *                      TileRowMajor pack). Defaults to 0, meaning reuse in1_per_core_w. For most factories
 *                      in1 read stride and output pack stride coincide. DRAM-sharded is
 *                      the exception: it reads in1 at per_core_N_in1_sender (unpadded shard
 *                      width) but packs output at per_core_N_compute (padded after subblock-
 *                      growth); those factories must pass the larger pack stride here.
 *
 * @example
 *   // Single-core matmul with defaults — no transpose, no L1 accumulation,
 *   // no activation fusion, SubblockMajor pack, init_mode=Short (default).
 *   // The kernel issues mm_block_init() ONCE at boot; the helper handles every
 *   // subsequent reconfig + short init internally. Valid for any (M, K, N)
 *   // whose K dimension fits in a single K-block (= all Kt tiles fit alongside
 *   // one M and N sub-block in L1). Caller constructs CircularBuffer (or
 *   // DataflowBuffer) once per CB and passes the object; the helper wraps
 *   // wait_front / pop_front / reserve_back / push_back on it. Passes out_buf
 *   // itself as the interm placeholder — see the interm_buf runtime-param
 *   // doc above for why that is the canonical pattern when num_k_blocks == 1.
 *   //
 *   // Boot-time init (caller-owned, runs ONCE at the top of kernel_main):
 *   //   mm_block_init(cb_in0, cb_in1, cb_out,
 *   //                 0,         // transpose
 *   //                 1,         // ct_dim   (= out_subblock_w)
 *   //                 1,         // rt_dim   (= out_subblock_h)
 *   //                 Kt);       // kt_dim   (= in0_block_k)
 *   CircularBuffer in0_buf(cb_in0);
 *   CircularBuffer in1_buf(cb_in1);
 *   CircularBuffer out_buf(cb_out);
 *   matmul_block<>(
 *       in0_buf, in1_buf, out_buf,
 *       out_buf,  // interm placeholder — unread when num_k_blocks == 1
 *       MatmulBlockShape::of(
 *           Mt,    // in0_num_subblocks
 *           Nt,    // in1_num_subblocks
 *           1,     // out_subblock_h
 *           1,     // out_subblock_w
 *           Kt,    // in0_block_k
 *           1));   // num_k_blocks
 *
 * @example
 *   // Row-major output + packer-L1 accumulation across K, no fused bias.
 *   // Template order: transpose, packer_l1_acc, last_block_target, tile_order.
 *   // Buf is deduced from the buffer-object arguments.
 *   matmul_block<false, true, LastBlockTarget::Out, OutputCBLayout::TileRowMajor>(
 *       in0_buf, in1_buf, out_buf, interm_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_k, num_k_blocks));
 *
 * @example
 *   // SDPA-style: row-major pack, retain in0 to reuse Q across K chunks, masked post-compute.
 *   // The SDPA-side wrapper does its own [mm_block_init_short, reconfig_data_format]
 *   // pair externally (for ordering parity with OptionalMaskPostCompute), so the
 *   // helper is invoked with init_mode=None.
 *   // Template slot order: transpose, packer_l1_acc, last_block_target, tile_order,
 *   // init_mode, in0_policy, in1_policy, PostComputeFn.
 *   matmul_block<transpose, false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor,
 *                matmul_config::InitMode::None,
 *                InputPolicy::WaitAndRetainOnLastBlock, InputPolicy::WaitAndPopPerKBlock,
 *                OptionalMaskPostCompute>(
 *       in0_buf, in1_buf, out_buf, in0_buf,  // interm unused when num_k_blocks==1
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             subblock_h, subblock_w, in0_block_k, num_blocks),
 *       OptionalMaskPostCompute{...},
 *       NoPreKBlock{});
 *
 * @example
 *   // FUSE_BIAS path: last K-block packs to interm_buf so add_bias_bcast_rows reads it.
 *   // DRAM-sharded passes explicit in1_per_core_w (shard width) and
 *   // out_row_width (padded pack width). The kernel ran mm_block_init() once at
 *   // boot — the helper's Short default reconfigs + short-inits per call.
 *   matmul_block<in1_transpose_tile, l1_acc, LastBlockTarget::Interm,
 *                output_layout, matmul_config::InitMode::Short,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                PostFn, PreFn>(
 *       in0_buf, in1_buf, out_buf, mm_partials_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_k, num_blocks_inner_dim),
 *       PostFn{}, PreFn{},
 *       in1_block_w,      // in1_per_core_w  (DRAM-sharded shard width)
 *       out_block_w);     // out_row_width   (DRAM-sharded padded pack width)
 *
 * @example
 *   // Per-K-block unpadded inner-dim: each K-block consumes a different chip's
 *   // shard width (ring-aware all-gather pattern). The kernel pads in0/in1 K-tiles
 *   // up to a uniform block_k so CB sizing stays static, then the helper's
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
 *   matmul_block<transpose, l1_acc, LastBlockTarget::Out, OutputCBLayout::SubblockMajor,
 *                matmul_config::InitMode::None,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                NoPostCompute, NoPreKBlock, false, NoPostKBlock,
 *                0, RingInnerDimFn>(
 *       in0_buf, in1_buf, out_buf, interm_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_k, num_blocks),
 *       NoPostCompute{}, NoPreKBlock{},
 *       in1_per_core_w, out_row_width,
 *       NoPostKBlock{}, RingInnerDimFn{unpadded, ring_idx, ring_size});
 *
 * @example
 *   // Ring-aware all-gather matmul: PreKBlockFn captures the next in1 rd_ptr,
 *   // PostKBlockFn commits it after the K-block consumes in1; OutWithUntilize
 *   // untilizes per-subblock output directly through the LLK pack_untilize.
 *   // Template slot order: transpose, packer_l1_acc, last_block_target, tile_order,
 *   // init_mode, in0_policy, in1_policy, PostComputeFn, PreKBlockFn,
 *   // pin_interm_to_captured_base, PostKBlockFn, untilize_block_ct_dim.
 *   matmul_block<in1_transpose_tile, l1_acc, LastBlockTarget::OutWithUntilize,
 *                OutputCBLayout::SubblockMajor, matmul_config::InitMode::None,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                NoPostCompute, RingPreKBlock, false,
 *                RingPostKBlock, out_subblock_num_tiles>(
 *       in0_buf, in1_buf, out_buf, mm_partials_buf,
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_k, num_blocks),
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
 *   // packer_l1_acc, last_block_target, tile_order, init_mode, in0_policy, in1_policy,
 *   // PostComputeFn, PreKBlockFn, pin_interm_to_captured_base.
 *   matmul_block<false, packer_l1_acc, LastBlockTarget::Interm,
 *                OutputCBLayout::SubblockMajor, matmul_config::InitMode::None,
 *                InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock,
 *                ConvSFPUPostCompute, ConvTilizePreKBlock,
 *                true>(  // pin_interm_to_captured_base
 *       cb_mm_in0, cb_in1, cb_matmul_partials, cb_matmul_partials,  // out==interm
 *       MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks,
 *                             out_subblock_h, out_subblock_w,
 *                             in0_block_k, in0_num_blocks_w),
 *       ConvSFPUPostCompute{}, conv_pre_k_block);
 *
 * ── Future perf followups ─────────────────────────────────────────────────
 *
 * Specialized-decode kloop variant (deferred). The pre-helper history had a
 * dedicated kloop matmul specialized for decode shapes — M is mostly 1, the
 * full in0 inner dim fits in L1, and the loop processes in1 in width chunks.
 * That pattern differs from matmul_block's K-major inner loop (which reuses
 * in0 across in1 width by holding a single in0 subblock while sweeping in1
 * subblocks within a K-block): kloop holds the entire in0 resident in L1 and
 * sweeps in1 width with no K-blocking pressure. matmul_block does not
 * currently cover this — decode-shaped configs that could fit full inner-dim
 * in0 and in1 in L1 (height-sharded with small M, large enough cores) pay
 * the K-block reload cost unnecessarily. A future helper variant or a
 * tile_order specialization that recognizes "full in0 inner-dim resident,
 * sweep in1 width only" could close this gap. Documented as an opportunity,
 * not scheduled — most decode workloads on this branch route through SDPA
 * or dedicated decode kernels rather than this helper. (Per Sofija's PR
 * review followup on the kloop removal.)
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
    bool pin_interm_to_captured_base = false,
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
