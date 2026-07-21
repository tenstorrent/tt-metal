// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"

/**
 * @file matmul_block_helpers.inl
 * @brief Implementation of matmul_block helper function.
 *
 * Single pipeline handles both pack strategies:
 *   tile_order=SubblockMajor → sequential pack_tile_block, per-subblock reserve/push
 *   tile_order=TileRowMajor      → absolute-offset pack_tile<true>, per-row-group reserve/push
 *
 * Both modes share K-loop, reload, L1_ACC management, and pre/post callbacks.
 * SKIP_COMPUTE (microbench define) elides the inner matmul LLK call only.
 */

namespace compute_kernel_lib {

/**
 * Pack a (subblock_height × subblock_width) DST sub-block to absolute row-major
 * positions in the output CB — one pack_tile<true>(dst_idx, cb, abs_tile_idx) per
 * tile, so the helper needs no access to CB-interface internals.
 *   dst_start_idx    start in DST; DST[dst_start_idx .. +subblock_height*subblock_width-1]
 *                    packed row-first.
 *   pack_target_id   output CB id.
 *   col_base         column offset within the row group (tiles).
 *   row_stride       row stride in tiles (= out_row_width).
 *   subblock_height  sub-block height in tiles.
 *   subblock_width   sub-block width in tiles.
 * PRECONDITION: caller reserved >= subblock_height * row_stride tiles in pack_target_id
 * (one M-row-group) and is responsible for the matching cb_push_back.
 */
ALWI void pack_subblock_row_strided(
    uint32_t dst_start_idx,
    uint32_t pack_target_id,
    uint32_t col_base,
    uint32_t row_stride,
    uint32_t subblock_height,
    uint32_t subblock_width) {
    for (uint32_t r = 0; r < subblock_height; r++) {
        const uint32_t row_base = r * row_stride + col_base;
        for (uint32_t c = 0; c < subblock_width; c++) {
            pack_tile<true>(dst_start_idx + r * subblock_width + c, pack_target_id, row_base + c);
        }
    }
}

/**
 * Contiguous mirror of pack_subblock_row_strided for the SubblockMajor in-place accumulate
 * pack: pack ntiles contiguous DST tiles to the fixed absolute offset abs_off in pack_target_id
 * via pack_tile<true> (out-of-order). Used when the helper has done ONE reserve over the whole
 * output block (in-place accumulate) so pack_tile_block's wr_ptr-relative pack would clobber —
 * each SubblockMajor subblock instead lands at its own contiguous region
 * (subblock_idx * out_subblock_num_tiles). PRECONDITION: caller reserved the whole output block in
 * pack_target_id and owns the matching push_back.
 */
ALWI void pack_subblock_at_offset(uint32_t dst_start_idx, uint32_t pack_target_id, uint32_t abs_off, uint32_t ntiles) {
    for (uint32_t t = 0; t < ntiles; t++) {
        pack_tile<true>(dst_start_idx + t, pack_target_id, abs_off + t);
    }
}

/**
 * Read mirror of pack_subblock_row_strided: reload a row-strided-spilled
 * (subblock_height × subblock_width) sub-block into CONTIGUOUS DST. Each row's
 * subblock_width tiles sit at source offset (r * row_stride + col_base) from fifo_rd_ptr
 * and land at DST[r * subblock_width], so the matmul/pack sees the same row-major layout
 * the contiguous (SubblockMajor) reload produces.
 *
 * One copy_block_matmul_partials per row (reads at fifo_rd_ptr + src_base; does NOT advance
 * it). Caller waits the whole fronted row group (col_base + (subblock_height-1)*row_stride +
 * subblock_width tiles) and pops it when done. col_base / row_stride match the spill.
 */
ALWI void copy_subblock_row_strided(
    uint32_t src_cb_id,
    uint32_t col_base,
    uint32_t row_stride,
    uint32_t subblock_height,
    uint32_t subblock_width) {
    for (uint32_t r = 0; r < subblock_height; r++) {
        copy_block_matmul_partials(src_cb_id, r * row_stride + col_base, r * subblock_width, subblock_width);
    }
}

/**
 * Finalize copy: materialize the fully-accumulated tile-format block from interm into out.
 * Runs ONLY when the accumulated block isn't already the output (interm not aliased onto out,
 * or a transform is requested). The K-loop accumulates in the OUTPUT layout (tile_order), so
 * this is a straight 1:1 sequential tile copy (tile i of interm → tile i of out) — no reorder.
 *
 * Applies the requested transforms on the packer as it writes out:
 *   - dtype convert: pack_reconfig_data_format(out) (out may be a lower-precision / different
 *     format than the fp32/fp16_b accumulation).
 *   - relu (apply_relu_on_last_block): llk_pack_relu_config(zero) around the block, restored after.
 *   - Activation (SFPU): apply_activation_from_pack on the fully-accumulated values.
 *
 * Consumes the whole block from interm (wait_front + pop_front over out_block_num_tiles) and
 * produces it into out (reserve_back + push_back). Copies in out_subblock_num_tiles (subblock-sized)
 * DST chunks so DST capacity is never exceeded (out_subblock_h*out_subblock_w <= DST limit,
 * asserted by the caller).
 */
template <bool packer_l1_acc, bool apply_relu_on_last_block, typename Activation, typename Buf>
ALWI void matmul_block_finalize_copy(
    Buf& interm_buf,
    Buf& out_buf,
    uint32_t out_block_num_tiles,
    uint32_t out_subblock_num_tiles,
    uint32_t prev_srca_cb_id) {
    const uint32_t interm_cb_id = buf_id(interm_buf);
    const uint32_t out_cb_id = buf_id(out_buf);

    // Bring srcA onto interm and the packer onto out's format (dtype convert lands here).
    // The K-loop left the unpacker's srcA data-format programmed for the matmul's srcA operand
    // (in1 — see the matmul convention). copy_tile_to_dst_init_short_with_dt only reprograms the
    // srcA format when old_cbid != new_cbid, so old_cbid MUST be that prior srcA operand
    // (prev_srca_cb_id = in1), not interm — otherwise the reconfig is a no-op and the datacopy
    // reads interm's tiles through the stale (in1) source format, producing bit garbage / inf
    // whenever interm's format differs from in1's (e.g. fp32 interm + bfp-packed in1).
    copy_tile_to_dst_init_short_with_dt(prev_srca_cb_id, interm_cb_id);
    PACK((pack_reconfig_data_format(out_cb_id)));
    // The accumulation packed with l1_acc enabled; the copy must NOT re-accumulate.
    if constexpr (packer_l1_acc) {
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
    if constexpr (apply_relu_on_last_block) {
        PACK((llk_pack_relu_config(ReluConfig::zero())));
    }

    interm_buf.wait_front(out_block_num_tiles);
    out_buf.reserve_back(out_block_num_tiles);
    for (uint32_t off = 0; off < out_block_num_tiles; off += out_subblock_num_tiles) {
        tile_regs_acquire();
        copy_block_matmul_partials(interm_cb_id, off, 0, out_subblock_num_tiles);
        tile_regs_commit();
        // Activation (SFPU) applies to the fully-accumulated values as they pack out. With NONE,
        // the standard 4-phase wait; apply_activation_from_pack replaces tile_regs_wait otherwise.
        if constexpr (Activation::activation != KernelActivation::NONE) {
            apply_activation_from_pack<
                Activation::activation,
                Activation::param0,
                Activation::param1,
                Activation::param2>(out_subblock_num_tiles);
        } else {
            tile_regs_wait();
        }
        pack_tile_block(0, out_cb_id, out_subblock_num_tiles);
        tile_regs_release();
    }
    out_buf.push_back(out_block_num_tiles);
    interm_buf.pop_front(out_block_num_tiles);

    // Restore a clean packer relu state for the next consumer/op.
    if constexpr (apply_relu_on_last_block) {
        PACK((llk_pack_relu_config(ReluConfig::none())));
    }
}

template <
    bool transpose,
    bool packer_l1_acc,
    LastBlockTarget last_block_target,
    OutputCBLayout tile_order,
    matmul_config::InitMode init_mode,
    InputPolicy in0_policy,
    InputPolicy in1_policy,
    typename PostComputeFn,
    typename PreKBlockFn,
    typename PostKBlockFn,
    typename KBlockInnerDimFn,
    typename In0SourceFn,
    typename In1BaseOffsetFn,
    typename Activation,
    matmul_config::DataFormatReconfig reconfig,
    uint32_t c_in0_num_subblocks,
    uint32_t c_in1_num_subblocks,
    uint32_t c_out_subblock_h,
    uint32_t c_out_subblock_w,
    uint32_t c_in0_block_k,
    uint32_t c_num_k_blocks,
    uint32_t c_batch,
    uint32_t c_last_in1_subblock_w_valid,
    uint32_t c_in1_per_core_w,
    uint32_t c_out_row_width,
    typename Buf>
ALWI void matmul_block_impl(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    const MatmulBlockShape& shape_rt,
    PostComputeFn post_compute,
    PreKBlockFn pre_k_block,
    PostKBlockFn post_k_block,
    KBlockInnerDimFn k_block_inner_dim,
    In0SourceFn in0_source_fn,
    In1BaseOffsetFn in1_base_offset_fn) {

    // Compile-time block-shape opt-in (see .hpp): when c_num_k_blocks != 0 the caller passed the
    // block dims as template params — rebuild a constexpr MatmulBlockShape so its fields fold to
    // immediates. The runtime `shape_rt` reference does NOT const-fold through this ALWI at -O3,
    // leaving every loop bound / derived tile-count a runtime load (a per-subblock pack-thread
    // tax on high-iteration pack-bound callers). c_num_k_blocks == 0 (default) selects shape_rt,
    // so non-opting callers keep the exact runtime path. Field order mirrors MatmulBlockShape::of.
    constexpr MatmulBlockShape shape_ct{
        c_in0_num_subblocks,
        c_in1_num_subblocks,
        c_out_subblock_h,
        c_out_subblock_w,
        c_in0_block_k,
        c_num_k_blocks,
        c_batch,
        c_last_in1_subblock_w_valid,
        c_in1_per_core_w,
        c_out_row_width};
    const MatmulBlockShape& shape = (c_num_k_blocks != 0) ? shape_ct : shape_rt;

    // NoWaitNoPop is in1-only: cross-chip global-CB receivers and L1-sharded weight CBs
    // are in1-side patterns; in0 has no analogous external-management case.
    static_assert(
        in0_policy != InputPolicy::NoWaitNoPop,
        "InputPolicy::NoWaitNoPop is only valid for in1 — use WaitAndPopPerKBlock or "
        "WaitAndRetainOnLastBlock for in0.");
    // last_block_target == Interm feeds a downstream phase; whether activation belongs HERE
    // (matmul pack) or in that phase is the caller's choice via the Activation parameter (the
    // helper can't infer it). Set Activation here only when the downstream phase reads the
    // partials unchanged; otherwise leave it NoneActivation and activate downstream.

    // ══ OUTPUT PUBLICATION — one principle, four structural exceptions ═══════════
    // Every output block reaches its consumer by ONE rule:
    //
    //   Stream-publish each subblock (SubblockMajor) / row-group (TileRowMajor) the MOMENT its
    //   fully-accumulated sum is available to the PACKER, on the last K-block — so the output
    //   writer drains it while the remaining subblocks still compute. Fall back to a single
    //   whole-block publish + a SEPARATE finalize pass ONLY when the final sum is structurally
    //   not packer-ready per-subblock.
    //
    // The streaming case is the two paths where the last pack yields the finished sum:
    //   • non-l1_acc reload — the software reload already added every prior K-block into DST, so
    //                         the last pack IS the sum → pack straight to out, publish per-subblock.
    //   • l1_acc "alias_out" — the sum is accumulated in place directly in out_buf via
    //                         packer_l1_acc → the last pack completes it in out → publish per-subblock.
    //
    // The whole-block finalize is forced ONLY by these four STRUCTURAL reasons — each a property
    // of the config, never of the shape or a specific test:
    //   1. Interm target     — the consumer is a downstream IN-KERNEL op (bias-add, untilize), not
    //                          the NoC writer; the block is handed over whole (there is no "publish").
    //   2. converting format — packer_l1_acc must accumulate in a WIDE interm (you cannot l1_acc into
    //                          a narrow out, e.g. bf8); publishing needs a convert-READ of interm,
    //                          which the single-reserve accumulate has no pack→unpack barrier for →
    //                          the finalize's wait_front supplies the barrier and converts the copy.
    //   3. SFPU activation   — the activation runs in DST on the full sum; under l1_acc the sum lives
    //                          in L1, not DST, so the finalize brings it to DST first.
    //   4. relu under l1_acc — relu must apply to the POST-accumulation sum; the finalize relus the
    //                          full sum in DST (a per-partial relu would clamp each K-block pre-sum).
    //
    // Streaming is applied exactly where it helps (small-N reader/writer-bound paths, via writer
    // overlap), NOT forced uniformly onto paths a whole-block finalize already serves well.
    //
    // ── Implementation of the above (flags) ──────────────────────────────────────
    // needs_finalize = (buf_id(out) != buf_id(accum)); false → the accumulated block already IS out
    // (alias_out, streamed) → zero-copy skip. Interm target consumes from interm directly → no finalize.
    constexpr bool target_is_interm = (last_block_target == LastBlockTarget::Interm);
    // Relu on the FULLY-ACCUMULATED block (OutWithRelu). Drives relu on both the finalize copy
    // (in_place) and the non-l1_acc straight-to-out last pack — hence "last block", not "finalize".
    // Interm defers relu to its downstream phase.
    constexpr bool apply_relu_on_last_block = (last_block_target == LastBlockTarget::OutWithRelu);

    // in_place: the (packer_l1_acc + accumulate-to-interm) config packs in place — one reserve_back
    // over the whole output block before the K-loop, skipping the per-K-block reserve/push/drain. Each
    // K-block packs to absolute offsets in that fixed region and packer_l1_acc accumulates in place.
    // The block is then published (issued INTERNALLY here, per batch) EITHER as one push_back after the
    // K-loop (the default) OR, for the alias_out SubblockMajor case, per-subblock on the last K-block so
    // the output writer overlaps the remaining subblocks' compute (see alias_out below). packer_l1_acc
    // makes in-place L1 accumulation the whole point; without it the accumulation path is a software
    // spill+reload, which does NOT accumulate in interm and must not take this path. Both layouts place
    // each subblock correctly via an absolute-offset pack: TileRowMajor row-strided (M-row-group base
    // folded into the OOP offset), SubblockMajor contiguous (subblock_idx * out_subblock_num_tiles).
    constexpr bool in_place = packer_l1_acc;

    // pack_last_to_interm: land the FINAL K-block in interm (vs straight to out). Do this ONLY when
    // something downstream reads it from interm: a downstream in-kernel op (Interm target), or the
    // finalize that materializes an in-place-accumulated block into out (in_place). For the remaining
    // case — the non-l1_acc software-reload Out/OutWithRelu path — the last block is already the
    // finished sum in DST, so it packs STRAIGHT TO OUT. Routing that path through interm would (a) add
    // a redundant full-block copy and, worse, (b) DEADLOCK: the last block reloads the prior spill from
    // interm while re-reserving a row-group in the SAME headroom-free interm CB (factory sizes interm0
    // == out_block), a classic reserve-before-pop circular wait (see the TileRowMajor row-group reserve
    // below + docs/hangs.md "Compute-internal deadlock variant").
    constexpr bool pack_last_to_interm = target_is_interm || in_place;

    // activate_on_last_pack: the straight-to-out Out/OutWithRelu path (non-l1_acc, no interm round-trip)
    // has no finalize, so relu (apply_relu_on_last_block) and the SFPU Activation apply HERE, on the last-block
    // pack — correct because that block is the fully-accumulated sum (num_k_blocks==1, or the software
    // reload has already added every prior K-block into DST). in_place / Interm defer these to the
    // finalize / downstream phase (a per-partial relu under packer_l1_acc would relu each partial before
    // accumulation).
    constexpr bool activate_on_last_pack = !pack_last_to_interm;

    // Cache integer IDs for legacy LLK calls. buf_id() resolves to
    // get_cb_id() on CircularBuffer or get_id() on DataflowBuffer.
    const uint32_t in0_cb_id = buf_id(in0_buf);
    const uint32_t in1_cb_id = buf_id(in1_buf);
    const uint32_t out_cb_id = buf_id(out_buf);
    const uint32_t interm_cb_id = buf_id(interm_buf);

    // ── In-place-into-OUT alias (skip the redundant finalize copy) ───────────────
    // For the in_place (packer_l1_acc) Out-target path the K-loop accumulates into interm and a
    // finalize step then copies interm→out. When interm and out carry the SAME tile data format
    // (e.g. bf16 out) that finalize is a REDUNDANT same-format copy: it runs as a serial tail after
    // the whole K-loop and delays the output writer's drain (measured: BRISC +~4.4µs / +5% on the
    // reader/writer-bound small-N DeepSeek matmuls — the writer can't start until the copy pushes out).
    // alias_out routes the in-place accumulator DIRECTLY into out_buf instead, so needs_finalize
    // (out != accum) is false and the copy is skipped — while keeping the single-reserve +
    // pack_subblock_at_offset + packer_l1_acc in-place accumulation fully intact.
    //
    // Eligibility (compile-time). Aliasing is only safe when the finalize would be a PURE copy:
    //   in_place                     — only the packer_l1_acc in-place path has a finalize to skip.
    //   tile_order == SubblockMajor  — the regressing case; TileRowMajor is left on the finalize
    //                                  path unchanged (its row-strided reserve/pad geometry is not
    //                                  in scope here).
    //   !target_is_interm            — Interm feeds a downstream in-kernel op that reads from interm.
    //   !apply_relu_on_last_block && no SFPU Activation — relu / SFPU activation are finalize-only for in_place
    //                                  (a per-partial apply under packer_l1_acc would transform each
    //                                  partial before the sum); aliasing would silently drop them.
    //   !fp32_dest_acc_en            — with fp32 DEST the factory makes interm Float32 ≠ out; keep
    //                                  the converting finalize.
    // Runtime format guard: even with !fp32_dest_acc_en the factory sets a packer_l1_acc interm to
    // Float16_b while out may be a narrower format (e.g. bf8), so accumulating directly in out would
    // change numerics. unpack_src_format is the one data-format array emitted on ALL THREE TRISC
    // threads (re-emitted on PACK for the BH tilize workaround) and is equalized to each CB's real
    // format (incl. the pack-only out CB), so compare it. A mismatch keeps the down-converting finalize.
    constexpr bool alias_out_eligible =
        in_place && (tile_order == OutputCBLayout::SubblockMajor) &&
        (last_block_target != LastBlockTarget::Interm) && (last_block_target != LastBlockTarget::OutWithRelu) &&
        (Activation::activation == KernelActivation::NONE) && !get_fp32_dest_acc_enabled();
    const bool alias_out =
        alias_out_eligible && (unpack_src_format[out_cb_id] == unpack_src_format[interm_cb_id]);
    // The in-place accumulation buffer: out_buf when aliased (finalize skipped), else interm_buf
    // (identical to the prior behavior — accum_cb_id == interm_cb_id whenever not aliased).
    Buf& accum_buf = alias_out ? out_buf : interm_buf;
    const uint32_t accum_cb_id = alias_out ? out_cb_id : interm_cb_id;

    // Fail fast on shape / CB invariants before doing any init or pipeline work.
    ASSERT(shape.in0_block_k > 0);
    ASSERT(shape.in0_num_subblocks > 0);
    ASSERT(shape.in1_num_subblocks > 0);
    ASSERT(shape.num_k_blocks > 0);
    ASSERT(shape.out_subblock_h > 0);
    ASSERT(shape.out_subblock_w > 0);
    ASSERT(shape.batch > 0);
    ASSERT(in0_cb_id != out_cb_id);
    ASSERT(in1_cb_id != out_cb_id);
    ASSERT(shape.out_subblock_h * shape.out_subblock_w <= compute_kernel_lib::DEST_AUTO_LIMIT);
    // Unsupported (PR #47724): TileRowMajor + software-reload (non-l1_acc) with interm and out on the
    // SAME physical L1. The multi-K reload keeps partials subblock-major in interm while the last
    // K-block packs row-strided into out; sharing L1 lets the row-strided write clobber not-yet-
    // reloaded partials (silent value corruption). TRM software-reload needs interm as its OWN region;
    // packer_l1_acc TRM may alias (single reserve, no reload) and num_k_blocks == 1 has no reload.
    // Compare PHYSICAL L1 bases (fifo_limit - fifo_size), NOT CB indices: a factory can bind two
    // DISTINCT CB indices to the SAME globally-allocated address (e.g. conv's partials-onto-out alias),
    // which an index compare misses. Debug-only tripwire — the real guard is the factory refusing to
    // alias this combo (matmul do_not_inplace_interm0_out_CB / conv can_alias_partials_onto_out).
    if constexpr (tile_order == OutputCBLayout::TileRowMajor && !packer_l1_acc) {
        const uint32_t interm_l1_base =
            get_local_cb_interface(interm_cb_id).fifo_limit - get_local_cb_interface(interm_cb_id).fifo_size;
        const uint32_t out_l1_base =
            get_local_cb_interface(out_cb_id).fifo_limit - get_local_cb_interface(out_cb_id).fifo_size;
        ASSERT(shape.num_k_blocks == 1 || interm_l1_base != out_l1_base);
    }

    // Reconfig and init are independent compile-time gates (see the InitMode /
    // DataFormatReconfig enums). The pack reconfig targets interm_cb_id (where non-last
    // K-blocks spill); the last-block in-loop reconfig swaps to out_cb_id. The init is
    // always short — the hw_configure-bearing boot init is the caller's.
    // ShortAfterPreKBlock relocates this whole block into the K-loop (after pre_k_block),
    // so it is skipped here for that mode.
    if constexpr (init_mode != matmul_config::InitMode::ShortAfterPreKBlock) {
        if constexpr (
            reconfig == matmul_config::DataFormatReconfig::Input ||
            reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
            // Matmul convention: srca takes in1, srcb takes in0.
            reconfig_data_format(in1_cb_id, in0_cb_id);
        }
        // NOTE: this pack reconfig targets interm_cb_id even when num_k_blocks == 1 (no
        // spill). The last-block swap-to-out reconfig is gated on l1_acc / fp32 DEST and may
        // not fire, so a placeholder interm whose format differs from out_buf can leave the
        // packer mis-configured (silent corruption) — the .hpp interm_buf doc requires a
        // same-format placeholder; pass out_buf.
        if constexpr (
            reconfig == matmul_config::DataFormatReconfig::Output ||
            reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
            PACK((pack_reconfig_data_format(accum_cb_id)));
        }
        if constexpr (init_mode == matmul_config::InitMode::Short) {
            matmul_block_init(
                in0_cb_id, in1_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
        }
    }

    const uint32_t out_subblock_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * shape.in0_block_k;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks;
    // in1_per_core_w: actual N-width the producer pushes per K-block (shape.in1_per_core_w).
    // Derived from subblocks by default; callers that pad the in1 width set shape.in1_per_core_w
    // to the real value to avoid wait/pop mismatches.
    uint32_t in1_per_core_w = shape.in1_per_core_w;
    if (in1_per_core_w == 0) {
        in1_per_core_w = shape.out_subblock_w * shape.in1_num_subblocks;
    }
    // out_row_width: N-tiles per row of the OUTPUT CB (TileRowMajor row stride, shape.out_row_width).
    // Defaults to in1_per_core_w (read and pack widths usually coincide); callers that pad the output
    // width above the in1 read width set shape.out_row_width to the larger value.
    uint32_t out_row_width = shape.out_row_width;
    if (out_row_width == 0) {
        out_row_width = in1_per_core_w;
    }
    const uint32_t in1_block_num_tiles = in1_per_core_w * shape.in0_block_k;
    const uint32_t out_block_num_tiles = out_subblock_num_tiles * shape.in0_num_subblocks * shape.in1_num_subblocks;
    const uint32_t row_group_tiles = shape.out_subblock_h * out_row_width;

    for (uint32_t b = 0; b < shape.batch; b++) {
        bool enable_reload = false;

        // in_place: reserve the whole output block ONCE per batch before the K-loop. Each
        // K-block packs to absolute offsets in this fixed region with packer_l1_acc accumulating in
        // place; the helper skips its own per-block reserve/push/drain (gated below on in_place).
        if constexpr (in_place) {
            accum_buf.reserve_back(out_block_num_tiles);
        }

        for (uint32_t block = 0; block < shape.num_k_blocks; block++) {
            const bool last_out = block == (shape.num_k_blocks - 1);

            // Relu is a finalize transform on the in_place (l1_acc) path — applied to the fully-
            // accumulated block; the non-l1_acc straight-to-out path applies it on the last-block pack.

            pre_k_block(block, shape.num_k_blocks, last_out);

            // Per-K-block FMA step count (default = full shape.in0_block_k). The LLK kt_dim
            // below stays shape.in0_block_k (in1 row stride), independent of this loop bound.
            const uint32_t inner_steps = k_block_inner_dim(block, shape.in0_block_k);

            // Per-K-block in0 source (default = bound in0_cb_id). Alternates MUST share
            // in0_cb_id's dataformat — the unpacker config keys on the bound id (see NoIn0Source).
            const uint32_t active_in0_cb_id = in0_source_fn(block, in0_cb_id);
            Buf active_in0_buf(active_in0_cb_id);

            // ShortAfterPreKBlock: restore matmul state HERE, after pre_k_block()'s (possibly
            // state-dirtying) work, so the PreKBlockFn never does the matmul restore. Same gated
            // reconfig + short init as the pre-loop path, but per K-block and keyed on
            // active_in0_cb_id (so an In0SourceFn that swaps in0 restores the right operand).
            if constexpr (init_mode == matmul_config::InitMode::ShortAfterPreKBlock) {
                if constexpr (
                    reconfig == matmul_config::DataFormatReconfig::Input ||
                    reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
                    reconfig_data_format(in1_cb_id, active_in0_cb_id);
                }
                if constexpr (
                    reconfig == matmul_config::DataFormatReconfig::Output ||
                    reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
                    PACK((pack_reconfig_data_format(accum_cb_id)));
                }
                matmul_block_init(
                    active_in0_cb_id,
                    in1_cb_id,
                    transpose,
                    shape.out_subblock_w,
                    shape.out_subblock_h,
                    shape.in0_block_k);
            }

            // Per-K-block in1 base tile offset (default 0). Nonzero reads a different slice
            // of the same fronted in1 region (see NoIn1BaseOffset).
            const uint32_t in1_base_offset = in1_base_offset_fn(block);

            // Full-block wait for both modes. Every caller has the
            // full in0 block resident before invoking the helper, so progressive
            // per-subblock waits are pure polling overhead on TRISCs.
            active_in0_buf.wait_front(in0_block_num_tiles);
            // in1_policy=NoWaitNoPop: caller manages in1 lifecycle externally
            // (cross-chip global-CB receivers; pre-populated L1-sharded in1).
            if constexpr (in1_policy != InputPolicy::NoWaitNoPop) {
                in1_buf.wait_front(in1_block_num_tiles);
            }

            // Pick the buffer the last K-block packs to. The reference here lets the
            // sync calls below dispatch through the right object regardless of branch.
            // For pack_last_to_interm the accumulator is accum_buf (out_buf when aliased, else
            // interm_buf); the non-l1_acc straight-to-out path packs to out_buf.
            Buf& pack_target_buf = pack_last_to_interm ? accum_buf : out_buf;
            const uint32_t pack_target_id = pack_last_to_interm ? accum_cb_id : out_cb_id;

            // SubblockMajor: reserve the full out_block on the first non-last K-block so
            // interm spills don't clobber output when interm shares out's L1 region (the
            // factory's share-buffer layout), and so reserve/wait increments stay uniform
            // across the K-loop. Skipped in the in_place path (the helper reserved the whole
            // block once before the K-loop). (in_place is TileRowMajor-only, so this SBM branch
            // is inert under it — the gate is defensive.)
            if constexpr (tile_order == OutputCBLayout::SubblockMajor && !in_place) {
                if (block == 0 && !last_out) {
                    out_buf.reserve_back(out_block_num_tiles);
                }
            }

            // Non-last K-blocks spill into interm_buf. With TileRowMajor + L1_ACC the spill must
            // match the last block's row-strided layout (Interm accumulates in the same region;
            // Out reloads from it row-strided), so per-address accumulation is correct. Otherwise
            // keep subblock-major so the last-block per-subblock reload reads partials contiguously.
            constexpr bool spill_row_grouped = (tile_order == OutputCBLayout::TileRowMajor) && packer_l1_acc;

            // in_place (packer_l1_acc): the packer's output format (accum_cb_id) and l1_acc mode are
            // BLOCK-invariant — every subblock of this K-block packs to accum_cb_id with the same
            // accumulate flag — so configure the packer ONCE here, exactly like main's hand-written
            // kernel. The per-subblock version (kept below for the fp32-only non-l1_acc path, whose
            // last-vs-non-last pack targets differ) was redundant block-invariant work repeated
            // num_subblocks× per K-block, costing ~30-100% on many-subblock configs (scales with
            // subblock×K-block count; confirmed by per-RISC).
            if constexpr (in_place) {
                PACK((pack_reconfig_data_format(accum_cb_id)));
                PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
            }

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
                if constexpr (tile_order == OutputCBLayout::TileRowMajor && !in_place) {
                    // Row-major path reserves per M-row-group (one row of all N-subblocks).
                    // Smaller than full-block reserve, so shared out/interm buffers don't deadlock.
                    if (last_out) {
                        pack_target_buf.reserve_back(row_group_tiles);
                    } else if constexpr (spill_row_grouped) {
                        interm_buf.reserve_back(row_group_tiles);
                    }
                }

                int in1_index_subblock_offset = in1_base_offset;
                for (uint32_t in1_subblock = 0; in1_subblock < shape.in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    // last_in1_subblock_w_valid: narrow the last in1 subblock's FMA ct_dim to the
                    // columns actually pushed. DST/pack stays full-width. Inert when 0.
                    const uint32_t effective_subblock_w =
                        (shape.last_in1_subblock_w_valid != 0 && in1_subblock == shape.in1_num_subblocks - 1)
                            ? shape.last_in1_subblock_w_valid
                            : shape.out_subblock_w;

                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(in1_cb_id, interm_cb_id);
                        if constexpr (spill_row_grouped) {
                            // Spills landed row-strided, pushed per M-row-group. Front the whole row
                            // group on the first in1 sub-block, gather this sub-block's row-strided
                            // slice into contiguous DST, and pop the group after the last in1 sub-block
                            // — matching the producer's per-row-group reserve/push so increments balance.
                            if (in1_subblock == 0) {
                                interm_buf.wait_front(row_group_tiles);
                            }
                            copy_subblock_row_strided(
                                interm_cb_id,
                                in1_subblock * shape.out_subblock_w,
                                out_row_width,
                                shape.out_subblock_h,
                                shape.out_subblock_w);
                            if (in1_subblock == shape.in1_num_subblocks - 1) {
                                interm_buf.pop_front(row_group_tiles);
                            }
                        } else {
                            interm_buf.wait_front(out_subblock_num_tiles);
                            copy_block_matmul_partials(interm_cb_id, 0, 0, out_subblock_num_tiles);
                            interm_buf.pop_front(out_subblock_num_tiles);
                        }
#ifndef ARCH_QUASAR
                        reconfig_data_format_srca(interm_cb_id, in1_cb_id);
                        matmul_block_init(
                            in0_cb_id, in1_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
#endif
                    }

                    // Compute the output sub-block. SKIP_COMPUTE (microbench) omits only the LLK call.
                    uint32_t dst_index = 0;
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < inner_steps; inner_dim++) {
#ifndef SKIP_COMPUTE
                        // ckernel:: disambiguates the LLK matmul_block from this helper.
                        ckernel::matmul_block(
                            active_in0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            transpose,
                            effective_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_k);
#else
                        (void)in0_index;
                        (void)in1_index;
                        (void)dst_index;
#endif
                        in0_index++;
                        in1_index += in1_per_core_w;
                    }

                    if (last_out) {
                        post_compute(out_subblock_num_tiles);

                        tile_regs_commit();
                        if constexpr (tile_order == OutputCBLayout::SubblockMajor && !in_place) {
                            pack_target_buf.reserve_back(out_subblock_num_tiles);
                        }
                        // Straight-to-out (activate_on_last_pack): apply the SFPU Activation to the
                        // fully-accumulated block as it packs (apply_activation_from_pack replaces the
                        // wait). Accumulate-to-interm paths pack the RAW sum — the finalize copy (or the
                        // downstream phase) applies the Activation there (applying it per-block under
                        // packer_l1_acc would activate each partial before accumulation, not the final sum).
                        if constexpr (activate_on_last_pack && Activation::activation != KernelActivation::NONE) {
                            apply_activation_from_pack<
                                Activation::activation,
                                Activation::param0,
                                Activation::param1,
                                Activation::param2>(out_subblock_num_tiles);
                        } else {
                            tile_regs_wait();
                        }

                        // in_place hoists both pack reconfigs to once-per-K-block (above); only the
                        // fp32-only (non-l1_acc) last-block Out reconfig remains per-subblock here. Its
                        // target is pack_target_id (= out_cb_id for that path), distinct from the spill.
                        if constexpr ((packer_l1_acc || get_fp32_dest_acc_enabled()) && !in_place) {
                            PACK((pack_reconfig_data_format(pack_target_id)));
                        }

                        // Straight-to-out relu (OutWithRelu, non-l1_acc): configure the packer to relu
                        // this block as it writes out; restored after the pack below. In-place / Interm
                        // relu happens in the finalize / downstream phase, so it is not set here.
                        if constexpr (activate_on_last_pack && apply_relu_on_last_block) {
                            PACK((llk_pack_relu_config(ReluConfig::zero())));
                        }

                        if constexpr (in_place && tile_order == OutputCBLayout::SubblockMajor) {
                            // SubblockMajor in-place accumulate: there is no per-subblock reserve
                            // (the helper did ONE reserve over the whole block, FIFO wr_ptr fixed at
                            // the base), so each subblock must pack to its own CONTIGUOUS absolute
                            // region (subblock_idx * out_subblock_num_tiles); pack_tile_block's wr_ptr-relative
                            // pack would clobber all subblocks onto the same slot. Contiguous mirror of
                            // the TileRowMajor row-strided pack below.
                            //
                            // alias_out: pack at the CURRENT fifo base (offset 0) and push_back this
                            // subblock immediately (below) so the output writer can drain it while the
                            // remaining subblocks of this LAST K-block still compute — restoring the
                            // per-subblock reader/writer overlap that main's straight-to-out pack has and
                            // that a single push-at-end loses (the measured BRISC regression). pack_tile<true>
                            // and packer_l1_acc both address off the live fifo_wr_ptr, which each push_back
                            // advances by out_subblock_num_tiles; so after i pushes, offset 0 == base + i*out_subblock_num_tiles
                            // == subblock i's in-place partial → the l1_acc read-add-write lands correctly and
                            // the subblock's absolute layout is preserved. Single reserve + in-place l1_acc are
                            // unchanged; only the push granularity moves from one block-push to per-subblock.
                            const uint32_t abs_off =
                                alias_out ? 0u : (in0_subblock * shape.in1_num_subblocks + in1_subblock) * out_subblock_num_tiles;
                            pack_subblock_at_offset(0, pack_target_id, abs_off, out_subblock_num_tiles);
                        } else if constexpr (tile_order == OutputCBLayout::TileRowMajor) {
                            // Absolute-offset per-tile pack into the row-group reserve; row stride
                            // = out_row_width. The per-row-group reserve supplies the M-row-group
                            // base, leaving only the in1 col offset.
                            //
                            // in_place: there is no per-row-group reserve (the helper did ONE
                            // reserve over the whole block, FIFO wr_ptr fixed at the base), so the
                            // M-row-group base must be folded into the absolute offset here
                            // (in0_subblock * row_group_tiles); otherwise every row group packs onto
                            // row 0 (latent when in0_num_subblocks == 1, garbles when > 1).
                            const uint32_t row_base = in_place ? in0_subblock * row_group_tiles : 0;
                            const uint32_t col_base = row_base + in1_subblock * shape.out_subblock_w;
                            pack_subblock_row_strided(
                                0, pack_target_id, col_base, out_row_width, shape.out_subblock_h, shape.out_subblock_w);
                        } else {
                            pack_tile_block(0, pack_target_id, out_subblock_num_tiles);
                        }

                        tile_regs_release();
                        // Restore a clean (non-relu) packer state for the next consumer/op.
                        if constexpr (activate_on_last_pack && apply_relu_on_last_block) {
                            PACK((llk_pack_relu_config(ReluConfig::none())));
                        }
                        if constexpr (tile_order == OutputCBLayout::SubblockMajor) {
                            if constexpr (!in_place) {
                                pack_target_buf.push_back(out_subblock_num_tiles);
                            } else if (alias_out) {
                                // Aliased in-place: publish this fully-accumulated subblock now so the
                                // writer overlaps the remaining subblocks' compute (see the pack above).
                                // Total per-subblock pushes == out_block_num_tiles, so the block-push at
                                // the end of the batch loop is skipped for alias_out.
                                pack_target_buf.push_back(out_subblock_num_tiles);
                            }
                        }

                    } else {
                        // Non-last K-block: spill partial to interm_buf. spill_row_grouped picks
                        // row-major (to match the last block when accumulating in the same interm
                        // region) or subblock-major (compatible with the software per-subblock reload).
                        tile_regs_commit();
                        // in_place hoists both pack reconfigs to once-per-K-block (above). Only the
                        // fp32-only (non-l1_acc) spill reconfig remains per-subblock; its pack-DF must
                        // match the accumulator's format else spills land in the previous op's format.
                        // Issued before tile_regs_wait: a PACK-thread op that does not depend on the DST
                        // being waited on, so it overlaps the wait.
                        if constexpr ((packer_l1_acc || get_fp32_dest_acc_enabled()) && !in_place) {
                            PACK((pack_reconfig_data_format(accum_cb_id)));
                        }
                        if constexpr (!spill_row_grouped && !in_place) {
                            interm_buf.reserve_back(out_subblock_num_tiles);
                        }
                        tile_regs_wait();

                        if constexpr (in_place && tile_order == OutputCBLayout::SubblockMajor) {
                            // SubblockMajor in-place accumulate spill: same contiguous absolute offset
                            // as the last-block pack, so packer_l1_acc accumulates each subblock in the
                            // same L1 cells across K-blocks.
                            const uint32_t abs_off =
                                (in0_subblock * shape.in1_num_subblocks + in1_subblock) * out_subblock_num_tiles;
                            pack_subblock_at_offset(0, accum_cb_id, abs_off, out_subblock_num_tiles);
                        } else if constexpr (spill_row_grouped) {
                            // in_place: fold the M-row-group base into the offset, same as the
                            // last-block pack above.
                            const uint32_t row_base = in_place ? in0_subblock * row_group_tiles : 0;
                            const uint32_t col_base = row_base + in1_subblock * shape.out_subblock_w;
                            pack_subblock_row_strided(
                                0, interm_cb_id, col_base, out_row_width, shape.out_subblock_h, shape.out_subblock_w);
                        } else {
                            pack_tile_block(0, interm_cb_id, out_subblock_num_tiles);
                        }
                        tile_regs_release();
                        if constexpr (!spill_row_grouped && !in_place) {
                            interm_buf.push_back(out_subblock_num_tiles);
                        }
                    }

                    in1_index_subblock_offset += shape.out_subblock_w;
                }

                if constexpr (tile_order == OutputCBLayout::TileRowMajor && !in_place) {
                    if (last_out) {
                        pack_target_buf.push_back(row_group_tiles);
                    } else if constexpr (spill_row_grouped) {
                        interm_buf.push_back(row_group_tiles);
                    }
                }

                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if constexpr (packer_l1_acc) {
                // Drain the L1_ACC partials in increments matching the producer's push
                // granularity (row_group_tiles when spill_row_grouped, else subblock-sized);
                // the CB API requires uniform increments. Skipped in the in_place path (the
                // helper pushes nothing per block, so there is nothing to drain).
                const uint32_t drain_step = spill_row_grouped ? row_group_tiles : out_subblock_num_tiles;
                if constexpr (pack_last_to_interm) {
                    // No software reload: Interm accumulates in place (and SBM-contiguous reload
                    // offsets wouldn't match the row-strided spill anyway).
                    if constexpr (!in_place) {
                        if (block < shape.num_k_blocks - 1) {
                            for (uint32_t s = 0; s < out_block_num_tiles; s += drain_step) {
                                interm_buf.wait_front(drain_step);
                                interm_buf.pop_front(drain_step);
                            }
                        }
                    }
                    enable_reload = false;
                } else {
                    if constexpr (!in_place) {
                        if (shape.num_k_blocks >= 2 && block < shape.num_k_blocks - 2) {
                            for (uint32_t s = 0; s < out_block_num_tiles; s += drain_step) {
                                interm_buf.wait_front(drain_step);
                                interm_buf.pop_front(drain_step);
                            }
                        }
                    }
                    if (block == shape.num_k_blocks - 2) {
                        enable_reload = true;
                    }
                }
            } else {
                if (shape.num_k_blocks > 1) {
                    enable_reload = true;
                }
            }

            // WaitAndRetainOnLastBlock: skip the pop on the last K-block (caller reuses in0);
            // intermediate blocks always pop. Pops target active_in0_buf (the In0SourceFn CB).
            if constexpr (in0_policy == InputPolicy::WaitAndPopPerKBlock) {
                active_in0_buf.pop_front(in0_block_num_tiles);
            } else {
                if (!last_out) {
                    active_in0_buf.pop_front(in0_block_num_tiles);
                }
            }
            // WaitAndRetainOnLastBlock: skip the pop on the last K-block (caller reuses in1);
            // intermediate blocks pop. NoWaitNoPop: no pop at all (caller manages in1's rd_ptr).
            if constexpr (in1_policy == InputPolicy::WaitAndPopPerKBlock) {
                in1_buf.pop_front(in1_block_num_tiles);
            } else if constexpr (in1_policy == InputPolicy::WaitAndRetainOnLastBlock) {
                if (!last_out) {
                    in1_buf.pop_front(in1_block_num_tiles);
                }
            }

            // PostKBlockFn: after the L1_ACC drain and both input pops, so callers can advance
            // CB rd_ptrs (or other bookkeeping) only once the consumer has read the block.
            post_k_block(block, shape.num_k_blocks, last_out);
        }

        // in_place: publish the fully-accumulated output block (matches the reserve_back at
        // the top of the batch loop). Block 0 packed with l1_acc=0 (fresh) and later blocks with
        // l1_acc=1 (accumulate) — handled per-block above — so no pre-reserve reset is needed here;
        // the post-push reconfig_l1_acc(0) restores a clean packer state for the next consumer/op.
        if constexpr (in_place) {
            // alias_out (SubblockMajor) already published the block per-subblock on the last K-block for
            // writer overlap; every other in_place path publishes the whole block once here.
            if (!alias_out) {
                accum_buf.push_back(out_block_num_tiles);
            }
            if constexpr (packer_l1_acc) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
        }

        // ── Finalize ─────────────────────────────────────────────────────────────
        // The finalize materializes out from the in-place-accumulated interm block. It runs ONLY
        // when the K-loop actually landed the final block in interm (pack_last_to_interm) — i.e. the
        // in_place path — AND that interm isn't already out:
        //   Interm target        → downstream in-kernel op consumes interm → NO finalize (untilize
        //                           output is one such downstream op: LastBlockTarget::Interm + a
        //                           reblock_and_untilize phase in the kernel — see the gather kernels).
        //   Out / OutWithRelu, non-l1_acc → the last block packed STRAIGHT TO OUT (pack_last_to_interm
        //                           is false), relu/Activation already applied → NO finalize.
        //   Out / OutWithRelu, in_place   → needs_finalize = (out != accum). accum is out_buf when the
        //                           block was aliased directly into out (alias_out: SubblockMajor + Out +
        //                           no relu/activation + same tile format + !fp32_dest) → needs_finalize
        //                           false → the accumulated block already IS out → zero-copy skip. Else
        //                           accum is interm_buf → copy interm→out with dtype-convert + relu +
        //                           Activation (fp32 interm, or a bf8/narrower out vs the Float16_b interm).
        if constexpr (pack_last_to_interm && !target_is_interm) {
            const bool needs_finalize = (out_cb_id != accum_cb_id);
            if (needs_finalize) {
                matmul_block_finalize_copy<packer_l1_acc, apply_relu_on_last_block, Activation>(
                    interm_buf, out_buf, out_block_num_tiles, out_subblock_num_tiles, /*prev_srca_cb_id=*/in1_cb_id);
                // The finalize copy left the unpacker in datacopy mode and the packer on out's
                // format. If another batch's K-loop follows, restore matmul unpack/math state and
                // point the packer back at interm so the next accumulation is correct. (Single-batch
                // and Interm targets don't reach here; the next helper invocation's own init restores
                // state when init_mode != None.)
                if (b + 1 < shape.batch) {
                    reconfig_data_format(in1_cb_id, in0_cb_id);
                    PACK((pack_reconfig_data_format(interm_cb_id)));
                    matmul_block_init(
                        in0_cb_id, in1_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
                }
            }
        }
    }
}

}  // namespace compute_kernel_lib
