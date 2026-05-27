// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
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
 * Pack a (h × w) DST sub-block to absolute row-major positions in the output CB.
 * One pack_tile<true>(dst_idx, cb, abs_tile_idx) call per tile — the LLK takes
 * an absolute tile index into the caller's reserved region, so we don't have
 * to touch fifo_wr_ptr / fifo_wr_tile_ptr at all. Earlier versions of this
 * helper batched into h pack_tile_block calls with manual fifo_wr_ptr striding;
 * llk_matmul_pack expands to a per-tile _llk_pack_ loop internally so the
 * underlying packer call count was identical, and the per-call C++ savings
 * (one ASSERT/get_output_id per row instead of per tile) didn't justify the
 * dependency on LLK CB-interface internals. See PR #43676 review thread for
 * the discussion.
 *
 * dst_start_idx     Start position in DST. Tiles DST[dst_start_idx..+h*w-1] are
 *                   packed in row-major order (DST iterates row-first within the
 *                   subblock).
 * pack_target_id    Output CB id.
 * col_base          Column offset within the row group (in tiles).
 * row_stride        Row stride in tiles (= out_row_width).
 *
 * PRECONDITION: caller has reserved at least h * row_stride tiles in pack_target_id
 * (one M-row-group). Caller is responsible for cb_push_back.
 */
ALWI void pack_subblock_row_strided(
    uint32_t dst_start_idx,
    uint32_t pack_target_id,
    uint32_t col_base,
    uint32_t row_stride,
    uint32_t h,
    uint32_t w) {
    for (uint32_t r = 0; r < h; r++) {
        const uint32_t row_base = r * row_stride + col_base;
        for (uint32_t c = 0; c < w; c++) {
            pack_tile<true>(dst_start_idx + r * w + c, pack_target_id, row_base + c);
        }
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
    bool pin_interm_to_captured_base,
    typename PostKBlockFn,
    uint32_t untilize_block_ct_dim,
    typename KBlockInnerDimFn,
    typename In0SourceFn,
    typename In1BaseOffsetFn,
    bool caller_owns_pack_target,
    typename Activation,
    matmul_config::DataFormatReconfig reconfig,
    typename Buf>
ALWI void matmul_block(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    const MatmulBlockShape& shape,
    PostComputeFn post_compute,
    PreKBlockFn pre_k_block,
    uint32_t in1_per_core_w,
    uint32_t out_row_width,
    PostKBlockFn post_k_block,
    KBlockInnerDimFn k_block_inner_dim,
    In0SourceFn in0_source_fn,
    In1BaseOffsetFn in1_base_offset_fn) {

    // OutWithUntilize requires the SubblockMajor pack path: pack_untilize_dest is
    // initialized for a fixed block_ct_dim and packs from DST starting at offset 0,
    // which doesn't compose with the row-major pack_subblock_row_strided
    // that manipulates fifo_wr_ptr per row. The Interm + reblock_and_untilize
    // path handles row-major untilize end-to-end via add_bias_bcast_rows /
    // reblock_and_untilize, so callers needing row-major untilize go that route.
    static_assert(
        last_block_target != LastBlockTarget::OutWithUntilize || tile_order == OutputCBLayout::SubblockMajor,
        "OutWithUntilize requires tile_order == SubblockMajor; route row-major untilize via Interm + reblock_and_untilize");
    // pack_untilize_dest_init's block_ct_dim is a compile-time template arg, so the
    // caller must supply it explicitly when opting into OutWithUntilize.
    static_assert(
        last_block_target != LastBlockTarget::OutWithUntilize || untilize_block_ct_dim > 0,
        "OutWithUntilize requires untilize_block_ct_dim > 0 (= shape.out_subblock_h * shape.out_subblock_w)");
    // NoWaitNoPop is in1-only: cross-chip global-CB receivers and L1-sharded weight CBs
    // are in1-side patterns; in0 has no analogous external-management case.
    static_assert(
        in0_policy != InputPolicy::NoWaitNoPop,
        "InputPolicy::NoWaitNoPop is only valid for in1 — use WaitAndPopPerKBlock or "
        "WaitAndRetainOnLastBlock for in0.");
    // last_block_target == Interm is overloaded between two distinct downstream phases:
    // (1) FUSE_BIAS, where activation BELONGS on the bias helper (after bias add), and
    // (2) untilize_out without FUSE_BIAS, where activation runs HERE (matmul pack)
    //     because the untilize phase just reads the activated partials unchanged.
    // The helper can't tell these apart from its template parameters, so the constraint
    // is enforced at the call site: callers on path (1) pass Activation=NoneActivation
    // here and route activation to the bias helper's Activation slot; callers on path
    // (2) pass Activation=ActivationOp<activation_type, …> here.

    // Decode the LastBlockTarget enum into the legacy bool pair the body branches on.
    // The (Interm + Relu) combination is unrepresentable, so the previous static_assert
    // on the bool pair is now structural.
    constexpr bool pack_last_to_interm = (last_block_target == LastBlockTarget::Interm);
    constexpr bool pack_relu = (last_block_target == LastBlockTarget::OutWithRelu);

    // Cache integer IDs for legacy LLK calls. buf_id() resolves to
    // get_cb_id() on CircularBuffer or get_id() on DataflowBuffer.
    const uint32_t in0_cb_id = buf_id(in0_buf);
    const uint32_t in1_cb_id = buf_id(in1_buf);
    const uint32_t out_cb_id = buf_id(out_buf);
    const uint32_t interm_cb_id = buf_id(interm_buf);

    // Fail fast on shape / CB invariants before doing any init or pipeline work.
    ASSERT(shape.in0_block_w > 0);
    ASSERT(shape.in0_num_subblocks > 0);
    ASSERT(shape.in1_num_subblocks > 0);
    ASSERT(shape.num_k_blocks > 0);
    ASSERT(shape.out_subblock_h > 0);
    ASSERT(shape.out_subblock_w > 0);
    ASSERT(shape.batch > 0);
    ASSERT(in0_cb_id != out_cb_id);
    ASSERT(in1_cb_id != out_cb_id);
    ASSERT(shape.out_subblock_h * shape.out_subblock_w <= compute_kernel_lib::DEST_AUTO_LIMIT);

    // Data-format reconfig and init are two independent switches, mirroring the
    // tilize_helpers / reduce_helpers / binary_op_helpers pattern. Each side fires
    // on its own compile-time gate; callers that already issued an external reconfig
    // (e.g. SDPA wrappers, conv2d's per-K-block tilize PreKBlockFn) pass
    // reconfig=NONE; callers that already issued an external mm_block_init_short
    // (e.g. the same wrappers) pass init_mode=None. The pack reconfig targets
    // interm_cb_id to match the OLD mm_block_init's 3rd-arg behavior — the first
    // non-last K-block spills there; the in-loop reconfig at the last block
    // (gated on l1_acc / fp32 DEST) handles the final swap to out_cb_id. The init
    // is always a short init (never a Full / hw_configure-bearing init — those
    // are caller's boot-time responsibility at the very top of kernel_main).
    // ActivationInitHelper::init() is the caller's boot-time responsibility
    // regardless of either switch — the helper does not issue it.
    if constexpr (
        reconfig == matmul_config::DataFormatReconfig::INPUT ||
        reconfig == matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT) {
        // Matmul convention: srca takes in1, srcb takes in0 (verified against
        // mm_block_init_short_with_dt at matmul.h:378 and the existing reload-time
        // call in the K-loop below).
        reconfig_data_format(in1_cb_id, in0_cb_id);
    }
    if constexpr (
        reconfig == matmul_config::DataFormatReconfig::OUTPUT ||
        reconfig == matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT) {
        PACK((pack_reconfig_data_format(interm_cb_id)));
    }
    if constexpr (init_mode == matmul_config::InitMode::Short) {
        mm_block_init_short(
            in0_cb_id, in1_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_w);
    }

    const uint32_t out_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * shape.in0_block_w;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks;
    // in1_per_core_w: actual N-width of the in1 CB per K-block.
    // Derived from subblocks by default; callers with padded per_core_N_compute must
    // pass the real shard width (per_core_N_in1_sender) to avoid CB wait/pop mismatches.
    if (in1_per_core_w == 0) {
        in1_per_core_w = shape.out_subblock_w * shape.in1_num_subblocks;
    }
    // out_row_width: N-tiles per row of the OUTPUT CB layout (row stride for row_major pack).
    // For most factories the in1 CB width and output pack width coincide, so we default to
    // in1_per_core_w. DRAM-sharded passes the larger padded per_core_N_compute here to keep
    // row_group_tiles / row_pos aligned with what the compute actually packs.
    if (out_row_width == 0) {
        out_row_width = in1_per_core_w;
    }
    const uint32_t in1_block_num_tiles = in1_per_core_w * shape.in0_block_w;
    const uint32_t out_block_num_tiles = out_num_tiles * shape.in0_num_subblocks * shape.in1_num_subblocks;
    const uint32_t row_group_tiles = shape.out_subblock_h * out_row_width;

    // Capture interm_buf rd/wr ptrs once at entry. Used by the pin_interm_to_captured_base
    // path to keep interm_buf operating at a fixed L1 base across the K-loop, matching the
    // original conv2d kernel's per-K-block fifo reset behavior. The initializers are
    // unconditional so the compiler keeps the locals when pin=true, but they're unused
    // (and DCE'd) when pin=false.
    [[maybe_unused]] uint32_t interm_pin_rd_ptr = 0;
    [[maybe_unused]] uint32_t interm_pin_wr_ptr = 0;
    if constexpr (pin_interm_to_captured_base) {
        UNPACK((interm_pin_rd_ptr = get_local_cb_interface(interm_cb_id).fifo_rd_ptr));
        PACK((interm_pin_wr_ptr = get_local_cb_interface(interm_cb_id).fifo_wr_ptr));
    }

    for (uint32_t b = 0; b < shape.batch; b++) {
        bool enable_reload = false;

        for (uint32_t block = 0; block < shape.num_k_blocks; block++) {
            const bool last_out = block == (shape.num_k_blocks - 1);

            if constexpr (pack_relu && !pack_last_to_interm) {
                if (last_out) {
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                }
            }

            pre_k_block(block, shape.num_k_blocks, last_out);

            // Per-K-block inner-dim step count. Default no-op returns shape.in0_block_w so the
            // loop runs the full K-tile span; ring-aware callers override this to
            // shrink the FMA loop on K-blocks whose unpadded width is < shape.in0_block_w.
            // The LLK call's kt_dim arg below stays shape.in0_block_w — that's the in1 row
            // stride in L1, not the FMA step count.
            const uint32_t inner_steps = k_block_inner_dim(block, shape.in0_block_w);

            // Per-K-block in0 source. Default no-op returns the bound in0_cb_id, so
            // active_in0_buf aliases in0_buf and behavior is unchanged. Ring-aware
            // callers swap to an alternate CB on chosen K-blocks; that alternate CB
            // MUST share the same dataformat as in0_cb_id (the kernel-entry
            // mm_block_init and the reload's mm_block_init_short_with_dt below keep
            // using the bound in0_cb_id, so the unpacker config doesn't re-issue
            // when the source flips).
            const uint32_t active_in0_cb_id = in0_source_fn(block, in0_cb_id);
            Buf active_in0_buf(active_in0_cb_id);

            // Per-K-block in1 starting tile offset. Default no-op returns 0, matching
            // the prior behavior of starting the in1 stride from the front of the
            // CB's fronted region. Ring-aware callers without rd_ptr rotation return
            // a non-zero base (e.g. in1_block_num_tiles * curr_ring_idx) to read a
            // different slice of the same fronted in1 buffer per K-block.
            const uint32_t in1_base_offset = in1_base_offset_fn(block);

            // Full-block wait for both modes. Every caller (matmul + SDPA) has the
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
            Buf& pack_target_buf = pack_last_to_interm ? interm_buf : out_buf;
            const uint32_t pack_target_id = pack_last_to_interm ? interm_cb_id : out_cb_id;

            // Legacy/sequential path: reserve the full out_block on the first
            // non-last K-block so interm spills don't overwrite output data
            // when out_buf and interm_buf share the same L1 region (multicast
            // factory layout). Single reserve here keeps all wait_front /
            // reserve_back increments identical across the K-loop, as the
            // CB-API contract requires. Skipped when caller owns pack lifecycle.
            if constexpr (tile_order == OutputCBLayout::SubblockMajor && !caller_owns_pack_target) {
                if (block == 0 && !last_out) {
                    out_buf.reserve_back(out_block_num_tiles);
                }
            }

            // Non-last K-blocks spill into interm_buf. When FUSE_BIAS (pack_last_to_interm)
            // also runs through interm_buf as pack_target, all blocks — non-last and last —
            // write to the same buffer at overlapping positions, so L1_ACC only accumulates
            // correctly if non-last and last share the same layout. In that case we must
            // spill row-major too. Otherwise (software reload path, or !pack_last_to_interm
            // where the last block writes to out_buf), keep non-last subblock-major so the
            // per-subblock reload at the last K-block can read partials contiguously.
            constexpr bool spill_row_grouped = (tile_order == OutputCBLayout::TileRowMajor) && packer_l1_acc && pack_last_to_interm;

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
                if constexpr (tile_order == OutputCBLayout::TileRowMajor && !caller_owns_pack_target) {
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

                    // shape.last_in1_subblock_w_valid narrowing: on the last in1 subblock, if the
                    // caller set the override, narrow the matmul FMA's ct_dim so the unpacker
                    // touches only the columns the reader actually pushed. The dst/pack region
                    // stays full-width; the writer drops padded output columns. Inert (==0)
                    // for non-DRAM-sharded callers.
                    const uint32_t effective_subblock_w =
                        (shape.last_in1_subblock_w_valid != 0 && in1_subblock == shape.in1_num_subblocks - 1)
                            ? shape.last_in1_subblock_w_valid
                            : shape.out_subblock_w;

                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(in1_cb_id, interm_cb_id);
                        interm_buf.wait_front(out_num_tiles);
                        copy_block_matmul_partials(interm_cb_id, 0, 0, out_num_tiles);
                        interm_buf.pop_front(out_num_tiles);
                        mm_block_init_short_with_dt(
                            in0_cb_id, in1_cb_id, interm_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_w);
                    }

                    // Compute output sub-block via hardware block matmul.
                    // SKIP_COMPUTE (microbench) keeps the surrounding pipeline intact but
                    // omits the actual matmul LLK call.
                    uint32_t dst_index = 0;
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < inner_steps; inner_dim++) {
#ifndef SKIP_COMPUTE
                        // ckernel:: disambiguates the LLK matmul_block from this helper.
                        // active_in0_cb_id reflects the In0SourceFn callback's choice
                        // for this K-block; defaults to in0_cb_id.
                        ckernel::matmul_block(
                            active_in0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            transpose,
                            effective_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_w);
#else
                        (void)in0_index;
                        (void)in1_index;
                        (void)dst_index;
#endif
                        in0_index++;
                        in1_index += in1_per_core_w;
                    }

                    if (last_out) {
                        post_compute(out_num_tiles);

                        // OutWithUntilize: bracket the per-subblock pack with the
                        // pack_untilize init/uninit pair so other ops (or the next K-loop
                        // iteration's reload) can resume their own packer config. Init
                        // before commit, uninit after release — matches the original
                        // gathered kernel pattern (one MMIO write per subblock; the host
                        // factory caps the case to a single subblock per K-block via the
                        // out_block_num_subblocks==1 || !untilize_out FATAL).
                        if constexpr (last_block_target == LastBlockTarget::OutWithUntilize) {
                            pack_untilize_dest_init<untilize_block_ct_dim>(pack_target_id);
                        }

                        tile_regs_commit();
                        if constexpr (tile_order == OutputCBLayout::SubblockMajor && !caller_owns_pack_target) {
                            pack_target_buf.reserve_back(out_num_tiles);
                        }
                        // Pack-side sync: apply_activation_from_pack runs SFPU on the
                        // packer thread (TRISC2) and includes its own math/pack semaphore
                        // wait + dest-offset flip + STALLWAIT, replacing (not augmenting)
                        // tile_regs_wait. When Activation::activation == NONE the standard
                        // 4-phase pack-side wait is used.
                        if constexpr (Activation::activation != KernelActivation::NONE) {
                            apply_activation_from_pack<
                                Activation::activation,
                                Activation::param0,
                                Activation::param1,
                                Activation::param2>(out_num_tiles);
                        } else {
                            tile_regs_wait();
                        }

                        if constexpr (packer_l1_acc || get_fp32_dest_acc_enabled()) {
                            PACK((pack_reconfig_data_format(pack_target_id)));
                        }

                        if constexpr (packer_l1_acc) {
                            if constexpr (pack_last_to_interm) {
                                // FUSE_BIAS path: L1 accumulates across all blocks.
                                PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
                            } else {
                                PACK((llk_pack_reconfig_l1_acc(0)));
                            }
                        }

                        if constexpr (tile_order == OutputCBLayout::TileRowMajor) {
                            // Absolute-offset per-tile pack into the caller's row-group
                            // reserve. Row stride uses out_row_width (padded output-pack
                            // width on DRAM-sharded; equal to in1_per_core_w on most
                            // factories).
                            const uint32_t col_base = in1_subblock * shape.out_subblock_w;
                            pack_subblock_row_strided(
                                0, pack_target_id, col_base, out_row_width, shape.out_subblock_h, shape.out_subblock_w);
                        } else if constexpr (last_block_target == LastBlockTarget::OutWithUntilize) {
                            pack_untilize_dest<untilize_block_ct_dim>(pack_target_id);
                        } else {
                            pack_tile_block(0, pack_target_id, out_num_tiles);
                        }

                        tile_regs_release();
                        if constexpr (last_block_target == LastBlockTarget::OutWithUntilize) {
                            pack_untilize_uninit(pack_target_id);
                        }
                        if constexpr (tile_order == OutputCBLayout::SubblockMajor && !caller_owns_pack_target) {
                            pack_target_buf.push_back(out_num_tiles);
                        }

                    } else {
                        // Non-last K-block: spill partial to interm_buf. spill_row_grouped (defined
                        // at the top of the K-block loop body) decides whether to match the
                        // last-block row-major layout (needed when pack_last_to_interm + L1_ACC
                        // accumulate into the same interm_buf buffer) or keep legacy subblock-
                        // major (compatible with software reload's per-subblock read).
                        tile_regs_commit();
                        if constexpr (!spill_row_grouped && !caller_owns_pack_target) {
                            interm_buf.reserve_back(out_num_tiles);
                        }
                        tile_regs_wait();

                        if constexpr (packer_l1_acc || get_fp32_dest_acc_enabled()) {
                            // Pack-DF must match interm_cb format for non-last spills. Without this,
                            // pack DF stays at whatever the previous op (kernel-entry mm_block_init,
                            // or a tilize/transpose pre_k_block) configured — typically the output
                            // CB's format — and intermediate spills land in the wrong format. Mirrors
                            // conv2d's per-K-block `pack_reconfig_data_format(curr_matmul_out_cb)`.
                            PACK((pack_reconfig_data_format(interm_cb_id)));
                        }
                        if constexpr (packer_l1_acc) {
                            PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
                        }

                        if constexpr (spill_row_grouped) {
                            const uint32_t col_base = in1_subblock * shape.out_subblock_w;
                            pack_subblock_row_strided(
                                0, interm_cb_id, col_base, out_row_width, shape.out_subblock_h, shape.out_subblock_w);
                        } else {
                            pack_tile_block(0, interm_cb_id, out_num_tiles);
                        }
                        tile_regs_release();
                        if constexpr (!spill_row_grouped && !caller_owns_pack_target) {
                            interm_buf.push_back(out_num_tiles);
                        }
                    }

                    in1_index_subblock_offset += shape.out_subblock_w;
                }

                if constexpr (tile_order == OutputCBLayout::TileRowMajor && !caller_owns_pack_target) {
                    if (last_out) {
                        pack_target_buf.push_back(row_group_tiles);
                    } else if constexpr (spill_row_grouped) {
                        interm_buf.push_back(row_group_tiles);
                    }
                }

                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if constexpr (packer_l1_acc) {
                // Wait/pop the L1_ACC partials in increments that match the producer's push
                // granularity: row_group_tiles when spill_row_grouped (FUSE_BIAS + L1_ACC path
                // pushes per M-row-group), otherwise subblock-sized. The CB API requires
                // identical increments across all waits. Skipped on the caller-owns-pack
                // path because the helper isn't pushing per block — there's nothing to drain.
                const uint32_t drain_step = spill_row_grouped ? row_group_tiles : out_num_tiles;
                if constexpr (pack_last_to_interm) {
                    if constexpr (!caller_owns_pack_target) {
                        if (block < shape.num_k_blocks - 1) {
                            for (uint32_t s = 0; s < out_block_num_tiles; s += drain_step) {
                                interm_buf.wait_front(drain_step);
                                interm_buf.pop_front(drain_step);
                            }
                        }
                    }
                    enable_reload = false;
                } else {
                    if constexpr (!caller_owns_pack_target) {
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

            // pin_interm_to_captured_base: mirror conv2d's per-K-block fifo reset so the
            // K-loop's natural rd/wr ptr advance doesn't desync from the output buffer when
            // interm_buf aliases out_buf in L1. See the helper's @param docstring for the
            // why; the iteration logic is conv2d's original (lines 478-510) verbatim.
            //
            // pack_last_to_interm  : reset rd+wr while block < num-1     (original FUSE_BIAS path).
            // !pack_last_to_interm : reset rd while block < num-1, reset wr while block < num-2 —
            //                        the wr ptr is allowed to advance one block on the
            //                        second-to-last iteration so the last K-block's reload
            //                        finds those partials at advanced wr_ptr (original
            //                        !FUSE_BIAS path).
            // packer_l1_acc tightens the wr/rd reset to block < num-2 in the !pack_last path,
            // matching the original kernel's L1_acc drain bound.
            if constexpr (pin_interm_to_captured_base) {
                if (shape.num_k_blocks > 1) {
                    if constexpr (pack_last_to_interm) {
                        if (block < shape.num_k_blocks - 1) {
                            UNPACK((get_local_cb_interface(interm_cb_id).fifo_rd_ptr = interm_pin_rd_ptr));
                            PACK((get_local_cb_interface(interm_cb_id).fifo_wr_ptr = interm_pin_wr_ptr));
                        }
                    } else if constexpr (packer_l1_acc) {
                        if (block < shape.num_k_blocks - 2) {
                            UNPACK((get_local_cb_interface(interm_cb_id).fifo_rd_ptr = interm_pin_rd_ptr));
                            PACK((get_local_cb_interface(interm_cb_id).fifo_wr_ptr = interm_pin_wr_ptr));
                        }
                    } else {
                        if (block < shape.num_k_blocks - 1) {
                            UNPACK((get_local_cb_interface(interm_cb_id).fifo_rd_ptr = interm_pin_rd_ptr));
                        }
                        if (block < shape.num_k_blocks - 2) {
                            PACK((get_local_cb_interface(interm_cb_id).fifo_wr_ptr = interm_pin_wr_ptr));
                        }
                    }
                }
            }

            // in0_policy=WaitAndRetainOnLastBlock: SDPA reuses Q across K chunks, so
            // caller keeps in0 front on the last iteration. Intermediate blocks always
            // pop. Pop targets active_in0_buf (the In0SourceFn-selected CB for this
            // iteration) so ring-aware callers swapping CBs per K-block pop from the
            // right one.
            if constexpr (in0_policy == InputPolicy::WaitAndPopPerKBlock) {
                active_in0_buf.pop_front(in0_block_num_tiles);
            } else {
                if (!last_out) {
                    active_in0_buf.pop_front(in0_block_num_tiles);
                }
            }
            // in1_policy=WaitAndRetainOnLastBlock: conv3d reuses weights across multiple
            // matmul invocations (each invocation has shape.num_k_blocks=1, last_out is always
            // true on its only K-block, so the helper never pops). Intermediate K-blocks
            // within a multi-K-block invocation still pop.
            // in1_policy=NoWaitNoPop elides the pop entirely; the helper isn't managing
            // in1 rd_ptr at all on that path.
            if constexpr (in1_policy == InputPolicy::WaitAndPopPerKBlock) {
                in1_buf.pop_front(in1_block_num_tiles);
            } else if constexpr (in1_policy == InputPolicy::WaitAndRetainOnLastBlock) {
                if (!last_out) {
                    in1_buf.pop_front(in1_block_num_tiles);
                }
            }

            // PostKBlockFn: symmetric counterpart to PreKBlockFn. Fires after the
            // L1_ACC partial drain and after both input pop_front calls, so callers
            // can advance ring CB rd_ptrs (or other per-K-block bookkeeping) only
            // once the consumer has finished reading.
            post_k_block(block, shape.num_k_blocks, last_out);
        }

        // pin_interm_to_captured_base + pack_last_to_interm: the last K-block above advanced
        // wr_ptr by one block (no per-iter reset on the last block), and rd_ptr by one block
        // (from its reload pop). Restore rd_ptr so the downstream consumer (bias-add, untilize)
        // reads from the same L1 cell where the last K-block packed. Mirrors conv2d's
        // original line 535: `matmul_partials_cb == mm_out_cb_id && partials_cb_uses_output`.
        if constexpr (pin_interm_to_captured_base && pack_last_to_interm) {
            UNPACK((get_local_cb_interface(interm_cb_id).fifo_rd_ptr = interm_pin_rd_ptr));
        }
    }
}

}  // namespace compute_kernel_lib
