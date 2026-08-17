// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "buffer_compat.hpp"
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
 * Pack a (h × w) DST sub-block to absolute row-major positions in the output CB —
 * one pack_tile<true>(dst_idx, cb, abs_tile_idx) per tile, so the helper needs no
 * access to CB-interface internals.
 *   dst_start_idx   start in DST; DST[dst_start_idx .. +h*w-1] packed row-first.
 *   pack_target_id  output CB id.
 *   col_base        column offset within the row group (tiles).
 *   row_stride      row stride in tiles (= out_row_width).
 * PRECONDITION: caller reserved >= h * row_stride tiles in pack_target_id (one
 * M-row-group) and is responsible for the matching cb_push_back.
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

/**
 * Read mirror of pack_subblock_row_strided: reload a row-strided-spilled (h × w) sub-block
 * into CONTIGUOUS DST. Each row's w tiles sit at source offset (r * row_stride + col_base)
 * from fifo_rd_ptr and land at DST[r * w], so the matmul/pack sees the same row-major
 * layout the contiguous (SubblockMajor) reload produces.
 *
 * When the sub-block spans the complete row width, all h*w source tiles are contiguous and one
 * block copy reloads them. Otherwise one copy_block_matmul_partials per row gathers across the
 * stride. Neither form advances fifo_rd_ptr. Caller waits the whole fronted row group
 * (col_base + (h-1)*row_stride + w tiles) and pops it when done. col_base / row_stride match the
 * spill.
 */
ALWI void copy_subblock_row_strided(
    uint32_t src_cb_id,
    uint32_t col_base,
    uint32_t row_stride,
    uint32_t h,
    uint32_t w) {
    if (w == row_stride) {
        copy_block_matmul_partials(src_cb_id, col_base, 0, h * w);
        return;
    }
    for (uint32_t r = 0; r < h; r++) {
        copy_block_matmul_partials(src_cb_id, r * row_stride + col_base, r * w, w);
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
    uint32_t untilize_block_ct_dim,
    typename KBlockInnerDimFn,
    typename In0SourceFn,
    typename In1BaseOffsetFn,
    bool caller_owns_pack_target,
    typename Activation,
    matmul_config::DataFormatReconfig reconfig,
    typename Buf,
    typename In0BaseOffsetFn>
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
    In1BaseOffsetFn in1_base_offset_fn,
    uint32_t out_col_offset,
    In0BaseOffsetFn in0_base_offset_fn) {

    // OutWithUntilize needs SubblockMajor: pack_untilize_dest packs from DST offset 0 for a
    // fixed block_ct_dim and can't compose with the per-tile absolute-offset row-major pack.
    // Row-major untilize goes through Interm + reblock_and_untilize instead.
    static_assert(
        last_block_target != LastBlockTarget::OutWithUntilize || tile_order == OutputCBLayout::SubblockMajor,
        "OutWithUntilize requires tile_order == SubblockMajor; route row-major untilize via Interm + reblock_and_untilize");
    // block_ct_dim is a compile-time arg, so the caller must supply it for OutWithUntilize.
    static_assert(
        last_block_target != LastBlockTarget::OutWithUntilize || untilize_block_ct_dim > 0,
        "OutWithUntilize requires untilize_block_ct_dim > 0 (= shape.out_subblock_h * shape.out_subblock_w)");
    // NoWaitNoPop on in0 hands the caller the WHOLE in0 lifecycle: one wait_front before the
    // call and one pop_front after, for as many matmul_block calls as read that block. It is the
    // only policy that fits an in0 which is (a) read by SEVERAL matmuls, so no call may pop, and
    // (b) M-major over the full K extent, so successive K-blocks are strided column windows the
    // pop lifecycle cannot reach — pair it with In0BaseOffsetFn, which shifts the K origin
    // instead. Without a base-offset functor at num_k_blocks > 1 it degenerates to "every K-block
    // reads block 0" — a SILENT WRONG ANSWER, not a hang. That pairing cannot be asserted here
    // (the default functor is the right choice at one K-block), so it is a review item.
    // last_block_target == Interm feeds a downstream phase; whether activation belongs HERE
    // (matmul pack) or in that phase is the caller's choice via the Activation parameter (the
    // helper can't infer it). Set Activation here only when the downstream phase reads the
    // partials unchanged; otherwise leave it NoneActivation and activate downstream.

    // Decode LastBlockTarget into the bool pair the body branches on. (Interm + Relu) is
    // unrepresentable by construction.
    constexpr bool pack_last_to_interm = (last_block_target == LastBlockTarget::Interm);
    constexpr bool pack_relu = (last_block_target == LastBlockTarget::OutWithRelu);
    constexpr bool caller_owned_plain_out = caller_owns_pack_target && (last_block_target == LastBlockTarget::Out);

    // caller_owns_pack_target is correct for two fixed-address L1-accumulation lifecycles:
    // Interm accumulates every K-block in place; plain Out accumulates non-last K-blocks in the
    // unpushed interm scratch, reloads it without FIFO bookkeeping on the last K-block, and packs
    // directly to caller-reserved output. See caller_owns_pack_target_supported for the contract.
    static_assert(
        caller_owns_pack_target_supported(
            caller_owns_pack_target,
            tile_order == OutputCBLayout::TileRowMajor,
            packer_l1_acc,
            pack_last_to_interm,
            last_block_target == LastBlockTarget::Out),
        "caller_owns_pack_target requires TileRowMajor + packer_l1_acc + Interm or plain Out");
    static_assert(
        in1_policy != InputPolicy::WaitAndRetainPerMSubblock,
        "WaitAndRetainPerMSubblock is an in0-only policy");

    // Cache integer IDs for legacy LLK calls. buf_id() resolves to
    // get_cb_id() on CircularBuffer or get_id() on DataflowBuffer.
    const uint32_t in0_cb_id = buf_id(in0_buf);
    const uint32_t in1_cb_id = buf_id(in1_buf);
    const uint32_t out_cb_id = buf_id(out_buf);
    const uint32_t interm_cb_id = buf_id(interm_buf);

    // Fail fast on shape / CB invariants before doing any init or pipeline work.
    ASSERT(shape.in0_block_k > 0);
    ASSERT(shape.in0_num_subblocks > 0);
    ASSERT(shape.in1_num_subblocks > 0);
    ASSERT(shape.num_k_blocks > 0);
    ASSERT(shape.out_subblock_h > 0);
    ASSERT(shape.out_subblock_w > 0);
    ASSERT(shape.batch > 0);
    if constexpr (in0_policy == InputPolicy::WaitAndRetainPerMSubblock) {
        ASSERT(shape.num_k_blocks == 1);
    }
    ASSERT(in0_cb_id != out_cb_id);
    ASSERT(in1_cb_id != out_cb_id);
    ASSERT(shape.out_subblock_h * shape.out_subblock_w <= compute_kernel_lib::DEST_AUTO_LIMIT);

    // Reconfig and init are independent compile-time gates (see the InitMode /
    // DataFormatReconfig enums). The pack reconfig targets interm_cb_id (where non-last
    // K-blocks spill); the last-block in-loop reconfig swaps to out_cb_id. The init is
    // always short — the hw_configure-bearing boot init is the caller's.
    // ShortAfterPreKBlock relocates this whole block into the K-loop (after pre_k_block),
    // so it is skipped here for that mode.
    if constexpr (init_mode != matmul_config::InitMode::ShortAfterPreKBlock) {
        if constexpr (
            reconfig == matmul_config::DataFormatReconfig::INPUT ||
            reconfig == matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT) {
            // Matmul convention: srca takes in1, srcb takes in0.
            reconfig_data_format(in1_cb_id, in0_cb_id);
        }
        // NOTE: this pack reconfig targets interm_cb_id even when num_k_blocks == 1 (no
        // spill). The last-block swap-to-out reconfig is gated on l1_acc / fp32 DEST and may
        // not fire, so a placeholder interm whose format differs from out_buf can leave the
        // packer mis-configured (silent corruption) — the .hpp interm_buf doc requires a
        // same-format placeholder; pass out_buf.
        if constexpr (
            reconfig == matmul_config::DataFormatReconfig::OUTPUT ||
            reconfig == matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT) {
            PACK((pack_reconfig_data_format(interm_cb_id)));
        }
        if constexpr (init_mode == matmul_config::InitMode::Short) {
            mm_block_init_short(
                in0_cb_id, in1_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
        }
    }

    const uint32_t out_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * shape.in0_block_k;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks;
    // in1_per_core_w: actual N-width the producer pushes per K-block. Derived from subblocks
    // by default; callers that pad the in1 width must pass the real value to avoid wait/pop
    // mismatches.
    if (in1_per_core_w == 0) {
        in1_per_core_w = shape.out_subblock_w * shape.in1_num_subblocks;
    }
    // out_row_width: N-tiles per row of the OUTPUT CB (TileRowMajor row stride). Defaults to
    // in1_per_core_w (read and pack widths usually coincide); callers that pad the output
    // width above the in1 read width pass the larger value here.
    if (out_row_width == 0) {
        out_row_width = in1_per_core_w;
    }
    const uint32_t in1_block_num_tiles = in1_per_core_w * shape.in0_block_k;
    const uint32_t out_block_num_tiles = out_num_tiles * shape.in0_num_subblocks * shape.in1_num_subblocks;
    const uint32_t row_group_tiles = shape.out_subblock_h * out_row_width;

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
                    reconfig == matmul_config::DataFormatReconfig::INPUT ||
                    reconfig == matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT) {
                    reconfig_data_format(in1_cb_id, active_in0_cb_id);
                }
                if constexpr (
                    reconfig == matmul_config::DataFormatReconfig::OUTPUT ||
                    reconfig == matmul_config::DataFormatReconfig::INPUT_AND_OUTPUT) {
                    PACK((pack_reconfig_data_format(interm_cb_id)));
                }
                mm_block_init_short(
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

            // Full-block wait for the ordinary modes. WaitAndRetainPerMSubblock instead
            // fronts each cumulative M prefix immediately before it is consumed below,
            // allowing a producer to publish a resident in0 one row-group at a time.
            // in0_policy=NoWaitNoPop: the caller fronted the whole in0 region once (it is read by
            // several matmuls and/or the K-blocks are strided windows of it), so a per-K-block
            // wait here would over-wait the region's own size and deadlock.
            if constexpr (in0_policy != InputPolicy::NoWaitNoPop) {
                if constexpr (in0_policy == InputPolicy::WaitAndRetainPerMSubblock) {
                    if (!shape.wait_in0_per_m_subblock) {
                        active_in0_buf.wait_front(in0_block_num_tiles);
                    }
                } else {
                    active_in0_buf.wait_front(in0_block_num_tiles);
                }
            }
            // in1_policy=NoWaitNoPop: caller manages in1 lifecycle externally
            // (cross-chip global-CB receivers; pre-populated L1-sharded in1).
            if constexpr (in1_policy != InputPolicy::NoWaitNoPop) {
                in1_buf.wait_front(in1_block_num_tiles);
            }

            // A caller-owned plain-Out matmul deliberately leaves the fixed intermediate scratch
            // unpushed so its address cannot walk when runtime M is smaller than the physical CB.
            // That also removes the ordinary CB credit which orders PACK's final accumulated spill
            // before UNPACK reloads it. Synchronize the two Tensix threads once, immediately before
            // the only software reload (the last K-block); tile_regs_release alone protects DEST
            // reuse, not an L1 PACK-write -> UNPACK-read dependency.
            if constexpr (caller_owned_plain_out) {
                if (enable_reload) {
                    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
                    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
                }
            }

            // Pick the buffer the last K-block packs to. The reference here lets the
            // sync calls below dispatch through the right object regardless of branch.
            Buf& pack_target_buf = pack_last_to_interm ? interm_buf : out_buf;
            const uint32_t pack_target_id = pack_last_to_interm ? interm_cb_id : out_cb_id;

            // SubblockMajor: reserve the full out_block on the first non-last K-block so
            // interm spills don't clobber output when interm shares out's L1 region (the
            // factory's share-buffer layout), and so reserve/wait increments stay uniform
            // across the K-loop. Skipped when the caller owns the pack lifecycle.
            if constexpr (tile_order == OutputCBLayout::SubblockMajor && !caller_owns_pack_target) {
                if (block == 0 && !last_out) {
                    out_buf.reserve_back(out_block_num_tiles);
                }
            }

            // Non-last K-blocks spill into interm_buf. With TileRowMajor + L1_ACC the spill must
            // match the last block's row-strided layout (Interm accumulates in the same region;
            // Out reloads from it row-strided), so per-address accumulation is correct. Otherwise
            // keep subblock-major so the last-block per-subblock reload reads partials contiguously.
            constexpr bool spill_row_grouped = (tile_order == OutputCBLayout::TileRowMajor) && packer_l1_acc;

            // Per-K-block in0 base tile offset (default 0). Nonzero reads a different K window of
            // the same fronted in0 region — the in0 mirror of in1_base_offset (see NoIn0BaseOffset).
            int in0_index_subblock_offset = static_cast<int>(in0_base_offset_fn(block));
            for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
                if constexpr (in0_policy == InputPolicy::WaitAndRetainPerMSubblock) {
                    if (shape.wait_in0_per_m_subblock) {
                        // No pop occurs between waits, so the CB contract requires cumulative counts.
                        // One increment is exactly the M-row-group this subblock will read.
                        active_in0_buf.wait_front((in0_subblock + 1) * in0_subblock_num_tiles);
                    }
                }
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
                            // Under caller_owns, the fixed intermediate is private scratch: it was
                            // never pushed, so reload it directly without touching FIFO state.
                            if constexpr (!caller_owns_pack_target) {
                                if (in1_subblock == 0) {
                                    interm_buf.wait_front(row_group_tiles);
                                }
                            }
                            // Ordinary FIFO reload advances the read pointer after each M row
                            // group. Caller-owned scratch never pops, so fold that row-group base
                            // into the absolute source offset just as the pack path does.
                            const uint32_t reload_row_base =
                                caller_owns_pack_target ? in0_subblock * row_group_tiles : 0;
                            copy_subblock_row_strided(
                                interm_cb_id,
                                reload_row_base + in1_subblock * shape.out_subblock_w,
                                out_row_width,
                                shape.out_subblock_h,
                                shape.out_subblock_w);
                            if constexpr (!caller_owns_pack_target) {
                                if (in1_subblock == shape.in1_num_subblocks - 1) {
                                    interm_buf.pop_front(row_group_tiles);
                                }
                            }
                        } else {
                            if constexpr (!caller_owns_pack_target) {
                                interm_buf.wait_front(out_num_tiles);
                            }
                            copy_block_matmul_partials(interm_cb_id, 0, 0, out_num_tiles);
                            if constexpr (!caller_owns_pack_target) {
                                interm_buf.pop_front(out_num_tiles);
                            }
                        }
                        mm_block_init_short_with_dt(
                            in0_cb_id, in1_cb_id, interm_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
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
                        post_compute(out_num_tiles);

                        // OutWithUntilize: bracket the per-subblock pack with pack_untilize
                        // init/uninit so later ops can resume their own packer config (init
                        // before commit, uninit after release).
                        if constexpr (last_block_target == LastBlockTarget::OutWithUntilize) {
                            pack_untilize_dest_init<untilize_block_ct_dim>(pack_target_id);
                        }

                        tile_regs_commit();
                        if constexpr (tile_order == OutputCBLayout::SubblockMajor && !caller_owns_pack_target) {
                            pack_target_buf.reserve_back(out_num_tiles);
                        }
                        // Pack-side sync: apply_activation_from_pack does SFPU on the packer
                        // thread with its own math/pack wait + dest-offset flip + STALLWAIT,
                        // REPLACING tile_regs_wait. With NONE, use the standard 4-phase wait.
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
                                // Interm target: L1 accumulates across all blocks in the same region.
                                PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
                            } else {
                                // Out target: the last block's partial was reloaded into DST, so the
                                // pack must NOT re-accumulate.
                                PACK((llk_pack_reconfig_l1_acc(0)));
                            }
                        }

                        if constexpr (tile_order == OutputCBLayout::TileRowMajor) {
                            // Absolute-offset per-tile pack into the row-group reserve; row stride
                            // = out_row_width. The per-row-group reserve supplies the M-row-group
                            // base, leaving only the in1 col offset.
                            //
                            // caller_owns_pack_target: there is no per-row-group reserve (the caller
                            // did ONE reserve over the whole block, FIFO wr_ptr fixed at the base),
                            // so the M-row-group base must be folded into the absolute offset here
                            // (in0_subblock * row_group_tiles); otherwise every row group packs onto
                            // row 0 (latent when in0_num_subblocks == 1, garbles when > 1).
                            const uint32_t row_base =
                                caller_owns_pack_target ? in0_subblock * row_group_tiles : 0;
                            // out_col_offset shifts this call's columns inside the row-major block,
                            // so a caller streaming the in1 width in chunks packs chunk c at its own
                            // columns rather than producing a chunk-major block. Applied to the
                            // spill and the last-block pack alike: a K-block partial must land at
                            // the same absolute address the reload will read it from. 0 by default.
                            const uint32_t col_base =
                                row_base + in1_subblock * shape.out_subblock_w + out_col_offset;
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
                        // Non-last K-block: spill partial to interm_buf. spill_row_grouped picks
                        // row-major (to match the last block when accumulating in the same interm
                        // region) or subblock-major (compatible with the software per-subblock reload).
                        tile_regs_commit();
                        if constexpr (!spill_row_grouped && !caller_owns_pack_target) {
                            interm_buf.reserve_back(out_num_tiles);
                        }
                        tile_regs_wait();

                        if constexpr (packer_l1_acc || get_fp32_dest_acc_enabled()) {
                            // Pack-DF must match interm's format for non-last spills, else spills
                            // land in whatever format the previous op left (typically out's).
                            PACK((pack_reconfig_data_format(interm_cb_id)));
                        }
                        if constexpr (packer_l1_acc) {
                            PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
                        }

                        if constexpr (spill_row_grouped) {
                            // caller_owns_pack_target: fold the M-row-group base into the offset,
                            // same as the last-block pack above.
                            const uint32_t row_base =
                                caller_owns_pack_target ? in0_subblock * row_group_tiles : 0;
                            // out_col_offset shifts this call's columns inside the row-major block,
                            // so a caller streaming the in1 width in chunks packs chunk c at its own
                            // columns rather than producing a chunk-major block. Applied to the
                            // spill and the last-block pack alike: a K-block partial must land at
                            // the same absolute address the reload will read it from. 0 by default.
                            const uint32_t col_base =
                                row_base + in1_subblock * shape.out_subblock_w + out_col_offset;
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

            if constexpr (caller_owned_plain_out) {
                if (block == shape.num_k_blocks - 2) {
                    // Publish only after every sub-block's tile_regs_release, so one semaphore
                    // handshake covers the complete fixed-scratch region.
                    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
                }
            }

            if constexpr (packer_l1_acc) {
                // Drain the L1_ACC partials in increments matching the producer's push
                // granularity (row_group_tiles when spill_row_grouped, else subblock-sized);
                // the CB API requires uniform increments. Skipped under caller-owns (the helper
                // pushes nothing, so there is nothing to drain).
                const uint32_t drain_step = spill_row_grouped ? row_group_tiles : out_num_tiles;
                if constexpr (pack_last_to_interm) {
                    // No software reload: Interm accumulates in place (and SBM-contiguous reload
                    // offsets wouldn't match the row-strided spill anyway).
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

            // WaitAndRetainOnLastBlock: skip the pop on the last K-block (caller reuses in0);
            // intermediate blocks always pop. Pops target active_in0_buf (the In0SourceFn CB).
            if constexpr (in0_policy == InputPolicy::WaitAndPopPerKBlock) {
                active_in0_buf.pop_front(in0_block_num_tiles);
            } else if constexpr (in0_policy == InputPolicy::WaitAndRetainOnLastBlock) {
                if (!last_out) {
                    active_in0_buf.pop_front(in0_block_num_tiles);
                }
            }
            // WaitAndRetainPerMSubblock and NoWaitNoPop: no pop at all — the caller owns in0's rd_ptr.
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
    }
}

}  // namespace compute_kernel_lib
