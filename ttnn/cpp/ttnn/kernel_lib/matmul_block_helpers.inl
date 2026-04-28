// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/cb_helpers_compute.hpp"

/**
 * @file matmul_block_helpers.inl
 * @brief Implementation of matmul_block helper function.
 *
 * Single pipeline handles both pack strategies:
 *   layout=SubblockMajor → sequential pack_tile_block, per-subblock reserve/push
 *   layout=RowMajor      → absolute-offset pack_tile<true>, per-row-group reserve/push
 *
 * Both modes share K-loop, reload, L1_ACC management, and pre/post callbacks.
 * SKIP_COMPUTE (microbench define) elides the inner matmul LLK call only.
 */

namespace compute_kernel_lib {

template <
    bool transpose,
    bool packer_l1_acc,
    bool pack_last_to_interm,
    bool pack_relu,
    OutputLayout layout,
    matmul_config::InitMode init_mode,
    typename PostComputeFn,
    typename PreKBlockFn,
    typename Buf>
ALWI void matmul_block(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    MatmulBlockShape shape,
    PostComputeFn post_compute,
    PreKBlockFn pre_k_block,
    bool retain_in0,
    uint32_t in1_per_core_w,
    uint32_t out_row_width) {

    // Cache integer IDs for legacy LLK calls. buf_id() resolves to
    // get_cb_id() on CircularBuffer or get_id() on DataflowBuffer.
    const uint32_t in0_cb_id = buf_id(in0_buf);
    const uint32_t in1_cb_id = buf_id(in1_buf);
    const uint32_t out_cb_id = buf_id(out_buf);
    const uint32_t interm_cb_id = buf_id(interm_buf);

    // Hoist shape fields into local names so the existing body reads unchanged.
    const uint32_t in0_num_subblocks = shape.in0_num_subblocks;
    const uint32_t in1_num_subblocks = shape.in1_num_subblocks;
    const uint32_t out_subblock_h = shape.out_subblock_h;
    const uint32_t out_subblock_w = shape.out_subblock_w;
    const uint32_t block_w = shape.in0_block_w;
    const uint32_t num_k_blocks = shape.num_k_blocks;
    const uint32_t batch = shape.batch;

    // Init dispatch: helper owns mm_block_init / mm_block_init_short. Caller does
    // compute_kernel_hw_startup once at boot; everything else is internal.
    if constexpr (init_mode == matmul_config::InitMode::Full) {
        mm_block_init(in0_cb_id, in1_cb_id, interm_cb_id, transpose, out_subblock_w, out_subblock_h, block_w);
    } else if constexpr (init_mode == matmul_config::InitMode::Short) {
        mm_block_init_short(in0_cb_id, in1_cb_id, transpose, out_subblock_w, out_subblock_h, block_w);
    }

    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;
    const uint32_t in0_subblock_num_tiles = out_subblock_h * block_w;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * in0_num_subblocks;
    // in1_per_core_w: actual N-width of the in1 CB per K-block.
    // Derived from subblocks by default; callers with padded per_core_N_compute must
    // pass the real shard width (per_core_N_in1_sender) to avoid CB wait/pop mismatches.
    if (in1_per_core_w == 0) {
        in1_per_core_w = out_subblock_w * in1_num_subblocks;
    }
    // out_row_width: N-tiles per row of the OUTPUT CB layout (row stride for row_major pack).
    // For most factories the in1 CB width and output pack width coincide, so we default to
    // in1_per_core_w. DRAM-sharded passes the larger padded per_core_N_compute here to keep
    // row_group_tiles / row_pos aligned with what the compute actually packs.
    if (out_row_width == 0) {
        out_row_width = in1_per_core_w;
    }
    const uint32_t in1_block_num_tiles = in1_per_core_w * block_w;
    const uint32_t out_block_num_tiles = out_num_tiles * in0_num_subblocks * in1_num_subblocks;
    const uint32_t row_group_tiles = out_subblock_h * out_row_width;

    ASSERT(block_w > 0);
    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(num_k_blocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(batch > 0);
    ASSERT(in0_cb_id != out_cb_id);
    ASSERT(in1_cb_id != out_cb_id);

    ASSERT(out_num_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);

    for (uint32_t b = 0; b < batch; b++) {
        bool enable_reload = false;

        for (uint32_t block = 0; block < num_k_blocks; block++) {
            const bool last_out = block == (num_k_blocks - 1);

            if constexpr (pack_relu && !pack_last_to_interm) {
                if (last_out) {
                    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
                }
            }

            pre_k_block(block, num_k_blocks, last_out);

            // Full-block wait for both modes. Every caller (matmul + SDPA) has the
            // full in0 block resident before invoking the helper, so progressive
            // per-subblock waits are pure polling overhead on TRISCs.
            in0_buf.wait_front(in0_block_num_tiles);
            in1_buf.wait_front(in1_block_num_tiles);

            // Pick the buffer the last K-block packs to. The reference here lets the
            // sync calls below dispatch through the right object regardless of branch.
            Buf& pack_target_buf = pack_last_to_interm ? interm_buf : out_buf;
            const uint32_t pack_target_id = pack_last_to_interm ? interm_cb_id : out_cb_id;

            // Legacy/sequential path: reserve the full out_block on the first
            // non-last K-block so interm spills don't overwrite output data
            // when out_buf and interm_buf share the same L1 region (multicast
            // factory layout). Single reserve here keeps all wait_front /
            // reserve_back increments identical across the K-loop, as the
            // CB-API contract requires.
            if constexpr (layout == OutputLayout::SubblockMajor) {
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
            constexpr bool spill_row_major = (layout == OutputLayout::RowMajor) && packer_l1_acc && pack_last_to_interm;

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                if constexpr (layout == OutputLayout::RowMajor) {
                    // Row-major path reserves per M-row-group (one row of all N-subblocks).
                    // Smaller than full-block reserve, so shared out/interm buffers don't deadlock.
                    if (last_out) {
                        pack_target_buf.reserve_back(row_group_tiles);
                    } else if constexpr (spill_row_major) {
                        interm_buf.reserve_back(row_group_tiles);
                    }
                }

                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(in1_cb_id, interm_cb_id);
                        interm_buf.wait_front(out_num_tiles);
                        copy_block_matmul_partials(interm_cb_id, 0, 0, out_num_tiles);
                        interm_buf.pop_front(out_num_tiles);
                        mm_block_init_short_with_dt(
                            in0_cb_id, in1_cb_id, interm_cb_id, transpose, out_subblock_w, out_subblock_h, block_w);
                    }

                    // Compute output sub-block via hardware block matmul.
                    // SKIP_COMPUTE (microbench) keeps the surrounding pipeline intact but
                    // omits the actual matmul LLK call.
                    uint32_t dst_index = 0;
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < block_w; inner_dim++) {
#ifndef SKIP_COMPUTE
                        // ckernel:: disambiguates the LLK matmul_block from this helper.
                        ckernel::matmul_block(
                            in0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            transpose,
                            out_subblock_w,
                            out_subblock_h,
                            block_w);
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

                        tile_regs_commit();
                        if constexpr (layout == OutputLayout::SubblockMajor) {
                            pack_target_buf.reserve_back(out_num_tiles);
                        }
                        tile_regs_wait();

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

                        if constexpr (layout == OutputLayout::RowMajor) {
                            // Single-row subblock: DST tiles are already contiguous in
                            // row-major order and consecutive in1_subblock iterations land
                            // at adjacent CB positions, so pack_tile_block produces the
                            // correct layout with fewer per-tile LLK calls than the
                            // absolute-offset path below. Multi-row subblocks need the
                            // absolute-offset pack because DST tiles are packed column-first
                            // within a subblock while row-major output wants row-first.
                            if (out_subblock_h == 1) {
                                pack_tile_block(0, pack_target_id, out_subblock_w);
                            } else {
                                uint32_t dst_idx = 0;
                                uint32_t col_base = in1_subblock * out_subblock_w;
                                for (uint32_t r = 0; r < out_subblock_h; r++) {
                                    // Row stride uses out_row_width (padded output-pack width),
                                    // not in1_per_core_w — those differ for DRAM-sharded.
                                    uint32_t row_pos = r * out_row_width;
                                    for (uint32_t c = 0; c < out_subblock_w; c++) {
                                        pack_tile<true>(dst_idx, pack_target_id, row_pos + col_base + c);
                                        dst_idx++;
                                    }
                                }
                            }
                        } else {
                            pack_tile_block(0, pack_target_id, out_num_tiles);
                        }

                        tile_regs_release();
                        if constexpr (layout == OutputLayout::SubblockMajor) {
                            pack_target_buf.push_back(out_num_tiles);
                        }

                    } else {
                        // Non-last K-block: spill partial to interm_buf. spill_row_major (defined
                        // at the top of the K-block loop body) decides whether to match the
                        // last-block row-major layout (needed when pack_last_to_interm + L1_ACC
                        // accumulate into the same interm_buf buffer) or keep legacy subblock-
                        // major (compatible with software reload's per-subblock read).
                        tile_regs_commit();
                        if constexpr (!spill_row_major) {
                            interm_buf.reserve_back(out_num_tiles);
                        }
                        tile_regs_wait();

                        if constexpr (packer_l1_acc) {
                            PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
                        }

                        if constexpr (spill_row_major) {
                            if (out_subblock_h == 1) {
                                pack_tile_block(0, interm_cb_id, out_subblock_w);
                            } else {
                                uint32_t dst_idx = 0;
                                uint32_t col_base = in1_subblock * out_subblock_w;
                                for (uint32_t r = 0; r < out_subblock_h; r++) {
                                    uint32_t row_pos = r * out_row_width;
                                    for (uint32_t c = 0; c < out_subblock_w; c++) {
                                        pack_tile<true>(dst_idx, interm_cb_id, row_pos + col_base + c);
                                        dst_idx++;
                                    }
                                }
                            }
                        } else {
                            pack_tile_block(0, interm_cb_id, out_num_tiles);
                        }
                        tile_regs_release();
                        if constexpr (!spill_row_major) {
                            interm_buf.push_back(out_num_tiles);
                        }
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }

                if constexpr (layout == OutputLayout::RowMajor) {
                    if (last_out) {
                        pack_target_buf.push_back(row_group_tiles);
                    } else if constexpr (spill_row_major) {
                        interm_buf.push_back(row_group_tiles);
                    }
                }

                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if constexpr (packer_l1_acc) {
                // Wait/pop the L1_ACC partials in increments that match the producer's push
                // granularity: row_group_tiles when spill_row_major (FUSE_BIAS + L1_ACC path
                // pushes per M-row-group), otherwise subblock-sized. The CB API requires
                // identical increments across all waits.
                const uint32_t drain_step = spill_row_major ? row_group_tiles : out_num_tiles;
                if constexpr (pack_last_to_interm) {
                    if (block < num_k_blocks - 1) {
                        for (uint32_t s = 0; s < out_block_num_tiles; s += drain_step) {
                            interm_buf.wait_front(drain_step);
                            interm_buf.pop_front(drain_step);
                        }
                    }
                    enable_reload = false;
                } else {
                    if (num_k_blocks >= 2 && block < num_k_blocks - 2) {
                        for (uint32_t s = 0; s < out_block_num_tiles; s += drain_step) {
                            interm_buf.wait_front(drain_step);
                            interm_buf.pop_front(drain_step);
                        }
                    }
                    if (block == num_k_blocks - 2) {
                        enable_reload = true;
                    }
                }
            } else {
                if (num_k_blocks > 1) {
                    enable_reload = true;
                }
            }

            // retain_in0: SDPA reuses Q across K chunks, so caller keeps in0 front
            // on the last iteration. Intermediate blocks always pop.
            if (!retain_in0 || !last_out) {
                in0_buf.pop_front(in0_block_num_tiles);
            }
            in1_buf.pop_front(in1_block_num_tiles);
        }
    }
}

template <typename Buf>
ALWI void matmul_reduce_inplace(
    Buf& in_out_buf,
    Buf& in1_buf,
    uint32_t num_subblocks,
    uint32_t subblock_h,
    uint32_t subblock_w,
    uint32_t block_kt) {

    const uint32_t in_out_cb_id = buf_id(in_out_buf);
    const uint32_t in1_cb_id = buf_id(in1_buf);

    const uint32_t subblock_tiles = subblock_h * subblock_w;
    const uint32_t total_in_tiles = num_subblocks * subblock_tiles;

    // Init + reconfig + input waits. in1_buf holds a single column-identity tile
    // (fronted for the life of the helper); in_out_buf must have the full input
    // population fronted before the reduce begins.
    mm_block_init_short(in_out_cb_id, in1_cb_id, /*transpose=*/false, subblock_w, subblock_h, block_kt);
    reconfig_data_format(in1_cb_id, in_out_cb_id);
    in1_buf.wait_front(1);
    in_out_buf.wait_front(total_in_tiles);

    for (uint32_t sub = 0; sub < num_subblocks; ++sub) {
        tile_regs_acquire();
        ckernel::matmul_block(
            in_out_cb_id, in1_cb_id, 0, 0, 0,
            /*transpose=*/false, subblock_w, subblock_h, block_kt);
        tile_regs_commit();
        // Pop must happen after commit and before the back-pack so the read pointer
        // advances past the tiles we just consumed, making room for the write.
        in_out_buf.pop_front(subblock_tiles);
        tile_regs_wait();
        in_out_buf.reserve_back(subblock_tiles);
        for (uint32_t i = 0; i < subblock_tiles; i++) {
            pack_tile(i, in_out_cb_id);
        }
        tile_regs_release();
        in_out_buf.push_back(subblock_tiles);
    }
}

}  // namespace compute_kernel_lib
