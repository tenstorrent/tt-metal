// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of bmm_large_block_zm_fused_bias_activation_gathered.cpp, which lives beside it. The
// Metal 2.0 gather_in0 matmul binds this fork; the original serves the legacy MeshWorkload builder and
// the fused reduce-scatter matmul. Until the original is retired, changes to either copy likely belong
// in the other too.
//
// The binding and argument names below are this fork's interface: every factory that later ports
// onto it inherits them and cannot rename them.
//
// This fork carries the path the Metal 2.0 gather_in0 factory builds: one weight, every in0 shard
// holding the same unpadded K / ring_size tiles, and in1 consumed front to back, one K-block per
// ring step, as its reader publishes it. The legacy kernel's batched GlobalCircularBuffer read (in1
// fully resident, addressed by rewriting the read pointer) and its resident L1-sharded in1 are not
// carried.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef SFPU_ACTIVATION
#include "bmm_fused_activation.hpp"
#endif

FORCE_INLINE void reload_from_dfb_to_dst(
    uint32_t in0_dfb_id,
    uint32_t in1_dfb_id,
    uint32_t mm_partials_dfb_id,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);
    // Reconfigure input
    reconfig_data_format_srca(in1_dfb_id, mm_partials_dfb_id);
    copy_init(mm_partials_dfb_id);
    mm_partials_dfb.wait_front(out_subblock_num_tiles);

    const uint32_t start_dst_index = 0;
    const uint32_t start_tile_index = 0;
    copy_block(mm_partials_dfb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);

    mm_partials_dfb.pop_front(out_subblock_num_tiles);
    // Reconfigure srcA back
    reconfig_data_format_srca(mm_partials_dfb_id, in1_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, /*transpose=*/false, out_subblock_w, out_subblock_h, in0_block_w);
}

void kernel_main() {
    constexpr auto in0_block_w = get_arg(args::in0_block_w);              // inner block size in tiles
    constexpr auto in0_num_subblocks = get_arg(args::in0_num_subblocks);  // outer row block size (in inner row blocks)
    constexpr auto in0_block_num_tiles =
        get_arg(args::in0_block_num_tiles);  // out_subblock_h*in0_block_w*in0_num_subblocks
    constexpr auto in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);  // out_subblock_h*in0_block_w
    constexpr auto in1_num_subblocks =
        get_arg(args::in1_num_subblocks);  // outer column block size (in inner column blocks)
    constexpr auto in1_block_num_tiles =
        get_arg(args::in1_block_num_tiles);                         // out_subblock_w*in0_block_w*in1_num_subblocks
    constexpr auto in1_block_w = get_arg(args::in1_block_w);        // out_subblock_w*in1_num_subblocks
    constexpr auto num_blocks = get_arg(args::num_blocks);          // K-blocks, one per ring position
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);  // inner row block size in tiles
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);  // inner column block size in tiles
    constexpr auto out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);  // out_subblock_h * out_subblock_w
    constexpr auto out_block_num_tiles = get_arg(args::out_block_num_tiles);        // number of tiles in out_block
    constexpr bool untilize_out = get_arg(args::untilize_out);

#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type = static_cast<KernelActivation>(get_arg(args::activation_type));
    constexpr auto activation_param0 = get_arg(args::activation_param0);
    constexpr auto activation_param1 = get_arg(args::activation_param1);
    constexpr auto activation_param2 = get_arg(args::activation_param2);
#endif

    // in0 is this core's own shard and feeds the first K-block; in2 holds the shards the ring
    // delivers for the rest, in arrival order.
    constexpr uint32_t in0_dfb_id = dfb::in0;
    constexpr uint32_t in2_dfb_id = dfb::in2;
    constexpr uint32_t in1_dfb_id = dfb::in1;
    constexpr uint32_t mm_out_dfb_id = dfb::out;
    constexpr uint32_t mm_partials_dfb_id = dfb::intermed0;

    // From the binding token, not the bare id: in1 is a relay over a PrefetcherPipe ring, and only
    // the token constructor re-aligns it to the pipe's durable read cursor (firmware resets the
    // buffer's pointers at every launch).
    DataflowBuffer in1_dfb(dfb::in1);
    DataflowBuffer mm_out_dfb(mm_out_dfb_id);
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);

#ifdef SFPU_ACTIVATION
    ActivationInitHelper<activation_type, activation_param0, activation_param1>::init();
#endif

    // With a single output subblock the partial sums stay in DEST across K-blocks; only several
    // subblocks have to spill them to the partials buffer between blocks.
    constexpr bool spill = num_blocks > 1 && (out_block_num_tiles / out_subblock_num_tiles) > 1;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_dfb_id, in1_dfb_id, mm_partials_dfb_id);
    matmul_block_init(in0_dfb_id, in1_dfb_id, /*transpose=*/false, out_subblock_w, out_subblock_h, in0_block_w);

    bool enable_reload = false;
    for (uint32_t block = 0; block < num_blocks; block++) {
        // The in1 reader publishes K-blocks in the ring-rotated order the in0 shards arrive in, so
        // block `block` of in1 pairs with whichever in0 shard this step holds.
        in1_dfb.wait_front(in1_block_num_tiles);

        const uint32_t input0_dfb_id = block == 0 ? in0_dfb_id : in2_dfb_id;
        DataflowBuffer input0_dfb(static_cast<uint16_t>(input0_dfb_id));
        const bool last_out = block == (num_blocks - 1);
// Configure packer once for pack out without Bias
#if defined PACK_RELU
        if (last_out) {
            // if last block we pack the final result with relu enabled
            PACK((llk_pack_relu_config(ReluConfig::zero())));
        }
#endif

        input0_dfb.wait_front(in0_block_num_tiles);

        uint32_t in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            uint32_t in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                tile_regs_acquire();
                if (enable_reload) {
                    reload_from_dfb_to_dst(
                        input0_dfb_id,
                        in1_dfb_id,
                        mm_partials_dfb_id,
                        out_subblock_num_tiles,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                }

                // Compute output sub-block
                const uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
                uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w
                for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                    // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                    // accumulation is done by iterating matmul_block across inner dim
                    // in0_block_w is passed as inner dim (kt) to matmul_block, internally used to stride in0
                    matmul_block(
                        input0_dfb_id,
                        in1_dfb_id,
                        in0_index,
                        in1_index,
                        dst_index,
                        /*transpose=*/false,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                    in0_index++;               // stride right by 1
                    in1_index += in1_block_w;  // to stride down by 1 need to stride by in1_block_w
                }

                if (last_out) {
                    if constexpr (untilize_out) {
                        pack_untilize_dest_init<out_subblock_num_tiles>(mm_out_dfb_id);
                    }
                    tile_regs_commit();
                    // Pack out to output buffer
                    mm_out_dfb.reserve_back(out_subblock_num_tiles);

#if defined SFPU_ACTIVATION
                    apply_activation_from_pack<
                        activation_type,
                        activation_param0,
                        activation_param1,
                        activation_param2>(out_subblock_num_tiles);
#else
                    tile_regs_wait();
#endif

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(mm_out_dfb_id)));
#endif

#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                    const uint32_t start_dst_index = 0;
                    if constexpr (untilize_out) {
                        pack_untilize_dest<out_subblock_num_tiles>(mm_out_dfb_id);
                    } else {
                        pack_block(start_dst_index, mm_out_dfb_id, out_subblock_num_tiles);
                    }

                    tile_regs_release();
                    if constexpr (untilize_out) {
                        pack_untilize_uninit(mm_out_dfb_id);
                    }
                    mm_out_dfb.push_back(out_subblock_num_tiles);

                } else if (spill) {
                    tile_regs_commit();
                    // Move partial result to interm buffer
                    mm_partials_dfb.reserve_back(out_subblock_num_tiles);
                    tile_regs_wait();

#ifdef PACKER_L1_ACC
                    if (block == 0) {  // no accumulation for first iteration
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    } else if (block == 1) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
#endif

                    const uint32_t start_dst_index = 0;
                    pack_block(start_dst_index, mm_partials_dfb_id, out_subblock_num_tiles);

                    tile_regs_release();
                    mm_partials_dfb.push_back(out_subblock_num_tiles);
                }

                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

#ifdef PACKER_L1_ACC
        // Last iteration does spill and reload to output buffer
        if (block < num_blocks - 2 && spill) {
            mm_partials_dfb.wait_front(out_block_num_tiles);
            mm_partials_dfb.pop_front(out_block_num_tiles);
        }
        if (block == num_blocks - 2 && spill) {
            enable_reload = true;
        }  // reload when last iteration
#else
        if constexpr (spill) {
            enable_reload = true;
        }
#endif

        input0_dfb.pop_front(in0_block_num_tiles);
        // Popping in1 is what lets the reader hand this K-block's ring entry back to the sender:
        // the pop publishes the consumer ack only after the unpacker has drained the block.
        in1_dfb.pop_front(in1_block_num_tiles);
    }
}
