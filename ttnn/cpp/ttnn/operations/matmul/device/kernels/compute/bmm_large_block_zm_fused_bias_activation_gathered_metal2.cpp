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
// This fork carries the path the Metal 2.0 gather_in0 factory builds: one weight, consumed one K-block
// per ring step, in the ring order the in0 shards arrive in, this core's own first. in1 arrives either
// in that order, a K-block at a time, or in K order, the whole layer staying in in1 until this kernel
// is done with it; then this kernel walks the layer in ring order by stepping over the K-blocks it does
// not need yet (see below), where the legacy kernel rewrites the read pointer. in0 shards may be padded:
// shard p holds the K tiles from p * in0_block_w up to K, and compute accumulates only those. The legacy
// kernel's multi-weight batch and its resident L1-sharded in1 are not carried.

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

// Moves in1's front past num_tiles tiles without reading them, once they are published.
FORCE_INLINE void step_over_in1(DataflowBuffer& in1_dfb, uint32_t num_tiles) {
    if (num_tiles > 0) {
        in1_dfb.wait_front(num_tiles);
        in1_dfb.pop_front(num_tiles);
    }
}

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
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::intermed0);

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
    constexpr uint32_t ring_size = num_blocks;
    // The activation's K, without the padding of its last shards when K does not fill the ring.
    constexpr auto k_tiles = get_arg(args::k_tiles);
    // Whether in1's K-blocks arrive in ring order, this core's own first, or in K order, a whole layer.
    constexpr bool in1_in_ring_order = get_arg(args::in1_in_ring_order);

    // This core's position in the ring: it holds activation shard ring_idx.
    const uint32_t ring_idx = get_arg(args::ring_idx);

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

    matmul_block_init(in0_dfb_id, in1_dfb_id, /*transpose=*/false, out_subblock_w, out_subblock_h, in0_block_w);

    // In K order the layer starts at in1's front. This kernel steps over its first ring_idx K-blocks to
    // reach its own, and after the layer's last K-block over the rest of the ring, which holds none of
    // this layer, to come back round to the layer's first. in1's ring is the PrefetcherPipe's, so its
    // size is read off in1 (as 0 on the math thread, which leaves in1's pointers to the unpacker). The
    // in1 reader publishes both spans for this kernel to step over.
    constexpr uint32_t layer_tiles = num_blocks * in1_block_num_tiles;
    if constexpr (!in1_in_ring_order) {
        step_over_in1(in1_dfb, ring_idx * in1_block_num_tiles);
    }

    bool enable_reload = false;
    for (uint32_t block = 0; block < num_blocks; block++) {
        // The in1 K-block at the front pairs with the in0 shard this step holds, that of ring position
        // ring_pos. Shard ring_pos holds the K tiles from shard_k_start up to K: a padded shard fewer,
        // or none.
        const uint32_t ring_pos = (ring_idx + block) % ring_size;
        const uint32_t shard_k_start = ring_pos * in0_block_w;
        const uint32_t k_tiles_left = shard_k_start < k_tiles ? k_tiles - shard_k_start : 0;
        const uint32_t unpadded_in0_block_w = k_tiles_left < in0_block_w ? k_tiles_left : in0_block_w;

        in1_dfb.wait_front(in1_block_num_tiles);

        const uint32_t input0_dfb_id = block == 0 ? in0_dfb_id : in2_dfb_id;
        DataflowBuffer input0_dfb(static_cast<uint16_t>(input0_dfb_id));
        const bool last_out = block == (num_blocks - 1);
// Configure packer once for pack out without Bias
#if defined PACK_RELU
        if (last_out) {
            // if last block we pack the final result with relu enabled
            pack_relu_config(ReluConfig::zero());
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
                // inner dim that we accumulate is the inner dim of in0/in1: the shard's unpadded width
                for (uint32_t inner_dim_idx = 0; inner_dim_idx < unpadded_in0_block_w; ++inner_dim_idx) {
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
                    pack_reconfig_data_format(mm_out_dfb_id);
#endif

#ifdef PACKER_L1_ACC
                    pack_reconfig_l1_acc(0);
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
                        pack_reconfig_l1_acc(0);
                    } else if (block == 1) {
                        pack_reconfig_l1_acc(1);
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
        if constexpr (!in1_in_ring_order) {
            if (ring_pos == ring_size - 1 && block + 1 < num_blocks) {
                const uint32_t ring_tiles = in1_dfb.get_total_num_entries();
                step_over_in1(in1_dfb, ring_tiles > layer_tiles ? ring_tiles - layer_tiles : 0);
            }
        }
    }
}
