// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp
// (block-sharded conv2d weights+bias mcast receiver; also does the split-reader second-half
// activation reads).  Algorithm body identical to the legacy kernel; only the host-binding surface
// is migrated:
//   - CB-index CTAs -> dfb:: tokens (weights / bias / act_second_reader / act_sharded / reader_indices)
//   - weights mcast semaphore-id RTAs -> sem::weights_mcast_sender / sem::weights_mcast_receiver
//   - remaining positional CTAs -> get_arg(args::name); remaining RTAs -> get_arg(args::name)
//   - DataflowBuffer -> DataflowBuffer; get_tile_size(cb) -> cb.get_entry_size()
//
// SCOPE: split_reader_cb_shared (the side-channel-semaphore second-writer overlap) is DROPPED.
// The Metal 2.0 sharded factory TT_FATAL-rejects split_reader_cb_shared on the host, so this fork
// only implements the NON-OVERLAP topology where ACT is single-producer (the reserve_done/write_done
// semaphores and the raw co-fill into the shared ACT CB are gone).  Deferred per CB_TAXONOMY_ANALYSIS
// resolution #3.
//
// Despite the "tiled_out" filename this kernel never writes OUT; the factory binds OUT as a
// degenerate consumer (resolution #1).

#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"
#include "api/debug/ring_buffer.h"
// DEBUG: weights-mcast-receiver deadlock localization. Marker 0xBP_00CC, P=phase, CC=load counter.

void kernel_main() {
    uint32_t rb_wcnt = 0;  // DEBUG: weight-load counter for ring buffer markers
    constexpr uint32_t num_blocks_weight_h = get_arg(args::num_blocks_weight_h);
    constexpr uint32_t weight_block_num_tiles = get_arg(args::weight_block_num_tiles);
    constexpr uint32_t weight_block_height_num_outer = get_arg(args::weight_block_height_num_outer);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_arg(args::bias_ntiles);
    constexpr uint32_t out_num_blocks_h = get_arg(args::out_num_blocks_h);
    constexpr uint32_t out_num_blocks_w = get_arg(args::out_num_blocks_w);

    constexpr bool fuse_bias = get_arg(args::fuse_bias);

    constexpr bool split_reader_enabled = get_arg(args::split_reader_enabled);

    // Split reader args
    constexpr uint32_t window_outer = get_arg(args::window_outer);  // num_blocks_act_w
    constexpr bool sliced_inner_dim = window_outer > 1;             // Derived like block sharded reader
    constexpr uint32_t act_block_num_tiles_split_last = get_arg(args::act_block_num_tiles_split_last);
    constexpr uint32_t conv_act_c_read_bytes = get_arg(args::conv_act_c_read_bytes);
    constexpr uint32_t weight_size_w = get_arg(args::weight_size_w);
    constexpr uint32_t padded_conv_act_size_w = get_arg(args::padded_conv_act_size_w);
    constexpr uint32_t act_block_w_extra_align_bytes = get_arg(args::act_block_w_extra_align_bytes);
    constexpr bool needs_act_block_zero_out = get_arg(args::needs_act_block_zero_out) == 1;
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t weight_size_h = get_arg(args::weight_size_h);

    // mcast args
    const uint32_t weights_mcast_sender_noc_x = get_arg(args::weights_mcast_sender_noc_x);
    const uint32_t weights_mcast_sender_noc_y = get_arg(args::weights_mcast_sender_noc_y);

    // Experimental API objects
    Noc noc;
    Semaphore weights_mcast_sender_sem(sem::weights_mcast_sender);
    Semaphore weights_mcast_receiver_sem(sem::weights_mcast_receiver);
    DataflowBuffer cb_weight_obj(dfb::weights);
#ifdef FUSE_BIAS
    DataflowBuffer cb_bias_obj(dfb::bias);
#endif
#ifdef SPLIT_READER
    DataflowBuffer cb_act_second_obj(dfb::act_second_reader);
    DataflowBuffer cb_reader_indices_obj(dfb::reader_indices);
    DataflowBuffer cb_sharded_act_obj(dfb::act_sharded);
#endif

    const bool is_sender_core = get_arg(args::is_sender_core) > 0;

    // Split reader configuration
    if constexpr (split_reader_enabled) {
#ifdef SPLIT_READER
#ifdef CONFIG_TENSOR_IN_DRAM
        cb_reader_indices_obj.wait_front(1);
#endif
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<dfb::act_second_reader>(noc, cb_act_second_obj);
        }
#endif
    }

#ifdef SPLIT_READER
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        (split_reader_enabled && is_sender_core)
            ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_indices_obj.get_write_ptr())
            : nullptr;

    // Initial setup for second reader (starting from second reader's data)
    // Only read reader indices on cores that have sharded input (is_sender_core).
    uint32_t start_reader_idx =
        (split_reader_enabled && is_sender_core) ? (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1 : 0;
    uint32_t reader_idx = start_reader_idx;

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    constexpr uint32_t window_outer_offset = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;

    const uint32_t act_l1_read_addr = split_reader_enabled ? cb_sharded_act_obj.get_read_ptr() : 0;
#endif

    // read in bias if enabled (done only once for all batches)
    bool load_bias = true;

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
#ifdef SPLIT_READER
    uint32_t l1_write_addr_act = 0;
    uint32_t reader_offset = 0;
#endif
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
#ifdef SPLIT_READER
            if constexpr (split_reader_enabled) {
                // Read activation data using block sharded pattern (for second reader)
                reader_offset = act_l1_read_addr;
            }
#endif
            for (uint32_t height_block_index = 0; height_block_index < num_blocks_weight_h; height_block_index++) {
#ifdef SPLIT_READER
                if constexpr (split_reader_enabled) {
                    reader_idx = start_reader_idx;
                    cb_act_second_obj.reserve_back(act_block_num_tiles_split_last);

                    if (is_sender_core) {
                        l1_write_addr_act = cb_act_second_obj.get_write_ptr();
                        experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);
                        read_activation_data<
                            sliced_inner_dim,
                            dilation_w,
                            coalesced_read_bytes,
                            conv_act_c_read_bytes,
                            act_block_w_extra_align_bytes,
                            stride_w_bytes,
                            weight_size_w,
                            stride_w,
                            weight_size_h,
                            window_outer_offset>(
                            noc,
                            packed_reader_indices_ptr,
                            reader_offset,
                            l1_write_addr_act,
                            reader_idx,
                            act_l1_read_addr,
                            stride_h_bytes);
                    }
                    cb_act_second_obj.push_back(act_block_num_tiles_split_last);
                }
#endif
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
                    // MCAST RECEIVE WEIGHTS
                    // read weight blocks inner dim
                    // read weight slice - 1 block of weights in width dim and full weight matrix height
                    // read slice only once for all activation blocks
                    WATCHER_RING_BUFFER_PUSH(0xB1000000u | (rb_wcnt & 0xffff));  // recv: pre reserve cb_weight
                    cb_weight_obj.reserve_back(weight_block_num_tiles);
                    // Set weights semaphore value to INVALID
                    weights_mcast_receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    weights_mcast_sender_sem.up(noc, weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, 1);

                    WATCHER_RING_BUFFER_PUSH(0xB3000000u | (rb_wcnt & 0xffff));  // recv: bumped, pre wait VALID
                    // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                    weights_mcast_receiver_sem.wait(VALID);
                    WATCHER_RING_BUFFER_PUSH(0xB4000000u | (rb_wcnt & 0xffff));  // recv: got VALID (weight loaded)
                    rb_wcnt++;

                    cb_weight_obj.push_back(weight_block_num_tiles);
                }  // for weight_block_height_num_outer
            }
#ifdef SPLIT_READER
            if constexpr (split_reader_enabled) {
                // Update reader index for next iteration (split reader increment)
                // Only read reader indices on cores that have sharded input (is_sender_core).
                if (is_sender_core) {
                    start_reader_idx =
                        reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
                }
            }
#endif
            if constexpr (fuse_bias) {
                if (load_bias) {
#ifdef FUSE_BIAS
                    cb_bias_obj.reserve_back(bias_ntiles);

                    // Set weights semaphore value to INVALID
                    weights_mcast_receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    weights_mcast_sender_sem.up(noc, weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, 1);

                    // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                    weights_mcast_receiver_sem.wait(VALID);

                    cb_bias_obj.push_back(bias_ntiles);
                    load_bias = false;
#endif
                }
            }

        }  // out_num_blocks_h
    }  // out_num_blocks_w

    noc.async_write_barrier();
}
