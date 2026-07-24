// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp
// (height-sharded conv2d weights mcast receiver; also reads the activation second-reader half on
// split-reader paths).
//
// Despite the filename this kernel never writes the OUT buffer: it receives multicast weights (and
// bias) and, when split reader is on, reads the second-reader half of the activation. In the Metal
// 2.0 factory OUT is bound to this kernel as a DEGENERATE CONSUMER (resolution #1); there is no
// out-CB code in the body.
//
// Algorithm body identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs -> dfb:: tokens (weights / bias / act_second_reader / act_sharded / reader_indices)
//   - weights-mcast semaphore RTAs -> Semaphore(sem::weights_mcast_sender / weights_mcast_receiver)
//   - remaining positional CTAs -> get_arg(args::name); remaining RTAs -> get_arg(args::name)
//   - DataflowBuffer -> DataflowBuffer (objects passed to conv_reader_common.hpp helpers stay
//     experimental::CB); dfb::bias gated behind FUSE_BIAS; dfb::act_second_reader behind SPLIT_READER

#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"

void kernel_main() {
    constexpr uint32_t num_blocks_weight_h = get_arg(args::num_blocks_weight_h);
    constexpr uint32_t weight_block_num_tiles = get_arg(args::weight_block_num_tiles);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_arg(args::bias_ntiles);

    constexpr uint32_t out_num_blocks_h = get_arg(args::out_num_blocks_h);

    constexpr bool fuse_bias = get_arg(args::fuse_bias);

    constexpr bool split_reader_enabled = get_arg(args::split_reader_enabled);
    constexpr bool activation_reuse_enabled = get_arg(args::activation_reuse_enabled);

    // Split reader args
    constexpr uint32_t act_block_num_tiles = get_arg(args::act_block_num_tiles);
    constexpr uint32_t conv_act_c_read_bytes = get_arg(args::conv_act_c_read_bytes);
    constexpr uint32_t weight_size_w = get_arg(args::weight_size_w);
    constexpr uint32_t conv_act_size_w_padded = get_arg(args::conv_act_size_w_padded);
    constexpr uint32_t act_block_w_extra_align_bytes = get_arg(args::act_block_w_extra_align_bytes);
    constexpr bool needs_act_block_zero_out = get_arg(args::needs_act_block_zero_out) == 1;
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t weights_size_h = get_arg(args::weights_size_h);

    // Activation reuse args
    constexpr uint32_t act_reuse_cb_tiles = get_arg(args::act_reuse_cb_tiles);
    constexpr uint32_t act_block_w_tiles = get_arg(args::act_block_w_tiles);
    constexpr bool readers_process_full_image_widths = get_arg(args::readers_process_full_image_widths) == 1;
    constexpr uint32_t image_width_tiles = get_arg(args::image_width_tiles);
    constexpr uint32_t output_image_width = get_arg(args::output_image_width);
    constexpr uint32_t window_reuse_offset = get_arg(args::window_reuse_offset);
    constexpr bool need_to_push_remaining_tiles = get_arg(args::need_to_push_remaining_tiles) == 1;
    constexpr bool single_core_processes_multiple_batches = get_arg(args::single_core_processes_multiple_batches) == 1;

    const uint32_t noop = get_arg(args::noop);

    if (noop) {
        return;
    }

    // Experimental API objects
    Noc noc;

#ifdef SPLIT_READER
    DataflowBuffer cb_act_second_obj(dfb::act_second_reader);
    if constexpr (split_reader_enabled) {
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<dfb::act_second_reader>(noc, cb_act_second_obj);
        }
    }
#endif

    // mcast args
    const uint32_t weights_mcast_sender_noc_x = get_arg(args::weights_mcast_sender_noc_x);
    const uint32_t weights_mcast_sender_noc_y = get_arg(args::weights_mcast_sender_noc_y);
    Semaphore<> weights_mcast_sender_sem(sem::weights_mcast_sender);
    Semaphore<> weights_mcast_receiver_sem(sem::weights_mcast_receiver);
    DataflowBuffer cb_weight_obj(dfb::weights);
#ifdef SPLIT_READER
    DataflowBuffer cb_reader_indices_obj(dfb::reader_indices);
    DataflowBuffer cb_sharded_act_obj(dfb::act_sharded);
#endif
#ifdef FUSE_BIAS
    DataflowBuffer cb_bias_obj(dfb::bias);
#endif

#ifdef SPLIT_READER
    const uint32_t remaining_tiles_to_push =
        split_reader_enabled && activation_reuse_enabled ? get_arg(args::remaining_tiles_to_push) : 0;

    // Split reader configuration
    if constexpr (split_reader_enabled) {
#ifdef CONFIG_TENSOR_IN_DRAM
        cb_reader_indices_obj.wait_front(1);
#endif
    }
    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        split_reader_enabled ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_indices_obj.get_write_ptr())
                             : nullptr;
    uint32_t reader_idx = 0;
    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    const uint32_t act_l1_read_addr = split_reader_enabled ? cb_sharded_act_obj.get_read_ptr() : 0;
    uint32_t start_reader_idx =
        split_reader_enabled ? (uint32_t)(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1 : 0;
    const uint32_t cb_start_addr = split_reader_enabled ? cb_act_second_obj.get_write_ptr() : 0;
    uint32_t reader_offset = 0;
#endif

    // read in bias if enabled (done only once for all batches)
#ifdef FUSE_BIAS
    bool load_bias = true;
#endif
    [[maybe_unused]] uint32_t l1_write_addr_act = 0;
    for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        // MCAST RECEIVE WEIGHTS
        // read weight blocks inner dim
        // read weight slice - 1 block of weights in width dim and full weight matrix height
        // read slice only once for all activation blocks

#ifdef SPLIT_READER
        if constexpr (split_reader_enabled) {
            if constexpr (activation_reuse_enabled) {
                l1_write_addr_act = cb_start_addr;
                cb_act_second_obj.evil_set_write_ptr(l1_write_addr_act);
            }
            reader_offset = act_l1_read_addr;
        }
#endif
        for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
#ifdef SPLIT_READER
            if constexpr (split_reader_enabled) {
                // Do the second half of the reads for act
                experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);
                reader_idx = start_reader_idx;

                if constexpr (!activation_reuse_enabled) {
                    cb_act_second_obj.reserve_back(act_block_num_tiles);
                    l1_write_addr_act = cb_act_second_obj.get_write_ptr();
                    read_sticks<
                        dilation_w,
                        coalesced_read_bytes,
                        conv_act_c_read_bytes,
                        act_block_w_extra_align_bytes,
                        stride_w_bytes,
                        weight_size_w,
                        stride_w>(noc, packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);
                    noc.async_read_barrier();
                    cb_act_second_obj.push_back(act_block_num_tiles);

                    reader_offset += window_outer_offset;
                } else {
                    read_sticks_activation_reuse<
                        coalesced_read_bytes,
                        conv_act_c_read_bytes,
                        act_block_w_extra_align_bytes,
                        window_outer_offset,
                        weight_size_w,
                        stride_w,
                        weights_size_h,
                        dfb::act_second_reader,
                        act_reuse_cb_tiles,
                        act_block_w_tiles,
                        readers_process_full_image_widths,
                        image_width_tiles,
                        output_image_width,
                        window_reuse_offset,
                        single_core_processes_multiple_batches>(
                        noc,
                        cb_act_second_obj,
                        packed_reader_indices_ptr,
                        act_l1_read_addr,
                        l1_write_addr_act,
                        reader_idx,
                        cb_start_addr);

                    if constexpr (need_to_push_remaining_tiles) {
                        if (block_weight_h == num_blocks_weight_h - 1) {
                            // Last core sometimes has less work to do, but we still need to push the same number of
                            // tiles to avoid blocking compute kernels
                            push_remaining_tiles<dfb::act_second_reader, act_block_w_tiles, image_width_tiles>(
                                cb_act_second_obj, remaining_tiles_to_push, cb_start_addr);
                        }
                    }
                }
            }
#endif

            // Receive weights
            cb_weight_obj.reserve_back(weight_block_num_tiles);
            if (bh == 0) {
                // Set weights semaphore value to INVALID
                weights_mcast_receiver_sem.set(INVALID);

                // Atomic increment source core counter
                weights_mcast_sender_sem.up(noc, weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, 1);

                // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts
                // data)
                weights_mcast_receiver_sem.wait(VALID);
            }

            cb_weight_obj.push_back(weight_block_num_tiles);
        }

#ifdef FUSE_BIAS
        if constexpr (fuse_bias) {
            if (load_bias) {
                cb_bias_obj.reserve_back(bias_ntiles);

                // Set weights semaphore value to INVALID
                weights_mcast_receiver_sem.set(INVALID);

                // Atomic increment source core counter
                weights_mcast_sender_sem.up(noc, weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, 1);

                // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                weights_mcast_receiver_sem.wait(VALID);

                cb_bias_obj.push_back(bias_ntiles);
                load_bias = false;
            }
        }
#endif

#ifdef SPLIT_READER
        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
        }
#endif
    }  // out_num_blocks_h

    // Drain outstanding NOC writes AND atomics (weights_mcast_sender_sem.up) before returning. Under
    // Metal 2.0 the FW kernel epilogue does not drain the kernel's outstanding NOC transactions like
    // the legacy runtime did, so returning with an un-acked atomic leaves the core "running" and it
    // never signals program completion -> dispatch process_wait hangs.
    noc.async_full_barrier();
}
