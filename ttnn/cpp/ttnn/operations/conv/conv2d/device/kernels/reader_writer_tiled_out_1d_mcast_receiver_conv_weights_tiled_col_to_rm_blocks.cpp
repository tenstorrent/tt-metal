// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metalium 2.0 reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks kernel
// (height-sharded conv2d weights mcast receiver; also reads the activation second-reader half on
// split-reader paths).
//
// Despite the filename this kernel never writes the OUT buffer: it receives multicast weights (and
// bias) and, when split reader is on, reads the second-reader half of the activation. In the Metal
// 2.0 factory OUT is bound to this kernel as a DEGENERATE CONSUMER (resolution #1); there is no
// out-CB code in the body.
//
// Uses the typed kernel-binding surface:
//   - CB-index CTAs -> dfb:: tokens (weights / bias / act_second_reader / act_sharded / reader_indices)
//   - weights-mcast semaphore RTAs -> Semaphore(sem::weights_mcast_sender / weights_mcast_receiver)
//   - compile-time choices and RTAs -> TT_KERNEL template and function arguments
//   - DataflowBuffer -> DataflowBuffer (objects passed to conv_reader_common.hpp helpers stay
//     experimental::CB); optional bias and split-reader paths use named constexpr arguments

#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"

template <
    uint32_t num_blocks_weight_h,
    uint32_t weight_block_num_tiles,
    uint32_t bias_ntiles,
    uint32_t out_num_blocks_h,
    uint32_t fuse_bias,
    uint32_t split_reader_enabled,
    uint32_t config_tensor_in_dram,
    uint32_t activation_reuse_enabled,
    uint32_t act_block_num_tiles,
    uint32_t conv_act_c_read_bytes,
    uint32_t weight_size_w,
    uint32_t conv_act_size_w_padded,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t needs_act_block_zero_out,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t stride_w,
    uint32_t weights_size_h,
    uint32_t skip_mcast,
    uint32_t act_reuse_cb_tiles,
    uint32_t act_block_w_tiles,
    uint32_t readers_process_full_image_widths,
    uint32_t image_width_tiles,
    uint32_t output_image_width,
    uint32_t window_reuse_offset,
    uint32_t need_to_push_remaining_tiles,
    uint32_t single_core_processes_multiple_batches>
TT_KERNEL void kernel_main(
    uint32_t noop,
    uint32_t weights_mcast_sender_noc_x,
    uint32_t weights_mcast_sender_noc_y,
    uint32_t remaining_tiles_to_push) {
    if (noop) {
        return;
    }

    // Experimental API objects
    Noc noc;

    DataflowBuffer cb_act_second_obj(dfb::act_second_reader);
    if constexpr (split_reader_enabled) {
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<dfb::act_second_reader>(noc, cb_act_second_obj);
        }
    }

    // mcast args
    Semaphore<> weights_mcast_sender_sem(sem::weights_mcast_sender);
    Semaphore<> weights_mcast_receiver_sem(sem::weights_mcast_receiver);
    DataflowBuffer cb_weight_obj(dfb::weights);
    DataflowBuffer cb_bias_obj(dfb::bias);
    DataflowBuffer cb_reader_indices_obj(dfb::reader_indices);

    if constexpr (split_reader_enabled && config_tensor_in_dram) {
        cb_reader_indices_obj.wait_front(1);
    }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr = nullptr;
    if constexpr (split_reader_enabled) {
        packed_reader_indices_ptr =
            config_tensor_in_dram
                ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_indices_obj.get_write_ptr())
                : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                      (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::split_reader_indices).get_noc_addr(0)));
    }
    uint32_t reader_idx = 0;
    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    const uint32_t act_l1_read_addr =
        split_reader_enabled
            ? (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::split_act_sharded).get_noc_addr(0))
            : 0;
    uint32_t start_reader_idx =
        split_reader_enabled ? (uint32_t)(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1 : 0;
    const uint32_t cb_start_addr = split_reader_enabled ? cb_act_second_obj.get_write_ptr() : 0;
    uint32_t reader_offset = 0;

    // read in bias if enabled (done only once for all batches)
    bool load_bias = true;

    [[maybe_unused]] uint32_t l1_write_addr_act = 0;
    for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        // MCAST RECEIVE WEIGHTS
        // read weight blocks inner dim
        // read weight slice - 1 block of weights in width dim and full weight matrix height
        // read slice only once for all activation blocks

        if constexpr (split_reader_enabled) {
            if constexpr (activation_reuse_enabled) {
                l1_write_addr_act = cb_start_addr;
                get_local_cb_interface(dfb::act_second_reader).fifo_wr_ptr = l1_write_addr_act;
            }
            reader_offset = act_l1_read_addr;
        }
        for (uint32_t block_weight_h = 0; block_weight_h < num_blocks_weight_h; block_weight_h++) {
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
                            push_remaining_tiles<act_block_w_tiles, image_width_tiles>(
                                cb_act_second_obj, remaining_tiles_to_push, cb_start_addr);
                        }
                    }
                }
            }

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

        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
        }
    }  // out_num_blocks_h

    // Drain outstanding NOC writes AND atomics (weights_mcast_sender_sem.up) before returning. Under
    // Metal 2.0 the FW kernel epilogue does not drain the kernel's outstanding NOC transactions like
    // the legacy runtime did, so returning with an un-acked atomic leaves the core "running" and it
    // never signals program completion -> dispatch process_wait hangs.
    noc.async_full_barrier();
}
