// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metalium 2.0 writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks kernel
// (block-sharded conv2d weights+bias mcast receiver; also does the split-reader second-half
// activation reads), using the typed kernel-binding surface:
//   - CB-index CTAs -> dfb:: tokens (weights / bias / act_second_reader / act_sharded / reader_indices)
//   - weights mcast semaphore-id RTAs -> sem::weights_mcast_sender / sem::weights_mcast_receiver
//   - compile-time choices and RTAs -> TT_KERNEL template and function arguments
//   - DataflowBuffer -> DataflowBuffer; get_tile_size(cb) -> cb.get_entry_size()
//
// Shared split-reader overlap writes directly into the existing ACT DFB subrange and synchronizes with
// the activation reader through named semaphores; it allocates no staging DFB.
//
// Despite the "tiled_out" filename this kernel never writes OUT; the factory binds OUT as a
// degenerate consumer (resolution #1).

#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"

template <
    uint32_t num_blocks_weight_h,
    uint32_t weight_block_num_tiles,
    uint32_t weight_block_height_num_outer,
    uint32_t bias_ntiles,
    uint32_t out_num_blocks_h,
    uint32_t out_num_blocks_w,
    uint32_t fuse_bias,
    uint32_t split_reader_enabled,
    uint32_t config_tensor_in_dram,
    uint32_t window_outer,
    uint32_t act_block_num_tiles_split_last,
    uint32_t conv_act_c_read_bytes,
    uint32_t weight_size_w,
    uint32_t padded_conv_act_size_w,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t needs_act_block_zero_out,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t stride_w,
    uint32_t weight_size_h,
    uint32_t split_reader_cb_shared,
    uint32_t act_write_offset,
    uint32_t act_write_offset_last,
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
    uint32_t weights_mcast_sender_noc_x, uint32_t weights_mcast_sender_noc_y, uint32_t is_sender_core) {
    constexpr bool sliced_inner_dim = window_outer > 1;  // Derived like block sharded reader

    // Experimental API objects
    Noc noc;
    Semaphore weights_mcast_sender_sem(sem::weights_mcast_sender);
    Semaphore weights_mcast_receiver_sem(sem::weights_mcast_receiver);
    Semaphore act_split_reserve_done_sem(sem::act_split_reserve_done);
    Semaphore act_split_write_done_sem(sem::act_split_write_done);
    DataflowBuffer cb_weight_obj(dfb::weights);
    uint32_t split_reader_cb_write_addr = 0;
    uint32_t split_reader_cb_write_addr_last = 0;
    uint32_t split_reader_cb_write_addr_sum = 0;
    DataflowBuffer cb_bias_obj(dfb::bias);
    DataflowBuffer cb_reader_indices_obj(dfb::reader_indices);
    DataflowBuffer cb_act_second_obj(dfb::act_second_reader);
    split_reader_cb_write_addr = split_reader_cb_shared ? cb_act_second_obj.get_write_ptr() + act_write_offset : 0;
    split_reader_cb_write_addr_last =
        split_reader_cb_shared ? cb_act_second_obj.get_write_ptr() + act_write_offset_last : 0;
    split_reader_cb_write_addr_sum = split_reader_cb_write_addr + split_reader_cb_write_addr_last;

    const bool sender_core = is_sender_core > 0;

    // Split reader configuration
    if constexpr (split_reader_enabled) {
        if constexpr (config_tensor_in_dram) {
            cb_reader_indices_obj.wait_front(1);
        }
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<dfb::act_second_reader>(noc, cb_act_second_obj);
        }
    }

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr = nullptr;
    if constexpr (split_reader_enabled) {
        packed_reader_indices_ptr =
            config_tensor_in_dram
                ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_indices_obj.get_write_ptr())
                : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                      (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::split_reader_indices).get_noc_addr(0)));
    }
    packed_reader_indices_ptr = sender_core ? packed_reader_indices_ptr : nullptr;

    // Initial setup for second reader (starting from second reader's data)
    // Only read reader indices on cores that have sharded input (sender_core).
    uint32_t start_reader_idx =
        (split_reader_enabled && sender_core) ? (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1 : 0;
    uint32_t reader_idx = start_reader_idx;

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    constexpr uint32_t window_outer_offset = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;

    const uint32_t act_l1_read_addr =
        split_reader_enabled
            ? (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::split_act_sharded).get_noc_addr(0))
            : 0;

    // read in bias if enabled (done only once for all batches)
    bool load_bias = true;

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t l1_write_addr_act = split_reader_cb_shared ? split_reader_cb_write_addr : 0;
    uint32_t previous_write_addr = 0;
    uint32_t reader_offset = 0;
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            if constexpr (split_reader_enabled) {
                // Read activation data using block sharded pattern (for second reader)
                reader_offset = act_l1_read_addr;
            }
            for (uint32_t height_block_index = 0; height_block_index < num_blocks_weight_h; height_block_index++) {
                if constexpr (split_reader_enabled) {
                    reader_idx = start_reader_idx;
                    if constexpr (!split_reader_cb_shared) {
                        cb_act_second_obj.reserve_back(act_block_num_tiles_split_last);
                    }

                    if (sender_core) {
                        if constexpr (split_reader_cb_shared) {
                            act_split_reserve_done_sem.wait(VALID);
                            act_split_reserve_done_sem.set(INVALID);
                            previous_write_addr = l1_write_addr_act;
                        } else {
                            l1_write_addr_act = cb_act_second_obj.get_write_ptr();
                        }
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
                        if constexpr (split_reader_cb_shared) {
                            l1_write_addr_act = split_reader_cb_write_addr_sum - previous_write_addr;
                            act_split_write_done_sem.set(VALID);
                        }
                    }
                    if constexpr (!split_reader_cb_shared) {
                        cb_act_second_obj.push_back(act_block_num_tiles_split_last);
                    }
                }
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
                    // MCAST RECEIVE WEIGHTS
                    // read weight blocks inner dim
                    // read weight slice - 1 block of weights in width dim and full weight matrix height
                    // read slice only once for all activation blocks
                    cb_weight_obj.reserve_back(weight_block_num_tiles);
                    // Set weights semaphore value to INVALID
                    weights_mcast_receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    weights_mcast_sender_sem.up(noc, weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, 1);

                    // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                    weights_mcast_receiver_sem.wait(VALID);
                    cb_weight_obj.push_back(weight_block_num_tiles);
                }  // for weight_block_height_num_outer
            }
            if constexpr (split_reader_enabled) {
                // Update reader index for next iteration (split reader increment)
                // Only read reader indices on cores that have sharded input (sender_core).
                if (sender_core) {
                    start_reader_idx =
                        reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
                }
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

        }  // out_num_blocks_h
    }  // out_num_blocks_w

    // Drain non-posted semaphore atomics before the Metalium 2.0 kernel epilogue returns.
    noc.async_full_barrier();
}
