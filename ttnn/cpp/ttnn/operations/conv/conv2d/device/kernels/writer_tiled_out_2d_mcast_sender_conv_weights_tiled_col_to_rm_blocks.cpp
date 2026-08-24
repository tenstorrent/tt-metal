// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metalium 2.0 writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks kernel
// (block-sharded conv2d weights+bias mcast sender; also does the split-reader second-half
// activation reads), using the typed kernel-binding surface:
//   - CB-index CTAs -> dfb:: tokens (weights / bias / act_second_reader / act_sharded / reader_indices)
//   - weight/bias TensorAccessorArgs + base-address RTAs -> tensor::weights / tensor::bias bindings
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
    uint32_t weight_block_height_ntiles,
    uint32_t weight_block_width_ntiles,
    uint32_t weight_stride_h,
    uint32_t weight_next_block_stride_w,
    uint32_t bias_ntiles,
    uint32_t out_num_blocks_h,
    uint32_t out_num_blocks_w,
    uint32_t weight_block_height_num_outer_in,
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
    uint32_t out_start_tile_id_w,
    uint32_t bias_tile_offset,
    uint32_t mcast_dest_noc_start_x,
    uint32_t mcast_dest_noc_start_y,
    uint32_t mcast_dest_noc_end_x,
    uint32_t mcast_dest_noc_end_y,
    uint32_t weights_mcast_num_dests,
    uint32_t weights_mcast_num_cores,
    uint32_t is_sender_core,
    uint32_t skip_work) {
    constexpr bool sliced_inner_dim = num_blocks_weight_h > 1;  // Derived like block sharded reader

    // mcast args
    const McastRect mcast_rect = {
        mcast_dest_noc_start_x, mcast_dest_noc_start_y, mcast_dest_noc_end_x, mcast_dest_noc_end_y};

    // Experimental API objects
    Noc noc;
    Semaphore weights_mcast_sender_sem(sem::weights_mcast_sender);
    Semaphore weights_mcast_receiver_sem(sem::weights_mcast_receiver);
    Semaphore act_split_reserve_done_sem(sem::act_split_reserve_done);
    Semaphore act_split_write_done_sem(sem::act_split_write_done);
    MulticastEndpoint mcast_ep;
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
    // Pre-built mcast destination; .addr is updated per mcast call
    McastDst mcast_dst = {
        .noc_x_start = mcast_rect.noc_x_start,
        .noc_y_start = mcast_rect.noc_y_start,
        .noc_x_end = mcast_rect.noc_x_end,
        .noc_y_end = mcast_rect.noc_y_end,
        .addr = 0};

    const bool sender_core = is_sender_core > 0;
    const bool skip_this_work = skip_work > 0;

    if (skip_this_work && !split_reader_enabled) {
        return;
    }

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
                      TensorAccessor(tensor::split_reader_indices).get_bank_base_address());
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
        split_reader_enabled ? TensorAccessor(tensor::split_act_sharded).get_bank_base_address() : 0;

    if constexpr (!skip_mcast) {
        weights_mcast_receiver_sem.set(VALID);
    }

    // read in bias if enabled (done only once for all batches)
    constexpr uint32_t bias_pagesize = fuse_bias ? get_tile_size(dfb::bias) : 0;
    const auto s_bias = TensorAccessor(tensor::bias);

    bool load_bias = true;

    constexpr uint32_t weight_tile_nbytes = get_tile_size(dfb::weights);
    const auto s_weight = TensorAccessor(tensor::weights);
    constexpr uint32_t weights_block_size_bytes = weight_tile_nbytes * weight_block_num_tiles;

    // Pre-compute constants used in tile_id calculation (preserving exact original logic)
    constexpr uint32_t tiles_per_full_block =
        num_blocks_weight_h * weight_block_height_ntiles * weight_block_height_num_outer_in * weight_block_width_ntiles;
    constexpr uint32_t height_stride_factor = weight_block_height_ntiles * weight_stride_h;

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t reader_offset = 0;
    uint32_t weight_start_tile_id = out_start_tile_id_w;
    uint32_t l1_write_addr_act = split_reader_cb_shared ? split_reader_cb_write_addr : 0;
    uint32_t previous_write_addr = 0;
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            if constexpr (split_reader_enabled) {
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
                            // Ping-pong the shared CB write address; without double buffering the alternate
                            // address is identical and this assignment is intentionally a no-op.
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
                    if (skip_this_work) {
                        continue;
                    }
                }
                // Compute height block offset once per outer loop iteration
                const uint32_t height_block_offset = height_block_index * height_stride_factor;
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
                    cb_weight_obj.reserve_back(weight_block_num_tiles);

                    const uint32_t outer_block_offset = weight_tile_h_outer_i * tiles_per_full_block;
                    uint32_t tile_id = weight_start_tile_id + height_block_offset + outer_block_offset;
                    uint32_t weight_write_offset = 0;
                    for (uint32_t block_weight_h = 0; block_weight_h < weight_block_height_ntiles; block_weight_h++) {
                        uint32_t weight_tile_id = tile_id;

                        for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles;
                             ++weight_tile_w_i) {
                            noc.async_read(
                                s_weight,
                                cb_weight_obj,
                                weight_tile_nbytes,
                                {.page_id = weight_tile_id++},
                                {.offset_bytes = weight_write_offset});
                            weight_write_offset += weight_tile_nbytes;
                        }
                        tile_id += weight_stride_h;
                    }
                    noc.async_read_barrier();

                    if constexpr (!skip_mcast) {
                        // wait until all weights mcast destinations have atomically incremented the weights
                        // semaphore_addr (i.e. its value should be weights_mcast_num_dests), then reset the
                        // semaphore_addr value back to zero for the next block
                        weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                        weights_mcast_sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        mcast_dst.addr = cb_weight_obj.get_write_ptr();
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(cb_weight_obj.get_write_ptr()),
                            mcast_ep,
                            weights_block_size_bytes,
                            weights_mcast_num_cores,
                            {},
                            mcast_dst,
                            true);

                        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                        // statically (using NOC_CMD_STATIC_VC). We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        weights_mcast_receiver_sem.set_multicast(
                            noc,
                            mcast_rect.noc_x_start,
                            mcast_rect.noc_y_start,
                            mcast_rect.noc_x_end,
                            mcast_rect.noc_y_end,
                            weights_mcast_num_cores);
                    }
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
                if (skip_this_work) {
                    continue;
                }
            }
            if constexpr (fuse_bias) {
                if (load_bias) {
                    cb_bias_obj.reserve_back(bias_ntiles);

                    uint32_t bias_write_offset = 0;
                    uint32_t bias_block_size_bytes = 0;
                    for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles;
                         ++bias_tile) {
                        noc.async_read(
                            s_bias,
                            cb_bias_obj,
                            bias_pagesize,
                            {.page_id = bias_tile},
                            {.offset_bytes = bias_write_offset});
                        bias_write_offset += bias_pagesize;
                        bias_block_size_bytes += bias_pagesize;
                    }
                    noc.async_read_barrier();

                    // MCAST BIAS (shares some mcast args with weights)
                    if constexpr (!skip_mcast) {
                        // wait until all weights mcast destinations have atomically incremented the weights
                        // semaphore_addr (i.e. its value should be weights_mcast_num_dests), then reset the
                        // semaphore_addr value back to zero for the next block
                        weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                        weights_mcast_sender_sem.set(0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        mcast_dst.addr = cb_bias_obj.get_write_ptr();
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(cb_bias_obj.get_write_ptr()),
                            mcast_ep,
                            bias_block_size_bytes,
                            weights_mcast_num_cores,
                            {},
                            mcast_dst,
                            true);

                        // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                        // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                        // statically (using NOC_CMD_STATIC_VC). We should also multicast the flag to destinations
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        weights_mcast_receiver_sem.set_multicast(
                            noc,
                            mcast_rect.noc_x_start,
                            mcast_rect.noc_y_start,
                            mcast_rect.noc_x_end,
                            mcast_rect.noc_y_end,
                            weights_mcast_num_cores);
                    }

                    cb_bias_obj.push_back(bias_ntiles);
                    load_bias = false;
                }
            }

        }  // out_num_blocks_h

        // Increment weight start tile id for next block in width dim
        weight_start_tile_id += weight_next_block_stride_w;
    }  // out_num_blocks_w

    noc.async_write_barrier();
}
