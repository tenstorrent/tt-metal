// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/dataflow/dataflow_api.h>
#include "conv_reader_common.hpp"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);

    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(8);
    constexpr uint32_t weight_block_height_ntiles = get_compile_time_arg_val(9);
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(10);
    constexpr uint32_t weight_stride_h = get_compile_time_arg_val(11);
    constexpr uint32_t weight_next_block_stride_w = get_compile_time_arg_val(13);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);

    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(16);
    constexpr uint32_t weight_block_height_num_outer_in = get_compile_time_arg_val(17);

    constexpr bool fuse_bias = get_compile_time_arg_val(18);

    constexpr bool split_reader_enabled = get_compile_time_arg_val(19);

    // Split reader args
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(5);
    constexpr uint32_t window_outer = get_compile_time_arg_val(6);  // num_blocks_act_w
    constexpr bool sliced_inner_dim = num_blocks_weight_h > 1;      // Derived like block sharded reader
    constexpr uint32_t act_block_num_tiles_split_last = get_compile_time_arg_val(21);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(22);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(23);
    constexpr uint32_t padded_conv_act_size_w = get_compile_time_arg_val(24);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(25);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(27);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(28);
    constexpr uint32_t stride_w = get_compile_time_arg_val(29);
    constexpr uint32_t weight_size_h = get_compile_time_arg_val(30);  // Input filter window height

    constexpr bool split_reader_cb_shared = get_compile_time_arg_val(31) == 1;

    // When the split reader CB is shared, both readers write to the same circular buffer.
    // Synchronization is required: the main reader signals when CB space is reserved,
    // and the second reader signals when it has finished writing its portion.
    experimental::Semaphore<> reserve_done_sem((split_reader_cb_shared) ? get_compile_time_arg_val(32) : 0);
    experimental::Semaphore<> write_done_sem((split_reader_cb_shared) ? get_compile_time_arg_val(33) : 0);
    constexpr uint32_t act_write_offset = get_compile_time_arg_val(34);
    constexpr uint32_t act_write_offset_last = get_compile_time_arg_val(35);

    experimental::CB cb_act_second_obj(cb_id_act_second_reader);
    const uint32_t split_reader_cb_write_addr =
        (split_reader_cb_shared) ? cb_act_second_obj.get_write_ptr() + act_write_offset : 0;
    // In case of double buffering the split reader can write to two different addresses
    const uint32_t split_reader_cb_write_addr_last =
        (split_reader_cb_shared) ? cb_act_second_obj.get_write_ptr() + act_write_offset_last : 0;
    const uint32_t split_reader_cb_write_addr_sum = split_reader_cb_write_addr + split_reader_cb_write_addr_last;

    constexpr uint32_t ct_arg_idx = 36;
    constexpr auto s_weight_args = TensorAccessorArgs<ct_arg_idx>();
    constexpr auto s_bias_args = TensorAccessorArgs<s_weight_args.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i++);
    // Bias arg. Unused if bias fusion is not enabled.
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);
    const uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i++);
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i++);

    // mcast args
    const McastRect mcast_rect = {
        get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++)};
    const uint32_t weights_mcast_num_dests = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_cores = get_arg_val<uint32_t>(i++);

    // Experimental API objects
    experimental::Noc noc;
    experimental::Semaphore<> weights_mcast_sender_sem(get_arg_val<uint32_t>(i++));
    experimental::Semaphore<> weights_mcast_receiver_sem(get_arg_val<uint32_t>(i++));
    experimental::MulticastEndpoint mcast_ep;
    experimental::CB cb_weight_obj(cb_id_weight);
    experimental::CB cb_bias_obj(bias_cb_id);
    experimental::CB cb_reader_indices_obj(cb_reader_indices);
    experimental::CB cb_sharded_act_obj(cb_id_sharded_act);
    // Pre-built mcast destination; .addr is updated per mcast call
    McastDst mcast_dst = {
        .noc_x_start = mcast_rect.noc_x_start,
        .noc_y_start = mcast_rect.noc_y_start,
        .noc_x_end = mcast_rect.noc_x_end,
        .noc_y_end = mcast_rect.noc_y_end,
        .addr = 0};

    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;
    const bool skip_work = get_arg_val<uint32_t>(i++) > 0;

    if (skip_work && !split_reader_enabled) {
        return;
    }

    // Split reader configuration
    if constexpr (split_reader_enabled) {
#ifdef CONFIG_TENSOR_IN_DRAM
        cb_reader_indices_obj.wait_front(1);
#endif
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<cb_id_act_second_reader>(noc, cb_act_second_obj);
        }
    }

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

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    weights_mcast_receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
#endif

    // read in bias if enabled (done only once for all batches)
    constexpr uint32_t bias_pagesize =
        fuse_bias ? get_tile_size(bias_cb_id) : 0;  // dummy value in case bias is not enabled
    const auto s_bias = TensorAccessor(s_bias_args, bias_addr, bias_pagesize);

    bool load_bias = true;

    constexpr uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    const auto s_weight = TensorAccessor(s_weight_args, weight_addr_dram_base, weight_tile_nbytes);
    constexpr uint32_t weights_block_size_bytes = weight_tile_nbytes * weight_block_num_tiles;

    // Pre-compute constants used in tile_id calculation (preserving exact original logic)
    constexpr uint32_t tiles_per_full_block =
        num_blocks_weight_h * weight_block_height_ntiles * weight_block_height_num_outer_in * weight_block_width_ntiles;
    constexpr uint32_t height_stride_factor = weight_block_height_ntiles * weight_stride_h;

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t reader_offset = 0;
    uint32_t weight_start_tile_id = out_start_tile_id_w;
    uint32_t l1_write_addr_act = split_reader_cb_write_addr;
    uint32_t prev_addr = 0;
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
                    if (is_sender_core) {
                        if constexpr (split_reader_cb_shared) {
                            reserve_done_sem.wait(VALID);
                            reserve_done_sem.set(INVALID);
                            prev_addr = l1_write_addr_act;
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
                            // in case of shared cb we update the write address (it will remain the same if double
                            // buffering is not enabled)
                            l1_write_addr_act = split_reader_cb_write_addr_sum - prev_addr;
                            write_done_sem.set(VALID);
                        }
                    }
                    if constexpr (!split_reader_cb_shared) {
                        cb_act_second_obj.push_back(act_block_num_tiles_split_last);
                    }
                    if (skip_work) {
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

#ifndef SKIP_MCAST
                    // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
                    // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to
                    // zero for the next block
                    weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                    weights_mcast_sender_sem.set(0);

                    // Now we have the block in the CB address, we can mcast to dests!
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    mcast_dst.addr = cb_weight_obj.get_write_ptr();
                    noc.async_write_multicast(
                        experimental::use<experimental::CB::AddrSelector::WRITE_PTR>(cb_weight_obj),
                        mcast_ep,
                        weights_block_size_bytes,
                        weights_mcast_num_cores,
                        {},
                        mcast_dst,
                        true);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and same
                    // vc even though cmd bufs are different Also, this only works because we are setting VCs statically
                    // (using NOC_CMD_STATIC_VC).
                    // We should also multicast the flag to destinations
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    weights_mcast_receiver_sem.set_multicast(
                        noc,
                        mcast_rect.noc_x_start,
                        mcast_rect.noc_y_start,
                        mcast_rect.noc_x_end,
                        mcast_rect.noc_y_end,
                        weights_mcast_num_cores);
#endif
                    cb_weight_obj.push_back(weight_block_num_tiles);
                }  // for weight_block_height_num_outer
            }
            if constexpr (split_reader_enabled) {
                // Update reader index for next iteration (split reader increment)
                // Only read reader indices on cores that have sharded input (is_sender_core).
                if (is_sender_core) {
                    start_reader_idx =
                        reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
                }
                if (skip_work) {
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
#ifndef SKIP_MCAST
                    // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
                    // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to
                    // zero for the next block
                    weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                    weights_mcast_sender_sem.set(0);

                    // Now we have the block in the CB address, we can mcast to dests!
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    mcast_dst.addr = cb_bias_obj.get_write_ptr();
                    noc.async_write_multicast(
                        experimental::use<experimental::CB::AddrSelector::WRITE_PTR>(cb_bias_obj),
                        mcast_ep,
                        bias_block_size_bytes,
                        weights_mcast_num_cores,
                        {},
                        mcast_dst,
                        true);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and same
                    // vc even though cmd bufs are different Also, this only works because we are setting VCs statically
                    // (using NOC_CMD_STATIC_VC).
                    // We should also multicast the flag to destinations
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    weights_mcast_receiver_sem.set_multicast(
                        noc,
                        mcast_rect.noc_x_start,
                        mcast_rect.noc_y_start,
                        mcast_rect.noc_x_end,
                        mcast_rect.noc_y_end,
                        weights_mcast_num_cores);
#endif

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
