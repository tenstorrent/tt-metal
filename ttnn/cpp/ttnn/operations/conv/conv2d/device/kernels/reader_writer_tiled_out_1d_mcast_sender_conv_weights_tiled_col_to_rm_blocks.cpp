// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <api/dataflow/dataflow_api.h>
#include "conv_reader_common.hpp"

void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_act_second_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(7);

    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(8);
    constexpr uint32_t weight_block_height_ntiles = get_compile_time_arg_val(9);
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(10);
    constexpr uint32_t weight_stride_h = get_compile_time_arg_val(11);
    constexpr uint32_t weight_next_block_stride_h = get_compile_time_arg_val(12);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(14);

    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(15);

    constexpr bool fuse_bias = get_compile_time_arg_val(18);

    constexpr bool split_reader_enabled = get_compile_time_arg_val(19);
    constexpr bool activation_reuse_enabled = get_compile_time_arg_val(20);

    // Split reader args
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(22);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(23);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(24);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(25);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t dilation_h = get_compile_time_arg_val(27);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(28);
    constexpr uint32_t stride_w = get_compile_time_arg_val(29);
    constexpr uint32_t weights_size_h = get_compile_time_arg_val(30);

    // Activation reuse args
    constexpr uint32_t act_reuse_cb_tiles = get_compile_time_arg_val(31);
    constexpr uint32_t act_block_w_tiles = get_compile_time_arg_val(32);
    constexpr bool readers_process_full_image_widths = get_compile_time_arg_val(33) == 1;
    constexpr uint32_t image_width_tiles = get_compile_time_arg_val(34);
    constexpr uint32_t output_image_width = get_compile_time_arg_val(35);
    constexpr uint32_t window_reuse_offset = get_compile_time_arg_val(36);
    constexpr bool need_to_push_remaining_tiles = get_compile_time_arg_val(37) == 1;
    constexpr bool single_core_processes_multiple_batches = get_compile_time_arg_val(38) == 1;

    constexpr auto s_weight_args = TensorAccessorArgs<39>();
    constexpr auto s_bias_args = TensorAccessorArgs<s_weight_args.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i++);
    // Bias arg. Unused if bias fusion is not enabled.
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);

    const uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i++);
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i++);

    // Experimental API objects
    experimental::Noc noc;

    if constexpr (split_reader_enabled) {
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<cb_id_act_second_reader>(noc, experimental::CB(cb_id_act_second_reader));
        }
    }

    // mcast args
    const struct {
        uint32_t noc_x_start, noc_y_start, noc_x_end, noc_y_end;
    } mcast_rect = {
        get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++)};
    const uint32_t weights_mcast_num_dests = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_cores = get_arg_val<uint32_t>(i++);
    experimental::Semaphore<> weights_mcast_sender_sem(get_arg_val<uint32_t>(i++));
    experimental::Semaphore<> weights_mcast_receiver_sem(get_arg_val<uint32_t>(i++));
    experimental::MulticastEndpoint mcast_ep;
    experimental::CB cb_weight_obj(cb_id_weight);
    experimental::CB cb_bias_obj(bias_cb_id);
    experimental::CB cb_act_second_obj(cb_id_act_second_reader);

    const uint32_t remaining_tiles_to_push =
        split_reader_enabled && activation_reuse_enabled ? get_arg_val<uint32_t>(i++) : 0;

    // Split reader configuration
    if constexpr (split_reader_enabled) {
#ifdef CONFIG_TENSOR_IN_DRAM
        cb_wait_front(cb_reader_indices, 1);
#endif
    }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        split_reader_enabled ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices))
                             : nullptr;
    uint32_t reader_idx = 0;
    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    uint32_t act_l1_read_addr = split_reader_enabled ? get_read_ptr(cb_id_sharded_act) : 0;
    // coalesce reads along weight_size_w
    uint32_t start_reader_idx = split_reader_enabled ? (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1 : 0;
    uint32_t cb_start_addr = split_reader_enabled ? cb_act_second_obj.get_write_ptr() : 0;
    uint32_t reader_offset = act_l1_read_addr;

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    weights_mcast_receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
#endif

    // read in bias if enabled (done only once for all batches)
    constexpr uint32_t bias_pagesize =
        fuse_bias ? get_tile_size(bias_cb_id) : 0;  // dummy but valid value in case bias is not enabled
    const auto s_bias = TensorAccessor(s_bias_args, bias_addr, bias_pagesize);
    bool load_bias = true;

    constexpr uint32_t weight_tile_nbytes = get_tile_size(cb_id_weight);
    const auto s_weight = TensorAccessor(s_weight_args, weight_addr_dram_base, weight_tile_nbytes);

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    constexpr uint32_t weight_inner_block_stride_h =
        weight_next_block_stride_h / weight_block_height_num_outer;  // TODO: Pass as args

    uint32_t l1_write_addr_act = 0;
    for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
        // READ WEIGHTS + MCAST SEND WEIGHTS
        // read weight blocks inner dim
        // read weight slice - 1 block of weights in width dim and full weight matrix height
        // read slice only once for all activation blocks
        uint32_t weight_h_offset = 0;

        uint32_t weight_current_block_start_tile_id = 0;

        if constexpr (split_reader_enabled) {
            if constexpr (activation_reuse_enabled) {
                l1_write_addr_act = cb_start_addr;
                get_local_cb_interface(cb_id_act_second_reader).fifo_wr_ptr = l1_write_addr_act;
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
                        cb_id_act_second_reader,
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
                            push_remaining_tiles<cb_id_act_second_reader, act_block_w_tiles, image_width_tiles>(
                                cb_act_second_obj, remaining_tiles_to_push, cb_start_addr);
                        }
                    }
                }
            }

            // Do weights read + mcast
            cb_weight_obj.reserve_back(weight_block_num_tiles);
            if (bh == 0) {
                uint32_t weight_row_start_tile_id = weight_current_block_start_tile_id + weight_h_offset;

                uint32_t weight_write_offset = 0;
                uint32_t weights_block_size_bytes = 0;

                // loop over weight block tiles along h
                for (uint32_t weight_tile_h_i = 0; weight_tile_h_i < weight_block_height_ntiles; ++weight_tile_h_i) {
                    uint32_t weight_tile_id = weight_row_start_tile_id;
                    // loop over weight block tiles along w
                    for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles; ++weight_tile_w_i) {
                        // DPRINT << "weight_tile_id=" << weight_tile_id << ENDL();
                        noc.async_read(
                            s_weight,
                            cb_weight_obj,
                            weight_tile_nbytes,
                            {.page_id = weight_tile_id},
                            {.offset_bytes = weight_write_offset});
                        weight_write_offset += weight_tile_nbytes;
                        weights_block_size_bytes += weight_tile_nbytes;
                        weight_tile_id += 1;
                    }  // for weight_block_w
                    weight_row_start_tile_id += weight_stride_h;
                }  // for weight_block_h
                noc.async_read_barrier();

#ifndef SKIP_MCAST
                // wait until all weights mcast destinations have atomically incremented the weights
                // semaphore_addr (i.e. its value should be weights_mcast_num_dests), then reset the
                // semaphore_addr value back to zero for the next block
                weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                weights_mcast_sender_sem.set(0);

                // Now we have the block in the CB address, we can mcast to dests!
                // num_dests must not include source, since we are NOT really doing a local copy!
                noc.async_write_multicast(
                    experimental::use<experimental::CB::AddrSelector::WRITE_PTR>(cb_weight_obj),
                    mcast_ep,
                    weights_block_size_bytes,
                    weights_mcast_num_cores,
                    {},
                    {.noc_x_start = mcast_rect.noc_x_start,
                     .noc_y_start = mcast_rect.noc_y_start,
                     .noc_x_end = mcast_rect.noc_x_end,
                     .noc_y_end = mcast_rect.noc_y_end,
                     .addr = cb_weight_obj.get_write_ptr()},
                    true);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                // statically (using NOC_CMD_STATIC_VC).
                // We should also multicast the flag to destinations
                // num_dests must not include source, since we are NOT really doing a local copy!
                weights_mcast_receiver_sem.set_multicast(
                    noc,
                    mcast_rect.noc_x_start,
                    mcast_rect.noc_y_start,
                    mcast_rect.noc_x_end,
                    mcast_rect.noc_y_end,
                    weights_mcast_num_cores,
                    false);
#endif

                weight_current_block_start_tile_id += weight_next_block_stride_h;
            }

            cb_weight_obj.push_back(weight_block_num_tiles);
        }  // for num_blocks_weight_h
        weight_h_offset += weight_inner_block_stride_h;

        if constexpr (fuse_bias) {
            if (load_bias) {
                cb_bias_obj.reserve_back(bias_ntiles);

                uint32_t bias_write_offset = 0;
                uint32_t bias_block_size_bytes = 0;
                for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles; ++bias_tile) {
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
                // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to zero
                // for the next block
                weights_mcast_sender_sem.wait(weights_mcast_num_dests);
                weights_mcast_sender_sem.set(0);

                // Now we have the block in the CB address, we can mcast to dests!
                // num_dests must not include source, since we are NOT really doing a local copy!
                noc.async_write_multicast(
                    experimental::use<experimental::CB::AddrSelector::WRITE_PTR>(cb_bias_obj),
                    mcast_ep,
                    bias_block_size_bytes,
                    weights_mcast_num_cores,
                    {},
                    {.noc_x_start = mcast_rect.noc_x_start,
                     .noc_y_start = mcast_rect.noc_y_start,
                     .noc_x_end = mcast_rect.noc_x_end,
                     .noc_y_end = mcast_rect.noc_y_end,
                     .addr = cb_bias_obj.get_write_ptr()},
                    true);

                // Note: no need for write barrier, since these two multicasts are done on the same noc id and same vc
                // even though cmd bufs are different Also, this only works because we are setting VCs statically (using
                // NOC_CMD_STATIC_VC).
                // We should also multicast the flag to destinations
                // num_dests must not include source, since we are NOT really doing a local copy!
                weights_mcast_receiver_sem.set_multicast(
                    noc,
                    mcast_rect.noc_x_start,
                    mcast_rect.noc_y_start,
                    mcast_rect.noc_x_end,
                    mcast_rect.noc_y_end,
                    weights_mcast_num_cores,
                    false);
#endif

                cb_bias_obj.push_back(bias_ntiles);
                load_bias = false;
            }
        }
        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx = reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
        }
    }  // out_num_blocks_h
    noc.async_write_barrier();
}
