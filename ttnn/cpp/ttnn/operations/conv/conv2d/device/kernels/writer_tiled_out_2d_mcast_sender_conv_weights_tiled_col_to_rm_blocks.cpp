// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "conv_reader_common.hpp"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
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
    const uint32_t act_split_reader_reserve_done_semaphore_addr =
        (split_reader_cb_shared) ? get_semaphore(get_compile_time_arg_val(32)) : 0;
    const uint32_t act_split_reader_write_done_semaphore_addr =
        (split_reader_cb_shared) ? get_semaphore(get_compile_time_arg_val(33)) : 0;
    constexpr uint32_t act_write_offset = get_compile_time_arg_val(34);
    constexpr uint32_t act_write_offset_last = get_compile_time_arg_val(35);

    volatile tt_l1_ptr uint32_t* act_split_reader_reserve_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_reserve_done_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_split_reader_write_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_write_done_semaphore_addr);

    const uint32_t split_reader_cb_write_addr =
        (split_reader_cb_shared) ? get_write_ptr(cb_id_act_second_reader) + act_write_offset : 0;
    // In case of double buffering the split reader can write to two different addresses
    const uint32_t split_reader_cb_write_addr_last =
        (split_reader_cb_shared) ? get_write_ptr(cb_id_act_second_reader) + act_write_offset_last : 0;
    const uint32_t split_reader_cb_write_addr_sum = split_reader_cb_write_addr + split_reader_cb_write_addr_last;

    constexpr bool transpose_mcast = get_compile_time_arg_val(36) == 1;
    constexpr uint32_t act_mcast_reserve_done_semaphore_id = get_compile_time_arg_val(37);
    const uint32_t act_mcast_reserve_done_semaphore_addr = get_semaphore(act_mcast_reserve_done_semaphore_id);
    // DPRINT << "ACT_MCAST_RESERVE_DONE_SEMaphore ID: " << act_mcast_reserve_done_semaphore_id << ENDL();
    constexpr uint32_t act_mcast_receiver_second_semaphore_id = get_compile_time_arg_val(38);
    const uint32_t act_mcast_receiver_semaphore_addr = get_semaphore(act_mcast_receiver_second_semaphore_id);
    constexpr uint32_t act_cb_id = get_compile_time_arg_val(39);
    constexpr uint32_t act_tilized_cb = get_compile_time_arg_val(40);
    constexpr uint32_t cb_l1_array = get_compile_time_arg_val(41);
    // act_mcast_write_offset represents the offset to the second half of the ACT CB (which will be written to by the
    // second reader)
    constexpr uint32_t act_mcast_write_offset = get_compile_time_arg_val(42);
    // act_mcast_write_offset_last represents the offset to the second half of the second block of ACT CB (which will be
    // written to by the main reader, when double buffering is enabled)
    constexpr uint32_t act_mcast_write_offset_last = get_compile_time_arg_val(43);
    constexpr uint32_t act_mcast_num_cores = get_compile_time_arg_val(44);
    constexpr uint32_t act_mcast_sender_size_bytes = get_compile_time_arg_val(45);
    constexpr bool skip_mcast = get_compile_time_arg_val(46) == 1;

    volatile tt_l1_ptr uint32_t* act_mcast_reserve_done_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_reserve_done_semaphore_addr);
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr);

    // L1 array
    volatile tt_l1_ptr uint32_t* l1_array = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = &l1_array[0];
    act_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers
    uint32_t act_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(&l1_array[0]);

    constexpr uint32_t act_tilized_offset =
        act_mcast_write_offset;  // This is the offset to the first half of the ACT_TILIZED CB (which will be read by
                                 // the second reader)
    const uint32_t tilized_act_start_address = get_read_ptr(act_tilized_cb);
    const uint32_t base_act_address = get_write_ptr(act_cb_id);
    constexpr uint64_t act_mcast_write_offset_sum = act_mcast_write_offset + act_mcast_write_offset_last;

    constexpr uint32_t ct_arg_idx = 47;
    constexpr auto s_weight_args = TensorAccessorArgs<ct_arg_idx>();
    constexpr auto s_bias_args = TensorAccessorArgs<s_weight_args.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t weight_addr_dram_base = get_arg_val<uint32_t>(i++);
    // Bias arg. Unused if bias fusion is not enabled.
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);
    const uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i++);
    const uint32_t bias_tile_offset = get_arg_val<uint32_t>(i++);

    // mcast args
    const uint32_t weights_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_dests = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_num_cores = get_arg_val<uint32_t>(i++);
    const uint32_t weights_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const uint32_t weights_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(i++));
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;
    const bool is_receiver_core = get_arg_val<uint32_t>(i++) > 0;
    const bool skip_work = get_arg_val<uint32_t>(i++) > 0;
    // DPRINT << "skip_work: " << (skip_work ? "true" : "false") << ENDL();
    uint32_t act_mcast_dest_noc_start_x = 0;
    uint32_t act_mcast_dest_noc_start_y = 0;
    uint32_t act_mcast_dest_noc_end_x = 0;
    uint32_t act_mcast_dest_noc_end_y = 0;
    uint32_t act_mcast_sender_id = 0;
    if constexpr (split_reader_enabled) {
        act_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
        act_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
        act_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
        act_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
        act_mcast_sender_id = get_arg_val<uint32_t>(i++);
    }
    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x, act_mcast_dest_noc_start_y, act_mcast_dest_noc_end_x, act_mcast_dest_noc_end_y, 0);
    uint64_t act_multicast_receiver_semaphore_noc_addr = act_multicast_noc_addr | act_mcast_receiver_semaphore_addr;
    if (skip_work && !split_reader_enabled) {
        return;
    }

    // Split reader configuration
    if constexpr (split_reader_enabled) {
#ifdef CONFIG_TENSOR_IN_DRAM
        cb_wait_front(cb_reader_indices, 1);
#endif
        if constexpr (needs_act_block_zero_out) {
            zero_out_tiles<cb_id_act_second_reader>();
        }
    }

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        split_reader_enabled ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices))
                             : nullptr;
    // Initial setup for second reader (starting from second reader's data)
    uint32_t start_reader_idx = split_reader_enabled ? (uint32_t)(packed_reader_indices_ptr[0] & 0xffff) + 1 : 0;
    uint32_t reader_idx = start_reader_idx;

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);
    constexpr uint32_t window_outer_offset = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;

    const uint32_t act_l1_read_addr = split_reader_enabled ? get_read_ptr(cb_id_sharded_act) : 0;

#ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    *(weights_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* weights_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_sender_semaphore_addr);

    const uint64_t weights_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        weights_mcast_dest_noc_start_x,
        weights_mcast_dest_noc_start_y,
        weights_mcast_dest_noc_end_x,
        weights_mcast_dest_noc_end_y,
        weights_mcast_receiver_semaphore_addr);
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
    uint64_t act_write_offset_current = act_mcast_write_offset;
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        for (uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            if constexpr (split_reader_enabled) {
                reader_offset = act_l1_read_addr;
            }
            for (uint32_t height_block_index = 0; height_block_index < num_blocks_weight_h; height_block_index++) {
                if constexpr (split_reader_enabled) {
                    reader_idx = start_reader_idx;
                    if constexpr (!split_reader_cb_shared) {
                        cb_reserve_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
                    }
                    if (is_sender_core) {
                        if constexpr (split_reader_cb_shared) {
                            wait_reserve_done(act_split_reader_reserve_done_semaphore_addr_ptr);
                            prev_addr = l1_write_addr_act;
                        } else {
                            l1_write_addr_act = get_write_ptr(cb_id_act_second_reader);
                        }
                        noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                        {
                            DeviceZoneScopedN("IMG2COL");
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
                                packed_reader_indices_ptr,
                                reader_offset,
                                l1_write_addr_act,
                                reader_idx,
                                act_l1_read_addr,
                                stride_h_bytes);
                        }
                        if constexpr (split_reader_cb_shared) {
                            // in case of shared cb we update the write address (it will remain the same if double
                            // buffering is not enabled)
                            l1_write_addr_act = split_reader_cb_write_addr_sum - prev_addr;
                            signal_write_done(act_split_reader_write_done_semaphore_addr_ptr);
                        }
                    }
                    if constexpr (!split_reader_cb_shared) {
                        cb_push_back(cb_id_act_second_reader, act_block_num_tiles_split_last);
                    } else {
                        act_write_offset_current = act_mcast_write_offset_sum - act_write_offset_current;
                    }
                }
                // Compute height block offset once per outer loop iteration
                const uint32_t height_block_offset = height_block_index * height_stride_factor;
                for (uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer;
                     weight_tile_h_outer_i++) {
                    if (!skip_work) {
                        cb_reserve_back(cb_id_weight, weight_block_num_tiles);
                        uint32_t weight_write_l1_addr = get_write_ptr(cb_id_weight);

                        const uint32_t outer_block_offset = weight_tile_h_outer_i * tiles_per_full_block;
                        uint32_t tile_id = weight_start_tile_id + height_block_offset + outer_block_offset;
                        // mcast args
                        uint32_t weights_start_address = weight_write_l1_addr;
                        {
                            DeviceZoneScopedN("READ WEIGHTS");
                            for (uint32_t block_weight_h = 0; block_weight_h < weight_block_height_ntiles;
                                 block_weight_h++) {
                                uint32_t weight_tile_id = tile_id;

                                for (uint32_t weight_tile_w_i = 0; weight_tile_w_i < weight_block_width_ntiles;
                                     ++weight_tile_w_i) {
                                    noc_async_read_tile(weight_tile_id++, s_weight, weight_write_l1_addr);
                                    weight_write_l1_addr += weight_tile_nbytes;
                                }
                                tile_id += weight_stride_h;
                            }
                            noc_async_read_barrier();
                        }
#ifndef SKIP_MCAST
                        // wait until all weights mcast destinations have atomically incremented the weights
                        // semaphore_addr (i.e. its value should be weights_mcast_num_dests), then reset the
                        // semaphore_addr value back to zero for the next block
                        noc_semaphore_wait(weights_mcast_sender_semaphore_addr_ptr, weights_mcast_num_dests);
                        noc_semaphore_set(weights_mcast_sender_semaphore_addr_ptr, 0);

                        // Now we have the block in the CB address, we can mcast to dests!
                        uint64_t weights_multicast_data_addr = get_noc_multicast_addr(
                            weights_mcast_dest_noc_start_x,
                            weights_mcast_dest_noc_start_y,
                            weights_mcast_dest_noc_end_x,
                            weights_mcast_dest_noc_end_y,
                            weights_start_address);
                        // num_dests must not include source, since we are NOT really doing a local copy!
                        {
                            DeviceZoneScopedN("WEIGHTS MCAST SEND");
                            noc_async_write_multicast(
                                weights_start_address,
                                weights_multicast_data_addr,
                                weights_block_size_bytes,
                                weights_mcast_num_cores,
                                true);

                            // Note: no need for write barrier, since these two multicasts are done on the same noc id
                            // and same vc even though cmd bufs are different Also, this only works because we are
                            // setting VCs statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                            // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs
                            // and may not be sent in order they are issued
                            noc_async_writes_flushed();
#endif
                            // We should also multicast the flag to destinations
                            // num_dests must not include source, since we are NOT really doing a local copy!
                            noc_semaphore_set_multicast(
                                weights_mcast_receiver_semaphore_addr,
                                weights_mcast_receiver_semaphore_noc_addr,
                                weights_mcast_num_cores);
                        }
#endif
                        cb_push_back(cb_id_weight, weight_block_num_tiles);
                    }
                    if constexpr (split_reader_enabled && !skip_mcast) {
                        if (weight_tile_h_outer_i == act_mcast_sender_id) {
                            uint64_t act_address = base_act_address + act_write_offset_current;
                            uint64_t act_multicast_data_addr = act_multicast_noc_addr | act_address;

                            // DPRINT << "WAIT RESERVE"
                            // << reinterpret_cast<uint32_t>(act_mcast_reserve_done_semaphore_addr_ptr) << ENDL();
                            wait_reserve_done(act_mcast_reserve_done_semaphore_addr_ptr);
                            // DPRINT << "WAIT RESERVE DONE" << ENDL();
                            cb_wait_front(act_tilized_cb, act_block_num_tiles_split_last);
                            {
                                DeviceZoneScopedN("MCAST SEND");
                                if (is_receiver_core) {
                                    if constexpr (act_mcast_num_cores) {
                                        // num_dests will source, since we are copying to a different local CB as well
                                        noc_async_write_multicast_loopback_src(
                                            tilized_act_start_address,
                                            act_multicast_data_addr,
                                            act_mcast_sender_size_bytes,
                                            act_mcast_num_cores + 1,
                                            true);
                                    } else {
                                        // In this case sender core is the only reciever in the grid,
                                        // we can't use the multicast_loopback_src (hang)
                                        noc_async_write(
                                            get_noc_addr(tilized_act_start_address),
                                            get_noc_addr(act_address),
                                            act_mcast_sender_size_bytes);
                                        noc_async_write_barrier();
                                    }
                                } else {
                                    // If sender core is not the reciever core as well we can't use the loopback mcast.
                                    // (hang)
                                    noc_async_write_multicast(
                                        tilized_act_start_address,
                                        act_multicast_data_addr,
                                        act_mcast_sender_size_bytes,
                                        act_mcast_num_cores + 1,
                                        true);
                                }

                                // Note: no need for write barrier, since these two multicasts are done on the same noc
                                // id and same vc even though cmd bufs are different Also, this only works because we
                                // are setting VCs statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                                // On Blackhole the flush is needed because the commands go into separate cmd buffer
                                // FIFOs and may not be sent in order they are issued
                                noc_async_writes_flushed();
#endif

                                // DPRINT << "SET MULTICAST LOOPBACK SRC" << ENDL();
                                if (is_receiver_core) {
                                    // We should also multicast VALID flag to destinations for receiver semaphore
                                    if constexpr (act_mcast_num_cores) {
                                        noc_semaphore_set_multicast_loopback_src(
                                            act_mcast_sender_semaphore_valid_addr,
                                            act_multicast_receiver_semaphore_noc_addr,
                                            act_mcast_num_cores + 1);
                                    }
                                } else {
                                    noc_semaphore_set_multicast(
                                        act_mcast_sender_semaphore_valid_addr,
                                        act_multicast_receiver_semaphore_noc_addr,
                                        act_mcast_num_cores + 1);
                                }
                            }
                        }
                        act_write_offset_current = act_mcast_write_offset_sum - act_write_offset_current;
                    }

                }  // for weight_block_height_num_outer
                if constexpr (split_reader_enabled && !skip_mcast) {
                    cb_pop_front(act_tilized_cb, act_block_num_tiles_split_last);
                }
            }
            if constexpr (split_reader_enabled) {
                // Update reader index for next iteration (split reader increment)
                start_reader_idx =
                    reader_idx + static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1;
                if (skip_work) {
                    continue;
                }
            }
            if constexpr (fuse_bias) {
                if (load_bias) {
                    cb_reserve_back(bias_cb_id, bias_ntiles);
                    uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);

                    // mcast args
                    uint32_t bias_start_address = bias_l1_addr;
                    uint32_t bias_block_size_bytes = 0;
                    for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles;
                         ++bias_tile) {
                        noc_async_read_tile(bias_tile, s_bias, bias_l1_addr);
                        bias_l1_addr += bias_pagesize;
                        bias_block_size_bytes += bias_pagesize;
                    }
                    noc_async_read_barrier();

// MCAST BIAS (shares some mcast args with weights)
#ifndef SKIP_MCAST
                    // wait until all weights mcast destinations have atomically incremented the weights semaphore_addr
                    // (i.e. its value should be weights_mcast_num_dests), then reset the semaphore_addr value back to
                    // zero for the next block
                    noc_semaphore_wait(weights_mcast_sender_semaphore_addr_ptr, weights_mcast_num_dests);
                    noc_semaphore_set(weights_mcast_sender_semaphore_addr_ptr, 0);

                    // Now we have the block in the CB address, we can mcast to dests!
                    uint64_t bias_multicast_data_addr = get_noc_multicast_addr(
                        weights_mcast_dest_noc_start_x,
                        weights_mcast_dest_noc_start_y,
                        weights_mcast_dest_noc_end_x,
                        weights_mcast_dest_noc_end_y,
                        bias_start_address);
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_async_write_multicast(
                        bias_start_address,
                        bias_multicast_data_addr,
                        bias_block_size_bytes,
                        weights_mcast_num_cores,
                        true);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and same
                    // vc even though cmd bufs are different Also, this only works because we are setting VCs statically
                    // (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                    // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and may
                    // not be sent in order they are issued
                    noc_async_writes_flushed();
#endif
                    // We should also multicast the flag to destinations
                    // num_dests must not include source, since we are NOT really doing a local copy!
                    noc_semaphore_set_multicast(
                        weights_mcast_receiver_semaphore_addr,
                        weights_mcast_receiver_semaphore_noc_addr,
                        weights_mcast_num_cores);
#endif

                    cb_push_back(bias_cb_id, bias_ntiles);
                    load_bias = false;
                }
            }

        }  // out_num_blocks_h

        // Increment weight start tile id for next block in width dim
        weight_start_tile_id += weight_next_block_stride_w;
    }  // out_num_blocks_w

    noc_async_write_barrier();
}
