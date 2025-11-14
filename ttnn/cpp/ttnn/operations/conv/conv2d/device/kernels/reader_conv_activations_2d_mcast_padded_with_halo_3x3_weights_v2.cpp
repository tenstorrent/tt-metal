// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "conv_reader_common.hpp"
#include "noc/noc_parameters.h"
#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

// Multicasts activation data from the local circular buffer to multiple destinations depending on the core role.
template <uint32_t act_mcast_num_cores>
void multicast_data(
    bool is_receiver_core,        // Whether this core is also a receiver of its own multicast (for loopback).
    uint32_t src_l1_addr,         // SRC address in L1
    uint32_t dst_l1_addr,         // DST address in L1
    uint64_t multicast_noc_addr,  // Multicast NOC address
    uint32_t total_bytes)         // Total bytes to send
{
    uint64_t multicast_write_addr = multicast_noc_addr | dst_l1_addr;
    if (is_receiver_core) {
        if constexpr (act_mcast_num_cores) {
            // num_dests will source, since we are copying to a different local CB as well
            noc_async_write_multicast_loopback_src(
                src_l1_addr, multicast_write_addr, total_bytes, act_mcast_num_cores + 1, true);
        } else {
            // In this case sender core is the only reciever in the grid,
            // we can't use the multicast_loopback_src (hang)
            noc_async_write(get_noc_addr(src_l1_addr), get_noc_addr(dst_l1_addr), total_bytes);
        }
    } else {
        // If sender core is not the reciever core as well we can't use the loopback mcast. (hang)
        noc_async_write_multicast(src_l1_addr, multicast_write_addr, total_bytes, act_mcast_num_cores + 1, true);
    }
}

// Multicast activation data from the local circular buffer to multiple destinations (dst_cb in receiver cores).
// This function sends a block of data (the activation block) using NOC multicast commands, it avoids waiting for the
// whole block to be available in the source CB before starting the multicast, instead waits for enough tiles to do one
// multicast of NOC_MAX_BURST_SIZE size. This is because under the hood, the multicast splits the data into chunks of
// NOC_MAX_BURST_SIZE size
// It calls the multicast_data function for each chunk of maximum size NOC_MAX_BURST_SIZE bytes.
// Said function does mcast loopback when the sender core is also a receiver core (it is both in ouput and input grids)
// or mcast when the sender core is not a receiver core (it is only present in the input grid, mcast loopback will hang
// if the core isn't one of receivers) or just local write when it is in both input and output grids but is the only
// receiver core (will hang if mcast loopback is used)
template <
    uint32_t act_mcast_num_dest_cores,
    uint32_t mcast_noc_burst_size,
    uint32_t block_tile_count,
    uint32_t tile_size>
void mcast_block_chunked(bool is_receiver_core, uint32_t src_cb, uint32_t dst_cb, uint64_t multicast_noc_addr) {
    // number of full bursts
    constexpr uint32_t mcast_full_burst_cnt = block_tile_count * tile_size / mcast_noc_burst_size;
    // size of the leftover burst, if 0 means we have no leftover burst
    constexpr uint32_t mcast_leftover_burst_size = block_tile_count * tile_size % mcast_noc_burst_size;
    // number of tiles that we need to wait for to cover the full burst size
    constexpr uint32_t wait_tile_full_cnt = (mcast_noc_burst_size + tile_size - 1) / tile_size;

    // In full burst iterations we wait for a bit more than the full burst size in case where the
    // tile size does not divide the burst size evenly.
    // we need to insure that we don't wait for more tiles than we have in the block
    constexpr uint32_t wait_tile_full_done = std::min(mcast_full_burst_cnt * wait_tile_full_cnt, block_tile_count);

    // optimization to avoid unnecessary branching in the loop
    constexpr bool no_need_partial_wait_tile = mcast_full_burst_cnt * wait_tile_full_cnt <= block_tile_count;

    // number of times we need to increase the wait_tile_curr for the full burst iterations
    constexpr uint32_t wait_tile_full_iter_cnt = (wait_tile_full_done / wait_tile_full_cnt) - 1;

    uint32_t src_l1_addr = get_read_ptr(src_cb);
    uint32_t dst_l1_addr = get_write_ptr(dst_cb);

    constexpr uint32_t wait_tile_start_cnt = std::min(block_tile_count, wait_tile_full_cnt);
    uint32_t wait_tile_curr = wait_tile_start_cnt;
    for (uint32_t i = 0; i < mcast_full_burst_cnt; i++) {
        cb_wait_front(src_cb, wait_tile_curr);
        multicast_data<act_mcast_num_dest_cores>(
            is_receiver_core, src_l1_addr, dst_l1_addr, multicast_noc_addr, mcast_noc_burst_size);
        src_l1_addr += mcast_noc_burst_size;
        dst_l1_addr += mcast_noc_burst_size;

        if constexpr (no_need_partial_wait_tile) {
            wait_tile_curr += wait_tile_full_cnt;
        } else {
            // we shouldn't wait for more than the number of tiles in the block
            if (i < wait_tile_full_iter_cnt) {
                wait_tile_curr += wait_tile_full_cnt;
            } else {
                wait_tile_curr = block_tile_count;
            }
        }
    }
    if constexpr (mcast_leftover_burst_size > 0) {
        cb_wait_front(src_cb, block_tile_count);
        multicast_data<act_mcast_num_dest_cores>(
            is_receiver_core, src_l1_addr, dst_l1_addr, multicast_noc_addr, mcast_leftover_burst_size);
    }

    // In case we only do local l1 writes, we need to wait for the barrier to complete
    if constexpr (act_mcast_num_dest_cores == 0) {
        if (is_receiver_core) {
            noc_async_write_barrier();
        }
    }
}

constexpr uint32_t DILATION_W = get_compile_time_arg_val(1);
void kernel_main() {
    constexpr uint32_t dilation_h = get_compile_time_arg_val(0);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t window_outer = get_compile_time_arg_val(4);
    constexpr uint32_t act_block_num_tiles_read = get_compile_time_arg_val(6);
    constexpr uint32_t weight_size_h = get_compile_time_arg_val(7);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(8);
    constexpr uint32_t padded_conv_act_size_w = get_compile_time_arg_val(9);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(11);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t act_w_num_outer = get_compile_time_arg_val(13);
    constexpr uint32_t act_mcast_num_dests = get_compile_time_arg_val(14);
    constexpr uint32_t act_mcast_num_cores = get_compile_time_arg_val(15);
    const uint32_t act_mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    const uint32_t act_mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(17));
    constexpr uint32_t act_mcast_tile_size_bytes = get_compile_time_arg_val(18);
    constexpr bool transpose_mcast = get_compile_time_arg_val(19) == 1;
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t cb_id_act = get_compile_time_arg_val(21);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(22);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(23);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t cb_id_act_row_major_bfloat16 = get_compile_time_arg_val(25);
    constexpr uint32_t cb_l1_array = get_compile_time_arg_val(26);
    constexpr bool split_reader_enabled = get_compile_time_arg_val(27);

    constexpr bool split_reader_cb_shared = get_compile_time_arg_val(32) == 1;
    volatile tt_l1_ptr uint32_t* act_split_reader_reserve_done_semaphore_addr_ptr = nullptr;
    volatile tt_l1_ptr uint32_t* act_split_reader_write_done_semaphore_addr_ptr = nullptr;
    if constexpr (split_reader_cb_shared) {  // When the split reader CB is shared, both readers write to the same
                                             // circular
                                             // buffer.
        // Synchronization is required: the main reader signals when CB space is reserved,
        // and the second reader signals when it has finished writing its portion.
        const uint32_t act_split_reader_reserve_done_semaphore_addr = get_semaphore(get_compile_time_arg_val(33));
        const uint32_t act_split_reader_write_done_semaphore_addr = get_semaphore(get_compile_time_arg_val(34));

        act_split_reader_reserve_done_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_reserve_done_semaphore_addr);
        act_split_reader_write_done_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_split_reader_write_done_semaphore_addr);
    }

    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_row_major_bfloat16>();
    }

    uint32_t i = 0;
    uint32_t act_mcast_dest_noc_start_x = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_start_y = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_end_x = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_dest_noc_end_y = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_id = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    const bool is_receiver_core = get_arg_val<uint32_t>(i++) > 0;
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;
    uint32_t dram_config_reader_index = get_arg_val<uint32_t>(i++);

    tt_l1_ptr uint32_t* act_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(i));

    load_config_tensor_if_in_dram<29, 30, 31, cb_reader_indices>(dram_config_reader_index);

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // L1 array
    volatile tt_l1_ptr uint32_t* l1_array = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_l1_array));
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_valid_addr_ptr = &l1_array[0];
    act_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers
    uint32_t act_mcast_sender_semaphore_valid_addr = reinterpret_cast<uint32_t>(&l1_array[0]);
    // Set up remote VALID value
    volatile tt_l1_ptr uint32_t* act_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_receiver_semaphore_addr);
    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* act_mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_mcast_sender_semaphore_addr);

    uint64_t act_multicast_noc_addr = get_noc_multicast_addr(
        act_mcast_dest_noc_start_x, act_mcast_dest_noc_start_y, act_mcast_dest_noc_end_x, act_mcast_dest_noc_end_y, 0);

    uint64_t act_mcast_receiver_semaphore_noc_addr = act_multicast_noc_addr | act_mcast_receiver_semaphore_addr;

    // TODO: need to make the read coalescing optimization cleaner
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);

    // Fully create act matrix and tilize it before mcast
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

    if constexpr (!split_reader_cb_shared) {
        noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
    }

    constexpr uint32_t window_outer_offset = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_h_bytes = padded_conv_act_size_w * conv_act_c_read_bytes * dilation_h;
    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    constexpr bool sliced_inner_dim = window_outer > 1;

    // Reset reader_idx to finish act_block_h_datums
    uint32_t reader_idx = 0;
    uint32_t start_reader_idx = 0;
    for (uint32_t nbh = 0; nbh < act_num_blocks_h; nbh++) {
        uint32_t reader_offset = act_l1_read_addr;
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            reader_idx = start_reader_idx;
            cb_reserve_back(cb_id_act_row_major_bfloat16, act_block_num_tiles_read);
            if (is_sender_core) {
                uint32_t l1_write_addr_act = get_write_ptr(cb_id_act_row_major_bfloat16);
                if constexpr (split_reader_cb_shared) {
                    signal_reserve_done(act_split_reader_reserve_done_semaphore_addr_ptr);
                    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
                }
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
                if constexpr (split_reader_cb_shared) {
                    wait_write_done(act_split_reader_write_done_semaphore_addr_ptr);
                }
            }
            cb_push_back(cb_id_act_row_major_bfloat16, act_block_num_tiles_read);

#ifndef SKIP_MCAST
            // Round robin self-mcast and receive tilized act matrix in cb_id_act
            // Compute should function like regular mm
            for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
                cb_reserve_back(cb_id_act, act_block_num_tiles);
                if (act_w_outer_i == act_mcast_sender_id) {
                    // MCAST SENDER: send entire tilized input to other cores in column
                    // wait until all act mcast destinations have atomically incremented the act semaphore_addr
                    // (i.e. its value should be act_mcast_num_dests), then reset the semaphore_addr value back to
                    // zero for the next block
                    noc_semaphore_wait(
                        act_mcast_sender_semaphore_addr_ptr, act_mcast_num_dests + (is_receiver_core ? 0 : 1));
                    noc_semaphore_set(act_mcast_sender_semaphore_addr_ptr, 0);

                    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                    mcast_block_chunked<
                        act_mcast_num_cores,
                        NOC_MAX_BURST_SIZE,
                        act_block_num_tiles,
                        act_mcast_tile_size_bytes>(
                        is_receiver_core, tilized_in0_cb_id, cb_id_act, act_multicast_noc_addr);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                    // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                    // statically (using NOC_CMD_STATIC_VC).
#ifdef ARCH_BLACKHOLE
                    // On Blackhole the flush is needed because the commands go into separate cmd buffer FIFOs and
                    // may not be sent in order they are issued
                    noc_async_writes_flushed();
#endif

                    if (is_receiver_core) {
                        // We should also multicast VALID flag to destinations for receiver semaphore
                        if constexpr (act_mcast_num_cores) {
                            noc_semaphore_set_multicast_loopback_src(
                                act_mcast_sender_semaphore_valid_addr,
                                act_mcast_receiver_semaphore_noc_addr,
                                act_mcast_num_cores + 1);
                            noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
                        }
                    } else {
                        noc_semaphore_set_multicast(
                            act_mcast_sender_semaphore_valid_addr,
                            act_mcast_receiver_semaphore_noc_addr,
                            act_mcast_num_cores + 1);
                    }
                } else if (is_receiver_core) {
                    // MCAST RECEIVER: receive entire tilized input from sender core
                    // Set act semaphore value to INVALID
                    noc_semaphore_set(act_mcast_receiver_semaphore_addr_ptr, INVALID);

                    // Atomic increment source core counter
                    uint64_t act_mcast_sender_semaphore_noc_addr;
                    if constexpr (transpose_mcast) {
                        act_mcast_sender_semaphore_noc_addr = get_noc_addr(
                            act_mcast_sender_noc_x,
                            act_mcast_sender_noc_y[act_w_outer_i],
                            act_mcast_sender_semaphore_addr);
                    } else {
                        act_mcast_sender_semaphore_noc_addr = get_noc_addr(
                            act_mcast_sender_noc_y[act_w_outer_i],
                            act_mcast_sender_noc_x,
                            act_mcast_sender_semaphore_addr);
                    }
                    noc_semaphore_inc(act_mcast_sender_semaphore_noc_addr, 1);

                    // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)
                    noc_semaphore_wait(act_mcast_receiver_semaphore_addr_ptr, VALID);
                }
                cb_push_back(cb_id_act, act_block_num_tiles);
            }  // act_w_num_outer

            cb_pop_front(tilized_in0_cb_id, act_block_num_tiles);
#endif
        }
        start_reader_idx = reader_idx;
        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx += (static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1);
        }
    }

    noc_async_write_barrier();
}
