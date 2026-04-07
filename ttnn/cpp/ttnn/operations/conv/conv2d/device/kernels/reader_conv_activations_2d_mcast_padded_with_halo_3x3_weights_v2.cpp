// SPDX-FileCopyrightText: (c) 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "conv_reader_common.hpp"
#include "noc/noc_parameters.h"
#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

struct McastRect {
    uint32_t noc_x_start, noc_y_start, noc_x_end, noc_y_end;
};

using McastDst = experimental::noc_traits_t<experimental::MulticastEndpoint>::dst_args_mcast_type;

// Multicasts activation data from src_cb to dst_cb across cores in the multicast rectangle.
// Three cases depending on the sender's role and number of multicast destinations:
//   is_receiver_core && act_mcast_num_cores > 0:  mcast with INCLUDE_SRC loopback
//   is_receiver_core && act_mcast_num_cores == 0: local self-write (mcast loopback hangs with 0 destinations)
//   !is_receiver_core:                            standard mcast EXCLUDE_SRC (even when act_mcast_num_cores == 0,
//                                                 because the sender still needs to send to the output core)
template <uint32_t act_mcast_num_cores>
void multicast_data(
    experimental::Noc& noc,
    experimental::MulticastEndpoint& mcast_ep,
    bool is_receiver_core,
    experimental::CB& src_cb,
    uint32_t src_offset,
    McastDst& dst,
    uint32_t total_bytes) {
    auto src = experimental::use<experimental::CircularBuffer::AddrSelector::READ_PTR>(src_cb);
    if (is_receiver_core) {
        if constexpr (act_mcast_num_cores > 0) {
            noc.async_write_multicast<experimental::Noc::McastMode::INCLUDE_SRC>(
                src, mcast_ep, total_bytes, act_mcast_num_cores + 1, {.offset_bytes = src_offset}, dst, true);
        } else {
            // Sender is the only receiver — can't use multicast loopback (hangs with 0 destinations)
            noc.async_write(
                src,
                experimental::UnicastEndpoint{},
                total_bytes,
                {.offset_bytes = src_offset},
                {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = dst.addr});
        }
    } else {
        noc.async_write_multicast(
            src, mcast_ep, total_bytes, act_mcast_num_cores + 1, {.offset_bytes = src_offset}, dst, true);
    }
}

// Multicast activation data from the local circular buffer to multiple destinations (dst_cb in receiver cores).
// This function sends a block of data (the activation block) using NOC multicast commands, it avoids waiting for the
// whole block to be available in the source CB before starting the multicast, instead waits for enough tiles to do one
// multicast of NOC_MAX_BURST_SIZE size. This is because under the hood, the multicast splits the data into chunks of
// NOC_MAX_BURST_SIZE size
// It calls the multicast_data function for each chunk of maximum size NOC_MAX_BURST_SIZE bytes.
// Said function does mcast loopback when the sender core is also a receiver core (it is both in output and input grids)
// or mcast when the sender core is not a receiver core (it is only present in the input grid, mcast loopback will hang
// if the core isn't one of receivers) or just local write when it is in both input and output grids but is the only
// receiver core (will hang if mcast loopback is used)
template <
    uint32_t act_mcast_num_dest_cores,
    uint32_t mcast_noc_burst_size,
    uint32_t block_tile_count,
    uint32_t tile_size>
void mcast_block_chunked(
    experimental::Noc& noc,
    experimental::MulticastEndpoint& mcast_ep,
    experimental::CB& src_cb_obj,
    bool is_receiver_core,
    experimental::CB& dst_cb_obj,
    const McastRect& rect) {
    // Build mcast dst once; only .addr is updated per burst
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

    uint32_t src_offset = 0;
    McastDst dst = {
        .noc_x_start = rect.noc_x_start,
        .noc_y_start = rect.noc_y_start,
        .noc_x_end = rect.noc_x_end,
        .noc_y_end = rect.noc_y_end,
        .addr = dst_cb_obj.get_write_ptr()};

    constexpr uint32_t wait_tile_start_cnt = std::min(block_tile_count, wait_tile_full_cnt);
    uint32_t wait_tile_curr = wait_tile_start_cnt;
    for (uint32_t i = 0; i < mcast_full_burst_cnt; i++) {
        src_cb_obj.wait_front(wait_tile_curr);
        multicast_data<act_mcast_num_dest_cores>(
            noc, mcast_ep, is_receiver_core, src_cb_obj, src_offset, dst, mcast_noc_burst_size);
        src_offset += mcast_noc_burst_size;
        dst.addr += mcast_noc_burst_size;

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
        src_cb_obj.wait_front(block_tile_count);
        multicast_data<act_mcast_num_dest_cores>(
            noc, mcast_ep, is_receiver_core, src_cb_obj, src_offset, dst, mcast_leftover_burst_size);
    }

    // In case we only do local l1 writes, we need to wait for the barrier to complete
    if constexpr (act_mcast_num_dest_cores == 0) {
        if (is_receiver_core) {
            noc.async_write_barrier();
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

    constexpr bool split_reader_cb_shared = get_compile_time_arg_val(33) == 1;

    // Experimental API objects
    experimental::Noc noc;
    experimental::Semaphore<> act_mcast_sender_sem(get_compile_time_arg_val(16));
    experimental::Semaphore<> act_mcast_receiver_sem(get_compile_time_arg_val(17));
    experimental::MulticastEndpoint mcast_ep;
    experimental::CB cb_act_obj(cb_id_act);
    experimental::CB cb_act_rm_obj(cb_id_act_row_major_bfloat16);
    experimental::CB cb_tilized_in0_obj(tilized_in0_cb_id);
    experimental::CB cb_reader_indices_obj(cb_reader_indices);

    experimental::Semaphore<> reserve_done_sem(0);
    experimental::Semaphore<> write_done_sem(0);
    if constexpr (split_reader_cb_shared) {
        // When the split reader CB is shared, both readers write to the same circular buffer.
        // Synchronization is required: the main reader signals when CB space is reserved,
        // and the second reader signals when it has finished writing its portion.
        reserve_done_sem = experimental::Semaphore<>(get_compile_time_arg_val(34));
        write_done_sem = experimental::Semaphore<>(get_compile_time_arg_val(35));
    }

    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_row_major_bfloat16>(noc, cb_act_rm_obj);
    }

    uint32_t i = 0;
    const McastRect act_mcast_rect = {
        get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++), get_arg_val<uint32_t>(i++)};
    uint32_t act_mcast_sender_id = get_arg_val<uint32_t>(i++);
    uint32_t act_mcast_sender_noc_x = get_arg_val<uint32_t>(i++);
    const bool is_receiver_core = get_arg_val<uint32_t>(i++) > 0;
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;
    uint32_t dram_config_reader_index = get_arg_val<uint32_t>(i++);

    tt_l1_ptr uint32_t* act_mcast_sender_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(i));

    load_config_tensor_if_in_dram<29, 30, 31, cb_reader_indices>(noc, cb_reader_indices_obj, dram_config_reader_index);

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    // Set up receiver semaphore VALID value, to be mcasted to destinations after the data has been mcasted
    act_mcast_receiver_sem.set(VALID);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast

    // TODO: need to make the read coalescing optimization cleaner
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);

    // Fully create act matrix and tilize it before mcast
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

    if constexpr (!split_reader_cb_shared) {
        experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);
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
            cb_act_rm_obj.reserve_back(act_block_num_tiles_read);
            if (is_sender_core) {
                uint32_t l1_write_addr_act = cb_act_rm_obj.get_write_ptr();
                if constexpr (split_reader_cb_shared) {
                    reserve_done_sem.set(VALID);
                    experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);
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
                    noc,
                    packed_reader_indices_ptr,
                    reader_offset,
                    l1_write_addr_act,
                    reader_idx,
                    act_l1_read_addr,
                    stride_h_bytes);
                if constexpr (split_reader_cb_shared) {
                    write_done_sem.wait(VALID);
                    write_done_sem.set(INVALID);
                }
            }
            cb_act_rm_obj.push_back(act_block_num_tiles_read);

#ifndef SKIP_MCAST
            // Round robin self-mcast and receive tilized act matrix in cb_id_act
            // Compute should function like regular mm
            for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
                cb_act_obj.reserve_back(act_block_num_tiles);
                if (act_w_outer_i == act_mcast_sender_id) {
                    // MCAST SENDER: send entire tilized input to other cores in column
                    // wait until all act mcast destinations have atomically incremented the act semaphore_addr
                    // (i.e. its value should be act_mcast_num_dests), then reset the semaphore_addr value back to
                    // zero for the next block
                    act_mcast_sender_sem.wait(act_mcast_num_dests + (is_receiver_core ? 0 : 1));
                    act_mcast_sender_sem.set(0);

                    act_mcast_receiver_sem.set(INVALID);

                    mcast_block_chunked<
                        act_mcast_num_cores,
                        NOC_MAX_BURST_SIZE,
                        act_block_num_tiles,
                        act_mcast_tile_size_bytes>(
                        noc, mcast_ep, cb_tilized_in0_obj, is_receiver_core, cb_act_obj, act_mcast_rect);

                    // Note: no need for write barrier, since these two multicasts are done on the same noc id and
                    // same vc even though cmd bufs are different Also, this only works because we are setting VCs
                    // statically (using NOC_CMD_STATIC_VC).

                    if (is_receiver_core) {
                        // We should also multicast VALID flag to destinations for receiver semaphore
                        if constexpr (act_mcast_num_cores) {
                            act_mcast_receiver_sem.set(VALID);
                            act_mcast_receiver_sem.set_multicast<experimental::Noc::McastMode::INCLUDE_SRC>(
                                noc,
                                act_mcast_rect.noc_x_start,
                                act_mcast_rect.noc_y_start,
                                act_mcast_rect.noc_x_end,
                                act_mcast_rect.noc_y_end,
                                act_mcast_num_cores + 1);
                            // Use write barrier instead of wait(VALID) since set(VALID) above
                            // made the local semaphore immediately VALID. The write barrier
                            // ensures all prior multicasts (data + semaphore) are delivered.
                            noc.async_write_barrier();
                        }
                    } else {
                        act_mcast_receiver_sem.set(VALID);
                        act_mcast_receiver_sem.set_multicast(
                            noc,
                            act_mcast_rect.noc_x_start,
                            act_mcast_rect.noc_y_start,
                            act_mcast_rect.noc_x_end,
                            act_mcast_rect.noc_y_end,
                            act_mcast_num_cores + 1);
                    }
                } else if (is_receiver_core) {
                    // MCAST RECEIVER: receive entire tilized input from sender core
                    // Set act semaphore value to INVALID
                    act_mcast_receiver_sem.set(INVALID);

                    // Atomic increment source core counter
                    if constexpr (transpose_mcast) {
                        act_mcast_sender_sem.up(noc, act_mcast_sender_noc_x, act_mcast_sender_noc_y[act_w_outer_i], 1);
                    } else {
                        act_mcast_sender_sem.up(noc, act_mcast_sender_noc_y[act_w_outer_i], act_mcast_sender_noc_x, 1);
                    }

                    // wait on act semaphore value to become VALID (set by mcast sender after it multicasts data)
                    act_mcast_receiver_sem.wait(VALID);
                }
                cb_act_obj.push_back(act_block_num_tiles);
            }  // act_w_num_outer

            cb_tilized_in0_obj.pop_front(act_block_num_tiles);
#endif
        }
        start_reader_idx = reader_idx;
        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            // Only read reader indices on cores that have sharded input (is_sender_core).
            if (is_sender_core) {
                start_reader_idx += (static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1);
            }
        }
    }

    noc.async_write_barrier();
}
