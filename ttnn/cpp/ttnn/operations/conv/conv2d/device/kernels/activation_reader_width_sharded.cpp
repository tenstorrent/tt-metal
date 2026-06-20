// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/dataflow/dataflow_api.h>
#include "conv_reader_common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

constexpr uint32_t weight_size_h = get_compile_time_arg_val(5);
constexpr uint32_t weight_size_w = get_compile_time_arg_val(6);
// Only a part of the total channel depth (width) is used in one block.
template <int window_height, int window_width>
FORCE_INLINE void read_channels(
    Noc noc,
    uint32_t& l1_write_addr_act,
    const uint32_t act_l1_read_addr,
    const uint32_t reader_channel_idx,
    const uint32_t conv_act_c_bytes,
    const uint32_t conv_act_c_read_bytes,
    const uint32_t stride_h_bytes,
    const uint32_t stride_w_bytes) {
    uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_bytes);
#pragma GCC unroll weight_size_h
    for (uint32_t outer = 0; outer < window_height; outer++) {
        uint32_t act_l1_read_addr_row_offset = act_l1_read_addr_plus_offset;
#pragma GCC unroll weight_size_w
        for (uint32_t inner = 0; inner < window_width; inner++) {
            // Read the partial depth.
            experimental::read_with_state(noc, l1_write_addr_act, act_l1_read_addr_row_offset);
            // Increment by full depth to go to the next pixel
            l1_write_addr_act += conv_act_c_read_bytes;
            act_l1_read_addr_row_offset += stride_w_bytes;
        }
        // Go to the next row
        act_l1_read_addr_plus_offset += stride_h_bytes;
    }
}

void kernel_main() {
    constexpr uint32_t stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t dilation_h = get_compile_time_arg_val(1);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_size_w = get_compile_time_arg_val(3);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t act_block_h_datums = get_compile_time_arg_val(7);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_input_cores = get_compile_time_arg_val(9);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_w = get_compile_time_arg_val(11);
    Semaphore<> act_mcast_sender_sem(get_compile_time_arg_val(12));
    Semaphore<> act_mcast_receiver_sem(get_compile_time_arg_val(13));
    constexpr McastRect mcast_rect = {
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17)};
    constexpr uint32_t act_mcast_sender_size_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(19);
    constexpr uint32_t num_reader_cores = get_compile_time_arg_val(20);

    constexpr uint32_t cb_id_act = get_compile_time_arg_val(21);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(22);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(23);
    constexpr uint32_t cb_id_act_row_major_bfloat16 = get_compile_time_arg_val(25);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(26);

    constexpr uint32_t num_mcast_cores = num_input_cores > num_output_cores ? num_input_cores : num_output_cores;
    uint32_t i = 0;  // Runtime arg index

    uint32_t this_core_x = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t this_core_y = get_arg_val<uint32_t>(i);
    i += 1;

    // Num of cols of compute cores. (Total Cores, not active cores.)
    uint32_t num_cores_x = get_arg_val<uint32_t>(i);
    i += 1;

    // X and Y lookup are independent.
    // X Lookup table for translating logical to physical cores.
    tt_l1_ptr uint32_t* act_mcast_x_lookup = (tt_l1_ptr uint32_t*)(get_arg_addr(i));
    i += num_cores_x;
    // Y lookup.
    tt_l1_ptr uint32_t* act_mcast_y_lookup = (tt_l1_ptr uint32_t*)(get_arg_addr(i));

    // Equivalent to Core Index.
    uint32_t this_core_id = this_core_x + (num_cores_x * this_core_y);

    if (this_core_id >= num_mcast_cores) {
        return;
    }

    // Experimental API objects
    experimental::CB reader_indices_cb(cb_reader_indices);
    experimental::CB act_rm_cb(cb_id_act_row_major_bfloat16);
    experimental::CB act_cb(cb_id_act);
    experimental::CB tilized_in0_cb(tilized_in0_cb_id);
    experimental::CB sharded_act_cb(cb_id_sharded_act);
    Noc noc;

    load_config_tensor_if_in_dram<27, 28, 29, cb_reader_indices>(noc, reader_indices_cb, 0);

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_cb.get_write_ptr());

    // mcast_pipe v8 DUAL-PIPE (SenderPipe + ReceiverPipe) round-robin self-mcast.
    // Each core is the sender on its own round and a receiver on the others.
    //
    // SENDER face (built once below, PRE_HANDSHAKE=true):
    //   DATA_READY_SEM_ID    = act_mcast_receiver_sem (CTA13) — the VALID/INVALID data-ready flag.
    //   CONSUMER_READY_SEM_ID = act_mcast_sender_sem (CTA12) — the receiver->sender fan-in counter.
    //   consumer_ack_count   = num_mcast_cores - 1 (the raw fan-in wait count). DIVERGENT from the
    //     mcast fan-out: the rect area is num_reader_cores (the broadcast lands on every reader core),
    //     but only num_mcast_cores-1 cores actually ack. v8's runtime consumer_ack_count decouples the
    //     two (v7 could express only one count for both -> the original hang). The ctor sets the local
    //     data-ready cell VALID; send() re-asserts VALID per call (the M12b per-send re-assert), so the
    //     stale-INVALID left by this core's own receive rounds is refreshed before each broadcast.
    //
    // RECEIVER face (built per round inside the loop): ReceiverPipe with the SAME sem ids +
    //   PRE_HANDSHAKE=true. .receive(sx,sy) folds BOTH the raw fan-in ack (up on the consumer-ready
    //   counter) AND the raw wait(VALID); it clears the flag after the wait (H11, clear-after-wait).
    //   The raw top-of-round act_mcast_receiver_sem.set(INVALID) is kept — it also clears the stale
    //   VALID this core's own sender round leaves behind.
    dataflow_kernel_lib::SenderPipe<
        noc_index,
        get_compile_time_arg_val(13),
        /*PRE_HANDSHAKE=*/true,
        get_compile_time_arg_val(12)>
        act_send_pipe(
            noc,
            dataflow_kernel_lib::McastRect<>{
                mcast_rect.noc_x_start, mcast_rect.noc_y_start, mcast_rect.noc_x_end, mcast_rect.noc_y_end},
            /*consumer_ack_count=*/num_mcast_cores - 1);

    // Compute is divided along the width to reduce the size of CBs.
    // Only a part of the width on each core is used in one block.
    // Bytes read is conv_act_c_read_bytes.
    // Size of channel in bytes on this core is conv_act_c_bytes.
    constexpr uint32_t conv_act_c_bytes = conv_act_c_read_bytes * act_num_blocks_w;

    // Stride after each channel read.
    constexpr uint32_t stride_w_bytes = conv_act_c_bytes * dilation_w;

    // Striding to next row happens using stride_h_bytes
    constexpr uint32_t stride_h_bytes = (conv_act_size_w)*conv_act_c_bytes * dilation_h;

    uint32_t act_l1_read_addr = sharded_act_cb.get_read_ptr();
    experimental::set_read_state<conv_act_c_read_bytes>(noc, act_l1_read_addr);
    uint32_t reader_idx = 0;
    uint32_t l1_write_addr_act = 0;

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t ntile_height = act_block_h_datums / TILE_HEIGHT;
    constexpr uint32_t ntile_width = act_block_num_tiles / ntile_height;

    // Reset reader_idx to finish act_block_h_datums
    for (uint32_t block_h_index = 0; block_h_index < act_num_blocks_h; block_h_index++) {
        act_l1_read_addr = sharded_act_cb.get_read_ptr();
        uint32_t old_reader_idx = reader_idx;
        for (uint32_t block_w_index = 0; block_w_index < act_num_blocks_w; block_w_index++) {
            reader_idx = old_reader_idx;
            if (this_core_id < num_input_cores) {
                uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
                uint16_t num_elems = two_reader_indices & 0xffff;

                uint16_t remaining_indexes = TILE_HEIGHT;
                while (num_elems--) {
                    reader_idx++;
                    two_reader_indices = packed_reader_indices_ptr[reader_idx];
                    uint16_t start_ind = two_reader_indices & 0xffff;
                    uint16_t end_ind = two_reader_indices >> 16;
                    for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                        if (remaining_indexes == TILE_HEIGHT) {
                            l1_write_addr_act = act_rm_cb.get_write_ptr();
                            act_rm_cb.reserve_back(ntile_width);
                        }
                        read_channels<weight_size_h, weight_size_w>(
                            noc,
                            l1_write_addr_act,
                            act_l1_read_addr,
                            ind,
                            conv_act_c_bytes,
                            conv_act_c_read_bytes,
                            stride_h_bytes,
                            stride_w_bytes);

                        if (--remaining_indexes == 0) {
                            noc.async_read_barrier();
                            act_rm_cb.push_back(ntile_width);
                            l1_write_addr_act = act_rm_cb.get_write_ptr();
                            remaining_indexes = TILE_HEIGHT;
                        }
                    }
                }
                if (remaining_indexes && remaining_indexes != TILE_HEIGHT) {
                    noc.async_read_barrier();
                    act_rm_cb.push_back(ntile_width);
                }
                reader_idx++;

                // After reading one block, increment the starting read pointer by the width of the block.
                // Next read uses the next set of channels.
                act_l1_read_addr += conv_act_c_read_bytes;
            } else {
                for (uint32_t tile_h_index = 0; tile_h_index < ntile_height; tile_h_index++) {
                    act_rm_cb.reserve_back(ntile_width);
                    act_rm_cb.push_back(ntile_width);
                }
            }

            // Round robin self-mcast and receive tilized act matrix in cb_id_act
            // Compute should function like regular mm
#ifndef SKIP_MCAST
            for (uint32_t act_w_outer_i = 0; act_w_outer_i < num_input_cores; act_w_outer_i++) {
                act_cb.reserve_back(act_block_num_tiles);
                if (act_w_outer_i == this_core_id) {
                    // MCAST SENDER: send entire tilized input to other cores in column.
                    // compute tilizes and pops cb_id_act and pushes to tilized_in0_cb_id
                    tilized_in0_cb.wait_front(act_block_num_tiles);

                    // mcast_pipe SENDER face: send() absorbs the consumer-ready fan-in wait
                    // (PRE_HANDSHAKE; ack_count == num_mcast_cores-1) + reset, the data mcast
                    // (INCLUDE_SRC loopback inferred: in-rect && src!=dst), the data-ready VALID flag
                    // mcast on the same VC (with per-send VALID re-assert), and the flush.
                    act_send_pipe.send(
                        tilized_in0_cb.get_read_ptr(), act_cb.get_write_ptr(), act_mcast_sender_size_bytes);
                } else {
                    // MCAST RECEIVER: receive entire tilized input from sender core.
                    // Clear the flag before receiving — also clears the stale VALID this core's own
                    // sender round left behind (the helper's clear-after-wait ends INVALID; the next
                    // sender's send() re-asserts VALID, so this raw reset and H11 coexist).
                    act_mcast_receiver_sem.set(INVALID);

                    // Compute sender's logical coordinates from iteration index
                    uint32_t sender_logical_x = act_w_outer_i % num_cores_x;
                    uint32_t sender_logical_y = act_w_outer_i / num_cores_x;

                    // Lookup physical coordinates
                    uint32_t sender_x = act_mcast_x_lookup[sender_logical_x];
                    uint32_t sender_y = act_mcast_y_lookup[sender_logical_y];

                    // mcast_pipe RECEIVER face: receive() acks the sender (up on the consumer-ready
                    // counter), waits VALID, then clears the flag (H11). Folds the raw ack + wait.
                    dataflow_kernel_lib::ReceiverPipe<
                        get_compile_time_arg_val(13),
                        /*PRE_HANDSHAKE=*/true,
                        get_compile_time_arg_val(12)>
                        act_recv_pipe(noc);
                    act_recv_pipe.receive(sender_x, sender_y);
                }

                act_cb.push_back(act_block_num_tiles);

            }  // num_input_cores
            tilized_in0_cb.pop_front(act_block_num_tiles);
#endif
        }
    }
    noc.async_read_barrier();
    noc.async_write_barrier();
}
