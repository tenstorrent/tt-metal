// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/dataflow/dataflow_api.h>
#include "conv_reader_common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

#include <optional>

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
    constexpr dataflow_kernel_lib::McastArgs<12, 3> act_mcast_args;
    constexpr uint32_t act_post_mcast_ct_offset = act_mcast_args.next_compile_time_args_offset();
    constexpr uint32_t act_mcast_sender_size_bytes = get_compile_time_arg_val(act_post_mcast_ct_offset);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(act_post_mcast_ct_offset + 1);

    constexpr uint32_t cb_id_act = get_compile_time_arg_val(act_post_mcast_ct_offset + 3);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(act_post_mcast_ct_offset + 4);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(act_post_mcast_ct_offset + 5);
    constexpr uint32_t cb_id_act_row_major_bfloat16 = get_compile_time_arg_val(act_post_mcast_ct_offset + 7);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(act_post_mcast_ct_offset + 8);

    constexpr uint32_t num_mcast_cores = num_input_cores > num_output_cores ? num_input_cores : num_output_cores;
    uint32_t i = 0;  // Runtime arg index

    uint32_t this_core_x = get_arg_val<uint32_t>(i);
    i += 1;
    uint32_t this_core_y = get_arg_val<uint32_t>(i);
    i += 1;

    // Num of cols of compute cores. (Total Cores, not active cores.)
    uint32_t num_cores_x = get_arg_val<uint32_t>(i);
    i += 1;

    // Equivalent to Core Index.
    uint32_t this_core_id = this_core_x + (num_cores_x * this_core_y);

    if (this_core_id >= num_mcast_cores) {
        return;
    }

    // Experimental API objects
    DataflowBuffer reader_indices_dfb(cb_reader_indices);
    DataflowBuffer act_rm_dfb(cb_id_act_row_major_bfloat16);
    DataflowBuffer act_dfb(cb_id_act);
    DataflowBuffer tilized_in0_dfb(tilized_in0_cb_id);
    DataflowBuffer sharded_act_dfb(cb_id_sharded_act);
    Noc noc;

    using ActSendPipe = decltype(act_mcast_args.sender(noc));
    using ActRecvPipe = decltype(act_mcast_args.receiver(noc));
    std::optional<ActSendPipe> act_send_pipe;
    std::optional<ActRecvPipe> act_recv_pipe;
    if (act_mcast_args.can_send()) {
        act_send_pipe.emplace(act_mcast_args.sender(noc));
    }
    if (act_mcast_args.can_receive()) {
        act_recv_pipe.emplace(act_mcast_args.receiver(noc));
    }

    load_config_tensor_if_in_dram<
        act_post_mcast_ct_offset + 9,
        act_post_mcast_ct_offset + 10,
        act_post_mcast_ct_offset + 11,
        cb_reader_indices>(noc, reader_indices_dfb, 0);

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reader_indices_dfb.get_write_ptr());

    // Compute is divided along the width to reduce the size of CBs.
    // Only a part of the width on each core is used in one block.
    // Bytes read is conv_act_c_read_bytes.
    // Size of channel in bytes on this core is conv_act_c_bytes.
    constexpr uint32_t conv_act_c_bytes = conv_act_c_read_bytes * act_num_blocks_w;

    // Stride after each channel read.
    constexpr uint32_t stride_w_bytes = conv_act_c_bytes * dilation_w;

    // Striding to next row happens using stride_h_bytes
    constexpr uint32_t stride_h_bytes = (conv_act_size_w)*conv_act_c_bytes * dilation_h;

    uint32_t act_l1_read_addr = sharded_act_dfb.get_read_ptr();
    experimental::set_read_state<conv_act_c_read_bytes>(noc, act_l1_read_addr);
    uint32_t reader_idx = 0;
    uint32_t l1_write_addr_act = 0;

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t ntile_height = act_block_h_datums / TILE_HEIGHT;
    constexpr uint32_t ntile_width = act_block_num_tiles / ntile_height;

    // Reset reader_idx to finish act_block_h_datums
    for (uint32_t block_h_index = 0; block_h_index < act_num_blocks_h; block_h_index++) {
        act_l1_read_addr = sharded_act_dfb.get_read_ptr();
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
                            l1_write_addr_act = act_rm_dfb.get_write_ptr();
                            act_rm_dfb.reserve_back(ntile_width);
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
                            act_rm_dfb.push_back(ntile_width);
                            l1_write_addr_act = act_rm_dfb.get_write_ptr();
                            remaining_indexes = TILE_HEIGHT;
                        }
                    }
                }
                if (remaining_indexes && remaining_indexes != TILE_HEIGHT) {
                    noc.async_read_barrier();
                    act_rm_dfb.push_back(ntile_width);
                }
                reader_idx++;

                // After reading one block, increment the starting read pointer by the width of the block.
                // Next read uses the next set of channels.
                act_l1_read_addr += conv_act_c_read_bytes;
            } else {
                for (uint32_t tile_h_index = 0; tile_h_index < ntile_height; tile_h_index++) {
                    act_rm_dfb.reserve_back(ntile_width);
                    act_rm_dfb.push_back(ntile_width);
                }
            }

            // Round robin self-mcast and receive tilized act matrix in cb_id_act
            // Compute should function like regular mm
#ifndef SKIP_MCAST
            for (uint32_t act_w_outer_i = 0; act_w_outer_i < num_input_cores; act_w_outer_i++) {
                act_dfb.reserve_back(act_block_num_tiles);
                if (act_mcast_args.should_send(act_w_outer_i)) {
                    // compute tilizes and pops cb_id_act and pushes to tilized_in0_cb_id
                    tilized_in0_dfb.wait_front(act_block_num_tiles);

                    act_send_pipe->send(
                        tilized_in0_dfb.get_read_ptr(), act_dfb.get_write_ptr(), act_mcast_sender_size_bytes);
                } else {
                    ASSERT(act_mcast_args.can_receive());
                    act_recv_pipe->receive(act_w_outer_i);
                }

                act_dfb.push_back(act_block_num_tiles);

            }  // num_input_cores
            tilized_in0_dfb.pop_front(act_block_num_tiles);
#endif
        }
    }
    noc.async_read_barrier();
    noc.async_write_barrier();
}
