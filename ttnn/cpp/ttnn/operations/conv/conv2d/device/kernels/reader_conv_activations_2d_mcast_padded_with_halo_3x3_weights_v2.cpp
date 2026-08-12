// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <api/dataflow/dataflow_api.h>
#include "conv_reader_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

using namespace dataflow_kernel_lib;

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
    constexpr uint32_t act_mcast_tile_size_bytes = get_compile_time_arg_val(18);
    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t cb_id_act = get_compile_time_arg_val(21);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(22);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(23);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(24);
    constexpr uint32_t cb_id_act_row_major_bfloat16 = get_compile_time_arg_val(25);
    constexpr bool split_reader_enabled = get_compile_time_arg_val(27);
    constexpr bool activation_reuse_enabled = get_compile_time_arg_val(28) == 1;
    constexpr uint32_t act_mcast_ct_base = 33 + (activation_reuse_enabled ? 8 : 0);
    constexpr auto act_mcast = McastArgs<act_mcast_ct_base, 0>();
    constexpr uint32_t post_mcast_ct_base = act_mcast.next_compile_time_args_offset();
    constexpr bool split_reader_cb_shared = get_compile_time_arg_val(post_mcast_ct_base) == 1;

    // Experimental API objects
    Noc noc;
    DataflowBuffer dfb_act_obj(cb_id_act);
    DataflowBuffer dfb_act_rm_obj(cb_id_act_row_major_bfloat16);
    DataflowBuffer dfb_tilized_in0_obj(tilized_in0_cb_id);
    DataflowBuffer dfb_reader_indices_obj(cb_reader_indices);
    DataflowBuffer dfb_sharded_act_obj(cb_id_sharded_act);

    Semaphore<> reserve_done_sem(0);
    Semaphore<> write_done_sem(0);
    if constexpr (split_reader_cb_shared) {
        // When the split reader CB is shared, both readers write to the same circular buffer.
        // Synchronization is required: the main reader signals when CB space is reserved,
        // and the second reader signals when it has finished writing its portion.
        reserve_done_sem = Semaphore<>(get_compile_time_arg_val(post_mcast_ct_base + 1));
        write_done_sem = Semaphore<>(get_compile_time_arg_val(post_mcast_ct_base + 2));
    }

    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act_row_major_bfloat16>(noc, dfb_act_rm_obj);
    }

    uint32_t i = act_mcast.next_runtime_args_offset();
    const bool is_receiver_core = get_arg_val<uint32_t>(i++) > 0;
    const bool is_sender_core = get_arg_val<uint32_t>(i++) > 0;
    uint32_t dram_config_reader_index = get_arg_val<uint32_t>(i++);

    auto act_mcast_sender = act_mcast.sender(noc);
    auto act_mcast_receiver = act_mcast.receiver(noc);

    load_config_tensor_if_in_dram<29, 30, 31, cb_reader_indices>(noc, dfb_reader_indices_obj, dram_config_reader_index);

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_reader_indices_obj.get_write_ptr());

    // TODO: need to make the read coalescing optimization cleaner
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? weight_size_w * conv_act_c_read_bytes : conv_act_c_read_bytes);

    // Fully create act matrix and tilize it before mcast
    uint32_t act_l1_read_addr = dfb_sharded_act_obj.get_read_ptr();

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
            dfb_act_rm_obj.reserve_back(act_block_num_tiles_read);
            if (is_sender_core) {
                uint32_t l1_write_addr_act = dfb_act_rm_obj.get_write_ptr();
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
            dfb_act_rm_obj.push_back(act_block_num_tiles_read);

#ifndef SKIP_MCAST
            // Round robin self-mcast and receive tilized act matrix in cb_id_act
            // Compute should function like regular mm
            for (uint32_t act_w_outer_i = 0; act_w_outer_i < act_w_num_outer; act_w_outer_i++) {
                dfb_act_obj.reserve_back(act_block_num_tiles);
                if (act_mcast.is_sender(act_w_outer_i)) {
                    act_mcast_sender.send_from_cb<act_block_num_tiles, act_mcast_tile_size_bytes>(
                        dfb_tilized_in0_obj, dfb_act_obj.get_write_ptr());
                } else if (is_receiver_core) {
                    act_mcast_receiver.receive(act_w_outer_i);
                }
                dfb_act_obj.push_back(act_block_num_tiles);
            }  // act_w_num_outer

            dfb_tilized_in0_obj.pop_front(act_block_num_tiles);
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
