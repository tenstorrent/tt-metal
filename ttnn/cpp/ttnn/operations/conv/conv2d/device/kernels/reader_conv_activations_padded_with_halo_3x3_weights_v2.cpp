// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "height_sharded_reader_common.hpp"

void kernel_main() {
    constexpr uint32_t dilation_h = get_compile_time_arg_val(0);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(3);
    // need to have these as compile-time, they are inner loop bouds / unroll loops / constexpr conditionals based on
    // them
    constexpr uint32_t window_outer = get_compile_time_arg_val(4);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t weight_size_h = get_compile_time_arg_val(7);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(8);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(9);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(11);

    constexpr bool needs_act_block_zero_out = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t cb_id_act = get_compile_time_arg_val(21);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(22);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(23);

    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t core_index = get_arg_val<uint32_t>(0);
    load_config_tensor_if_in_dram<27, 28, 29, cb_reader_indices>(core_index);

#ifdef ACTIVATION_REUSE
    constexpr uint32_t act_reuse_cb_tiles = get_compile_time_arg_val(30);
    constexpr uint32_t act_block_w_tiles = get_compile_time_arg_val(31);
    constexpr bool readers_process_full_image_widths = get_compile_time_arg_val(32) == 1;
    constexpr uint32_t image_width_tiles = get_compile_time_arg_val(33);
    constexpr uint32_t output_image_width = get_compile_time_arg_val(34);
    constexpr uint32_t window_reuse_offset = get_compile_time_arg_val(35);
    constexpr bool need_to_push_remaining_tiles = get_compile_time_arg_val(36) == 1;

    uint32_t remaining_tiles_to_push = get_arg_val<uint32_t>(0);
#endif


    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act>();
    }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;

    // LOOP TO FILL READER INDICES

    uint32_t reader_idx = 0;

    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner
    // so the loop can be removed
    constexpr bool coalesce_window_inner_reads = true;
    constexpr uint32_t num_coalesced_reads = weight_size_w;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? num_coalesced_reads * conv_act_c_read_bytes : conv_act_c_read_bytes);
    // the conditional selecting between coalescing and no-colescing must be constexpr to that compiler can optimized
    // the other path away this has shown to be a big perf win

    // coalesce reads along weight_size_w
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

    static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    uint32_t start_reader_idx = 0;
    const uint32_t cb_start_addr = get_write_ptr(cb_id_act);
    for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
#ifdef ACTIVATION_REUSE
        uint32_t l1_write_addr_act = cb_start_addr;
#endif
        uint32_t reader_offset = act_l1_read_addr;
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            reader_idx = start_reader_idx;

#ifndef ACTIVATION_REUSE
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);

            read_sticks<
                dilation_w,
                coalesced_read_bytes,
                conv_act_c_read_bytes,
                act_block_w_extra_align_bytes,
                stride_w_bytes,
                weight_size_w,
                stride_w>(packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);

            noc_async_read_barrier();
            cb_push_back(cb_id_act, act_block_num_tiles);
            reader_offset += window_outer_offset;
#else
            read_sticks_activation_reuse<
                coalesced_read_bytes,
                conv_act_c_read_bytes,
                act_block_w_extra_align_bytes,
                window_outer_offset,
                weight_size_w,
                stride_w,
                weight_size_h,
                cb_id_act,
                act_reuse_cb_tiles,
                act_block_w_tiles,
                readers_process_full_image_widths,
                image_width_tiles,
                output_image_width,
                window_reuse_offset>(
                packed_reader_indices_ptr, act_l1_read_addr, l1_write_addr_act, reader_idx, cb_start_addr);

#endif
        }

        start_reader_idx = reader_idx;
#ifdef SPLIT_READER
        // Increment reader index for the next number of segments (number of segments for other reader)
        start_reader_idx += (static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1);
#endif
    }

#ifdef ACTIVATION_REUSE
    // Last core sometimes has less work to do, but we still need to push the same number of tiles
    // to avoid blocking compute kernels
    if constexpr (need_to_push_remaining_tiles) {
        push_remaining_tiles<cb_id_act, act_block_w_tiles, image_width_tiles>(remaining_tiles_to_push, cb_start_addr);
    }
#endif

    noc_async_write_barrier();
}
