// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "debug/dprint.h"


void kernel_main() {
    uint32_t i = 0;
    uint32_t conv_act_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_w = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_num_blocks_h = get_arg_val<uint32_t>(i); i+=1;
    // inner loop bounds as compile-time args improve pef
    // uint32_t act_block_h_datums = get_arg_val<uint32_t>(i); i+=1;
    // i+=1; // skip an arg

    uint32_t act_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;

    uint32_t first_partial_right_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_partial_right_aligned_row  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t first_partial_image_num_rows          = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_first_partial_image_row    = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_full_images                       = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_full_image                 = get_arg_val<uint32_t>(i); i+=1;
    uint32_t last_partial_image_num_rows           = get_arg_val<uint32_t>(i); i+=1;
    uint32_t last_partial_left_aligned_row_width   = get_arg_val<uint32_t>(i); i+=1;

    // moved these to compile-time args
    // uint32_t window_outer                          = get_arg_val<uint32_t>(i); i+=1;
    // uint32_t window_inner                          = get_arg_val<uint32_t>(i); i+=1;
    i+=2; // skip 2 rt args

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_size_w_ = get_compile_time_arg_val(3);
    constexpr uint32_t conv_output_w_last_index = get_compile_time_arg_val(4) - 1;
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t log_base_2_of_conv_act_size_c_bytes = get_compile_time_arg_val(6);
    // TODO delete unused: get_compile_time_arg_val(7); (8), (9)
    // need to have these as compile-time, they are inner loop bouds / unroll loops / constexpr conditionals based on them
    constexpr uint32_t window_outer                        = get_compile_time_arg_val(10);
    constexpr uint32_t window_inner                        = get_compile_time_arg_val(11);
    constexpr uint32_t act_block_h_datums                  = get_compile_time_arg_val(12);

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t cb_id_sharded_act = 3;

    // Assumptions. Must be true. Validate on host.
    // assert(act_block_w_datums == C * weight_size_w)
    // assert(num_blocks_act_w == weight_size_h)
    // assert(act_block_w_datums % C == 0)
    // assert(act_block_w_datums % 32 == 0)
    // assert(act_block_h_datums % 32 == 0)
    // assert(act_block_h_ntiles == act_block_h_datums/32)
    // assert(act_block_w_ntiles == act_block_w_datums/32)
    // assert(act_block_num_tiles == (act_block_h_datums * act_block_w_datums)/1024)

    // LOOP TO FILL READER INDICES
    constexpr uint32_t cb_reader_indices = tt::CB::c_in4;
    volatile tt_l1_ptr uint32_t* reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t weights_top_left_corner_idx = 0;
    uint32_t reader_idx = 0;

    // First partial right-aligned row
    for (uint32_t k = 0; k < first_partial_right_aligned_row_width; k++) {
        reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
    }
    weights_top_left_corner_idx += skip_after_partial_right_aligned_row; // Skip padded width

    // First partial image
    for (uint32_t j = 0; j < first_partial_image_num_rows; j++) {
        for (uint32_t k = 0; k < conv_act_size_w_; k++) {
            reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
        }
        weights_top_left_corner_idx += weight_size_w - 1;
    }
    weights_top_left_corner_idx += skip_after_first_partial_image_row; // Skip padded rows

    // Full images
    for (uint32_t i = 0; i < num_full_images; i++) {
        for (uint32_t j = 0; j < conv_act_size_h; j++) {
            for (uint32_t k = 0; k < conv_act_size_w; k++) {
                reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
            }
            weights_top_left_corner_idx += weight_size_w - 1;
        }
        weights_top_left_corner_idx += skip_after_full_image; // Skip padded rows
    }

    // Last partial image
    for (uint32_t j = 0; j < last_partial_image_num_rows; j++) {
        for (uint32_t k = 0; k < conv_act_size_w; k++) {
            reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
        }
        weights_top_left_corner_idx += weight_size_w - 1;
    }

    // Last partial left-alighted row
    for (uint32_t k = 0; k < last_partial_left_aligned_row_width; k++) {
        reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
    }


    // LOOP TO FILL READER OFFSETS
    /* We can add another loop to read chunks of a stick as well.
     * - Duplicate reader_offset for same stick X times (window_inner must be 1)
     * - New loop between outer and inner that loops X times reading from same stick
     * - Read conv_act_c_read_bytes / X each time
     * - Update l1_write_addr_act by conv_act_c_read_bytes
     */
    constexpr uint32_t cb_reader_offsets = tt::CB::c_in5;
    volatile tt_l1_ptr uint32_t* reader_offsets_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_offsets));
    uint32_t reader_offset = 0; // Constant offset for each pixel within filter window
    uint32_t reader_offset_idx = 0;
    for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
        for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
            reader_offsets_ptr[reader_offset_idx++] = reader_offset++;
        }
        // -1 to go back to previous reader_offset
        reader_offset += conv_act_size_w - 1; // Assuming (weight_size_w - 1) / 2 == pad_w
    }


    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both src/dst side
    // we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner so the loop can be removed
    constexpr bool coalesce_window_inner_reads = true;
    constexpr uint32_t num_coalesced_reads = 3;
    constexpr uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
    // the conditional selecting between coalescing and no-colescing must be constexpr to that compiler can optimized the other path away
    // this has shown to be a big perf win
    if constexpr (coalesce_window_inner_reads and window_inner == num_coalesced_reads) {
        // coalesce reads along weight_size_w
        reader_offset_idx = 0;
        uint32_t act_l1_offset = 0;
        uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

        static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
        // set_state uses just x/y from the get_noc_addr, addr is ignored
        noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
        uint32_t start_reader_idx = 0;
        for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
            for (uint32_t outer = 0; outer < window_outer; outer++) {
                // Reset reader_idx to finish act_block_h_datums
                reader_idx = start_reader_idx;

                cb_reserve_back(cb_id_act, act_block_num_tiles);
                uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
                uint32_t reader_offset = act_l1_read_addr + (reader_offsets_ptr[reader_offset_idx] << log_base_2_of_conv_act_size_c_bytes);
                // #pragma GCC unroll 4 // unroll didn't help, but act_block_h_datums (loop bound) being const does help
                for (uint32_t bhd = 0; bhd < act_block_h_datums; bhd++) {
                    // local read from reader_index + reader_offset;
                    act_l1_offset = reader_offset + (reader_indices_ptr[reader_idx] << log_base_2_of_conv_act_size_c_bytes);
                    noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                    l1_write_addr_act += coalesced_read_bytes;
                    reader_idx++;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_act, act_block_num_tiles);

                reader_offset_idx += window_inner;
            }
            reader_offset_idx = 0;
            start_reader_idx = reader_idx;
        }

    } else {
        // no coalescing of reads
        reader_offset_idx = 0;
        uint32_t act_l1_offset = 0;
        uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

        static_assert(conv_act_c_read_bytes <= NOC_MAX_BURST_SIZE);
        // set_state uses just x/y from the get_noc_addr, addr is ignored
        noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), conv_act_c_read_bytes);

        uint32_t start_reader_idx = 0;
        for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
            for (uint32_t outer = 0; outer < window_outer; outer++) {
                // Reset reader_idx to finish act_block_h_datums
                reader_idx = start_reader_idx;
                cb_reserve_back(cb_id_act, act_block_num_tiles);
                uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
                for (uint32_t bhd = 0; bhd < act_block_h_datums; bhd++) {
                    // when no read coalesing, main use case is window_inner == 1,
                    // and if window_inner is const this loop should be removed by the compiler
                    for (uint32_t inner = 0; inner < window_inner; inner++) {
                        // local read from reader_index + reader_offset;
                        act_l1_offset = act_l1_read_addr + ((reader_indices_ptr[reader_idx] + reader_offsets_ptr[reader_offset_idx + inner]) << log_base_2_of_conv_act_size_c_bytes);
                        noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                        l1_write_addr_act += conv_act_c_read_bytes;

                    }
                    reader_idx++;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_act, act_block_num_tiles);

            reader_offset_idx += window_inner;
            reader_offset_idx += window_inner;
                reader_offset_idx += window_inner;
            }
            reader_offset_idx = 0;
            start_reader_idx = reader_idx;
        }
    }
}
