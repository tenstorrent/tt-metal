// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "firmware_common.h"

// conv1D reader kernel
void kernel_main() {
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(3);
    // need to have these as compile-time, they are inner loop bouds / unroll loops / constexpr conditionals based on
    // them
    constexpr uint32_t window_outer = get_compile_time_arg_val(4);
    constexpr uint32_t window_inner = get_compile_time_arg_val(5);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t weight_size_h = get_compile_time_arg_val(7);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(8);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(9);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(11);
    constexpr uint32_t cb_id_act = get_compile_time_arg_val(21);
    constexpr uint32_t cb_id_sharded_act = get_compile_time_arg_val(22);
    constexpr uint32_t cb_reader_indices = get_compile_time_arg_val(23);

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i);
    i += 1;

    if (noop) {
        return;
    }

    // LOOP TO FILL READER OFFSETS
    /* We can add another loop to read chunks of a stick as well.
     * - Duplicate reader_offset for same stick X times (window_inner must be 1)
     * - New loop between outer and inner that loops X times reading from same stick
     * - Read conv_act_c_read_bytes / X each time
     * - Update l1_write_addr_act by conv_act_c_read_bytes
     */
    uint32_t reader_offsets[weight_size_w * weight_size_h];
    uint32_t reader_offset = 0;  // Constant offset for each pixel within filter window
    uint32_t reader_offset_idx = 0;
    for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
        uint32_t reader_offset_row = reader_offset;
        for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
            reader_offsets[reader_offset_idx++] = reader_offset_row++;
        }
        // -1 to go back to previous reader_offset
        reader_offset += conv_act_size_w_padded;
    }

    // LOOP TO FILL READER INDICES
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t reader_idx = 0;

    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner
    // so the loop can be removed
    constexpr bool coalesce_window_inner_reads = true;
    constexpr uint32_t num_coalesced_reads = weight_size_w;
    constexpr uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
    // the conditional selecting between coalescing and no-colescing must be constexpr to that compiler can optimized
    // the other path away this has shown to be a big perf win
    reader_offset_idx = 0;
    uint32_t act_l1_offset = 0;
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);

    // static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
    //  set_state uses just x/y from the get_noc_addr, addr is ignored
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
    uint32_t start_reader_idx = 0;
    for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            // Reset reader_idx to finish act_block_h_datums
            reader_idx = start_reader_idx;

            cb_reserve_back(cb_id_act, act_block_num_tiles);
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
            uint32_t reader_offset = act_l1_read_addr + (reader_offsets[reader_offset_idx] * conv_act_c_read_bytes);
            // #pragma GCC unroll 4 // unroll didn't help, but act_block_h_datums (loop bound) being const does help
            uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];

            uint16_t num_elems = two_reader_indices & 0xffff;

            while (num_elems--) {
                reader_idx++;
                two_reader_indices = packed_reader_indices_ptr[reader_idx];

                uint16_t start_ind = two_reader_indices & 0xffff;
                uint16_t end_ind = two_reader_indices >> 16;

                for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                    act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                    noc_async_read(get_noc_addr(act_l1_offset), l1_write_addr_act, coalesced_read_bytes);
                    l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_act, act_block_num_tiles);

            reader_offset_idx += window_inner;
        }
        reader_offset_idx = 0;

        start_reader_idx = reader_idx;
    }
}
