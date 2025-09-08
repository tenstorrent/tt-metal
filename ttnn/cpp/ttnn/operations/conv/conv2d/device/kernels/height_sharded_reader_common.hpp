// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

// Zero out all tiles for a given circular buffer.
template <uint32_t cb_id>
FORCE_INLINE void zero_out_tiles() {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    static_assert(
        tile_size % MEM_ZEROS_SIZE == 0, "Tile size must be a multiple of MEM_ZEROS_BASE for zeroing out tiles");
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_zeros_reads = (tile_size / MEM_ZEROS_SIZE) * num_tiles;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

template <
    uint32_t dilation_w,
    uint32_t coalesced_read_bytes,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t stride_w_bytes,
    uint32_t weight_size_w,
    uint32_t stride_w>
FORCE_INLINE void read_sticks(
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr,
    uint32_t reader_offset,
    uint32_t& l1_write_addr_act,
    uint32_t& reader_idx) {
    uint16_t num_elems = packed_reader_indices_ptr[reader_idx] & 0xffff;

    while (num_elems--) {
        reader_idx++;
        uint16_t start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
        uint16_t end_ind = packed_reader_indices_ptr[reader_idx] >> 16;

        if constexpr (dilation_w == 1) {
            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
            }
        } else {
            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                    noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                    l1_write_addr_act += conv_act_c_read_bytes;
                    act_l1_offset += stride_w_bytes;
                }
                l1_write_addr_act += act_block_w_extra_align_bytes;
            }
        }
    }
    reader_idx++;
}

template <uint32_t dram_addr_index, uint32_t page_size_index, uint32_t tensor_args_index, uint32_t cb_reader_index>
void load_config_tensor_if_in_dram(uint32_t core_index) {
#ifdef CONFIG_TENSOR_IN_DRAM
    // TODO: Instead of all cores reading from dram, only the first column reads, and does an MCAST to all the other
    // cores in the row.
    constexpr uint32_t config_dram_addr = get_compile_time_arg_val(dram_addr_index);
    constexpr uint32_t config_page_size = get_compile_time_arg_val(page_size_index);
    const auto config_tensor_args = TensorAccessorArgs<tensor_args_index>();
    const auto config_accessor = TensorAccessor(config_tensor_args, config_dram_addr, config_page_size);
    uint64_t src_noc_addr = get_noc_addr(core_index, config_accessor);

    noc_async_read(src_noc_addr, get_write_ptr(cb_reader_index), config_page_size);
    noc_async_read_barrier();
    cb_push_back(cb_reader_index, 1);
#endif
}
