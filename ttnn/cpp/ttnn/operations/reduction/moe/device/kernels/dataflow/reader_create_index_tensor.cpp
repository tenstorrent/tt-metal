// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

/**
 * add a cb full of indices for the tile
 * each row is identical in the index tensor, so we just need to add an offset based on which row tile it is
 * first 32 elements are {0,..31}, then next 32 are {32,..64}
 * wt is which tile it is along the row [0, Wt) so j + 32*wt is the value in the tile at each element
 */
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // TODO: investigate moving to compile time (binary size is at risk)
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint16_t wt_offset = wt << 5;

    uint32_t count = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t k = 0; k < 16; ++k) {
                for (uint32_t l = 0; l < 16; l += 2) {
                    uint16_t value = l + 16 * j + wt_offset;
                    ptr[count] = (value + 1) << 16 | value;
                    count++;
                }
            }
        }
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t topk_addr = get_arg_val<uint32_t>(1);
    uint32_t expert_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_intermed_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_topk_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_expert_mask = get_compile_time_arg_val(3);
    constexpr bool src_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool topk_mask_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool expert_mask_is_dram = get_compile_time_arg_val(6) == 1;

    constexpr uint32_t Ht = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t K = get_compile_time_arg_val(9);
    constexpr uint32_t Kt = K % 32 == 0 ? K / 32 : K / 32 + 1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes_input = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format_input = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s0 = {
        .bank_base_address = src_addr, .page_size = tile_bytes_input, .data_format = data_format_input};

    constexpr uint32_t tile_bytes_topk = get_tile_size(cb_topk_mask);
    constexpr DataFormat data_format_topk = get_dataformat(cb_topk_mask);

    const InterleavedAddrGenFast<topk_mask_is_dram> s1 = {
        .bank_base_address = topk_addr, .page_size = tile_bytes_topk, .data_format = data_format_topk};

    constexpr uint32_t tile_bytes_expert = get_tile_size(cb_expert_mask);
    constexpr DataFormat data_format_expert = get_dataformat(cb_expert_mask);

    const InterleavedAddrGenFast<expert_mask_is_dram> s2 = {
        .bank_base_address = expert_addr, .page_size = tile_bytes_expert, .data_format = data_format_expert};

    // Stream in input tensor, buffer has four tiles as we double-buffer to continue streaming while waiting for compute
    // and we need two tiles for the bitonic sort llk We could load in an entire row of tiles at a time but that would
    // require substantially more memory (we would be double buffering four Wt sized CBs)

    uint32_t tile_id = 0;
    uint32_t tile_id_topk = 0;
    uint32_t tile_id_expert = 0;
    for (uint32_t i = 0; i < Ht; ++i) {
        // input
        cb_reserve_back(cb_id_in0, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(tile_id, s0, l1_write_addr);
            l1_write_addr += tile_bytes_input;
            tile_id++;
            generate_index_tile(cb_intermed_index, j);
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, Wt);

        // topk mask
        cb_reserve_back(cb_topk_mask, Kt);
        uint32_t l1_write_addr_topk = get_write_ptr(cb_topk_mask);
        for (uint32_t j = 0; j < Kt; ++j) {
            noc_async_read_tile(tile_id_topk, s1, l1_write_addr_topk);
            l1_write_addr_topk += tile_bytes_topk;
            tile_id_topk++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_topk_mask, Kt);

        // expert mask
        cb_reserve_back(cb_expert_mask, Wt);
        uint32_t l1_write_addr_expert = get_write_ptr(cb_expert_mask);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(tile_id_expert, s2, l1_write_addr_expert);
            l1_write_addr_expert += tile_bytes_expert;
            tile_id_expert++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_expert_mask, Wt);
    }
}
