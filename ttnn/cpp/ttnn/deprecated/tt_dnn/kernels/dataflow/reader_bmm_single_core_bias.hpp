// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"

template <bool bias_in_dram>
FORCE_INLINE void read_bias(
    uint32_t bias_addr,
    uint32_t bias_ntiles,
    uint32_t bias_cb_id,
    uint32_t bias_log2_of_pagesize,
    uint32_t bias_pagesize) {
    const InterleavedPow2AddrGenFast<bias_in_dram> s_bias = {
        .bank_base_address = bias_addr, .log_base_2_of_page_size = bias_log2_of_pagesize};

    cb_reserve_back(bias_cb_id, bias_ntiles);
    uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);
    for (uint32_t bias_tile = 0; bias_tile < bias_ntiles; ++bias_tile) {
        noc_async_read_page(bias_tile, s_bias, bias_l1_addr);
        bias_l1_addr += bias_pagesize;
    }
    noc_async_read_barrier();
    cb_push_back(bias_cb_id, bias_ntiles);
}  // read_bias()

template <bool bias_in_dram>
FORCE_INLINE void read_bias_with_offset(
    uint32_t bias_addr,
    uint32_t bias_tile_offset,
    uint32_t bias_ntiles,
    uint32_t bias_cb_id,
    uint32_t bias_log2_of_pagesize,
    uint32_t bias_pagesize) {
    const InterleavedPow2AddrGenFast<bias_in_dram> s_bias = {
        .bank_base_address = bias_addr, .log_base_2_of_page_size = bias_log2_of_pagesize};

    cb_reserve_back(bias_cb_id, bias_ntiles);
    uint32_t bias_l1_addr = get_write_ptr(bias_cb_id);
    for (uint32_t bias_tile = bias_tile_offset; bias_tile < bias_tile_offset + bias_ntiles; ++bias_tile) {
        noc_async_read_page(bias_tile, s_bias, bias_l1_addr);
        bias_l1_addr += bias_pagesize;
    }
    noc_async_read_barrier();
    cb_push_back(bias_cb_id, bias_ntiles);
}  // read_bias()
