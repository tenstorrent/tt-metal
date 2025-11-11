// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Status:
// Functionally passes, but seems to sporadically fail. Not sure if there's
// a race condition somewhere ...
//
// Resources:
// Removal of dram_sharded APIs: https://github.com/tenstorrent/tt-metal/pull/30687
// Metal 2.0: https://github.com/tenstorrent/tt-metal/pull/31376

void kernel_main() {
    constexpr uint32_t N = get_named_compile_time_arg_val("N");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr uint32_t cb_depth = get_named_compile_time_arg_val("cb_depth");
    constexpr auto src_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);

    const auto s0 = TensorAccessor(src_args, src_addr, page_size);

    // NOC setup
    // uint64_t src_base_addr = s0.get_noc_addr(0);
    // size_t src_base_addr2 = s0.get_bank_and_offset(0).bank_id;
    // noc_async_read_one_packet_set_state<true>(src_base_addr, page_size, vc);
    // noc_async_read_set_state<true>(src_base_addr);
    // Reset the barrier counter in case trids are in a non-zero state
    constexpr uint32_t noc_index = 0;  // TODO avoid hardcoding tt::tt_metal::NOC::NOC_0 ?
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    // Transaction ids
    static_assert(
        cb_depth < NOC_MAX_TRANSACTION_ID, "Circular buffer depth exceeds max NOC transaction ID");  // max trid = 0xF
    constexpr uint32_t start_trid = 1;
    constexpr uint32_t end_trid = cb_depth;
    uint32_t prev_trid = start_trid;
    uint32_t curr_trid = start_trid;
    // warmup=true for first few iterations until we reach end_trid, during which don't use barrier
    bool warmup = true;

    // Manually keep track of CB pointer, since the get_write_ptr() API only
    // updates after a cb_push_back().
    uint32_t l1_addr_start = get_write_ptr(tt::CBIndex::c_0);
    uint32_t l1_addr = l1_addr_start;

    for (uint32_t row = start_row; row < end_row; ++row) {
        cb_reserve_back(tt::CBIndex::c_0, 1);

        noc_async_read_tile_dram_sharded_set_trid(curr_trid);
        // noc_async_read_page(row, s0, l1_addr);
        // uint64_t src_offset = s0.get_noc_addr(row) - src_base_addr;
        // noc_async_read_tile_dram_sharded_with_state_with_trid(src_base_addr, src_offset, l1_addr, curr_trid);
        // noc_async_read_one_packet_with_state<true, true>(src_offset, l1_addr, vc);
        // noc_async_read_with_state(s0.get_noc_addr(row), l1_addr, page_size);
        noc_async_read_one_packet(s0.get_noc_addr(row), l1_addr, page_size, noc_index, vc);

        if (warmup) {
            warmup = curr_trid < end_trid - 1;
        } else {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(tt::CBIndex::c_0, 1);
            prev_trid = (prev_trid == end_trid) ? start_trid : (prev_trid + 1);
        }
        l1_addr = (curr_trid == end_trid) ? l1_addr_start : (l1_addr + page_size);
        curr_trid = (curr_trid == end_trid) ? start_trid : (curr_trid + 1);
    }
    while (prev_trid != curr_trid) {
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(tt::CBIndex::c_0, 1);
        prev_trid = (prev_trid == end_trid) ? start_trid : (prev_trid + 1);
    }
}
