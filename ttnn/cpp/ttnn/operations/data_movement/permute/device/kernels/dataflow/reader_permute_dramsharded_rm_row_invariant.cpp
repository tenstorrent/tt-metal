// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// TODO
// Currently this is not functionally correct.
// After PR https://github.com/tenstorrent/tt-metal/pull/30687
// use noc_async_read_one_packet_set_state and don't use bank_id,
// that will likely make this functionally correct.

void kernel_main() {
    constexpr uint32_t N = get_named_compile_time_arg_val("N");
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t num_rows = get_named_compile_time_arg_val("num_rows");
    constexpr auto src_args = TensorAccessorArgs<0>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);
    const uint32_t bank_id = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);

    const auto s0 = TensorAccessor(src_args, src_addr, page_size);

    // NOC setup
    uint32_t src_base_addr = noc_async_read_tile_dram_sharded_set_state<true>(src_addr, page_size, bank_id, vc);
    // Reset the barrier counter in case trids are in a non-zero state
    constexpr uint32_t noc_index = 0;  // TODO avoid hardcoding tt::tt_metal::NOC::NOC_0 ?
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    // Transaction ids
    constexpr uint32_t max_trid = NOC_MAX_TRANSACTION_ID;  // 0xF
    constexpr uint32_t num_reads_in_flight = 4;
    uint32_t prev_trid = 1;
    uint32_t curr_trid = prev_trid;
    // warmup=true for first few iterations until we reach num_reads_in_flight, during which don't use barrier
    bool warmup = true;

    for (uint32_t row = start_row; row < end_row; ++row) {
        cb_reserve_back(tt::CBIndex::c_0, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);

        noc_async_read_tile_dram_sharded_set_trid(curr_trid);
        // noc_async_read_page(row, s0, src_buffer_l1_addr);
        uint64_t src_offset = s0.get_noc_addr(row) - src_base_addr;
        noc_async_read_tile_dram_sharded_with_state_with_trid(src_base_addr, src_offset, src_buffer_l1_addr, curr_trid);

        if (warmup) {
            warmup = curr_trid < num_reads_in_flight;
        } else {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(tt::CBIndex::c_0, 1);
            prev_trid = (prev_trid == max_trid) ? 1 : (prev_trid + 1);
        }
        curr_trid = (curr_trid == max_trid) ? 1 : (curr_trid + 1);
    }
    while (prev_trid != curr_trid) {
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(tt::CBIndex::c_0, 1);
        prev_trid = (prev_trid == max_trid) ? 1 : (prev_trid + 1);
    }
}
