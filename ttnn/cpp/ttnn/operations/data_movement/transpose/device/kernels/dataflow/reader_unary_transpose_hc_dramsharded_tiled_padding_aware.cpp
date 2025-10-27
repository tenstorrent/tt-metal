// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Forked from:
// reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t bank_id = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_writes = get_named_compile_time_arg_val("num_writes");
    constexpr uint32_t padding_val_packed = get_named_compile_time_arg_val("padding_val_packed");
    constexpr uint32_t needs_padding = get_named_compile_time_arg_val("needs_padding") == 1;
    constexpr uint32_t swap_hw = get_named_compile_time_arg_val("swap_hw") == 1;
    constexpr uint32_t H = get_named_compile_time_arg_val("H");
    constexpr uint32_t W = get_named_compile_time_arg_val("W");
    constexpr uint32_t accumulated_outer_dims = get_named_compile_time_arg_val("accumulated_outer_dims");
    constexpr uint32_t TILE_HEIGHT = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t TILE_WIDTH = get_named_compile_time_arg_val("tile_width");
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t H_p = tt::data_movement::common::round_up<H, TILE_HEIGHT>();
    constexpr uint32_t W_p = tt::data_movement::common::round_up<W, TILE_WIDTH>();

    constexpr uint32_t Wt = W_p / TILE_WIDTH;
    constexpr uint32_t Ht = H_p / TILE_HEIGHT;

    constexpr uint32_t HtWt = Ht * Wt;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    // dynamic noc
    uint32_t src_base_addr = noc_async_read_tile_dram_sharded_set_state<true>(src_addr, tile_bytes, bank_id, vc);
    // reset the barrier counter in case trids are in a non-zero state
    constexpr uint32_t noc_index = 0;  // TODO avoid hardcoding tt::tt_metal::NOC::NOC_0 ?
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    // transaction ids
    constexpr uint32_t max_trid = NOC_MAX_TRANSACTION_ID;  // 0xF
    constexpr uint32_t num_reads_in_flight = 4;
    uint32_t prev_trid = 1;
    uint32_t curr_trid = prev_trid;
    bool warmup =
        true;  // = true for first few iterations until we reach num_reads_in_flight, during which don't use barrier

    // gate all the writes until reads complete
    cb_reserve_back(tt::CBIndex::c_2, 1);

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        uint32_t linear_tile_index = 0;
        if constexpr (swap_hw) {
            uint32_t rem = i;
            uint32_t ht = rem % Ht;
            rem /= Ht;
            uint32_t wt = rem % Wt;
            rem /= Wt;
            uint32_t offset = rem % accumulated_outer_dims;
            linear_tile_index = offset * HtWt + ht * Wt + wt;
        } else {
            linear_tile_index = i;
        }
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        noc_async_read_tile_dram_sharded_set_trid(curr_trid);
        // noc_async_read_tile(linear_tile_index, s, l1_write_addr);
        uint64_t src_offset = s.get_noc_addr(linear_tile_index) - src_base_addr;
        noc_async_read_tile_dram_sharded_with_state_with_trid(src_base_addr, src_offset, l1_write_addr, curr_trid);

        if (warmup) {
            warmup = curr_trid < num_reads_in_flight;
        } else {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(cb_id_in0, onetile);
            prev_trid = (prev_trid == max_trid) ? 1 : (prev_trid + 1);
        }
        curr_trid = (curr_trid == max_trid) ? 1 : (curr_trid + 1);
    }
    while (prev_trid != curr_trid) {
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(cb_id_in0, onetile);
        prev_trid = (prev_trid == max_trid) ? 1 : (prev_trid + 1);
    }

    if constexpr (needs_padding) {
        // Add padding
        cb_reserve_back(tt::CBIndex::c_1, 1);
        uint32_t l1_write_addr = get_write_ptr(tt::CBIndex::c_1);
        // Fill with padding value
        // if bfloat16 num_writes = FACE_WIDTH / (sizeof(uint32_t))/(element_size)
        tt::data_movement::common::fill_with_val(l1_write_addr, num_writes, padding_val_packed);
        cb_push_back(tt::CBIndex::c_1, 1);
    }

    // done reads, ungate writes
    cb_push_back(tt::CBIndex::c_2, 1);
}
