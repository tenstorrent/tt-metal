// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DM0 Kernel: Weight + Input Reader (RISCV_1, NOC 0)
//
// Reads weight and input tiles from DRAM in blocks so compute can start
// matmul before all tiles arrive. Workers additionally read one bias tile.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t tile_size = get_named_compile_time_arg_val("tile_size_bf16");
    constexpr uint32_t n_tiles_total = get_named_compile_time_arg_val("n_tiles");

    // Run-time arguments (shared layout with dm1 and compute)
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto weight_addr = get_arg_val<uint32_t>(argidx++);
    const auto input_addr = get_arg_val<uint32_t>(argidx++);
    const auto bias_addr = get_arg_val<uint32_t>(argidx++);
    const auto sem_partial_ready = get_arg_val<uint32_t>(argidx++);
    const auto is_sender = get_arg_val<uint32_t>(argidx++);
    const auto is_worker = get_arg_val<uint32_t>(argidx++);
    const auto is_collector = get_arg_val<uint32_t>(argidx++);
    const auto num_k_tiles = get_arg_val<uint32_t>(argidx++);
    const auto k_tile_offset = get_arg_val<uint32_t>(argidx++);
    const auto n_tile_id = get_arg_val<uint32_t>(argidx++);
    const auto worker_phys_x = get_arg_val<uint32_t>(argidx++);
    const auto worker_phys_y = get_arg_val<uint32_t>(argidx++);
    const auto sender_slot = get_arg_val<uint32_t>(argidx++);
    const auto worker_gather_slot = get_arg_val<uint32_t>(argidx++);
    const auto sem_topk_ready = get_arg_val<uint32_t>(argidx++);
    const auto indices_rm_addr = get_arg_val<uint32_t>(argidx++);
    const auto weights_rm_addr = get_arg_val<uint32_t>(argidx++);
    const auto aligned_page_size = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_weight = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_bias = tt::CBIndex::c_4;

    const InterleavedAddrGenFast</*DRAM=*/true> input_addrgen = {
        .bank_base_address = input_addr, .page_size = tile_size, .data_format = get_dataformat(cb_input)};

    const InterleavedAddrGenFast</*DRAM=*/true> weight_addrgen = {
        .bank_base_address = weight_addr, .page_size = tile_size, .data_format = get_dataformat(cb_weight)};

    // Push tiles in blocks so compute can start matmul before all tiles arrive.
    constexpr uint32_t BLOCK_SIZE = 2;
    uint32_t tiles_done = 0;

    while (tiles_done < num_k_tiles) {
        uint32_t block = num_k_tiles - tiles_done;
        if (block > BLOCK_SIZE) {
            block = BLOCK_SIZE;
        }

        cb_reserve_back(cb_input, block);
        cb_reserve_back(cb_weight, block);
        uint32_t inp_wr = get_write_ptr(cb_input);
        uint32_t wt_wr = get_write_ptr(cb_weight);

        for (uint32_t k = 0; k < block; k++) {
            uint32_t kg = k_tile_offset + tiles_done + k;
            noc_async_read_tile(kg, input_addrgen, inp_wr + k * tile_size);
            noc_async_read_tile(kg * n_tiles_total + n_tile_id, weight_addrgen, wt_wr + k * tile_size);
        }
        noc_async_read_barrier();
        cb_push_back(cb_input, block);
        cb_push_back(cb_weight, block);

        tiles_done += block;
    }

    // Read bias (worker only, 1 tile)
    if (is_worker) {
        const InterleavedAddrGenFast</*DRAM=*/true> bias_addrgen = {
            .bank_base_address = bias_addr, .page_size = tile_size, .data_format = get_dataformat(cb_bias)};

        cb_reserve_back(cb_bias, 1);
        uint32_t bias_write_ptr = get_write_ptr(cb_bias);
        noc_async_read_tile(n_tile_id, bias_addrgen, bias_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_bias, 1);
    }
}
