// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM0 Kernel: Weight + Input Reader (RISCV_1, NOC 0)
//
// Reads weight and input tiles from DRAM in blocks so compute can start
// matmul before all tiles arrive. Workers additionally read one bias tile.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // Compile-time args
    constexpr uint32_t tile_size = get_arg(args::tile_size_bf16);
    constexpr uint32_t n_tiles_total = get_arg(args::n_tiles);

    // Run-time arguments (shared layout with dm1 and compute)
    const auto dram_bank_id = get_arg(args::dram_bank_id);
    const auto vchannel = get_arg(args::vchannel);
    const auto is_sender = get_arg(args::is_sender);
    const auto is_worker = get_arg(args::is_worker);
    const auto is_collector = get_arg(args::is_collector);
    const auto num_k_tiles = get_arg(args::num_k_tiles);
    const auto k_tile_offset = get_arg(args::k_tile_offset);
    const auto n_tile_id = get_arg(args::n_tile_id);
    const auto worker_phys_x = get_arg(args::worker_phys_x);
    const auto worker_phys_y = get_arg(args::worker_phys_y);
    const auto sender_slot = get_arg(args::sender_slot);
    const auto worker_gather_slot = get_arg(args::worker_gather_slot);

    // DFBs
    DataflowBuffer dfb_weight(dfb::weight);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_bias(dfb::bias);

    const auto input_addrgen = TensorAccessor(tensor::input);
    const auto weight_addrgen = TensorAccessor(tensor::weight);

    // Push tiles in blocks so compute can start matmul before all tiles arrive.
    constexpr uint32_t BLOCK_SIZE = 2;
    uint32_t tiles_done = 0;

    while (tiles_done < num_k_tiles) {
        uint32_t block = num_k_tiles - tiles_done;
        if (block > BLOCK_SIZE) {
            block = BLOCK_SIZE;
        }

        dfb_input.reserve_back(block);
        dfb_weight.reserve_back(block);
        uint32_t inp_wr = dfb_input.get_write_ptr();
        uint32_t wt_wr = dfb_weight.get_write_ptr();

        for (uint32_t k = 0; k < block; k++) {
            uint32_t kg = k_tile_offset + tiles_done + k;
            noc.async_read(
                input_addrgen, CoreLocalMem<uint32_t>(inp_wr + k * tile_size), tile_size, {.page_id = kg}, {});
            noc.async_read(
                weight_addrgen,
                CoreLocalMem<uint32_t>(wt_wr + k * tile_size),
                tile_size,
                {.page_id = kg * n_tiles_total + n_tile_id},
                {});
        }
        noc.async_read_barrier();
        dfb_input.push_back(block);
        dfb_weight.push_back(block);

        tiles_done += block;
    }

    // Read bias (worker only, 1 tile)
    if (is_worker) {
        const auto bias_addrgen = TensorAccessor(tensor::bias);

        dfb_bias.reserve_back(1);
        uint32_t bias_write_ptr = dfb_bias.get_write_ptr();
        noc.async_read(bias_addrgen, CoreLocalMem<uint32_t>(bias_write_ptr), tile_size, {.page_id = n_tile_id}, {});
        noc.async_read_barrier();
        dfb_bias.push_back(1);
    }
}
