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
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // Compile-time args
    constexpr uint32_t tile_size = get_named_compile_time_arg_val("tile_size_bf16");
    constexpr uint32_t n_tiles_total = get_named_compile_time_arg_val("n_tiles");

    // Tensor accessors (compile-time args from TensorAccessorArgs)
    constexpr auto input_accessor_args = TensorAccessorArgs<0>();
    constexpr auto weight_accessor_args = TensorAccessorArgs<input_accessor_args.next_compile_time_args_offset()>();
    constexpr auto bias_accessor_args = TensorAccessorArgs<weight_accessor_args.next_compile_time_args_offset()>();

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
    constexpr auto cb_weight_id = tt::CBIndex::c_0;
    constexpr auto cb_input_id = tt::CBIndex::c_1;
    constexpr auto cb_bias_id = tt::CBIndex::c_4;

    CircularBuffer cb_weight(cb_weight_id);
    CircularBuffer cb_input(cb_input_id);
    CircularBuffer cb_bias(cb_bias_id);

    const auto input_addrgen = TensorAccessor(input_accessor_args, input_addr);
    const auto weight_addrgen = TensorAccessor(weight_accessor_args, weight_addr);

    // Push tiles in blocks so compute can start matmul before all tiles arrive.
    constexpr uint32_t BLOCK_SIZE = 2;
    uint32_t tiles_done = 0;

    while (tiles_done < num_k_tiles) {
        uint32_t block = num_k_tiles - tiles_done;
        if (block > BLOCK_SIZE) {
            block = BLOCK_SIZE;
        }

        cb_input.reserve_back(block);
        cb_weight.reserve_back(block);
        uint32_t inp_wr = cb_input.get_write_ptr();
        uint32_t wt_wr = cb_weight.get_write_ptr();

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
        cb_input.push_back(block);
        cb_weight.push_back(block);

        tiles_done += block;
    }

    // Read bias (worker only, 1 tile)
    if (is_worker) {
        const auto bias_addrgen = TensorAccessor(bias_accessor_args, bias_addr);

        cb_bias.reserve_back(1);
        uint32_t bias_write_ptr = cb_bias.get_write_ptr();
        noc.async_read(bias_addrgen, CoreLocalMem<uint32_t>(bias_write_ptr), tile_size, {.page_id = n_tile_id}, {});
        noc.async_read_barrier();
        cb_bias.push_back(1);
    }
}
