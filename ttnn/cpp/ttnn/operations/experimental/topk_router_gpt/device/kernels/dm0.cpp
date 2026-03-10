// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DM0 Kernel: Weight + Input Reader (RISCV_0, NOC 0)
//
// Triple-buffered pipeline for weight reads from DRAM.
// Input tiles read with double-buffered blocks so compute can start early.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#define NUM_SLOTS 3

#define ADVANCE_SLOT(s)       \
    do {                      \
        (s)++;                \
        if ((s) >= NUM_SLOTS) \
            (s) = 0;          \
    } while (0)
#define ADVANCE_TRID(t)      \
    do {                     \
        (t)++;               \
        if ((t) > NUM_SLOTS) \
            (t) = 1;         \
    } while (0)

void kernel_main() {
    // Compile-time args
    constexpr uint32_t tile_size = get_named_compile_time_arg_val("tile_size_bf16");
    constexpr uint32_t n_tiles_total = get_named_compile_time_arg_val("n_tiles");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    // Runtime args
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t input_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_bank_id = get_arg_val<uint32_t>(2);
    uint32_t vchannel = get_arg_val<uint32_t>(3);
    uint32_t num_k_tiles = get_arg_val<uint32_t>(4);
    uint32_t k_tile_offset = get_arg_val<uint32_t>(5);
    uint32_t n_tile_id = get_arg_val<uint32_t>(6);
    uint32_t is_worker = get_arg_val<uint32_t>(7);
    uint32_t bias_addr = get_arg_val<uint32_t>(8);

    constexpr uint32_t CB_WEIGHT = tt::CBIndex::c_0;
    constexpr uint32_t CB_INPUT = tt::CBIndex::c_1;
    constexpr uint32_t CB_BIAS = tt::CBIndex::c_4;

    const InterleavedAddrGenFast</*DRAM=*/true> input_addrgen = {
        .bank_base_address = input_addr, .page_size = tile_size, .data_format = get_dataformat(CB_INPUT)};

    const InterleavedAddrGenFast</*DRAM=*/true> weight_addrgen = {
        .bank_base_address = weight_addr, .page_size = tile_size, .data_format = get_dataformat(CB_WEIGHT)};

    // ----- Triple-buffered weight + input reads -----
    // Push tiles in blocks so compute can start matmul before all tiles arrive.
    constexpr uint32_t BLOCK_SIZE = 2;
    uint32_t tiles_done = 0;

    while (tiles_done < num_k_tiles) {
        uint32_t block = num_k_tiles - tiles_done;
        if (block > BLOCK_SIZE) {
            block = BLOCK_SIZE;
        }

        cb_reserve_back(CB_INPUT, block);
        cb_reserve_back(CB_WEIGHT, block);
        uint32_t inp_wr = get_write_ptr(CB_INPUT);
        uint32_t wt_wr = get_write_ptr(CB_WEIGHT);

        for (uint32_t k = 0; k < block; k++) {
            uint32_t kg = k_tile_offset + tiles_done + k;
            noc_async_read_tile(kg, input_addrgen, inp_wr + k * tile_size);
            noc_async_read_tile(kg * n_tiles_total + n_tile_id, weight_addrgen, wt_wr + k * tile_size);
        }
        noc_async_read_barrier();
        cb_push_back(CB_INPUT, block);
        cb_push_back(CB_WEIGHT, block);

        tiles_done += block;
    }

    // ----- Read bias (worker only, 1 tile) -----
    if (is_worker) {
        const InterleavedAddrGenFast</*DRAM=*/true> bias_addrgen = {
            .bank_base_address = bias_addr, .page_size = tile_size, .data_format = get_dataformat(CB_BIAS)};

        cb_reserve_back(CB_BIAS, 1);
        uint32_t bias_write_ptr = get_write_ptr(CB_BIAS);
        noc_async_read_tile(n_tile_id, bias_addrgen, bias_write_ptr);
        noc_async_read_barrier();
        cb_push_back(CB_BIAS, 1);
    }
}

#undef ADVANCE_TRID
#undef ADVANCE_SLOT
#undef NUM_SLOTS
