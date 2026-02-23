// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "elemwise_args_kernel.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

#include <tools/profiler/kernel_profiler.hpp>

// #define ARCH_GRAYSKULL
constexpr uint32_t num_trids = 4;
uint32_t get_prev_trid(uint32_t trid) { return trid == 1 ? num_trids : (trid - 1); }
uint32_t get_next_trid(uint32_t trid) { return trid == num_trids ? 1 : (trid + 1); }

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace eltwise_dram_optimized;
    auto args = make_runtime_struct_from_args<EltwiseReaderArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<EltwiseReaderCTArgs>();

    constexpr auto a_tensor_args = TensorAccessorArgs<amount_of_fields<EltwiseReaderCTArgs>()>();
    constexpr auto b_tensor_args = TensorAccessorArgs<a_tensor_args.next_compile_time_args_offset()>();

    const auto a_tensor = TensorAccessor(a_tensor_args, args.a_tensor_base_addr, get_tile_size(c_args.a_tensor_cb));
    const auto b_tensor = TensorAccessor(b_tensor_args, args.b_tensor_base_addr, get_tile_size(c_args.b_tensor_cb));

    constexpr uint32_t num_tiles_per_cycle = 1;  // c_args.num_tiles_per_cycle;
    const uint32_t a_tile_size = get_tile_size(c_args.a_tensor_cb);
    const uint32_t b_tile_size = get_tile_size(c_args.b_tensor_cb);

#ifdef ARCH_GRAYSKULL
    // for (uint32_t tile_id = args.tile_ofs; tile_id < args.tile_ofs + args.num_tiles; tile_id += args.tile_stride) {

    DPRINT << "READER KERNEL: tile_ofs: " << args.tile_ofs << " num_tiles: " << args.num_tiles << ", tile_stride "
           << args.tile_stride << ENDL();

    auto tile_id = args.tile_ofs;
    for (auto num_processed_tiles = 0u; num_processed_tiles < args.num_tiles;
         num_processed_tiles += num_tiles_per_cycle, tile_id += num_tiles_per_cycle * args.tile_stride) {
        DeviceZoneScopedN("READER_KERNEL_DATA_MOVEMENT");
        //
        // DPRINT << "Reading tile " << tile_id << " from input circular buffers" << ENDL();
        cb_reserve_back(c_args.a_tensor_cb, num_tiles_per_cycle);
        cb_reserve_back(c_args.b_tensor_cb, num_tiles_per_cycle);

        uint32_t a_write_ptr = get_write_ptr(c_args.a_tensor_cb);
        uint32_t b_write_ptr = get_write_ptr(c_args.b_tensor_cb);

        for (uint32_t k = 0; k < num_tiles_per_cycle; k += args.tile_stride) {
            noc_async_read_tile(tile_id + k, a_tensor, a_write_ptr + k * a_tile_size);
            noc_async_read_tile(tile_id + k, b_tensor, b_write_ptr + k * b_tile_size);
        }
        {
            DeviceZoneScopedN("READER_KERNEL_BARRIER");
            noc_async_read_barrier();
        }

        cb_push_back(c_args.a_tensor_cb, num_tiles_per_cycle);
        cb_push_back(c_args.b_tensor_cb, num_tiles_per_cycle);
    }
#else
    // auto a_noc_addr = a_tensor.pages(args.tile_ofs, args.tile_ofs + args.num_tiles).begin()->noc_addr();
    // auto b_noc_addr = b_tensor.pages(args.tile_ofs, args.tile_ofs + args.num_tiles).begin()->noc_addr();

    // uint64_t a_noc_addr = get_noc_addr_from_bank_id<true>(args.tile_ofs, args.a_tensor_base_addr);
    // uint64_t b_noc_addr = get_noc_addr_from_bank_id<true>(args.tile_ofs, args.b_tensor_base_addr);
    uint64_t a_noc_addr = a_tensor.get_noc_addr(args.tile_ofs);
    uint64_t b_noc_addr = b_tensor.get_noc_addr(args.tile_ofs);

    uint32_t a_addr_ofs = 0;
    uint32_t b_addr_ofs = 0;
    auto trid = 1u;
    for (auto i = 0u; i < args.num_tiles; i++) {
        DeviceZoneScopedN("READER_KERNEL_DATA_MOVEMENT");
        noc_async_read_set_trid(trid);

        cb_reserve_back(c_args.a_tensor_cb, num_tiles_per_cycle);
        cb_reserve_back(c_args.b_tensor_cb, num_tiles_per_cycle);

        uint32_t a_write_ptr = get_write_ptr(c_args.a_tensor_cb);
        uint32_t b_write_ptr = get_write_ptr(c_args.b_tensor_cb);

        noc_async_read_one_packet_set_state<true>(a_noc_addr, a_tile_size, args.vc);
        noc_async_read_one_packet_with_state_with_trid(a_noc_addr, a_addr_ofs, a_write_ptr, trid);

        noc_async_read_one_packet_set_state<true>(b_noc_addr, b_tile_size, args.vc);
        noc_async_read_one_packet_with_state_with_trid(b_noc_addr, b_addr_ofs, b_write_ptr, trid);

        a_addr_ofs += a_tile_size;
        b_addr_ofs += b_tile_size;

        if (i != 0) {
            DeviceZoneScopedN("READER_KERNEL_BARRIER");
            noc_async_read_barrier_with_trid(get_prev_trid(trid));
            cb_push_back(c_args.a_tensor_cb, num_tiles_per_cycle);
            cb_push_back(c_args.b_tensor_cb, num_tiles_per_cycle);
        }

        trid = get_next_trid(trid);
    }

    noc_async_read_barrier_with_trid(get_prev_trid(trid));
    cb_push_back(c_args.a_tensor_cb, num_tiles_per_cycle);
    cb_push_back(c_args.b_tensor_cb, num_tiles_per_cycle);
#endif
    DPRINT << "Reader kernel completed" << ENDL();
}
