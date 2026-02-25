// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "elemwise_args_kernel.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

#include <tools/profiler/kernel_profiler.hpp>

// #include "api/debug/dprint_pages.h"

// inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     DPRINT << "======" << ENDL();
//     for (uint32_t r = 0; r < 32; ++r) {
//         SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
//         DPRINT_DATA0({ DPRINT << r << ": " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL(); });
//     }
//     DPRINT << "++++++" << ENDL();
// }

// #define ARCH_GRAYSKULL
constexpr uint32_t num_trids = 3;  // NOC_MAX_TRANSACTION_ID - 1;
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

    uint64_t a_noc_addr = a_tensor.get_noc_addr(args.tile_ofs);
    uint64_t b_noc_addr = b_tensor.get_noc_addr(args.tile_ofs);

    uint32_t a_addr_ofs = 0;
    uint32_t b_addr_ofs = 0;
    uint32_t trid = 1u;  // MUST START WITH ONE
    uint32_t trid_to_wait = trid;

    constexpr uint32_t num_batches = c_args.num_batches;  // noc transcation must be overlapped
    constexpr uint32_t max_num_tiles_per_batch = c_args.num_tiles_per_batch;
    const uint32_t num_tiles_per_batch =
        args.num_tiles > max_num_tiles_per_batch ? max_num_tiles_per_batch : args.num_tiles;

    cb_reserve_back(c_args.a_tensor_cb, num_tiles_per_batch);
    cb_reserve_back(c_args.b_tensor_cb, num_tiles_per_batch);
    uint32_t a_write_base_ptr = get_write_ptr(c_args.a_tensor_cb);  // must be after reserve_back
    uint32_t b_write_base_ptr = get_write_ptr(c_args.b_tensor_cb);

    uint32_t a_write_end_ptr = a_write_base_ptr + CB_PAGE_COUNT(c_args.a_tensor_cb) * a_tile_size;
    uint32_t b_write_end_ptr = b_write_base_ptr + CB_PAGE_COUNT(c_args.a_tensor_cb) * b_tile_size;

    auto next_a_cb_addr = [&](uint32_t addr, uint32_t num_tiles = 1) {
        for (auto j = 0u; j < num_tiles; j++) {
            addr += a_tile_size;
            if (addr >= a_write_end_ptr) {
                addr = a_write_base_ptr;
            }
        }

        return addr;
    };

    auto next_b_cb_addr = [&](uint32_t addr, uint32_t num_tiles = 1) {
        for (auto j = 0u; j < num_tiles; j++) {
            addr += b_tile_size;
            if (addr >= b_write_end_ptr) {
                addr = b_write_base_ptr;
            }
        }
        return addr;
    };

    uint32_t a_write_ptr = a_write_base_ptr;
    uint32_t b_write_ptr = b_write_base_ptr;

    // DPRINT << "Reader kernel. num_tiles: " << args.num_tiles << ", num_tiles_to_read: " << num_tiles_per_batch
    //        << ENDL();

    auto n_tiles_proc = num_batches * num_tiles_per_batch;

    auto num_tail_tiles = args.num_tiles % n_tiles_proc;
    auto num_tiles = args.num_tiles - num_tail_tiles;
    auto transfer_sz = n_tiles_proc * a_tile_size;
    for (auto i = 0u; i < num_tiles; i += n_tiles_proc) {
        DPRINT << "1. reading range " << i << " of " << num_tiles << " with trid " << trid << " and n_tiles_proc "
               << n_tiles_proc << ENDL();
        noc_async_read_set_trid(trid);

        // DeviceZoneScopedN("READ_TILES");

        noc_async_read_one_packet_set_state<true>(a_noc_addr, transfer_sz, args.vc);
        noc_async_read_one_packet_with_state_with_trid(a_noc_addr, a_addr_ofs, a_write_ptr, trid);
        a_addr_ofs += transfer_sz;
        a_write_ptr = next_a_cb_addr(a_write_ptr, n_tiles_proc);

        // noc_async_read_one_packet_set_state<true>(b_noc_addr, num_tiles_per_batch * b_tile_size, args.vc);
        // noc_async_read_one_packet_with_state_with_trid(b_noc_addr, b_addr_ofs, b_write_ptr, trid);
        // b_addr_ofs += num_tiles_per_batch * b_tile_size;
        // for (auto j = 0u; j < num_tiles_per_batch; j++) {
        //     b_write_ptr = next_b_cb_addr(b_write_ptr);
        // }

        if (i != 0) {
            // DeviceZoneScopedN("BARRIER");
            noc_async_read_barrier_with_trid(trid_to_wait);
            trid_to_wait = get_next_trid(trid_to_wait);

            // cb_push_back(c_args.a_tensor_cb, n_tiles_proc);
            //  cb_push_back(c_args.b_tensor_cb, num_tiles_per_batch);

            // cb_reserve_back(c_args.a_tensor_cb, n_tiles_proc);
            //  cb_reserve_back(c_args.b_tensor_cb, num_tiles_per_batch);
        }
        trid = get_next_trid(trid);
    }

    auto prev_n_tiles_proc = n_tiles_proc;
    n_tiles_proc = num_tiles_per_batch;

    transfer_sz = n_tiles_proc * a_tile_size;
    num_tiles = args.num_tiles - num_tiles;
    num_tail_tiles = num_tiles % n_tiles_proc;
    num_tiles = num_tiles - num_tail_tiles;

    for (auto i = 0u; i < num_tiles; i += n_tiles_proc) {
        DPRINT << "2. reading range " << i << " of " << num_tiles << " with trid " << trid << " and n_tiles_proc "
               << n_tiles_proc << ENDL();
        noc_async_read_set_trid(trid);

        // DeviceZoneScopedN("READ_TILES");

        noc_async_read_one_packet_set_state<true>(a_noc_addr, transfer_sz, args.vc);
        noc_async_read_one_packet_with_state_with_trid(a_noc_addr, a_addr_ofs, a_write_ptr, trid);
        a_addr_ofs += transfer_sz;
        a_write_ptr = next_a_cb_addr(a_write_ptr, n_tiles_proc);

        // noc_async_read_one_packet_set_state<true>(b_noc_addr, num_tiles_per_batch * b_tile_size, args.vc);
        // noc_async_read_one_packet_with_state_with_trid(b_noc_addr, b_addr_ofs, b_write_ptr, trid);
        // b_addr_ofs += num_tiles_per_batch * b_tile_size;
        // for (auto j = 0u; j < num_tiles_per_batch; j++) {
        //     b_write_ptr = next_b_cb_addr(b_write_ptr);
        // }

        if (i != 0) {
            // DeviceZoneScopedN("BARRIER");
            noc_async_read_barrier_with_trid(trid_to_wait);
            trid_to_wait = get_next_trid(trid_to_wait);

            // cb_push_back(c_args.a_tensor_cb, n_tiles_proc);
            // cb_push_back(c_args.b_tensor_cb, n_tiles_proc);

            // cb_reserve_back(c_args.a_tensor_cb, n_tiles_proc);
            // cb_reserve_back(c_args.b_tensor_cb, n_tiles_proc);
        } else {
            noc_async_read_barrier_with_trid(trid_to_wait);
            trid_to_wait = get_next_trid(trid_to_wait);

            // cb_push_back(c_args.a_tensor_cb, prev_n_tiles_proc);
            // cb_push_back(c_args.b_tensor_cb, prev_n_tiles_proc);

            // cb_reserve_back(c_args.a_tensor_cb, prev_n_tiles_proc);
            // cb_reserve_back(c_args.b_tensor_cb, prev_n_tiles_proc);
        }
        trid = get_next_trid(trid);
    }

    prev_n_tiles_proc = n_tiles_proc;
    n_tiles_proc = num_tail_tiles;
    transfer_sz = n_tiles_proc * a_tile_size;
    // handle tail tiles
    if (n_tiles_proc != 0) {
        DPRINT << "3. reading tail tiles with trid " << trid << " and n_tiles_proc " << n_tiles_proc << ENDL();
        // DeviceZoneScopedN("READER_KERNEL_DATA_MOVEMENT");
        // DPRINT << "Schedule read with trid " << trid << ENDL();
        noc_async_read_set_trid(trid);

        noc_async_read_one_packet_set_state<true>(a_noc_addr, transfer_sz, args.vc);
        noc_async_read_one_packet_with_state_with_trid(a_noc_addr, a_addr_ofs, a_write_ptr, trid);
        a_addr_ofs += transfer_sz;
        a_write_ptr = next_a_cb_addr(a_write_ptr, n_tiles_proc);

        // noc_async_read_one_packet_set_state<true>(b_noc_addr, b_tile_size, args.vc);
        // for (auto j = 0u; j < num_tiles_per_batch; j++) {
        //     noc_async_read_one_packet_with_state_with_trid(b_noc_addr, b_addr_ofs, b_write_ptr, trid);

        //     b_addr_ofs += b_tile_size;
        //     b_write_ptr = next_b_cb_addr(b_write_ptr);
        // }

        trid = get_next_trid(trid);
        noc_async_read_barrier_with_trid(trid_to_wait);

        trid_to_wait = get_next_trid(trid_to_wait);

        // DPRINT << "2.push data from trid " << trid_to_wait << ENDL();

        // cb_push_back(c_args.a_tensor_cb, prev_n_tiles_proc);
        // cb_push_back(c_args.b_tensor_cb, prev_n_tiles_proc);

        // cb_reserve_back(c_args.a_tensor_cb, n_tiles_proc);
        // cb_reserve_back(c_args.b_tensor_cb, n_tiles_proc);
    }

    {
        DeviceZoneScopedN("READER_KERNEL_BARRIER_FINAL");
        noc_async_read_barrier_with_trid(trid_to_wait);
    }

    // DPRINT << "3.push data from trid " << trid_to_wait << ENDL();

    // cb_push_back(c_args.a_tensor_cb, n_tiles_proc);
    // cb_push_back(c_args.b_tensor_cb, n_tiles_proc);

#endif

    // DPRINT << "Reader kernel completed" << ENDL();
}
