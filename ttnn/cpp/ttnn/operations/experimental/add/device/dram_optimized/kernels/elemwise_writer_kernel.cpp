// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace eltwise_dram_optimized;
    auto args = make_runtime_struct_from_args<EltwiseWriterArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<EltwiseWriterCTArgs>();

    const uint32_t tile_size = get_tile_size(c_args.cb_dst);
    constexpr auto dst_args = TensorAccessorArgs<amount_of_fields<EltwiseWriterCTArgs>()>();
    const auto dst_tensor = TensorAccessor(dst_args, args.dst_base_addr, tile_size);

    constexpr uint32_t max_num_tiles_per_batch = 4;
    uint32_t num_tiles_per_batch = args.num_tiles > max_num_tiles_per_batch ? max_num_tiles_per_batch : args.num_tiles;
    uint32_t end_id = args.tile_ofs + args.num_tiles;

    auto page_id = args.tile_ofs;
    // DPRINT << "WRITER KERNEL: page_id " << page_id << ", num_tiles: " << args.num_tiles << ENDL();

    //
    uint64_t dst_noc_addr = dst_tensor.get_noc_addr(args.tile_ofs);
    auto dst_noc_ofs = 0u;
    // dst_noc_ofs += a_tile_size;

    auto num_tail_tiles = args.num_tiles % num_tiles_per_batch;
    auto num_tiles = args.num_tiles - num_tail_tiles;

    for (auto tile_id = 0u; tile_id < num_tiles; tile_id += num_tiles_per_batch) {
        // DeviceZoneScopedN("WRITER_KERNEL_DATA_MOVEMENT");
        {
            DeviceZoneScopedN("WAIT_CB_DATA");
            cb_wait_front(c_args.cb_dst, num_tiles_per_batch);
        }

        uint32_t l1_read_addr = get_read_ptr(c_args.cb_dst);
        // print_full_tile(c_args.cb_dst, tile_id, true);

        for (uint32_t k = 0; k < num_tiles_per_batch; k++) {
            noc_async_write(l1_read_addr + k * tile_size, dst_noc_addr + dst_noc_ofs, tile_size);
            // DPRINT << "wrote tile " << dst_noc_ofs << ENDL();
            dst_noc_ofs += tile_size;
        }
        {
            DeviceZoneScopedN("WRITER_KERNEL_BARRIER");
            noc_async_write_barrier();
        }
        cb_pop_front(c_args.cb_dst, num_tiles_per_batch);
    }

    if (num_tail_tiles != 0) {
        num_tiles_per_batch = num_tail_tiles;
        cb_wait_front(c_args.cb_dst, num_tiles_per_batch);

        uint32_t l1_read_addr = get_read_ptr(c_args.cb_dst);
        // print_full_tile(c_args.cb_dst, tile_id, true);

        for (uint32_t k = 0; k < num_tiles_per_batch; k++) {
            noc_async_write(l1_read_addr + k * tile_size, dst_noc_addr + dst_noc_ofs, tile_size);
            // DPRINT << "wrote tile " << dst_noc_ofs << ENDL();
            dst_noc_ofs += tile_size;
        }
        {
            DeviceZoneScopedN("TILE WRITER_KERNEL_BARRIER");
            noc_async_write_barrier();
        }
        cb_pop_front(c_args.cb_dst, num_tiles_per_batch);
    }
    // DPRINT << "Writing kernel finish" << ENDL();
}
