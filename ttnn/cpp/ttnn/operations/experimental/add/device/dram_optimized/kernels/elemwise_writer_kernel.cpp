// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "elemwise_args_kernel.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace eltwise_dram_optimized;
    auto args = make_runtime_struct_from_args<EltwiseWriterArgs>();
    constexpr auto c_args = make_compile_time_struct_from_args<EltwiseWriterCTArgs>();

    constexpr auto dst_args = TensorAccessorArgs<amount_of_fields<EltwiseWriterCTArgs>()>();
    const auto dst_tensor = TensorAccessor(dst_args, args.dst_base_addr, get_tile_size(c_args.cb_dst));

    constexpr uint32_t num_tiles_per_cycle = c_args.num_tiles_per_cycle;
    const uint32_t tile_size = get_tile_size(c_args.cb_dst);
    uint32_t end_id = args.tile_ofs + args.num_tiles;

    auto tile_id = args.tile_ofs;
    DPRINT << "WRITER KERNEL: tile_id " << tile_id << ", num_tiles: " << args.num_tiles << ", tile_stride "
           << args.tile_stride << ENDL();
    for (auto num_processed_tiles = 0u; num_processed_tiles < args.num_tiles;
         num_processed_tiles += num_tiles_per_cycle, tile_id += num_tiles_per_cycle * args.tile_stride) {
        DeviceZoneScopedN("WRITER_KERNEL_DATA_MOVEMENT");
        // DPRINT << "[WR] Waiting tile " << tile_id << " from output circular buffer" << ENDL();
        cb_wait_front(c_args.cb_dst, num_tiles_per_cycle);
        // DPRINT << "cb_dst is ready" << ENDL();
        uint32_t l1_read_addr = get_read_ptr(c_args.cb_dst);
        for (uint32_t k = 0; k < num_tiles_per_cycle; k += args.tile_stride) {
            noc_async_write_tile(tile_id + k, dst_tensor, l1_read_addr + k * tile_size);
        }
        {
            DeviceZoneScopedN("WRITER_KERNEL_BARRIER");
            noc_async_write_barrier();
        }
        cb_pop_front(c_args.cb_dst, num_tiles_per_cycle);
    }
    DPRINT << "Writing kernel finish" << ENDL();
}
