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
    constexpr auto ct_args = make_compile_time_struct_from_args<EltwiseWriterCTArgs>();

    const uint32_t tile_size = get_tile_size(ct_args.cb_dst);
    constexpr auto dst_args = TensorAccessorArgs<amount_of_fields<EltwiseWriterCTArgs>()>();
    const auto dst_tensor = TensorAccessor(dst_args, args.dst_base_addr, tile_size);

    uint64_t dst_noc_addr = dst_tensor.get_noc_addr(args.tile_ofs);
    auto dst_noc_ofs = 0u;

    const uint32_t large_chunk = ct_args.num_batches * ct_args.num_tiles_per_batch;
    uint32_t remaining = args.num_tiles;

    while (remaining > 0) {
        uint32_t n_tiles_proc;
        if (remaining >= large_chunk) {
            n_tiles_proc = large_chunk;
        } else if (remaining >= ct_args.num_tiles_per_batch) {
            n_tiles_proc = ct_args.num_tiles_per_batch;
        } else {
            n_tiles_proc = remaining;
        }

        // DeviceZoneScopedN("WRITER_KERNEL_DATA_MOVEMENT");
        {
            DeviceZoneScopedN("WAIT_CB_DATA");
            cb_wait_front(ct_args.cb_dst, n_tiles_proc);
        }

        uint32_t l1_read_addr = get_read_ptr(ct_args.cb_dst);

        for (uint32_t k = 0; k < n_tiles_proc; k++) {
            noc_async_write(l1_read_addr + k * tile_size, dst_noc_addr + dst_noc_ofs, tile_size);
            dst_noc_ofs += tile_size;
        }
        {
            DeviceZoneScopedN("WRITER_KERNEL_BARRIER");
            noc_async_write_barrier();
        }
        cb_pop_front(ct_args.cb_dst, n_tiles_proc);
        remaining -= n_tiles_proc;
    }

    DPRINT << "Writer kernel completed" << ENDL();
}
