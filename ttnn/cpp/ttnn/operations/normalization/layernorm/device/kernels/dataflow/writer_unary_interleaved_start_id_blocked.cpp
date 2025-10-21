// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/dataflow/tile_generate.h"
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
    // DeviceZoneScopedN("FULL-WRITER");
    namespace kutil = norm::kernel_util;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    constexpr bool use_welford = get_compile_time_arg_val(1) == 1;
    constexpr auto dst_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_id_integers = tt::CBIndex::c_25;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);

    // Generate the integers tile if using Welford's algorithm
    // if constexpr (use_welford) {
    //     {
    //         DeviceZoneScopedN("TEST-GENERATE-INTEGER-TILE");
    //         kutil::dataflow::generate_incremented_tile<float, kutil::generic::policies::CBWaitPolicy::NoWait>(
    //             cb_id_integers);
    //     }
    // }

    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        cb_wait_front(cb_id_out0, blk);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j < blk; j++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, blk);
    }
}
