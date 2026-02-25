// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "elemwise_args_kernel.hpp"
#include "ttnn/kernel/kernel_utils.hpp"

#include <tools/profiler/kernel_profiler.hpp>

constexpr uint32_t num_trids = 3;
uint32_t get_next_trid(uint32_t trid) { return trid == num_trids ? 1 : (trid + 1); }

void kernel_main() {
    using namespace ttnn::kernel_utils;
    using namespace eltwise_dram_optimized;
    auto args = make_runtime_struct_from_args<EltwiseWriterArgs>();
    constexpr auto ct_args = make_compile_time_struct_from_args<EltwiseWriterCTArgs>();

    const uint32_t tile_size = get_tile_size(ct_args.cb_dst);
    constexpr auto dst_args = TensorAccessorArgs<amount_of_fields<EltwiseWriterCTArgs>()>();
    const auto dst_tensor = TensorAccessor(dst_args, args.dst_base_addr, tile_size);

    uint64_t dst_noc_addr = dst_tensor.get_noc_addr(args.tile_ofs);
    uint32_t dst_noc_ofs = 0;

    uint32_t trid = 1u;  // MUST START WITH ONE
    uint32_t trid_to_wait = trid;

    constexpr uint32_t num_batches = ct_args.num_batches;
    const uint32_t num_tiles_per_batch = ct_args.num_tiles_per_batch;

    const uint32_t large_chunk = num_batches * num_tiles_per_batch;
    uint32_t remaining = args.num_tiles;
    bool first_iter = true;
    uint32_t prev_chunk = 0;
    auto noc_index = 0u;

    // cb_wait_front(ct_args.cb_dst, large_chunk);
    while (remaining > 0) {
        uint32_t chunk;
        if (remaining >= large_chunk) {
            chunk = large_chunk;
        } else if (remaining >= num_tiles_per_batch) {
            chunk = num_tiles_per_batch;
        } else {
            chunk = remaining;
        }

        uint32_t transfer_sz = chunk * tile_size;

        if (!first_iter) {
            noc_async_write_barrier_with_trid(trid_to_wait);
            trid_to_wait = get_next_trid(trid_to_wait);
            cb_pop_front(ct_args.cb_dst, prev_chunk);
        }

        cb_wait_front(ct_args.cb_dst, chunk);
        uint32_t read_ptr = get_read_ptr(ct_args.cb_dst);

        noc_async_write_one_packet_set_state(dst_noc_addr, transfer_sz, noc_index, args.vc);
        noc_async_write_one_packet_with_trid(read_ptr, dst_noc_addr + dst_noc_ofs, transfer_sz, trid);
        dst_noc_ofs += transfer_sz;

        trid = get_next_trid(trid);
        first_iter = false;
        prev_chunk = chunk;
        remaining -= chunk;
    }

    noc_async_write_barrier_with_trid(trid_to_wait);
    cb_pop_front(ct_args.cb_dst, prev_chunk);
}
