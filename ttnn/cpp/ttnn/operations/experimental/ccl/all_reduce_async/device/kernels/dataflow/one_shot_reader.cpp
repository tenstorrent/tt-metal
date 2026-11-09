// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_reduce_async one_shot: link sender (local NoC) side. See create_one_shot_descriptor in the program factory.
//   1. Stage the input tiles of every remote-owned chunk of this link in the RS CB, in the writer's send order, a batch
//      of chunks per read barrier.
//   2. Wait until every owner core of this link has reduced its chunk (ag_ready semaphore), then stage those reduced
//      chunks (the owners' output shards) in the AG CB for the writer's multicast.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include <algorithm>
#include <cstdint>

constexpr uint32_t cb_rs_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_ag_id = get_compile_time_arg_val(1);
constexpr uint32_t in_page_size = get_compile_time_arg_val(2);
constexpr uint32_t out_page_size = get_compile_time_arg_val(3);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(4);
constexpr uint32_t batch_chunks = get_compile_time_arg_val(5);
constexpr uint32_t ag_ready_sem_id = get_compile_time_arg_val(6);
// Live-rows packing: stage only the first half (faces 0/1 = rows 0..15) of every tile, packed back to back within each
// chunk's CB slot (chunk i of a batch still starts at i * chunk_tiles pages, so the writer's CB stepping is unchanged).
constexpr bool half_tile = get_compile_time_arg_val(7) != 0;
constexpr uint32_t half_in_page = in_page_size / 2;
constexpr uint32_t half_out_page = out_page_size / 2;

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);
    size_t arg_idx = 0;
    const uint32_t num_rs_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_ag_chunks = get_arg_val<uint32_t>(arg_idx++);

    CircularBuffer cb_rs(cb_rs_id);
    CircularBuffer cb_ag(cb_ag_id);

    // 1. Remote-owned chunks: per chunk, num_segs then (noc x << 16 | y, first tile in that input shard, tiles).
    for (uint32_t done = 0; done < num_rs_chunks;) {
        const uint32_t n = std::min(batch_chunks, num_rs_chunks - done);
        cb_rs.reserve_back(n * chunk_tiles);
        uint32_t l1_write_addr = cb_rs.get_write_ptr();
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t num_segs = get_arg_val<uint32_t>(arg_idx++);
            for (uint32_t s = 0; s < num_segs; s++) {
                const uint32_t xy = get_arg_val<uint32_t>(arg_idx++);
                const uint32_t first_tile = get_arg_val<uint32_t>(arg_idx++);
                const uint32_t tiles = get_arg_val<uint32_t>(arg_idx++);
                if constexpr (half_tile) {
                    for (uint32_t t = 0; t < tiles; t++) {
                        noc_async_read(
                            get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + (first_tile + t) * in_page_size),
                            l1_write_addr + t * half_in_page,
                            half_in_page);
                    }
                    l1_write_addr += tiles * half_in_page;
                } else {
                    noc_async_read(
                        get_noc_addr(xy >> 16, xy & 0xFFFF, in_addr + first_tile * in_page_size),
                        l1_write_addr,
                        tiles * in_page_size);
                    l1_write_addr += tiles * in_page_size;
                }
            }
            if constexpr (half_tile) {
                // Next chunk starts at its full-size CB slot (the writer pops chunk_tiles pages per chunk).
                l1_write_addr += chunk_tiles * half_in_page;
            }
        }
        noc_async_read_barrier();
        cb_rs.push_back(n * chunk_tiles);
        done += n;
    }

    // 2. This device's reduced chunks of this link, once their owner cores are done.
    if (num_ag_chunks > 0) {
        auto* ag_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ag_ready_sem_id));
        noc_semaphore_wait(ag_ready, num_ag_chunks);
        cb_ag.reserve_back(num_ag_chunks * chunk_tiles);
        uint32_t l1_write_addr = cb_ag.get_write_ptr();
        for (uint32_t j = 0; j < num_ag_chunks; j++) {
            const uint32_t xy = get_arg_val<uint32_t>(arg_idx++);
            if constexpr (half_tile) {
                for (uint32_t t = 0; t < chunk_tiles; t++) {
                    noc_async_read(
                        get_noc_addr(xy >> 16, xy & 0xFFFF, out_addr + t * out_page_size),
                        l1_write_addr + t * half_out_page,
                        half_out_page);
                }
            } else {
                noc_async_read(
                    get_noc_addr(xy >> 16, xy & 0xFFFF, out_addr), l1_write_addr, chunk_tiles * out_page_size);
            }
            l1_write_addr += chunk_tiles * out_page_size;
        }
        noc_async_read_barrier();
        cb_ag.push_back(num_ag_chunks * chunk_tiles);
    }
}
