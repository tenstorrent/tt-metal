// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// height_server_read_split micro-bench: one read engine (this RISC on its default NoC) reads num_tiles one-tile pages
// of an interleaved DRAM tensor (pages page_start + i * page_step) into L1, chunk_tiles per transaction id, issuing
// chunk c before waiting chunk c - 1 (the conv2d_3x3 weight-server reader's loop). Timed zone "mb_read". Afterwards
// (untimed) it writes the tiles to the same pages of the output tensor, so the host can check a bit-exact copy.
// mode 1: bank-grouped stateful issue (same tiles; per bank one coordinate set-up, then only addresses per read).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t page_start = get_arg_val<uint32_t>(3);
    const uint32_t page_step = get_arg_val<uint32_t>(4);
    const uint32_t l1_tile0 = get_arg_val<uint32_t>(5);  // landing slot (tiles from the CB base)
    if (num_tiles == 0) {
        return;
    }
    const uint32_t tile_bytes = get_tile_size(cb);
    const auto in = TensorAccessor(in_args, in_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t l1 = get_write_ptr(cb) + l1_tile0 * tile_bytes;
    const Noc noc;
    const uint32_t num_chunks = (num_tiles + chunk_tiles - 1) / chunk_tiles;
    {
        DeviceZoneScopedN("mb_read");
        for (uint32_t c = 0; c <= num_chunks; ++c) {
            if (c < num_chunks) {
                dataflow_kernel_lib::set_read_trid(noc, 1 + c % 4);
                const uint32_t t1 = (c + 1) * chunk_tiles < num_tiles ? (c + 1) * chunk_tiles : num_tiles;
                for (uint32_t t = c * chunk_tiles; t < t1; ++t) {
                    noc_async_read(in.get_noc_addr(page_start + t * page_step), l1 + t * tile_bytes, tile_bytes);
                }
                dataflow_kernel_lib::set_read_trid(noc, 0);
            }
            if (c > 0) {
                dataflow_kernel_lib::async_read_barrier_with_trid(noc, 1 + (c - 1) % 4);
            }
        }
    }
    for (uint32_t t = 0; t < num_tiles; ++t) {
        noc_async_write(l1 + t * tile_bytes, out.get_noc_addr(page_start + t * page_step), tile_bytes);
    }
    noc_async_write_barrier();
}
