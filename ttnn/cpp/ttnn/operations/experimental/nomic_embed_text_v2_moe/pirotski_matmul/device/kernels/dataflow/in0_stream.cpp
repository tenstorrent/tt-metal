// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 (NOC0) of every core: the activation stream of the core's column. The column's top core reads each
// (M block, K block) of the column's rows and multicasts it down the column. With several N blocks a core, compute
// keeps the last K block of an N block for the next one, which runs K the other way, as minimal_matmul's does; that
// block is not sent again. With split_drain it then writes the last output block's odd rows as RISCV_1 posts them
// (in1_stream.cpp).

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "stream_protocol.hpp"

void kernel_main() {
    constexpr uint32_t K_tiles = get_named_compile_time_arg_val("K_tiles");
    constexpr uint32_t M_block_tiles = get_named_compile_time_arg_val("M_block_tiles");
    constexpr uint32_t K_block_tiles = get_named_compile_time_arg_val("K_block_tiles");
    constexpr uint32_t M_blocks_per_core = get_named_compile_time_arg_val("M_blocks_per_core");
    constexpr uint32_t N_blocks_per_core = get_named_compile_time_arg_val("N_blocks_per_core");
    constexpr uint32_t K_num_blocks = get_named_compile_time_arg_val("K_num_blocks");
    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("in0_cb");
    // The split norm's layout: K tiles [s * shard_width, (s + 1) * shard_width) of the column's rows sit in the L1 of
    // the column's core at NoC row shard_noc_y[s], at the input's address, row-major, M_tiles_per_core rows high.
    constexpr uint32_t shard_width = get_named_compile_time_arg_val("shard_width");
    constexpr bool split_drain = get_named_compile_time_arg_val("split_drain") == 1;
    constexpr auto in0_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr uint32_t block_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t shards = shard_width > 0 ? K_tiles / shard_width : 0;

    uint32_t arg = 0;
    const uint32_t in0_address = get_arg_val<uint32_t>(arg++);
    const bool is_sender = get_arg_val<uint32_t>(arg++) == 1;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(arg++);
    // The core's rows end here or at M, whichever is first; a partial last block stops there too.
    const uint32_t M_end_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    pirotski_matmul::Multicast receivers;
    receivers.start_x = get_arg_val<uint32_t>(arg++);
    receivers.start_y = get_arg_val<uint32_t>(arg++);
    receivers.end_x = get_arg_val<uint32_t>(arg++);
    receivers.end_y = get_arg_val<uint32_t>(arg++);
    receivers.count = get_arg_val<uint32_t>(arg++);
    uint32_t shard_noc_y[shards > 0 ? shards : 1];
    for (uint32_t shard = 0; shard < shards; ++shard) {
        shard_noc_y[shard] = get_arg_val<uint32_t>(arg++);
    }

    const Noc noc;
    DataflowBuffer in0_cb(in0_cb_id);
    Semaphore<> granted(get_named_compile_time_arg_val("in0_granted"));
    Semaphore<> arrived(get_named_compile_time_arg_val("in0_arrived"));
    const auto in0 = TensorAccessor(in0_args, in0_address);
    constexpr uint32_t tile_bytes = get_tile_size(in0_cb_id);

    // Rows past the core's end are left as they are: they feed output rows nobody writes.
    auto read_block = [&](uint32_t m_tile, uint32_t k_block, uint32_t address) {
        const uint32_t m_tile_end = m_tile + M_block_tiles < M_end_tile ? m_tile + M_block_tiles : M_end_tile;
        const uint32_t k_start = k_block * K_block_tiles;
        for (uint32_t row = m_tile; row < m_tile_end; ++row) {
            if constexpr (shard_width > 0) {
                // A row's tiles in one shard are contiguous: one read a row a shard.
                const uint32_t shard_row = (row - M_start_tile) * shard_width;
                for (uint32_t k = k_start; k < k_start + K_block_tiles;) {
                    const uint32_t shard = k / shard_width;
                    const uint32_t shard_end = (shard + 1) * shard_width;
                    const uint32_t run =
                        (shard_end < k_start + K_block_tiles ? shard_end : k_start + K_block_tiles) - k;
                    noc.async_read(
                        UnicastEndpoint{},
                        CoreLocalMem<uint32_t>(address),
                        run * tile_bytes,
                        {.noc_x = sender_x,
                         .noc_y = shard_noc_y[shard],
                         .addr = in0_address + (shard_row + k % shard_width) * tile_bytes},
                        {});
                    address += run * tile_bytes;
                    k += run;
                }
            } else {
                uint32_t page = row * K_tiles + k_start;
                for (uint32_t col = 0; col < K_block_tiles; ++col, ++page) {
                    noc.async_read(in0, CoreLocalMem<uint32_t>(address), tile_bytes, {.page_id = page}, {});
                    address += tile_bytes;
                }
            }
        }
        noc.async_read_barrier();
    };

    uint32_t block = 0;
    for (uint32_t m_block = 0; m_block < M_blocks_per_core; ++m_block) {
        const uint32_t m_tile = M_start_tile + m_block * M_block_tiles;
        bool k_forward = true;
        for (uint32_t n_block = 0; n_block < N_blocks_per_core; ++n_block, k_forward = !k_forward) {
            for (uint32_t k_iter = n_block > 0 ? 1 : 0; k_iter < K_num_blocks; ++k_iter, ++block) {
                const uint32_t k_block = k_forward ? k_iter : K_num_blocks - 1 - k_iter;
                in0_cb.reserve_back(block_tiles);
                const uint32_t slot = in0_cb.get_write_ptr();
                if (is_sender) {
                    read_block(m_tile, k_block, slot);
                    // Compute reads the slot while the multicast does; send_block returns once the slot is read.
                    in0_cb.push_back(block_tiles);
                    if (receivers.count > 0) {
                        pirotski_matmul::send_block(
                            noc, granted, arrived, receivers, slot, block_tiles * tile_bytes, block);
                    }
                } else {
                    pirotski_matmul::receive_block(noc, granted, arrived, sender_x, sender_y, block);
                    in0_cb.push_back(block_tiles);
                }
            }
        }
    }

    if constexpr (split_drain) {
        // The last output block's odd rows, as RISCV_1 posts them. The output ring holds one block, so the last
        // block starts at its base.
        constexpr uint32_t N_tiles = get_named_compile_time_arg_val("N_tiles");
        constexpr uint32_t N_block_tiles = get_named_compile_time_arg_val("N_block_tiles");
        constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("out_cb");
        constexpr uint32_t out_tile_bytes = get_tile_size(out_cb_id);
        const uint32_t out_address = get_arg_val<uint32_t>(arg++);
        const uint32_t N_start_tile = get_arg_val<uint32_t>(arg++);
        const auto out = TensorAccessor(out_args, out_address);
        const uint32_t out_base = DataflowBuffer(out_cb_id).get_read_ptr();
        Semaphore<> drain_posted(get_named_compile_time_arg_val("drain_posted"));
        Semaphore<> drain_done(get_named_compile_time_arg_val("drain_done"));
        const uint32_t last = M_start_tile + (M_blocks_per_core - 1) * M_block_tiles;
        const uint32_t rows = pirotski_matmul::valid_rows(last, M_end_tile, M_block_tiles);
        for (uint32_t row = 1; row < rows; row += 2) {
            drain_posted.wait_min(row / 2 + 1);
            uint32_t address = out_base + row * N_block_tiles * out_tile_bytes;
            uint32_t page = (last + row) * N_tiles + N_start_tile + (N_blocks_per_core - 1) * N_block_tiles;
            for (uint32_t n = 0; n < N_block_tiles; ++n, ++page, address += out_tile_bytes) {
                noc.async_write(CoreLocalMem<uint32_t>(address), out, out_tile_bytes, {}, {.page_id = page});
            }
            noc.async_writes_flushed();
            drain_done.set(row / 2 + 1);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
