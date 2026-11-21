// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_1 (NOC1) of every core: the weight stream of the core's row, the bias and the output. The row's right core
// reads each (K block, N block of the row's lane) and multicasts it west along the row. Output blocks go M block by
// M block, N block by N block within one, and every second N block of an M block runs K backwards, as in
// minimal_matmul.
//
// Output block b - 1 is written during block b's K loop: with streamed weights at K step defer_write_k_block
// (never 0 when there is a later one), before that step's weights are taken; with resident weights after all of
// block b's weight pushes, which wait only for compute to free their slots. Either way compute has pushed block
// b - 1 by then, since it pushes a block before it needs the next block's first K block. The last block goes out
// row by row as compute pushes it; with split_drain RISCV_0 writes its odd rows. With direct_pack the output ring
// is the output's own shard and nothing here writes or pops it. A strip core of a sharded output (shard_drain)
// writes each row's tiles into the shards of its column's main cores, N_tiles_per_shard to a shard.

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
    constexpr uint32_t N_tiles = get_named_compile_time_arg_val("N_tiles");
    constexpr uint32_t M_block_tiles = get_named_compile_time_arg_val("M_block_tiles");
    constexpr uint32_t K_block_tiles = get_named_compile_time_arg_val("K_block_tiles");
    constexpr uint32_t N_block_tiles = get_named_compile_time_arg_val("N_block_tiles");
    constexpr uint32_t M_blocks_per_core = get_named_compile_time_arg_val("M_blocks_per_core");
    constexpr uint32_t N_blocks_per_core = get_named_compile_time_arg_val("N_blocks_per_core");
    constexpr uint32_t K_num_blocks = get_named_compile_time_arg_val("K_num_blocks");
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    // The weight ring holds the lane's whole K and N: the first M block streams the weights and later ones, pushing
    // the same blocks in the same order, find each in the slot it was streamed into.
    constexpr bool resident = get_named_compile_time_arg_val("resident") == 1;
    constexpr bool split_drain = get_named_compile_time_arg_val("split_drain") == 1;
    constexpr bool direct_pack = get_named_compile_time_arg_val("direct_pack") == 1;
    static_assert(!(direct_pack && split_drain));
    static_assert(N_blocks_per_core == 1 || !direct_pack);
    constexpr bool shard_drain = get_named_compile_time_arg_val("shard_drain") == 1;
    // The sender issues block b + 1's reads before it multicasts block b, overlapping the two.
    constexpr bool read_ahead = get_named_compile_time_arg_val("read_ahead") == 1;
    static_assert(!read_ahead || (M_blocks_per_core == 1 && N_blocks_per_core == 1 && !resident));
    static_assert(!shard_drain || !(direct_pack || split_drain));
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("in1_cb");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t bias_cb_id = get_named_compile_time_arg_val("bias_cb");
    constexpr auto in1_args = TensorAccessorArgs<0>();
    constexpr auto bias_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr uint32_t block_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_tiles = M_block_tiles * N_block_tiles;

    uint32_t arg = 0;
    const uint32_t in1_address = get_arg_val<uint32_t>(arg++);
    const uint32_t bias_address = get_arg_val<uint32_t>(arg++);
    const uint32_t out_address = get_arg_val<uint32_t>(arg++);
    const bool is_sender = get_arg_val<uint32_t>(arg++) == 1;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(arg++);
    // The core's rows end here or at M, whichever is first: no output row past it is written.
    const uint32_t M_end_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    pirotski_matmul::Multicast receivers;
    receivers.start_x = get_arg_val<uint32_t>(arg++);
    receivers.start_y = get_arg_val<uint32_t>(arg++);
    receivers.end_x = get_arg_val<uint32_t>(arg++);
    receivers.end_y = get_arg_val<uint32_t>(arg++);
    receivers.count = get_arg_val<uint32_t>(arg++);
    // shard_drain: the shard row of the core's first row, the tiles a shard takes of a row, and the NoC coordinates
    // of the shards' cores in N order.
    constexpr uint32_t max_shards = 16;
    uint32_t shard_row = 0;
    uint32_t N_tiles_per_shard = 0;
    uint32_t shards = 0;
    uint32_t shard_x[max_shards];
    uint32_t shard_y[max_shards];
    if constexpr (shard_drain) {
        shard_row = get_arg_val<uint32_t>(arg++);
        N_tiles_per_shard = get_arg_val<uint32_t>(arg++);
        shards = get_arg_val<uint32_t>(arg++);
        ASSERT(shards <= max_shards && shards * N_tiles_per_shard == N_block_tiles);
        for (uint32_t shard = 0; shard < shards; ++shard) {
            shard_x[shard] = get_arg_val<uint32_t>(arg++);
            shard_y[shard] = get_arg_val<uint32_t>(arg++);
        }
    }
    // A deferred write at or past the last K block never pops its block: the last rows would go out stale.
    ASSERT(M_blocks_per_core * N_blocks_per_core == 1 || resident || direct_pack || defer_write_k_block < K_num_blocks);

    const Noc noc;
    DataflowBuffer in1_cb(in1_cb_id);
    DataflowBuffer out_cb(out_cb_id);
    DataflowBuffer bias_cb(bias_cb_id);
    Semaphore<> granted(get_named_compile_time_arg_val("in1_granted"));
    Semaphore<> arrived(get_named_compile_time_arg_val("in1_arrived"));
    const auto in1 = TensorAccessor(in1_args, in1_address);
    const auto out = TensorAccessor(out_args, out_address);
    constexpr uint32_t in1_tile_bytes = get_tile_size(in1_cb_id);
    constexpr uint32_t out_tile_bytes = get_tile_size(out_cb_id);

    auto read_weights = [&](uint32_t k_block, uint32_t n_tile, uint32_t address) {
        for (uint32_t k = k_block * K_block_tiles; k < (k_block + 1) * K_block_tiles; ++k) {
            uint32_t page = k * N_tiles + n_tile;
            for (uint32_t n = 0; n < N_block_tiles; ++n, ++page) {
                noc.async_read(in1, CoreLocalMem<uint32_t>(address), in1_tile_bytes, {.page_id = page}, {});
                address += in1_tile_bytes;
            }
        }
    };
    auto write_rows = [&](uint32_t read_address, uint32_t first_row, uint32_t end_row, uint32_t n_tile) {
        for (uint32_t row = first_row; row < end_row; ++row) {
            uint32_t page = row * N_tiles + n_tile;
            for (uint32_t n = 0; n < N_block_tiles; ++n, ++page) {
                noc.async_write(CoreLocalMem<uint32_t>(read_address), out, out_tile_bytes, {}, {.page_id = page});
                read_address += out_tile_bytes;
            }
        }
    };
    // An earlier output block, whole, once compute has pushed it.
    auto write_block = [&](uint32_t m_tile, uint32_t n_tile) {
        out_cb.wait_front(out_block_tiles);
        write_rows(
            out_cb.get_read_ptr(),
            m_tile,
            m_tile + pirotski_matmul::valid_rows(m_tile, M_end_tile, M_block_tiles),
            n_tile);
        noc.async_writes_flushed();
        out_cb.pop_front(out_block_tiles);
    };

    uint32_t block = 0;
    uint32_t output_block = 0;
    uint32_t previous_m_tile = 0;
    uint32_t previous_n_tile = 0;
    [[maybe_unused]] uint32_t bias_slot = 0;
    for (uint32_t m_block = 0; m_block < M_blocks_per_core; ++m_block) {
        const uint32_t m_tile = M_start_tile + m_block * M_block_tiles;
        bool k_forward = true;
        for (uint32_t n_block = 0; n_block < N_blocks_per_core; ++n_block, ++output_block, k_forward = !k_forward) {
            const uint32_t n_tile = N_start_tile + n_block * N_block_tiles;
            for (uint32_t k_iter = 0; k_iter < K_num_blocks; ++k_iter, ++block) {
                const uint32_t k_block = k_forward ? k_iter : K_num_blocks - 1 - k_iter;
                if (!direct_pack && !resident && output_block > 0 && k_iter == defer_write_k_block) {
                    write_block(previous_m_tile, previous_n_tile);
                }
                in1_cb.reserve_back(block_tiles);
                const uint32_t slot = in1_cb.get_write_ptr();
                if (resident && m_block > 0) {
                    // Already in the slot this block was streamed into for the first M block.
                    in1_cb.push_back(block_tiles);
                } else if (is_sender) {
                    if (!read_ahead || k_iter == 0) {
                        read_weights(k_block, n_tile, slot);
                    }
                    noc.async_read_barrier();
                    in1_cb.push_back(block_tiles);
                    if (read_ahead && k_iter + 1 < K_num_blocks) {
                        // K runs forward here: one N block a core.
                        in1_cb.reserve_back(block_tiles);
                        read_weights(k_block + 1, n_tile, in1_cb.get_write_ptr());
                    }
                    if (receivers.count > 0) {
                        pirotski_matmul::send_block(
                            noc, granted, arrived, receivers, slot, block_tiles * in1_tile_bytes, block);
                    }
                } else {
                    pirotski_matmul::receive_block(noc, granted, arrived, sender_x, sender_y, block);
                    in1_cb.push_back(block_tiles);
                }
                if constexpr (has_bias) {
                    // Read after the block's last weight block, which compute takes before it adds the bias: earlier,
                    // the first block's bias reads held up the weight reads behind them. The slot holds one N block:
                    // with one a core it is read once and later blocks re-push it unchanged.
                    if (k_iter == K_num_blocks - 1 && (N_blocks_per_core > 1 || output_block == 0)) {
                        const auto bias = TensorAccessor(bias_args, bias_address);
                        constexpr uint32_t bias_tile_bytes = get_tile_size(bias_cb_id);
                        bias_cb.reserve_back(N_block_tiles);
                        bias_slot = bias_cb.get_write_ptr();
                        uint32_t address = bias_slot;
                        for (uint32_t n = n_tile; n < n_tile + N_block_tiles; ++n) {
                            noc.async_read(bias, CoreLocalMem<uint32_t>(address), bias_tile_bytes, {.page_id = n}, {});
                            address += bias_tile_bytes;
                        }
                        noc.async_read_barrier();
                    }
                }
            }
            if constexpr (has_bias) {
                if (N_blocks_per_core == 1 && output_block > 0) {
                    bias_cb.reserve_back(N_block_tiles);
                    // Re-pushed unread: the slot must be the one the bias was read into.
                    ASSERT(bias_cb.get_write_ptr() == bias_slot);
                }
                bias_cb.push_back(N_block_tiles);
            }
            if (!direct_pack && resident && output_block > 0) {
                write_block(previous_m_tile, previous_n_tile);
            }
            previous_m_tile = m_tile;
            previous_n_tile = n_tile;
        }
    }

    // The last block, row by row; flushed before each pop, since compute refills a popped row. With split_drain
    // rows go in pairs: RISCV_0 writes the second once posted, and the pair is popped once it is done.
    const uint32_t last = M_start_tile + (M_blocks_per_core - 1) * M_block_tiles;
    const uint32_t last_n_tile = N_start_tile + (N_blocks_per_core - 1) * N_block_tiles;
    const uint32_t rows = pirotski_matmul::valid_rows(last, M_end_tile, M_block_tiles);
    Semaphore<> drain_posted(get_named_compile_time_arg_val("drain_posted"));
    Semaphore<> drain_done(get_named_compile_time_arg_val("drain_done"));
    constexpr uint32_t rows_a_step = split_drain ? 2 : 1;
    for (uint32_t row = 0; row < (direct_pack ? 0 : M_block_tiles); row += rows_a_step) {
        const uint32_t step = row + rows_a_step <= M_block_tiles ? rows_a_step : M_block_tiles - row;
        out_cb.wait_front(step * N_block_tiles);
        const bool shared = split_drain && row + 1 < rows;
        if (shared) {
            drain_posted.set(row / 2 + 1);
        }
        if (row < rows) {
            if constexpr (shard_drain) {
                uint32_t address = out_cb.get_read_ptr();
                const uint32_t bytes = N_tiles_per_shard * out_tile_bytes;
                for (uint32_t shard = 0; shard < shards; ++shard, address += bytes) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(address),
                        UnicastEndpoint{},
                        bytes,
                        {},
                        {.noc_x = shard_x[shard],
                         .noc_y = shard_y[shard],
                         .addr = out_address + (shard_row + row) * bytes});
                }
            } else {
                write_rows(out_cb.get_read_ptr(), last + row, last + row + 1, last_n_tile);
            }
        }
        noc.async_writes_flushed();
        if (shared) {
            drain_done.wait_min(row / 2 + 1);
        }
        out_cb.pop_front(step * N_block_tiles);
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
