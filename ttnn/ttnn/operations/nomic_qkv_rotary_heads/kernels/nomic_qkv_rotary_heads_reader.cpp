// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_qkv_rotary_heads — reader. RISCV_0 (BRISC) on NOC0: in triage a hung x stream shows on BRISC.
//
// load_x_block (every core): the column's in0 stream. The column's top core (the x_mcast sender) gathers each
//   (M block, K block) of the column's rows from the norm's ND shards (x source table: per row and K part the
//   holder's NoC xy and the L1 address of the row's first tile in that part), writes the ones column of the bias K
//   tile, retires everything with one barrier, hands the slot to its own compute and multicasts it down the column.
//   Receivers reserve the same slot, ack and wait. Rows past the column's end are left stale: they feed output rows
//   nobody writes.
// load_rotary_table_block (q/k cores): C and S of each row of the M block (fp32 pages 2*s_t, 2*s_t+1), requested
//   after in0 K block table_load_k_block of the block is multicast (host knob TABLE_LOAD_K_BLOCK, default n-2).
//   cb_rotary_table is single-slot, so the request waits for the previous M block's last K block; issuing it late
//   keeps that wait and the table read out of the column's x stream (op_design.md stall-shadow table).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_block = get_named_compile_time_arg_val("cb_x_block");
    constexpr uint32_t cb_rotary_table = get_named_compile_time_arg_val("cb_rotary_table");
    constexpr uint32_t block_m_tiles = get_named_compile_time_arg_val("block_m_tiles");
    constexpr uint32_t k_block_tiles = get_named_compile_time_arg_val("k_block_tiles");
    constexpr uint32_t num_k_blocks = get_named_compile_time_arg_val("num_k_blocks");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t bias_k_tile = get_named_compile_time_arg_val("bias_k_tile");
    constexpr uint32_t x_part_tiles = get_named_compile_time_arg_val("x_part_tiles");
    constexpr uint32_t x_num_parts = get_named_compile_time_arg_val("x_num_parts");
    constexpr uint32_t seq_tiles = get_named_compile_time_arg_val("seq_tiles");
    constexpr bool load_tables = get_named_compile_time_arg_val("load_tables") == 1;
    constexpr auto table_args = TensorAccessorArgs<0>();
    constexpr auto x_mcast = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();

    constexpr uint32_t x_block_tiles = block_m_tiles * k_block_tiles;
    constexpr uint32_t table_block_tiles = get_named_compile_time_arg_val("table_block_tiles");
    static_assert(table_block_tiles == 2 * block_m_tiles, "one C and one S page per row of the M block");
    // The bias K tile's position inside its K block.
    constexpr uint32_t bias_k_block = bias_k_tile / k_block_tiles;
    constexpr uint32_t bias_k_in_block = bias_k_tile % k_block_tiles;
    constexpr uint32_t source_words_per_row = 2 * x_num_parts;

    uint32_t arg = 0;
    const uint32_t table_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t first_row = get_arg_val<uint32_t>(arg++);
    const uint32_t end_row = get_arg_val<uint32_t>(arg++);
    const uint32_t num_m_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t source_table_arg = arg;  // the top core's x source table (rows x parts x {noc_xy, addr})

    Noc noc;
    CircularBuffer x_cb(cb_x_block);
    CircularBuffer table_cb(cb_rotary_table);
    constexpr uint32_t x_tile_bytes = get_tile_size(cb_x_block);
    constexpr uint32_t table_tile_bytes = get_tile_size(cb_rotary_table);
    const bool is_sender = x_mcast.can_send();

    // Top core: gather rows [row0, row0 + block_m_tiles) ∩ [.., end_row) of K block k_block into the slot.
    auto gather_block = [&](uint32_t row0, uint32_t k_block, uint32_t slot) {
        const uint32_t k_start = k_block * k_block_tiles;
        const uint32_t k_hidden_end = k_start + k_block_tiles < hidden_tiles ? k_start + k_block_tiles : hidden_tiles;
        const uint32_t row_end = row0 + block_m_tiles < end_row ? row0 + block_m_tiles : end_row;
        for (uint32_t row = row0; row < row_end; ++row) {
            const uint32_t entry = source_table_arg + (row - first_row) * source_words_per_row;
            uint32_t dst = slot + (row - row0) * k_block_tiles * x_tile_bytes;
            for (uint32_t k = k_start; k < k_hidden_end;) {
                const uint32_t part = k / x_part_tiles;
                const uint32_t part_end = (part + 1) * x_part_tiles;
                const uint32_t run = (part_end < k_hidden_end ? part_end : k_hidden_end) - k;
                const uint32_t noc_xy = get_arg_val<uint32_t>(entry + 2 * part);
                const uint32_t addr = get_arg_val<uint32_t>(entry + 2 * part + 1);
                noc.async_read(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(dst),
                    run * x_tile_bytes,
                    {.noc_x = noc_xy >> 16, .noc_y = noc_xy & 0xFFFF, .addr = addr + (k % x_part_tiles) * x_tile_bytes},
                    {});
                dst += run * x_tile_bytes;
                k += run;
            }
        }
    };

    // The bias K tile of x: column 0 = 1.0 in every row, all else 0 (zeroed over the NoC, then 32 bf16 stores).
    auto write_ones_tiles = [&](uint32_t slot) {
        for (uint32_t r = 0; r < block_m_tiles; ++r) {
            const uint32_t tile = slot + (r * k_block_tiles + bias_k_in_block) * x_tile_bytes;
            noc.async_write_zeros(CoreLocalMem<uint32_t>(tile), x_tile_bytes);
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t r = 0; r < block_m_tiles; ++r) {
            volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                slot + (r * k_block_tiles + bias_k_in_block) * x_tile_bytes);
            // Row i, column 0: face 0 (rows 0-15) or face 2 (rows 16-31), 16 elements a face row.
            for (uint32_t i = 0; i < 16; ++i) {
                tile[i * 16] = 0x3F80;
                tile[512 + i * 16] = 0x3F80;
            }
        }
    };

    auto load_table_block = [&](uint32_t row0) {
        const auto table = TensorAccessor(table_args, table_addr, table_tile_bytes);
        table_cb.reserve_back(table_block_tiles);
        const uint32_t base = table_cb.get_write_ptr();
        for (uint32_t r = 0; r < block_m_tiles; ++r) {
            const uint32_t row = row0 + r;
            // Garbage rows of a ragged block read page 0 (their output is never written).
            const uint32_t s_t = row < end_row ? row % seq_tiles : 0;
            for (uint32_t half = 0; half < 2; ++half) {
                noc.async_read(
                    table,
                    CoreLocalMem<uint32_t>(base + (2 * r + half) * table_tile_bytes),
                    table_tile_bytes,
                    {.page_id = 2 * s_t + half},
                    {});
            }
        }
        noc.async_read_barrier();
        table_cb.push_back(table_block_tiles);
    };

    // Pipes are built once; each core constructs only the face(s) its role allows.
    auto sender = x_mcast.optional_sender(noc);
    auto receiver = x_mcast.optional_receiver(noc);

    uint32_t round = 0;
    for (uint32_t m_block = 0; m_block < num_m_blocks; ++m_block) {
        const uint32_t row0 = first_row + m_block * block_m_tiles;
        for (uint32_t k_block = 0; k_block < num_k_blocks; ++k_block, ++round) {
            x_cb.reserve_back(x_block_tiles);
            const uint32_t slot = x_cb.get_write_ptr();
            if (is_sender) {
                if (k_block == bias_k_block) {
                    write_ones_tiles(slot);
                }
                if constexpr (get_named_compile_time_arg_val("ablate_gather") == 0) {
                    gather_block(row0, k_block, slot);
                }
                noc.async_read_barrier();
                // Compute reads the slot while the multicast does; send() returns once the slot has been read.
                x_cb.push_back(x_block_tiles);
                sender->send(slot, slot, x_block_tiles * x_tile_bytes);
            } else {
                receiver->receive(round);
                x_cb.push_back(x_block_tiles);
            }
            if constexpr (load_tables) {
                if (k_block == get_named_compile_time_arg_val("table_load_k_block")) {
                    load_table_block(row0);
                }
            }
        }
    }
    noc.async_write_barrier();
}
