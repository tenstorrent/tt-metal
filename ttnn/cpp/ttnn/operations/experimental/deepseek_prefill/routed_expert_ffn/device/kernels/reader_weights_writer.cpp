// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NCRISC dataflow kernel — reads gate_proj and up_proj tiles, writes gate_out
// and up_out back to DRAM.
//
// Weight read order mirrors the compute kernel: M_outer → K_outer → N_inner.
// For each (M_block, K_block, N_block):
//   1. Push K_block × N_block gate_proj tiles into CB_IN1_GATE.
//   2. Push K_block × N_block up_proj   tiles into CB_IN1_UP.
//
// After all K_blocks for a given M_block, read M_block × N_tiles output tiles
// from CB_GATE_OUT and CB_UP_OUT and write them to DRAM.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ── Compile-time args ────────────────────────────────────────────────────
    constexpr uint32_t M_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t N_num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(8);

    // TensorAccessor offsets in the compile-time arg list.
    constexpr uint32_t gate_ta_offset = 9;
    constexpr auto gate_ta_args = TensorAccessorArgs<gate_ta_offset>();
    constexpr uint32_t up_ta_offset = gate_ta_args.next_compile_time_args_offset();
    constexpr auto up_ta_args = TensorAccessorArgs<up_ta_offset>();
    constexpr uint32_t gate_out_ta_off = up_ta_args.next_compile_time_args_offset();
    constexpr auto gate_out_ta_args = TensorAccessorArgs<gate_out_ta_off>();
    constexpr uint32_t up_out_ta_off = gate_out_ta_args.next_compile_time_args_offset();
    constexpr auto up_out_ta_args = TensorAccessorArgs<up_out_ta_off>();

    // ── Runtime args ─────────────────────────────────────────────────────────
    uint32_t argidx = 0;
    const uint32_t gate_proj_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t up_proj_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gate_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t up_out_addr = get_arg_val<uint32_t>(argidx++);

    const auto gate_acc = TensorAccessor(gate_ta_args, gate_proj_addr, in1_tile_size);
    const auto up_acc = TensorAccessor(up_ta_args, up_proj_addr, in1_tile_size);
    const auto gate_out_acc = TensorAccessor(gate_out_ta_args, gate_out_addr, out_tile_size);
    const auto up_out_acc = TensorAccessor(up_out_ta_args, up_out_addr, out_tile_size);

    constexpr uint32_t cb_in1_gate = tt::CBIndex::c_1;
    constexpr uint32_t cb_in1_up = tt::CBIndex::c_2;
    constexpr uint32_t cb_gate_out = tt::CBIndex::c_5;
    constexpr uint32_t cb_up_out = tt::CBIndex::c_6;

    constexpr uint32_t in1_block_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t full_out_tiles = M_block_tiles * N_tiles;

    // Weight layout: tile (k_tile, n_tile) has
    //   tile_id = k_tile * N_tiles + n_tile
    constexpr uint32_t K_tiles_total = K_num_blocks * K_block_tiles;

    for (uint32_t m = 0; m < M_num_blocks; m++) {
        uint32_t m_tile_base = m * M_block_tiles;

        // ── Supply weight tiles: K_outer → N_inner ────────────────────────
        for (uint32_t k = 0; k < K_num_blocks; k++) {
            uint32_t k_tile_base = k * K_block_tiles;

            for (uint32_t n = 0; n < N_num_blocks; n++) {
                uint32_t n_tile_base = n * N_block_tiles;

                // ---- gate_proj k-block × n-block ----------------------------
                cb_reserve_back(cb_in1_gate, in1_block_tiles);
                uint32_t gate_write_ptr = get_write_ptr(cb_in1_gate);
                for (uint32_t kt = 0; kt < K_block_tiles; kt++) {
                    for (uint32_t nt = 0; nt < N_block_tiles; nt++) {
                        uint32_t tile_id = (k_tile_base + kt) * N_tiles + (n_tile_base + nt);
                        noc_async_read_page(tile_id, gate_acc, gate_write_ptr);
                        gate_write_ptr += in1_tile_size;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_in1_gate, in1_block_tiles);

                // ---- up_proj k-block × n-block ------------------------------
                cb_reserve_back(cb_in1_up, in1_block_tiles);
                uint32_t up_write_ptr = get_write_ptr(cb_in1_up);
                for (uint32_t kt = 0; kt < K_block_tiles; kt++) {
                    for (uint32_t nt = 0; nt < N_block_tiles; nt++) {
                        uint32_t tile_id = (k_tile_base + kt) * N_tiles + (n_tile_base + nt);
                        noc_async_read_page(tile_id, up_acc, up_write_ptr);
                        up_write_ptr += in1_tile_size;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_in1_up, in1_block_tiles);
            }
        }

        // ── Write gate_out M-block to DRAM ────────────────────────────────
        cb_wait_front(cb_gate_out, full_out_tiles);
        uint32_t gate_read_ptr = get_read_ptr(cb_gate_out);
        for (uint32_t mt = 0; mt < M_block_tiles; mt++) {
            for (uint32_t nt = 0; nt < N_tiles; nt++) {
                uint32_t tile_id = (m_tile_base + mt) * N_tiles + nt;
                noc_async_write_page(tile_id, gate_out_acc, gate_read_ptr);
                gate_read_ptr += out_tile_size;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_gate_out, full_out_tiles);

        // ── Write up_out M-block to DRAM ──────────────────────────────────
        cb_wait_front(cb_up_out, full_out_tiles);
        uint32_t up_read_ptr = get_read_ptr(cb_up_out);
        for (uint32_t mt = 0; mt < M_block_tiles; mt++) {
            for (uint32_t nt = 0; nt < N_tiles; nt++) {
                uint32_t tile_id = (m_tile_base + mt) * N_tiles + nt;
                noc_async_write_page(tile_id, up_out_acc, up_read_ptr);
                up_read_ptr += out_tile_size;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_up_out, full_out_tiles);
    }
}
