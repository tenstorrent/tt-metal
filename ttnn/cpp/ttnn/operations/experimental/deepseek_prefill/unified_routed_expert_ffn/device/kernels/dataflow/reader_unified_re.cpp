// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for unified_routed_expert_ffn.
//
// Per-core responsibilities:
//   - At program start: read counts and global_expert_idx_table from DRAM into
//     L1 scratch CBs, look up count = counts[global_expert_idx_table[local_expert_id]].
//     Compute num_active_M_tiles = ceil_tile(count) which bounds how many
//     chunks this core has to feed. For chunks past num_active_M_tiles we still
//     have to push tiles (the writer will discard them) so that the compute
//     kernel's deterministic CB protocol stays in sync.
//   - For each chunk (chunks are looped at the program-factory level: one
//     compute kernel invocation == one chunk in v1), stream into the CBs:
//       Phase 1 (gate matmul):
//         For each K-block in 0..num_blocks_gu:
//           push g_in0_block_num_tiles tiles of x (this core's per_core_M rows ×
//                in0_block_w columns) into cb_in0_x
//           push g_in1_block_num_tiles tiles of gate_proj (in0_block_w × per_core_N)
//                into cb_in1_gate
//       Phase 2 (up matmul):
//         For each K-block in 0..num_blocks_gu:
//           push x tiles AGAIN into cb_in0_x (kernel pops between phases)
//           push up_proj tiles into cb_in1_up
//       Phase 3 (multiply): compute-only, no reader involvement.
//       Phase 4 (down matmul):
//         For each K-block in 0..num_blocks_d:
//           NOTE: in0 for down is already in L1 (cb_activated) — reader does
//           NOT push to cb_in0_x. Only cb_in1_down is fed.
//           push down_proj tiles into cb_in1_down
//
// This v1 reader uses no NoC multicast — each core reads its own data directly
// from DRAM. The mcast variant is a follow-up.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    // -------------------------- runtime args ------------------------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t up_addr = get_arg_val<uint32_t>(2);
    const uint32_t down_addr = get_arg_val<uint32_t>(3);
    const uint32_t counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_table_addr = get_arg_val<uint32_t>(5);

    // Which (mt, nt) block of the output this core produces. mt is the chunk-
    // local M-row block (per_core_M tiles tall), nt is the N-col block.
    const uint32_t my_mt = get_arg_val<uint32_t>(6);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(7);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(8);
    const uint32_t chunk_start_tile_row = get_arg_val<uint32_t>(9);

    // -------------------------- compile-time args -------------------------
    constexpr uint32_t cb_in0_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1_down = get_compile_time_arg_val(3);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(4);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(5);

    constexpr uint32_t local_expert_id = get_compile_time_arg_val(6);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(7);    // tiles, gate/up/down share same M block
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(8);  // gate/up N tiles per core
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(9);   // down N tiles per core
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(10);  // K for gate/up = emb / TILE
    constexpr uint32_t K_down_tiles = get_compile_time_arg_val(11);  // K for down = hidden / TILE
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(12);
    constexpr uint32_t in0_block_w_d = get_compile_time_arg_val(13);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(14);  // hidden / TILE
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(15);  // emb / TILE
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(16);       // M_max / TILE

    // Tile counts pushed per K-block (per phase).
    constexpr uint32_t g_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;
    constexpr uint32_t d_in1_block_num_tiles = per_core_N_d * in0_block_w_d;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t num_blocks_d = K_down_tiles / in0_block_w_d;

    constexpr uint32_t x_accessor_offset = 17;
    constexpr auto x_args = TensorAccessorArgs<x_accessor_offset>();
    const auto x_acc = TensorAccessor(x_args, x_addr, get_tile_size(cb_in0_x));

    constexpr uint32_t gate_accessor_offset = x_args.next_compile_time_args_offset();
    constexpr auto gate_args = TensorAccessorArgs<gate_accessor_offset>();
    const auto gate_acc = TensorAccessor(gate_args, gate_addr, get_tile_size(cb_in1_gate));

    constexpr uint32_t up_accessor_offset = gate_args.next_compile_time_args_offset();
    constexpr auto up_args = TensorAccessorArgs<up_accessor_offset>();
    const auto up_acc = TensorAccessor(up_args, up_addr, get_tile_size(cb_in1_up));

    constexpr uint32_t down_accessor_offset = up_args.next_compile_time_args_offset();
    constexpr auto down_args = TensorAccessorArgs<down_accessor_offset>();
    const auto down_acc = TensorAccessor(down_args, down_addr, get_tile_size(cb_in1_down));

    constexpr uint32_t counts_accessor_offset = down_args.next_compile_time_args_offset();
    constexpr auto counts_args = TensorAccessorArgs<counts_accessor_offset>();
    const auto counts_acc = TensorAccessor(counts_args, counts_addr);

    constexpr uint32_t idx_accessor_offset = counts_args.next_compile_time_args_offset();
    constexpr auto idx_args = TensorAccessorArgs<idx_accessor_offset>();
    const auto idx_acc = TensorAccessor(idx_args, idx_table_addr);

    // Look up active token count for this expert from device-side buffers.
    const uint32_t counts_l1 = get_write_ptr(cb_counts_scratch);
    const uint32_t idx_l1 = get_write_ptr(cb_idx_scratch);
    noc_async_read_page(0, counts_acc, counts_l1);
    noc_async_read_page(0, idx_acc, idx_l1);
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    const uint32_t global_expert_id = idx_ptr[local_expert_id];
    const uint32_t count_value = counts_ptr[global_expert_id];
    const uint32_t count_tiles = (count_value + TILE_HEIGHT - 1) / TILE_HEIGHT;

    // If this core's M-row block is entirely beyond the active token count,
    // we still need to feed the CBs (the writer will discard the result).
    // Skipping work would desynchronize the compute kernel's wait_front /
    // pop_front pattern. We still push the same number of tile-block pushes;
    // the data we push can be from the padded region (which the compute kernel
    // will multiply into a result the writer discards).
    const uint32_t this_core_first_row = chunk_start_tile_row + my_mt * per_core_M;
    const bool core_is_active = this_core_first_row < count_tiles;
    (void)core_is_active;  // v1: always feed the same number of tiles regardless

    // For v1 we always read the actual tile region (even if past count) — the
    // padded rows contain undefined-but-safe garbage that won't corrupt valid
    // output rows (those belong to active cores). This keeps the per-core
    // workload constant which simplifies the kernel and matches the existing
    // routed_expert_ffn behavior pre-Milestone 4.

    const uint32_t x_tile_bytes = get_tile_size(cb_in0_x);
    const uint32_t gate_tile_bytes = get_tile_size(cb_in1_gate);
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    const uint32_t down_tile_bytes = get_tile_size(cb_in1_down);

    // -------------------------------------------------------------------
    // PHASE 1 — gate matmul reader feed.
    // For each K-block:
    //   push per_core_M * in0_block_w_gu tiles of x to cb_in0_x
    //   push per_core_N_gu * in0_block_w_gu tiles of gate_proj to cb_in1_gate
    // -------------------------------------------------------------------
    for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
        cb_reserve_back(cb_in0_x, g_in0_block_num_tiles);
        uint32_t l1_x = get_write_ptr(cb_in0_x);
        for (uint32_t m = 0; m < per_core_M; ++m) {
            for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                const uint32_t row = this_core_first_row + m;
                const uint32_t col = kb * in0_block_w_gu + k;
                const uint32_t tile_idx = row * (K_gate_tiles) + col;
                noc_async_read_tile(tile_idx, x_acc, l1_x);
                l1_x += x_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0_x, g_in0_block_num_tiles);

        cb_reserve_back(cb_in1_gate, g_in1_block_num_tiles);
        uint32_t l1_w = get_write_ptr(cb_in1_gate);
        for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
            for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                const uint32_t row = kb * in0_block_w_gu + k;
                const uint32_t col = my_nt_gu * per_core_N_gu + n;
                const uint32_t tile_idx = row * N_gate_tiles_full + col;
                noc_async_read_tile(tile_idx, gate_acc, l1_w);
                l1_w += gate_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in1_gate, g_in1_block_num_tiles);
    }

    // -------------------------------------------------------------------
    // PHASE 2 — up matmul reader feed (re-streams x; pushes up_proj).
    // -------------------------------------------------------------------
    for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
        cb_reserve_back(cb_in0_x, g_in0_block_num_tiles);
        uint32_t l1_x = get_write_ptr(cb_in0_x);
        for (uint32_t m = 0; m < per_core_M; ++m) {
            for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                const uint32_t row = this_core_first_row + m;
                const uint32_t col = kb * in0_block_w_gu + k;
                const uint32_t tile_idx = row * (K_gate_tiles) + col;
                noc_async_read_tile(tile_idx, x_acc, l1_x);
                l1_x += x_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0_x, g_in0_block_num_tiles);

        cb_reserve_back(cb_in1_up, g_in1_block_num_tiles);
        uint32_t l1_w = get_write_ptr(cb_in1_up);
        for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
            for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                const uint32_t row = kb * in0_block_w_gu + k;
                const uint32_t col = my_nt_gu * per_core_N_gu + n;
                const uint32_t tile_idx = row * N_gate_tiles_full + col;
                noc_async_read_tile(tile_idx, up_acc, l1_w);
                l1_w += up_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in1_up, g_in1_block_num_tiles);
    }

    // -------------------------------------------------------------------
    // PHASE 4 — down matmul reader feed (in0=cb_activated already lives in
    // L1 from the compute kernel; only cb_in1_down is fed here).
    // -------------------------------------------------------------------
    for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
        cb_reserve_back(cb_in1_down, d_in1_block_num_tiles);
        uint32_t l1_w = get_write_ptr(cb_in1_down);
        for (uint32_t k = 0; k < in0_block_w_d; ++k) {
            for (uint32_t n = 0; n < per_core_N_d; ++n) {
                const uint32_t row = kb * in0_block_w_d + k;
                const uint32_t col = my_nt_d * per_core_N_d + n;
                const uint32_t tile_idx = row * N_down_tiles_full + col;
                noc_async_read_tile(tile_idx, down_acc, l1_w);
                l1_w += down_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in1_down, d_in1_block_num_tiles);
    }
}
