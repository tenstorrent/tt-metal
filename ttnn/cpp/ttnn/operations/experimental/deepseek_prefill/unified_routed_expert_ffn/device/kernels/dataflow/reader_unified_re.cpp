// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for unified_routed_expert_ffn.
//
// Per-core responsibilities, sequenced in one pass:
//   - Read counts/idx_table scratch and discover this expert's active token
//     count (currently used only for diagnostics; the host pre-sized the
//     output buffer, the writer drops chunks beyond the count).
//   - Phase 1 (gate matmul): per K-block, push per_core_M * in0_block_w_gu
//     tiles of x to cb_in0_x and per_core_N_gu * in0_block_w_gu tiles of
//     gate_proj to cb_in1_gate.
//   - Phase 2 (up matmul): re-stream x and stream up_proj.
//   - WAIT on a global semaphore until it reaches `total_cores`. The writer
//     kernel of each core increments the semaphore once it's done draining
//     cb_activated to the per-program DRAM scratch tensor, so once the
//     semaphore equals total_cores every core's M-rows × hidden columns of
//     activated are coherent in scratch.
//   - Phase 4 (down matmul): per K-block, read this core's per_core_M rows
//     × in0_block_w_d columns from the activated scratch into
//     cb_in0_down_full, and stream the matching in1 K-block of down_proj
//     into cb_in1_down.

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
    const uint32_t scratch_addr = get_arg_val<uint32_t>(6);  // activated scratch DRAM tensor
    const uint32_t sem_id = get_arg_val<uint32_t>(7);        // semaphore id (resolves to L1 addr via get_semaphore)
    const uint32_t sem_addr = get_semaphore(sem_id);

    const uint32_t my_mt = get_arg_val<uint32_t>(8);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(9);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(10);
    const uint32_t chunk_start_tile_row = get_arg_val<uint32_t>(11);

    // Weight-multicast runtime args (indices 12..21).
    const uint32_t is_in1_sender_u32 = get_arg_val<uint32_t>(12);
    const bool is_in1_sender = is_in1_sender_u32 != 0;
    const uint32_t in1_ready_sem_id = get_arg_val<uint32_t>(13);
    const uint32_t in1_valid_sem_id = get_arg_val<uint32_t>(14);
    const uint32_t in1_num_receivers = get_arg_val<uint32_t>(15);
    const uint32_t in1_mcast_nx_start = get_arg_val<uint32_t>(16);
    const uint32_t in1_mcast_ny_start = get_arg_val<uint32_t>(17);
    const uint32_t in1_mcast_nx_end = get_arg_val<uint32_t>(18);
    const uint32_t in1_mcast_ny_end = get_arg_val<uint32_t>(19);
    const uint32_t in1_sender_nx = get_arg_val<uint32_t>(20);
    const uint32_t in1_sender_ny = get_arg_val<uint32_t>(21);
    const uint32_t in1_ready_sem_addr = get_semaphore(in1_ready_sem_id);
    const uint32_t in1_valid_sem_addr = get_semaphore(in1_valid_sem_id);

    // -------------------------- compile-time args -------------------------
    constexpr uint32_t cb_in0_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in1_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1_down = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in0_down_full = get_compile_time_arg_val(4);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(5);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(6);

    constexpr uint32_t local_expert_id = get_compile_time_arg_val(7);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(8);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(9);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(10);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t K_down_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(13);
    constexpr uint32_t in0_block_w_d = get_compile_time_arg_val(14);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(15);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(16);
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(17);
    constexpr uint32_t total_cores = get_compile_time_arg_val(18);

    constexpr uint32_t g_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;
    constexpr uint32_t d_in0_block_num_tiles = per_core_M * in0_block_w_d;
    constexpr uint32_t d_in1_block_num_tiles = per_core_N_d * in0_block_w_d;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t num_blocks_d = K_down_tiles / in0_block_w_d;

    constexpr uint32_t x_accessor_offset = 19;
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

    constexpr uint32_t scratch_accessor_offset = idx_args.next_compile_time_args_offset();
    constexpr auto scratch_args = TensorAccessorArgs<scratch_accessor_offset>();
    const auto scratch_acc = TensorAccessor(scratch_args, scratch_addr, get_tile_size(cb_in0_down_full));

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
    (void)count_value;
    (void)M_tiles_full;

    const uint32_t this_core_first_row = chunk_start_tile_row + my_mt * per_core_M;

    const uint32_t x_tile_bytes = get_tile_size(cb_in0_x);
    const uint32_t gate_tile_bytes = get_tile_size(cb_in1_gate);
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    const uint32_t down_tile_bytes = get_tile_size(cb_in1_down);
    const uint32_t scratch_tile_bytes = get_tile_size(cb_in0_down_full);

    // Weight-multicast helper. For each in1 K-block:
    //   * Sender (gy=0): wait for all GRID_Y-1 receivers to inc the local
    //     ready_sem. Reset ready_sem. Read in1 from DRAM into local cb_in1.
    //     Multicast the L1 region to receivers. Multicast valid_sem=1.
    //   * Receiver: reserve cb space. Reset local valid_sem=0. Increment
    //     sender's ready_sem at sender's NoC coord. Wait local valid_sem=1.
    //
    // Both sender and receiver finish with the K-block of in1 in their own
    // cb_in1 L1, ready for cb_push_back/compute.
    constexpr uint32_t IN1_VALID = 1;
    volatile tt_l1_ptr uint32_t* in1_ready_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* in1_valid_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_valid_sem_addr);
    const uint64_t in1_sender_ready_noc = get_noc_addr(in1_sender_nx, in1_sender_ny, in1_ready_sem_addr);
    const uint64_t in1_mcast_valid_noc = get_noc_multicast_addr(
        in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, in1_valid_sem_addr);

    // -------- PHASE 1 — gate matmul feed (push x + gate_proj K-blocks) ----
    for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
        // x: every core reads its own M slice (no x mcast yet).
        cb_reserve_back(cb_in0_x, g_in0_block_num_tiles);
        uint32_t l1_x = get_write_ptr(cb_in0_x);
        for (uint32_t m = 0; m < per_core_M; ++m) {
            for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                const uint32_t row = this_core_first_row + m;
                const uint32_t col = kb * in0_block_w_gu + k;
                const uint32_t tile_idx = row * K_gate_tiles + col;
                noc_async_read_tile(tile_idx, x_acc, l1_x);
                l1_x += x_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0_x, g_in0_block_num_tiles);

        // in1_gate via multicast within N-col group.
        cb_reserve_back(cb_in1_gate, g_in1_block_num_tiles);
        if (is_in1_sender) {
            // Wait until all receivers have signalled "ready".
            noc_semaphore_wait(in1_ready_local, in1_num_receivers);
            *in1_ready_local = 0;

            // DRAM read the K-block into our local cb_in1_gate.
            uint32_t l1_w = get_write_ptr(cb_in1_gate);
            const uint32_t block_start = l1_w;
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

            // Multicast block to all receivers' cb_in1_gate L1 (same offset).
            const uint64_t mcast_data_noc = get_noc_multicast_addr(
                in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, block_start);
            const uint32_t block_bytes = g_in1_block_num_tiles * gate_tile_bytes;
            noc_async_write_multicast(block_start, mcast_data_noc, block_bytes, in1_num_receivers, /*linked=*/false);
            noc_async_writes_flushed();

            // Multicast valid=1 to receivers.
            *in1_valid_local = IN1_VALID;
            noc_semaphore_set_multicast(in1_valid_sem_addr, in1_mcast_valid_noc, in1_num_receivers);
        } else {
            // Reset local valid sem before signalling sender.
            *in1_valid_local = 0;

            // Signal sender we're ready to receive.
            noc_semaphore_inc(in1_sender_ready_noc, 1);

            // Wait for sender to mcast data + valid signal.
            noc_semaphore_wait(in1_valid_local, IN1_VALID);
        }
        cb_push_back(cb_in1_gate, g_in1_block_num_tiles);
    }

    // -------- PHASE 2 — up matmul feed (re-stream x + up_proj K-blocks via mcast) ---
    // up_proj uses the SAME mcast topology as gate_proj (N-col groups).
    // ready_sem / valid_sem from phase 1 are already at 0 (sender reset
    // ready, receivers reset valid before sending the last "ready" inc).
    for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
        cb_reserve_back(cb_in0_x, g_in0_block_num_tiles);
        uint32_t l1_x = get_write_ptr(cb_in0_x);
        for (uint32_t m = 0; m < per_core_M; ++m) {
            for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                const uint32_t row = this_core_first_row + m;
                const uint32_t col = kb * in0_block_w_gu + k;
                const uint32_t tile_idx = row * K_gate_tiles + col;
                noc_async_read_tile(tile_idx, x_acc, l1_x);
                l1_x += x_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0_x, g_in0_block_num_tiles);

        // in1_up via multicast within N-col group (same topology as gate).
        cb_reserve_back(cb_in1_up, g_in1_block_num_tiles);
        if (is_in1_sender) {
            noc_semaphore_wait(in1_ready_local, in1_num_receivers);
            *in1_ready_local = 0;

            uint32_t l1_w = get_write_ptr(cb_in1_up);
            const uint32_t block_start = l1_w;
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

            const uint64_t mcast_data_noc = get_noc_multicast_addr(
                in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, block_start);
            const uint32_t block_bytes = g_in1_block_num_tiles * up_tile_bytes;
            noc_async_write_multicast(block_start, mcast_data_noc, block_bytes, in1_num_receivers, /*linked=*/false);
            noc_async_writes_flushed();

            *in1_valid_local = IN1_VALID;
            noc_semaphore_set_multicast(in1_valid_sem_addr, in1_mcast_valid_noc, in1_num_receivers);
        } else {
            *in1_valid_local = 0;
            noc_semaphore_inc(in1_sender_ready_noc, 1);
            noc_semaphore_wait(in1_valid_local, IN1_VALID);
        }
        cb_push_back(cb_in1_up, g_in1_block_num_tiles);
    }

    // Wait for the controller writer to broadcast "go" (sem == 1).
    {
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
        noc_semaphore_wait(sem_ptr, 1);
    }
    (void)total_cores;

    // -------- PHASE 4 — down matmul feed (activated scratch + down_proj) --
    for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
        cb_reserve_back(cb_in0_down_full, d_in0_block_num_tiles);
        uint32_t l1_a = get_write_ptr(cb_in0_down_full);
        for (uint32_t m = 0; m < per_core_M; ++m) {
            for (uint32_t k = 0; k < in0_block_w_d; ++k) {
                const uint32_t row = this_core_first_row + m;
                const uint32_t col = kb * in0_block_w_d + k;
                const uint32_t tile_idx = row * N_gate_tiles_full + col;  // scratch is M_full x N_gate
                noc_async_read_tile(tile_idx, scratch_acc, l1_a);
                l1_a += scratch_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0_down_full, d_in0_block_num_tiles);

        // in1_down via multicast within N-col group. Note: per_core_N_d is
        // larger than per_core_N_gu (28 vs 8), so the block is much bigger
        // — this phase benefits most from mcast.
        cb_reserve_back(cb_in1_down, d_in1_block_num_tiles);
        if (is_in1_sender) {
            noc_semaphore_wait(in1_ready_local, in1_num_receivers);
            *in1_ready_local = 0;

            uint32_t l1_w = get_write_ptr(cb_in1_down);
            const uint32_t block_start = l1_w;
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

            const uint64_t mcast_data_noc = get_noc_multicast_addr(
                in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, block_start);
            const uint32_t block_bytes = d_in1_block_num_tiles * down_tile_bytes;
            noc_async_write_multicast(block_start, mcast_data_noc, block_bytes, in1_num_receivers, /*linked=*/false);
            noc_async_writes_flushed();

            *in1_valid_local = IN1_VALID;
            noc_semaphore_set_multicast(in1_valid_sem_addr, in1_mcast_valid_noc, in1_num_receivers);
        } else {
            *in1_valid_local = 0;
            noc_semaphore_inc(in1_sender_ready_noc, 1);
            noc_semaphore_wait(in1_valid_local, IN1_VALID);
        }
        cb_push_back(cb_in1_down, d_in1_block_num_tiles);
    }
}
