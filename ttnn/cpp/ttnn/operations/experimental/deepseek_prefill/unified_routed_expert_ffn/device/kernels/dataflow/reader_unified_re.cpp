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

    // x (in0) multicast runtime args (indices 22..31).
    const uint32_t is_in0_sender_u32 = get_arg_val<uint32_t>(22);
    const bool is_in0_sender = is_in0_sender_u32 != 0;
    const uint32_t in0_ready_sem_id = get_arg_val<uint32_t>(23);
    const uint32_t in0_valid_sem_id = get_arg_val<uint32_t>(24);
    const uint32_t in0_num_receivers = get_arg_val<uint32_t>(25);
    const uint32_t in0_mcast_nx_start = get_arg_val<uint32_t>(26);
    const uint32_t in0_mcast_ny_start = get_arg_val<uint32_t>(27);
    const uint32_t in0_mcast_nx_end = get_arg_val<uint32_t>(28);
    const uint32_t in0_mcast_ny_end = get_arg_val<uint32_t>(29);
    const uint32_t in0_sender_nx = get_arg_val<uint32_t>(30);
    const uint32_t in0_sender_ny = get_arg_val<uint32_t>(31);
    const uint32_t in0_ready_sem_addr = get_semaphore(in0_ready_sem_id);
    const uint32_t in0_valid_sem_addr = get_semaphore(in0_valid_sem_id);

    // done_sem: legacy from the DRAM-scratch barrier — not used after the L1
    // mcast switch.
    const uint32_t done_sem_id = get_arg_val<uint32_t>(32);
    const uint32_t done_sem_addr = get_semaphore(done_sem_id);
    (void)done_sem_id;
    (void)done_sem_addr;

    // Activated L1 mcast sems. Sender (gx == kb at phase-4 K-block kb) waits
    // on its act_ready_sem for GRID_X - 1 incs from the 7 receivers; then
    // mcasts cb_activated -> all M-row cores' cb_in0_down_full L1; then
    // mcasts act_valid_sem to release receivers.
    const uint32_t act_ready_sem_id = get_arg_val<uint32_t>(33);
    const uint32_t act_valid_sem_id = get_arg_val<uint32_t>(34);
    const uint32_t act_ready_sem_addr = get_semaphore(act_ready_sem_id);
    const uint32_t act_valid_sem_addr = get_semaphore(act_valid_sem_id);

    // M-row NoC coord table: 8 (x, y) pairs at runtime args 35..50. Used to
    // resolve the sender's NoC addr per phase-4 K-block kb (= gx).
    constexpr uint32_t M_ROW_NOC_RT_OFFSET = 35;

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
    constexpr uint32_t num_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(20);
    constexpr uint32_t cb_activated = get_compile_time_arg_val(21);
    constexpr uint32_t GRID_X_NOC = get_compile_time_arg_val(22);  // M-row mcast group size (11 in v2)
    constexpr uint32_t K_down_tiles_padded = get_compile_time_arg_val(23);

    constexpr uint32_t g_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;
    constexpr uint32_t d_in0_block_num_tiles = per_core_M * in0_block_w_d;
    constexpr uint32_t d_in1_block_num_tiles = per_core_N_d * in0_block_w_d;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t num_blocks_d = K_down_tiles_padded / in0_block_w_d;

    constexpr uint32_t x_accessor_offset = 24;
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
    // Reserve+read+push so the compute kernel (TRISC) and writer kernel
    // (NCRISC) can cb_wait_front on these CBs and read the same L1 data.
    cb_reserve_back(cb_counts_scratch, 1);
    cb_reserve_back(cb_idx_scratch, 1);
    const uint32_t counts_l1 = get_write_ptr(cb_counts_scratch);
    const uint32_t idx_l1 = get_write_ptr(cb_idx_scratch);
    noc_async_read_page(0, counts_acc, counts_l1);
    noc_async_read_page(0, idx_acc, idx_l1);
    noc_async_read_barrier();
    cb_push_back(cb_counts_scratch, 1);
    cb_push_back(cb_idx_scratch, 1);

    const volatile tt_l1_ptr uint32_t* counts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    const uint32_t global_expert_id = idx_ptr[local_expert_id];
    const uint32_t count_value = counts_ptr[global_expert_id];
    // count_value is in TOKEN rows. Convert to tile rows (ceil) then to chunks.
    // For count=0 the loop is empty (no chunks processed). For count > 0 we
    // process ceil(count_tiles / chunk_M_tiles) chunks; the remaining chunks
    // (if any) are skipped — no DRAM reads, no mcasts, no compute.
    const uint32_t count_tiles = (count_value + 31) / 32;
    const uint32_t effective_chunks_runtime = (count_tiles + chunk_M_tiles - 1) / chunk_M_tiles;
    // Clamp to compile-time num_chunks just in case (defensive against bad input).
    const uint32_t effective_chunks = effective_chunks_runtime < num_chunks ? effective_chunks_runtime : num_chunks;
    (void)M_tiles_full;

    // Per-chunk M-row base: recomputed inside the chunk loop below.
    (void)chunk_start_tile_row;  // legacy runtime arg, base is now derived per chunk

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

    constexpr uint32_t IN0_VALID = 1;
    volatile tt_l1_ptr uint32_t* in0_ready_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* in0_valid_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_valid_sem_addr);
    const uint64_t in0_sender_ready_noc = get_noc_addr(in0_sender_nx, in0_sender_ny, in0_ready_sem_addr);
    const uint64_t in0_mcast_valid_noc = get_noc_multicast_addr(
        in0_mcast_nx_start, in0_mcast_ny_start, in0_mcast_nx_end, in0_mcast_ny_end, in0_valid_sem_addr);

    // Bound the chunk loop by effective_chunks (= ceil_div(count, chunk_M_tiles))
    // so this expert only does work proportional to its actual token count,
    // not the max-tokens-padded shape of the input. Eliminates the host-side
    // count read that previously had to narrow the input tensor.
    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        const uint32_t this_core_first_row = chunk * chunk_M_tiles + my_mt * per_core_M;

        // -------- PHASES 1+2 fused — push x ONCE per K-block, then gate then up.
        // Compute does both matmuls per K-block sharing the same x. This halves
        // the x DRAM mcast bytes vs sequential phases.
        for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
            // x via multicast in M-row direction.
            cb_reserve_back(cb_in0_x, g_in0_block_num_tiles);
            if (is_in0_sender) {
                noc_semaphore_wait(in0_ready_local, in0_num_receivers);
                *in0_ready_local = 0;

                uint32_t l1_x = get_write_ptr(cb_in0_x);
                const uint32_t block_start = l1_x;
                for (uint32_t m = 0; m < per_core_M; ++m) {
                    for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                        const uint32_t row = this_core_first_row + m;
                        const uint32_t col = kb * in0_block_w_gu + k;
                        const uint32_t tile_idx = row * K_gate_tiles + col;
                        noc_async_read_tile(tile_idx, x_acc, l1_x, /*offset=*/0, /*noc=*/0);
                        l1_x += x_tile_bytes;
                    }
                }
                noc_async_read_barrier(/*noc=*/0);

                const uint64_t mcast_data_noc = get_noc_multicast_addr(
                    in0_mcast_nx_start, in0_mcast_ny_start, in0_mcast_nx_end, in0_mcast_ny_end, block_start);
                const uint32_t block_bytes = g_in0_block_num_tiles * x_tile_bytes;
                noc_async_write_multicast(
                    block_start, mcast_data_noc, block_bytes, in0_num_receivers, /*linked=*/false);
                noc_async_writes_flushed();

                *in0_valid_local = IN0_VALID;
                noc_semaphore_set_multicast(in0_valid_sem_addr, in0_mcast_valid_noc, in0_num_receivers);
            } else {
                *in0_valid_local = 0;
                noc_semaphore_inc(in0_sender_ready_noc, 1);
                noc_semaphore_wait(in0_valid_local, IN0_VALID);
            }
            cb_push_back(cb_in0_x, g_in0_block_num_tiles);

            // in1_gate AND in1_up via a SINGLE mcast handshake. We issue two
            // back-to-back NoC writes (gate L1 region, then up L1 region) but
            // share one ready/valid sem pair. Halves in1 mcast handshake count
            // for the fused phases 1+2 (from 28 to 14 sem handshakes).
            cb_reserve_back(cb_in1_gate, g_in1_block_num_tiles);
            cb_reserve_back(cb_in1_up, g_in1_block_num_tiles);
            if (is_in1_sender) {
                noc_semaphore_wait(in1_ready_local, in1_num_receivers);
                *in1_ready_local = 0;

                // DRAM read gate region first.
                uint32_t l1_w_gate = get_write_ptr(cb_in1_gate);
                const uint32_t gate_block_start = l1_w_gate;
                for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                    for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                        const uint32_t row = kb * in0_block_w_gu + k;
                        const uint32_t col = my_nt_gu * per_core_N_gu + n;
                        if (col < N_gate_tiles_full) {
                            const uint32_t tile_idx = row * N_gate_tiles_full + col;
                            noc_async_read_tile(tile_idx, gate_acc, l1_w_gate, /*offset=*/0, /*noc=*/0);
                        } else {
                            volatile tt_l1_ptr uint64_t* p = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w_gate);
                            for (uint32_t i = 0; i < gate_tile_bytes / 8; ++i) {
                                p[i] = 0;
                            }
                        }
                        l1_w_gate += gate_tile_bytes;
                    }
                }
                // DRAM read up region (queued behind gate reads on the same NoC).
                uint32_t l1_w_up = get_write_ptr(cb_in1_up);
                const uint32_t up_block_start = l1_w_up;
                for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                    for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                        const uint32_t row = kb * in0_block_w_gu + k;
                        const uint32_t col = my_nt_gu * per_core_N_gu + n;
                        if (col < N_gate_tiles_full) {
                            const uint32_t tile_idx = row * N_gate_tiles_full + col;
                            noc_async_read_tile(tile_idx, up_acc, l1_w_up, /*offset=*/0, /*noc=*/0);
                        } else {
                            volatile tt_l1_ptr uint64_t* p = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w_up);
                            for (uint32_t i = 0; i < up_tile_bytes / 8; ++i) {
                                p[i] = 0;
                            }
                        }
                        l1_w_up += up_tile_bytes;
                    }
                }
                noc_async_read_barrier(/*noc=*/0);

                // Issue TWO mcast writes back-to-back on NoC 1 (default).
                const uint64_t gate_mcast_noc = get_noc_multicast_addr(
                    in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, gate_block_start);
                const uint32_t gate_block_bytes = g_in1_block_num_tiles * gate_tile_bytes;
                noc_async_write_multicast(
                    gate_block_start, gate_mcast_noc, gate_block_bytes, in1_num_receivers, /*linked=*/false);

                const uint64_t up_mcast_noc = get_noc_multicast_addr(
                    in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, up_block_start);
                const uint32_t up_block_bytes = g_in1_block_num_tiles * up_tile_bytes;
                noc_async_write_multicast(
                    up_block_start, up_mcast_noc, up_block_bytes, in1_num_receivers, /*linked=*/false);

                noc_async_writes_flushed();

                // ONE valid mcast covers both gate and up data — receivers
                // wait once and find both pushed.
                *in1_valid_local = IN1_VALID;
                noc_semaphore_set_multicast(in1_valid_sem_addr, in1_mcast_valid_noc, in1_num_receivers);
            } else {
                *in1_valid_local = 0;
                noc_semaphore_inc(in1_sender_ready_noc, 1);
                noc_semaphore_wait(in1_valid_local, IN1_VALID);
            }
            cb_push_back(cb_in1_gate, g_in1_block_num_tiles);
            cb_push_back(cb_in1_up, g_in1_block_num_tiles);
        }

        (void)total_cores;
        (void)sem_addr;
        (void)scratch_acc;
        (void)scratch_tile_bytes;

        // -------- PHASE 4 — down matmul feed via L1 mcast of activated + down weight mcast --
        //
        // For each K-block kb (0..num_blocks_d-1=7), exactly one core in this
        // M-row is the "activated sender": the core at gx == kb. Its
        // cb_activated holds the per_core_M x per_core_N_gu tiles whose
        // hidden-column range matches K-block kb. Sender mcasts those tiles
        // (with loopback to its own L1) to every M-row core's cb_in0_down_full.
        //
        // We compute the mcast destination rectangle from the M-row NoC table
        // once: corners are mrow[0] (top-left) and mrow[GRID_X-1] (bottom-right).
        // Sender NoC addr for the per-K-block ready-sem inc is looked up by
        // index kb from the same table.
        // GRID_X_NOC comes from compile-time arg now (= GRID_X in program factory).
        const uint32_t mrow_first_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 0);
        const uint32_t mrow_first_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 1);
        const uint32_t mrow_last_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 0);
        const uint32_t mrow_last_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * (GRID_X_NOC - 1) + 1);
        constexpr uint32_t ACT_VALID = 1;
        volatile tt_l1_ptr uint32_t* act_ready_local =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_ready_sem_addr);
        volatile tt_l1_ptr uint32_t* act_valid_local =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_valid_sem_addr);
        const uint32_t intermed_tile_bytes = get_tile_size(cb_in0_down_full);

        for (uint32_t kb = 0; kb < num_blocks_d; ++kb) {
            const bool is_act_sender = (my_nt_d == kb);

            // Pre-stage in1_down: receivers ack ready early, sender issues
            // DRAM reads asynchronously so the read traffic overlaps with the
            // activated L1 mcast that follows. Without this prefetch, in1_down
            // sender's DRAM read (~30us) and activated mcast (~15us) ran
            // back-to-back per K-block (11 K-blocks * 30us = 330us serialized
            // DRAM time). With async reads here, the in1_down read overlaps
            // the activated mcast NoC writes.
            cb_reserve_back(cb_in1_down, d_in1_block_num_tiles);
            uint32_t in1_block_start = 0;
            if (is_in1_sender) {
                noc_semaphore_wait(in1_ready_local, in1_num_receivers);
                *in1_ready_local = 0;
                uint32_t l1_w = get_write_ptr(cb_in1_down);
                in1_block_start = l1_w;
                for (uint32_t k = 0; k < in0_block_w_d; ++k) {
                    for (uint32_t n = 0; n < per_core_N_d; ++n) {
                        const uint32_t row = kb * in0_block_w_d + k;
                        const uint32_t col = my_nt_d * per_core_N_d + n;
                        if (row < K_down_tiles && col < N_down_tiles_full) {
                            const uint32_t tile_idx = row * N_down_tiles_full + col;
                            // DRAM read on NoC 0 (BRISC default is NoC 1) so the
                            // read traffic runs on a separate NoC channel from the
                            // activated mcast that follows on NoC 1.
                            noc_async_read_tile(tile_idx, down_acc, l1_w, /*offset=*/0, /*noc=*/0);
                        } else {
                            volatile tt_l1_ptr uint64_t* p = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w);
                            for (uint32_t i = 0; i < down_tile_bytes / 8; ++i) {
                                p[i] = 0;
                            }
                        }
                        l1_w += down_tile_bytes;
                    }
                }
                // DON'T barrier yet — reads continue (on NoC 0) while we do
                // activated mcast (on NoC 1).
            } else {
                *in1_valid_local = 0;
                noc_semaphore_inc(in1_sender_ready_noc, 1);
                // DON'T wait_valid yet — wait after activated mcast.
            }

            // Activated L1 mcast (runs concurrently with in1_down DRAM read).
            cb_reserve_back(cb_in0_down_full, d_in0_block_num_tiles);
            if (is_act_sender) {
                cb_wait_front(cb_activated, d_in0_block_num_tiles);
                noc_semaphore_wait(act_ready_local, GRID_X_NOC - 1);
                *act_ready_local = 0;

                const uint32_t src_l1 = get_read_ptr(cb_activated);
                const uint32_t dst_l1 = get_write_ptr(cb_in0_down_full);
                const uint32_t mcast_bytes = d_in0_block_num_tiles * intermed_tile_bytes;
                const uint64_t data_mcast_noc =
                    get_noc_multicast_addr(mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny, dst_l1);
                noc_async_write_multicast_loopback_src(
                    src_l1, data_mcast_noc, mcast_bytes, GRID_X_NOC, /*linked=*/false);
                noc_async_writes_flushed();

                *act_valid_local = ACT_VALID;
                const uint64_t valid_mcast_noc = get_noc_multicast_addr(
                    mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny, act_valid_sem_addr);
                noc_semaphore_set_multicast_loopback_src(act_valid_sem_addr, valid_mcast_noc, GRID_X_NOC);

                cb_pop_front(cb_activated, d_in0_block_num_tiles);
            } else {
                *act_valid_local = 0;
                const uint32_t sender_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * kb + 0);
                const uint32_t sender_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * kb + 1);
                const uint64_t sender_ready_noc = get_noc_addr(sender_nx, sender_ny, act_ready_sem_addr);
                noc_semaphore_inc(sender_ready_noc, 1);
                noc_semaphore_wait(act_valid_local, ACT_VALID);
            }
            cb_push_back(cb_in0_down_full, d_in0_block_num_tiles);

            // Finish in1_down: sender barriers on the DRAM reads (which have
            // been in flight during the activated mcast), then mcasts.
            if (is_in1_sender) {
                noc_async_read_barrier(/*noc=*/0);  // reads were on NoC 0
                const uint64_t mcast_data_noc = get_noc_multicast_addr(
                    in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, in1_block_start);
                const uint32_t block_bytes = d_in1_block_num_tiles * down_tile_bytes;
                noc_async_write_multicast(
                    in1_block_start, mcast_data_noc, block_bytes, in1_num_receivers, /*linked=*/false);
                noc_async_writes_flushed();

                *in1_valid_local = IN1_VALID;
                noc_semaphore_set_multicast(in1_valid_sem_addr, in1_mcast_valid_noc, in1_num_receivers);
            } else {
                noc_semaphore_wait(in1_valid_local, IN1_VALID);
            }
            cb_push_back(cb_in1_down, d_in1_block_num_tiles);
        }
    }  // end chunk loop
}
