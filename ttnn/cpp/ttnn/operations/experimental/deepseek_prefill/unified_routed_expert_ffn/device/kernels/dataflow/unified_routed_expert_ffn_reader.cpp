// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for unified_routed_expert_ffn.
//
// Per-core responsibilities, sequenced over `effective_chunks` chunks
// (effective_chunks = ceil(this expert's token count / chunk_M_tiles)):
//   - Read counts/idx_table scratch once at kernel start to discover this
//     expert's active token count. x is the already-extracted per-expert
//     token tensor — tile reads always start at row 0.
//   - Phase 1 (gate matmul, fused with phase 2): per K-block, sender at
//     gx=0 NoC-mcasts the x K-block to its M-row receivers (in0 mcast);
//     sender at gy=0 NoC-mcasts the gate+up K-block to its N-col
//     receivers (in1 mcast). Handshakes are per-K-block ready/valid
//     sems (in0_*_sem, in1_*_sem).
//   - Phase 4 (down matmul): per K-block kb, exactly one core in the
//     M-row (gx==kb) is the "activated sender" — it L1-mcasts its
//     cb_activated tiles to all M-row cores' cb_in0_down_full (with
//     loopback) using act_{ready,valid}_sem. The in1_down sender at
//     gy=0 mcasts the down K-block weight in parallel on the other NoC.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // -------------------------- runtime args ------------------------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t up_addr = get_arg_val<uint32_t>(2);
    const uint32_t down_addr = get_arg_val<uint32_t>(3);
    const uint32_t counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t idx_table_addr = get_arg_val<uint32_t>(5);

    const uint32_t my_mt = get_arg_val<uint32_t>(6);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(7);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(8);

    // Weight-multicast runtime args (indices 9..18).
    const uint32_t is_in1_sender_u32 = get_arg_val<uint32_t>(9);
    const bool is_in1_sender = is_in1_sender_u32 != 0;
    const uint32_t in1_ready_sem_id = get_arg_val<uint32_t>(10);
    const uint32_t in1_valid_sem_id = get_arg_val<uint32_t>(11);
    const uint32_t in1_num_receivers = get_arg_val<uint32_t>(12);
    const uint32_t in1_mcast_nx_start = get_arg_val<uint32_t>(13);
    const uint32_t in1_mcast_ny_start = get_arg_val<uint32_t>(14);
    const uint32_t in1_mcast_nx_end = get_arg_val<uint32_t>(15);
    const uint32_t in1_mcast_ny_end = get_arg_val<uint32_t>(16);
    const uint32_t in1_sender_nx = get_arg_val<uint32_t>(17);
    const uint32_t in1_sender_ny = get_arg_val<uint32_t>(18);
    const uint32_t in1_ready_sem_addr = get_semaphore(in1_ready_sem_id);
    const uint32_t in1_valid_sem_addr = get_semaphore(in1_valid_sem_id);

    // x (in0) multicast runtime args (indices 19..28).
    const uint32_t is_in0_sender_u32 = get_arg_val<uint32_t>(19);
    const bool is_in0_sender = is_in0_sender_u32 != 0;
    const uint32_t in0_ready_sem_id = get_arg_val<uint32_t>(20);
    const uint32_t in0_valid_sem_id = get_arg_val<uint32_t>(21);
    const uint32_t in0_num_receivers = get_arg_val<uint32_t>(22);
    const uint32_t in0_mcast_nx_start = get_arg_val<uint32_t>(23);
    const uint32_t in0_mcast_ny_start = get_arg_val<uint32_t>(24);
    const uint32_t in0_mcast_nx_end = get_arg_val<uint32_t>(25);
    const uint32_t in0_mcast_ny_end = get_arg_val<uint32_t>(26);
    const uint32_t in0_sender_nx = get_arg_val<uint32_t>(27);
    const uint32_t in0_sender_ny = get_arg_val<uint32_t>(28);
    const uint32_t in0_ready_sem_addr = get_semaphore(in0_ready_sem_id);
    const uint32_t in0_valid_sem_addr = get_semaphore(in0_valid_sem_id);

    // Activated L1 mcast sems. Sender (gx == kb at phase-4 K-block kb) waits
    // on its act_ready_sem for GRID_X - 1 incs from the receivers; then
    // mcasts cb_activated -> all M-row cores' cb_in0_down_full L1; then
    // mcasts act_valid_sem to release receivers.
    const uint32_t act_ready_sem_id = get_arg_val<uint32_t>(29);
    const uint32_t act_valid_sem_id = get_arg_val<uint32_t>(30);
    const uint32_t act_ready_sem_addr = get_semaphore(act_ready_sem_id);
    const uint32_t act_valid_sem_addr = get_semaphore(act_valid_sem_id);

    // M-row NoC coord table: GRID_X (x, y) pairs starting at runtime arg 31.
    // Used to resolve the sender's NoC addr per phase-4 K-block kb (= gx).
    constexpr uint32_t M_ROW_NOC_RT_OFFSET = 31;

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
    constexpr uint32_t num_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t cb_activated = get_compile_time_arg_val(20);
    constexpr uint32_t GRID_X_NOC = get_compile_time_arg_val(21);  // M-row mcast group size
    constexpr uint32_t K_down_tiles_padded = get_compile_time_arg_val(22);

    constexpr uint32_t g_in0_block_num_tiles = per_core_M * in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;
    constexpr uint32_t d_in0_block_num_tiles = per_core_M * in0_block_w_d;
    constexpr uint32_t d_in1_block_num_tiles = per_core_N_d * in0_block_w_d;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t num_blocks_d = K_down_tiles_padded / in0_block_w_d;

    constexpr uint32_t x_accessor_offset = 23;
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
    // Reserve+read+push so the compute kernel (TRISC) and writer kernel
    // (NCRISC) can cb_wait_front on these CBs and read the same L1 data.
    //
    // Each scratch CB is a single page sized (host-side) to hold up to
    // MAX_GLOBAL_EXPERTS UINT32 entries, so `1` here is the whole buffer and a
    // single noc_async_read_page lands the entire counts / idx vector. The
    // later counts_ptr[global_expert_id] / idx_ptr[local_expert_id] indexing
    // therefore stays in-bounds for any model up to MAX_GLOBAL_EXPERTS experts
    // (DeepSeek V3 256, Kimi 384, ... up to 1024).
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

    // Per-chunk pre-zero bookkeeping. For each chunk we decide whether
    // THIS core (as in0 sender) needs to zero its cb_in0_x slots before
    // starting the K-loop: it does iff some of the chunk's per_core_M rows
    // are past min(count_tiles, M_tiles_full). The K-loop then SKIPS writes
    // for invalid rows; the pre-zero ensures the slot's invalid-row L1
    // regions are zero across all K-blocks (the K-loop only overwrites
    // valid rows). One pre-zero per chunk replaces the prior per-K-block
    // memset (~14× savings on RISC-V CPU stores).
    //
    // Need to re-pre-zero per chunk because: the L1 carries content from
    // the previous chunk's K-blocks. Multi-chunk cases (e.g. 3.2k with
    // chunk_M=56 num_chunks=2: chunk 0 all-valid, chunk 1 has invalid
    // rows) would otherwise leave chunk 1's invalid rows holding chunk 0's
    // real data — matmul wastes cycles on the garbage even if writer
    // skips the OOB output writes downstream.
    const uint32_t M_bound = (count_tiles < M_tiles_full) ? count_tiles : M_tiles_full;

    const uint32_t x_tile_bytes = get_tile_size(cb_in0_x);
    const uint32_t gate_tile_bytes = get_tile_size(cb_in1_gate);
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    const uint32_t down_tile_bytes = get_tile_size(cb_in1_down);

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

        // Pre-zero both DB slots of cb_in0_x for this chunk IFF this core
        // (as in0 sender) has any M-rows past M_bound. The K-loop below
        // overwrites valid rows but skips writes for invalid rows — those
        // rows must already be zero in L1 to avoid feeding garbage (or
        // leftover chunk-N-1 real data) into the matmul. Fires only on
        // tail chunks of non-aligned M.
        if (is_in0_sender) {
            const uint32_t this_core_last_row = this_core_first_row + per_core_M;
            if (this_core_last_row > M_bound) {
                const uint32_t slot_size_bytes = g_in0_block_num_tiles * get_tile_size(cb_in0_x);
                const uint32_t both_slots_bytes = 2 * slot_size_bytes;
                volatile tt_l1_ptr uint64_t* zero_dst =
                    reinterpret_cast<volatile tt_l1_ptr uint64_t*>(get_write_ptr(cb_in0_x));
                const size_t num_u64_words = both_slots_bytes / sizeof(uint64_t);
                for (size_t word = 0; word < num_u64_words; ++word) {
                    zero_dst[word] = 0;
                }
            }
        }

        // -------- PHASES 1+2 fused — push x ONCE per K-block, then gate then up.
        //
        // Per-K-block restructure for parallelism:
        //   Previously each K-block ran in0 mcast section FULLY then in1 mcast
        //   section FULLY in series on every core. That meant the kernel walked
        //   through every K-block as ~30µs(in0) + ~30µs(in1) = 60µs.
        //   Now we ack BOTH ready semaphores up-front (receivers signal both
        //   senders before either sender starts work), then both senders run
        //   their DRAM-read + NoC-mcast in parallel (in0 sender on M-row,
        //   in1 sender on N-col are disjoint sets of cores except (0,0)).
        //   Receivers wait for BOTH valid semaphores at the end. Halves the
        //   per-K-block elapsed time at small per_core_M where mcast/handshake
        //   overhead dominates compute.
        for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
            cb_reserve_back(cb_in0_x, g_in0_block_num_tiles);
            cb_reserve_back(cb_in1_gate, g_in1_block_num_tiles);
            cb_reserve_back(cb_in1_up, g_in1_block_num_tiles);

            // Step 1: receivers ack BOTH senders upfront so both senders can
            // proceed in parallel. The senders are usually disjoint sets of
            // cores; the only core that's both senders is (0,0) which doesn't
            // self-inc (it's its own sender for both).
            if (!is_in0_sender) {
                *in0_valid_local = 0;
                noc_semaphore_inc(in0_sender_ready_noc, 1);
            }
            if (!is_in1_sender) {
                *in1_valid_local = 0;
                noc_semaphore_inc(in1_sender_ready_noc, 1);
            }

            // Step 2: senders run their work. in0 sender path and in1 sender
            // path can each start as soon as their ready sem is satisfied —
            // for the common case where a core is one type of sender, the
            // work begins immediately. For core (0,0) (both senders), in0
            // runs first then in1, ~60µs sequentially — same as before.
            if (is_in0_sender) {
                noc_semaphore_wait(in0_ready_local, in0_num_receivers);
                *in0_ready_local = 0;

                uint32_t l1_x = get_write_ptr(cb_in0_x);
                const uint32_t block_start = l1_x;
                for (uint32_t m = 0; m < per_core_M; ++m) {
                    const uint32_t row = this_core_first_row + m;
                    // count_tiles is the runtime tile-row count for this expert.
                    // Rows past it are NOT filled by extract — they hold
                    // uninitialized DRAM bytes. Reading them would feed garbage
                    // (potentially NaN/Inf in bf8 representation) into the
                    // matmul, which propagates through the per-K-block L1_ACC
                    // accumulation and contaminates the FFN output. Zero-fill
                    // the L1 region for those rows instead — silu(0) = 0,
                    // 0 * up = 0, 0 @ W_down = 0 (safe and free of NaN).
                    const bool row_valid = row < count_tiles;
                    if (row_valid) {
                        for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                            const uint32_t col = kb * in0_block_w_gu + k;
                            const uint32_t tile_idx = row * K_gate_tiles + col;
                            noc_async_read_page(tile_idx, x_acc, l1_x, /*offset=*/0, /*noc=*/0);
                            l1_x += x_tile_bytes;
                        }
                    } else {
                        // Pre-zero already covered these L1 bytes. Just advance.
                        l1_x += in0_block_w_gu * x_tile_bytes;
                    }
                }
                noc_async_read_barrier(/*noc=*/0);

                const uint64_t mcast_data_noc = get_noc_multicast_addr(
                    in0_mcast_nx_start, in0_mcast_ny_start, in0_mcast_nx_end, in0_mcast_ny_end, block_start);
                const uint32_t block_bytes = g_in0_block_num_tiles * x_tile_bytes;
                noc_async_write_multicast(
                    block_start, mcast_data_noc, block_bytes, in0_num_receivers, /*linked=*/false);
                cb_push_back(cb_in0_x, g_in0_block_num_tiles);

                noc_async_writes_flushed();
                *in0_valid_local = IN0_VALID;
                noc_semaphore_set_multicast(in0_valid_sem_addr, in0_mcast_valid_noc, in0_num_receivers);
            }

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
                            noc_async_read_page(tile_idx, gate_acc, l1_w_gate, /*offset=*/0, /*noc=*/0);
                        } else {
                            volatile tt_l1_ptr uint64_t* p = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w_gate);
                            for (uint32_t i = 0; i < gate_tile_bytes / 8; ++i) {
                                p[i] = 0;
                            }
                        }
                        l1_w_gate += gate_tile_bytes;
                    }
                }
                uint32_t l1_w_up = get_write_ptr(cb_in1_up);
                const uint32_t up_block_start = l1_w_up;
                for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                    for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                        const uint32_t row = kb * in0_block_w_gu + k;
                        const uint32_t col = my_nt_gu * per_core_N_gu + n;
                        if (col < N_gate_tiles_full) {
                            const uint32_t tile_idx = row * N_gate_tiles_full + col;
                            noc_async_read_page(tile_idx, up_acc, l1_w_up, /*offset=*/0, /*noc=*/0);
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

                const uint64_t gate_mcast_noc = get_noc_multicast_addr(
                    in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, gate_block_start);
                const uint32_t gate_block_bytes = g_in1_block_num_tiles * gate_tile_bytes;
                // linked=true on gate, linked=false on up — chains the two
                // mcasts so they share NoC path setup, saving a few cycles
                // of programming overhead per K-block.
                noc_async_write_multicast(
                    gate_block_start, gate_mcast_noc, gate_block_bytes, in1_num_receivers, /*linked=*/true);

                const uint64_t up_mcast_noc = get_noc_multicast_addr(
                    in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, up_block_start);
                const uint32_t up_block_bytes = g_in1_block_num_tiles * up_tile_bytes;
                noc_async_write_multicast(
                    up_block_start, up_mcast_noc, up_block_bytes, in1_num_receivers, /*linked=*/false);

                cb_push_back(cb_in1_gate, g_in1_block_num_tiles);
                cb_push_back(cb_in1_up, g_in1_block_num_tiles);

                noc_async_writes_flushed();

                *in1_valid_local = IN1_VALID;
                noc_semaphore_set_multicast(in1_valid_sem_addr, in1_mcast_valid_noc, in1_num_receivers);
            }

            // Step 3: receivers wait for both valid semaphores and push.
            if (!is_in0_sender) {
                noc_semaphore_wait(in0_valid_local, IN0_VALID);
                cb_push_back(cb_in0_x, g_in0_block_num_tiles);
            }
            if (!is_in1_sender) {
                noc_semaphore_wait(in1_valid_local, IN1_VALID);
                cb_push_back(cb_in1_gate, g_in1_block_num_tiles);
                cb_push_back(cb_in1_up, g_in1_block_num_tiles);
            }
        }

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

            cb_reserve_back(cb_in1_down, d_in1_block_num_tiles);
            cb_reserve_back(cb_in0_down_full, d_in0_block_num_tiles);

            // Step 1: receivers ack BOTH senders (in1_down and act) at the
            // top of the K-block iter. The in1_down ack lets the in1_down
            // sender immediately start DRAM reads; the act ack lets the act
            // sender start mcasting as soon as compute pushes cb_activated.
            // Without the early act ack the sender would only see receivers
            // after the in1_down section finishes, serializing the two paths.
            if (!is_in1_sender) {
                *in1_valid_local = 0;
                noc_semaphore_inc(in1_sender_ready_noc, 1);
            }
            if (!is_act_sender) {
                *act_valid_local = 0;
                const uint32_t sender_nx = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * kb + 0);
                const uint32_t sender_ny = get_arg_val<uint32_t>(M_ROW_NOC_RT_OFFSET + 2 * kb + 1);
                const uint64_t sender_ready_noc = get_noc_addr(sender_nx, sender_ny, act_ready_sem_addr);
                noc_semaphore_inc(sender_ready_noc, 1);
            }

            // Step 2: in1_down sender kicks off DRAM reads (NoC 0) without
            // barriering — reads run concurrently with the activated mcast
            // below on NoC 1.
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
                            noc_async_read_page(tile_idx, down_acc, l1_w, /*offset=*/0, /*noc=*/0);
                        } else {
                            volatile tt_l1_ptr uint64_t* p = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w);
                            for (uint32_t i = 0; i < down_tile_bytes / 8; ++i) {
                                p[i] = 0;
                            }
                        }
                        l1_w += down_tile_bytes;
                    }
                }
            }

            // Step 3: activated L1 mcast (sender for this K-block = gx==kb).
            // act_sender starts as soon as compute pushes cb_activated AND
            // the ready acks are in (already done in step 1).
            if (is_act_sender) {
                cb_wait_front(cb_activated, d_in0_block_num_tiles);
                noc_semaphore_wait(act_ready_local, GRID_X_NOC - 1);
                *act_ready_local = 0;

                const uint32_t src_l1 = get_read_ptr(cb_activated);
                const uint32_t dst_l1 = get_write_ptr(cb_in0_down_full);
                const uint32_t mcast_bytes = d_in0_block_num_tiles * intermed_tile_bytes;
                const uint64_t data_mcast_noc =
                    get_noc_multicast_addr(mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny, dst_l1);
                // linked=true keeps the multicast path RESERVED so the
                // valid-semaphore multicast below travels the SAME path and is
                // delivered AFTER the data at every receiver. With linked=false
                // the path is released and the (posted) valid-sem multicast can
                // overtake the bulk data multicast at a receiver -> the receiver
                // observes act_valid, pushes cb_in0_down_full, and compute reads
                // stale L1 -> that core's whole down-matmul output block is wrong
                // (run-to-run nondeterministic). A write barrier does NOT fix this
                // on Blackhole (multicast writes are posted; no completion ack to
                // wait on) — only path-linking orders the sem behind the data.
                // Mirrors the canonical matmul in0 sender
                // (reader_bmm_tile_layout_in0_sender_padding.cpp).
                noc_async_write_multicast_loopback_src(
                    src_l1, data_mcast_noc, mcast_bytes, GRID_X_NOC, /*linked=*/true);
                noc_async_writes_flushed();

                *act_valid_local = ACT_VALID;
                const uint64_t valid_mcast_noc = get_noc_multicast_addr(
                    mrow_first_nx, mrow_first_ny, mrow_last_nx, mrow_last_ny, act_valid_sem_addr);
                noc_semaphore_set_multicast_loopback_src(act_valid_sem_addr, valid_mcast_noc, GRID_X_NOC);

                cb_pop_front(cb_activated, d_in0_block_num_tiles);
            }

            // Step 4: in1_down sender finishes — barrier on DRAM reads (NoC 0,
            // in flight during step 3 activated mcast on NoC 1), then mcast.
            if (is_in1_sender) {
                noc_async_read_barrier(/*noc=*/0);
                const uint64_t mcast_data_noc = get_noc_multicast_addr(
                    in1_mcast_nx_start, in1_mcast_ny_start, in1_mcast_nx_end, in1_mcast_ny_end, in1_block_start);
                const uint32_t block_bytes = d_in1_block_num_tiles * down_tile_bytes;
                // linked=true so the in1_valid-sem multicast is ordered behind
                // the weight data on the same reserved path (see the activated
                // mcast above for the full rationale).
                noc_async_write_multicast(
                    in1_block_start, mcast_data_noc, block_bytes, in1_num_receivers, /*linked=*/true);
                noc_async_writes_flushed();

                *in1_valid_local = IN1_VALID;
                noc_semaphore_set_multicast(in1_valid_sem_addr, in1_mcast_valid_noc, in1_num_receivers);
            }

            // Step 5: receivers wait for both valid sems and push.
            if (!is_act_sender) {
                noc_semaphore_wait(act_valid_local, ACT_VALID);
            }
            cb_push_back(cb_in0_down_full, d_in0_block_num_tiles);

            if (!is_in1_sender) {
                noc_semaphore_wait(in1_valid_local, IN1_VALID);
            }
            cb_push_back(cb_in1_down, d_in1_block_num_tiles);
        }
    }  // end chunk loop

    // The last in-flight noc_semaphore_set_multicast (act_valid / in1_valid)
    // is a posted atomic; without an explicit barrier it can still be in
    // flight at kernel exit, leading to timing-dependent corruption.
    noc_async_atomic_barrier();
}
