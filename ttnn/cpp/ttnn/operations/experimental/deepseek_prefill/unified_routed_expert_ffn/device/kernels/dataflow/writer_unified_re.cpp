// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Two responsibilities, sequenced in one pass:
//   (1) Phase 3 -> 4 bridge: drain `cb_activated` (the multiply output, full
//       per-core block of per_core_M * per_core_N_gu tiles) into the per-
//       program DRAM scratch tensor at this core's (mt, nt_gu) tile region.
//       After the writes complete, atomically increment a global semaphore
//       so all cores' reader kernels know the scratch is consistent.
//   (2) Phase 4 drain: pop `cb_out` (the down matmul's per-core final block,
//       packed one subblock at a time) and write tiles to the DRAM-
//       interleaved output tensor at this core's (mt, nt_d) tile region.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

constexpr uint32_t TILE_HEIGHT = 32;

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t scratch_addr = get_arg_val<uint32_t>(1);
    const uint32_t sem_id = get_arg_val<uint32_t>(2);  // semaphore id (resolves via get_semaphore)
    const uint32_t sem_addr = get_semaphore(sem_id);
    const uint32_t num_sync_cores = get_arg_val<uint32_t>(3);
    // Multicast destination grid (one rectangle covering every compute core)
    // and the number of dests inside it. After the phase-3 drain, every
    // writer issues a SINGLE multicast atomic increment to bump every core's
    // local sem by 1; the reader spins on its local sem until it reads
    // total_cores, at which point every writer has finished its drain.
    const uint32_t mcast_x_start = get_arg_val<uint32_t>(4);
    const uint32_t mcast_y_start = get_arg_val<uint32_t>(5);
    const uint32_t mcast_x_end = get_arg_val<uint32_t>(6);
    const uint32_t mcast_y_end = get_arg_val<uint32_t>(7);
    const uint32_t my_mt = get_arg_val<uint32_t>(8);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(9);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(10);
    const uint32_t chunk_start_tile_row = get_arg_val<uint32_t>(11);
    (void)chunk_start_tile_row;  // legacy; base derived per chunk below.
    (void)num_sync_cores;

    constexpr uint32_t cb_activated = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(4);
    constexpr uint32_t gu_out_subblock_h = get_compile_time_arg_val(5);
    constexpr uint32_t gu_out_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(9);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(10);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(12);

    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;

    // done_sem runtime arg index. CORE_COORDS_RT_OFFSET sits AFTER done_sem.
    const uint32_t done_sem_id = get_arg_val<uint32_t>(12);
    const uint32_t done_sem_addr = get_semaphore(done_sem_id);

    // NoC virtual coords of every compute core live in runtime args at index
    // CORE_COORDS_RT_OFFSET (interleaved x, y). Used by the controller writer
    // to atomic-inc each core's done_sem slot per chunk.
    constexpr uint32_t CORE_COORDS_RT_OFFSET = 13;

    constexpr uint32_t out_accessor_offset = 13;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    constexpr uint32_t scratch_accessor_offset = out_args.next_compile_time_args_offset();
    constexpr auto scratch_args = TensorAccessorArgs<scratch_accessor_offset>();
    const auto scratch_acc = TensorAccessor(scratch_args, scratch_addr, get_tile_size(cb_activated));

    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const uint32_t scratch_tile_bytes = get_tile_size(cb_activated);

    // done_sem lives at the same offset on every core. We point a volatile
    // pointer at it once and reuse below for both the chunk-top wait and
    // the do_barrier lambda.
    volatile tt_l1_ptr uint32_t* done_sem_ptr_top = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Wait for the PREVIOUS chunk's BARRIER B (phase 4 done on every
        // core). Without this gate a fast core could start chunk's phase 3
        // drain (overwriting scratch row/col slots that share my_mt) while a
        // slow core in the same M-row is still reading them for chunk-1's
        // phase 4. Skipped on chunk 0 (no prior barrier B).
        if (chunk > 0) {
            noc_semaphore_wait(done_sem_ptr_top, 2 * chunk);
        }

        // Drain cb_activated subblock-by-subblock, writing each tile into the
        // scratch DRAM tensor so phase 4's down matmul can pull this core's
        // full K_down = N_gate_tiles_full activated columns back via the
        // reader.
        {
            const uint32_t gu_in0_num_subblocks = per_core_M / gu_out_subblock_h;
            const uint32_t gu_in1_num_subblocks = per_core_N_gu / gu_out_subblock_w;
            const uint32_t row0 = chunk * chunk_M_tiles + my_mt * per_core_M;
            const uint32_t col0 = my_nt_gu * per_core_N_gu;
            for (uint32_t sb_m = 0; sb_m < gu_in0_num_subblocks; ++sb_m) {
                for (uint32_t sb_n = 0; sb_n < gu_in1_num_subblocks; ++sb_n) {
                    cb_wait_front(cb_activated, gu_out_subblock_num_tiles);
                    uint32_t l1_read = get_read_ptr(cb_activated);
                    for (uint32_t i = 0; i < gu_out_subblock_h; ++i) {
                        for (uint32_t j = 0; j < gu_out_subblock_w; ++j) {
                            const uint32_t row = row0 + sb_m * gu_out_subblock_h + i;
                            const uint32_t col = col0 + sb_n * gu_out_subblock_w + j;
                            const uint32_t tile_idx = row * N_gate_tiles_full + col;
                            noc_async_write_tile(tile_idx, scratch_acc, l1_read);
                            l1_read += scratch_tile_bytes;
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_activated, gu_out_subblock_num_tiles);
                }
            }
        }

        // BARRIER A (phase 3 done): tell every core's reader that scratch is
        // consistent so phase 4 reads may begin.
        //
        // We use two sems with monotonically-accumulating chunk-indexed targets
        // so the same sems can survive multiple chunks per program without
        // resets that would race:
        //   * `sem_addr` (a.k.a. ready_sem) lives ONLY on the controller's L1.
        //     Each non-controller atomic-incs it once per barrier; the
        //     controller waits >= target.
        //   * `done_sem_addr` lives on EVERY core's L1. The controller DMA-
        //     writes the new "done" value to every core's slot (including
        //     itself) once it sees ready_sem reach the per-barrier target.
        //
        // Per chunk we issue TWO barriers:
        //   A — "phase 3 done": release readers to start phase 4 reads of
        //     scratch. Target for non-controllers: done_sem >= 2*chunk + 1.
        //   B — "phase 4 done": release ALL writers (this core's NEXT chunk's
        //     phase-3 drain) to start overwriting scratch. Without this barrier
        //     a fast core could start writing chunk N+1's activated into a
        //     scratch slot that a slow core is still reading for chunk N's
        //     phase 4. Target for non-controllers: done_sem >= 2*chunk + 2.
        //
        // Controller-side ready_sem target is 2*(chunk+1)*(N-1) after barrier B
        // of chunk c (it accumulates (N-1) incs from non-controllers per barrier).
        const bool is_controller = (my_x[0] == mcast_x_start) && (my_y[0] == mcast_y_start);
        volatile tt_l1_ptr uint32_t* ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
        volatile tt_l1_ptr uint32_t* done_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);

        auto do_barrier = [&](uint32_t ready_target, uint32_t done_target_value) {
            if (is_controller) {
                noc_semaphore_wait(ready_sem_ptr, ready_target);
                // Stage the new done value locally so the cross-core DMA picks
                // up an L1-coherent source.
                *done_sem_ptr = done_target_value;
                for (uint32_t c = 0; c < num_sync_cores; ++c) {
                    const uint32_t nx = get_arg_val<uint32_t>(CORE_COORDS_RT_OFFSET + 2 * c);
                    const uint32_t ny = get_arg_val<uint32_t>(CORE_COORDS_RT_OFFSET + 2 * c + 1);
                    if (nx == my_x[0] && ny == my_y[0]) {
                        continue;
                    }
                    const uint64_t dst = get_noc_addr(nx, ny, done_sem_addr);
                    noc_async_write(reinterpret_cast<uint32_t>(done_sem_ptr), dst, 4);
                }
                noc_async_write_barrier();
            } else {
                const uint64_t ctrl_ready_noc = get_noc_addr(mcast_x_start, mcast_y_start, sem_addr);
                noc_semaphore_inc(ctrl_ready_noc, 1);
                noc_async_atomic_barrier();
            }
        };

        // BARRIER A: phase 3 done.
        do_barrier(
            /*ready_target=*/(2 * chunk + 1) * (num_sync_cores - 1),
            /*done_target_value=*/2 * chunk + 1);

        // ----- Phase 4 drain -> DRAM output ------------------------------------
        {
            const uint32_t row0 = chunk * chunk_M_tiles + my_mt * per_core_M;
            const uint32_t col0 = my_nt_d * per_core_N_d;
            for (uint32_t sb_m = 0; sb_m < d_in1_num_subblocks_M; ++sb_m) {
                for (uint32_t sb_n = 0; sb_n < d_in1_num_subblocks_N; ++sb_n) {
                    cb_wait_front(cb_out, d_out_subblock_num_tiles);
                    uint32_t l1_read = get_read_ptr(cb_out);
                    for (uint32_t i = 0; i < d_out_subblock_h; ++i) {
                        for (uint32_t j = 0; j < d_out_subblock_w; ++j) {
                            const uint32_t row = row0 + sb_m * d_out_subblock_h + i;
                            const uint32_t col = col0 + sb_n * d_out_subblock_w + j;
                            const uint32_t tile_idx = row * N_down_tiles_full + col;
                            noc_async_write_tile(tile_idx, out_acc, l1_read);
                            l1_read += out_tile_bytes;
                        }
                    }
                    noc_async_write_barrier();
                    cb_pop_front(cb_out, d_out_subblock_num_tiles);
                }
            }
        }

        // BARRIER B: phase 4 done. Without this, a fast core could start
        // chunk+1's phase 3 drain (overwriting scratch row/col slots) while a
        // slow core is still reading those same slots for chunk's phase 4.
        // The last chunk skips B since there's no follow-up phase 3 drain to
        // protect — every reader is already done before the program exits.
        if (chunk + 1 < num_chunks) {
            do_barrier(
                /*ready_target=*/(2 * chunk + 2) * (num_sync_cores - 1),
                /*done_target_value=*/2 * chunk + 2);
        }
    }  // end chunk loop
}
