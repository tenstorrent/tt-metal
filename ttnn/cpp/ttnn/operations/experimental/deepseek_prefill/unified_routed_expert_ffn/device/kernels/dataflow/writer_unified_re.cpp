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
    const uint32_t sem_addr = get_arg_val<uint32_t>(2);  // L1 offset of the semaphore on every core
    const uint32_t num_sync_cores = get_arg_val<uint32_t>(3);
    // num_sync_cores pairs of (sem_core_x, sem_core_y) follow at args 4 ..
    // (4 + 2*num_sync_cores - 1). Each writer NoC-increments every core's
    // slot so every reader's noc_semaphore_wait(sem == total_cores) finishes
    // exactly once all writers have hit this point.
    const uint32_t sync_coords_base = 4;
    const uint32_t after_coords_base = sync_coords_base + 2 * num_sync_cores;
    const uint32_t my_mt = get_arg_val<uint32_t>(after_coords_base + 0);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(after_coords_base + 1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(after_coords_base + 2);
    const uint32_t chunk_start_tile_row = get_arg_val<uint32_t>(after_coords_base + 3);

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

    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;

    constexpr uint32_t out_accessor_offset = 11;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    constexpr uint32_t scratch_accessor_offset = out_args.next_compile_time_args_offset();
    constexpr auto scratch_args = TensorAccessorArgs<scratch_accessor_offset>();
    const auto scratch_acc = TensorAccessor(scratch_args, scratch_addr, get_tile_size(cb_activated));

    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const uint32_t scratch_tile_bytes = get_tile_size(cb_activated);

    // ----- Phase 3 -> scratch ----------------------------------------------
    // Drain cb_activated subblock-by-subblock and write each tile into the
    // scratch DRAM tensor. The compute kernel packs subblocks in row-major
    // order over (in0_sub, in1_sub) where in0_num_subblocks = per_core_M /
    // gu_out_subblock_h and in1_num_subblocks = per_core_N_gu /
    // gu_out_subblock_w. We mirror that ordering.
    {
        const uint32_t gu_in0_num_subblocks = per_core_M / gu_out_subblock_h;
        const uint32_t gu_in1_num_subblocks = per_core_N_gu / gu_out_subblock_w;
        const uint32_t row0 = chunk_start_tile_row + my_mt * per_core_M;
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

    // DIAGNOSTIC: For now, just increment THIS core's own sem so reader on
    // this same core can proceed. Each core syncs only with itself (= local
    // barrier between phases 3 and 4). True cross-core sync (every writer
    // increments every reader's slot) is the path forward; this self-only
    // variant is to isolate whether the NoC sem broadcast pattern is what's
    // wedging the kernel.
    {
        const uint32_t target_x = get_arg_val<uint32_t>(sync_coords_base + 0);
        const uint32_t target_y = get_arg_val<uint32_t>(sync_coords_base + 1);
        const uint64_t remote_sem = get_noc_addr(target_x, target_y, sem_addr);
        noc_semaphore_inc(remote_sem, 1);
    }
    // Suppress unused-var warning until full cross-core sync is wired back.
    (void)num_sync_cores;

    // ----- Phase 4 drain -> DRAM output ------------------------------------
    {
        const uint32_t row0 = chunk_start_tile_row + my_mt * per_core_M;
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
}
