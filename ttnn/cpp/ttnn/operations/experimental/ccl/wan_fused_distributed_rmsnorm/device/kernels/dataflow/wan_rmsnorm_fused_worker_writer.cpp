// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Per-worker writer for the fused Wan2.2 distributed RMSNorm AG (forwarder model).
 *
 * The worker holds NO fabric connection — its forwarder core does. Per row the
 * worker:
 *   1. Takes compute's row-0 transposed stat tile (stats_transposed_local_cb)
 *      and NoC-writes its 128 B stick (two contiguous 64 B face-rows packed
 *      contiguous) into its forwarder's packet_buf[round%2] + slot*128 B, then
 *      increments the forwarder's fwd_arrival_sem.
 *   2. Waits on its own go-sem (forwarder sets it once that round's ring gather
 *      has landed in this chip's DRAM scratch).
 *   3. Reads its ring_size gathered sticks from DRAM (page(d, forwarder, round)
 *      + slot*128 B for each device d) into ROW 0 of stats_transposed_gathered_cb
 *      tiles, and pushes them to compute (which FPU-adds + transpose_wh_dest).
 *   4. Drains the row's output_cb tiles to the output tensor.
 *
 * Also populates compute's reduce-scalar / epsilon / trans_mat CBs up front
 * (shared helper) so the reader starts the input read ASAP.
 *
 * is_tp_1 (ring==1 / per_head_norm) never reaches this kernel — that path keeps
 * stats local in compute and uses the plain drain-only writer.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "wan_rmsnorm_scalar_setup.hpp"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t output_cb = get_compile_time_arg_val(0);
constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t stats_transposed_local_cb = get_compile_time_arg_val(3);
constexpr uint32_t stats_transposed_gathered_cb = get_compile_time_arg_val(4);
constexpr uint32_t ring_size = get_compile_time_arg_val(5);
constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(6);
constexpr uint32_t total_num_tile_rows = get_compile_time_arg_val(7);
constexpr uint32_t max_rounds = get_compile_time_arg_val(8);              // pages per (device,forwarder)
constexpr uint32_t stick_bytes = get_compile_time_arg_val(9);             // 128
constexpr uint32_t num_chunks_per_device = get_compile_time_arg_val(10);  // num_forwarders*max_rounds
// Shared packet CB (created on the whole core grid -> uniform L1 addr, so this
// worker's get_write_ptr(packet_cb) == the forwarder core's packet base) and
// grid-uniform sync sem ids.
constexpr uint32_t packet_cb = get_compile_time_arg_val(11);
constexpr uint32_t arrival_sem_id = get_compile_time_arg_val(12);
constexpr uint32_t go_sem_id = get_compile_time_arg_val(13);
constexpr uint32_t padded_row_tiles = ((num_tile_cols + block_size - 1u) / block_size) * block_size;
// Tile row-0 layout (post transpose_wh): face_00 row0 = bytes [0,64), face_01
// row0 = bytes [1024,1088). 32 fp32 = 128 B real data per stat tile.
constexpr uint32_t kFaceRowBytes = 64u;
constexpr uint32_t kFace01Off = 1024u;

// Scalar/eps/trans_mat population args (after the output + dram accessors).
constexpr auto output_args = TensorAccessorArgs<14>();
constexpr auto stats_dram_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
constexpr uint32_t SCB = stats_dram_args.next_compile_time_args_offset();
constexpr uint32_t w_sum_cb = get_compile_time_arg_val(SCB + 0);
constexpr uint32_t w_avg_cb = get_compile_time_arg_val(SCB + 1);
constexpr uint32_t w_eps_cb = get_compile_time_arg_val(SCB + 2);
constexpr uint32_t w_transmat_cb = get_compile_time_arg_val(SCB + 3);
constexpr uint32_t w_reduce_factor = get_compile_time_arg_val(SCB + 4);
constexpr uint32_t w_eps_bits = get_compile_time_arg_val(SCB + 5);
constexpr uint32_t w_fuse_rope = get_compile_time_arg_val(SCB + 6);
constexpr auto w_transmat_args = TensorAccessorArgs<SCB + 7>();

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t transformation_mat_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stats_dram_addr = get_arg_val<uint32_t>(arg_idx++);
    // Forwarder core NoC coords (which core to write the stick to / inc arrival),
    // plus this worker's per-core forwarder group + slot (runtime, differs per core).
    const uint32_t fwd_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t fwd_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_forwarder_index = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_slot = get_arg_val<uint32_t>(arg_idx++);

    // Grid-uniform: my own packet_cb base == the forwarder's packet base; sem
    // addrs are the same on me and the forwarder.
    const uint32_t fwd_packet_buf_addr = get_write_ptr(packet_cb);
    const uint32_t packet_slot_bytes = get_tile_size(packet_cb);  // unit_packet_bytes (per round%2 slot)
    const uint32_t fwd_arrival_sem_addr = get_semaphore(arrival_sem_id);
    const uint32_t go_sem_addr = get_semaphore(go_sem_id);

    const uint32_t output_tile_bytes = get_tile_size(output_cb);
    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
    const uint32_t gathered_tile_bytes = get_tile_size(stats_transposed_gathered_cb);
    const uint32_t stat_tile_bytes = get_tile_size(stats_transposed_local_cb);

    // Populate compute's scalar/eps/trans_mat CBs before anything else.
    wan_rmsnorm_generate_scalars_and_transmat<
        w_sum_cb,
        w_avg_cb,
        w_eps_cb,
        w_transmat_cb,
        w_reduce_factor,
        static_cast<bool>(w_fuse_rope)>(w_eps_bits, TensorAccessor(w_transmat_args, transformation_mat_addr));

    volatile tt_l1_ptr uint32_t* go_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem_addr);
    const uint64_t fwd_arrival_noc = safe_get_noc_addr(fwd_x, fwd_y, fwd_arrival_sem_addr, 0);

    uint32_t go_target = 0;
    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        const uint32_t round = tile_row - tile_row_start;

        // ---- 1. push my stick into the forwarder's packet_buf, then inc arrival ----
        {
            DeviceZoneScopedN("W_PUSH");
            cb_wait_front(stats_transposed_local_cb, 1);
            const uint32_t src = get_read_ptr(stats_transposed_local_cb);
            const uint32_t dst = fwd_packet_buf_addr + (round & 1u) * packet_slot_bytes + my_slot * stick_bytes;
            const uint64_t dst_noc0 = safe_get_noc_addr(fwd_x, fwd_y, dst, 0);
            const uint64_t dst_noc1 = safe_get_noc_addr(fwd_x, fwd_y, dst + kFaceRowBytes, 0);
            noc_async_write(src, dst_noc0, kFaceRowBytes);               // face_00 row0
            noc_async_write(src + kFace01Off, dst_noc1, kFaceRowBytes);  // face_01 row0
            noc_async_write_barrier();
            noc_semaphore_inc(fwd_arrival_noc, 1);
            noc_async_atomic_barrier();
            cb_pop_front(stats_transposed_local_cb, 1);
        }

        // ---- 2. wait for the forwarder's go (this round's ring gather landed) ----
        {
            DeviceZoneScopedN("W_AGWAIT");
            go_target += 1;
            noc_semaphore_wait_min(go_sem_ptr, go_target);
        }

        // ---- 3. read ring_size gathered sticks from DRAM into ROW 0 of gathered tiles ----
        cb_reserve_back(stats_transposed_gathered_cb, ring_size);
        const uint32_t gbase = get_write_ptr(stats_transposed_gathered_cb);
#ifndef WAN_ABL_SKIP_GATHER_SCATTER
        for (uint32_t d = 0; d < ring_size; d++) {
            const uint32_t page_idx = d * num_chunks_per_device + my_forwarder_index * max_rounds + round;
            const uint32_t tile_dst = gbase + d * gathered_tile_bytes;
            const uint64_t src = get_noc_addr(page_idx, stats_dram, my_slot * stick_bytes);
            noc_async_read(src, tile_dst, kFaceRowBytes);                               // -> face_00 row0
            noc_async_read(src + kFaceRowBytes, tile_dst + kFace01Off, kFaceRowBytes);  // -> face_01 row0
        }
        noc_async_read_barrier();
#endif
        cb_push_back(stats_transposed_gathered_cb, ring_size);

        // ---- 4. drain this row's output_cb tiles ----
        {
            DeviceZoneScopedN("W_DRAIN");
            uint32_t row_base_rd = 0;
            uint32_t cumulative = 0;
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cumulative += block_size;
                cb_wait_front(output_cb, cumulative);
                if (col_tile == 0) {
                    row_base_rd = get_read_ptr(output_cb);
                }
                uint32_t rd = row_base_rd + col_tile * output_tile_bytes;
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t c = col_tile + i;
                    const uint32_t h = c / head_dim_tiles;
                    const uint32_t t_col = c - h * head_dim_tiles;
                    const uint32_t out_idx =
                        h * total_num_tile_rows * head_dim_tiles + tile_row * head_dim_tiles + t_col;
#ifndef WAN_ABL_SKIP_OUTPUT_WRITE
                    noc_async_write_tile(out_idx, output_accessor, rd);
#endif
                    rd += output_tile_bytes;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(output_cb, padded_row_tiles);
        }
    }
    noc_async_write_barrier();
    noc_semaphore_set(go_sem_ptr, 0);
}
