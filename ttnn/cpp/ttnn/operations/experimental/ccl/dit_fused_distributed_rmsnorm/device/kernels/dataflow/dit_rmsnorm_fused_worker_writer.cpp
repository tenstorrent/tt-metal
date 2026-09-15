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
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "dit_rmsnorm_scalar_setup.hpp"
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
// worker's CircularBuffer(packet_cb).get_write_ptr() == the forwarder core's
// packet base) and grid-uniform sync sem ids.
constexpr uint32_t packet_cb = get_compile_time_arg_val(11);
constexpr uint32_t arrival_sem_id = get_compile_time_arg_val(12);
constexpr uint32_t go_sem_id = get_compile_time_arg_val(13);
// Tile row-0 layout (post transpose_wh): face_00 row0 = bytes [0,64), face_01
// row0 = bytes [1024,1088). 32 fp32 = 128 B real data per stat tile.
constexpr uint32_t kFaceRowBytes = 64u;
constexpr uint32_t kFace01Off = 1024u;
// Stats transported per token-tile: 1 for RMSNorm (sum-of-squares), 2 for Welford
// LayerNorm (mean, variance). The physical stick is num_stats * 128 B; each stat is
// one 128 B packed row-0 stick (two 64 B face-rows). num_stats==1 -> RMS layout.
constexpr uint32_t kStatBytes = 128u;
static_assert(stick_bytes % kStatBytes == 0, "stick_bytes must be a whole multiple of the 128 B packed stat stick");
constexpr uint32_t num_stats = stick_bytes / kStatBytes;

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

    Noc noc;

    CircularBuffer cb_packet(packet_cb);
    CircularBuffer cb_output(output_cb);
    CircularBuffer cb_stats_local(stats_transposed_local_cb);
    CircularBuffer cb_stats_gathered(stats_transposed_gathered_cb);

    // Grid-uniform: my own packet_cb base == the forwarder's packet base, and both
    // semaphores resolve to the same L1 offset on me and on the forwarder.
    const uint32_t fwd_packet_buf_addr = cb_packet.get_write_ptr();
    const uint32_t packet_slot_bytes = cb_packet.get_tile_size();  // unit_packet_bytes (per round%2 slot)
    Semaphore<> fwd_arrival_sem(arrival_sem_id);
    Semaphore<> go_sem(go_sem_id);

    const uint32_t output_tile_bytes = cb_output.get_tile_size();
    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const uint32_t output_page_bytes = output_accessor.get_aligned_page_size();
    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
    const uint32_t gathered_tile_bytes = cb_stats_gathered.get_tile_size();
    const uint32_t stat_tile_bytes = cb_stats_local.get_tile_size();

    // Populate compute's scalar/eps/trans_mat CBs before anything else.
    dit_rmsnorm_generate_scalars_and_transmat<
        w_sum_cb,
        w_avg_cb,
        w_eps_cb,
        w_transmat_cb,
        w_reduce_factor,
        static_cast<bool>(w_fuse_rope)>(w_eps_bits, TensorAccessor(w_transmat_args, transformation_mat_addr));

    uint32_t go_target = 0;
    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        const uint32_t round = tile_row - tile_row_start;

        // ---- 1. push my num_stats sticks into the forwarder's packet_buf, inc arrival ----
        // Each stat occupies a 128 B sub-stick at dst + s*128 (mean then variance for
        // LayerNorm); the worker contributes one slot (my_slot*stick_bytes) regardless.
        {
            DeviceZoneScopedN("W_PUSH");
            cb_stats_local.wait_front(num_stats);
            const uint32_t src0 = cb_stats_local.get_read_ptr();
            const uint32_t dst = fwd_packet_buf_addr + (round & 1u) * packet_slot_bytes + my_slot * stick_bytes;
            // The forwarder's packet buffer is a grid-uniform CB, so its address is our own.
            UnicastEndpoint fwd_core;
            for (uint32_t s = 0; s < num_stats; s++) {
                const uint32_t src = src0 + s * stat_tile_bytes;
                const uint32_t sub = dst + s * kStatBytes;
                noc.async_write(  // face_00 row0
                    CoreLocalMem<uint32_t>(src),
                    fwd_core,
                    kFaceRowBytes,
                    {},
                    {.noc_x = fwd_x, .noc_y = fwd_y, .addr = sub});
                noc.async_write(  // face_01 row0
                    CoreLocalMem<uint32_t>(src + kFace01Off),
                    fwd_core,
                    kFaceRowBytes,
                    {},
                    {.noc_x = fwd_x, .noc_y = fwd_y, .addr = sub + kFaceRowBytes});
            }
            noc.async_write_barrier();
            // Arrival handshake: the stick must be visible in the forwarder's packet buffer
            // *before* the forwarder sees the count go up, hence the write barrier above and
            // the atomic barrier below.
            fwd_arrival_sem.up(noc, fwd_x, fwd_y, 1);
            noc.async_atomic_barrier();
            cb_stats_local.pop_front(num_stats);
        }

        // ---- 2. wait for the forwarder's go (this round's ring gather landed) ----
        {
            DeviceZoneScopedN("W_AGWAIT");
            go_target += 1;
            go_sem.wait_min(go_target);
        }

        // ---- 3. read num_stats*ring gathered sticks from DRAM into ROW 0 of gathered tiles ----
        // Device-major, stat-minor order: gathered tile (d*num_stats + s). For LayerNorm
        // this yields interleaved [mean_d, var_d] per device, as combine_welford_partials wants.
        cb_stats_gathered.reserve_back(num_stats * ring_size);
        const uint32_t gbase = cb_stats_gathered.get_write_ptr();
        for (uint32_t d = 0; d < ring_size; d++) {
            const uint32_t page_idx = d * num_chunks_per_device + my_forwarder_index * max_rounds + round;
            for (uint32_t s = 0; s < num_stats; s++) {
                const uint32_t tile_dst = gbase + (d * num_stats + s) * gathered_tile_bytes;
                const uint32_t src_off = my_slot * stick_bytes + s * kStatBytes;
                noc.async_read(  // -> face_00 row0
                    stats_dram,
                    CoreLocalMem<uint32_t>(tile_dst),
                    kFaceRowBytes,
                    {.page_id = page_idx, .offset_bytes = src_off},
                    {});
                noc.async_read(  // -> face_01 row0
                    stats_dram,
                    CoreLocalMem<uint32_t>(tile_dst + kFace01Off),
                    kFaceRowBytes,
                    {.page_id = page_idx, .offset_bytes = src_off + kFaceRowBytes},
                    {});
            }
        }
        noc.async_read_barrier();
        cb_stats_gathered.push_back(num_stats * ring_size);

        // ---- 4. drain this row's output_cb tiles ----
        // Per-block wait + pop (NOT a cumulative wait with a single end-of-row pop):
        // under block_major_post the factory sizes output_cb to just 2*block_size
        // (block-local), NOT the whole row, so a cumulative output_cb.wait_front(
        // 3*block_size...) could never be satisfied — compute can't push a 3rd block
        // into a 2-block CB it never popped → deadlock (this is why is_tp_1 wide, which
        // uses the per-block drain-only writer, worked while TP>1 wide hung). Compute
        // pushes block_size-padded slots per col-block; wait/pop the full block, but
        // only NoC-write the valid tiles. Matches the drain-only writer's drain loop.
        {
            DeviceZoneScopedN("W_DRAIN");
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_output.wait_front(block_size);
                uint32_t rd = cb_output.get_read_ptr();
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    const uint32_t c = col_tile + i;
                    const uint32_t h = c / head_dim_tiles;
                    const uint32_t t_col = c - h * head_dim_tiles;
                    const uint32_t out_idx =
                        h * total_num_tile_rows * head_dim_tiles + tile_row * head_dim_tiles + t_col;
                    noc.async_write(
                        CoreLocalMem<uint32_t>(rd), output_accessor, output_page_bytes, {}, {.page_id = out_idx});
                    rd += output_tile_bytes;
                }
                noc.async_writes_flushed();
                cb_output.pop_front(block_size);
            }
        }
    }
    noc.async_write_barrier();
    // Reset the go-sem for the next invocation. Trace replay re-runs this kernel without
    // re-running host-side semaphore init, so leaving a stale non-zero count here would let
    // the next replay's very first go_sem.wait_min(1) fall straight through.
    go_sem.set(0);
}
