// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Writer for the fused Wan2.2 distributed RMSNorm op.
 *
 * For TP=1 (is_tp_1 == 1):
 *   Just drains output_cb to DRAM. The compute kernel pushes stats directly
 *   into stats_gathered_cb, so no AG work is needed here.
 *
 * For TP>1 (is_tp_1 == 0):
 *   First does a ring AG of per-row stats across the TP cluster axis, then
 *   drains output_cb. Per-core fabric forwarder pattern:
 *     1. For each row this core handles, pull stats_local_cb (compute pushed
 *        a row of partial sum-of-squares), then do a fabric multicast write
 *        + atomic_inc that lands the tile in slot[r * ring_size + my_device_index]
 *        of stats_gathered_cb on every chip in the ring (including locally —
 *        fused_write_atomic_and_advance does a local noc_async_write to the
 *        same dest noc xy).
 *     2. After all rows are written, wait for the GlobalSemaphore on this
 *        core to reach num_tile_rows * (ring_size - 1) — that count of
 *        atomic increments arriving over fabric from the other (ring_size-1)
 *        chips, one per row.
 *     3. Push stats_gathered_cb to compute kernel, which then runs the post
 *        phase reading ring_size stats tiles per row.
 *
 * The pattern was modeled on rms_writer.cpp from the rms_allgather op, but
 * adapted to a fully distributed AG: every worker core runs its own fabric
 * loop for the rows it owns, rather than a single all_to_all_worker writing
 * all the chip's stats. Each chip writes to slot[my_device_index] on every
 * chip — that slot offset is unique per source chip but lives at the same
 * L1 address on every ring chip (since CBs have identical layout across
 * chips running the same program), so a single multicast destination NoC
 * address suffices.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
// Writer always populates compute's reduce-scalar / epsilon / trans_mat CBs
// (shared helper, used by both writers) so the reader starts input reads ASAP.
#include "dit_rmsnorm_scalar_setup.hpp"

void kernel_main() {
    // ---------- compile-time args ----------
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t is_tp_1 = get_compile_time_arg_val(3);
    constexpr uint32_t stats_local_cb = get_compile_time_arg_val(4);
    constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(5);
    constexpr uint32_t reserved_packet_header_cb = get_compile_time_arg_val(6);
    constexpr uint32_t ring_size = get_compile_time_arg_val(7);
    constexpr uint32_t my_device_index = get_compile_time_arg_val(8);
    constexpr uint32_t num_targets_forward = get_compile_time_arg_val(9);
    constexpr uint32_t num_targets_backward = get_compile_time_arg_val(10);
    // Output tensor has shape [1, num_heads_per_device, N, head_dim]; the
    // compute kernel processes the input as if it were [1, 1, N, H_full] where
    // H_full = num_heads_per_device * head_dim. The two tile-page layouts
    // only agree when total_num_tile_rows == 1. Otherwise the writer must
    // map (tile_row, col_tile) → (head, t_row, t_col_in_head) to address the
    // right page in the 4D output.
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t total_num_tile_rows = get_compile_time_arg_val(12);
    constexpr auto output_args = TensorAccessorArgs<13>();

    // Scalar/eps/trans_mat population args (appended after the output accessor).
    // The writer always populates compute's reduce_scalar_* / epsilon /
    // transformation_mat CBs so the reader can start the input read immediately.
    constexpr uint32_t SCB_BASE = output_args.next_compile_time_args_offset();
    constexpr uint32_t reduce_scalar_sum_cb = get_compile_time_arg_val(SCB_BASE + 0);
    constexpr uint32_t reduce_scalar_avg_cb = get_compile_time_arg_val(SCB_BASE + 1);
    constexpr uint32_t epsilon_cb = get_compile_time_arg_val(SCB_BASE + 2);
    constexpr uint32_t transformation_mat_cb = get_compile_time_arg_val(SCB_BASE + 3);
    constexpr uint32_t reduce_factor = get_compile_time_arg_val(SCB_BASE + 4);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(SCB_BASE + 5);
    constexpr uint32_t fuse_rope = get_compile_time_arg_val(SCB_BASE + 6);
    constexpr auto transmat_args = TensorAccessorArgs<SCB_BASE + 7>();

    // ---------- runtime args ----------
    // size_t arg_idx (not uint32_t) so it can bind to FabricConnectionManager::build_from_args
    // which takes the index by reference and advances it past the fabric rt args.
    size_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);
    // trans_mat base address for the writer-side scalar/trans_mat population
    // (only read when fuse_rope). 0 when no RoPE.
    const uint32_t transformation_mat_addr = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t output_tile_bytes = get_tile_size(output_cb);
    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const uint32_t num_tile_rows = tile_row_end - tile_row_start;

    // Populate compute's reduce-scalar / epsilon / trans_mat CBs before any AG
    // or output work — compute blocks on these at its very top. Independent of
    // fabric (uses this writer's own NoC), so it overlaps the fabric handshake.
    dit_rmsnorm_generate_scalars_and_transmat<
        reduce_scalar_sum_cb,
        reduce_scalar_avg_cb,
        epsilon_cb,
        transformation_mat_cb,
        reduce_factor,
        static_cast<bool>(fuse_rope)>(epsilon_bits, TensorAccessor(transmat_args, transformation_mat_addr));

    // =================== TP>1: stats fabric AG ===================
    if constexpr (is_tp_1 == 0) {
        // The next runtime args carry the GlobalSemaphore L1 address + fabric
        // connection setup info. Layout (set by host):
        //   [3]  out_ready_sem_bank_addr  (L1 address of the GlobalSemaphore
        //                                  on this core, same on every chip)
        //   [4]  has_forward_fabric (0/1)
        //   ...  forward fabric connection rt args (if has_forward_fabric)
        //   [N]  has_backward_fabric (0/1)
        //   ...  backward fabric connection rt args (if has_backward_fabric)
        const uint32_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);

        auto fabric_connection =
            FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
                arg_idx);

        // Reserve two packet header slots, one for each direction.
        cb_reserve_back(reserved_packet_header_cb, 1);
        auto pkt_hdr_fwd_addr = get_write_ptr(reserved_packet_header_cb);
        cb_push_back(reserved_packet_header_cb, 1);
        cb_reserve_back(reserved_packet_header_cb, 1);
        auto pkt_hdr_bwd_addr = get_write_ptr(reserved_packet_header_cb);
        cb_push_back(reserved_packet_header_cb, 1);

        volatile PACKET_HEADER_TYPE* pkt_hdr_forward = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_fwd_addr);
        volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_bwd_addr);

        // Unconditionally init both routing headers (matches rms_writer.cpp).
        // num_targets=0 is a valid no-op routing config.
        pkt_hdr_forward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward)});
        pkt_hdr_backward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward)});

        // Guard open_finish on is_logically_connected(), matching the canonical CCL writers.
        if (fabric_connection.is_logically_connected()) {
            fabric_connection.open_finish();
        }

        // Reserve the worker's FULL gathered-stats region up front so the absolute
        // slot addresses base + (r*ring_size + device) are valid for every row
        // (the multicast targets that same L1 address on each ring chip). But
        // RELEASE (push) each row's ring_size slots to compute as soon as that row's
        // gather completes, instead of one push at the end. The compute interleaves
        // PRE/POST per chunk (1 row): it pushes stats_local[r] in chunk r's PRE, then
        // BLOCKS in chunk r's POST on stats_gathered[r]. Pushing only at the end
        // deadlocks at ring_size>1 with >1 tile-row — we'd wait for stats_local[r+1]
        // that compute can't produce until it gets stats_gathered[r] (the LTX-audio
        // TP=2 small-shape hang).
        const uint32_t total_stats_tiles = num_tile_rows * ring_size;
        cb_reserve_back(stats_gathered_cb, total_stats_tiles);
        const uint32_t stats_gathered_base = get_write_ptr(stats_gathered_cb);
        const uint32_t stats_tile_bytes = get_tile_size(stats_gathered_cb);

        // GlobalSemaphore on this core; remote chips atomic_inc it via fabric.
        const uint64_t out_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addr, 0);
        volatile tt_l1_ptr uint32_t* out_ready_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);

        // For each row this core owns: fabric-mcast its stats tile + local write to my
        // slot, then wait for that row's incs and release the row's gathered slots.
        // fused_write_atomic_and_advance_local_read_address_for_fabric_write does both
        // the local noc_async_write and the fabric mcast write+atomic_inc to remote
        // chips, advancing l1_read_addr afterwards.
        uint32_t cumulative_incs = 0;
        for (uint32_t r = 0; r < num_tile_rows; r++) {
            cb_wait_front(stats_local_cb, 1);
            size_t l1_read_addr = get_read_ptr(stats_local_cb);

            const uint32_t my_slot_offset = (r * ring_size + my_device_index) * stats_tile_bytes;
            const uint32_t my_slot_addr = stats_gathered_base + my_slot_offset;
            const uint64_t noc0_dest_noc_addr = safe_get_noc_addr(my_x[0], my_y[0], my_slot_addr, 0);

            // flush=true: order the payload write before the atomic_inc at the receiver,
            // so a satisfied semaphore guarantees the remote slot is committed (without
            // it the sem-gated read can see a stale slot).
            fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                stats_tile_bytes,
                out_ready_sem_noc_addr_in_pkt,
                /*val=*/1,
                /*flush=*/true);

            cb_pop_front(stats_local_cb, 1);

            // Wait for this row's incs (cumulative across rows so far): each of the
            // (ring_size-1) remote chips atomic_inc's once per row. The remote runs the
            // identical per-row loop, so both chips stay in lockstep — no deadlock.
            cumulative_incs += (ring_size - 1);
            if (cumulative_incs > 0) {
                noc_semaphore_wait_min(out_ready_sem_ptr, cumulative_incs);
            }
            // Local write of my own slot for row r must land before compute reads it.
            noc_async_write_barrier();
            // Release this row's ring_size gathered slots so compute's chunk r can run.
            cb_push_back(stats_gathered_cb, ring_size);
        }
        // Reset for any subsequent invocations of this op.
        noc_semaphore_set(out_ready_sem_ptr, 0);

        // Guard close on is_logically_connected(), matching the canonical CCL writers.
        if (fabric_connection.is_logically_connected()) {
            fabric_connection.close_start();
            fabric_connection.close_finish();
        }
    }

    // =================== Drain output_cb to DRAM ===================
    // Output page index for [1, num_heads, N, head_dim] tile (h, t_row, t_col):
    //   page = h * total_num_tile_rows * head_dim_tiles + t_row * head_dim_tiles + t_col
    // The kernel produces tile (tile_row, col_tile + i) where
    //   h = (col_tile + i) / head_dim_tiles
    //   t_col = (col_tile + i) % head_dim_tiles
    // For num_heads_per_device == 1 (head_dim_tiles == num_tile_cols), this
    // collapses to tile_row * num_tile_cols + col_tile + i.
    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
            const uint32_t tiles_in_block =
                ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
            // Compute pushes `block_size` slots per col-block even when only
            // `tiles_in_block` are valid; wait+pop the full block but only
            // NoC-write the valid tiles. Without this, multi-chunk overflows
            // output_cb because over-pushed garbage slots never drain.
            cb_wait_front(output_cb, block_size);
            uint32_t output_rd_ptr = get_read_ptr(output_cb);

            for (uint32_t i = 0; i < tiles_in_block; i++) {
                const uint32_t c = col_tile + i;
                const uint32_t h = c / head_dim_tiles;
                const uint32_t t_col = c - h * head_dim_tiles;
                const uint32_t output_tile_idx =
                    h * total_num_tile_rows * head_dim_tiles + tile_row * head_dim_tiles + t_col;
                noc_async_write_page(output_tile_idx, output_accessor, output_rd_ptr);
                output_rd_ptr += output_tile_bytes;
            }
            // _flushed (write request committed to NoC) instead of _barrier
            // (round-trip ACK). L1 source can be reused once the write has
            // left the core. Final barrier at end of kernel ensures all
            // writes complete before exit.
            noc_async_writes_flushed();
            cb_pop_front(output_cb, block_size);
        }
    }
    // Final barrier — all in-flight output writes must complete before exit.
    noc_async_write_barrier();
}
