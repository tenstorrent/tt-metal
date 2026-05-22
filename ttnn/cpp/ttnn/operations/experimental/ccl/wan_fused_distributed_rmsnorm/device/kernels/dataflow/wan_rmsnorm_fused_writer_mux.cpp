// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Multi-worker MUX writer for the fused Wan2.2 distributed RMSNorm op (TP>1).
 *
 * Same logical job as `wan_rmsnorm_fused_writer.cpp`'s TP>1 path: ring-AG the
 * per-row stats this worker owns, then drain output_cb to DRAM. The
 * difference: this kernel uses the fabric MUX (`WorkerToFabricMuxSender`)
 * instead of opening a dedicated `FabricConnectionManager` channel per
 * worker. That lets many worker cores per chip share a small number of MUX
 * cores (and thus a small number of physical fabric channels), so we can
 * lift the previous single-AG-core restriction and parallelize the compute
 * across N worker cores per chip.
 *
 * Per-worker layout:
 *   - Worker owns rows [tile_row_start, tile_row_end).
 *   - Worker connects to up to two MUX cores: one for forward fabric, one for
 *     backward (whichever direction(s) has a valid neighbor in this topology).
 *   - For each row: local NoC write to my own stats_gathered slot, plus a
 *     fabric multicast fused write+atomic_inc in each valid direction.
 *   - Then waits for `num_tile_rows * (ring_size - 1)` atomic incs on the
 *     per-worker GlobalSemaphore (each remote chip's matching worker writes
 *     to my sem once per row).
 *   - Pushes the gathered stats batch to compute, then drains output_cb.
 *
 * Termination protocol (mirrors all_gather_async's minimal_default_writer):
 *   - Each worker disconnects from its MUX(es).
 *   - The "termination master" worker (one per MUX) waits for (num_mux_clients - 1)
 *     ack incs on a local sem, then sends the graceful-terminate signal to
 *     the MUX. Non-master workers signal the master via that same sem.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

// =============================================================================
// Compile-time args
// =============================================================================
constexpr uint32_t output_cb = get_compile_time_arg_val(0);
constexpr uint32_t num_tile_cols = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t stats_local_cb = get_compile_time_arg_val(3);
constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(4);
constexpr uint32_t ring_size = get_compile_time_arg_val(5);
constexpr uint32_t my_device_index = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward = get_compile_time_arg_val(8);
constexpr uint32_t chunk_size_rows = get_compile_time_arg_val(9);

// MUX CT args (5, in canonical order — matches ccl::fabric_mux_connection_ct_args).
// Same for forward and backward MUX (we use the same FabricMuxConfig for both).
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(10);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(11);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(12);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(13);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(14);

constexpr auto output_args = TensorAccessorArgs<15>();
// Persistent DRAM stats buffer accessor args (Phase 1).
constexpr auto stats_dram_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

// =============================================================================
// Per-direction MUX runtime-arg parsing
// =============================================================================
struct MuxRtArgs {
    bool connection_valid;
    bool is_termination_master;
    uint8_t mux_x;
    uint8_t mux_y;
    uint32_t channel_base_address;
    uint32_t connection_info_address;
    uint32_t connection_handshake_address;
    uint32_t flow_control_address;
    uint32_t buffer_index_address;
    uint8_t channel_id;
    uint32_t termination_sync_addr;
    uint32_t local_fabric_mux_status_addr;
    uint32_t local_flow_control_addr;
    uint32_t local_teardown_addr;
    uint32_t local_buffer_index_addr;
    uint8_t termination_master_noc_x;
    uint8_t termination_master_noc_y;
};

FORCE_INLINE MuxRtArgs parse_mux_rt_args(size_t& arg_idx) {
    MuxRtArgs a;
    a.connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    a.is_termination_master = get_arg_val<uint32_t>(arg_idx++) == 1;
    a.mux_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.mux_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    a.connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    a.connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    a.flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    a.buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    a.channel_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.termination_sync_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_fabric_mux_status_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_flow_control_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_teardown_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.local_buffer_index_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    a.termination_master_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    a.termination_master_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    return a;
}

FORCE_INLINE tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> build_mux_sender(
    const MuxRtArgs& a) {
    return tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        a.mux_x,
        a.mux_y,
        a.channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        a.channel_base_address,
        a.connection_info_address,
        a.connection_handshake_address,
        a.flow_control_address,
        a.buffer_index_address,
        a.local_flow_control_addr,
        a.local_teardown_addr,
        a.local_buffer_index_addr);
}

// Signal-and-or-wait termination handshake on a single MUX. Mirrors
// minimal_default_writer.cpp:662-676 — non-masters atomic_inc the master's
// termination_sync sem; the master waits for (num_mux_clients - 1) incs then
// graceful-terminates the MUX. We collapse this into one helper since both
// directions use identical logic.
FORCE_INLINE void terminate_mux_handshake(const MuxRtArgs& a) {
    if (a.is_termination_master) {
        auto* sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a.termination_sync_addr);
        noc_semaphore_wait(sync_ptr, num_mux_clients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(a.mux_x, a.mux_y, fabric_mux_termination_signal_address);
    } else {
        uint64_t dest_addr =
            safe_get_noc_addr(a.termination_master_noc_x, a.termination_master_noc_y, a.termination_sync_addr, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
}

void kernel_main() {
    // ---------- runtime args ----------
    size_t arg_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    // Persistent DRAM stats buffer (Phase 1). The buffer holds
    // [num_workers, chunk_size_rows, ring_size] fp32 tiles. This worker owns
    // tiles [worker_tile_base, worker_tile_base + chunk_size_rows*ring_size).
    const uint32_t stats_dram_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_tile_base = get_arg_val<uint32_t>(arg_idx++);

    // Two MUX rt blocks: forward first, then backward. Both blocks present
    // always (set by host), with `connection_valid=false` if that direction
    // has no neighbor in this topology (e.g. line endpoints).
    MuxRtArgs fwd_mux_args = parse_mux_rt_args(arg_idx);
    MuxRtArgs bwd_mux_args = parse_mux_rt_args(arg_idx);

    const uint32_t output_tile_bytes = get_tile_size(output_cb);
    const auto output_accessor = TensorAccessor(output_args, output_addr);
    const auto stats_dram_accessor = TensorAccessor(stats_dram_args, stats_dram_addr);
    const uint32_t num_tile_rows = tile_row_end - tile_row_start;

    // ---------- Build + connect MUX senders ----------
    auto fwd_mux_conn = build_mux_sender(fwd_mux_args);
    auto bwd_mux_conn = build_mux_sender(bwd_mux_args);

    if (fwd_mux_args.connection_valid) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fwd_mux_args.mux_x,
            fwd_mux_args.mux_y,
            fabric_mux_status_address,
            fwd_mux_args.local_fabric_mux_status_addr);
    }
    if (bwd_mux_args.connection_valid) {
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            bwd_mux_args.mux_x,
            bwd_mux_args.mux_y,
            fabric_mux_status_address,
            bwd_mux_args.local_fabric_mux_status_addr);
    }
    if (fwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_connect(fwd_mux_conn);
    }
    if (bwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_connect(bwd_mux_conn);
    }

    // ---------- Allocate packet headers ----------
    auto pkt_hdr_forward = PacketHeaderPool::allocate_header();
    auto pkt_hdr_backward = PacketHeaderPool::allocate_header();

    const uint32_t stats_tile_bytes = get_tile_size(stats_gathered_cb);

    // GlobalSemaphore lives on THIS worker core; remote chips' matching workers
    // atomic_inc it via fabric. Compute its NoC-0 addr for the packet headers
    // and a local L1 pointer for our cumulative wait + reset.
    const uint64_t out_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addr, 0);
    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);

    // ---------- Chunked AG + output drain ----------
    // For each chunk: AG that chunk's rows, push stats_gathered for compute's
    // post phase, then drain that chunk's output_cb to DRAM. The sem is NOT
    // reset between chunks — we wait on a cumulative count so concurrent
    // incoming incs for chunk N+1 don't race with chunk N's wait. Reset once
    // at the very end so a subsequent invocation starts at 0.
    uint32_t row_processed = 0;
    uint32_t cumulative_expected_incs = 0;
    while (row_processed < num_tile_rows) {
        const uint32_t rows_in_chunk =
            ((row_processed + chunk_size_rows) <= num_tile_rows) ? chunk_size_rows : (num_tile_rows - row_processed);
        const uint32_t chunk_stats_tiles = rows_in_chunk * ring_size;

        // ---- Phase A: AG into persistent DRAM ----
        for (uint32_t r = 0; r < rows_in_chunk; r++) {
            cb_wait_front(stats_local_cb, 1);
            uint32_t l1_read_addr = get_read_ptr(stats_local_cb);

            // This chunk's row r, this chip's slot in DRAM.
            const uint32_t dram_tile_idx = worker_tile_base + r * ring_size + my_device_index;
            const uint64_t dram_dest_noc_addr = get_noc_addr(dram_tile_idx, stats_dram_accessor);

            // Local DRAM write (this chip's own slot).
            noc_async_write(l1_read_addr, dram_dest_noc_addr, stats_tile_bytes);

            // Fabric mcasts to remote chips' matching DRAM slot. Same tile_idx,
            // same DRAM layout → same NoC0 address on every chip.
            if constexpr (num_targets_forward > 0) {
                if (fwd_mux_args.connection_valid) {
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &fwd_mux_conn,
                        pkt_hdr_forward,
                        l1_read_addr,
                        stats_tile_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            dram_dest_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, false},
                        /*start_distance=*/1,
                        static_cast<uint8_t>(num_targets_forward));
                }
            }
            if constexpr (num_targets_backward > 0) {
                if (bwd_mux_args.connection_valid) {
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &bwd_mux_conn,
                        pkt_hdr_backward,
                        l1_read_addr,
                        stats_tile_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            dram_dest_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, false},
                        /*start_distance=*/1,
                        static_cast<uint8_t>(num_targets_backward));
                }
            }

            cb_pop_front(stats_local_cb, 1);
        }

        cumulative_expected_incs += rows_in_chunk * (ring_size - 1);
        if (cumulative_expected_incs > 0) {
            noc_semaphore_wait_min(out_ready_sem_ptr, cumulative_expected_incs);
        }
        // Wait for local DRAM writes to commit before reading them back.
        noc_async_write_barrier();

        // ---- Phase A.5: Read DRAM stats back into stats_gathered_cb ----
        // The compute kernel still consumes stats from stats_gathered_cb in L1
        // (one ring_size-wide tile-row per reduce). DRAM is just the safe
        // transport medium for cross-chip AG.
        cb_reserve_back(stats_gathered_cb, chunk_stats_tiles);
        uint32_t cb_wr_ptr = get_write_ptr(stats_gathered_cb);
        for (uint32_t r = 0; r < rows_in_chunk; r++) {
            for (uint32_t d = 0; d < ring_size; d++) {
                const uint32_t dram_tile_idx = worker_tile_base + r * ring_size + d;
                noc_async_read_tile(dram_tile_idx, stats_dram_accessor, cb_wr_ptr);
                cb_wr_ptr += stats_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(stats_gathered_cb, chunk_stats_tiles);

        // Drain this chunk's output_cb tiles to DRAM. Compute always pushes
        // `block_size` slots per col-block (LLK packer requirement) even when
        // only `tiles_in_block` (clamped) are valid; we wait+pop the full
        // `block_size` to keep the CB level zero per row, but only NoC-write
        // the valid tiles. Without this, multi-chunk overflows output_cb
        // because over-pushed garbage slots never drain.
        for (uint32_t r = 0; r < rows_in_chunk; r++) {
            const uint32_t tile_row = tile_row_start + row_processed + r;
            uint32_t output_tile_idx = tile_row * num_tile_cols;
            for (uint32_t col_tile = 0; col_tile < num_tile_cols; col_tile += block_size) {
                const uint32_t tiles_in_block =
                    ((num_tile_cols - col_tile) >= block_size) ? block_size : (num_tile_cols - col_tile);
                cb_wait_front(output_cb, block_size);
                uint32_t output_rd_ptr = get_read_ptr(output_cb);
                for (uint32_t i = 0; i < tiles_in_block; i++) {
                    noc_async_write_tile(output_tile_idx, output_accessor, output_rd_ptr);
                    output_rd_ptr += output_tile_bytes;
                    output_tile_idx++;
                }
                noc_async_write_barrier();
                cb_pop_front(output_cb, block_size);
            }
        }

        row_processed += rows_in_chunk;
    }

    // Reset sem for any subsequent invocation.
    noc_semaphore_set(out_ready_sem_ptr, 0);

    // ---------- Disconnect MUX + terminate ----------
    noc_async_write_barrier();
    noc_async_atomic_barrier();

    if (fwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(fwd_mux_conn);
        terminate_mux_handshake(fwd_mux_args);
    }
    if (bwd_mux_args.connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(bwd_mux_conn);
        terminate_mux_handshake(bwd_mux_args);
    }
    noc_async_write_barrier();
}
