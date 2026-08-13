// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Coalescing fabric forwarder for the fused Wan2.2 distributed RMSNorm AG.
 *
 * One forwarder core per fabric link (num_forwarders = min(num_links,
 * num_workers)). It owns a contiguous GROUP of worker cores on this chip and
 * holds the fwd+bwd fabric connections for its link. Its job: coalesce the
 * group's tiny 128 B stat "sticks" into one large fabric packet per row-round
 * and ring-multicast it, instead of each worker sending its own tiny 128 B
 * (≈3% utilization) fabric packet.
 *
 * Per row-round r (r in [0, max_rounds)):
 *   1. Wait for present_count(r) arrivals on fwd_arrival_sem — each present
 *      worker NoC-writes its 128 B stick into packet_buf[r%2] + slot*128 B and
 *      increments the sem. (present_count shrinks on the last, remainder round;
 *      the present set is always a contiguous slot prefix [0, present_count).)
 *   2. Ring-multicast packet_buf[r%2][0 : present_count(r)*128 B] to the DRAM
 *      scratch page (my_device, forwarder, r) on every chip (local write + fwd
 *      + bwd fabric), fusing an atomic_inc on every PEER forwarder's
 *      out_ready_sem (flush=true: payload committed before the inc is seen).
 *   3. Wait out_ready_sem >= cumulative (ring_size-1) incs for this round —
 *      i.e. all peer forwarders' packets have landed in THIS chip's DRAM.
 *   4. On-chip: increment each group worker's go-sem so they read their
 *      gathered sticks from DRAM and run POST for row r.
 *
 * Symmetric across devices (TP shards features, so every chip has identical
 * num_tile_rows / partition), so the per-round present set + inc counts match
 * on both ends — no deadlock, no orphaned waits. Remote packets land in DRAM
 * (workers read them), so the forwarder holds only its outbound packet_buf.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tools/profiler/kernel_profiler.hpp"

// ---------- compile-time args ----------
constexpr uint32_t packet_cb = get_compile_time_arg_val(0);  // outbound packet buffer (unit_packet x2)
constexpr uint32_t reserved_packet_header_cb = get_compile_time_arg_val(1);
constexpr uint32_t ring_size = get_compile_time_arg_val(2);
constexpr uint32_t my_device_index = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward = get_compile_time_arg_val(5);
constexpr uint32_t forwarder_index = get_compile_time_arg_val(6);  // which link/group on this chip
constexpr uint32_t num_forwarders = get_compile_time_arg_val(7);
constexpr uint32_t group_size = get_compile_time_arg_val(8);    // workers in this forwarder's group
constexpr uint32_t max_rounds = get_compile_time_arg_val(9);    // ceil(num_tile_rows / num_workers)
constexpr uint32_t stick_bytes = get_compile_time_arg_val(10);  // 128 (32 fp32)
constexpr uint32_t num_chunks_per_device =
    get_compile_time_arg_val(11);  // = num_forwarders * max_rounds (pages/device)
// Grid-uniform semaphore ids (created on the whole core grid -> same L1 addr on
// every worker + forwarder core, so no cross-core address args are needed).
constexpr uint32_t arrival_sem_id = get_compile_time_arg_val(12);  // workers inc; forwarder waits
constexpr uint32_t go_sem_id = get_compile_time_arg_val(13);       // forwarder incs; workers wait
constexpr auto stats_dram_args = TensorAccessorArgs<14>();

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t stats_dram_addr = get_arg_val<uint32_t>(arg_idx++);
    // out_ready_sem: a GlobalSemaphore — PEER forwarders fuse-inc it over fabric
    // (same L1 addr on every chip). Local pointer for the wait + reset.
    const uint32_t out_ready_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    // group worker NoC coords (x,y) x group_size, then present_count[r]. Sized by
    // the constexpr CT args (group_size / max_rounds) so large-row configs (e.g.
    // wan self_sp4_N18944 -> 592 tile-rows / 32 workers = 19 rounds) don't overflow
    // a fixed bound -> -Werror=aggressive-loop-optimizations at JIT time.
    uint32_t worker_x[group_size];
    uint32_t worker_y[group_size];
    for (uint32_t i = 0; i < group_size; i++) {
        worker_x[i] = get_arg_val<uint32_t>(arg_idx++);
        worker_y[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    uint32_t present_count[max_rounds];
    for (uint32_t r = 0; r < max_rounds; r++) {
        present_count[r] = get_arg_val<uint32_t>(arg_idx++);
    }
    // Grid-uniform sem addresses (same on this forwarder + its workers).
    const uint32_t fwd_arrival_sem_addr = get_semaphore(arrival_sem_id);
    const uint32_t group_go_sem_addr = get_semaphore(go_sem_id);

    // Fabric connection (fwd+bwd) for this forwarder's link.
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    cb_reserve_back(reserved_packet_header_cb, 1);
    auto pkt_hdr_fwd_addr = get_write_ptr(reserved_packet_header_cb);
    cb_push_back(reserved_packet_header_cb, 1);
    cb_reserve_back(reserved_packet_header_cb, 1);
    auto pkt_hdr_bwd_addr = get_write_ptr(reserved_packet_header_cb);
    cb_push_back(reserved_packet_header_cb, 1);
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_fwd_addr);
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_bwd_addr);
    pkt_hdr_fwd->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward)});
    pkt_hdr_bwd->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward)});
    // Guard open_finish on is_logically_connected(), matching the canonical CCL writers
    // (all_reduce_async / all_to_all_async): a forwarder with no live fabric connection
    // (e.g. a line-topology edge or a degenerate ring) must not run the open handshake.
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
    const uint32_t packet_base = get_read_ptr(packet_cb);         // packet_cb is depth-2; we index by r%2 manually
    const uint32_t packet_tile_bytes = get_tile_size(packet_cb);  // = unit_packet_bytes (one slot)

    const uint64_t out_ready_sem_noc = safe_get_noc_addr(my_x[0], my_y[0], out_ready_sem_addr, 0);
    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* fwd_arrival_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_arrival_sem_addr);

    uint32_t cumulative_arrivals = 0;
    uint32_t cumulative_incs = 0;
    for (uint32_t r = 0; r < max_rounds; r++) {
        const uint32_t pc = present_count[r];
        // Zero-present round (uneven num_tile_rows / num_workers split, e.g. 38
        // tile-rows over 32 workers -> later forwarders have no worker on the
        // remainder round). Symmetric across devices, so every peer forwarder_index
        // also has pc==0 and skips: no arrivals, no fabric packet, no peer incs, no
        // workers to release. A 0-byte fused fabric write would NOT transmit its
        // atomic inc, so peers would wait forever -> must skip the round entirely.
        if (pc == 0) {
            continue;
        }
        const uint32_t packet_addr = packet_base + (r & 1u) * packet_tile_bytes;

        {
            DeviceZoneScopedN("F_COLLECT");
            // Workers write their stick into packet_addr + slot*stick_bytes and inc.
            cumulative_arrivals += pc;
            noc_semaphore_wait_min(fwd_arrival_sem_ptr, cumulative_arrivals);
        }

        // Page (my_device, forwarder, r) — same DRAM address on every chip.
        const uint32_t page_idx = my_device_index * num_chunks_per_device + forwarder_index * max_rounds + r;
        const uint64_t dram_dest = tt::tt_fabric::linear::addrgen_detail::get_noc_address(stats_dram, page_idx, 0);
        {
            DeviceZoneScopedN("F_FABRIC");
            size_t l1_read_addr = packet_addr;
            fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                dram_dest,
                pkt_hdr_fwd,
                pkt_hdr_bwd,
                fabric_connection,
                l1_read_addr,
                pc * stick_bytes,
                out_ready_sem_noc,
                /*val=*/1,
                /*flush=*/true);
            cumulative_incs += (ring_size - 1);
            if (cumulative_incs > 0) {
                noc_semaphore_wait_min(out_ready_sem_ptr, cumulative_incs);
            }
            noc_async_write_barrier();
            noc_async_atomic_barrier();
        }

        // Release this round's POST: inc each PRESENT group worker's go-sem. The
        // present set is the contiguous slot prefix [0, pc) (earlier workers get
        // the ceil row count), so only the first pc workers loop this round.
        for (uint32_t i = 0; i < pc; i++) {
            const uint64_t go = safe_get_noc_addr(worker_x[i], worker_y[i], group_go_sem_addr, 0);
            noc_semaphore_inc(go, 1);
        }
        noc_async_atomic_barrier();
    }

    // Reset BOTH op-managed semaphores this core owns to 0 so a traced replay
    // starts clean. Trace capture/replay does NOT re-run the host-side semaphore
    // init that eager launches get, so any sem left non-zero accumulates across
    // replays. out_ready (peers fuse-inc it) and fwd_arrival (workers inc it) both
    // live on this forwarder core; the workers reset their own go-sem. Without the
    // arrival reset the forwarder's wait_min(arrival, cumulative) passes instantly
    // on replay (sem already high) -> it stops waiting for the workers' sticks ->
    // races (fast-but-wrong) or desyncs into a hang. (go/out_ready already reset.)
    noc_semaphore_set(out_ready_sem_ptr, 0);
    noc_semaphore_set(fwd_arrival_sem_ptr, 0);
    // Guard close on is_logically_connected(), matching the canonical CCL writers.
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
        fabric_connection.close_finish();
    }
}
