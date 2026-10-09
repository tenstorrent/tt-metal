// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Multi-packet variant of dit_fused_norm_common's coalescing forwarder, used only by
 * the column-split RMSNorm layout. Same per-round protocol, except a round's group
 * sticks are carried by num_packets fabric packets (sticks_per_pk sticks each, one DRAM
 * scratch page each), so a group may exceed one packet's capacity. Each packet fuses
 * its own out_ready inc, so a round waits for (ring_size-1)*num_packets peer incs.
 * With stage_stats the forwarder then copies its group's ring_size*num_packets pages
 * from local DRAM into stage_cb (grid-uniform) before releasing the workers, who read
 * their sticks from this core's L1 instead of DRAM.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

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
constexpr auto forward_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<stats_dram_args.next_compile_time_args_offset()>();
constexpr auto backward_route = ccl_routing_utils::get_line_multicast_route_info_from_args<
    stats_dram_args.next_compile_time_args_offset() + ccl_routing_utils::num_line_multicast_args>();
constexpr uint32_t XCB =
    stats_dram_args.next_compile_time_args_offset() + 2 * ccl_routing_utils::num_line_multicast_args;
constexpr uint32_t num_packets = get_compile_time_arg_val(XCB + 0);
constexpr uint32_t sticks_per_pk = get_compile_time_arg_val(XCB + 1);
constexpr uint32_t stage_stats = get_compile_time_arg_val(XCB + 2);
constexpr uint32_t stage_cb = get_compile_time_arg_val(XCB + 3);
// One fabric packet / DRAM scratch page. Passed explicitly: CircularBuffer::get_tile_size()
// reports the fp32 tile size, not the packet CB's page size.
constexpr uint32_t unit_packet_bytes = get_compile_time_arg_val(XCB + 4);

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
    Noc noc;
    CircularBuffer cb_pkt_hdr(reserved_packet_header_cb);
    CircularBuffer cb_packet(packet_cb);
    Semaphore<> fwd_arrival_sem(arrival_sem_id);
    Semaphore<> group_go_sem(go_sem_id);

    // Fabric connection (fwd+bwd) for this forwarder's link.
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    cb_pkt_hdr.reserve_back(1);
    auto pkt_hdr_fwd_addr = cb_pkt_hdr.get_write_ptr();
    cb_pkt_hdr.push_back(1);
    cb_pkt_hdr.reserve_back(1);
    auto pkt_hdr_bwd_addr = cb_pkt_hdr.get_write_ptr();
    cb_pkt_hdr.push_back(1);
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_fwd_addr);
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_bwd_addr);
    // The hop-only header initializer is a no-op for 2D fabric headers.
    // Use the shared CCL helper to initialize the correct route for either fabric.
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_fwd, forward_route);
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_bwd, backward_route);
    // Guard open_finish on is_logically_connected(), matching the canonical CCL writers
    // (all_reduce_async / all_to_all_async): a forwarder with no live fabric connection
    // (e.g. a line-topology edge or a degenerate ring) must not run the open handshake.
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
    const uint32_t packet_base = cb_packet.get_read_ptr();         // packet_cb is depth-2; we index by r%2 manually
    const uint32_t packet_tile_bytes = cb_packet.get_tile_size();  // round stride, as the workers use

    const uint64_t out_ready_sem_noc = safe_get_noc_addr(my_x[0], my_y[0], out_ready_sem_addr, 0);
    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);

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
            // Workers write their stick into packet_addr + slot*stick_bytes and inc.
            cumulative_arrivals += pc;
            fwd_arrival_sem.wait_min(cumulative_arrivals);
        }

        // Page (my_device, forwarder, r, packet) — same DRAM address on every chip.
        const uint32_t page_base =
            my_device_index * num_chunks_per_device + (forwarder_index * max_rounds + r) * num_packets;
        {
            for (uint32_t p = 0; p < num_packets; p++) {
                const uint32_t first = p * sticks_per_pk;
                const uint32_t n = (pc > first) ? ((pc - first) < sticks_per_pk ? (pc - first) : sticks_per_pk) : 0u;
                if (n == 0) {
                    break;
                }
                const uint64_t dram_dest =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(stats_dram, page_base + p, 0);
                size_t l1_read_addr = packet_addr + p * unit_packet_bytes;
                fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                    dram_dest,
                    pkt_hdr_fwd,
                    pkt_hdr_bwd,
                    fabric_connection,
                    l1_read_addr,
                    n * stick_bytes,
                    out_ready_sem_noc,
                    /*val=*/1,
                    /*flush=*/true);
                cumulative_incs += (ring_size - 1);
                // The fwd/bwd headers are rewritten for the next packet; the non-blocking send
                // must have read them (and the payload) out of L1 first.
                noc_async_writes_flushed();
            }
            if (cumulative_incs > 0) {
                // out_ready is a GlobalSemaphore.
                noc_semaphore_wait_min(out_ready_sem_ptr, cumulative_incs);
            }
            noc.async_write_barrier();
            noc.async_atomic_barrier();
        }
        if constexpr (stage_stats) {
            CircularBuffer cb_stage(stage_cb);
            const uint32_t stage_base = cb_stage.get_write_ptr();
            const uint32_t used_packets = (pc + sticks_per_pk - 1) / sticks_per_pk;
            for (uint32_t d = 0; d < ring_size; d++) {
                for (uint32_t p = 0; p < used_packets; p++) {
                    const uint32_t page =
                        d * num_chunks_per_device + (forwarder_index * max_rounds + r) * num_packets + p;
                    const uint32_t bytes =
                        ((pc - p * sticks_per_pk) < sticks_per_pk ? (pc - p * sticks_per_pk) : sticks_per_pk) *
                        stick_bytes;
                    noc_async_read(
                        stats_dram.get_noc_addr(page), stage_base + (d * num_packets + p) * unit_packet_bytes, bytes);
                }
            }
            noc_async_read_barrier();
        }

        // Release this round's POST: inc each PRESENT group worker's go-sem. The
        // present set is the contiguous slot prefix [0, pc) (earlier workers get
        // the ceil row count), so only the first pc workers loop this round.
        for (uint32_t i = 0; i < pc; i++) {
            group_go_sem.up(noc, worker_x[i], worker_y[i], 1);
        }
        noc.async_atomic_barrier();
    }

    // Reset BOTH op-managed semaphores this core owns to 0 so a traced replay
    // starts clean. Trace capture/replay does NOT re-run the host-side semaphore
    // init that eager launches get, so any sem left non-zero accumulates across
    // replays. out_ready (peers fuse-inc it) and fwd_arrival (workers inc it) both
    // live on this forwarder core; the workers reset their own go-sem. Without the
    // arrival reset the forwarder's fwd_arrival_sem.wait_min(cumulative) passes
    // instantly on replay (sem already high) -> it stops waiting for the workers'
    // sticks -> races (fast-but-wrong) or desyncs into a hang. (go/out_ready already
    // reset.)
    //
    // out_ready is a GlobalSemaphore.
    noc_semaphore_set(out_ready_sem_ptr, 0);
    fwd_arrival_sem.set(0);
    // Guard close on is_logically_connected(), matching the canonical CCL writers.
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
        fabric_connection.close_finish();
    }
}
