// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Two-wave RMSNorm fork of the coalescing fabric forwarder
 * (dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp) for the column-split layout.
 * Consumer: ttnn::experimental::prim::DitFusedDistributedRmsnormMeshWorkloadFactory (col_split only).
 * Compile-time and runtime args are the common forwarder's, plus wave_slots / wave_span_bytes (CT).
 *
 * The group's 2 * wave_slots workers form two waves (two column-half workers per tile-row, half of the
 * rows per wave). The factory starts wave 1's input read only as wave 0's lands, so wave 0's ring gather
 * overlaps wave 1's DRAM read and wave 1's gather overlaps wave 0's output drain. Wave w's sticks are the
 * region [w * wave_span_bytes, (w + 1) * wave_span_bytes) of packet_buf[0] and of the scratch page
 * (my_device, forwarder, 0) on every chip.
 *
 * Each wave is counted in its own 16-bit field of the shared semaphores: a wave-w worker incs fwd_arrival
 * by 1 << (16*w), and wave w's fused fabric inc adds 1 << (16*w) to every peer's out_ready. So a fast
 * peer's wave-1 inc can never satisfy this chip's wave-0 wait, and wave 1's packet doesn't wait behind
 * wave 0's out_ready. One poll loop: send wave `sent` once its arrivals are complete; release wave
 * `released` (inc its workers' go-sems, slot order) once all peers' wave packets have landed.
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
#include "tools/profiler/kernel_profiler.hpp"

// ---------- compile-time args ----------
constexpr std::uint32_t packet_cb = get_compile_time_arg_val(0);  // outbound packet buffer (unit_packet x2)
constexpr std::uint32_t reserved_packet_header_cb = get_compile_time_arg_val(1);
constexpr std::uint32_t ring_size = get_compile_time_arg_val(2);
constexpr std::uint32_t my_device_index = get_compile_time_arg_val(3);
constexpr std::uint32_t num_targets_forward = get_compile_time_arg_val(4);
constexpr std::uint32_t num_targets_backward = get_compile_time_arg_val(5);
constexpr std::uint32_t forwarder_index = get_compile_time_arg_val(6);  // which link/group on this chip
constexpr std::uint32_t num_forwarders = get_compile_time_arg_val(7);
constexpr std::uint32_t group_size = get_compile_time_arg_val(8);  // workers in this forwarder's group
constexpr std::uint32_t max_rounds = get_compile_time_arg_val(9);  // 1 (one row per worker)
constexpr std::uint32_t stick_bytes = get_compile_time_arg_val(10);
constexpr std::uint32_t num_chunks_per_device = get_compile_time_arg_val(11);  // = num_forwarders * max_rounds
constexpr std::uint32_t arrival_sem_id = get_compile_time_arg_val(12);         // workers inc; forwarder waits
constexpr std::uint32_t go_sem_id = get_compile_time_arg_val(13);              // forwarder incs; workers wait
constexpr auto stats_dram_args = TensorAccessorArgs<14>();
constexpr auto forward_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<stats_dram_args.next_compile_time_args_offset()>();
constexpr auto backward_route = ccl_routing_utils::get_line_multicast_route_info_from_args<
    stats_dram_args.next_compile_time_args_offset() + ccl_routing_utils::num_line_multicast_args>();
constexpr std::uint32_t kWaveArgs =
    stats_dram_args.next_compile_time_args_offset() + 2 * ccl_routing_utils::num_line_multicast_args;
constexpr std::uint32_t wave_slots = get_compile_time_arg_val(kWaveArgs);           // sticks (workers) per wave
constexpr std::uint32_t wave_span_bytes = get_compile_time_arg_val(kWaveArgs + 1);  // wave region in the page
constexpr std::uint32_t kNumWaves = 2u;
static_assert(max_rounds == 1, "two-wave forwarder needs a single row-round");
static_assert(group_size == kNumWaves * wave_slots, "two-wave forwarder: group must be two full waves");
constexpr std::uint32_t kWaveFieldBits = 16u;
constexpr std::uint32_t kWaveFieldMask = (1u << kWaveFieldBits) - 1u;

void kernel_main() {
    size_t arg_idx = 0;
    const std::uint32_t stats_dram_addr = get_arg_val<std::uint32_t>(arg_idx++);
    // out_ready_sem: a GlobalSemaphore — PEER forwarders fuse-inc it over fabric
    // (same L1 addr on every chip). Local pointer for the wait + reset.
    const std::uint32_t out_ready_sem_addr = get_arg_val<std::uint32_t>(arg_idx++);
    // Group worker NoC coords (x,y), wave-major slot order, then present_count[0] (unused: every slot is
    // present with one row per worker).
    std::uint32_t worker_x[group_size];
    std::uint32_t worker_y[group_size];
    for (std::uint32_t i = 0; i < group_size; i++) {
        worker_x[i] = get_arg_val<std::uint32_t>(arg_idx++);
        worker_y[i] = get_arg_val<std::uint32_t>(arg_idx++);
    }
    arg_idx += max_rounds;
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
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_fwd, forward_route);
    ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_bwd, backward_route);
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
    const std::uint32_t packet_base = cb_packet.get_read_ptr();

    const std::uint64_t out_ready_sem_noc = safe_get_noc_addr(my_x[0], my_y[0], out_ready_sem_addr, 0);
    volatile tt_l1_ptr std::uint32_t* out_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(out_ready_sem_addr);
    volatile tt_l1_ptr std::uint32_t* arrival_ptr =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_semaphore(arrival_sem_id));

    const std::uint32_t page_idx = my_device_index * num_chunks_per_device + forwarder_index * max_rounds;
    std::uint32_t sent = 0;
    std::uint32_t released = 0;
    while (released < kNumWaves) {
        invalidate_l1_cache();
        if (sent < kNumWaves) {
            const std::uint32_t shift = sent * kWaveFieldBits;
            if (((*arrival_ptr >> shift) & kWaveFieldMask) >= wave_slots) {
                DeviceZoneScopedN("F_SEND");
                // Wave `sent`'s region of packet_buf[0] -> the same offset of page (my_device, fwd, 0).
                const std::uint32_t off = sent * wave_span_bytes;
                const std::uint64_t dram_dest =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(stats_dram, page_idx, off);
                size_t l1_read_addr = packet_base + off;
                fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                    dram_dest,
                    pkt_hdr_fwd,
                    pkt_hdr_bwd,
                    fabric_connection,
                    l1_read_addr,
                    wave_span_bytes,
                    out_ready_sem_noc,
                    /*val=*/1u << shift,
                    /*flush=*/true);
                sent++;
                continue;
            }
        }
        if (released < sent) {
            const std::uint32_t shift = released * kWaveFieldBits;
            if (((*out_ready_sem_ptr >> shift) & kWaveFieldMask) >= ring_size - 1) {
                DeviceZoneScopedN("F_GO");
                // Our own local write of this wave must have landed before its workers read it.
                noc.async_write_barrier();
                for (std::uint32_t i = released * wave_slots; i < (released + 1) * wave_slots; i++) {
                    group_go_sem.up(noc, worker_x[i], worker_y[i], 1);
                }
                released++;
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();

    // Reset BOTH op-managed semaphores this core owns to 0 so a traced replay starts clean (trace replay
    // does not re-run the host-side semaphore init). The workers reset their own go-sem.
    noc_semaphore_set(out_ready_sem_ptr, 0);
    fwd_arrival_sem.set(0);
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
        fabric_connection.close_finish();
    }
}
