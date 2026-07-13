// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fabric forwarder for fused distributed GroupNorm AG.
 *
 * Adapted from dit_rmsnorm_fused_forwarder.cpp with max_rounds=1 and a single
 * worker stick of stick_bytes = num_groups * 16 (contiguous fp32 sum/sumsq/count
 * per group, not face-row packed tiles).
 *
 * Per round r (r in [0, max_rounds); for GN max_rounds==1):
 *   1. Wait present_count(r) arrivals on fwd_arrival_sem
 *   2. Ring-multicast packet_buf[r%2][0 : pc*stick_bytes] to DRAM page
 *      (my_device, forwarder, r) on every chip + fused atomic_inc on peers
 *   3. Wait out_ready_sem >= cumulative (ring_size-1) incs
 *   4. Inc each present worker's go-sem
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t packet_cb = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb = get_compile_time_arg_val(1);
constexpr uint32_t ring_size = get_compile_time_arg_val(2);
constexpr uint32_t my_device_index = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward = get_compile_time_arg_val(5);
constexpr uint32_t forwarder_index = get_compile_time_arg_val(6);
constexpr uint32_t num_forwarders = get_compile_time_arg_val(7);
constexpr uint32_t group_size = get_compile_time_arg_val(8);
constexpr uint32_t max_rounds = get_compile_time_arg_val(9);
constexpr uint32_t stick_bytes = get_compile_time_arg_val(10);
constexpr uint32_t num_chunks_per_device = get_compile_time_arg_val(11);
constexpr uint32_t arrival_sem_id = get_compile_time_arg_val(12);
constexpr uint32_t go_sem_id = get_compile_time_arg_val(13);
constexpr auto stats_dram_args = TensorAccessorArgs<14>();

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t stats_dram_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_ready_sem_addr = get_arg_val<uint32_t>(arg_idx++);
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
    const uint32_t fwd_arrival_sem_addr = get_semaphore(arrival_sem_id);
    const uint32_t group_go_sem_addr = get_semaphore(go_sem_id);

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
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    const auto stats_dram = TensorAccessor(stats_dram_args, stats_dram_addr);
    const uint32_t packet_base = get_read_ptr(packet_cb);
    const uint32_t packet_tile_bytes = get_tile_size(packet_cb);

    const uint64_t out_ready_sem_noc = safe_get_noc_addr(my_x[0], my_y[0], out_ready_sem_addr, 0);
    volatile tt_l1_ptr uint32_t* out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* fwd_arrival_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_arrival_sem_addr);

    uint32_t cumulative_arrivals = 0;
    uint32_t cumulative_incs = 0;
    for (uint32_t r = 0; r < max_rounds; r++) {
        const uint32_t pc = present_count[r];
        if (pc == 0) {
            continue;
        }
        const uint32_t packet_addr = packet_base + (r & 1u) * packet_tile_bytes;

        {
            DeviceZoneScopedN("GN_F_COLLECT");
            cumulative_arrivals += pc;
            noc_semaphore_wait_min(fwd_arrival_sem_ptr, cumulative_arrivals);
        }

        const uint32_t page_idx = my_device_index * num_chunks_per_device + forwarder_index * max_rounds + r;
        const uint64_t dram_dest = tt::tt_fabric::linear::addrgen_detail::get_noc_address(stats_dram, page_idx, 0);
        {
            DeviceZoneScopedN("GN_F_FABRIC");
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

        for (uint32_t i = 0; i < pc; i++) {
            const uint64_t go = safe_get_noc_addr(worker_x[i], worker_y[i], group_go_sem_addr, 0);
            noc_semaphore_inc(go, 1);
        }
        noc_async_atomic_barrier();
    }

    noc_semaphore_set(out_ready_sem_ptr, 0);
    noc_semaphore_set(fwd_arrival_sem_ptr, 0);
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
        fabric_connection.close_finish();
    }
}
