// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "api/debug/device_print.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/overlap_config.hpp"
#include "hostdev/fabric_telemetry_msgs.h"
#include "dev_mem_map.h"

// ============================================================================
// writer_relay — relay ("R") core writer kernel = the combine FABRIC ENDPOINT.
// ============================================================================
//
// STAGE (relay-2): the relay took over the fabric endpoint role from the sender. It is the ONLY kernel
// on its core, so it uses NOC_0 for everything (the preferred write-to-eth NOC), while the sender/
// untilizer NOC assignments are unchanged from main. The relay:
//   1. opens the eth (EDM) fabric connections,
//   2. runs the cross-chip init/exit semaphore handshake (moved here from writer_combine),
//   3. sends tokens to fabric,
//   4. tears the connections down.
// writer_combine (the sender) now opens no connection and sends nothing (see its USE_RELAY early exit).
//
// This stage ports ONLY the MOCK, FABRIC_1D send path (matching MOCK_COMBINE_INTERNALS' constraints):
// the relay fabricates synthetic tokens (garbage payload from its own c_24 buffer scratch) and pushes
// them through the trid+header overlap pool. Later stages feed real tokens from the sender over the
// sender->relay CB. The token PAYLOAD is synthetic here (perf/liveness parity with today's mock).
#if !MOCK_COMBINE_INTERNALS
#error "writer_relay currently implements only the MOCK send path (MOCK_COMBINE_INTERNALS=1)."
#endif
#ifdef FABRIC_2D
#error "writer_relay is FABRIC_1D only for now (uses get_route/manhattan_distance + the 1D handshake)."
#endif
#ifndef DEST_CHIP_ID
#error "writer_relay requires a multi-chip build (DEST_CHIP_ID)."
#endif

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_COMBINE(...)
#endif

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile-time args =====
    constexpr uint32_t output_pages = get_compile_time_arg_val(0);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(2);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(3);
    constexpr uint32_t num_links = get_compile_time_arg_val(4);
    [[maybe_unused]] constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(5);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(6);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(7);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(8);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(9);
    [[maybe_unused]] constexpr uint32_t src_mesh_id = get_compile_time_arg_val(10);
    constexpr uint32_t num_chips = get_compile_time_arg_val(11);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(12);
    constexpr uint32_t cb_relay_buf = get_compile_time_arg_val(13);
    constexpr auto output_args = TensorAccessorArgs<14>();

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t combine_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t combine_devices = num_chips;
#endif
    constexpr uint32_t total_mesh_devices = mesh_rows * mesh_cols;
    constexpr uint8_t dest_chip_ids[total_mesh_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[total_mesh_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    // ===== Runtime args =====
    size_t rt_args_idx = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t exit_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t relay_index = get_arg_val<uint32_t>(rt_args_idx++);
    // [cmb-place] host-computed coordinate quad (order matches push_worker_coord_quad):
    const uint32_t self_logical_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_logical_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_virt_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_virt_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_phys_noc0_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_phys_noc0_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_noc1_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t self_noc1_y = get_arg_val<uint32_t>(rt_args_idx++);
    // fabric-connection args are appended after this by append_fabric_connection_rt_args (read below).

    // [debug] BW + per-token gap histogram (mirrors writer_combine so relay send perf is comparable).
    uint64_t bw_total_payload_bytes = 0;
    uint64_t hist_last_ts = 0;
    uint32_t hist_buckets[7] = {0, 0, 0, 0, 0, 0, 0};
    auto hist_record = [&hist_last_ts, &hist_buckets](uint64_t now) {
        const uint64_t d = now - hist_last_ts;
        hist_last_ts = now;
        if (d < 135ULL) {
            hist_buckets[0]++;
        } else if (d < 1350ULL) {
            hist_buckets[1]++;
        } else if (d < 13500ULL) {
            hist_buckets[2]++;
        } else if (d < 135000ULL) {
            hist_buckets[3]++;
        } else if (d < 1350000ULL) {
            hist_buckets[4]++;
        } else if (d < 13500000ULL) {
            hist_buckets[5]++;
        } else {
            hist_buckets[6]++;
        }
    };
    uint32_t mock_wait_buckets[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t mock_send_buckets[7] = {0, 0, 0, 0, 0, 0, 0};
    auto mock_bin = [](uint32_t* b, uint64_t d) {
        if (d < 135ULL) {
            b[0]++;
        } else if (d < 1350ULL) {
            b[1]++;
        } else if (d < 13500ULL) {
            b[2]++;
        } else if (d < 135000ULL) {
            b[3]++;
        } else if (d < 1350000ULL) {
            b[4]++;
        } else if (d < 13500000ULL) {
            b[5]++;
        } else {
            b[6]++;
        }
    };

    // ===== Fabric connections (FABRIC_1D array + per-target unicast handshake) =====
    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);
    auto* sem_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    open_direction_connections_barrier(directions, fabric_connections);

    // ===== Init semaphore exchange (moved here from writer_combine) =====
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        total_mesh_devices>(
        fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);
    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, combine_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);
    DPRINT_COMBINE("Relay fabric setup complete\n");

    // [debug] START marker in each connected eth router's telemetry scratch[0] (same scheme as sender).
    constexpr uint32_t combine_marker_l1_addr = MEM_AERISC_FABRIC_TELEMETRY_BASE + offsetof(FabricTelemetry, scratch);
    const uint32_t combine_marker_value = 100 + src_chip_id * 10 + relay_index;
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            noc_inline_dw_write<InlineWriteDst::L1>(
                get_noc_addr(fabric_connections[d].edm_noc_x, fabric_connections[d].edm_noc_y, combine_marker_l1_addr),
                combine_marker_value);
        }
    }

    // [debug][cmb-place] placement + downstream eth (now that the relay connects to fabric). Mirrors
    // [cmb-place sender]; send_noc reports the relay's NOC (expect 0 == NOC_0).
    DEVICE_PRINT(
        "[cmb-place relay] idx={} logical=({},{}) virt=({},{}) phys_noc0=({},{}) noc1=({},{}) | "
        "dev_virt=({},{}) dev_logical=({},{}) send_noc={}\n",
        relay_index,
        self_logical_x,
        self_logical_y,
        self_virt_x,
        self_virt_y,
        self_phys_noc0_x,
        self_phys_noc0_y,
        self_noc1_x,
        self_noc1_y,
        (uint32_t)my_x[noc_index],
        (uint32_t)my_y[noc_index],
        (uint32_t)get_absolute_logical_x(),
        (uint32_t)get_absolute_logical_y(),
        (uint32_t)noc_index);
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            DEVICE_PRINT(
                "[cmb-place relay]   idx={} dir={} downstream_eth_virt=({},{})\n",
                relay_index,
                d,
                (uint32_t)fabric_connections[d].edm_noc_x,
                (uint32_t)fabric_connections[d].edm_noc_y);
        }
    }

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

    // Lockstep trid + header pool (right after the 2 base headers in c_5). Each in-flight token uses
    // header pool[overlap_slot] and trid (overlap_slot+1), cycling over [0, OVERLAP_POOL_DEPTH).
    const uint32_t overlap_hdr_pool_base = packet_header_buffer_address + 2u * sizeof(PACKET_HEADER_TYPE);
    uint32_t overlap_slot = 0;

    // ===== MOCK writer-only synthetic fabric send loop (CB-free) =====
    {
        // (route direction, distance-in-hops) for every peer on the 1D combine line, out of the hot loop.
        uint8_t mock_route[total_mesh_devices];
        uint16_t mock_distance[total_mesh_devices];
        uint32_t mock_num_peers = 0;
        for (uint32_t dst = 0; dst < total_mesh_devices; dst++) {
            if (dst == linearized_mesh_coord) {
                continue;  // never send to self
            }
            mock_route[mock_num_peers] =
                static_cast<uint8_t>(get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst));
            mock_distance[mock_num_peers] =
                static_cast<uint16_t>(manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst));
            mock_num_peers++;
        }

        // Payload scratch: reuse the relay's own c_24 buffer (its first slot's data region) as a fixed
        // 14 KB source. Content is never refreshed (garbage, perf only) so nothing overwrites it while
        // the non-blocking overlapped reads are in flight -> race-free.
        const uint32_t mock_payload_addr = get_write_ptr(cb_relay_buf) + l1_alignment;
        const uint32_t mock_total_tokens = mock_num_peers * MOCK_COMBINE_TOKENS_PER_DEST;

        uint32_t peer = 0;
        for (uint32_t i = 0; i < mock_total_tokens; i++) {
            ccl_routing_utils::line_unicast_route_info_t pkt_route_info{};
            pkt_route_info.distance_in_hops = mock_distance[peer];
            auto& payload_sender = fabric_connections[mock_route[peer]];
            const uint32_t output_page_idx = i % output_pages;

            auto* overlap_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(
                overlap_hdr_pool_base + overlap_slot * sizeof(PACKET_HEADER_TYPE));
            const uint32_t overlap_trid = overlap_slot + 1;  // 1..OVERLAP_POOL_DEPTH (trid 0 reserved)

            const uint64_t t0 = get_timestamp();
            wait_on_flush_for_trid(overlap_trid);  // LAGGED: free this slot's DEPTH-ago use before reuse
            const uint64_t t1 = get_timestamp();

            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_for_route_helper(overlap_hdr), pkt_route_info);
            fabric_send_noc_unicast_with_trid<fabric_max_packet_size>(
                output_addr_gen,
                payload_sender,
                overlap_hdr,
                mock_payload_addr,
                output_page_idx,
                (int)aligned_output_page_size,
                l1_alignment,
                overlap_trid);
            const uint64_t t2 = get_timestamp();

            mock_bin(mock_wait_buckets, t1 - t0);  // lagged trid-completion wait (want ~0)
            mock_bin(mock_send_buckets, t2 - t1);  // issue + wait_for_empty_write_slot (fabric slot)
            hist_record(t2);

            bw_total_payload_bytes += aligned_output_page_size;
            overlap_slot = (overlap_slot + 1 == OVERLAP_POOL_DEPTH) ? 0u : overlap_slot + 1u;
            peer = (peer + 1 == mock_num_peers) ? 0u : peer + 1u;
        }
    }

    // Drain the up-to-OVERLAP_POOL_DEPTH still-in-flight overlapped sends before the exit handshake.
    for (uint32_t t = 1; t <= OVERLAP_POOL_DEPTH; t++) {
        wait_on_flush_for_trid(t);
    }
    noc_async_write_barrier();

    // [debug] END marker (0) after the barrier, so the last data packet has departed L1.
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            noc_inline_dw_write<InlineWriteDst::L1>(
                get_noc_addr(fabric_connections[d].edm_noc_x, fabric_connections[d].edm_noc_y, combine_marker_l1_addr),
                0);
        }
    }

    // ===== Exit semaphore exchange (flush=true so peers can't see the inc before our data lands) =====
    {
        const uint64_t exit_noc_semaphore_addr = get_noc_addr(exit_semaphore_address);
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            axis,
            total_mesh_devices>(
            fabric_connections,
            sem_packet_header,
            dest_chip_ids,
            dest_mesh_ids,
            exit_noc_semaphore_addr,
            /*flush=*/true);
        volatile tt_l1_ptr uint32_t* exit_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exit_semaphore_address);
        noc_semaphore_wait(exit_sem_ptr, combine_devices - 1);
        noc_semaphore_set(exit_sem_ptr, 0);
    }

    noc_async_full_barrier();
    close_direction_connections(directions, fabric_connections);

    // [debug] per-token gap + mock decomposition histograms (mirrors writer_combine).
    DEVICE_PRINT(
        "[combine-hist] relay per-token gap histogram (counts): "
        "<100ns={} <1us={} <10us={} <100us={} <1ms={} <10ms={} >=10ms={}\n",
        hist_buckets[0],
        hist_buckets[1],
        hist_buckets[2],
        hist_buckets[3],
        hist_buckets[4],
        hist_buckets[5],
        hist_buckets[6]);
    DEVICE_PRINT(
        "[combine-mock] relay LAGGED trid-wait gap histogram (counts): "
        "<100ns={} <1us={} <10us={} <100us={} <1ms={} <10ms={} >=10ms={}\n",
        mock_wait_buckets[0],
        mock_wait_buckets[1],
        mock_wait_buckets[2],
        mock_wait_buckets[3],
        mock_wait_buckets[4],
        mock_wait_buckets[5],
        mock_wait_buckets[6]);
    DEVICE_PRINT(
        "[combine-mock] relay SEND-call gap histogram (incl. wait_for_empty_write_slot) (counts): "
        "<100ns={} <1us={} <10us={} <100us={} <1ms={} <10ms={} >=10ms={}\n",
        mock_send_buckets[0],
        mock_send_buckets[1],
        mock_send_buckets[2],
        mock_send_buckets[3],
        mock_send_buckets[4],
        mock_send_buckets[5],
        mock_send_buckets[6]);
}
