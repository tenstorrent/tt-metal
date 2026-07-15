// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/relay_config.hpp"

// [debug] Fabric-supplied API to open/close a router's detailed flow-control logging window ([rxlog]/[txlog]).
#include "tt_metal/fabric/hw/inc/edm_fabric/detailed_fabric_log_command.hpp"

// ============================================================================
// writer_relay — relay ("R") core writer kernel = the combine FABRIC ENDPOINT.
// ============================================================================
//
// The relay is the ONLY kernel on its core, so it uses NOC_0 for everything (the preferred
// write-to-eth NOC), while the sender/untilizer NOC assignments are unchanged from the base op. It:
//   1. opens the eth (EDM) fabric connections,
//   2. runs the cross-chip init/exit semaphore handshake,
//   3. drains tokens (route_info + payload) from its c_24 receive ring — fed by the owning sender over
//      the sender->relay NOC pipe, credit-flow-controlled — and forwards each to fabric,
//   4. tears the connections down.
// writer_combine (the sender) opens no connection and sends nothing over eth (see its USE_RELAY block).
//
// FABRIC_1D only (1D init/exit handshake + line-unicast route info).
#ifdef FABRIC_2D
#error "writer_relay is FABRIC_1D only."
#endif
#ifndef DEST_CHIP_ID
#error "writer_relay requires a multi-chip build (DEST_CHIP_ID)."
#endif

// Matches writer_combine / reader_combine: a slot whose route_info[0] == this marks end-of-stream.
constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile-time args =====
    [[maybe_unused]] constexpr uint32_t output_pages = get_compile_time_arg_val(0);  // page idx comes from slot
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(2);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(3);
    [[maybe_unused]] constexpr uint32_t num_links = get_compile_time_arg_val(4);
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
    // sender->relay pipe args (owning sender NOC coords + 3 flow-control sem ids). Read BEFORE the
    // fabric-connection args (which follow), matching the factory append order.
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t relay_data_ready_sem_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t relay_credits_sem_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t relay_buf_addr_sem_id = get_arg_val<uint32_t>(rt_args_idx++);
    // [debug] per-relay index (== sender index); forms the eth-router detailed flow-log window id.
    const uint32_t combine_window_index = get_arg_val<uint32_t>(rt_args_idx++);
    // fabric-connection args are appended after this by append_fabric_connection_rt_args (read below).

    // ===== Fabric connections (FABRIC_1D array + per-target unicast handshake) =====
    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);
    auto* sem_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    open_direction_connections_barrier(directions, fabric_connections);

    // ===== Init semaphore exchange =====
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

    // [debug] Open the detailed flow-control logging window on each connected eth router. window_id =
    // 100 + chip*10 + relay_index so a dumped trace file ties back to this relay endpoint without host-side
    // mapping. The relay is the fabric endpoint under USE_RELAY, so it owns the window (the sender's
    // writer_combine fabric/logging path is dead code -- it returns before reaching it).
    const uint32_t combine_window_id = 100 + src_chip_id * 10 + combine_window_index;
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            tt::tt_fabric::start_detailed_logging(
                fabric_connections[d].edm_noc_x, fabric_connections[d].edm_noc_y, combine_window_id);
        }
    }

    // Publish our c_24 base to the owning sender (into its relay_buf_addr sem L1) so it knows where to
    // NOC-write tokens. Done AFTER the init handshake; the sender spins on this until non-zero.
    const uint32_t relay_buf_base = get_write_ptr(cb_relay_buf);
    noc_inline_dw_write<InlineWriteDst::L1>(
        get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(relay_buf_addr_sem_id)), relay_buf_base);
    noc_async_writes_flushed();

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

    // ===== CB consumer: read sender-produced tokens from c_24 and forward to fabric =====
    // The sender (writer_combine) produces route_info + payload into our c_24 ring and terminates the
    // stream with a ROUTE_INFO_SENTINEL slot. For each real slot we read its route_info (route dir,
    // distance, page idx), fabric-send its payload, then return the slot's credit so the sender can
    // refill it; a sentinel breaks the loop. The send is blocking (fabric_send_noc_unicast +
    // noc_async_writes_flushed), so the slot's payload read has completed before we return the credit —
    // the sender cannot overwrite the slot until it sees that credit, so there is no torn read.
    {
        const uint32_t relay_slot_size = l1_alignment + aligned_output_page_size;
        const uint32_t relay_buf_base_c = get_write_ptr(cb_relay_buf);
        volatile tt_l1_ptr uint32_t* data_ready_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(relay_data_ready_sem_id));
        const uint64_t sender_credits_noc_addr =
            get_noc_addr(sender_noc_x, sender_noc_y, get_semaphore(relay_credits_sem_id));

        uint32_t consumed = 0;
        while (true) {
            // Wait until slot `consumed` is filled: the producer monotonically ++ data_ready per token.
            while (true) {
                invalidate_l1_cache();
                if (*data_ready_ptr > consumed) {
                    break;
                }
            }
            const uint32_t r = consumed % RELAY_SLOTS;
            const uint32_t slot_base = relay_buf_base_c + r * relay_slot_size;
            volatile tt_l1_ptr uint32_t* ri = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base);
            const uint32_t route = ri[0];

            if (route == ROUTE_INFO_SENTINEL) {
                // Sender signalled end-of-stream. Free the sentinel slot and stop.
                noc_semaphore_inc<true>(sender_credits_noc_addr, 1);
                consumed++;
                break;
            }

            const uint32_t distance = ri[1];
            const uint32_t output_page_idx = ri[2];
            const uint32_t payload_addr = slot_base + l1_alignment;

            ccl_routing_utils::line_unicast_route_info_t pkt_route_info{};
            pkt_route_info.distance_in_hops = static_cast<uint16_t>(distance);
            auto& payload_sender = fabric_connections[route];

            ccl_routing_utils::fabric_set_line_unicast_route(
                pkt_hdr_for_route_helper(unicast_packet_header), pkt_route_info);
            fabric_send_noc_unicast<fabric_max_packet_size>(
                output_addr_gen,
                payload_sender,
                unicast_packet_header,
                payload_addr,
                output_page_idx,
                (int)aligned_output_page_size,
                l1_alignment);
            noc_async_writes_flushed();  // payload has departed L1 -> slot's read done, safe to free

            // Return the slot's credit so the sender can refill it.
            noc_semaphore_inc<true>(sender_credits_noc_addr, 1);
            consumed++;
        }
    }

    // Drain pending local NOC writes before the exit handshake / connection teardown.
    noc_async_write_barrier();

    // [debug] Close the detailed logging window on each connected eth router (after the barrier above, so the
    // last data packet has departed L1). The router finalizes its trace to DRAM on seeing this.
    for (uint32_t d = 0; d < 4; d++) {
        if (directions[d]) {
            tt::tt_fabric::stop_detailed_logging(fabric_connections[d].edm_noc_x, fabric_connections[d].edm_noc_y);
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
}
