// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

// FABRIC_2D vs 1D dispatch is handled portably via ccl_routing_utils::fabric_set_line_unicast_route
// (templated on packet-header type). Under 1D the helper consumes route_info.distance_in_hops,
// under 2D it consumes route_info.dst_chip_id + dst_mesh_id. The 2D fabric_route (EDM index)
// still has to be recomputed from dest_mesh_ids/dest_chip_ids because the reader's 1D-style
// route_info[0] doesn't match the 2D physical EDM index.

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-9)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_payload_for_writer_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_metadata_for_writer_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(9);

    // Page counts (indices 10-16)
    constexpr uint32_t input_pages = get_compile_time_arg_val(10);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(11);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(12);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(13);
    constexpr uint32_t output_pages = get_compile_time_arg_val(14);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(15);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(16);

    // Page sizes (indices 17-23)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(22);
    constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(23);

    // Operation parameters (indices 24-30)
    constexpr uint32_t num_devices = get_compile_time_arg_val(24);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(25);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(26);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(27);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(28);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(29);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(30);

    // Mesh information (indices 31-35)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(31);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(32);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(33);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(34);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(35);

    // Aligned page sizes (indices 36-42)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(38);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(39);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(40);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(41);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(42);

    // Fabric configuration (indices 43-46)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(43);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(44);
    constexpr uint32_t num_links = get_compile_time_arg_val(45);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(46);

    // Batch configuration (index 47) — read_batch_size not used by writer
    // Index 48 — max_dispatch_buffer_token_size (used by reader only)

    // TensorAccessorArgs for all 7 tensors (starting at index 49)
    constexpr auto input_args = TensorAccessorArgs<49>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatch_table_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatch_core_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_dispatch_cores = get_arg_val<uint32_t>(rt_args_idx++);
    // Separate semaphore for the exit handshake. Reusing init_semaphore_address
    // for both phases is racy: a fast partner's exit-inc can land inside the
    // post-init noc_semaphore_set(0) window and get wiped, deadlocking the
    // pair on dispatch_devices==2 (mesh-2x4 column pair). Mirrors the combine fix.
    uint32_t exit_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
#endif

    DPRINT_DISPATCH(
        "Writer kernel: dispatch_core={} / {} dispatch_devices={}\n",
        dispatch_core_idx,
        num_dispatch_cores,
        dispatch_devices);

#ifdef DEST_CHIP_ID
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* sem_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    open_direction_connections_barrier(directions, fabric_connections);

    // Init semaphore exchange
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        num_devices>(fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, dispatch_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);

    DPRINT_DISPATCH("Fabric setup complete\n");
#endif

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address);

    // Sentinel-terminated fabric send loop
    while (true) {
        cb_wait_front(cb_route_info_id, 1);
        volatile tt_l1_ptr uint32_t* route_info =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_route_info_id));

        uint32_t route = route_info[0];
        if (route == ROUTE_INFO_SENTINEL) {
            cb_pop_front(cb_route_info_id, 1);
            break;
        }
        uint32_t distance = route_info[1];
        uint32_t page_idx = route_info[2];

#ifdef DEST_CHIP_ID
        // CB layout (written by reader_dispatch): [0]=route (1D EDM index), [1]=distance_hops,
        // [2]=page_idx, [3]=dst_chip_index (2D only). Capture per-iteration route state from the
        // CB BEFORE cb_pop_front invalidates the pointer; under 2D recompute the EDM direction
        // from the destination since route_info[0] is 1D-style and doesn't match the 2D index.
        ccl_routing_utils::line_unicast_route_info_t pkt_route_info{};
        uint32_t fabric_route;
        if constexpr (
            std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader> ||
            std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
            const uint32_t dst_chip_device_id = route_info[3];
            pkt_route_info.dst_chip_id = dest_chip_ids[dst_chip_device_id];
            pkt_route_info.dst_mesh_id = dest_mesh_ids[dst_chip_device_id];
            // TODO(#46174): drop the private tt_fabric_api.h dependency once
            // RoutingPlaneConnectionManager exposes a portable (mesh, chip) -> slot lookup.
            fabric_route = static_cast<uint32_t>(
                get_next_hop_router_direction(dest_mesh_ids[dst_chip_device_id], dest_chip_ids[dst_chip_device_id]));
        } else {
            pkt_route_info.distance_in_hops = static_cast<uint16_t>(distance);
            fabric_route = route;
        }
#endif
        cb_pop_front(cb_route_info_id, 1);

        cb_wait_front(cb_payload_for_writer_id, 1);
        cb_wait_front(cb_metadata_for_writer_id, 1);
        uint32_t payload_addr = get_read_ptr(cb_payload_for_writer_id);
        uint32_t metadata_addr = get_read_ptr(cb_metadata_for_writer_id);

        DPRINT_DISPATCH("Fabric send: route={} distance={} page_idx={}\n", route, distance, page_idx);

#ifdef DEST_CHIP_ID
        // Send payload
        ccl_routing_utils::fabric_set_line_unicast_route(
            pkt_hdr_for_route_helper(unicast_packet_header), pkt_route_info);
        fabric_send_noc_unicast<fabric_max_packet_size>(
            output_addr_gen,
            fabric_connections[fabric_route],
            unicast_packet_header,
            payload_addr,
            page_idx,
            (int)aligned_output_page_size,
            l1_alignment);

        // Send metadata
        ccl_routing_utils::fabric_set_line_unicast_route(
            pkt_hdr_for_route_helper(unicast_packet_header), pkt_route_info);
        fabric_send_noc_unicast<fabric_max_packet_size>(
            metadata_addr_gen,
            fabric_connections[fabric_route],
            unicast_packet_header,
            metadata_addr,
            page_idx,
            (int)aligned_metadata_page_size,
            l1_alignment);
        noc_async_writes_flushed();  // Ensure payload+metadata departed L1 before freeing CB slots

#endif

        cb_pop_front(cb_payload_for_writer_id, 1);
        cb_pop_front(cb_metadata_for_writer_id, 1);
    }

#ifdef DEST_CHIP_ID
    // Defensive: drain any pending local NOC writes before fabric atomic-inc traffic,
    // so the exit-sem signal cannot reach peers ahead of the last metadata/payload writes.
    noc_async_write_barrier();

    // Exit semaphore exchange - uses a dedicated semaphore (exit_semaphore_address) and
    // the dedicated sem_packet_header. flush=true (vs the init handshake which uses the
    // default flush=false): the EDM on the receiver holds this atomic-inc until our prior
    // fabric writes (payload + metadata) to that chip have committed there. Without it the
    // small atomic-inc packet can overtake the larger data writes on B's local NOC and the
    // peer would observe sem-reached-threshold before the data has landed in DRAM. At init
    // there are no prior writes to order against, so flush=false saves one EDM round-trip
    // check.
    {
        const uint64_t exit_noc_semaphore_addr = get_noc_addr(exit_semaphore_address);
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            axis,
            num_devices>(
            fabric_connections,
            sem_packet_header,
            dest_chip_ids,
            dest_mesh_ids,
            exit_noc_semaphore_addr,
            /*flush=*/true);

        volatile tt_l1_ptr uint32_t* exit_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(exit_semaphore_address);
        noc_semaphore_wait(exit_sem_ptr, dispatch_devices - 1);
        noc_semaphore_set(exit_sem_ptr, 0);
    }

    // send_init_semaphore_to_configured_targets's portable helper uses
    // fabric_unicast_noc_unicast_atomic_inc (linear/api.h), which calls
    // send_payload_flush_non_blocking_from_address — confirms the write departed worker's NIU
    // but does not mean the packet bytes have landed in EDM's L1 inbox. If we exit the kernel
    // and close_direction_connections runs while the write is still mid-flight, EDM might
    // process its slot bookkeeping before the bytes arrive. A full barrier ensures all writes
    // (and atomics, defensively) have completed before we close.
    noc_async_full_barrier();

    close_direction_connections(directions, fabric_connections);
#endif
}
