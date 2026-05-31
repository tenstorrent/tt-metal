// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Sender RISCV_0 kernel — fabric writer.
//
// Tile layout (IS_TILE_LAYOUT): per-entry pipeline with no sender reader RISC. This
// kernel owns:
//   1. Startup address handshake to the untilize core (publishes c_4/c_5/c_6 base
//      L1 addresses, then NOC-incs untilize's addr_ready semaphore).
//   2. Fabric init / send.
//   3. Per-entry direct-consume of writer CBs (c_4/c_5/c_6, writer_cb_size slots each):
//        - noc_semaphore_wait_min(data_avail, consumed+1)
//        - read route_info[0] at slot = consumed % writer_cb_size
//        - ROUTE_INFO_SENTINEL: break.
//        - regular: fabric-send payload + metadata from the slot; flush; NOC-inc
//          untilize's space_avail directly so the slot becomes refillable.
//
// Row-major layout (no IS_TILE_LAYOUT): standard cb_wait_front / cb_pop_front on
// c_4 / c_5 / c_6 pushed by the row-major sender reader.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

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

    // TensorAccessorArgs for all 7 tensors (starting at index 49)
    constexpr auto input_args = TensorAccessorArgs<49>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

#ifdef IS_TILE_LAYOUT
    // Writer-only tile-layout extras appended after the shared TensorAccessorArgs prefix.
    constexpr uint32_t writer_extra_args_base = dispatch_table_args.next_compile_time_args_offset();
    constexpr uint32_t writer_cb_size = get_compile_time_arg_val(writer_extra_args_base + 0);
    constexpr uint32_t cb_route_info_2_id = get_compile_time_arg_val(writer_extra_args_base + 1);
    constexpr uint32_t cb_payload_for_writer_2_id = get_compile_time_arg_val(writer_extra_args_base + 2);
    constexpr uint32_t cb_metadata_for_writer_2_id = get_compile_time_arg_val(writer_extra_args_base + 3);
    constexpr uint32_t route_info_slot_stride = l1_alignment;
#endif

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

#ifdef IS_TILE_LAYOUT
    // u1 handshake + credit semaphores (c_4/c_5/c_6)
    uint32_t addr_ready_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_c4_addr_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_c5_addr_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_c6_addr_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t untilize_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t untilize_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t data_avail_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t space_avail_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    // u2 handshake + credit semaphores (c_16/c_17/c_18)
    uint32_t addr_ready_u2_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_c16_addr_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_c17_addr_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_c18_addr_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t untilize_u2_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t untilize_u2_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t data_avail_u2_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t space_avail_u2_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
#endif

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

#ifdef IS_TILE_LAYOUT
    // ---- u1 address handshake (c_4/c_5/c_6) ----
    uint32_t writer_route_base = get_write_ptr(cb_route_info_id);
    uint32_t writer_payload_base = get_write_ptr(cb_payload_for_writer_id);
    uint32_t writer_metadata_base = get_write_ptr(cb_metadata_for_writer_id);

    uint32_t addr_ready_sem_l1_offset = get_semaphore(addr_ready_semaphore_id);
    uint32_t cross_c4_addr_sem_l1_offset = get_semaphore(cross_c4_addr_semaphore_id);
    uint32_t cross_c5_addr_sem_l1_offset = get_semaphore(cross_c5_addr_semaphore_id);
    uint32_t cross_c6_addr_sem_l1_offset = get_semaphore(cross_c6_addr_semaphore_id);

    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cross_c4_addr_sem_l1_offset) = writer_route_base;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cross_c5_addr_sem_l1_offset) = writer_payload_base;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cross_c6_addr_sem_l1_offset) = writer_metadata_base;
    noc_async_write(
        cross_c4_addr_sem_l1_offset,
        get_noc_addr(untilize_noc_x, untilize_noc_y, cross_c4_addr_sem_l1_offset),
        sizeof(uint32_t));
    noc_async_write(
        cross_c5_addr_sem_l1_offset,
        get_noc_addr(untilize_noc_x, untilize_noc_y, cross_c5_addr_sem_l1_offset),
        sizeof(uint32_t));
    noc_async_write(
        cross_c6_addr_sem_l1_offset,
        get_noc_addr(untilize_noc_x, untilize_noc_y, cross_c6_addr_sem_l1_offset),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(untilize_noc_x, untilize_noc_y, addr_ready_sem_l1_offset), 1);
    noc_async_atomic_barrier();

    // ---- u2 address handshake (c_16/c_17/c_18) ----
    uint32_t writer_route_base_2 = get_write_ptr(cb_route_info_2_id);
    uint32_t writer_payload_base_2 = get_write_ptr(cb_payload_for_writer_2_id);
    uint32_t writer_metadata_base_2 = get_write_ptr(cb_metadata_for_writer_2_id);

    uint32_t addr_ready_u2_sem_l1_offset = get_semaphore(addr_ready_u2_semaphore_id);
    uint32_t cross_c16_addr_sem_l1_offset = get_semaphore(cross_c16_addr_semaphore_id);
    uint32_t cross_c17_addr_sem_l1_offset = get_semaphore(cross_c17_addr_semaphore_id);
    uint32_t cross_c18_addr_sem_l1_offset = get_semaphore(cross_c18_addr_semaphore_id);

    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cross_c16_addr_sem_l1_offset) = writer_route_base_2;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cross_c17_addr_sem_l1_offset) = writer_payload_base_2;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cross_c18_addr_sem_l1_offset) = writer_metadata_base_2;
    noc_async_write(
        cross_c16_addr_sem_l1_offset,
        get_noc_addr(untilize_u2_noc_x, untilize_u2_noc_y, cross_c16_addr_sem_l1_offset),
        sizeof(uint32_t));
    noc_async_write(
        cross_c17_addr_sem_l1_offset,
        get_noc_addr(untilize_u2_noc_x, untilize_u2_noc_y, cross_c17_addr_sem_l1_offset),
        sizeof(uint32_t));
    noc_async_write(
        cross_c18_addr_sem_l1_offset,
        get_noc_addr(untilize_u2_noc_x, untilize_u2_noc_y, cross_c18_addr_sem_l1_offset),
        sizeof(uint32_t));
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(untilize_u2_noc_x, untilize_u2_noc_y, addr_ready_u2_sem_l1_offset), 1);
    noc_async_atomic_barrier();

    DPRINT_DISPATCH(
        "Sender writer: addr handshake done u1=({},{}) u2=({},{})\n",
        untilize_noc_x,
        untilize_noc_y,
        untilize_u2_noc_x,
        untilize_u2_noc_y);
#endif

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

#ifdef IS_TILE_LAYOUT
    volatile tt_l1_ptr uint32_t* data_avail_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_avail_semaphore_id));
    uint64_t untilize_space_avail_noc_addr =
        get_noc_addr(untilize_noc_x, untilize_noc_y, get_semaphore(space_avail_semaphore_id));
    volatile tt_l1_ptr uint32_t* data_avail_u2_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_avail_u2_semaphore_id));
    uint64_t untilize_u2_space_avail_noc_addr =
        get_noc_addr(untilize_u2_noc_x, untilize_u2_noc_y, get_semaphore(space_avail_u2_semaphore_id));

    uint32_t consumed_1 = 0;
    uint32_t consumed_2 = 0;
    bool done_1 = false;
    bool done_2 = false;

        while (!done_1 || !done_2) {
            // ---- u1 entry ----
            if (!done_1) {
                noc_semaphore_wait_min(data_avail_sem_ptr, consumed_1 + 1);
                uint32_t slot = consumed_1 % writer_cb_size;
                volatile tt_l1_ptr uint32_t* route_info =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_route_base + slot * route_info_slot_stride);
                if (route_info[0] == ROUTE_INFO_SENTINEL) {
                    done_1 = true;
                } else {
                    uint32_t distance = route_info[1];
                    uint32_t page_idx = route_info[2];
                    uint32_t payload_addr = writer_payload_base + slot * aligned_output_page_size;
                    uint32_t metadata_addr = writer_metadata_base + slot * aligned_metadata_page_size;
                    DPRINT_DISPATCH("u1 send: route={} page={}\n", route_info[0], page_idx);
#ifdef DEST_CHIP_ID

                    fabric_set_unicast_route<false>(
                        (volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
                    fabric_send_noc_unicast<fabric_max_packet_size>(
                        output_addr_gen,
                        fabric_connections[route_info[0]],
                        unicast_packet_header,
                        payload_addr,
                        page_idx,
                        (int)aligned_output_page_size,
                        l1_alignment);
                    fabric_send_noc_unicast<fabric_max_packet_size>(
                        metadata_addr_gen,
                        fabric_connections[route_info[0]],
                        unicast_packet_header,
                        metadata_addr,
                        page_idx,
                        (int)aligned_metadata_page_size,
                        l1_alignment);
                    noc_async_writes_flushed();
#endif
                    noc_semaphore_inc<true>(untilize_space_avail_noc_addr, 1);
                    consumed_1++;
                }
            }
            // ---- u2 entry ----
            if (!done_2) {
                noc_semaphore_wait_min(data_avail_u2_sem_ptr, consumed_2 + 1);
                uint32_t slot = consumed_2 % writer_cb_size;
                volatile tt_l1_ptr uint32_t* route_info =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_route_base_2 + slot * route_info_slot_stride);
                if (route_info[0] == ROUTE_INFO_SENTINEL) {
                    done_2 = true;
                } else {
                    uint32_t distance = route_info[1];
                    uint32_t page_idx = route_info[2];
                    uint32_t payload_addr = writer_payload_base_2 + slot * aligned_output_page_size;
                    uint32_t metadata_addr = writer_metadata_base_2 + slot * aligned_metadata_page_size;
                    DPRINT_DISPATCH("u2 send: route={} page={}\n", route_info[0], page_idx);
#ifdef DEST_CHIP_ID
                    fabric_set_unicast_route<false>(
                        (volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
                    fabric_send_noc_unicast<fabric_max_packet_size>(
                        output_addr_gen,
                        fabric_connections[route_info[0]],
                        unicast_packet_header,
                        payload_addr,
                        page_idx,
                        (int)aligned_output_page_size,
                        l1_alignment);
                    fabric_send_noc_unicast<fabric_max_packet_size>(
                        metadata_addr_gen,
                        fabric_connections[route_info[0]],
                        unicast_packet_header,
                        metadata_addr,
                        page_idx,
                        (int)aligned_metadata_page_size,
                        l1_alignment);
                    noc_async_writes_flushed();
#endif
                    noc_semaphore_inc<true>(untilize_u2_space_avail_noc_addr, 1);
                    consumed_2++;
                }
            }
        }
#else
    // ===== Row-major path: standard CB protocol on c_4/c_5/c_6 pushed by the sender reader.
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
        cb_pop_front(cb_route_info_id, 1);

        cb_wait_front(cb_payload_for_writer_id, 1);
        cb_wait_front(cb_metadata_for_writer_id, 1);
        uint32_t payload_addr = get_read_ptr(cb_payload_for_writer_id);
        uint32_t metadata_addr = get_read_ptr(cb_metadata_for_writer_id);

        DPRINT_DISPATCH("Fabric send: route={} distance={} page_idx={}\n", route, distance, page_idx);

#ifdef DEST_CHIP_ID
        fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
        fabric_send_noc_unicast<fabric_max_packet_size>(
            output_addr_gen,
            fabric_connections[route],
            unicast_packet_header,
            payload_addr,
            page_idx,
            (int)aligned_output_page_size,
            l1_alignment);

        fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
        fabric_send_noc_unicast<fabric_max_packet_size>(
            metadata_addr_gen,
            fabric_connections[route],
            unicast_packet_header,
            metadata_addr,
            page_idx,
            (int)aligned_metadata_page_size,
            l1_alignment);

        noc_async_writes_flushed();
#endif

        cb_pop_front(cb_payload_for_writer_id, 1);
        cb_pop_front(cb_metadata_for_writer_id, 1);
    }
#endif

#ifdef DEST_CHIP_ID
        // Defensive: drain any pending local NOC writes before fabric atomic-inc traffic,
        // so the exit-sem signal cannot reach peers ahead of the last metadata/payload writes.
        noc_async_write_barrier();

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

    noc_async_full_barrier();

    close_direction_connections(directions, fabric_connections);
#endif
}
