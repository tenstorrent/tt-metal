// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric Gather Histogram Worker
//
// Subgroup-scoped all-gather along axis 0 of a per-chip `[N_ROWS, W]` UINT32 input,
// producing a `[subgroup_rows * N_ROWS, W]` output on each chip. Each chip writes its
// N_ROWS input pages into its own output block, then fabric-unicasts the same pages
// to every same-column peer's output block. Fabric traffic stays on axis 0 (ReplicateGroup::COLS),
// so distance-based 1D routing on the LowLatencyPacketHeader works on both 1D and 2D fabric.
// The composite op handles axis-1 gathering (if any) as a separate pre-pass.
//
// Synchronization pattern mirrors combine's writer (two init-semaphore exchanges, one
// up-front, one as the exit barrier), both axis-0-filtered.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile-time args =====
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t N_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(7);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(8);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(9);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(10);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(11);
    constexpr uint32_t subgroup_num_devices = get_compile_time_arg_val(12);
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(13);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(14);
    constexpr uint32_t num_links = get_compile_time_arg_val(15);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(16);

    constexpr auto input_args = TensorAccessorArgs<17>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // Peer fan-out is restricted to same-column (axis-0 line), matching the AXIS=cluster_axis=0 -> COLS
    // convention. On a 1D subgroup (mesh_cols=1) COLS reduces to "every peer".
    constexpr ReplicateGroup axis = ReplicateGroup::COLS;

    // The row-block this chip owns within the output is `local_row * N_ROWS`.
    constexpr uint32_t local_row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t output_block_offset = local_row * N_ROWS;

    // ===== Runtime args =====
    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);

    // ===== Read local N_ROWS input pages into L1 scratch =====
    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address);
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address);

    cb_reserve_back(cb_input_id, N_ROWS);
    uint32_t input_l1_addr = get_write_ptr(cb_input_id);
    for (uint32_t p = 0; p < N_ROWS; ++p) {
        noc_async_read_page(p, input_addr_gen, input_l1_addr + p * aligned_input_page_size);
    }
    noc_async_read_barrier();

#ifdef DEST_CHIP_ID
    constexpr uint8_t dest_chip_ids[subgroup_num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[subgroup_num_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* sem_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    open_direction_connections_barrier(directions, fabric_connections);

    // Init barrier: each chip notifies its same-column peers that it's ready to receive.
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        subgroup_num_devices>(
        fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, mesh_rows - 1);
    noc_semaphore_set(init_sem_ptr, 0);
#endif

    // ===== Local write: put my N_ROWS pages into my own output block =====
    for (uint32_t p = 0; p < N_ROWS; ++p) {
        uint64_t local_dst = output_addr_gen.get_noc_addr(output_block_offset + p);
        noc_async_write(input_l1_addr + p * aligned_input_page_size, local_dst, W * sizeof(uint32_t));
    }

#ifdef DEST_CHIP_ID
    // ===== Fabric send: for each same-column peer, write my N_ROWS pages into their output block =====
    for (uint32_t device_idx = 0; device_idx < subgroup_num_devices; ++device_idx) {
        if (device_idx == linearized_mesh_coord) {
            continue;
        }
        if (device_idx % mesh_cols != linearized_mesh_coord % mesh_cols) {
            continue;  // skip different-column peers; axis-1 was handled by the composite's pre-pass
        }

        uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, device_idx);
        uint32_t distance = manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, device_idx);

        for (uint32_t p = 0; p < N_ROWS; ++p) {
            fabric_set_unicast_route<false>(
                (volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
            fabric_send_noc_unicast<fabric_max_packet_size>(
                output_addr_gen,
                fabric_connections[route],
                unicast_packet_header,
                input_l1_addr + p * aligned_input_page_size,
                output_block_offset + p,
                (int)aligned_output_page_size,
                l1_alignment);
        }
    }

    noc_async_write_barrier();

    // Exit barrier: ensure every same-column peer has finished writing into my output before we return.
    {
        const uint64_t exit_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            axis,
            subgroup_num_devices>(
            fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, exit_noc_semaphore_addr);

        volatile tt_l1_ptr uint32_t* exit_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
        noc_semaphore_wait(exit_sem_ptr, mesh_rows - 1);
        noc_semaphore_set(exit_sem_ptr, 0);
    }

    close_direction_connections(directions, fabric_connections);
#else
    noc_async_write_barrier();
#endif

    cb_pop_front(cb_input_id, N_ROWS);
}
