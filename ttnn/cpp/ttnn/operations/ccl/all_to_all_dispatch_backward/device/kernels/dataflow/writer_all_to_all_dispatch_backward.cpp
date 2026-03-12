// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for all_to_all_dispatch backward.
//
// For each grad page, determines the source device from the page position
// in the expanded batch/seq dimension, and sends the gradient back to that
// source device's expanded output slot.
//
// Dispatch forward: input [1, B, S, H] → output [1, B*D_axis, S, H] (shard_dim=1)
//                   or [1, B, S*D_axis, H] (shard_dim=2)
// Dispatch backward (this kernel):
//   input:  grad [1, B*D_axis, S, H] per device
//   output: expanded [D_axis, B, S, H] per device (caller sums dim 0)
//
// Each device writes to its own slot (device_in_group) in the expanded output
// on the source device, avoiding write conflicts.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using tt::tt_fabric::NocUnicastAtomicIncCommandHeader;
using tt::tt_fabric::NocUnicastCommandHeader;
using tt::tt_fabric::WorkerToFabricEdmSender;
using namespace ttnn::operations::ccl::common;

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t packet_header_cb_id          = get_compile_time_arg_val(0);
    constexpr uint32_t data_cb_id                   = get_compile_time_arg_val(1);
    constexpr uint32_t batch_per_device             = get_compile_time_arg_val(2);
    constexpr uint32_t seq_per_device               = get_compile_time_arg_val(3);
    constexpr uint32_t dispatch_devices             = get_compile_time_arg_val(4);
    constexpr uint32_t grad_dim2                    = get_compile_time_arg_val(5);
    constexpr uint32_t device_in_group              = get_compile_time_arg_val(6);
    constexpr uint32_t output_shard_dim             = get_compile_time_arg_val(7);
    constexpr uint32_t num_devices                  = get_compile_time_arg_val(8);
    constexpr uint32_t src_chip_id                  = get_compile_time_arg_val(9);
    constexpr uint32_t data_size_bytes              = get_compile_time_arg_val(10);
    constexpr uint32_t alignment                    = get_compile_time_arg_val(11);
    constexpr uint32_t mesh_rows                    = get_compile_time_arg_val(12);
    constexpr uint32_t mesh_cols                    = get_compile_time_arg_val(13);
    constexpr uint32_t fabric_max_packet_size_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t linearized_mesh_coord        = get_compile_time_arg_val(15);
    constexpr tt::tt_fabric::Topology topology      = tt::tt_fabric::Topology(get_compile_time_arg_val(16));
    constexpr auto output_args = TensorAccessorArgs<17>();

#ifdef REPLICATE_GROUP_AXIS
    constexpr ReplicateGroup replicate_axis = ReplicateGroup(REPLICATE_GROUP_AXIS);
    constexpr uint8_t replicate_group_devices =
        num_devices / (replicate_axis == ReplicateGroup::COLS ? mesh_cols : mesh_rows);
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;

    constexpr uint32_t device_begin_idx = replicate_axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (replicate_axis == ReplicateGroup::COLS)
            ? (col + mesh_rows * mesh_cols)
            : (row * mesh_cols + mesh_cols);
    constexpr uint32_t device_stride = replicate_axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup replicate_axis = ReplicateGroup::NONE;
    constexpr uint8_t replicate_group_devices = num_devices;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_devices;
    constexpr uint32_t device_stride = 1;
#endif

    constexpr uint32_t pages_per_device = batch_per_device * seq_per_device;

    constexpr uint8_t Num_Directions = 4;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    const std::array<bool, Num_Directions> directions = DIRECTIONS;

    // ---- Runtime args ----
    size_t rt_arg_count = 0;
    const auto output_base_addr       = get_arg_val<uint32_t>(rt_arg_count++);
    const auto global_semaphore_addr  = get_arg_val<uint32_t>(rt_arg_count++);
    const auto init_semaphore_addr    = get_arg_val<uint32_t>(rt_arg_count++);
    const uint32_t page_start         = get_arg_val<uint32_t>(rt_arg_count++);
    const uint32_t page_end           = get_arg_val<uint32_t>(rt_arg_count++);

    std::array<WorkerToFabricEdmSender, Num_Directions> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_arg_count);

    const auto output_addrgen = TensorAccessor(output_args, output_base_addr, data_size_bytes);

    volatile PACKET_HEADER_TYPE* packet_headers[2];
    for (uint8_t i = 0; i < 2; ++i) {
        cb_reserve_back(packet_header_cb_id, 1);
        const uint32_t ph_addr = get_read_ptr(packet_header_cb_id);
        packet_headers[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(ph_addr);
        cb_push_back(packet_header_cb_id, 1);
    }

    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_addr);
    open_direction_connections_barrier(directions, fabric_connections);
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        replicate_axis,
        num_devices>(fabric_connections, packet_headers[1], dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    noc_semaphore_wait((uint32_t*)init_semaphore_addr, replicate_group_devices - 1);
    noc_semaphore_set((uint32_t*)init_semaphore_addr, 0);

    bool needs_barrier = false;

    for (uint32_t page_idx = page_start; page_idx < page_end; ++page_idx) {
        cb_wait_front(data_cb_id, 1);
        const uint32_t src_data_l1_ptr = get_read_ptr(data_cb_id);

        // Determine source device and local coordinates from page position
        uint32_t d_src_in_group, b_local, s_local;
        if constexpr (output_shard_dim == 1) {
            // grad shape [1, B*D_axis, S, H] → page = b_global * S + s
            const uint32_t b_global = page_idx / grad_dim2;
            const uint32_t s = page_idx % grad_dim2;
            d_src_in_group = b_global / batch_per_device;
            b_local = b_global % batch_per_device;
            s_local = s;
        } else {
            // grad shape [1, B, S*D_axis, H] → page = b * (S*D_axis) + s_global
            const uint32_t b = page_idx / grad_dim2;
            const uint32_t s_global = page_idx % grad_dim2;
            d_src_in_group = s_global / seq_per_device;
            b_local = b;
            s_local = s_global % seq_per_device;
        }

        // Output page index in expanded output [dispatch_devices, B, S, H]
        const uint32_t output_page_idx =
            device_in_group * pages_per_device + b_local * seq_per_device + s_local;

        // Flat device index of the source device
        const uint32_t dest_device_idx = device_begin_idx + d_src_in_group * device_stride;

        const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
        const uint8_t dest_chip_id_val = dest_chip_ids[dest_device_idx];

        if (dest_device_idx == linearized_mesh_coord) {
            // Local write
            noc_async_write(src_data_l1_ptr, output_noc_addr, data_size_bytes);
            needs_barrier = true;
            noc_async_writes_flushed();
        } else {
            if constexpr (is_1d_topology<topology>()) {
                fabric_send_chip_unicast_noc_unicast_1d<
                    linearized_mesh_coord,
                    topology,
                    mesh_rows,
                    mesh_cols,
                    fabric_max_packet_size_bytes>(
                    output_addrgen,
                    fabric_connections,
                    packet_headers[0],
                    dest_device_idx,
                    src_data_l1_ptr,
                    output_page_idx,
                    data_size_bytes,
                    alignment);
            } else {
                const auto& dest_mesh_id = dest_mesh_ids[dest_device_idx];
                fabric_send_chip_unicast_noc_unicast<
                    src_chip_id,
                    mesh_rows,
                    mesh_cols,
                    fabric_max_packet_size_bytes>(
                    output_addrgen,
                    fabric_connections,
                    packet_headers[0],
                    dest_chip_id_val,
                    dest_mesh_id,
                    src_data_l1_ptr,
                    output_page_idx,
                    data_size_bytes,
                    alignment);
            }
        }

        cb_pop_front(data_cb_id, 1);
    }

    if (needs_barrier) {
        noc_async_write_barrier();
    }

    // Synchronize completion with all devices in the replicate group
    const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore_addr);
    for (uint32_t device_idx = device_begin_idx; device_idx < device_end_idx; device_idx += device_stride) {
        const auto& d_chip_id = dest_chip_ids[device_idx];

        if (device_idx == linearized_mesh_coord) {
            noc_semaphore_inc(global_noc_semaphore_addr, 1);
            noc_async_atomic_barrier();
        } else if (is_configured_target<linearized_mesh_coord, mesh_rows, mesh_cols, replicate_axis>(device_idx)) {
            if constexpr (is_1d_topology<topology>()) {
                fabric_send_chip_unicast_noc_unicast_semaphore_only_1d<
                    linearized_mesh_coord,
                    topology,
                    mesh_rows,
                    mesh_cols>(
                    fabric_connections, packet_headers[1], device_idx, global_noc_semaphore_addr, 1, true);
            } else {
                const auto& d_mesh_id = dest_mesh_ids[device_idx];
                fabric_send_chip_unicast_noc_unicast_semaphore_only<src_chip_id, mesh_rows, mesh_cols>(
                    fabric_connections,
                    packet_headers[1],
                    d_chip_id,
                    d_mesh_id,
                    global_noc_semaphore_addr,
                    1,
                    true);
            }
        }
    }

    close_direction_connections(directions, fabric_connections);

    auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);
    noc_semaphore_wait(semaphore_ptr, replicate_group_devices);
    noc_semaphore_set(semaphore_ptr, 0);
}
