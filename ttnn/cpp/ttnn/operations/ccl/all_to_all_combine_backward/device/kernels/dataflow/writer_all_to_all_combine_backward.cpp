// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Backward writer kernel for all_to_all_combine.
//
// For each local token t_global and each k-slot, reads the k-th grad page and
// routes it to the expert device d' that contributed that slot in the forward pass.
//
// locally_reduced=false:
//   output_page_idx = e_local * (batch*seq) + t_global
//   Each (device, t_global) pair receives exactly one gradient (no conflict).
//
// locally_reduced=true:
//   output_page_idx = t_global  (only 1 expert slot per device)
//   At most one gradient is sent per destination device per token
//   (first k-slot mapping to that device wins, mirroring forward's break-after-first logic).

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
    constexpr uint32_t metadata_cb_id          = get_compile_time_arg_val(0);
    constexpr uint32_t expert_device_map_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id     = get_compile_time_arg_val(2);
    constexpr uint32_t data_cb_id              = get_compile_time_arg_val(3);
    constexpr uint32_t batch_size              = get_compile_time_arg_val(4);   // global batch
    constexpr uint32_t seq_size               = get_compile_time_arg_val(5);   // global seq
    constexpr uint32_t selected_experts_k     = get_compile_time_arg_val(6);
    constexpr uint32_t num_experts            = get_compile_time_arg_val(7);
    constexpr uint32_t num_devices            = get_compile_time_arg_val(8);
    constexpr uint32_t src_chip_id            = get_compile_time_arg_val(9);
    constexpr uint32_t data_size_bytes        = get_compile_time_arg_val(10);
    constexpr uint32_t alignment              = get_compile_time_arg_val(11);
    constexpr uint32_t mesh_rows              = get_compile_time_arg_val(12);
    constexpr uint32_t mesh_cols              = get_compile_time_arg_val(13);
    constexpr uint32_t fabric_max_packet_size_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t linearized_mesh_coord  = get_compile_time_arg_val(15);
    constexpr tt::tt_fabric::Topology topology = tt::tt_fabric::Topology(get_compile_time_arg_val(16));
    constexpr bool locally_reduced            = get_compile_time_arg_val(17);
    constexpr auto output_args = TensorAccessorArgs<18>();

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

    constexpr uint32_t global_tokens = batch_size * seq_size;

    constexpr uint8_t Num_Directions = 4;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    const std::array<bool, Num_Directions> directions = DIRECTIONS;

    // ---- Runtime args ----
    size_t rt_arg_count = 0;
    const auto output_base_addr       = get_arg_val<uint32_t>(rt_arg_count++);
    const auto global_semaphore_addr  = get_arg_val<uint32_t>(rt_arg_count++);
    const auto init_semaphore_addr    = get_arg_val<uint32_t>(rt_arg_count++);
    const uint32_t token_global_start = get_arg_val<uint32_t>(rt_arg_count++);
    const uint32_t token_global_end   = get_arg_val<uint32_t>(rt_arg_count++);

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
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] opening connections barrier\n";
    open_direction_connections_barrier(directions, fabric_connections);
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] sending init semaphore\n";
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        replicate_axis,
        num_devices>(fabric_connections, packet_headers[1], dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] waiting expert_device_map CB\n";
    // Wait for expert device map from reader (built once, reused for all tokens)
    cb_wait_front(expert_device_map_cb_id, 1);
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] got expert_device_map\n";
    auto expert_map_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(expert_device_map_cb_id));
    const volatile tt_l1_ptr uint16_t* expert_to_device_ptr    = expert_map_ptr;
    const volatile tt_l1_ptr uint16_t* expert_to_local_idx_ptr = expert_map_ptr + num_experts;

    bool needs_barrier = false;
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] waiting init semaphore (need " << (uint32_t)(replicate_group_devices - 1) << ")\n";
    noc_semaphore_wait((uint32_t*)init_semaphore_addr, replicate_group_devices - 1);
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] init sem passed, t=[" << token_global_start << "," << token_global_end << ")\n";
    noc_semaphore_set((uint32_t*)init_semaphore_addr, 0);

    // Per-device sent tracking for locally_reduced (small array, num_devices <= 32)
    uint8_t sent_to_device[num_devices] = {};

    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] entering token loop\n";
    for (uint32_t t_global = token_global_start; t_global < token_global_end; ++t_global) {
        const bool first_token = (t_global == token_global_start);
        if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0: waiting metadata\n"; }
        cb_wait_front(metadata_cb_id, 1);
        if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0: got metadata\n"; }
        const uint32_t metadata_l1_addr = get_read_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        if constexpr (locally_reduced) {
            // Reset per-device tracking for this token
            for (uint32_t d = 0; d < num_devices; ++d) {
                sent_to_device[d] = 0;
            }
        }

        for (uint32_t k = 0; k < selected_experts_k; ++k) {
            if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0 k=" << k << " waiting data\n"; }
            cb_wait_front(data_cb_id, 1);
            if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0 k=" << k << " got data\n"; }
            const uint32_t src_data_l1_ptr = get_read_ptr(data_cb_id);

            invalidate_l1_cache();
            const uint16_t expert_idx    = metadata_ptr[k];
            invalidate_l1_cache();
            const uint32_t dest_device_idx = expert_to_device_ptr[expert_idx];

            // Only send to devices within the replicate group (same column for axis=0,
            // same row for axis=1). The forward combine only gathers within the group,
            // so the backward should only scatter within the group.
            bool in_group = false;
            for (uint32_t d = device_begin_idx; d < device_end_idx; d += device_stride) {
                if (d == dest_device_idx) { in_group = true; break; }
            }
            bool do_send = in_group;
            if constexpr (locally_reduced) {
                // Only send to each device once per token (first k wins),
                // mirroring the forward's break-after-first-local-expert logic.
                if (do_send && sent_to_device[dest_device_idx]) {
                    do_send = false;
                } else if (do_send) {
                    sent_to_device[dest_device_idx] = 1;
                }
            }

            if (do_send) {
                uint32_t output_page_idx;
                if constexpr (locally_reduced) {
                    // Forward input was [1, B_global, S, H]: one expert slot per device
                    output_page_idx = t_global;
                } else {
                    invalidate_l1_cache();
                    const uint32_t e_local = expert_to_local_idx_ptr[expert_idx];
                    // Forward input was [experts_per_device, B_global, S, H]
                    output_page_idx = e_local * global_tokens + t_global;
                }

                const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
                const uint8_t dest_chip_id_val  = dest_chip_ids[dest_device_idx];

                if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0 k=" << k << " send e=" << expert_idx << " dest=" << dest_device_idx << " pg=" << output_page_idx << "\n"; }
                if (dest_device_idx == linearized_mesh_coord) {
                    noc_async_write(src_data_l1_ptr, output_noc_addr, data_size_bytes);
                    needs_barrier = true;
                    noc_async_writes_flushed();
                    if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0 k=" << k << " local write done\n"; }
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
                    if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0 k=" << k << " fabric send done\n"; }
                }
            }

            cb_pop_front(data_cb_id, 1);
        }

        if (first_token) { DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] t0 all k done, popping metadata\n"; }
        cb_pop_front(metadata_cb_id, 1);
    }

    cb_pop_front(expert_device_map_cb_id, 1);
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] token loop done, waiting final sem\n";

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

    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] sent final sems, closing connections\n";
    close_direction_connections(directions, fabric_connections);

    auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] waiting final sem (need " << (uint32_t)replicate_group_devices << ")\n";
    noc_semaphore_wait(semaphore_ptr, replicate_group_devices);
    DPRINT << "[BWD WRITER " << linearized_mesh_coord << "] DONE\n";
    noc_semaphore_set(semaphore_ptr, 0);
}
