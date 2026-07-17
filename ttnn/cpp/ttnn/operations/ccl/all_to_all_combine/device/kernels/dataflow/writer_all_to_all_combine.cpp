// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using tt::tt_fabric::NocUnicastAtomicIncCommandHeader;
using tt::tt_fabric::NocUnicastCommandHeader;
using tt::tt_fabric::WorkerToFabricEdmSender;
using namespace ttnn::operations::ccl::common;

namespace detail {

template <
    uint32_t LinearizedMeshCoord,
    uint32_t TokensPerDevice,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ReplicateGroup Axis>
inline uint32_t get_device_idx_from_global_token_idx(const uint32_t t) {
    constexpr uint32_t Replicate_Group = (Axis == ReplicateGroup::NONE)   ? MeshRows * MeshCols
                                         : (Axis == ReplicateGroup::COLS) ? MeshRows
                                                                          : MeshCols;
    const uint32_t device_in_group = t / TokensPerDevice;

    if constexpr (Axis == ReplicateGroup::NONE) {
        return device_in_group;
    } else if (Axis == ReplicateGroup::ROWS) {
        return (LinearizedMeshCoord / MeshCols) * MeshCols + device_in_group;
    } else {
        return device_in_group * MeshCols + LinearizedMeshCoord % MeshCols;
    }
}

template <uint32_t TokensPerDevice>
inline uint32_t get_output_page_idx(const uint32_t t, const uint32_t k) {
    uint32_t t_idx = t % TokensPerDevice;
    return k * TokensPerDevice + t_idx;
}

// Signal combine completion to the other devices in the replicate group.
//
// Templated (not inlined into kernel_main) so the `if constexpr` below genuinely discards the
// unused branch: fabric_multicast_bidirectional_atomic_inc_1d static_asserts a 1D topology, so it
// must not be instantiated for a 2D (Mesh/Torus) build — which only holds inside a template.
//
// 1D fix (matches selective_reduce_combine / all_to_all_dispatch_metadata): send the completion
// credit as a bidirectional ring/line multicast so it egresses on BOTH arcs and therefore follows
// this device's payloads on whichever arc they took (same-connection, in-order) — it can no longer
// overtake a payload on the opposite arc. Previously the credit re-derived its route via the
// stateful get_route(), which for the antipodal "tie" destination could pick the opposite arc from
// the payload and be observed before the data landed. DoubleAntipodalAtomicInc=true increments the
// antipodal device from both directions, so on a Ring every device receives exactly
// replicate_group_devices credits (see the wait in kernel_main); Linear does no antipodal doubling.
//
// 2D path is unchanged: the credit routes via the deterministic hop router (not get_route), so a
// per-device unicast credit cannot mis-route relative to the payload.
template <
    uint32_t LinearizedMeshCoord,
    tt::tt_fabric::Topology Topology,
    uint32_t MeshRows,
    uint32_t MeshCols,
    ReplicateGroup ReplicateAxis,
    uint32_t SrcChipId,
    uint32_t DeviceBeginIdx,
    uint32_t DeviceEndIdx,
    uint32_t DeviceStride,
    typename SenderType>
inline void send_completion_credits(
    std::array<SenderType, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* packet_header_pos,
    volatile PACKET_HEADER_TYPE* packet_header_neg,
    uint64_t global_noc_semaphore_addr,
    const uint8_t* dest_chip_ids,
    const uint8_t* dest_mesh_ids) {
    if constexpr (is_1d_topology<Topology>() && ReplicateAxis != ReplicateGroup::NONE) {
        fabric_multicast_bidirectional_atomic_inc_1d<
            LinearizedMeshCoord,
            Topology,
            MeshRows,
            MeshCols,
            ReplicateAxis,
            /*DoubleAntipodalAtomicInc=*/(Topology == tt::tt_fabric::Topology::Ring)>(
            fabric_connections, packet_header_pos, packet_header_neg, global_noc_semaphore_addr);
        noc_async_atomic_barrier();
    } else {
        for (uint32_t device_idx = DeviceBeginIdx; device_idx < DeviceEndIdx; device_idx += DeviceStride) {
            if (device_idx == LinearizedMeshCoord) {
                noc_semaphore_inc(global_noc_semaphore_addr, 1);
                noc_async_atomic_barrier();
            } else if (is_configured_target<LinearizedMeshCoord, MeshRows, MeshCols, ReplicateAxis>(device_idx)) {
                fabric_send_chip_unicast_noc_unicast_semaphore_only<SrcChipId, MeshRows, MeshCols>(
                    fabric_connections,
                    packet_header_pos,
                    dest_chip_ids[device_idx],
                    dest_mesh_ids[device_idx],
                    global_noc_semaphore_addr,
                    1,
                    true);
            }
        }
    }
}

}  // namespace detail

void kernel_main() {
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t batch_size = get_compile_time_arg_val(4);
    constexpr uint32_t seq_size = get_compile_time_arg_val(5);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(6);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(7);
    constexpr uint32_t num_devices = get_compile_time_arg_val(8);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(9);
    constexpr uint32_t data_size_bytes = get_compile_time_arg_val(10);  // hidden dim * datum size
    constexpr uint32_t alignment = get_compile_time_arg_val(11);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(12);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(13);  // ew_dim
    constexpr uint32_t fabric_max_packet_size_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(15);
    constexpr tt::tt_fabric::Topology topology = tt::tt_fabric::Topology(get_compile_time_arg_val(16));
    constexpr uint32_t locally_reduced = get_compile_time_arg_val(17);
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
            ? (col + mesh_rows * mesh_cols)   // last is col+(mesh_rows-1)*mesh_cols; add one stride
            : (row * mesh_cols + mesh_cols);  // last is row*mesh_cols+(mesh_cols-1); add one
    constexpr uint32_t device_stride = replicate_axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup replicate_axis = ReplicateGroup::NONE;
    constexpr uint8_t replicate_group_devices = num_devices;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_devices;
    constexpr uint32_t device_stride = 1;
#endif

    constexpr uint32_t Replicate_Group = (replicate_axis == ReplicateGroup::NONE)   ? mesh_rows * mesh_cols
                                         : (replicate_axis == ReplicateGroup::COLS) ? mesh_rows
                                                                                    : mesh_cols;

    constexpr uint32_t tokens = batch_size * seq_size;  // global token size
    constexpr uint32_t tokens_per_device = tokens / replicate_group_devices;

    constexpr uint8_t Num_Directions = 4;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    const std::array<bool, Num_Directions> directions = DIRECTIONS;

    size_t rt_arg_count = 0;
    const auto output_base_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto global_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto init_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const uint32_t token_start_idx = get_arg_val<uint32_t>(rt_arg_count++);
    const uint32_t token_end_idx = get_arg_val<uint32_t>(rt_arg_count++);

    std::array<WorkerToFabricEdmSender, Num_Directions> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_arg_count);

    const auto output_addrgen = TensorAccessor(output_args, output_base_addr);

    volatile PACKET_HEADER_TYPE* packet_headers[3];
    for (uint8_t i = 0; i < 3; ++i) {
        cb_reserve_back(packet_header_cb_id,1);
        const uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
        packet_headers[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
        cb_push_back(packet_header_cb_id,1);
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

    cb_wait_front(local_experts_cb_id,1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(local_experts_cb_id));
    bool needs_barrier = false;
    noc_semaphore_wait((uint32_t*)init_semaphore_addr, replicate_group_devices - 1);
    noc_semaphore_set((uint32_t*)init_semaphore_addr, 0);

    for (uint32_t t = token_start_idx; t < token_end_idx; ++t) {
        cb_wait_front(metadata_cb_id, 1);
        const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto & expert_idx = local_experts_ptr[e];
            const auto [found, k] = find_if<uint16_t, selected_experts_k, true>(metadata_ptr, expert_idx);

            if (found) {
                cb_wait_front(data_cb_id,1);
                const uint32_t src_data_l1_ptr = get_read_ptr(data_cb_id);

                // figure out output page index, noc address.
                const uint32_t output_page_idx = detail::get_output_page_idx<tokens_per_device>(t, k);
                const uint64_t output_noc_addr = output_addrgen.get_noc_addr(output_page_idx);

                // figure out which device to send data to and routing
                const auto dest_device_idx = detail::get_device_idx_from_global_token_idx<
                    linearized_mesh_coord,
                    tokens_per_device,
                    mesh_rows,
                    mesh_cols,
                    replicate_axis>(t);
                const auto& dest_chip_id = dest_chip_ids[dest_device_idx];

                if (dest_device_idx == linearized_mesh_coord) {
                    noc_async_write(src_data_l1_ptr,output_noc_addr,data_size_bytes);
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
                            dest_chip_id,
                            dest_mesh_id,
                            src_data_l1_ptr,
                            output_page_idx,
                            data_size_bytes,
                            alignment);
                    }
                }
                cb_pop_front(data_cb_id,1);

                if constexpr (locally_reduced) {
                    break;
                }
            }
        }

        cb_pop_front(metadata_cb_id, 1);
    }
    cb_pop_front(local_experts_cb_id, 1);
    if (needs_barrier) {
        noc_async_write_barrier();
    }
    const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore_addr);
    // Signal completion to the other devices in the replicate group (see send_completion_credits).
    detail::send_completion_credits<
        linearized_mesh_coord,
        topology,
        mesh_rows,
        mesh_cols,
        replicate_axis,
        src_chip_id,
        device_begin_idx,
        device_end_idx,
        device_stride>(
        fabric_connections,
        packet_headers[1],
        packet_headers[2],
        global_noc_semaphore_addr,
        dest_chip_ids,
        dest_mesh_ids);

    close_direction_connections(directions, fabric_connections);

    auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);
    // Ring (bidirectional, DoubleAntipodal): each device receives replicate_group_devices credits.
    // Linear (bidirectional): replicate_group_devices - 1 (no antipodal doubling).
    // 2D unicast path: local self-inc (1) + (replicate_group_devices - 1) remotes = replicate_group_devices.
    constexpr uint32_t expected_credits =
        (topology == tt::tt_fabric::Topology::Linear) ? (replicate_group_devices - 1) : replicate_group_devices;
    noc_semaphore_wait(semaphore_ptr, expected_credits);
    noc_semaphore_set(semaphore_ptr, 0);
}
