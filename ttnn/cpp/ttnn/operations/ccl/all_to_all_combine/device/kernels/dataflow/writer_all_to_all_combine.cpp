// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
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

// Signal combine completion to the replicate group.
// Templated so the `if constexpr` genuinely discards the unused branch
// (fabric_multicast_bidirectional_atomic_inc_1d static_asserts a 1D topology).
// 1D: send the credit as a bidirectional ring/line multicast so it follows the payload on
// whichever arc it took and cannot overtake it on the opposite arc. On a Ring the antipodal
// device is incremented from both directions (DoubleAntipodalAtomicInc), so every device
// receives replicate_group_devices credits; Linear does no antipodal doubling.
// 2D: credit routes via the deterministic hop router, so a unicast credit cannot mis-route.
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
    const Noc& noc,
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
        noc.async_atomic_barrier();
    } else {
        for (uint32_t device_idx = DeviceBeginIdx; device_idx < DeviceEndIdx; device_idx += DeviceStride) {
            if (device_idx == LinearizedMeshCoord) {
                noc_semaphore_inc(global_noc_semaphore_addr, 1);
                noc.async_atomic_barrier();
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
    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t seq_size = get_arg(args::seq_size);
    constexpr uint32_t selected_experts_k = get_arg(args::selected_experts_k);
    constexpr uint32_t num_local_experts = get_arg(args::num_local_experts);
    constexpr uint32_t num_devices = get_arg(args::num_devices);
    constexpr uint32_t src_chip_id = get_arg(args::src_chip_id);
    constexpr uint32_t data_size_bytes = get_arg(args::data_size_bytes);  // hidden dim * datum size
    constexpr uint32_t alignment = get_arg(args::alignment);
    constexpr uint32_t mesh_rows = get_arg(args::mesh_rows);
    constexpr uint32_t mesh_cols = get_arg(args::mesh_cols);  // ew_dim
    constexpr uint32_t fabric_max_packet_size_bytes = get_arg(args::fabric_max_packet_size_bytes);
    constexpr uint32_t linearized_mesh_coord = get_arg(args::linearized_mesh_coord);
    constexpr tt::tt_fabric::Topology topology = tt::tt_fabric::Topology(get_arg(args::topology));
    constexpr uint32_t locally_reduced = get_arg(args::locally_reduced);

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
    // Fabric connections in {East, West, North, South} order; a direction without a neighbor has no connection.
    const std::array<FabricConnectionBindingToken, Num_Directions> fabric_bindings = {
        fabric::east, fabric::west, fabric::north, fabric::south};
    std::array<bool, Num_Directions> directions;
    for (uint32_t i = 0; i < Num_Directions; i++) {
        directions[i] = tt::tt_fabric::has_fabric_connection(fabric_bindings[i]);
    }

    const uint32_t token_start_idx = get_arg(args::token_start_idx);
    const uint32_t token_end_idx = get_arg(args::token_end_idx);
    experimental::GlobalSemaphore global_semaphore(global_sem::global_semaphore);
    experimental::GlobalSemaphore init_semaphore(global_sem::init_semaphore);

    std::array<WorkerToFabricEdmSender, Num_Directions> fabric_connections;
    for (uint32_t i = 0; i < Num_Directions; i++) {
        if (directions[i]) {
            fabric_connections[i] = WorkerToFabricEdmSender::build_from_binding(fabric_bindings[i]);
            fabric_connections[i].open_start();
        }
    }

    const auto output_addrgen = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_metadata(dfb::metadata);
    DataflowBuffer dfb_local_experts(dfb::local_experts);
    Scratchpad<uint8_t> packet_header(scratch::packet_header);
    DataflowBuffer dfb_data(dfb::data);

    // 3 distinct header buffers: the bidirectional completion send fills packet_headers[1] and [2]
    // back-to-back with non-draining fabric sends, so they must not alias. Take the scratchpad base once
    // and offset each by sizeof(PACKET_HEADER_TYPE).
    volatile PACKET_HEADER_TYPE* packet_headers[3];
    const uint32_t packet_header_base = reinterpret_cast<uint32_t>(&packet_header[0]);
    for (uint8_t i = 0; i < 3; ++i) {
        packet_headers[i] =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_base + i * sizeof(PACKET_HEADER_TYPE));
    }
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore.address());

    for (uint32_t i = 0; i < Num_Directions; i++) {
        if (directions[i]) {
            fabric_connections[i].open_finish();
        }
    }
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        replicate_axis,
        num_devices>(fabric_connections, packet_headers[1], dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    dfb_local_experts.wait_front(1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dfb_local_experts.get_read_ptr());
    bool needs_barrier = false;
    init_semaphore.wait(replicate_group_devices - 1);
    init_semaphore.set(0);

    for (uint32_t t = token_start_idx; t < token_end_idx; ++t) {
        dfb_metadata.wait_front(1);
        // Consumer: read the front page via get_read_ptr (get_write_ptr only coincides while depth==1).
        const uint32_t metadata_l1_addr = dfb_metadata.get_read_ptr();
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto& expert_idx = local_experts_ptr[e];
            const auto [found, k] = find_if<uint16_t, selected_experts_k, true>(metadata_ptr, expert_idx);

            if (found) {
                dfb_data.wait_front(1);
                const uint32_t src_data_l1_ptr = dfb_data.get_read_ptr();

                // figure out output page index, noc address.
                const uint32_t output_page_idx = detail::get_output_page_idx<tokens_per_device>(t, k);

                // figure out which device to send data to and routing
                const auto dest_device_idx = detail::get_device_idx_from_global_token_idx<
                    linearized_mesh_coord,
                    tokens_per_device,
                    mesh_rows,
                    mesh_cols,
                    replicate_axis>(t);
                const auto& dest_chip_id = dest_chip_ids[dest_device_idx];

                if (dest_device_idx == linearized_mesh_coord) {
                    noc.async_write(
                        dfb_data,
                        output_addrgen,
                        data_size_bytes,
                        {.offset_bytes = 0},
                        {.page_id = output_page_idx, .offset_bytes = 0});
                    needs_barrier = true;
                    noc.async_writes_flushed();
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
                dfb_data.pop_front(1);

                if constexpr (locally_reduced) {
                    break;
                }
            }
        }

        dfb_metadata.pop_front(1);
    }
    dfb_local_experts.pop_front(1);
    if (needs_barrier) {
        noc.async_write_barrier();
    }
    const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore.address());
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
        noc,
        fabric_connections,
        packet_headers[1],
        packet_headers[2],
        global_noc_semaphore_addr,
        dest_chip_ids,
        dest_mesh_ids);

    for (uint32_t i = 0; i < Num_Directions; i++) {
        if (directions[i]) {
            fabric_connections[i].close();
        }
    }

    // Credits expected: Ring = replicate_group_devices (antipodal doubled); Linear = replicate_group_devices - 1;
    // 2D unicast = 1 self-inc + (replicate_group_devices - 1) remotes = replicate_group_devices.
    constexpr uint32_t expected_credits =
        (topology == tt::tt_fabric::Topology::Linear) ? (replicate_group_devices - 1) : replicate_group_devices;
    global_semaphore.wait(expected_credits);
    global_semaphore.set(0);
}
