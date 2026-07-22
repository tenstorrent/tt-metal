// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

using tt::tt_fabric::NocUnicastAtomicIncCommandHeader;
using tt::tt_fabric::NocUnicastCommandHeader;
using tt::tt_fabric::WorkerToFabricEdmSender;
using namespace ttnn::operations::ccl::common;

// packet size bytes 4352
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

// output is [select_experts_k ,tokens, hidden]
template <uint32_t TokensPerDevice>
inline uint32_t get_output_page_idx(const uint32_t t, const uint32_t k) {
    uint32_t t_idx = t % TokensPerDevice;
    return k * TokensPerDevice + t_idx;
}

template <bool Enable = false>
struct DoubleBuffer {
private:
    uint32_t idx;

public:
    DoubleBuffer(
        const Noc& /*noc*/,
        const uint32_t /*compute_cores_per_combine_core*/,
        const uint32_t /*sync_semaphore_addr*/,
        size_t& /*rt_arg_count*/) :
        idx(0) {};

    auto& operator++() {
        ++idx;
        return *this;
    }

    auto operator*() { return idx; }
};

template <>
struct DoubleBuffer<true> {
private:
    const Noc& noc;
    const uint32_t compute_cores_per_combine_core;
    const uint32_t sync_semaphore_addr;
    volatile tt_l1_ptr uint32_t* core_coords_ptr;

    bool idx;

public:
    DoubleBuffer(
        const Noc& noc,
        const uint32_t compute_cores_per_combine_core,
        const uint32_t sync_semaphore_addr,
        size_t& rt_arg_count) :
        noc(noc),
        compute_cores_per_combine_core(compute_cores_per_combine_core),
        sync_semaphore_addr(sync_semaphore_addr),
        core_coords_ptr(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(rt_arg_count))),
        idx(false) {
        rt_arg_count += 2 * compute_cores_per_combine_core;
    };

    DoubleBuffer& operator++() {
        noc.async_writes_flushed();

        for (uint32_t c = 0; c < compute_cores_per_combine_core; ++c) {
            const uint64_t sem_noc_addr = safe_get_noc_addr(
                core_coords_ptr[2 * c],
                core_coords_ptr[2 * c + 1],
                sync_semaphore_addr,
                /*noc_id=*/1);
            // Device 2.0 migration: legacy primitive retained: sync_semaphore_addr is a
            // fabric-mux sync address.
            noc_semaphore_inc</*posted=*/true>(sem_noc_addr, 1, /*noc_id=*/1);
        }
        idx = !idx;
        return *this;
    }

    auto operator*() { return idx; }
};

template <uint8_t NumBuffers, uint8_t NumDirections>
void mux_channel_writes_flushed(
    const std::array<bool, NumDirections>& directions,
    const std::array<WorkerToFabricMuxSender<NumBuffers>, NumDirections>& connections) {
    for (uint8_t d = 0; d < NumDirections; ++d) {
        if (directions[d]) {
            while (connections[d].get_num_free_write_slots() != NumBuffers) {
            };
        }
    }
}

}  // namespace detail

void kernel_main() {
    DeviceZoneScopedN("Combine-writer");

    constexpr uint32_t dense_token_maps_cb_id = get_named_compile_time_arg_val("dense_token_maps_cb_id");
    constexpr uint32_t token_counts_cb_id = get_named_compile_time_arg_val("token_counts_cb_id");
    constexpr uint32_t data_cb_id = get_named_compile_time_arg_val("data_cb_id");
    constexpr uint32_t token_activations_cb_id = get_named_compile_time_arg_val("token_activations_cb_id");
    constexpr uint32_t activations_stride_elm = get_named_compile_time_arg_val("activations_stride_elm");
    constexpr uint32_t packet_header_cb_id = get_named_compile_time_arg_val("packet_header_cb_id");
    constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val("num_token_parallel_cores");
    constexpr uint32_t num_data_parallel_cores = get_named_compile_time_arg_val("num_data_parallel_cores");
    constexpr uint32_t num_workers_per_link = get_named_compile_time_arg_val("num_workers_per_link");
    constexpr bool use_init_semaphore = get_named_compile_time_arg_val("use_init_semaphore") == 1;
    constexpr uint32_t noc_x_start = get_named_compile_time_arg_val("noc_x_start");
    constexpr uint32_t noc_y_start = get_named_compile_time_arg_val("noc_y_start");
    constexpr uint32_t noc_x_end = get_named_compile_time_arg_val("noc_x_end");
    constexpr uint32_t noc_y_end = get_named_compile_time_arg_val("noc_y_end");
    constexpr uint32_t num_local_experts = get_named_compile_time_arg_val("num_local_experts");
    constexpr uint32_t global_num_tokens = get_named_compile_time_arg_val("global_num_tokens");  // global token size
    constexpr uint32_t source_token_segment_buffer_size_bytes =
        get_named_compile_time_arg_val("source_token_segment_buffer_size_bytes");
    constexpr uint32_t source_block_size_bytes = get_named_compile_time_arg_val("source_expert_block_size_bytes");
    constexpr uint32_t dense_token_maps_stride_elm = get_named_compile_time_arg_val("dense_token_maps_stride_elm");
    constexpr uint32_t alignment = get_named_compile_time_arg_val("alignment");
    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t src_chip_id = get_named_compile_time_arg_val("src_chip_id");
    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");  // ew_dim
    constexpr uint32_t fabric_max_packet_size_bytes = get_named_compile_time_arg_val("fabric_max_packet_size_bytes");
    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr auto topology = tt::tt_fabric::Topology(get_named_compile_time_arg_val("topology"));
    constexpr uint32_t num_mux_workers_per_link = get_named_compile_time_arg_val("num_mux_workers_per_link");
    constexpr uint32_t compute_sync_semaphore_id = get_named_compile_time_arg_val("compute_sync_semaphore_id");
    constexpr uint32_t compute_cores_per_combine_core =
        get_named_compile_time_arg_val("compute_cores_per_combine_core");
    constexpr bool double_buffer_source = get_named_compile_time_arg_val("double_buffer_source") == 1;
    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(0);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(4);

    constexpr auto output_ta_args = TensorAccessorArgs<5>();

    constexpr ReplicateGroup replicate_axis = ReplicateGroup(REPLICATE_GROUP_AXIS);
    constexpr uint32_t replicate_factor = (replicate_axis == ReplicateGroup::COLS) ? mesh_cols : mesh_rows;
    constexpr uint8_t replicate_group_devices = num_devices / replicate_factor;
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;

    constexpr uint32_t tokens_per_device = global_num_tokens / replicate_group_devices;

    constexpr uint8_t Num_Directions = 4;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    const std::array<bool, Num_Directions> directions = DIRECTIONS;

    size_t rt_arg_count = 0;
    const auto output_base_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto source_token_segment_size_bytes = get_arg_val<uint32_t>(rt_arg_count++);
    const auto dest_token_segment_offset_bytes = get_arg_val<uint32_t>(rt_arg_count++);
    // Device 2.0 migration: legacy primitive retained: init_semaphore_addr is a GlobalSemaphore address.
    const auto init_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto global_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const bool is_init_sync_core = get_arg_val<uint32_t>(rt_arg_count++);

    Noc noc1_obj(1);
    CircularBuffer cb_packet_header(packet_header_cb_id);
    CircularBuffer cb_token_counts(token_counts_cb_id);
    CircularBuffer cb_data(data_cb_id);
    CircularBuffer cb_dense_token_maps(dense_token_maps_cb_id);
    CircularBuffer cb_token_activations(token_activations_cb_id);
    Semaphore<> compute_sync_sem(compute_sync_semaphore_id);

    const auto compute_sync_semaphore_addr = get_semaphore(compute_sync_semaphore_id);

    // rt_arg_count is incremented
    detail::DoubleBuffer<double_buffer_source> db(
        noc1_obj, compute_cores_per_combine_core, compute_sync_semaphore_addr, rt_arg_count);

    // rt_arg_count does not get incremented
    MuxSyncCoreArgs sync_args(rt_arg_count);

    std::array<WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>, Num_Directions> fabric_connections;

    // rt_arg_count does not get incremented
    open_direction_connections_async<
        Num_Directions,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_status_address>(directions, fabric_connections, rt_arg_count);

    const auto output_addrgen = TensorAccessor(output_ta_args, output_base_addr);

    volatile PACKET_HEADER_TYPE* packet_headers[3];
    for (uint8_t i = 0; i < 3; ++i) {
        cb_packet_header.reserve_back(1);
        const uint32_t packet_header_addr = cb_packet_header.get_write_ptr();
        packet_headers[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
        cb_packet_header.push_back(1);
    }

    // mux_rt_arg_count does not get incremented
    open_direction_connections_barrier<Num_Directions, fabric_mux_num_buffers_per_channel, fabric_mux_status_address>(
        directions, fabric_connections, rt_arg_count);

    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_addr);
    if (is_init_sync_core && use_init_semaphore) {
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            replicate_axis,
            num_devices>(fabric_connections, packet_headers[1], dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);
    }

    cb_token_counts.wait_front(1);
    const uint32_t token_counts_l1_addr = cb_token_counts.get_write_ptr();
    auto* token_counts_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_counts_l1_addr);
    uint32_t token_split_offsets[num_local_experts];
    uint32_t token_split_counts[num_local_experts];
    uint32_t token_activation_offsets[num_local_experts];
    for (uint32_t e = 0; e < num_local_experts; ++e) {
        token_split_offsets[e] = token_counts_l1_ptr[num_local_experts + e];
        token_split_counts[e] = token_counts_l1_ptr[num_local_experts + num_local_experts + e];
        token_activation_offsets[e] = token_counts_l1_ptr[num_local_experts + 2 * num_local_experts + e];
    }
    cb_token_counts.pop_front(1);

    cb_data.reserve_back(1);
    const uint32_t src_data_l1_base_addr = cb_data.get_write_ptr();

    cb_dense_token_maps.wait_front(num_local_experts);
    const uint32_t dense_token_maps_l1_addr = cb_dense_token_maps.get_write_ptr();
    auto* dense_token_maps_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense_token_maps_l1_addr);

    cb_token_activations.wait_front(1);
    const uint32_t token_activations_l1_addr = cb_token_activations.get_write_ptr();
    auto* token_activations_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_activations_l1_addr);

    if constexpr (use_init_semaphore) {
        auto* init_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_addr);
        if (is_init_sync_core) {
            // Device 2.0 migration: legacy primitive retained: init_semaphore_addr is a GlobalSemaphore address.
            noc_semaphore_wait(init_semaphore_ptr, replicate_group_devices - 1);
            // swap start/end coordinates because this kernel is using NOC1
            const uint64_t semaphore_mc_addr =
                get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, init_semaphore_addr, /*noc=*/1);
            noc_semaphore_set_multicast(
                init_semaphore_addr,
                semaphore_mc_addr,
                num_token_parallel_cores * num_data_parallel_cores - 1,
                /*linked=*/false,
                /*noc=*/1);
            noc1_obj.async_writes_flushed();

        } else {
            noc_semaphore_wait(init_semaphore_ptr, replicate_group_devices - 1);
        }
        noc_semaphore_set(init_semaphore_ptr, 0);
    }

    uint32_t compute_sync_semaphore_val = compute_cores_per_combine_core;
    for (uint32_t e = 0; e < num_local_experts; ++e) {
        auto* expert_token_activations_ptr =
            token_activations_l1_ptr + token_activation_offsets[e] * activations_stride_elm;

        compute_sync_sem.wait(compute_sync_semaphore_val);

        for (uint32_t dt = 0; dt < token_split_counts[e]; ++dt) {
            const uint32_t st = dense_token_maps_l1_ptr
                [(e * (global_num_tokens + 1) + token_split_offsets[e] + dt) * dense_token_maps_stride_elm];
            uint32_t guard = 0;
            while (expert_token_activations_ptr[0] != st) {
                expert_token_activations_ptr += activations_stride_elm;
                ASSERT(guard++ < global_num_tokens);
            }
            const uint32_t k = expert_token_activations_ptr[1 + e];

            // figure out output page index, noc address.
            const uint32_t output_page_idx = detail::get_output_page_idx<tokens_per_device>(st, k);

            const uint32_t src_data_l1_addr =
                src_data_l1_base_addr + *db * source_block_size_bytes + dt * source_token_segment_buffer_size_bytes;

            // figure out which device to send data to and routing
            const auto dest_device_idx = detail::get_device_idx_from_global_token_idx<
                linearized_mesh_coord,
                tokens_per_device,
                mesh_rows,
                mesh_cols,
                replicate_axis>(st);

            if (dest_device_idx == linearized_mesh_coord) {
                noc1_obj.async_write(
                    CoreLocalMem<uint8_t>(src_data_l1_addr),
                    output_addrgen,
                    source_token_segment_size_bytes,
                    {},
                    {.page_id = output_page_idx, .offset_bytes = dest_token_segment_offset_bytes});
                noc1_obj.async_writes_flushed();
            } else {
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
                    src_data_l1_addr,
                    output_page_idx,
                    source_token_segment_size_bytes,
                    alignment,
                    dest_token_segment_offset_bytes);
            }
        }
        compute_sync_semaphore_val += compute_cores_per_combine_core;
        ++db;
    }

    compute_sync_sem.set(0);

    cb_dense_token_maps.pop_front(num_local_experts);
    cb_token_activations.pop_front(1);
    cb_data.push_back(1);

    noc1_obj.async_write_barrier();

    // In order to ensure that the barrier semaphores land after all of the data has arrived we must wait for the mux
    // cores to send off all of their transactions to the EDM.
    detail::mux_channel_writes_flushed<fabric_mux_num_buffers_per_channel, Num_Directions>(
        directions, fabric_connections);

    if (sync_args.is_sync_core) {
        auto* termination_sync_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_args.termination_sync_address);

        // Device 2.0 migration: legacy primitive retained: termination_sync_address is a
        // fabric-mux sync address (not a get_semaphore<>(id) address).
        noc_semaphore_wait(termination_sync_semaphore_ptr, num_workers_per_link - 1);
        noc_semaphore_set(termination_sync_semaphore_ptr, 0);

        const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore_addr, /*noc=*/1);

        // Topology is a CT arg from the host (already auto-downgraded via ttnn::ccl::get_usable_topology
        // at the op API boundary in moe_compute_device_operation.cpp). For Linear topology, the helper
        // derives per-device positive/negative ranges from linearized_mesh_coord so endpoints elide the
        // absent-direction send via `if constexpr` — required to avoid UB on LINE endpoints.
        fabric_multicast_bidirectional_atomic_inc_1d<
            linearized_mesh_coord,
            topology,
            mesh_rows,
            mesh_cols,
            replicate_axis,
            /*DoubleAntipodalAtomicInc=*/(topology == tt::tt_fabric::Topology::Ring)>(
            fabric_connections, packet_headers[1], packet_headers[2], global_noc_semaphore_addr);

        auto* semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);

        noc1_obj.async_write_barrier();
        noc1_obj.async_atomic_barrier();

        close_direction_connections<
            Num_Directions,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_termination_signal_address,
            num_mux_workers_per_link>(directions, fabric_connections, true, rt_arg_count);

        // Ring + DoubleAntipodalAtomicInc=true: each device receives `replicate_group_devices` inc's
        //   (the antipodal device is incremented from both directions, summing to N senders).
        // Linear: each device receives `replicate_group_devices - 1` inc's (no antipodal doubling;
        //   each sender on the line sends to exactly N-1 other devices).
        constexpr uint32_t expected_dispatch_device_inc =
            (topology == tt::tt_fabric::Topology::Linear) ? (replicate_group_devices - 1) : replicate_group_devices;
        noc_semaphore_wait(semaphore_ptr, expected_dispatch_device_inc);
        noc_semaphore_set(semaphore_ptr, 0);
    } else {
        // get sync core semaphore noc address
        close_direction_connections<
            Num_Directions,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_termination_signal_address>(directions, fabric_connections, false);

        const uint64_t safe_termination_sync_address = safe_get_noc_addr(
            sync_args.termination_master_noc_x,
            sync_args.termination_master_noc_y,
            sync_args.termination_sync_address,
            /*noc=*/1);
        // Device 2.0 migration: legacy primitive retained: termination_sync_address is a
        // fabric-mux sync address (not a get_semaphore<>(id) address)
        noc_semaphore_inc(safe_termination_sync_address, 1, /*noc=*/1);

        noc1_obj.async_write_barrier();
        noc1_obj.async_atomic_barrier();
    }
}
