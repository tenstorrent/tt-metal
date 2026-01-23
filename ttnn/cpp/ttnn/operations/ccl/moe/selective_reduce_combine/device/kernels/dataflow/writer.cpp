// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include "api/debug/dprint_pages.h"

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

// output is [token, k, hidden]
template <uint32_t TokensPerDevice, uint32_t SelectExpertsK>
inline uint32_t get_output_page_idx(const uint32_t t, const uint32_t k) {
    uint32_t t_idx = t % TokensPerDevice;
    return t_idx*SelectExpertsK + k;
}
}  // namespace detail

void kernel_main() {
    constexpr uint32_t metadata_cb_id = get_named_compile_time_arg_val("metadata_cb_id");
    constexpr uint32_t data_cb_id = get_named_compile_time_arg_val("data_cb_id");

    constexpr uint32_t packet_header_cb_id = get_named_compile_time_arg_val("packet_header_cb_id");
    constexpr uint32_t metadata_entry_size = get_named_compile_time_arg_val("metadata_entry_size");

    constexpr uint32_t num_token_parallel_cores = get_named_compile_time_arg_val("num_token_parallel_cores");
    constexpr uint32_t num_data_parallel_cores = get_named_compile_time_arg_val("num_data_parallel_cores");

    constexpr uint32_t noc_x_start = get_named_compile_time_arg_val("noc_x_start");
    constexpr uint32_t noc_y_start = get_named_compile_time_arg_val("noc_y_start");
    constexpr uint32_t noc_x_end = get_named_compile_time_arg_val("noc_x_end");
    constexpr uint32_t noc_y_end = get_named_compile_time_arg_val("noc_y_end");

    constexpr uint32_t select_experts_k = get_named_compile_time_arg_val("select_experts_k");
    constexpr uint32_t num_local_experts = get_named_compile_time_arg_val("num_local_experts");
    constexpr uint32_t global_num_tokens = get_named_compile_time_arg_val("global_num_tokens");  // global token size

    constexpr uint32_t source_token_segment_buffer_size_bytes =
        get_named_compile_time_arg_val("source_token_segment_buffer_size_bytes");
    constexpr uint32_t source_expert_block_size_bytes =
        get_named_compile_time_arg_val("source_expert_block_size_bytes");
    constexpr uint32_t token_size_bytes = get_named_compile_time_arg_val("token_size_bytes");

    constexpr uint32_t alignment = get_named_compile_time_arg_val("alignment");

    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t src_chip_id = get_named_compile_time_arg_val("src_chip_id");
    constexpr uint32_t mesh_rows = get_named_compile_time_arg_val("mesh_rows");
    constexpr uint32_t mesh_cols = get_named_compile_time_arg_val("mesh_cols");  // ew_dim
    constexpr uint32_t fabric_max_packet_size_bytes = get_named_compile_time_arg_val("fabric_max_packet_size_bytes");
    constexpr uint32_t linearized_mesh_coord = get_named_compile_time_arg_val("linearized_mesh_coord");
    constexpr auto topology = tt::tt_fabric::Topology(get_named_compile_time_arg_val("topology"));
    constexpr uint32_t num_mux_workers = get_named_compile_time_arg_val("num_mux_workers");

    constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(0);
    constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(1);
    constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(2);
    constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(3);
    constexpr uint32_t num_mux_clients = get_compile_time_arg_val(4);

    constexpr auto output_ta_args = TensorAccessorArgs<5>();

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
    constexpr uint32_t Replicate_Group = (replicate_axis == ReplicateGroup::COLS) ? mesh_rows : mesh_cols;

    constexpr uint32_t tokens_per_device = global_num_tokens / replicate_group_devices;

    constexpr uint8_t Num_Directions = 4;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    const std::array<bool, Num_Directions> directions = DIRECTIONS;

    size_t rt_arg_count = 0;
    const auto output_base_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto source_token_segment_size_bytes = get_arg_val<uint32_t>(rt_arg_count++);
    const auto dest_token_segment_offset_bytes = get_arg_val<uint32_t>(rt_arg_count++);
    const auto init_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);
    const auto global_semaphore_addr = get_arg_val<uint32_t>(rt_arg_count++);

    // rt_arg_count does not get incremented
    MuxSyncCoreArgs sync_args(rt_arg_count);

    std::array<WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>, Num_Directions> fabric_connections;

    // rt_arg_count does not get incremented
    open_direction_connections_async<
        Num_Directions,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_status_address>(directions, fabric_connections, rt_arg_count);

    const auto output_addrgen = TensorAccessor(output_ta_args, output_base_addr, token_size_bytes);

    volatile PACKET_HEADER_TYPE * packet_headers[2];
    for(uint8_t i =0;i<2;++i){
        cb_reserve_back(packet_header_cb_id,1);
        const uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
        packet_headers[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
        cb_push_back(packet_header_cb_id,1);
    }

    // mux_rt_arg_count does not get incremented
    open_direction_connections_barrier<Num_Directions, fabric_mux_num_buffers_per_channel, fabric_mux_status_address>(
        directions, fabric_connections, rt_arg_count);

    DPRINT << "MUX STARTED" << "\n";

    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_addr);
    if (sync_args.is_sync_core) {
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            replicate_axis,
            num_devices>(fabric_connections, packet_headers[1], dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);
    }

    DPRINT << "INIT SEMAPHORE SENT" << "\n";

    cb_reserve_back(data_cb_id, global_num_tokens);
    const uint32_t src_data_l1_base_addr = get_read_ptr(data_cb_id);

    // if(linearized_mesh_coord==1)
    // tt::data_movement::common::print_bf16_pages(src_data_l1_base_addr,num_local_experts*global_num_tokens*source_token_segment_size_bytes/2,1);

    cb_wait_front(metadata_cb_id, 1);
    const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
    auto * metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_l1_addr);

    // stashed these values in metadata
    const uint32_t token_start = metadata_ptr[global_num_tokens * metadata_entry_size];
    const uint32_t token_end = metadata_ptr[global_num_tokens * metadata_entry_size + 1];
    metadata_ptr += token_start * metadata_entry_size;

    auto* init_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_addr);
    if (sync_args.is_sync_core) {
        noc_semaphore_wait(init_semaphore_ptr, replicate_group_devices - 1);
        const uint64_t semaphore_mc_addr =
            get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, init_semaphore_addr);
        noc_semaphore_set_multicast(
            init_semaphore_addr, semaphore_mc_addr, num_token_parallel_cores * num_data_parallel_cores - 1);
        noc_async_write_barrier();
    } else {
        noc_semaphore_wait(init_semaphore_ptr, replicate_group_devices - 1);
    }
    noc_semaphore_set(init_semaphore_ptr, 0);

    DPRINT << "INIT SEMAPHORE RECEIVED" << "\n";

    uint32_t edt[num_local_experts];
    for (uint32_t e = 0; e < num_local_experts; ++e) {
        edt[e] = 0;
    }

    bool needs_barrier = false;
    for (uint32_t dt = token_start; dt < token_end; ++dt) {
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const uint32_t k = metadata_ptr[e + 1];
            const uint32_t st = metadata_ptr[0];

            if (k != select_experts_k+1) {

                // figure out output page index, noc address.
                const uint32_t output_page_idx =
                    detail::get_output_page_idx<tokens_per_device, select_experts_k>(st, k);

                const uint32_t src_data_l1_addr = src_data_l1_base_addr + e * source_expert_block_size_bytes +
                                                  edt[e]++ * source_token_segment_buffer_size_bytes;

                // figure out which device to send data to and routing
                const auto dest_device_idx = detail::get_device_idx_from_global_token_idx<
                    linearized_mesh_coord,
                    tokens_per_device,
                    mesh_rows,
                    mesh_cols,
                    replicate_axis>(st);

                // DPRINT<<"dt: "<<dt<<" edt: "<<edt[e]<<" st: "<<st<<" e: "<<e<<" dest_device_idx:"
                // <<dest_device_idx<<" output_page_idx:"<<" k: "<<k<<" output_page_idx:"<<output_page_idx<<"\n";
                // tt::data_movement::common::print_bf16_pages(src_data_l1_addr,source_token_segment_size_bytes/2,1);

                // if(linearized_mesh_coord==1){
                //                     DPRINT<<"dt: "<<dt<<" edt: "<<edt[e]<<" st: "<<st<<" e: "<<e<<" dest_device_idx:
                //                     "<<dest_device_idx<<" output_page_idx:"<<" k: "<<k<<" output_page_idx:
                //                     "<<output_page_idx<<"\n";
                //                     tt::data_movement::common::print_bf16_pages(src_data_l1_addr,
                //                     source_token_segment_size_bytes/2,1);
                //                 }

                if (dest_device_idx == linearized_mesh_coord) {
                    const uint64_t output_noc_addr =
                        get_noc_addr(output_page_idx, output_addrgen, dest_token_segment_offset_bytes);
                    noc_async_write(src_data_l1_addr, output_noc_addr, source_token_segment_size_bytes);
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
                            src_data_l1_addr,
                            output_page_idx,
                            source_token_segment_size_bytes,
                            alignment,
                            dest_token_segment_offset_bytes);

                        // DPRINT<<"SENT PAYLOAD"<<"\n";

                    } else {
                        const auto& dest_chip_id = dest_chip_ids[dest_device_idx];
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
                            src_data_l1_addr,
                            output_page_idx,
                            source_token_segment_size_bytes,
                            alignment,
                            dest_token_segment_offset_bytes);
                    }
                }
            }
        }

        metadata_ptr+=metadata_entry_size;
    }
    cb_pop_front(metadata_cb_id, 1);

    if (needs_barrier) {
        noc_async_write_barrier();
    }
    cb_push_back(data_cb_id, 1);
    DPRINT << "PASSED PAYLOAD LOOP \n";

    if (sync_args.is_sync_core) {
        DPRINT << "TERMINATION MASTER WAITING FOR WORKERS \n";
        auto termination_sync_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_args.termination_sync_address);

        noc_semaphore_wait(termination_sync_semaphore_ptr, (num_token_parallel_cores * num_data_parallel_cores) - 1);

        DPRINT << "TERMINATION MASTER WAITING FOR SENDING DEVICE SEMAPHORES \n";

        const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore_addr);
        // "multicast" semaphore increment to let other devices know we are done
        for (uint32_t device_idx = device_begin_idx; device_idx < device_end_idx; device_idx += device_stride) {
            const auto & dest_chip_id = dest_chip_ids[device_idx];

            if (device_idx == linearized_mesh_coord) {
                noc_semaphore_inc(global_noc_semaphore_addr, 1);
                noc_async_atomic_barrier();
            } else if (is_configured_target<linearized_mesh_coord, mesh_rows, mesh_cols, replicate_axis>(device_idx)) {
                if constexpr (is_1d_topology<topology>()) {
                    fabric_send_chip_unicast_noc_unicast_semaphore_only_1d<
                        linearized_mesh_coord,
                        topology,
                        mesh_rows,
                        mesh_cols>(fabric_connections, packet_headers[1], device_idx, global_noc_semaphore_addr, 1, true);
                } else {
                    const auto& dest_mesh_id = dest_mesh_ids[device_idx];
                    const auto& dest_chip_id = dest_chip_ids[device_idx];
                    fabric_send_chip_unicast_noc_unicast_semaphore_only<src_chip_id, mesh_rows, mesh_cols>(
                        fabric_connections,
                        packet_headers[1],
                        dest_chip_id,
                        dest_mesh_id,
                        global_noc_semaphore_addr,
                        1,
                        true);
                }
            }
        }

        auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);
        DPRINT << "TERMINATION MASTER CLOSING MUX \n";

        close_direction_connections<
            Num_Directions,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_termination_signal_address,
            num_mux_workers>(directions, fabric_connections, true, rt_arg_count);

        DPRINT << "TERMINATION MASTER MUX CLOSED \n";

        noc_semaphore_wait(semaphore_ptr, replicate_group_devices);
        noc_semaphore_set(semaphore_ptr, 0);

        DPRINT << "TERMINATION MASTER DONE \n";
    } else {
        // get sync core semaphore noc address
        DPRINT << "termination_master_noc_x: " << sync_args.termination_master_noc_x
               << " termination_master_noc_y: " << sync_args.termination_master_noc_y << "\n";
        uint64_t safe_termination_sync_address = safe_get_noc_addr(
            sync_args.termination_master_noc_x,
            sync_args.termination_master_noc_y,
            sync_args.termination_sync_address,
            0);
        noc_semaphore_inc(safe_termination_sync_address, 1);
        close_direction_connections<
            Num_Directions,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_termination_signal_address>(directions, fabric_connections, false);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
}
