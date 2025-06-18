// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "../common.hpp"

using tt::tt_fabric::NocUnicastAtomicIncCommandHeader;
using tt::tt_fabric::NocUnicastCommandHeader;
using tt::tt_fabric::WorkerToFabricEdmSender;

inline void print_uint16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << *ptr << " ";
        }
        DPRINT << ENDL();
    }
}

template <uint32_t BatchSize, uint32_t NumDevices, uint32_t ReplicateDim>
inline uint32_t get_device_idx_from_batch_idx(const uint32_t b) {
    constexpr uint32_t Batch_Per_Device = BatchSize / NumDevices * ReplicateDim;

    return b / Batch_Per_Device;
}

// output per device is [K, B/D* replicate_dim, 1, H]
template <uint32_t BatchSize, uint32_t NumDevices, uint32_t ReplicateDim>
inline uint32_t get_output_page_idx(const uint32_t b, const uint32_t k) {
    constexpr uint32_t Batch_Per_Device = BatchSize / NumDevices * ReplicateDim;
    return k * Batch_Per_Device + b % Batch_Per_Device;  // :(
}

// commonize me!
template <size_t Size>
inline void open_direction_connections(
    const std::array<bool, Size>& directions,
    std::array<WorkerToFabricEdmSender, Size>& connections,
    size_t rt_args_idx) {
    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i]) {
            connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            connections[i].open();
            DPRINT << "COnnection opened: " << i << "\n";
        }
    }
}

// commonize me!

template <size_t Size>
inline void close_direction_connections(
    const std::array<bool, Size>& directions, std::array<WorkerToFabricEdmSender, Size>& connections) {
    for (size_t i = 0; i < Size; ++i) {
        if (directions[i]) {
            connections[i].close();
        }
    }
}

// ! commonize me!
/*
enum eth_chan_directions {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    COUNT = 4,
};*/

template <uint32_t SrcChipId, uint32_t MeshCols, uint32_t MeshRows>
inline uint32_t get_direction(uint32_t dest_chip_id) {
    // if along the same row, we go east or west

    constexpr uint32_t Src_Row = SrcChipId / MeshCols, Src_Col = SrcChipId % MeshCols;
    if (Src_Row == dest_chip_id / MeshCols) {
        return SrcChipId < dest_chip_id ? eth_chan_directions::EAST : eth_chan_directions::WEST;
    }
    // if along the same column, we go north or south
    else if (Src_Col == dest_chip_id % MeshCols) {
        return SrcChipId < dest_chip_id ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
    }
    // if not along the same row or column, we go north or south; north if dest_chip_id is smaller than src_chip_id
    return dest_chip_id < SrcChipId ? eth_chan_directions::NORTH : eth_chan_directions::SOUTH;
}

template <uint32_t SrcChipId, uint32_t MeshCols, uint32_t MeshRows, int32_t MaxPacketSzBytes>
inline void dispatch_input_remote_device(
    const uint32_t dest_chip_id,
    const uint32_t dest_mesh_id,
    int32_t size_bytes,
    uint32_t input_token_read_addr,
    uint64_t output_token_write_addr,
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4>& fabric_connections,
    volatile PACKET_HEADER_TYPE* token_unicast_packet_header) {
    // Clear the header buffer region.

    const uint32_t route = get_direction<SrcChipId, MeshCols, MeshRows>(dest_chip_id);

    // Populate packet header with routing information
    zero_l1_buf((uint32_t*)token_unicast_packet_header, sizeof(PACKET_HEADER_TYPE));
    fabric_set_unicast_route(
        const_cast<LowLatencyMeshPacketHeader*>(token_unicast_packet_header),
        static_cast<eth_chan_directions>(fabric_connections[route].direction),
        SrcChipId,
        dest_chip_id,
        dest_mesh_id,
        MeshCols);

    while (size_bytes > 0) {
        uint32_t curr_packet_size = std::min(MaxPacketSzBytes, size_bytes);

        token_unicast_packet_header->to_noc_unicast_write(
            NocUnicastCommandHeader{output_token_write_addr}, curr_packet_size);

        fabric_connections[route].wait_for_empty_write_slot();

        fabric_connections[route].send_payload_without_header_non_blocking_from_address(
            input_token_read_addr, curr_packet_size);

        fabric_connections[route].send_payload_flush_blocking_from_address(
            (uint32_t)token_unicast_packet_header, sizeof(PACKET_HEADER_TYPE));

        input_token_read_addr += curr_packet_size;
        output_token_write_addr += curr_packet_size;
        size_bytes -= curr_packet_size;
    }
}

void kernel_main() {
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t batch_size = get_compile_time_arg_val(4);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(5);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(6);
    constexpr uint32_t num_devices = get_compile_time_arg_val(7);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(8);
    constexpr uint32_t data_size_bytes = get_compile_time_arg_val(9);  // hidden dim * datum size
    constexpr uint32_t output_is_dram = get_compile_time_arg_val(10);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(11);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(12);  // ew_dim
    constexpr uint32_t replicate_dim = get_compile_time_arg_val(13);
    constexpr uint32_t fabric_max_packet_size_bytes = get_compile_time_arg_val(14);

    // if(src_chip_id != 1 )return;

    constexpr uint8_t Num_Directions = 4;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    const std::array<bool, Num_Directions> directions = DIRECTIONS;

    const auto output_base_addr = get_arg_val<uint32_t>(0);
    const auto global_semaphore_addr = get_arg_val<uint32_t>(1);

    DPRINT << "WRITER 0" << "\n";

    const uint32_t rt_arg_count = 2;
    std::array<WorkerToFabricEdmSender, Num_Directions> fabric_connections;
    open_direction_connections(directions,fabric_connections,rt_arg_count);

    InterleavedAddrGen<output_is_dram> output_addrgen{
        .bank_base_address = output_base_addr, .page_size = data_size_bytes};

    DPRINT << "WRITER 0.5" << "\n";

    volatile PACKET_HEADER_TYPE * packet_headers[2];
    for(uint8_t i =0;i<2;++i){
        cb_reserve_back(packet_header_cb_id,1);
        const uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
        packet_headers[i] = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
        cb_push_back(packet_header_cb_id,1);
    }

    DPRINT << "WRITER 1" << "\n";

    cb_wait_front(local_experts_cb_id,1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(local_experts_cb_id));

    // print_uint16_pages(get_read_ptr(local_experts_cb_id),num_local_experts,1);

    for(uint32_t b=0;b<batch_size;++b){
        // DPRINT<< " WAIT FOR METADATA: "<<b<<"\n";
        cb_wait_front(metadata_cb_id, 1);
        const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);
        // DPRINT<<"METADATA: "<<b<<"\n";
        //         print_uint16_pages(metadata_l1_addr,selected_experts_k,1);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto & expert_idx = local_experts_ptr[e];
            const auto [found, k] = detail::find_if<uint16_t, selected_experts_k, true>(metadata_ptr, expert_idx);
            // DPRINT<<"WRITER 2 b: "<< b <<" e:" <<e <<" expert: "<<expert_idx<<"\n";

            if(found){
                // DPRINT<<"WRITER FOUND b: "<< b <<" e:" <<e <<"\n";

                cb_wait_front(data_cb_id,1);
                const uint32_t src_data_l1_ptr = get_write_ptr(data_cb_id);
                // DPRINT<<"WRITER GOT FOUND b: "<< b <<" e: " <<e <<" expert: "<<expert_idx<<"\n";

                // figure out output page index, noc address.
                const uint32_t output_page_idx = get_output_page_idx<batch_size, num_devices, replicate_dim>(b, k);
                // DPRINT<<"dest page idx: "<<output_page_idx<<"\n";

                const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
                // DPRINT<<"dest noc addr: "<<output_noc_addr<<"\n";

                // figure out which device to send data to and routing
                const auto dest_device_idx = get_device_idx_from_batch_idx<batch_size, num_devices, replicate_dim>(b);
                // DPRINT<<"device idx: "<<dest_device_idx<<"\n";

                const auto& dest_chip_id = dest_chip_ids[dest_device_idx];

                // DPRINT<<"WRITER 3.5 b: "<< b <<" e:" <<e <<"\n";

                if(dest_chip_id==src_chip_id){
                    noc_async_write(src_data_l1_ptr,output_noc_addr,data_size_bytes);
                    noc_async_write_barrier();
                } else if (false) {
                    const auto& dest_mesh_id = dest_mesh_ids[dest_device_idx];
                    dispatch_input_remote_device<src_chip_id, mesh_cols, mesh_rows, fabric_max_packet_size_bytes>(
                        dest_chip_id,
                        dest_mesh_id,
                        src_data_l1_ptr,
                        output_noc_addr,
                        data_size_bytes,
                        fabric_connections,
                        packet_headers[0]);

                    // DPRINT<<"WRITER 4 b: "<< b <<" e:" <<e <<"\n";
                }
                // DPRINT<<"WRITER POP b: "<< b <<"\n";
                cb_pop_front(data_cb_id,1);
            }
        }
        cb_pop_front(metadata_cb_id, 1);
    }
    cb_pop_front(local_experts_cb_id, 1);

    DPRINT << "WRITER 5: " << "\n";

    const uint64_t global_noc_semaphore_addr = get_noc_addr(global_semaphore_addr);
    // "multicast" semaphore increment to let other devices know we are done
    for(uint32_t device_idx=0;device_idx < num_devices;++device_idx){
        const auto & dest_chip_id = dest_chip_ids[device_idx];

        if (dest_chip_ids[device_idx] == src_chip_id) {
            // noc_semaphore_inc(get_noc_addr(global_semaphore_addr), 1);
            // noc_async_atomic_barrier();
        } else {
            const auto & dest_mesh_id = dest_mesh_ids[device_idx];
            const auto route = get_direction<src_chip_id, mesh_cols, mesh_rows>(dest_chip_id);

            packet_headers[1]->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{global_noc_semaphore_addr, 1, 32, true});

            fabric_set_unicast_route(
                const_cast<LowLatencyMeshPacketHeader*>(packet_headers[1]),
                static_cast<eth_chan_directions>(fabric_connections[route].direction),
                src_chip_id,
                dest_chip_id,
                dest_mesh_id,
                mesh_cols);

            if (!directions[route]) {
                DPRINT << "BAD ROUTE" << "\n";
            }

            fabric_connections[route].wait_for_empty_write_slot();

            fabric_connections[route].send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(packet_headers[1]), sizeof(PACKET_HEADER_TYPE));
        }
        // DPRINT<<"WRITER 6: "<<dest_chip_id<<"\n";
    }

    close_direction_connections(directions, fabric_connections);

    DPRINT << "WRITER 7: " << "\n";

    auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);
    noc_semaphore_wait(semaphore_ptr, num_devices);

    DPRINT << "WRITER 8: " << "\n";
}
