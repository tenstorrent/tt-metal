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

using tt::tt_fabric::NocUnicastAtomicIncCommandHeader;
using tt::tt_fabric::NocUnicastCommandHeader;
using tt::tt_fabric::WorkerToFabricEdmSender

template<uint32_t NumDevices>
inline uint32_t get_device_idx_from_batch_idx(const uint32_t b){
    return b/NumDevices
}

// output per device is [K, B/D, 1, H]
template<uint32_t BathSize, uint32_t NumDevices> 
inline uint32_t get_output_page_idx(const uint32_t b, const uint32_t k){
    return k*BatchSize/NumDevices + b;
}

// commonize me!
template<uint8_t Size>
inline void open_direction_connections(
    const std::array<bool,Size> & directions, 
    std::array<WorkerToFabricEdmSender,Size> & connections, 
    size_t rt_args_idx){
    for (uint32_t i = 0; i < 4; i++) {
        if (directions[i]) {
            fabric_connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
            fabric_connections[i].open();
        }
    }
}

template<uint8_t Size>
inline void close_direction_connections(std::array<WorkerToFabricEdmSender,Size> & connections){
    for(auto i = 0; i< 4;++i){
        if(connections[i].is_logically_connected()){
            connections[i].close();
        }
    }
}


void kernel_main() {

    constexpr uint32_t metadata_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_experts_cb_id
    constexpr uint32_t packet_header_cb_id
    constexpr uint32_t batch_size
    constexpr uint32_t num_devices
    constexpr uint32_t src_chip_id =
    constexpr uint32_t data_size_bytes 
    
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);  // ew_dim
    constexpr std::array<bool,4> directions = {
        (bool)get_compile_time_arg_val(32),
        (bool)get_compile_time_arg_val(33),
        (bool)get_compile_time_arg_val(34),
        (bool)get_compile_time_arg_val(35)};

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr uint8_t routes[num_devices] = ROUTE;
    
    const uint32_t global_semaphore_address
    
    const uint32_t rt_arg_count = ...
    std::array<WorkerToFabricEdmSender,Size> fabric_connections;
    open_direction_connections(directions,fabric_connections,rt_arg_count);
    
    InterleavedAddrGen output_addrgen{};
    
    volatile PACKET_HEADER_TYPE * packet_headers[2];
    for(uint8_t i =0;i<2;++i){
        cb_reserve_back(packet_header_cb_id,1);
        const uint32_t packet_header_addr = get_read_ptr(packet_header_cb_id);
        packet_headers[i] = unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_addr);
        cb_push_back(packet_header_cb_id,1);
    }
    
    cb_wait_front(local_experts_cb_id,1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(local_experts_cb_id);

    
    for(uint32_t b=0;b<batch_size;++b){
        cb_wait_front(metadata_tensor_cb_id,1);
        const uint32_t metadata_l1_addr = get_read_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(metadata_l1_addr);
        
        for(uint32_t e=0; e<num_local_experts;++e){
            
            const auto & expert_idx = local_experts_ptr[e];
            const auto [found, k_idx] find_if<selected_experts_k,true>(metadata_ptr,expert_idx)
            if(found){
                cb_wait_front(data_cb_id,1);
                const uint32_t src_data_l1_ptr = get_write_ptr(data_cb_id,);
                
                // figure out output page index, noc address.
                const uint32_t output_page_idx = get_output_page_idx(b,k);
                const uint64_t output_noc_addr = get_noc_addr(output_page_idx, output_addrgen);
                
                // figure out which device to send data to and routing
                const auto & device_idx = get_device_idx_from_batch_idx(b);
                const auto & dest_chip_id = dest_chip_ids[device_idx];
                
                if(dest_chip_id==src_chip_id){
                    noc_async_write(src_data_l1_ptr,output_noc_addr,data_size_bytes);
                    noc_async_write_barrier();
                }
                else{
                    const auto & dest_mesh_id = dest_mesh_ids[device_idx];
                    const auto & route = routers[device_idx];
                    
                    packet_headers[0]->to_noc_unicast_write(
                        NocUnicastCommandHeader{output_noc_addr}, data_size_bytes);
                    
                    fabric_set_unicast_route(
                        (LowLatencyMeshPacketHeader*)packet_header_buffer_address,
                        (eth_chan_directions)fabric_connections[route].direction,
                        src_chip_id,
                        dest_chip_id,
                        dest_mesh_id,
                        mesh_cols);
                        
                    // move this from Saad's kernel into shared code.
                    send_packet(
                        unicast_packet_header,
                        output_token_write_addr,
                        input_token_read_addr,
                        input_page_size,
                        fabric_connections[route]);
                    }
                }
                                
                cb_pop_front(data_cb_id,1);
            }
        }
    }
    
    const uint64_t global_noc_semaphore_address = get_noc_addr(global_semaphore_address);
    // "multicast" semaphore increment to let other devices know we are done
    for(uint32_t device_idx=0;device_idx < num_devices;++device_idx){
        const auto & dest_chip_id = dest_chip_ids[device_idx];
        
        if (dest_chip_ids[d] == src_chip_id) {

            noc_semaphore_inc(get_noc_addr(global_semaphore_address), 1);
            noc_async_atomic_barrier();
        } else {
            const auto & dest_mesh_id = dest_mesh_ids[device_idx];
            const auto & route = routers[device_idx];
            
            packet_header[1]->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{global_noc_semaphore_address, 1, 32, true});

            fabric_set_unicast_route(
                (LowLatencyMeshPacketHeader*)packet_header2[1],
                (eth_chan_directions)fabric_connections[route].direction,
                src_chip_id,
                dest_chip_id,
                dest_mesh_id,
                mesh_cols);
            }
    }
    
    close_direction_connections(fabric_connections);
    
    auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_noc_semaphore_address);
    noc_semaphore_wait(semaphore_ptr, 1);

}
