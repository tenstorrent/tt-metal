// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS - Unified P2P + Broadcast
///////////////////////////////////////////////////
constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(4);
constexpr bool is_sender = get_compile_time_arg_val(5);
constexpr uint32_t start_distance_in_hops_forward = get_compile_time_arg_val(6);
constexpr uint32_t range_hops_forward = get_compile_time_arg_val(7);
constexpr uint32_t start_distance_in_hops_backward = get_compile_time_arg_val(8);
constexpr uint32_t range_hops_backward = get_compile_time_arg_val(9);

inline constexpr uint32_t sharded_args_start_idx = 10;

/*
 * Fused TP Partner Writer:
 * 1. Receives data from TP reader via CB
 * 2. Writes data locally to tensor (P2P receiver writer functionality)
 * 3. Multicasts data via fabric down column (Broadcast writer functionality)
 * 4. Synchronizes properly between reader and writer using CB
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // RUNTIME ARGS - Unified P2P + Broadcast
    ///////////////////////////////////////////////////
    DPRINT << "start of TP writer" << ENDL();
    size_t arg_idx = 0;

    // Local tensor args (P2P writer part)
    const uint32_t local_tensor_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_page_id = get_arg_val<uint32_t>(arg_idx++);

    // Broadcast args (fabric multicast part)
    address_t broadcast_tensor_addr = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t page_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t page_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t fabric_arg_idx = arg_idx;

    // Debug: Print compile-time args
    DPRINT << "TP WRITER: cb_id_in=" << (uint32_t)cb_id_in
           << ", packet_size_in_pages=" << (uint32_t)packet_size_in_pages << ", page_size=" << (uint32_t)page_size
           << ENDL();
    DPRINT << "TP WRITER: num_targets_forward_direction=" << (uint32_t)num_targets_forward_direction
           << ", num_targets_backward_direction=" << (uint32_t)num_targets_backward_direction << ENDL();
    DPRINT << "TP WRITER: start_distance_in_hops_forward=" << (uint32_t)start_distance_in_hops_forward
           << ", range_hops_forward=" << (uint32_t)range_hops_forward << ENDL();
    DPRINT << "TP WRITER: start_distance_in_hops_backward=" << (uint32_t)start_distance_in_hops_backward
           << ", range_hops_backward=" << (uint32_t)range_hops_backward << ENDL();

    // Debug: Print runtime args
    DPRINT << "TP WRITER: local_tensor_addr=" << (uint32_t)local_tensor_addr << ", num_pages=" << (uint32_t)num_pages
           << ", start_page_id=" << (uint32_t)start_page_id << ENDL();
    DPRINT << "TP WRITER: broadcast_tensor_addr=" << (uint32_t)broadcast_tensor_addr
           << ", out_ready_sem_bank_addr=" << (uint32_t)out_ready_sem_bank_addr << ENDL();
    DPRINT << "TP WRITER: page_id_start=" << (uint32_t)page_id_start << ", page_id_end=" << (uint32_t)page_id_end
           << ENDL();
    DPRINT << "TP WRITER: wait_output_semaphore=" << (uint32_t)wait_output_semaphore
           << ", reset_global_semaphore=" << (uint32_t)reset_global_semaphore << ENDL();
    DPRINT << "TP WRITER: out_ready_sem_noc0_x=" << (uint32_t)out_ready_sem_noc0_x
           << ", out_ready_sem_noc0_y=" << (uint32_t)out_ready_sem_noc0_y << ENDL();
    DPRINT << "TP WRITER: out_ready_sem_wait_value=" << (uint32_t)out_ready_sem_wait_value
           << ", barrier_sem=" << (uint32_t)barrier_sem << ENDL();
    DPRINT << "TP WRITER: barrier_sem_noc0_x=" << (uint32_t)barrier_sem_noc0_x
           << ", barrier_sem_noc0_y=" << (uint32_t)barrier_sem_noc0_y << ENDL();
    DPRINT << "TP WRITER: num_connections=" << (uint32_t)num_connections << ENDL();

    // Setup local tensor accessor (P2P writer part)
    constexpr auto local_tensor_args = TensorAccessorArgs<10>();
    const auto local_tensor_accessor = TensorAccessor(local_tensor_args, local_tensor_addr, page_size);

    // Setup broadcast fabric connection and tensor accessor
    auto unicast_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto scatter_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    /*
    #ifdef SHARDED
        typedef ShardedInfo<
            get_compile_time_arg_val(sharded_args_start_idx),
            get_compile_time_arg_val(sharded_args_start_idx + 1),
            get_compile_time_arg_val(sharded_args_start_idx + 2),
            get_compile_time_arg_val(sharded_args_start_idx + 3),
            get_compile_time_arg_val(sharded_args_start_idx + 4),
            get_compile_time_arg_val(sharded_args_start_idx + 5),
            get_compile_time_arg_val(sharded_args_start_idx + 6)>
            tensor_shard_info;

        const auto [mapping_table, rt_increment] =
            experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(fabric_arg_idx));
        experimental::ShardedAddrGen<tensor_shard_info> broadcast_tensor_addrgen = {
            .bank_base_address = broadcast_tensor_addr, .shard_array = mapping_table};
        size_t fab_idx = fabric_arg_idx + rt_increment;
        open_connections(fabric_connection, num_connections, fab_idx);
    */
    // #else
    constexpr auto broadcast_tensor_args = TensorAccessorArgs<10>();
    auto broadcast_tensor_addrgen = TensorAccessor(broadcast_tensor_args, broadcast_tensor_addr, page_size);
    open_connections(fabric_connection, num_connections, fabric_arg_idx);
    // #endif

    // Setup broadcast fabric routing
    uint8_t starts[] = {
        static_cast<uint8_t>(start_distance_in_hops_forward), static_cast<uint8_t>(start_distance_in_hops_backward)};
    uint8_t ranges[] = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }

    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        fabric_connection, unicast_route_id, starts, ranges, nullptr, page_size);
    fabric_multicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        fabric_connection,
        scatter_route_id,
        starts,
        ranges,
        NocUnicastScatterCommandHeader{
            {0, 0},  // ignore
            static_cast<uint16_t>(page_size)},
        page_size * 2);

    uint32_t num_total_targets = num_targets_forward_direction + num_targets_backward_direction;

    // Barrier synchronization with column devices
    uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        sem_route_id,
        starts,
        ranges,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,  // ignore
            static_cast<uint32_t>(1)});
    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);

    ///////////////////////////////////////////////////
    // UNIFIED MAIN LOOP: Local NOC Write + Fabric Multicast
    ///////////////////////////////////////////////////
    constexpr uint32_t onetile = 1;
    uint32_t local_page_id = start_page_id;
    uint32_t broadcast_page_id = page_id_start;
    uint32_t end_page_id = start_page_id + num_pages;

    while (local_page_id < end_page_id && broadcast_page_id < page_id_end) {
        // Wait for data from TP reader
        cb_wait_front(cb_id_in, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in);

        // 1. LOCAL NOC WRITE (P2P receiver writer functionality)
        const uint64_t local_dst_noc_addr = get_noc_addr(local_page_id, local_tensor_accessor);
        noc_async_write(l1_read_addr, local_dst_noc_addr, page_size);

        // 2. FABRIC MULTICAST (Broadcast writer functionality)
        // Write to local broadcast tensor and multicast to column devices
        noc_async_write(l1_read_addr, broadcast_tensor_addrgen.get_noc_addr(broadcast_page_id, 0), page_size);
        fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
            fabric_connection,
            unicast_route_id,
            l1_read_addr,
            tt::tt_fabric::NocUnicastCommandHeader{
                linear::addrgen_detail::get_noc_address(broadcast_tensor_addrgen, broadcast_page_id, 0)});

        // Wait for NOC operations to complete
        noc_async_writes_flushed();

        // Release CB buffer after both operations complete
        cb_pop_front(cb_id_in, onetile);

        // Advance to next page
        local_page_id++;
        broadcast_page_id++;
    }

    // Signal column devices that broadcast is complete
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);

    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});

    // Increment local output ready semaphore
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    // Wait for broadcast completion if needed
    if (wait_output_semaphore) {
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
        noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
    }

    // Reset global semaphore if needed
    if (reset_global_semaphore) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
    }

    close_connections(fabric_connection);
    noc_async_write_barrier();
    DPRINT << "end of TP writer" << ENDL();
}
