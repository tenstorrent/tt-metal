// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
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
using tt::tt_metal::BufferType;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(0));
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t row_size = get_compile_time_arg_val(3);
constexpr uint32_t max_packet_size = get_compile_time_arg_val(4);
constexpr uint32_t num_packets_per_row = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);
constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<8>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<8 + ccl_routing_utils::num_line_multicast_args>();

inline constexpr uint32_t sharded_args_start_idx = 8 + 2 * ccl_routing_utils::num_line_multicast_args;

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    const address_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t row_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t row_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto data_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
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
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
        .bank_base_address = tensor_address0, .shard_array = mapping_table};
    size_t fab_idx = arg_for_fab + rt_increment;
    open_connections(fabric_connection, data_route_id, fab_idx);
#else
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    const auto tensor0_addrgen = get_interleaved_addr_gen<is_dram, row_size>(tensor_address0);
    open_connections(fabric_connection, data_route_id, arg_for_fab);
#endif
    uint8_t starts[] = {
        static_cast<uint8_t>(forward_multicast_route_info.start_distance_in_hops),
        static_cast<uint8_t>(backward_multicast_route_info.start_distance_in_hops)};
    uint8_t ranges[] = {
        static_cast<uint8_t>(forward_multicast_route_info.range_hops),
        static_cast<uint8_t>(backward_multicast_route_info.range_hops)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
        ranges[0] = ranges[1];
    }
    fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
        fabric_connection, data_route_id, starts, ranges);

    uint32_t num_total_targets = num_targets_forward_direction + num_targets_backward_direction;

    uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Wrap | UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        sem_route_id,
        starts,
        ranges,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0,                         // ignore
            static_cast<uint16_t>(1),  // increment 1
            32});

    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0, 0});

    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);

    // 1. mcast via fabric to remote tensor addresses
    uint32_t row_id = row_id_start;
    while (row_id < row_id_end) {
        size_t l1_read_addr = get_read_ptr(cb0_id);
        cb_wait_front(cb0_id, 1);

        uint32_t offset = 0;

        for (uint32_t j = 0; j < num_packets_per_row; j++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(row_id, tensor0_addrgen, offset, 0);

            uint32_t packet_size = std::min(max_packet_size, page_size);
            packet_size = std::min(packet_size, page_size - max_packet_size * j);

            const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
            noc_async_write(l1_read_addr, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), packet_size);
            fabric_multicast_noc_unicast_write_with_state<
                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                fabric_connection,
                data_route_id,
                l1_read_addr,
                tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr},
                packet_size);
            noc_async_writes_flushed();
            l1_read_addr += packet_size;
            offset += packet_size;  // advance the noc address for the next packet
        }
        row_id++;
        cb_pop_front(cb0_id, 1);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);

    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0, 0});
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
        noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value);
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
    }

    close_connections(fabric_connection);

    noc_async_write_barrier();
}
