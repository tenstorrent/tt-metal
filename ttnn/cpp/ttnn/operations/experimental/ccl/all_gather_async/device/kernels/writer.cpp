// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

#include <cstdint>
#include <array>
#include <type_traits>

// #include "api/debug/dprint.h"
#include "api/debug/device_print.h"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_page_size = get_compile_time_arg_val(1);
constexpr uint32_t out_page_size = get_compile_time_arg_val(2);
constexpr uint32_t packet_size = get_compile_time_arg_val(3);
// constexpr uint32_t num_targets_forward_direction = 4; //get_compile_time_arg_val(4);
// constexpr uint32_t num_targets_backward_direction = 3; //get_compile_time_arg_val(5);
// constexpr uint32_t start_distance_in_hops_forward = 1; //get_compile_time_arg_val(6);
constexpr uint32_t range_hops_forward = 4;  // get_compile_time_arg_val(7);
// constexpr uint32_t start_distance_in_hops_backward = 1; //get_compile_time_arg_val(8);
constexpr uint32_t range_hops_backward = 3;            // get_compile_time_arg_val(9);
constexpr bool load_balance_across_two_routes = true;  // TODO hardcoded, = true for ring with even devices

inline constexpr uint32_t sharded_args_start_idx = 10;

constexpr uint32_t num_out_pages_per_packet = packet_size / out_page_size;
constexpr uint32_t outputs_per_cb_page = cb_page_size / out_page_size;

constexpr bool unicast = packet_size <= out_page_size;
constexpr bool scatter = !unicast;

static_assert(
    (unicast && out_page_size % packet_size == 0)  // will be able to cover output page with unicast writes
    || (scatter && outputs_per_cb_page % num_out_pages_per_packet == 0 &&
        // will be able to cover cb page with scattered writes with one type of header
        num_out_pages_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS));

class FabricUnicastWriter {
public:
    FabricUnicastWriter(
        tt::tt_fabric::RoutingPlaneConnectionManager& manager,
        std::array<uint8_t, 2> starts,
        std::array<uint8_t, 2> ranges,
        uint32_t num_connections) :
        fabric_connection{manager}, unicast_route_id{PacketHeaderPool::allocate_header_n(num_connections)} {
        fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            fabric_connection, unicast_route_id, starts.data(), ranges.data(), nullptr, packet_size);
    }

    void send(uint32_t cb_out_page_start, uint64_t tensor_page_addr) {
        auto packet_read_addr = cb_out_page_start;
        auto dest_addr = tensor_page_addr;
        constexpr uint32_t packets_per_outpage = out_page_size / packet_size;
        for (uint32_t packet = 0; packet < packets_per_outpage; packet++) {
            noc_async_writes_flushed();
            fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                fabric_connection,
                unicast_route_id,
                packet_read_addr,
                tt::tt_fabric::NocUnicastCommandHeader{dest_addr},
                static_cast<uint16_t>(0u) /*packet_size*/);
            packet_read_addr += packet_size;
            dest_addr += packet_size;
        }
    }

private:
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t unicast_route_id;
};

class FabricScatterWriter {
public:
    FabricScatterWriter(
        tt::tt_fabric::RoutingPlaneConnectionManager& manager,
        std::array<uint8_t, 2> starts,
        std::array<uint8_t, 2> ranges,
        uint32_t num_connections) :
        fabric_connection{manager},
        route_id_1{PacketHeaderPool::allocate_header_n(num_connections)},
        route_id_2{load_balance_across_two_routes ? PacketHeaderPool::allocate_header_n(num_connections) : route_id_1},
        use_route_1{true},
        scatter_header({}, {}) {
        scatter_header.chunk_count = 0;
        const auto default_scatter_header = [] {
            if constexpr (num_out_pages_per_packet == 2) {
                return NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(out_page_size)});
            } else if constexpr (num_out_pages_per_packet == 3) {
                return NocUnicastScatterCommandHeader(
                    {0, 0, 0}, {static_cast<uint16_t>(out_page_size), static_cast<uint16_t>(out_page_size)});
            } else {
                return NocUnicastScatterCommandHeader(
                    {0, 0, 0, 0},
                    {static_cast<uint16_t>(out_page_size),
                     static_cast<uint16_t>(out_page_size),
                     static_cast<uint16_t>(out_page_size)});
            }
        }();

        fabric_multicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            fabric_connection, route_id_1, starts.data(), ranges.data(), default_scatter_header, packet_size);

        // Ring topology: alternate between two routes for load balancing.
        // Example for 8 device ring:
        //    route_1 = 4 devices forward and 3 devices backward
        //    route_2 = 3 devices forward and 4 devices backward
        if constexpr (load_balance_across_two_routes) {
            std::array<uint8_t, 2> swapped_ranges = {ranges[1], ranges[0]};
            fabric_multicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                fabric_connection,
                route_id_2,
                starts.data(),
                swapped_ranges.data(),
                default_scatter_header,
                packet_size);
        }
    }

    void send(uint32_t cb_out_page_start, uint64_t tensor_page_addr) {
        if (scatter_header.chunk_count == 0) {
            // save the address of the first chunk in packet
            // write function will start reading here
            packet_data_read_ptr = cb_out_page_start;
        }
        scatter_header.noc_address[scatter_header.chunk_count++] = tensor_page_addr;

        if (scatter_header.chunk_count == num_out_pages_per_packet) {
            noc_async_writes_flushed();
            fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                fabric_connection,
                use_route_1 ? route_id_1 : route_id_2,
                packet_data_read_ptr,
                scatter_header,
                out_page_size * num_out_pages_per_packet);

            scatter_header.chunk_count = 0;
            if constexpr (load_balance_across_two_routes) {
                use_route_1 = !use_route_1;  // alternate between routes for load balancing
            }
        }
    }

private:
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t route_id_1;
    uint8_t route_id_2;
    bool use_route_1;
    uint32_t packet_data_read_ptr;
    NocUnicastScatterCommandHeader scatter_header;
};

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    const address_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    constexpr auto tensor0_args = TensorAccessorArgs<sharded_args_start_idx>();
    auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0);

    // DPRINT << "out_page_size=" << out_page_size << " pages_per_packet=" << num_out_pages_per_packet << "
    // pages_per_cb=" << outputs_per_cb_page << ENDL();
    DEVICE_PRINT(
        "out_page_size={} pages_per_packet={} pages_per_cb={} scatter={}\n",
        out_page_size,
        num_out_pages_per_packet,
        outputs_per_cb_page,
        scatter);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    std::array starts = {static_cast<uint8_t>(1), static_cast<uint8_t>(1)};
    std::array ranges = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        ranges[0] = ranges[1];
    }

    uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        sem_route_id,
        starts.data(),
        ranges.data(),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0u,    // ignore
            1u});  // increment 1

    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

    uint32_t num_total_targets = range_hops_forward + range_hops_backward;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);

    // 1. mcast via fabric to remote tensor addresses

    using FabricWriter = std::conditional_t<unicast, FabricUnicastWriter, FabricScatterWriter>;
    FabricWriter writer(fabric_connection, starts, ranges, num_connections);

    for (uint32_t page_id = output_page_id_start; page_id < output_page_id_end;) {
        cb_wait_front(cb0_id, 1);
        auto l1_read_addr = get_read_ptr(cb0_id);

        const auto page_id_end = page_id + outputs_per_cb_page;
        for (; page_id < page_id_end && page_id < output_page_id_end; ++page_id) {
            auto fabric_tensor_page_addr =
                tt::tt_fabric::linear::addrgen_detail::get_noc_address(tensor0_addrgen, page_id, 0);
            writer.send(l1_read_addr, fabric_tensor_page_addr);
            auto local_tensor_page_addr = tensor0_addrgen.get_noc_addr(page_id, 0);
            noc_async_write(l1_read_addr, local_tensor_page_addr, out_page_size);
            l1_read_addr += out_page_size;
        }

        noc_async_writes_flushed();
        cb_pop_front(cb0_id, 1);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);

    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
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
