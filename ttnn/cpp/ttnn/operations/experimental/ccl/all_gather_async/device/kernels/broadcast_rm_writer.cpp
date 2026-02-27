// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_page_size = get_compile_time_arg_val(1);
constexpr uint32_t out_page_size = get_compile_time_arg_val(2);
constexpr uint32_t packet_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr uint32_t start_distance_in_hops_forward = get_compile_time_arg_val(6);
constexpr uint32_t range_hops_forward = get_compile_time_arg_val(7);
constexpr uint32_t start_distance_in_hops_backward = get_compile_time_arg_val(8);
constexpr uint32_t range_hops_backward = get_compile_time_arg_val(9);

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
        constexpr uint32_t packets_per_outpage = out_page_size / packet_size;
        for (uint32_t packet = 0; packet < packets_per_outpage; packet++) {
            auto packet_read_addr = cb_out_page_start + packet * packet_size;
            auto write_offset = packet * packet_size;
            noc_async_writes_flushed();
            fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                fabric_connection,
                unicast_route_id,
                packet_read_addr,
                tt::tt_fabric::NocUnicastCommandHeader{tensor_page_addr + write_offset},
                static_cast<uint16_t>(0u) /*packet_size*/);
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
        scatter_route_id{PacketHeaderPool::allocate_header_n(num_connections)},
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
            fabric_connection, scatter_route_id, starts.data(), ranges.data(), default_scatter_header, packet_size);
    }

    void send(uint32_t cb_out_page_start, uint64_t tensor_page_addr) {
        if (scatter_header.chunk_count == 0) {
            // save the address of the first chunk in packet
            // write function will start reading here
            packet_data_read_ptr = cb_out_page_start;
        }
        scatter_header.noc_address[scatter_header.chunk_count++] = tensor_page_addr;

        if (scatter_header.chunk_count == num_out_pages_per_packet) {
            fabric_multicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                fabric_connection,
                scatter_route_id,
                packet_data_read_ptr,
                scatter_header,
                out_page_size * num_out_pages_per_packet);

            scatter_header.chunk_count = 0;
        }
    }

private:
    tt::tt_fabric::RoutingPlaneConnectionManager& fabric_connection;
    uint8_t scatter_route_id;
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
    auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0, out_page_size);

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    std::array starts = {
        static_cast<uint8_t>(start_distance_in_hops_forward), static_cast<uint8_t>(start_distance_in_hops_backward)};
    std::array ranges = {static_cast<uint8_t>(range_hops_forward), static_cast<uint8_t>(range_hops_backward)};
    if (ranges[0] == 0) {
        starts[0] = starts[1];
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

    uint32_t num_total_targets = num_targets_forward_direction + num_targets_backward_direction;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);

    // 1. mcast via fabric to remote tensor addresses

    using FabricWriter = std::conditional_t<unicast, FabricUnicastWriter, FabricScatterWriter>;
    FabricWriter writer(fabric_connection, starts, ranges, num_connections);

    for (uint32_t page_id = output_page_id_start; page_id < output_page_id_end;) {
        cb_wait_front(cb0_id, 1);
        auto l1_read_addr = get_read_ptr(cb0_id);

        for (uint32_t output = 0u; output < outputs_per_cb_page; output++) {
            if (page_id + output >= output_page_id_end) [[unlikely]] {
                break;
            };
            auto out_page_start = l1_read_addr + output * out_page_size;
            auto tensor_page_addr = tensor0_addrgen.get_noc_addr(page_id + output, 0);
            writer.send(out_page_start, tensor_page_addr);
            noc_async_write(out_page_start, tensor_page_addr, out_page_size);
        }
        page_id += outputs_per_cb_page;

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
