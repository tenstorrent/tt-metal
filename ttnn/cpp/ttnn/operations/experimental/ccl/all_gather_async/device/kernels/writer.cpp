// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>
#include <array>
#include <type_traits>

#include "common.hpp"

// #include "api/debug/dprint.h"
#include "api/debug/device_print.h"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t packet_size = get_compile_time_arg_val(3);
    constexpr uint8_t range_hops = get_compile_time_arg_val(4);
    constexpr uint8_t range_hops_alt = get_compile_time_arg_val(5);
    constexpr bool load_balance_across_two_routes = true;  // TODO hardcoded, = true for ring with even devices
    constexpr auto output_tensor_args = TensorAccessorArgs<6>();

    constexpr uint32_t pages_per_cb_entry = cb_page_size / output_page_size;
    constexpr uint32_t num_banks = NUM_DRAM_BANKS;  // compile-time constant available in kernels

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chip_id = get_arg_val<uint32_t>(arg_idx++);
    const bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    Noc noc;
    CircularBuffer cb(cb0_id);

    // DPRINT << "output_page_size=" << output_page_size << " pages_per_packet=" << pages_per_packet << " pages_per_cb="
    // << pages_per_cb_entry << ENDL();
    // DEVICE_PRINT(
    //    "output_page_size={} pages_per_packet={} pages_per_cb={}\n",
    //    output_page_size,
    //    pages_per_packet,
    //    pages_per_cb_entry);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    FabricWriter<output_page_size, packet_size, load_balance_across_two_routes> fabric(
        noc, fabric_connection, num_connections, range_hops, range_hops_alt);

    auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
    std::array starts = {static_cast<uint8_t>(1)};
    std::array ranges = {range_hops};
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        sem_route_id,
        starts.data(),
        ranges.data(),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0u,    // ignore
            1u});  // increment 1

    ///////////////////////////////////////////////////
    // STARTUP BARRIER
    ///////////////////////////////////////////////////

    fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        fabric_connection,
        sem_route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc_addr_in_pkt, 0});

    uint32_t num_total_targets = range_hops;  // range_hops_forward + range_hops_backward;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), num_total_targets);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // "iterator" for output_tensor
    // TODO for per-bank iteration, start from (input_page_id_start + bank) and incr by += num_banks
    auto output_page_id = output_page_id_start;
    auto next_output_page_id = [&]() __attribute__((always_inline)) { return output_page_id++; };

    // uint32_t bank = chip_id; // starting
    // for (uint32_t b = 0; b < num_banks; ++b) {
    // uint32_t first_page = output_page_id_start + bank;
    // bank = (bank == num_banks - 1) ? 0 : bank + 1;

    while (output_page_id < output_page_id_end) {
        cb.wait_front(1);
        auto l1_read_addr = cb.get_read_ptr();

        for (uint32_t i = 0; i < pages_per_cb_entry && output_page_id < output_page_id_end; ++i) {
            auto page_id = next_output_page_id();
            // Fabric write
            auto fabric_tensor_page_addr =
                tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_tensor_accessor, page_id, 0);
            fabric.send(l1_read_addr, fabric_tensor_page_addr);

            // Local write.
            // For local writes use posted writes (to skip waiting for ack) on different virtual channel
            // (to avoid interfering with Fabric writes on same noc)
            noc.async_write<Noc::TxnIdMode::DISABLED, Noc::ResponseMode::POSTED>(
                CoreLocalMem<uint32_t>(l1_read_addr),
                output_tensor_accessor,
                output_page_size,
                {},
                {.page_id = page_id},
                NOC_UNICAST_WRITE_VC + 1);

            l1_read_addr += output_page_size;
        }

        noc.async_writes_flushed<Noc::ResponseMode::POSTED>();  // wait for local writes
        fabric.flush();                                         // wait for Fabric writes
        cb.pop_front(1);
    }

    //}

    ///////////////////////////////////////////////////
    // COMPLETION BARRIER
    ///////////////////////////////////////////////////

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
        // noc_semaphore_wait(sem_ptr, out_ready_sem_wait_value); // TODO
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
    }

    close_connections(fabric_connection);
    noc.async_write_barrier();
}
