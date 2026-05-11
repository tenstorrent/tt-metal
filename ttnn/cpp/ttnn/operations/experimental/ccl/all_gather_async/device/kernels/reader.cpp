// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "experimental/core_local_mem.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

#include <cstdint>

#include "common.hpp"

#include "api/debug/dprint.h"
// #include "api/debug/device_print.h"

using address_t = uint32_t;
using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t packet_size = get_compile_time_arg_val(4);
    constexpr uint8_t range_hops = get_compile_time_arg_val(5);
    constexpr uint8_t range_hops_alt = get_compile_time_arg_val(6);
    constexpr bool load_balance_across_two_routes = true;  // TODO hardcoded, = true for ring with even devices
    constexpr auto input_tensor_args = TensorAccessorArgs<7>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t inputs_per_cb_page = cb_page_size / input_page_size;  // TODO duplicate of pages_per_cb_entry?

    constexpr uint32_t pages_per_packet = packet_size / output_page_size;
    constexpr uint32_t pages_per_cb_entry = cb_page_size / output_page_size;
    constexpr uint32_t num_banks = NUM_DRAM_BANKS;  // compile-time constant available in kernels
    static_assert(cb_page_size % input_page_size == 0);

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_tensor_address);
    auto output_tensor_accessor = TensorAccessor(output_tensor_args, output_tensor_address);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb0_id);

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_for_fab);

    FabricScatterWriter<output_page_size, pages_per_packet, load_balance_across_two_routes> writer(
        noc, fabric_connection, range_hops, range_hops_alt, num_connections);

    /*auto sem_route_id = PacketHeaderPool::allocate_header_n(num_connections);
    uint64_t barrier_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
    fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        fabric_connection,
        sem_route_id,
        starts.data(),
        ranges.data(),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            0u,    // ignore
            1u});  // increment 1*/

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    // TODO explain this (is loop general or only for 3 trids?)
    uint32_t max_trid = 3;  // TODO = cb depth, <= 15
    uint32_t curr_trid = 1;
    uint32_t wait_trid = 1;
    bool txns_in_flight = false;

    // Get write pointer (to write to CB) and read pointer (to read from CB).
    // We need to manually keep track of these pointers since we don't push_back
    // after every reserve_back when using NOC transaction IDs, so get_read/write_ptr()
    // will return stale values.
    auto l1_base_addr = cb.get_write_ptr();
    auto l1_end_addr = l1_base_addr + (3 * cb_page_size);  // TODO hardcoded number 3
    auto l1_write_addr = l1_base_addr;
    auto l1_read_addr = l1_base_addr;

    // "iterator" for input_tensor
    // TODO for per-bank iteration, start from (input_page_id_start + bank) and incr by += num_banks
    auto input_page_id = input_page_id_start;
    auto next_input_page_id = [&]() __attribute__((always_inline)) { return input_page_id++; };

    // "iterator" for output_tensor
    // TODO for per-bank iteration, start from (input_page_id_start + bank) and incr by += num_banks
    auto output_page_id = output_page_id_start;
    auto next_output_page_id = [&]() __attribute__((always_inline)) { return output_page_id++; };

    // We reserve one to kick start the pipeline, and then it is steady state
    cb.reserve_back(2);

    // uint32_t bank = chip_id; // starting
    // for (uint32_t b = 0; b < num_banks; ++b) {
    // uint32_t first_page = input_page_id_start + bank;
    // bank = (bank == num_banks - 1) ? 0 : bank + 1;

    while (input_page_id < input_page_id_end) {
        // fill CB page
        for (uint32_t i = 0; i < inputs_per_cb_page && input_page_id < input_page_id_end; ++i) {
            auto page_id = next_input_page_id();
            noc.async_read<experimental::Noc::TxnIdMode::ENABLED>(
                input_tensor_accessor,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr),
                input_page_size,
                {.page_id = page_id},
                {},
                NOC_UNICAST_WRITE_VC,
                curr_trid);
            l1_write_addr += input_page_size;

            // uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<true>(page_id);  // true for
            // DRAM uint32_t bank_id = interleaved_addr_gen::get_bank_index<true>(page_id, bank_offset_index); DPRINT <<
            // "page_id=" << page_id << " bank=" << bank_id << ENDL();
        }
        if (l1_write_addr == l1_end_addr) {
            l1_write_addr = l1_base_addr;
        }

        curr_trid = (curr_trid == max_trid) ? 1 : curr_trid + 1;
        if (txns_in_flight) {
            // push_back() will unblock the writer to send Fabric data in opposite dir
            noc.async_read_barrier<experimental::Noc::BarrierMode::TXN_ID>(wait_trid);
            cb.push_back(1);
            wait_trid = (wait_trid == max_trid) ? 1 : (wait_trid + 1);

            // Send Fabric data in our dir
            for (uint32_t i = 0; i < pages_per_cb_entry && output_page_id < output_page_id_end; ++i) {
                auto page_id = next_output_page_id();
                auto fabric_tensor_page_addr =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_tensor_accessor, page_id, 0);
                writer.send(l1_read_addr, fabric_tensor_page_addr);
                l1_read_addr += input_page_size;
            }
            if (l1_read_addr == l1_end_addr) {
                l1_read_addr = l1_base_addr;
            }

            // Reserve for next block
            // Reserve back is not incremental, so to reserve one more, we need to reserve 2
            // This accounts for the one we already have reserved (for in-flight read)
            cb.reserve_back(2);
        }
        txns_in_flight = true;
    }
    // Drain in-flight reads
    while (wait_trid != curr_trid) {
        // push_back() will unblock the writer to send Fabric data in opposite dir
        noc.async_read_barrier<experimental::Noc::BarrierMode::TXN_ID>(wait_trid);
        cb.push_back(1);
        wait_trid = (wait_trid == max_trid) ? 1 : (wait_trid + 1);

        // Send Fabric data in our dir
        for (uint32_t i = 0; i < pages_per_cb_entry && output_page_id < output_page_id_end; ++i) {
            auto page_id = next_output_page_id();
            auto fabric_tensor_page_addr =
                tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_tensor_accessor, page_id, 0);
            writer.send(l1_read_addr, fabric_tensor_page_addr);
            l1_read_addr += input_page_size;
        }
        if (l1_read_addr == l1_end_addr) {
            l1_read_addr = l1_base_addr;
        }
    }

    //}

    /*// NUM_DRAM_BANKS is a device compile-time constant available in kernels
    uint32_t num_banks = NUM_DRAM_BANKS;

    for (uint32_t bank = 0; bank < num_banks; bank++) {
        for (uint32_t page_id = bank; page_id < total_pages; page_id += num_banks) {
            // page_id hits bank `bank` every iteration
            // use with noc.async_read(tensor_accessor, dst, size, {.page_id = page_id}, ...);
        }
    }

    for (uint32_t bank = 0; bank < num_banks; bank++) {
    // First page_id in [start, end) that lands on this bank
    uint32_t remainder = start_page_id % num_banks;
    uint32_t first = (remainder <= bank)
        ? start_page_id + (bank - remainder)
        : start_page_id + (num_banks - remainder + bank);

        for (uint32_t page_id = first; page_id < end_page_id; page_id += num_banks) {
            // all iterations hit bank `bank`
        }
    }

    for (uint32_t i = 0; i < num_banks; i++) {
        uint32_t first = start_page_id + i;
        for (uint32_t page_id = first; page_id < end_page_id; page_id += num_banks) {
            // all iterations here hit the same bank
        }
    }*/

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    // 4. global semaphore reset
    // if (reset_global_semaphore) {
    //    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), 0);
    //}

    close_connections(fabric_connection);
    noc.async_write_barrier();
}
