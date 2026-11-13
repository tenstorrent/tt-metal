// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// routing_exchange_y_sender: the routing exchange's send in the first Y direction
// Runs on: the watcher core (one a chip), RISCV_0 (BRISC), NoC 0 (RISCV_0_default)
// Responsibilities:
//  - waits for this chip's planner instances (ready word), then sends their routing packets from L1 over the fabric
//    straight into each destination chip's DRAM slot, with a fused increment of its arrival counter on planner core 0;
//    this chip's own packet: a NoC write into its own slot
//  - Y hello (launch-skew measurement only): one fabric increment to the Y neighbour's watcher core; waits for the
//    hellos that land here (a runtime arg: 0, 1 or 2) and re-arms the word
// Note: the routing planner reads this chip's own packet from the sender's L1, not from the DRAM slot written on NoC 0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
// (no public device-side fabric API: the packet headers, the fabric connection, the header pool, the routes)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_fabric_sends.hpp"

// This kernel's runtime args (head_gather_program.cpp ChipProgram::y_args of the first Y direction), read once in
// kernel_main
struct RoutingExchangeYSenderArgs {
    RoutingExchangeSendArgs routing_exchange;  // the send block (ChipProgram::send_block)
    uint32_t has_remote_destinations;          // then: whether any destination is another chip
    // then (if so): my "yhello" word, the Y neighbour's fabric chip id and mesh id, the fabric connection (its eth
    // channel, teardown and buffer-index semaphores), the hellos that land here a call (the rows whose first Y
    // direction points at this row: one on a ring; on a line none at the near end, two at the row before the far end)
    uint32_t hello_address;
    uint32_t y_neighbour_chip_id, y_neighbour_mesh_id;
    uint32_t fabric_eth_channel, fabric_teardown_semaphore, fabric_buffer_index_semaphore;
    uint32_t hellos_expected;
};

FORCE_INLINE RoutingExchangeYSenderArgs read_routing_exchange_y_sender_args() {
    RoutingExchangeYSenderArgs args{};
    uint32_t arg_index = 0;
    args.routing_exchange = read_routing_exchange_send_args(arg_index);
    args.has_remote_destinations = moe_block_args::arg(arg_index++);
    if (args.has_remote_destinations) {
        args.hello_address = moe_block_args::arg(arg_index++);
        args.y_neighbour_chip_id = moe_block_args::arg(arg_index++);
        args.y_neighbour_mesh_id = moe_block_args::arg(arg_index++);
        args.fabric_eth_channel = moe_block_args::arg(arg_index++);
        args.fabric_teardown_semaphore = moe_block_args::arg(arg_index++);
        args.fabric_buffer_index_semaphore = moe_block_args::arg(arg_index++);
        args.hellos_expected = moe_block_args::arg(arg_index++);
    }
    return args;
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(uint32_t block, const RoutingExchangeYSenderArgs& args) {
    // (row blocks: every block allocates its packet headers anew - the last block's packets were flushed by its
    // connection's close)
    PacketHeaderPool::reset();
    RoutingExchangeSend routing_exchange{args.routing_exchange};
    if (args.has_remote_destinations) {
        auto connection = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
            args.fabric_eth_channel, args.fabric_teardown_semaphore, args.fabric_buffer_index_semaphore);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
        connection.open();
        // measurement only: "started" to the Y neighbour's watcher; the neighbour's landing here is stamped when seen
        // (between the sends' polls, never waited on before they are done) - the rows' launch skew
        (void)fabric_set_unicast_route(
            header, static_cast<uint16_t>(args.y_neighbour_chip_id), static_cast<uint16_t>(args.y_neighbour_mesh_id));
        header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(my_x[0], my_y[0], args.hello_address), 1, true});
        connection.wait_for_empty_write_slot();
        connection.send_payload_flush_blocking_from_address(
            reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
        volatile tt_l1_ptr uint32_t* hello_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.hello_address);
        bool hello_seen = args.hellos_expected == 0;
        while (!routing_exchange.done) {
            invalidate_l1_cache();
            if (!hello_seen && *hello_word) {
                DeviceTimestampedData("SE-YHELLO", 1);
                hello_seen = true;
            }
            routing_exchange.poll(
                connection,
                header,
                false,
                ROUTING_EXCHANGE_NUM_CHIPS,
                ROUTING_EXCHANGE_PLANNER_INSTANCES,
                ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
                ROUTING_EXCHANGE_MY_CHIP,
                ROUTING_EXCHANGE_HEADER_BYTES,
                ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
                ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
                ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
        }
        if (args.hellos_expected) {
            noc_semaphore_wait_min(hello_word, args.hellos_expected);
            if (!hello_seen) {
                DeviceTimestampedData("SE-YHELLO", 0);  // (late: after the sends)
            }
            noc_semaphore_inc(get_noc_addr(args.hello_address), 0u - args.hellos_expected);  // (re-arm)
            noc_async_atomic_barrier();
        }
        routing_exchange.poll(
            connection,
            header,
            true,
            ROUTING_EXCHANGE_NUM_CHIPS,
            ROUTING_EXCHANGE_PLANNER_INSTANCES,
            ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
            ROUTING_EXCHANGE_MY_CHIP,
            ROUTING_EXCHANGE_HEADER_BYTES,
            ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
            ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
            ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
        connection.close();
    } else {
        struct NoConnection {
            void wait_for_empty_write_slot() {}
            void send_current_slot_non_blocking(uint32_t, uint32_t, uint32_t) {}
            void send_payload_flush_blocking_from_address(uint32_t, uint32_t) {}
        } no_connection;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header = nullptr;
        routing_exchange.poll(
            no_connection,
            header,
            true,
            ROUTING_EXCHANGE_NUM_CHIPS,
            ROUTING_EXCHANGE_PLANNER_INSTANCES,
            ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
            ROUTING_EXCHANGE_MY_CHIP,
            ROUTING_EXCHANGE_HEADER_BYTES,
            ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
            ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
            ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
    }
    noc_async_atomic_barrier();
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const RoutingExchangeYSenderArgs args = read_routing_exchange_y_sender_args();
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(block, args);
        row_block::end(block);
    }
}
