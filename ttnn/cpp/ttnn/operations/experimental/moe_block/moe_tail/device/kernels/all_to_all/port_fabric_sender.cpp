// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// port_fabric_sender: an all-to-all port's fabric sender - the all-to-all runs into the destinations' outputs
// Runs on: all-to-all port cores (plan.port_union; idle on a chip where the core is no port), RISCV_0 (BRISC), NoC 1
// Responsibilities:
//  - pops whole groups of staging pages from the staging CB (pushed by port_staging_reader)
//  - each row block: sends readies over the fabric to the chips writing my output through me, awaits every
//    destination's ready (block b's output rows are the output's l x the block stride further on every chip)
//  - sends each run over the fabric (2D: routed by node, 1D: by hops) into the destination's output (DRAM), with a
//    fused arrival increment every arrival_increment_every-th run; then awaits every arrival expected here and re-arms
//    the counter; the fabric connection and the packet headers are set up once, for all blocks
// Note: the fabric routers write on NoC 1, so the destination DRAM address is resolved for NoC 1 (SYS-1419)

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/all_to_all/port_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/all_to_all/padded_row_tiles.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
using namespace tt::tt_fabric;

// 2D fabrics route by destination node (multi-hop); 1D by hop count
inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* header, uint16_t chip_id, uint16_t mesh_id, uint32_t hops) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)hops;
    (void)fabric_set_unicast_route(header, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(header), static_cast<uint16_t>(hops));
#endif
}

// the most destinations a port sends to (the host's kMaxPortDestinations: a port's destinations are TP group chips)
constexpr uint32_t max_destinations = 8;

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    constexpr auto output_args = TensorAccessorArgs<PORT_ACCESSOR_ARGS>();
    // row blocks: their count and, a block, the output's rows that much further in every bank (bytes)
    constexpr uint32_t row_blocks = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
    constexpr uint32_t output_block_stride_bytes =
        get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 1);
    size_t arg_index = 0;
    const uint32_t output_address = moe_block_args::arg(arg_index++), my_position = moe_block_args::arg(arg_index++);
    const uint32_t owned_row_tiles = moe_block_args::arg(arg_index++);
    const uint32_t block_tiles = moe_block_args::arg(arg_index++), link = moe_block_args::arg(arg_index++);
    const uint32_t num_links = moe_block_args::arg(arg_index++);
    const uint32_t arrival_address = moe_block_args::arg(arg_index++),
                   expected_arrivals = moe_block_args::arg(arg_index++);
    const uint32_t ready_address = moe_block_args::arg(arg_index++),
                   expected_readies = moe_block_args::arg(arg_index++);
    const uint32_t padding_args = arg_index;  // (the padding args, at a fixed place)
    if constexpr (skip_padding) {
        arg_index += 7;
    }
    const uint32_t num_readies = moe_block_args::arg(arg_index++);  // (readies I send: x | y << 16, chip, mesh, hops)
    const uint32_t readies = arg_index;
    arg_index += 4 * num_readies;
    // (destinations: position | part << 8, chip, mesh, hops, runs, its port's x | y << 16)
    const uint32_t num_destinations = moe_block_args::arg(arg_index++);
    const uint32_t destinations = arg_index;
    arg_index += 6 * num_destinations;
    const bool sends = num_destinations > 0 || num_readies > 0;
    WorkerToFabricEdmSender connection;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[runs_per_group + 1];  // (the last: a padding row tile's increments)
    if (sends) {
        // (the fabric connection's eth channel, teardown and buffer-index semaphores: the block's last three)
        connection = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
            moe_block_args::arg(arg_index), moe_block_args::arg(arg_index + 1), moe_block_args::arg(arg_index + 2));
        for (uint32_t header_index = 0; header_index < runs_per_group + 1; ++header_index) {
            headers[header_index] = PacketHeaderPool::allocate_header();
        }
        connection.open();
    }
    for (uint32_t block = 0; block < row_blocks; ++block) {
        const auto output = TensorAccessor(output_args, output_address + block * output_block_stride_bytes, page_bytes);
        uint64_t live_mask = 0;
        if constexpr (skip_padding) {
            live_mask = read_live_row_tile_mask(
                padding_args,
                get_write_ptr(CB_PADDING_SCRATCH) + PADDING_SCRATCH_WRITER_OFFSET,
                owned_row_tiles,
                block);
        }
        if (sends) {
            for (uint32_t i = 0; i < num_readies; ++i) {  // (my output may be written: say so to each of its senders)
                const uint32_t ready_port_xy = moe_block_args::arg(readies + 4 * i);
                route_to(
                    headers[0],
                    moe_block_args::arg(readies + 4 * i + 1),
                    moe_block_args::arg(readies + 4 * i + 2),
                    moe_block_args::arg(readies + 4 * i + 3));
                headers[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                    get_noc_addr(ready_port_xy & 0xFFFF, ready_port_xy >> 16, ready_address), 1, true});
                connection.wait_for_empty_write_slot();
                connection.send_payload_flush_blocking_from_address(
                    reinterpret_cast<uint32_t>(headers[0]), sizeof(PACKET_HEADER_TYPE));
            }
            if (num_destinations > 0) {
                volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_address);
                noc_semaphore_wait_min(ready, expected_readies);
                noc_semaphore_inc(get_noc_addr(ready_address), 0u - expected_readies);
                noc_async_atomic_barrier();
            }
            uint32_t runs_sent[max_destinations] = {};
            uint32_t next_header = 0, unflushed = 0;
            for (uint32_t row_tile = 0; row_tile < owned_row_tiles; ++row_tile) {
                for (uint32_t destination_index = 0; destination_index < num_destinations; ++destination_index) {
                    const uint32_t destination_args = destinations + 6 * destination_index;
                    const uint32_t part = (moe_block_args::arg(destination_args) >> 8) & 3;
                    const uint16_t chip = moe_block_args::arg(destination_args + 1),
                                   mesh = moe_block_args::arg(destination_args + 2);
                    const uint32_t hops = moe_block_args::arg(destination_args + 3),
                                   runs = moe_block_args::arg(destination_args + 4);
                    const uint32_t port_xy = moe_block_args::arg(destination_args + 5);
                    const uint64_t arrival = get_noc_addr(port_xy & 0xFFFF, port_xy >> 16, arrival_address);
                    // (my rows' row_tile in the destination's output)
                    const uint32_t row_tile_first_page = (my_position * owned_row_tiles + row_tile) * block_tiles;
                    // (a padding row tile: nothing sent, its runs counted - an increment due goes alone, so the
                    // destination's arrival count stays the plan's)
                    if (!row_tile_is_live<skip_padding>(live_mask, row_tile)) {
                        for_each_run(block_tiles, link, num_links, part, [&](uint32_t, uint32_t) {
                            if (++runs_sent[destination_index] % arrival_increment_every == 0 ||
                                runs_sent[destination_index] == runs) {
                                route_to(headers[runs_per_group], chip, mesh, hops);
                                headers[runs_per_group]->to_noc_unicast_atomic_inc(
                                    NocUnicastAtomicIncCommandHeader{arrival, 1, true});
                                connection.wait_for_empty_write_slot();
                                connection.send_payload_flush_blocking_from_address(
                                    reinterpret_cast<uint32_t>(headers[runs_per_group]), sizeof(PACKET_HEADER_TYPE));
                            }
                        });
                        continue;
                    }
                    for_each_run(block_tiles, link, num_links, part, [&](uint32_t run_first_page, uint32_t pages) {
                        cb_wait_front(staging_cb, pages_per_run * (unflushed + 1));
                        const uint32_t source = get_read_ptr(staging_cb) + unflushed * pages_per_run * page_bytes;
                        // fabric router writes on NOC1: resolve the DRAM endpoint for its NoC (SYS-1419)
                        // (pages run_first_page, + banks, ..: consecutive)
                        const uint64_t destination = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            output, row_tile_first_page + run_first_page);
                        volatile tt_l1_ptr PACKET_HEADER_TYPE* header = headers[next_header];
                        next_header = next_header + 1 == runs_per_group ? 0 : next_header + 1;
                        route_to(header, chip, mesh, hops);
                        if (++runs_sent[destination_index] % arrival_increment_every == 0 ||
                            runs_sent[destination_index] == runs) {
                            header->to_noc_fused_unicast_write_atomic_inc(
                                NocUnicastAtomicIncFusedCommandHeader{destination, arrival, 1, true},
                                pages * page_bytes);
                        } else {
                            header->to_noc_unicast_write(NocUnicastCommandHeader{destination}, pages * page_bytes);
                        }
                        connection.wait_for_empty_write_slot();
                        connection.send_current_slot_non_blocking(
                            source, pages * page_bytes, reinterpret_cast<uint32_t>(header));
                        if (++unflushed == runs_per_group) {
                            noc_async_writes_flushed();
                            cb_pop_front(staging_cb, pages_per_run * runs_per_group);
                            unflushed = 0;
                        }
                    });
                }
            }
            if (unflushed) {  // (a whole group: the reader pushed a whole group)
                noc_async_writes_flushed();
                cb_wait_front(staging_cb, pages_per_run * runs_per_group);
                cb_pop_front(staging_cb, pages_per_run * runs_per_group);
            }
        }
        if (expected_arrivals > 0) {  // (everything sent into my output through my side landed; then re-armed)
            volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_address);
            noc_semaphore_wait_min(arrived, expected_arrivals);
            noc_semaphore_inc(get_noc_addr(arrival_address), 0u - expected_arrivals);
            noc_async_atomic_barrier();
        }
    }
    if (sends) {
        connection.close();
    }
}
