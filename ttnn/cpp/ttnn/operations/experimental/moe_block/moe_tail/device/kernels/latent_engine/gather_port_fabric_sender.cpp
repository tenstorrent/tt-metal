// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_port_fabric_sender: a latent engine gather port's fabric sender - the latent all-gather into the next chip
// Runs on: latent engine gather port cores (plan.port_cores, the last grid column), RISCV_0 (BRISC), NoC 1
// Responsibilities:
//  - pops groups from staging_cb (their packet count from packet_count_cb), filled by gather_port_staging_reader
//  - sends every packet of its units over the fabric into the next chip's engine cores' landing slots (L1); a
//    watcher's last share of a group carries a fused increment on that core's landed word
//  - fence, each row block: after the zero gate (PORT_WRITER_GO_SEM) sends its ready to the port writing into this
//    chip, waits for the next chip's ready before writing there; the fabric connection and the packet headers are set
//    up once for all blocks

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/latent_engine/gather_port_common.hpp"
#include "api/debug/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    const uint32_t landing = get_write_ptr(landing_cb);
    size_t arg_index = 0;
    // (the next chip's landed words; my ready word, which the previous chip's port increments)
    const uint32_t step_words = get_write_ptr(CB_STEP_WORDS), ready_address = moe_block_args::arg(arg_index++);
    // (do I send a ready, and to which port core of the previous chip)
    const bool send_ready = moe_block_args::arg(arg_index++) != 0;
    const uint32_t ready_x = moe_block_args::arg(arg_index++), ready_y = moe_block_args::arg(arg_index++);
    const uint16_t mesh_id = moe_block_args::arg(arg_index++), chip_id = moe_block_args::arg(arg_index++);
    const uint32_t num_units = moe_block_args::arg(arg_index++), units = arg_index, share_k_ranges = units + num_units;
    const uint32_t column_xs = share_k_ranges + 2 * engine_columns, pair_ys = column_xs + engine_columns;
    arg_index = pair_ys + num_pairs;
    if (num_units == 0 &&
        !send_ready) {  // (no link: still in every gate - its go is incremented a block, re-armed here)
        for (uint32_t block = 0; block < row_blocks; ++block) {
            wait_zero_gate(PORT_WRITER_GO_SEM);
        }
        return;
    }
    // the fabric connection and the packet headers: once a call, for every block
    // (the fabric connection's eth channel, teardown and buffer-index semaphores: the block's last three)
    auto connection = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
        moe_block_args::arg(arg_index), moe_block_args::arg(arg_index + 1), moe_block_args::arg(arg_index + 2));
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[packets_per_group];
    for (uint32_t header_index = 0; header_index < packets_per_group; ++header_index) {
        headers[header_index] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(headers[header_index], chip_id, mesh_id);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* ready_header = headers[0];
    connection.open();
    for (uint32_t block = 0; block < row_blocks; ++block) {
        // (block 0: this chip's engine words zeroed; block l: everyone here done with block b - 1 - my ready goes out
        // only then: the previous chip's block b increments and shares land in them)
        wait_zero_gate(PORT_WRITER_GO_SEM);
        if (send_ready) {  // (my landing regions are free for this block)
            ready_header->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_address), 1, true});
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(ready_header), sizeof(PACKET_HEADER_TYPE));
        }
        if (num_units > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_address);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_address), 0u - 1u);  // (re-arm: one ready a block, one waiter)
            noc_async_atomic_barrier();
        }
        // the reader's groups: packets_per_group slots each, its packet count in packet_count_cb; a header a slot, one
        // flush a group
        uint32_t group_index = 0, group_packets = 0;
        auto next_group = [&]() {
            if (group_packets > 0) {
                noc_async_writes_flushed();  // (the group's packets and headers have left L1)
                cb_pop_front(staging_cb, packets_per_group * tiles_per_packet);
            }
            {
                DeviceZoneScopedN("TP-SGWAIT");  // (the reader's next group not read yet)
                cb_wait_front(packet_count_cb, 1);
            }
            group_packets = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(packet_count_cb));
            cb_pop_front(packet_count_cb, 1);
            cb_wait_front(staging_cb, packets_per_group * tiles_per_packet);
            group_index = 0;
        };
        for (uint32_t unit_index = 0; unit_index < num_units; ++unit_index) {
            DeviceZoneScopedN("TP-SUNIT");
            // (position: on the next chip)
            const uint32_t unit = moe_block_args::arg(units + unit_index), part = unit & 0xFF, position = unit >> 8;
            auto send = [&](uint32_t pair,
                            uint32_t step,
                            uint32_t column,
                            uint32_t k_first,
                            uint32_t packet_k_first,
                            uint32_t tiles,
                            bool last_packet) {
                if (group_index == group_packets) {
                    next_group();
                }
                const uint32_t owner_x = moe_block_args::arg(column_xs + column);
                const uint32_t owner_y = moe_block_args::arg(pair_ys + pair);
                const uint64_t destination = get_noc_addr(
                    owner_x, owner_y, landing + (step * max_share_tiles + packet_k_first - k_first) * tile_bytes, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* header = headers[group_index];
                const uint32_t first_column = part == 2 ? engine_columns / 2 : 0;
                const uint32_t end_column = part == 1 ? engine_columns / 2 : engine_columns;
                // (a watcher's share: its column group's last packet)
                if (last_packet && ((column - first_column) % landed_group_columns == landed_group_columns - 1 ||
                                    column == end_column - 1)) {
                    header->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{
                            destination,
                            get_noc_addr(owner_x, owner_y, step_words + STEP_WORD_BYTES * step, 0),
                            1,
                            true},
                        tiles * tile_bytes);
                } else {
                    header->to_noc_unicast_write(NocUnicastCommandHeader{destination}, tiles * tile_bytes);
                }
                connection.wait_for_empty_write_slot();
                connection.send_current_slot_non_blocking(
                    get_read_ptr(staging_cb) + group_index * tiles_per_packet * tile_bytes,
                    tiles * tile_bytes,
                    reinterpret_cast<uint32_t>(header));
                ++group_index;
            };
            walk(part, position, share_k_ranges, send);
        }
        if (group_packets > 0) {
            noc_async_writes_flushed();
            cb_pop_front(staging_cb, packets_per_group * tiles_per_packet);
        }
    }
    connection.close();
}
