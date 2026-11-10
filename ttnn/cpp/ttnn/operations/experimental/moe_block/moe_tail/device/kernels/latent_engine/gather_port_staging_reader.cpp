// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// gather_port_staging_reader: a latent engine gather port's reader - its units' shares from the owners to staging
// Runs on: latent engine gather port cores (plan.port_cores, the last grid column), RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - polls the owner engine cores' landed words (CB_STEP_WORDS) and reads the landed / tilized shares from their
//    landing slots (landing_cb, L1) over NoC 0, two groups in flight (transaction ids)
//  - pushes each group (packets_per_group packets' pages) into staging_cb and its packet count into packet_count_cb for
//    gather_port_fabric_sender
//  - waits on the zero gate (PORT_READER_GO_SEM) each row block; counts in on the engine gate lead's
//    BLOCK_GATE_COUNT_SEM when done with a block

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/latent_engine/gather_port_common.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    moe_block_args::load();  // (my args: this chip's block of the DRAM table, common/args_table.hpp)
    const uint32_t landing = get_write_ptr(landing_cb), poll = get_write_ptr(poll_cb);
    size_t arg_index = 0;
    const uint32_t step_words = get_write_ptr(CB_STEP_WORDS);  // (the owners' landed words: counted per block)
    const uint32_t num_units = moe_block_args::arg(arg_index++);
    // (unit: part | position here << 8; then the columns' share K ranges, the engine's NoC x by column and NoC y by
    // pair)
    const uint32_t units = arg_index, share_k_ranges = units + num_units,
                   column_xs = share_k_ranges + 2 * engine_columns;
    const uint32_t pair_ys = column_xs + engine_columns;
    const uint32_t lead_xy = moe_block_args::arg(pair_ys + num_pairs);  // (the engine's gate lead: x | y << 16)
    volatile tt_l1_ptr uint32_t* poll_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(poll);
    // a group: packets_per_group packets' slots reserved, read, one barrier, pushed whole (padded) with its packet
    // count in packet_count_cb - a share not landed yet ends the group early (never hold read packets while waiting)
    // (two groups in flight: a group's pages pushed once its reads are issued, its packet count (what the sender waits
    // for first) once they landed - its transaction id's barrier, after the next group's reads went out: a barrier a
    // group was the reader's - and the gather's - pace, ~16 GB/s a link of 23.7)
    uint32_t packets = 0, group_l1 = 0, transaction_id = 1, pending_packets = 0, pending_transaction_id = 0;
    auto land_pending = [&]() {
        if (pending_transaction_id) {
            noc_async_read_barrier_with_trid(pending_transaction_id);
            cb_reserve_back(packet_count_cb, 1);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(packet_count_cb)) = pending_packets;
            cb_push_back(packet_count_cb, 1);
            pending_transaction_id = 0;
        }
    };
    auto flush = [&]() {
        cb_push_back(staging_cb, packets_per_group * tiles_per_packet);
        land_pending();
        pending_packets = packets, pending_transaction_id = transaction_id, transaction_id = 3 - transaction_id,
        packets = 0;
    };
    // (the shares' landed words polled a unit at a time: every not-yet-seen share's word in one batch of NoC reads,
    // one barrier - one round trip a share in turn was the gather's pace; a share still not in is then waited alone)
    bool seen[row_tiles_per_chunk * engine_columns];
    const uint32_t poll_buffer = poll + STEP_WORD_BYTES;  // (16 B a share, past the poll word)
    for (uint32_t block = 0; block < row_blocks; ++block) {
        const uint32_t expected = block + 1;  // (block b's landed words: l + 1 once a share landed)
        wait_zero_gate(PORT_READER_GO_SEM);   // (block 0: the owners' words zeroed; block l: everyone done with l - 1)
        for (uint32_t unit_index = 0; unit_index < num_units; ++unit_index) {
            DeviceZoneScopedN("TP-RUNIT");
            const uint32_t unit = moe_block_args::arg(units + unit_index), part = unit & 0xFF, position = unit >> 8;
            for (uint32_t i = 0; i < row_tiles_per_chunk * engine_columns; ++i) {
                seen[i] = false;
            }
            auto poll_all = [&]() {
                for (uint32_t chunk_row_tile = 0; chunk_row_tile < row_tiles_per_chunk; ++chunk_row_tile) {
                    const uint32_t row_tile = position * row_tiles_per_chunk + chunk_row_tile;
                    for (uint32_t column = 0; column < engine_columns; ++column) {
                        if (seen[chunk_row_tile * engine_columns + column] ||
                            (part == 1 && column >= engine_columns / 2) || (part == 2 && column < engine_columns / 2)) {
                            continue;
                        }
                        noc_async_read(
                            get_noc_addr(
                                moe_block_args::arg(column_xs + column),
                                moe_block_args::arg(pair_ys + row_tile % num_pairs),
                                step_words + STEP_WORD_BYTES * (row_tile / num_pairs)),
                            poll_buffer + STEP_WORD_BYTES * (chunk_row_tile * engine_columns + column),
                            4);
                    }
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t i = 0; i < row_tiles_per_chunk * engine_columns; ++i) {
                    if (!seen[i] && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                        poll_buffer + STEP_WORD_BYTES * i) >= expected) {
                        seen[i] = true;
                    }
                }
            };
            walk(
                part,
                position,
                share_k_ranges,
                [&](uint32_t pair,
                    uint32_t step,
                    uint32_t column,
                    uint32_t k_first,
                    uint32_t packet_k_first,
                    uint32_t tiles,
                    bool) {
                    const uint32_t owner_x = moe_block_args::arg(column_xs + column);
                    const uint32_t owner_y = moe_block_args::arg(pair_ys + pair);
                    const uint32_t share_index =
                        (step * num_pairs + pair - position * row_tiles_per_chunk) * engine_columns + column;
                    // (the share landed / tilized at its owner here: its landed word, over the NoC)
                    if (packet_k_first == k_first && !seen[share_index]) {
                        if (packets == 0) {
                            poll_all();  // (no reads in flight of mine: the batch's barrier waits for its words only)
                        }
                        auto landed = [&]() {
                            noc_async_read(
                                get_noc_addr(owner_x, owner_y, step_words + STEP_WORD_BYTES * step), poll, 4);
                            noc_async_read_barrier();
                            invalidate_l1_cache();  // (the copy just landed: not the cached word)
                            return *poll_word >= expected;
                        };
                        if (!seen[share_index] && !landed()) {
                            if (packets > 0) {
                                flush();
                            }
                            land_pending();                 // (the sender has everything read so far while I wait)
                            DeviceZoneScopedN("TP-RWAIT");  // (a share not landed / tilized yet)
                            while (!landed()) {
                            }
                        }
                    }
                    if (packets == 0) {
                        cb_reserve_back(staging_cb, packets_per_group * tiles_per_packet);
                        group_l1 = get_write_ptr(staging_cb);
                    }
                    noc_async_read_set_trid(transaction_id);
                    noc_async_read(
                        get_noc_addr(
                            owner_x,
                            owner_y,
                            landing + (step * max_share_tiles + packet_k_first - k_first) * tile_bytes),
                        group_l1 + packets * tiles_per_packet * tile_bytes,
                        tiles * tile_bytes);
                    if (++packets == packets_per_group) {
                        flush();
                    }
                });
        }
        if (packets > 0) {
            flush();
        }
        land_pending();                // (every landing slot read of block b has landed here)
        if (block + 1 < row_blocks) {  // (done with block b's landing slots: count in on the engine's gate)
            noc_semaphore_inc(get_noc_addr(lead_xy & 0xFFFF, lead_xy >> 16, get_semaphore(BLOCK_GATE_COUNT_SEM)), 1);
            noc_async_atomic_barrier();
        }
    }
}
