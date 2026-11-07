// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_matmul_prefill: one sender of the TP all-gather, on a worker core beside the matmul grid.
//
// Each sender holds one fabric connection (one direction, one link) and sends this rank's shard rows to the ranks it
// reaches over it: the row blocks first_row_block, + row_block_step, ... (one block per in0-sender row of the matmul,
// row_block_tiles tile rows of the shard) into their places in the destination's gathered in0, K block by K block,
// three tiles a packet (scatter writes). Each row's K block ends with a packet fused with an increment of that rank's
// slice counter on the in0 sender of the row: the in0 sender starts a K block of the slice once the counter holds all
// its rows' blocks so far.
//
// The gathered buffer and the counters persist across calls: before its first packet a sender waits until every rank
// it writes to has started this call (its matmul no longer reads the previous call's gathered rows, and its counters
// are consumed), and it tells the senders that write into this rank the same.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric::linear::experimental;

// Connection arguments: direction tag, ETH channel, teardown semaphore id, buffer-index semaphore id.
constexpr size_t connection_args = 4;
constexpr size_t teardown_arg = 2;
constexpr size_t buffer_index_arg = 3;
constexpr uint32_t max_dests = 2;
constexpr uint32_t headers_per_dest = 4;
static_assert(
    max_dests * headers_per_dest <= MEM_PACKET_HEADER_POOL_SIZE / MaxDMProcessorsPerCoreType / PACKET_HEADER_MAX_SIZE,
    "the sender's packet headers exceed the packet header pool");

void kernel_main() {
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr uint32_t slice_tiles = get_arg(args::slice_tiles);          // the shard's width, in tiles
    constexpr uint32_t full_row_tiles = get_arg(args::full_row_tiles);    // the gathered in0's width, in tiles
    constexpr uint32_t row_block_tiles = get_arg(args::row_block_tiles);  // tile rows per in0-sender row block
    constexpr uint32_t num_row_blocks = get_arg(args::num_row_blocks);
    constexpr uint32_t num_named_runtime_args = get_arg(args::num_named_runtime_args);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);  // the matmul's K block, in tiles
    static_assert(block_tiles >= 2 && slice_tiles % block_tiles == 0, "a slice is whole K blocks of two tiles or more");
    constexpr uint32_t k_blocks = slice_tiles / block_tiles;
    constexpr uint32_t unit_bytes = block_tiles * tile_bytes;  // one row's K block: the staging unit
    constexpr uint32_t stage_units = 4;

    const uint32_t rank = get_arg(args::rank);
    const uint32_t first_row_block = get_arg(args::first_row_block);
    const uint32_t row_block_step = get_arg(args::row_block_step);
    const uint32_t copy_local = get_arg(args::copy_local);
    const uint32_t num_dests = get_arg(args::num_dests);
    const uint32_t ready_semaphore = get_arg(args::ready_semaphore);
    const uint32_t dest_chip[max_dests] = {get_arg(args::dest0_chip), get_arg(args::dest1_chip)};
    const uint32_t dest_mesh[max_dests] = {get_arg(args::dest0_mesh), get_arg(args::dest1_mesh)};
    const uint32_t dest_counter[max_dests] = {get_arg(args::dest0_counter), get_arg(args::dest1_counter)};
    // The sender on the destination that writes into this rank: x | y << 16.
    const uint32_t dest_partner[max_dests] = {get_arg(args::dest0_partner), get_arg(args::dest1_partner)};
    // 1D fabrics route by hop count (2D ones by the destination, set in each header below).
    const uint8_t dest_hops[max_dests] = {
        static_cast<uint8_t>(get_arg(args::dest0_hops)), static_cast<uint8_t>(get_arg(args::dest1_hops))};

    Noc noc;
    Scratchpad<uint32_t> scratch(scratch::stage);  // stage_units units
    const uint32_t stage = local_mem::detail::AddressAccess::address(scratch.local_mem());
    const auto input = TensorAccessor(tensor::input);
    const auto gathered = TensorAccessor(tensor::gathered);

    if (first_row_block >= num_row_blocks || (num_dests == 0 && !copy_local)) {
        return;
    }

    tt::tt_fabric::RoutingPlaneConnectionManager connections;
    size_t arg_index = num_named_runtime_args;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[max_dests][headers_per_dest];
    if (num_dests > 0) {
        // The host computes the connection arguments without this kernel's semaphore ids; bind them before opening.
        auto* connection_arg_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(arg_index));
        connection_arg_words[teardown_arg] = sem::teardown.id;
        connection_arg_words[buffer_index_arg] = sem::buffer_index.id;
        open_connections(connections, 1, arg_index);
        for (uint32_t d = 0; d < num_dests; ++d) {
            for (uint32_t h = 0; h < headers_per_dest; ++h) {
                headers[d][h] = PacketHeaderPool::allocate_header();
#if defined(FABRIC_2D)
                (void)fabric_set_unicast_route(headers[d][h], dest_chip[d], dest_mesh[d]);
#endif
            }
        }
    }
    // Then the in0 senders of the row blocks: x | y << 16 (the same cores on every rank).
    const size_t in0_sender_args = arg_index;
    auto* sender = num_dests > 0 ? &connections.get(0).sender : nullptr;

    // Entry: tell the senders writing into this rank that it has started, then wait for the ranks this one writes to.
    for (uint32_t d = 0; d < num_dests; ++d) {
        const uint32_t xy = dest_partner[d];
        fabric_unicast_noc_unicast_atomic_inc(
            sender,
            headers[d][0],
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                get_noc_addr(xy & 0xFFFF, xy >> 16, ready_semaphore), 1, true},
            dest_hops[d]);
    }

    // The tile rows this sender sends: row_block_tiles per row block, the blocks first_row_block, + row_block_step, ...
    // Per destination (nearest first) they go K block by K block, every row's block before the next block: the
    // matmul consumes a slice block by block across all its rows. A unit is one row's block: block_tiles tiles.
    const uint32_t blocks = (num_row_blocks - first_row_block + row_block_step - 1) / row_block_step;
    const uint32_t rows = blocks * row_block_tiles;
    const uint32_t units = rows * k_blocks;
    const auto unit_row = [&](uint32_t u) {
        const uint32_t r = u % rows;
        return (first_row_block + (r / row_block_tiles) * row_block_step) * row_block_tiles + r % row_block_tiles;
    };
    const auto unit_first_tile = [&](uint32_t u) { return (u / rows) * block_tiles; };  // within the row's slice
    const auto read_unit = [&](uint32_t u, uint32_t buffer) {
        const uint32_t first_page = unit_row(u) * slice_tiles + unit_first_tile(u);
        for (uint32_t t = 0; t < block_tiles; ++t) {
            noc.async_read(
                input,
                scratch,
                tile_bytes,
                {.page_id = first_page + t},
                {.offset_bytes = buffer * unit_bytes + t * tile_bytes});
        }
    };

    read_unit(0, 0);
    if (num_dests > 0) {
        noc_async_writes_flushed();
        auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_semaphore);
        noc_semaphore_wait_min(ready, num_dests);
        noc_semaphore_inc(get_noc_addr(ready_semaphore), 0u - num_dests);
        noc_async_atomic_barrier();
    }

    uint32_t header_slot = 0;
    const auto next_header = [&](uint32_t d) {
        if (header_slot == 0) {
            // Every header of the ring is about to be reused: their packets must have left L1.
            noc_async_writes_flushed();
        }
        auto* header = headers[d][header_slot];
        header_slot = header_slot + 1 == headers_per_dest ? 0 : header_slot + 1;
        return header;
    };
    // The own-row copy rides along the first destination's pass.
    const uint32_t passes = num_dests > 0 ? num_dests : 1;
    for (uint32_t step = 0; step < passes * units; ++step) {
        const uint32_t d = step / units;
        const uint32_t u = step % units;
        const uint32_t buffer = step % stage_units;
        noc.async_read_barrier();
        // The next unit's buffer held an earlier unit's packets: they must have left L1 before it is refilled.
        noc_async_writes_flushed();
        if (step + 1 < passes * units) {
            read_unit((step + 1) % units, (step + 1) % stage_units);
        }
        const uint32_t row = unit_row(u);
        const uint32_t xy = get_arg_val<uint32_t>(in0_sender_args + row / row_block_tiles);
        const uint32_t source = stage + buffer * unit_bytes;
        const uint32_t first_page = row * full_row_tiles + rank * slice_tiles + unit_first_tile(u);
        const auto tile_address = [&](uint32_t t) {
            return addrgen_detail::get_noc_address(gathered, first_page + t, 0);
        };
        if (d < num_dests) {
            // Three tiles a packet; the unit's last packet writes its last two and counts the row's block on the
            // destination's in0 sender (its flush lands the unit's earlier writes first).
            uint32_t t = 0;
            while (t + 2 < block_tiles) {
                const uint32_t n = block_tiles - t - 2 >= 3 ? 3 : block_tiles - t - 2;
                auto* header = next_header(d);
                if (n == 1) {
                    fabric_unicast_noc_unicast_write(
                        sender,
                        header,
                        source + t * tile_bytes,
                        tile_bytes,
                        tt::tt_fabric::NocUnicastCommandHeader{tile_address(t)},
                        dest_hops[d]);
                } else if (n == 2) {
                    fabric_unicast_noc_scatter_write(
                        sender,
                        header,
                        source + t * tile_bytes,
                        2 * tile_bytes,
                        tt::tt_fabric::NocUnicastScatterCommandHeader(
                            {tile_address(t), tile_address(t + 1)}, {static_cast<uint16_t>(tile_bytes)}),
                        dest_hops[d]);
                } else {
                    fabric_unicast_noc_scatter_write(
                        sender,
                        header,
                        source + t * tile_bytes,
                        3 * tile_bytes,
                        tt::tt_fabric::NocUnicastScatterCommandHeader(
                            {tile_address(t), tile_address(t + 1), tile_address(t + 2)},
                            {static_cast<uint16_t>(tile_bytes), static_cast<uint16_t>(tile_bytes)}),
                        dest_hops[d]);
                }
                t += n;
            }
            fabric_unicast_noc_fused_scatter_write_atomic_inc(
                sender,
                next_header(d),
                source + t * tile_bytes,
                2 * tile_bytes,
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader(
                    {tile_address(t), tile_address(t + 1)},
                    get_noc_addr(xy & 0xFFFF, xy >> 16, dest_counter[d]),
                    {static_cast<uint16_t>(tile_bytes)},
                    1,
                    true),
                dest_hops[d]);
        }
        if (copy_local && d == 0) {
            // This rank's own rows of the gathered in0, for its other readers (the matmul reads them from the shard).
            for (uint32_t t = 0; t < block_tiles; ++t) {
                noc.async_write(
                    scratch,
                    gathered,
                    tile_bytes,
                    {.offset_bytes = buffer * unit_bytes + t * tile_bytes},
                    {.page_id = first_page + t});
            }
        }
    }
    noc.async_read_barrier();
    noc_async_writes_flushed();
    if (num_dests > 0) {
        close_connections(connections);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
