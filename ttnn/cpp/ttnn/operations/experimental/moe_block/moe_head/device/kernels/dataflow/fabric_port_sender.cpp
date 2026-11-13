// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_port_sender: sends everything this chip sends through one fabric port core (a direction and link)
// Runs on: port cores (one per X direction and fabric link), RISCV_1 (NCRISC), NoC 0 (RISCV_0_default)
// Responsibilities:
//  - x all-gather: pops CB 0 (fabric_port_stager) and fabric-writes each packet into the neighbour's matmul cores'
//    x-own L1, incrementing the downstream port's arrival counter; +1 on the neighbour's x_gather_watcher a shard
//  - from CB 0: reduce-scatter sub-blocks (the downstream chip's zones), latent all-to-all rows (the destination's
//    LATZ); from L1: router logits partials (the owner chips' router arrival zones), the routing exchange's X packets
//    (the destination chips' DRAM slots)
//  - fabric fences and credits: "ready" / "started" words, zone credits after "router done"; re-arms its words
// Note: in this op the reading side runs on BRISC (RISCV_0) and the sending side on NCRISC (RISCV_1), opposite to
//  tt-metal's Reader/Writer config convention, on purpose
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
// (no public device-side fabric API: the packet headers, the fabric connection, the header pool, the routes)
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/x_gather.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/latent_all_to_all.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/reduce_scatter.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/router_fabric_sends.hpp"

// An entry of my x gather: a shard this port sends on, in place - 3 words each (head_gather_program.cpp
// ChipProgram::gather, `sender_entries`)
struct SenderEntries {
    volatile tt_l1_ptr uint32_t* words;
    uint32_t shard(uint32_t entry) const { return words[3 * entry]; }
    uint32_t part(uint32_t entry) const { return words[3 * entry + 1]; }  // 0: whole; 1 / 2: a ring half
    uint32_t packets_per_chunk(uint32_t entry) const { return words[3 * entry + 2]; }
};

// The logits' end-to-end fence (head_gather_program.cpp ChipProgram::logits_block): my fence word, the "started"s my
// logits need (one an owner chip I send to), then the chips whose logits come to me through this port's direction and
// link
struct RouterLogitsFenceArgs {
    uint32_t fence_address;
    uint32_t fence_count;
    FabricStartedTargets started_targets;
};

// This kernel's runtime args (head_gather_program.cpp ChipProgram::gather, `args`), read once in kernel_main. Words [6
// .. 16] are 0 on a port without a peer (no neighbour in its direction).
struct FabricPortSenderArgs {
    uint32_t x_own_address;            // [0] the matmul cores' x-own L1 (the neighbour's: the same address)
    uint32_t gather_link;              // [1] my fabric link
    uint32_t arrival_counter_address;  // [2] the port's arrival counter (mine, the downstream port's: the same)
    uint32_t expected_arrivals;        // [3] the upstream's increments of mine a call (every chunk live)
    uint32_t ready_address;            // [4] the "ready" fence word
    uint32_t wait_ready;               // [5] wait for my neighbour's "ready" (I have entries to send)
    uint32_t send_ready;               // [6] tell the neighbour's port facing me "ready" (it has entries to send)
    uint32_t facing_port_noc_x, facing_port_noc_y;          // [7], [8] the neighbour's port facing me
    uint32_t downstream_port_noc_x, downstream_port_noc_y;  // [9], [10] the neighbour's port on in my direction
    uint16_t neighbour_mesh_id, neighbour_chip_id;          // [11], [12]
    uint32_t watcher_noc_x, watcher_noc_y;                  // [13], [14] the neighbour's watcher core
    uint32_t watcher_counter_address;                       // [15] my counter there
    uint32_t num_entries;                                   // [16]
    SenderEntries entries;                                  // [17 ..] num_entries x 3 words
    XGatherDestinationTables destination_tables;            // then: the neighbour's (HeadModel::tables)
    ReduceScatterArgs reduce_scatter;                       // then: reduce_scatter_block
    RouterLogitsStreamArgs logits;                          // then: port_block
    LatentAllToAllArgs latent_all_to_all;                   // then: latent_block
    RouterLogitsFenceArgs logits_fence;                     // then: logits_block
    RoutingExchangeSendArgs routing_exchange;               // then: send_block
    // then (a port with a peer only: else past the args - read only when the port sends) the fabric connection, in
    // place: its eth channel, teardown and buffer-index semaphores
    volatile tt_l1_ptr uint32_t* fabric_connection;
#ifdef SKIP_PADDING_CHUNKS
    XGatherLiveChunkArgs live_chunks;  // (common args)
#endif
};

FORCE_INLINE FabricPortSenderArgs read_fabric_port_sender_args() {
    FabricPortSenderArgs args;
    uint32_t arg_index = 0;
    args.x_own_address = moe_block_args::arg(arg_index++);
    args.gather_link = moe_block_args::arg(arg_index++);
    args.arrival_counter_address = moe_block_args::arg(arg_index++);
    args.expected_arrivals = moe_block_args::arg(arg_index++);
    args.ready_address = moe_block_args::arg(arg_index++);
    args.wait_ready = moe_block_args::arg(arg_index++);
    args.send_ready = moe_block_args::arg(arg_index++);
    args.facing_port_noc_x = moe_block_args::arg(arg_index++);
    args.facing_port_noc_y = moe_block_args::arg(arg_index++);
    args.downstream_port_noc_x = moe_block_args::arg(arg_index++);
    args.downstream_port_noc_y = moe_block_args::arg(arg_index++);
    args.neighbour_mesh_id = moe_block_args::arg(arg_index++);
    args.neighbour_chip_id = moe_block_args::arg(arg_index++);
    args.watcher_noc_x = moe_block_args::arg(arg_index++);
    args.watcher_noc_y = moe_block_args::arg(arg_index++);
    args.watcher_counter_address = moe_block_args::arg(arg_index++);
    args.num_entries = moe_block_args::arg(arg_index++);
    args.entries.words = moe_block_args::arg_pointer(arg_index);
    arg_index += 3 * args.num_entries;
    args.destination_tables.words = moe_block_args::arg_pointer(arg_index);
    arg_index += XGatherDestinationTables::num_words;
    args.reduce_scatter = read_reduce_scatter_args(arg_index);
    args.logits = read_router_logits_stream_args(arg_index);
    args.latent_all_to_all = read_latent_all_to_all_args(arg_index);
    args.logits_fence.fence_address = moe_block_args::arg(arg_index++);
    args.logits_fence.fence_count = moe_block_args::arg(arg_index++);
    args.logits_fence.started_targets.count = moe_block_args::arg(arg_index++);
    args.logits_fence.started_targets.words = moe_block_args::arg_pointer(arg_index);
    arg_index += 2 * args.logits_fence.started_targets.count;
    args.routing_exchange = read_routing_exchange_send_args(arg_index);
    args.fabric_connection = moe_block_args::arg_pointer(arg_index);
    arg_index += 3;
#ifdef SKIP_PADDING_CHUNKS
    args.live_chunks = read_x_gather_live_chunk_args();
#endif
    return args;
}

// One row block of this kernel's work: kernel_main below runs it once per block (row_block.hpp).
void process_row_block(uint32_t block, const FabricPortSenderArgs& args) {
    // (row blocks: every block allocates its packet headers anew - the last block's packets were flushed by its
    // connection's close)
    PacketHeaderPool::reset();
#ifdef SKIP_PADDING_CHUNKS
    x_gather_find_live_chunks(args.live_chunks);  // (a padded forward: the row chunks up to the last with a real row)
#endif
    const ReduceScatterArgs& reduce_scatter = args.reduce_scatter;
    const LatentAllToAllArgs& latent = args.latent_all_to_all;
    const RouterLogitsFenceArgs& logits_fence = args.logits_fence;
    // (chunks: the zone credits - my zone-free word (my chip's cores, a chunk's arrivals from my peer read), the
    // signals a chunk; forwarded to the peer's port facing me (its zone-credit word): row_chunks - 1 a call)
    volatile tt_l1_ptr uint32_t* zone_free =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_scatter.zone_free_address);
    const uint32_t zone_free_signals = reduce_scatter.zone_free_signals_per_chunk;
    const uint32_t zone_credits_expected =
        reduce_scatter.zone_credits_expected
            ? reduce_scatter.zone_credits_expected - (row_chunks - x_gather_num_live_chunks)
            : 0;  // (the credits a call expects: one a live chunk on)
    const uint32_t zone_credit_address = reduce_scatter.zone_credit_address;
    const uint32_t zone_credits_to_forward = zone_free_signals && row_chunks > 1 ? x_gather_num_live_chunks - 1 : 0;
    // (the router's buffers over this chip's arrival zones / LATZ: every matmul core counts its router phase over on
    // my "router done" word (router_done_count of them) - only then the peer's chunk 0 credit and the latent's
    // "started" go)
    volatile tt_l1_ptr uint32_t* router_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_scatter.router_done_address);
    const uint32_t router_done_count = reduce_scatter.router_done_count();
    // (what waits for it: chunk 0's credit, "started")
    const bool router_done_gates_zone = reduce_scatter.router_done_gates_zone(),
               router_done_gates_latent = reduce_scatter.router_done_gates_latent();
    bool router_done_told = router_done_count == 0;
    uint32_t zone_credits_forwarded = 0;
    RouterLogitsStream logits_stream{args.logits};
    // (the latent's end-to-end fence: my fence word, the "started" count my sends need (one a destination chip), then
    // the chips I tell "started" - their latent ports toward me)
    const FabricStartedTargets& latent_started_targets = latent.started_targets;
    // (the logits' end-to-end fence: my fence word, the "started" count my logits need (one an owner chip I send to),
    // then the chips whose logits come to me through this port's direction and link)
    const FabricStartedTargets& logits_started_targets = logits_fence.started_targets;
    volatile tt_l1_ptr uint32_t* logits_fence_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(logits_fence.fence_address);
    RoutingExchangeSend routing_exchange{args.routing_exchange};
    if (args.num_entries > 0 || args.send_ready || reduce_scatter.num_units > 0 || args.logits.num_units > 0 ||
        latent.num_units > 0 || latent_started_targets.count > 0 || logits_started_targets.count > 0 ||
        !routing_exchange.done || zone_credits_to_forward || (router_done_gates_zone && zone_free_signals)) {
        auto connection = WorkerToFabricEdmSender::build_from_eth_channel<ProgrammableCoreType::TENSIX>(
            args.fabric_connection[0], args.fabric_connection[1], args.fabric_connection[2]);
        // (packets_per_group data headers, the blocking header, then one a latent destination)
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_headers[packets_per_group + 1 + 3];
        for (uint32_t header = 0; header <= packets_per_group; ++header) {
            packet_headers[header] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(packet_headers[header], args.neighbour_chip_id, args.neighbour_mesh_id);
        }
        // (the blocking header: fences, credits, the watcher's count, a padding tile's arrival count)
        volatile tt_l1_ptr PACKET_HEADER_TYPE* blocking_header = packet_headers[packets_per_group];
        volatile tt_l1_ptr PACKET_HEADER_TYPE* routing_exchange_header =
            PacketHeaderPool::allocate_header();  // (routing: routed a packet)
        volatile tt_l1_ptr PACKET_HEADER_TYPE* logits_header =
            PacketHeaderPool::allocate_header();                                // (logits: routed a unit)
        for (uint32_t unit = 0; unit < latent.num_units && unit < 3; ++unit) {  // (the latent's destinations)
            const LatentAllToAllUnit latent_unit = latent.unit(unit);
            packet_headers[packets_per_group + 1 + unit] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(
                packet_headers[packets_per_group + 1 + unit],
                latent_unit.chip_id(),
                static_cast<uint16_t>(latent_unit.mesh_id));
        }
        connection.open();
        // a fabric atomic increment of the word at `noc_address` through `header`, flushed blocking
        auto send_increment = [&](volatile tt_l1_ptr PACKET_HEADER_TYPE* header, uint64_t noc_address) {
            header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc_address, 1, true});
            connection.wait_for_empty_write_slot();
            connection.send_payload_flush_blocking_from_address(
                reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
        };
        // "started": +1 on the fence word of every target's port, `header` re-routed to each target
        auto tell_started = [&](volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
                                const FabricStartedTargets& targets,
                                uint32_t fence_address) {
            for (uint32_t i = 0; i < targets.count; ++i) {
                (void)fabric_set_unicast_route(header, targets.chip_id(i), targets.mesh_id(i));
                send_increment(header, get_noc_addr(targets.port_noc_x(i), targets.port_noc_y(i), fence_address));
            }
        };
        if (args.send_ready) {  // fence: I started this call - the neighbour's port facing me may write into my L1
            send_increment(
                blocking_header, get_noc_addr(args.facing_port_noc_x, args.facing_port_noc_y, args.ready_address));
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* latent_started_header =
            latent_started_targets.count > 0 ? PacketHeaderPool::allocate_header() : nullptr;
        // (every chip sending me latent may write my LATZ - its latent port's fence word)
        auto send_latent_started = [&]() {
            tell_started(latent_started_header, latent_started_targets, latent.fence_address);
        };
        if (!router_done_gates_latent) {
            send_latent_started();  // (I started this call: LATZ is free - no router buffers over it)
        }
        if (logits_started_targets.count > 0) {
            // (I started this call: every chip whose logits come to me through this port may write my router cores'
            // arrival zone - its logits port's fence word; my neighbour's ready only fences the first hop. The logits
            // header is free until the first logits packet: re-routed each)
            tell_started(logits_header, logits_started_targets, logits_fence.fence_address);
        }
        bool logits_owners_started = logits_fence.fence_count == 0;  // (every owner I send logits to started)
#ifdef ROUTER_LOGITS_ITERATIONS
        // router iterations: the iteration whose logits I send, and the "started" rounds told past the first - round k
        // once my chip's router cores are all done with iteration k - 1 (their count on my iteration word, + 16)
        uint32_t logits_iteration = 0, logits_started_rounds = 0;
        volatile tt_l1_ptr uint32_t* router_iterations_done =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.logits.iteration_word_address + 16);
        auto send_logits_started_rounds = [&]() {
            if (logits_started_rounds + 1 < ROUTER_LOGITS_ITERATIONS && logits_started_targets.count > 0) {
                invalidate_l1_cache();
                if (*router_iterations_done >= args.logits.router_counts_per_iteration * (logits_started_rounds + 1)) {
                    noc_async_writes_flushed();  // (the logits header's last packet left: re-routed here)
                    tell_started(logits_header, logits_started_targets, logits_fence.fence_address);
                    ++logits_started_rounds;
                }
            } else if (logits_started_rounds + 1 < ROUTER_LOGITS_ITERATIONS) {
                ++logits_started_rounds;  // (no chip sends me logits through this port)
            }
        };
#endif
        if (args.wait_ready) {
            DeviceZoneScopedN("SE-RDYWAIT");  // (its end: the neighbour's port started - launch skew + a hop)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.ready_address), 1);
            noc_semaphore_inc(get_noc_addr(args.ready_address), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t downstream_arrival_noc_address =
            get_noc_addr(args.downstream_port_noc_x, args.downstream_port_noc_y, args.arrival_counter_address);
        uint32_t next_header = 0, unflushed_packets = 0;
        // (every matmul core's router phase is over: the zones, LATZ free - also between the x gather's packets: the
        // upstream's chunk 0 reduce-scatter waits for it)
        auto poll_router_done = [&]() {
            if (!router_done_told && *router_done >= router_done_count) {
                if (router_done_gates_zone && zone_free_signals) {  // (chunk 0's credit)
                    send_increment(
                        blocking_header,
                        get_noc_addr(args.facing_port_noc_x, args.facing_port_noc_y, zone_credit_address));
                }
                if (router_done_gates_latent) {
                    send_latent_started();
                }
                router_done_told = true;
            }
        };
        // (my chip's cores read a chunk's zones: credit the peer's sends, a blocking header - the gather's fence /
        // watcher header, never in flight across this)
        auto poll_zone_free = [&]() {
            poll_router_done();
            if (router_done_told && zone_credits_forwarded < zone_credits_to_forward &&
                *zone_free >= (zone_credits_forwarded + 1) * zone_free_signals) {
                send_increment(
                    blocking_header, get_noc_addr(args.facing_port_noc_x, args.facing_port_noc_y, zone_credit_address));
                ++zone_credits_forwarded;
            }
        };
        // (what goes between the x gather's / the latent's / the reduce-scatter's packets: credits, logits, routing)
        auto poll_interleaved_sends = [&]() {
            poll_zone_free();
#ifdef ROUTER_LOGITS_ITERATIONS
            send_logits_started_rounds();
            if (!logits_owners_started) {
                invalidate_l1_cache();
                // (every owner I send to: its router cores free for iteration logits_iteration)
                logits_owners_started = *logits_fence_word >= logits_fence.fence_count * (logits_iteration + 1);
            }
            if (logits_owners_started) {
                logits_stream.poll(connection, logits_header, false);
                // (iteration logits_iteration's sub-blocks all left: its slots are free)
                if (logits_stream.done() && logits_iteration + 1 < ROUTER_LOGITS_ITERATIONS) {
                    noc_async_writes_flushed();
                    if (args.logits.active()) {
                        noc_semaphore_inc(
                            get_noc_addr(
                                args.logits.router_core_0_noc & 0xFFFF,
                                args.logits.router_core_0_noc >> 16,
                                args.logits.iteration_word_address),
                            1);
                    }
                    logits_stream.next_iteration();
                    ++logits_iteration;
                    logits_owners_started = logits_fence.fence_count == 0;
                }
            }
#else
            if (!logits_owners_started) {
                invalidate_l1_cache();
                logits_owners_started = *logits_fence_word >= logits_fence.fence_count;
            }
            if (logits_owners_started) {
                logits_stream.poll(connection, logits_header, false);
            }
#endif
            routing_exchange.poll(
                connection,
                routing_exchange_header,
                false,
                ROUTING_EXCHANGE_NUM_CHIPS,
                ROUTING_EXCHANGE_PLANNER_INSTANCES,
                ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
                ROUTING_EXCHANGE_MY_CHIP,
                ROUTING_EXCHANGE_HEADER_BYTES,
                ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
                ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
                ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);
        };
        for (uint32_t chunk = 0; chunk < x_gather_num_live_chunks && args.num_entries > 0; ++chunk) {
            for (uint32_t entry = 0; entry < args.num_entries; ++entry) {
                const uint32_t packets_per_chunk = args.entries.packets_per_chunk(entry);
                uint32_t packets_sent = 0;
                const auto real_row_tiles = x_gather_mask(chunk);  // (the chunk's real row tiles)
                x_gather_walk(
                    chunk,
                    args.entries.shard(entry),
                    args.entries.part(entry),
                    args.gather_link,
                    [&](uint32_t row_tile,
                        uint32_t k_step,
                        uint32_t,
                        uint32_t step_tile,
                        uint32_t packet_tiles,
                        uint32_t) {
                        // (chunks: the upstream's chunk 0 reduce-scatter comes before my gather of the last chunk's x
                        // ends; one chunk: after it - no blocking header between the x packets, it cost K3 640 ~3 us
                        // of gather)
                        if (row_chunks > 1 && !router_done_told) {
                            invalidate_l1_cache();
                            poll_router_done();
                        }
#if ROUTER_LOGITS_FIRST
                        poll_interleaved_sends();  // (GLM-type: the logits ahead of the x gather)
                        while (!cb_pages_available_at_front(0, tiles_per_packet * (unflushed_packets + 1))) {
                            poll_interleaved_sends();
                        }
#endif
                        cb_wait_front(0, tiles_per_packet * (unflushed_packets + 1));
                        const uint32_t source_address =
                            get_read_ptr(0) + unflushed_packets * tiles_per_packet * moe_block_tiles::kBf16TileBytes;
                        const bool carries_increment =
                            (++packets_sent % packets_per_arrival_increment) == 0 || packets_sent == packets_per_chunk;
                        if ((real_row_tiles >> (row_tile - chunk * chunk_row_tiles)) & 1) {
                            const XGatherDestination destination = args.destination_tables.destination(
                                args.x_own_address, chunk, row_tile, k_step, step_tile);
                            const uint64_t destination_noc_address =
                                get_noc_addr(destination.noc_x, destination.noc_y, destination.address);
                            volatile tt_l1_ptr PACKET_HEADER_TYPE* header = packet_headers[next_header];
                            next_header = next_header + 1 == packets_per_group ? 0 : next_header + 1;
                            if (carries_increment) {
                                header->to_noc_fused_unicast_write_atomic_inc(
                                    NocUnicastAtomicIncFusedCommandHeader{
                                        destination_noc_address, downstream_arrival_noc_address, 1, true},
                                    packet_tiles * moe_block_tiles::kBf16TileBytes);
                            } else {
                                header->to_noc_unicast_write(
                                    NocUnicastCommandHeader{destination_noc_address},
                                    packet_tiles * moe_block_tiles::kBf16TileBytes);
                            }
                            connection.wait_for_empty_write_slot();
                            connection.send_current_slot_non_blocking(
                                source_address,
                                packet_tiles * moe_block_tiles::kBf16TileBytes,
                                reinterpret_cast<uint32_t>(header));
                        } else if (carries_increment) {  // (a padding row tile: no x - only the count its packets
                                                         // carry)
                            send_increment(blocking_header, downstream_arrival_noc_address);
                        }
                        if (++unflushed_packets == packets_per_group || packets_sent == packets_per_chunk) {
                            noc_async_writes_flushed();
                            cb_pop_front(0, tiles_per_packet * packets_per_group);
                            unflushed_packets = 0;
                        }
                    });
                // this chunk's shard `entry` all sent: the neighbour's watcher (behind its packets on this link)
                send_increment(
                    blocking_header,
                    get_noc_addr(args.watcher_noc_x, args.watcher_noc_y, args.watcher_counter_address));
            }
        }
        poll_interleaved_sends();
        uint32_t packets_popped = 0;  // (as the stager's packets_pushed)
        for (uint32_t chunk = 0; chunk < x_gather_num_live_chunks; ++chunk) {
            latent_all_to_all_send(latent, connection, packet_headers, poll_interleaved_sends, packets_popped, chunk);
            reduce_scatter_send(
                reduce_scatter, connection, packet_headers, poll_interleaved_sends, packets_popped, chunk);
        }
        if constexpr (row_block::BLOCKS > 1) {
            if (packets_popped % packets_per_group) {  // (row blocks: the stager padded its last group - the next
                                                       // block's gather starts aligned)
                cb_wait_front(0, (packets_per_group - packets_popped % packets_per_group) * tiles_per_packet);
                cb_pop_front(0, (packets_per_group - packets_popped % packets_per_group) * tiles_per_packet);
            }
        }
        // (the chunks' credits still owed; the router-done credit / "started")
        while (zone_credits_forwarded < zone_credits_to_forward || !router_done_told) {
            invalidate_l1_cache();
            poll_interleaved_sends();
        }
#ifdef ROUTER_LOGITS_ITERATIONS
        while (logits_iteration + 1 < ROUTER_LOGITS_ITERATIONS || !logits_stream.done() ||
               logits_started_rounds + 1 < ROUTER_LOGITS_ITERATIONS) {  // (every iteration's logits; every "started")
            poll_interleaved_sends();
        }
#else
        if (logits_fence.fence_count) {
            noc_semaphore_wait_min(logits_fence_word, logits_fence.fence_count);
        }
        logits_stream.poll(connection, logits_header, true);  // (whatever is left of the logits)
#endif
        routing_exchange.poll(
            connection,
            routing_exchange_header,
            true,
            ROUTING_EXCHANGE_NUM_CHIPS,
            ROUTING_EXCHANGE_PLANNER_INSTANCES,
            ROUTING_EXCHANGE_EXPERTS_PER_CHIP,
            ROUTING_EXCHANGE_MY_CHIP,
            ROUTING_EXCHANGE_HEADER_BYTES,
            ROUTING_EXCHANGE_EXPERT_COUNT_BLOCK_BYTES,
            ROUTING_EXCHANGE_INSTANCE_BLOCK_BYTES,
            ROUTING_EXCHANGE_MAX_PACKET_PAYLOAD_BYTES);  // (the routing)
        noc_async_writes_flushed();
        if (zone_credits_to_forward) {
            noc_semaphore_inc(
                get_noc_addr(reinterpret_cast<uint32_t>(zone_free)),
                0u - zone_credits_to_forward * zone_free_signals);  // (re-arm)
        }
        if (zone_credits_expected) {  // (every credit of the call came: they were waited for, the last before the
                                      // last chunk)
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zone_credit_address), zone_credits_expected);
            noc_semaphore_inc(get_noc_addr(zone_credit_address), 0u - zone_credits_expected);
        }
        logits_stream.rearm();
        if (latent.fence_count) {  // (re-arm: a next call's early "started" stays)
            noc_semaphore_inc(get_noc_addr(latent.fence_address), 0u - latent.fence_count);
        }
#ifdef ROUTER_LOGITS_ITERATIONS
        // (a "started" a router iteration from each owner I send to)
        constexpr uint32_t logits_started_rounds_per_call = ROUTER_LOGITS_ITERATIONS;
#else
        constexpr uint32_t logits_started_rounds_per_call = 1;
#endif
        if (logits_fence.fence_count) {  // (re-arm, as the latent's)
            noc_semaphore_inc(
                get_noc_addr(logits_fence.fence_address),
                0u - logits_fence.fence_count * logits_started_rounds_per_call);
        }
        noc_async_atomic_barrier();
        connection.close();
    }
#ifdef ROUTER_LOGITS_ITERATIONS
    {  // (re-arm: my chip's router cores counted every iteration but the last here - also on a port with nothing to
       // send)
        volatile tt_l1_ptr uint32_t* router_iterations_done =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.logits.iteration_word_address + 16);
        noc_semaphore_wait_min(
            router_iterations_done, args.logits.router_counts_per_iteration * (ROUTER_LOGITS_ITERATIONS - 1));
        noc_semaphore_inc(
            get_noc_addr(args.logits.iteration_word_address + 16),
            0u - args.logits.router_counts_per_iteration * (ROUTER_LOGITS_ITERATIONS - 1));
        noc_async_atomic_barrier();
    }
#endif
    if (router_done_count) {  // (re-arm: every matmul core counted its router phase here - also on a port with
                              // nothing to send)
        noc_semaphore_wait_min(router_done, router_done_count);
        noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(router_done)), 0u - router_done_count);
        noc_async_atomic_barrier();
    }
    if (args.expected_arrivals > 0) {  // everything from upstream landed; re-arm for the next call
        volatile tt_l1_ptr uint32_t* arrived =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.arrival_counter_address);
        // (as many a chunk: the live ones)
        const uint32_t live_arrivals = args.expected_arrivals / row_chunks * x_gather_num_live_chunks;
        noc_semaphore_wait_min(arrived, live_arrivals);
        noc_semaphore_inc(get_noc_addr(args.arrival_counter_address), 0u - live_arrivals);
        noc_async_atomic_barrier();
    }
}

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    const FabricPortSenderArgs args = read_fabric_port_sender_args();
    for (uint32_t block = 0; block < row_block::BLOCKS; ++block) {
        row_block::begin(block);
        process_row_block(block, args);
        row_block::end(block);
    }
}
