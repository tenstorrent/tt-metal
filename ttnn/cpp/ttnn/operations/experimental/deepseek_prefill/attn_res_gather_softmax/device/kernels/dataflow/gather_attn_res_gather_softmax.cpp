// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The collective, on the one core per chip that holds a fabric connection. A link
// admits a single worker — every sender core in a fabric op takes its own link index
// — so the exchange is funnelled here rather than run from each worker.
//
// What crosses is this rank's whole statistics plane, written into the slot the peers'
// fold cores read for this rank. Peers write their own slots here symmetrically, so
// nothing has to be reordered on arrival and the reduction is the slot-wise sum the
// fold already performs.
//
// The workers pack the plane by column before signalling, so a plane is a page rather
// than a page per token row-tile. That is what makes this affordable from one core: the
// fabric charges per packet almost regardless of payload, and a tile-shaped plane costs
// Ht packets to carry 32 values each.
//
// Ordering: a peer's arrival increment is sent after every payload bound for that
// peer, on the same connection, and the fabric preserves order per connection. A
// non-zero arrival count therefore implies the payload behind it has landed.
//
// The arrival semaphore is a global semaphore, not a program-local one. Program-local
// semaphores are re-initialized at every launch, which would race a peer that is one
// program ahead; a global semaphore is written once and consumed here, in kernel, by
// subtracting what this launch waited for.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/dataflow/attn_res_stats_layout.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t num_stat_cores = get_compile_time_arg_val(2);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t stage_tiles = get_compile_time_arg_val(5);
    constexpr auto stats_args = TensorAccessorArgs<6>();

    // runtime args
    const auto stats_addr = get_arg_val<uint32_t>(0);
    const auto my_rank = get_arg_val<uint32_t>(1);
    const auto arrival_sem_addr = get_arg_val<uint32_t>(2);
    // The fold cores, as rectangles to multicast the release over: (start x, start y,
    // end x, end y, core count) each.
    const auto num_release_ranges = get_arg_val<uint32_t>(3);
    constexpr uint32_t kReleaseRangeArgIdx = 4;
    constexpr uint32_t kWordsPerReleaseRange = 5;

    constexpr uint32_t cb_id_headers = 8;
    constexpr uint32_t cb_id_stage = 9;

    constexpr uint32_t stat_tile_bytes = get_tile_size(cb_id_stage);
    constexpr uint32_t kStatsPerRow = 2;
    constexpr uint32_t kPeers = ring_size - 1;
    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    Noc noc;
    DataflowBuffer stage_buf(cb_id_stage);
    CircularBuffer cb_headers(cb_id_headers);
    Semaphore<> ready_sem(ready_sem_id);
    Semaphore<> done_sem(done_sem_id);

    // Page size is given explicitly — the accessor's compile-time value can be stale
    // on a program-cache hit — and it doubles as the fabric payload size.
    auto stats_accessor = TensorAccessor(stats_args, stats_addr, stat_tile_bytes);

    size_t fabric_arg_idx = kReleaseRangeArgIdx + kWordsPerReleaseRange * num_release_ranges;
    const auto num_connections = get_arg_val<uint32_t>(fabric_arg_idx++);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION>(fabric_arg_idx, num_connections);

    // One connection per distinct first-hop direction the peers need, so a peer is served
    // by whichever connection its own route's first hop names. A direction the host opened
    // nothing in is not a slot index and must never be used as one.
    constexpr uint8_t kNoConnection = 0xFF;
    uint8_t dir_to_slot[eth_chan_directions::COUNT];
    for (uint32_t d = 0; d < static_cast<uint32_t>(eth_chan_directions::COUNT); ++d) {
        dir_to_slot[d] = kNoConnection;
    }
    for (uint32_t i = 0; i < num_connections; ++i) {
        dir_to_slot[fabric_connections.get_tag(i)] = static_cast<uint8_t>(i);
    }

    // A peer's route follows the connection args, two words per peer in the order the
    // loop below takes them: the peer's mesh and chip, since the header type this kernel
    // is compiled against routes by node rather than by hop count.
    const size_t route_arg_idx = fabric_arg_idx;

    // One payload header and one increment header per peer: a header carries its
    // peer's route for the whole kernel, and peers in the same direction still differ by
    // route, so headers cannot be shared even within a direction.
    cb_headers.reserve_back(kStatsPerRow * kPeers);
    const uint32_t header_base = cb_headers.get_write_ptr();
    cb_headers.push_back(kStatsPerRow * kPeers);

    volatile PACKET_HEADER_TYPE* payload_headers[kPeers];
    volatile PACKET_HEADER_TYPE* inc_headers[kPeers];
    tt::tt_fabric::WorkerToFabricEdmSender* peer_connections[kPeers];

    for (uint32_t p = 0, slot = 0; p < ring_size; ++p) {
        if (p == my_rank) {
            continue;
        }
        payload_headers[slot] =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_base + (2 * slot) * packet_header_size_bytes);
        inc_headers[slot] =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_base + (2 * slot + 1) * packet_header_size_bytes);

        const ccl_routing_utils::line_unicast_route_info_t route_info{
            .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_arg_idx + 2 * slot)),
            .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_arg_idx + 2 * slot + 1))};
        ccl_routing_utils::fabric_set_line_unicast_route(payload_headers[slot], route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(inc_headers[slot], route_info);

        // The direction the routing tables will take this route's first hop in. Rank
        // order does not name it: on an axis that wraps, the shorter way round to a higher
        // rank runs in the lower direction, and a payload injected against its own route
        // sits on a router that will not carry it while the peer waits on an arrival that
        // never comes.
        const uint32_t first_hop =
            static_cast<uint32_t>(get_next_hop_router_direction(route_info.dst_mesh_id, route_info.dst_chip_id));
        ASSERT(first_hop < static_cast<uint32_t>(eth_chan_directions::COUNT));
        ASSERT(dir_to_slot[first_hop] != kNoConnection);
        peer_connections[slot] = &fabric_connections.get(dir_to_slot[first_hop]).sender;
        ++slot;
    }

    // A global semaphore is an allocator buffer named by a runtime address, so it cannot be
    // wrapped: the Semaphore object resolves its address from a program semaphore id, and its
    // decrement is a local read-modify-write, which is the race the subtraction below avoids.
    auto* arrival_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem_addr);
    // The peer's copy of this semaphore. Same logical core there — one program shape
    // per chip — so the local NOC address is the right one to name it by.
    const uint64_t peer_arrival_noc_addr = get_noc_addr(arrival_sem_addr);

    // Nothing can go out until every statistics core has parked its rows. Only those
    // cores signal; the fold cores beyond them have nothing to contribute here and
    // wait to be released like the rest.
    ready_sem.wait(num_stat_cores);
    ready_sem.set(0);

    // A rank's plane is a contiguous run of pages, and the two planes it owns are
    // adjacent, so the whole of what it sends is one run. It is staged in chunks and
    // sent behind one read barrier per chunk rather than one per page: a barrier per
    // page puts the exchange on the DRAM latency ladder, one round trip deep, at exactly
    // the moment every fold core is prefetching against the same DRAM.
    constexpr uint32_t pages_per_plane = stats_pages_per_plane(Ht, stat_tile_bytes);
    constexpr uint32_t kPlanePages = kStatsPerRow * pages_per_plane;
    const uint32_t first_page = kStatsPerRow * my_rank * pages_per_plane;

    for (uint32_t base = 0; base < kPlanePages; base += stage_tiles) {
        const uint32_t remaining = kPlanePages - base;
        const uint32_t chunk = remaining < stage_tiles ? remaining : stage_tiles;

        stage_buf.reserve_back(chunk);
        const uint32_t stage_base = stage_buf.get_write_ptr();
        for (uint32_t t = 0; t < chunk; ++t) {
            noc.async_read(
                stats_accessor,
                stage_buf,
                stat_tile_bytes,
                {.page_id = first_page + base + t},
                {.offset_bytes = t * stat_tile_bytes});
        }
        noc.async_read_barrier();

        for (uint32_t t = 0; t < chunk; ++t) {
            for (uint32_t slot = 0; slot < kPeers; ++slot) {
                tt::tt_fabric::linear::to_noc_unicast_write(
                    stat_tile_bytes, payload_headers[slot], first_page + base + t, stats_accessor);
                perform_payload_send<true, true>(
                    *peer_connections[slot], stage_base + t * stat_tile_bytes, stat_tile_bytes, payload_headers[slot]);
            }
        }
    }

    for (uint32_t slot = 0; slot < kPeers; ++slot) {
        inc_headers[slot]->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_arrival_noc_addr, 1});
        peer_connections[slot]->wait_for_empty_write_slot();
        peer_connections[slot]->send_payload_flush_blocking_from_address(
            (uint32_t)inc_headers[slot], packet_header_size_bytes);
    }

    // Consumed by subtraction, not by a store. A peer that is already sending for the
    // next read site can increment between the load and the store of a plain reset, and
    // that increment is then lost: the next wait is one arrival short of a count no one
    // will send again. Subtracting atomically carries any early arrival forward instead.
    noc_semaphore_wait_min(arrival_sem_ptr, kPeers);
    noc_semaphore_inc(get_noc_addr(arrival_sem_addr), uint32_t{0} - kPeers);

    fabric_connections.close();

    // Release every fold core, whether or not it produced statistics. The fold takes
    // most of the grid, so this is a multicast per rectangle rather than an increment
    // per core: a hundred serialized atomics would put the whole fold behind a wake-up
    // ramp longer than the exchange it is waiting on.
    for (uint32_t r = 0; r < num_release_ranges; ++r) {
        const uint32_t base = kReleaseRangeArgIdx + kWordsPerReleaseRange * r;
        done_sem.inc_multicast(
            noc,
            get_arg_val<uint32_t>(base),
            get_arg_val<uint32_t>(base + 1),
            get_arg_val<uint32_t>(base + 2),
            get_arg_val<uint32_t>(base + 3),
            1,
            get_arg_val<uint32_t>(base + 4));
    }
    noc.async_atomic_barrier();
}
