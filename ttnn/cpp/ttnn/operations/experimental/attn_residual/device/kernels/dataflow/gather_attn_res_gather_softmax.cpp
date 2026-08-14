// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The collective, on the one core per chip that holds a fabric connection. A link
// admits a single worker — every sender core in a fabric op takes its own link index
// — so the exchange is funnelled here rather than run from each worker.
//
// What crosses is this rank's whole statistics plane: two tiles per token row-tile,
// written into the slot the peers' fold cores read for this rank. Peers write
// their own slots here symmetrically, so nothing has to be reordered on arrival and
// the reduction is the slot-wise sum the fold already performs.
//
// Ordering: a peer's arrival increment is sent after every payload bound for that
// peer, on the same connection, and the fabric preserves order per connection. A
// non-zero arrival count therefore implies the payload behind it has landed.
//
// The arrival semaphore is a global semaphore, not a program-local one. Program-local
// semaphores are re-initialized at every launch, which would race a peer that is one
// program ahead; a global semaphore is written once and reset here, in kernel, after
// the wait.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

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

    // Page size is given explicitly — the accessor's compile-time value can be stale
    // on a program-cache hit — and it doubles as the fabric payload size.
    auto stats_accessor = TensorAccessor(stats_args, stats_addr, stat_tile_bytes);

    size_t fabric_arg_idx = kReleaseRangeArgIdx + kWordsPerReleaseRange * num_release_ranges;
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
            fabric_arg_idx);

    // A peer's route follows the connection args, two words per peer in the order the
    // loop below takes them. The words mean different things per fabric — a hop count on
    // 1D, the peer's node on 2D — so the host encodes them and the header type here
    // decides how to read them back.
    const size_t route_arg_idx = fabric_arg_idx;

    // One payload header and one increment header per peer: a header carries its
    // peer's route for the whole kernel, and peers in the same direction still differ by
    // route, so headers cannot be shared even within a direction.
    cb_reserve_back(cb_id_headers, kStatsPerRow * kPeers);
    const uint32_t header_base = get_write_ptr(cb_id_headers);
    cb_push_back(cb_id_headers, kStatsPerRow * kPeers);

    volatile PACKET_HEADER_TYPE* payload_headers[kPeers];
    volatile PACKET_HEADER_TYPE* inc_headers[kPeers];
    tt::tt_fabric::WorkerToFabricEdmSender* peer_connections[kPeers];

    for (uint32_t p = 0, slot = 0; p < ring_size; ++p) {
        if (p == my_rank) {
            continue;
        }
        const bool forward = p > my_rank;

        payload_headers[slot] =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_base + (2 * slot) * packet_header_size_bytes);
        inc_headers[slot] =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_base + (2 * slot + 1) * packet_header_size_bytes);

        const ccl_routing_utils::line_unicast_route_info_t route_info{
            .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_arg_idx + 2 * slot)),
            .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(route_arg_idx + 2 * slot + 1))};
        ccl_routing_utils::fabric_set_line_unicast_route(payload_headers[slot], route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(inc_headers[slot], route_info);

        peer_connections[slot] =
            forward ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection();
        ++slot;
    }

    auto* ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id));
    auto* arrival_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem_addr);
    // The peer's copy of this semaphore. Same logical core there — one program shape
    // per chip — so the local NOC address is the right one to name it by.
    const uint64_t peer_arrival_noc_addr = get_noc_addr(arrival_sem_addr);

    // Nothing can go out until every statistics core has parked its rows. Only those
    // cores signal; the fold cores beyond them have nothing to contribute here and
    // wait to be released like the rest.
    noc_semaphore_wait(ready_sem_ptr, num_stat_cores);
    noc_semaphore_set(ready_sem_ptr, 0);

    // A rank's plane is a contiguous run of pages — every row's sum of squares, then
    // every row's dots — so it is staged in chunks and sent behind one read barrier per
    // chunk. Barriering per tile instead makes the exchange a ladder of DRAM round
    // trips, one deep, which is the whole of its cost: the payload is well inside a
    // single link's bandwidth, and every fold core is prefetching against the same DRAM
    // while this runs.
    constexpr uint32_t kPlaneTiles = kStatsPerRow * Ht;
    const uint32_t first_page = kStatsPerRow * my_rank * Ht;

    for (uint32_t base = 0; base < kPlaneTiles; base += stage_tiles) {
        const uint32_t remaining = kPlaneTiles - base;
        const uint32_t chunk = remaining < stage_tiles ? remaining : stage_tiles;

        stage_buf.reserve_back(chunk);
        const uint32_t stage_base = get_write_ptr(cb_id_stage);
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

    noc_semaphore_wait(arrival_sem_ptr, kPeers);
    noc_semaphore_set(arrival_sem_ptr, 0);

    fabric_connection.close();

    // Release every fold core, whether or not it produced statistics. The fold takes
    // most of the grid, so this is a multicast per rectangle rather than an increment
    // per core: a hundred serialized atomics would put the whole fold behind a wake-up
    // ramp longer than the exchange it is waiting on.
    const uint32_t done_sem_addr = get_semaphore(done_sem_id);
    for (uint32_t r = 0; r < num_release_ranges; ++r) {
        const uint32_t base = kReleaseRangeArgIdx + kWordsPerReleaseRange * r;
        const uint64_t range_noc_addr = get_noc_multicast_addr(
            get_arg_val<uint32_t>(base),
            get_arg_val<uint32_t>(base + 1),
            get_arg_val<uint32_t>(base + 2),
            get_arg_val<uint32_t>(base + 3),
            done_sem_addr);
        noc_semaphore_inc_multicast(range_noc_addr, 1, get_arg_val<uint32_t>(base + 4));
    }
    noc_async_atomic_barrier();
}
