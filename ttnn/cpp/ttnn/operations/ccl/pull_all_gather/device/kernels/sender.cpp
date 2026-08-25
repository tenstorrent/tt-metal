// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/fabric_dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/fabric_pull.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

constexpr uint32_t txns_per_device = get_arg(args::txns_per_device);
constexpr uint32_t bytes_per_dma_txn = get_arg(args::bytes_per_dma_txn);
constexpr uint32_t in_shard_bytes = get_arg(args::in_shard_bytes);
constexpr uint32_t out_shard_bytes = get_arg(args::out_shard_bytes);
constexpr uint32_t out_shard_tiles = get_arg(args::out_shard_tiles);
constexpr uint32_t tiles_per_device = get_arg(args::tiles_per_device);
constexpr uint32_t block_bytes = get_arg(args::block_bytes);

// The block is a whole number of output shards, which is what makes the chunk
// walk identical on every device.
static_assert(block_bytes % out_shard_bytes == 0);

// Bytes this chunk may carry, starting `cursor` bytes into the device block.
// Three-way minimum: the packet cap, what is left of the current input shard
// (whose last one per block may be ragged), and what is left of the current
// output shard (uniform, because the block is a whole number of them). Same
// recurrence the producer runs, so entry N here is the chunk it put in entry N.
constexpr uint32_t txn_bytes_at(uint32_t cursor) {
    const uint32_t in_end = std::min((cursor / in_shard_bytes + 1) * in_shard_bytes, block_bytes);
    const uint32_t out_left = out_shard_bytes - (cursor % out_shard_bytes);
    return std::min(bytes_per_dma_txn, std::min(in_end - cursor, out_left));
}

// Reads the route arg block and builds the multicast route args.
//
// The host has already decided how many routes this device needs and what goes
// in each, so this only unpacks. One block per route:
//
//   h[0], h[1], h[2], h[3], port, dst_dev_id, dst_mesh_id
//
// `h` is indexed by eth_chan_directions, so the host names the cardinal
// directions and the kernel never does. There are always fabric_max_routes
// blocks; the ones past num_routes are zero and are not read.
template <tt::tt_fabric::Topology topology>
FabricMcastRouteArgs<topology> build_mcast_route(std::size_t& arg_idx, bool include_self) {
    FabricMcastRouteArgs<topology> route{};

    if constexpr (!tt::tt_fabric::is_forwarding_topology(topology)) {
        // One word, and it is the whole route: which fabric nodes to deliver
        // to. No hop counts, no directions, no anchors -- every peer is one hop.
        route.routes[0].peer_mask = get_arg_val<uint32_t>(arg_idx++);
        route.num_routes = 1;
        route.include_self = include_self;
        return route;
    }

    const uint32_t num_routes = get_arg_val<uint32_t>(arg_idx++);
    ASSERT(num_routes > 0 && num_routes <= fabric_max_routes<topology>);

    auto next = [&]() { return static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++)); };

    for (uint32_t r = 0; r < num_routes; ++r) {
        const uint8_t h0 = next(), h1 = next(), h2 = next(), h3 = next();
        const uint8_t port = next();
        const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
        const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));

        // make_fabric_range() collapses the dimensional difference: a
        // MeshMcastRange in 2D, the bare hop count in 1D where exactly one slot
        // is nonzero. Only the destination fields differ.
        if constexpr (tt::tt_fabric::is_2D_topology(topology)) {
            route.routes[r] = {make_fabric_range(h0, h1, h2, h3), dst_dev_id, dst_mesh_id, port};
        } else {
            route.routes[r] = {make_fabric_range(h0, h1, h2, h3), port};
        }
    }
    // Skip the zeroed tail so the caller's arg_idx lands on the next field.
    arg_idx += (fabric_max_routes<topology> - num_routes) * 7;

    route.num_routes = num_routes;
    route.include_self = include_self;
    return route;
}

void kernel_main() {
    //   device_idx, num_peers | num_routes | kMaxRoutes * 7 route words | sem addr, x, y
    std::size_t runtime_arg_index = 0;
    const uint32_t device_idx = get_arg_val<uint32_t>(runtime_arg_index++);
    const uint32_t num_peers = get_arg_val<uint32_t>(runtime_arg_index++);

    // Chip multicast excludes the source chip in every direction. An all-gather
    // needs its own block in its own replica too, hence include_self.
    const auto route = build_mcast_route<topology>(runtime_arg_index, /*include_self=*/true);

    const uint32_t barrier_sem_address = get_arg_val<uint32_t>(runtime_arg_index++);
    const uint8_t barrier_sem_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(runtime_arg_index++));
    const uint8_t barrier_sem_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(runtime_arg_index++));

    // num_peers is the barrier fan-in, not M. M is the route count: the router
    // issues one packet per direction and the chain forwards it.
    ASSERT(num_peers > 0);

    // One request set per distinct packet state: the payload multicast and the
    // header-only completion atomic. A set holds one slot per route, since each
    // route needs its own packet header.
    using RequestSet = FabricPullRequestSet<PACKET_HEADER_TYPE, fabric_max_routes<topology>>;
    Scratchpad<volatile RequestSet> requests(scratch::fabric_requests);
    auto* data_request = requests.local_mem() + 0;
    auto* barrier_request = requests.local_mem() + 1;

    FabricDataflowBuffer payload(dfb::payload);  // counters come from reserved L1
    const auto output_tensor = TensorAccessor(tensor::output_tensor);
    Noc noc;

    // Nothing is opened. Link parameters come from the connection table
    // device-init populated, looked up by each route's direction.
    Fabric fabric;

    // Fills one slot per route, each with its own range and anchor, and records
    // include_self and the outgoing port.
    //
    // The local copy reuses the same remote_noc_addr as a posted NoC write on
    // NOC_UNICAST_WRITE_VC + 1, and is not counted in M. That works only
    // because the output is fully replicated with an identical layout on every
    // device, so a page id names the same (core, offset) locally and on every
    // peer.
    fabric.set_async_write_multicast_state(data_request, route);

    // Row-dim gather: this device owns one contiguous page range of the output,
    // in local tile order, so output_page(t) = base_page + t.
    const uint32_t base_page = device_idx * tiles_per_device;

    // Same walk the producer ran, so entry N here is the chunk the producer put
    // in entry N. The cursor is bytes into this device's block.
    uint32_t cursor = 0;
    for (uint32_t entry = 0; entry < txns_per_device; ++entry) {
        const uint32_t payload_bytes = txn_bytes_at(cursor);

        // A chunk is a byte range inside one output shard: page at that shard's
        // start, plus an offset.
        const uint32_t out_shard = cursor / out_shard_bytes;
        const uint32_t output_page = base_page + out_shard * out_shard_tiles;
        const uint32_t offset_bytes = cursor - out_shard * out_shard_bytes;

        // A shard's pages are one contiguous address run, so page + offset is a
        // single address covering the whole chunk. The accessor resolves the
        // destination core, which changes every shard.
        const uint64_t output_noc_address =
            tt::tt_fabric::addrgen_detail::get_noc_address(output_tensor, output_page, offset_bytes);

        // One call, every route. Internally: wait_for_txn_id ->
        // wait_for_next_issue -> get_read_ptr -> prepare_transaction(M) ->
        // publish each claimed slot under the same txn id -> local posted write
        // (include_self) -> commit_transaction, which advances the read pointer
        // ONCE for the whole multicast -> try_complete_front_transaction
        fabric.async_write_multicast_with_state(data_request, payload, output_noc_address, payload_bytes);

        cursor += payload_bytes;
    }

    // Completion atomic. Header-only, so it consumes no transaction ID. This is
    // just another request pushed into the DE send queue, behind every data
    // request; the DE drains that queue in order. Nothing here waits.
    //
    // Flush is load-bearing: the semaphore lives on the mirror worker core,
    // while the payload went to the output tensor's shard cores. Waiting on the
    // semaphore is only meaningful because Flush makes the *receiving* side
    // drain its preceding NoC writes before the increment lands.
    fabric.set_atomic_inc_multicast_state(barrier_request, route);
    fabric.atomic_inc_multicast_with_state(
        barrier_request,
        safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem_address, noc.get_noc_id()),
        /*value=*/1,
        // The semaphore is on the mirror worker core while the payload went to
        // the shard cores, so without this a peer could see the increment
        // before its data.
        /*flush=*/true);

    // No explicit teardown: ~FabricDataflowBuffer() drains our transaction
    // counters at scope exit, so no SWQ still references a payload page when the
    // kernel returns. The source-local posted writes were already flushed inside
    // each async_write_multicast_with_state.

    // One increment per peer, plus our own: the route carries include_self, so
    // atomic_inc_multicast_with_state() bumps this device's semaphore locally
    // as well. Fan-in is therefore the peer count plus one.
    const uint32_t barrier_arrivals = num_peers + 1;
    auto* barrier_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_address);
    noc_semaphore_wait_min(barrier_sem, barrier_arrivals);
    // Decrement rather than reset: increments from other phases must survive.
    noc_semaphore_inc(
        safe_get_noc_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem_address),
        static_cast<uint32_t>(-static_cast<int32_t>(barrier_arrivals)));

    noc.async_write_barrier();
}
