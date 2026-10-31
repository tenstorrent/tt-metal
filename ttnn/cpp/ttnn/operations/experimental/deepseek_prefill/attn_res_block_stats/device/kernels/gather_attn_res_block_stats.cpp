// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The gather across the tensor-parallel axis, on the one core per chip that holds a fabric
// connection. A link admits a single worker, so every row's statistics are funnelled here.
//
// What crosses is this rank's tile column of the gathered statistics, written into the same
// column on each peer; peers fill this chip's other columns symmetrically, so the gathered
// tensor is complete once every peer's arrivals have landed. It goes out as it is parked: a
// row's leader parks the row's candidates in order and counts them here, and whatever is
// parked is sent, so the rows of cores that finish early are on the wire while the slowest
// are still reducing.
//
// A route is a multicast along one direction of the axis, reaching every peer that lies
// that way, so a page crosses each link once however many peers are beyond it. A core opens
// one connection per direction, so each link has a gather core of its own, and a
// candidate's pages alternate between them.
//
// Ordering: each route's arrival increment follows every payload on that route, and the
// fabric preserves order per connection, so a peer that has counted an arrival from each of
// a sender's routes that reach it has every page that sender sent it.
//
// The arrival semaphore is a global semaphore, not a program-local one: program-local
// semaphores are re-initialized at every launch, which would race a peer one program ahead.
// It is consumed here by subtracting what this launch waited for.

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
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/attn_res_block_stats_items.hpp"

void kernel_main() {
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t num_candidates = get_compile_time_arg_val(2);
    // The per-row counts of parked candidates, from this semaphore on.
    constexpr uint32_t row_sem_id = get_compile_time_arg_val(3);
    constexpr AttnResProgressFields row_fields{get_compile_time_arg_val(4)};
    // The arrivals this core waits for: one from the same link's gather core of each peer.
    constexpr uint32_t num_arrivals = get_compile_time_arg_val(5);
    constexpr uint32_t num_links = get_compile_time_arg_val(6);
    constexpr auto gathered_args = TensorAccessorArgs<7>();

    const auto gathered_addr = get_arg_val<uint32_t>(0);
    const auto my_rank = get_arg_val<uint32_t>(1);
    const auto arrival_sem_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_headers = 8;
    constexpr uint32_t cb_stage = 9;
    constexpr uint32_t tile_bytes = get_tile_size(cb_stage);
    // A route each way along the axis.
    constexpr uint32_t kMaxRoutes = 2;
    constexpr uint32_t kWordsPerRoute = ccl_routing_utils::num_line_multicast_args;
    constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    Noc noc;
    DataflowBuffer stage_buf(cb_stage);
    CircularBuffer cb_headers_obj(cb_headers);

    // Page size is given explicitly — the accessor's compile-time value can be stale on a
    // program-cache hit — and it doubles as the fabric payload size.
    auto gathered = TensorAccessor(gathered_args, gathered_addr, tile_bytes);

    // This core's link, the routes' connections, one per route in route order, then each
    // route's multicast.
    const auto link = get_arg_val<uint32_t>(3);
    size_t arg_idx = 4;
    const auto num_routes = get_arg_val<uint32_t>(arg_idx++);
    ASSERT(num_routes <= kMaxRoutes);
    auto fabric_connections = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args<
        tt::tt_fabric::RoutingPlaneConnectionManager::BUILD_AND_OPEN_CONNECTION>(arg_idx, num_routes);

    cb_headers_obj.reserve_back(2 * num_routes);
    const uint32_t header_base = cb_headers_obj.get_write_ptr();
    cb_headers_obj.push_back(2 * num_routes);

    volatile PACKET_HEADER_TYPE* payload_headers[kMaxRoutes];
    volatile PACKET_HEADER_TYPE* inc_headers[kMaxRoutes];
    for (uint32_t r = 0; r < num_routes; ++r) {
        payload_headers[r] =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_base + (2 * r) * packet_header_size_bytes);
        inc_headers[r] =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_base + (2 * r + 1) * packet_header_size_bytes);
        const size_t base = arg_idx + kWordsPerRoute * r;
        const ccl_routing_utils::line_multicast_route_info_t route_info{
            .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(base)),
            .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(base + 1)),
            .e_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(base + 2)),
            .w_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(base + 3)),
            .n_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(base + 4)),
            .s_num_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(base + 5))};
        ccl_routing_utils::fabric_set_line_multicast_route(payload_headers[r], route_info);
        ccl_routing_utils::fabric_set_line_multicast_route(inc_headers[r], route_info);
    }

    // Whatever is parked and not yet sent, in batches: read back behind one barrier, then
    // multicast both ways at the same page. A page is this core's when it falls to its link.
    constexpr uint32_t kBatch = 16;
    stage_buf.reserve_back(kBatch);
    const uint32_t stage_base = stage_buf.get_write_ptr();
    uint32_t sent[Ht] = {};
    uint32_t pending = 0;
    for (uint32_t page = 0; page < num_candidates * Ht; ++page) {
        pending += page % num_links == link;
    }
    uint32_t batch_pages[kBatch];
    while (pending > 0) {
        uint32_t batch = 0;
        for (uint32_t g = 0; g < Ht && batch < kBatch; ++g) {
            const uint32_t parked = row_fields.count(
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(row_sem_id + row_fields.word(g))), g);
            for (; sent[g] < parked && batch < kBatch; ++sent[g]) {
                const uint32_t index = sent[g] * Ht + g;
                if (index % num_links == link) {
                    batch_pages[batch++] = index * ring_size + my_rank;
                }
            }
        }
        if (batch == 0) {
            invalidate_l1_cache();
            continue;
        }
        for (uint32_t i = 0; i < batch; ++i) {
            noc.async_read(
                gathered, stage_buf, tile_bytes, {.page_id = batch_pages[i]}, {.offset_bytes = i * tile_bytes});
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < batch; ++i) {
            for (uint32_t r = 0; r < num_routes; ++r) {
                tt::tt_fabric::linear::to_noc_unicast_write(tile_bytes, payload_headers[r], batch_pages[i], gathered);
                perform_payload_send<true, true>(
                    fabric_connections.get(r).sender, stage_base + i * tile_bytes, tile_bytes, payload_headers[r]);
            }
        }
        pending -= batch;
    }

    // The peer's copy of the arrival semaphore: same logical core there, one program shape
    // per chip, so the local NOC address names it.
    const uint64_t peer_arrival_noc_addr = get_noc_addr(arrival_sem_addr);
    for (uint32_t r = 0; r < num_routes; ++r) {
        auto& sender = fabric_connections.get(r).sender;
        inc_headers[r]->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_arrival_noc_addr, 1});
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_blocking_from_address((uint32_t)inc_headers[r], packet_header_size_bytes);
    }

    // Consumed by subtraction, not by a store: a peer already sending for the next call can
    // increment between the load and the store of a plain reset, and that arrival is lost.
    auto* arrival_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem_addr);
    noc_semaphore_wait_min(arrival_sem_ptr, num_arrivals);
    noc_semaphore_inc(get_noc_addr(arrival_sem_addr), uint32_t{0} - num_arrivals);

    fabric_connections.close();
}
