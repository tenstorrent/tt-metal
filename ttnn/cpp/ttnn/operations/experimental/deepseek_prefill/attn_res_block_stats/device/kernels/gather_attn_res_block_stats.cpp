// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The gather across the tensor-parallel axis, on the gather cores: one per fabric link,
// since a link admits a single worker.
//
// What crosses is this rank's tile column of the gathered statistics, written into the same
// column on each peer; peers fill this chip's other columns symmetrically, so the gathered
// tensor is complete once every peer's arrivals have landed. It goes out as it is parked: a
// row's leader parks the row's candidates in order and counts them here, and whatever is
// parked is sent, so early rows are on the wire while the slowest are still reducing.
//
// A route is a multicast along one direction of the axis, reaching every peer that lies
// that way, so a page crosses each link once however many peers are beyond it. A core opens
// one connection per direction; pages alternate between the links' cores.
//
// Ordering: each route's arrival increment follows every payload on that route, and the
// fabric preserves order per connection, so a peer that has counted an arrival from each of
// a sender's routes that reach it has every page that sender sent it.
//
// Readiness: the gathered tensor is kept across calls, so a peer's write must not land while
// this chip still reads the previous call's statistics out of it. Each gather core tells its
// peers it has started before it sends anything, and sends nothing until every peer has told
// it the same. A chip runs its programs in order, so a peer that has started this op is done
// with everything before it, the previous call's consumer included.
//
// Both counts share the caller's semaphore, readiness in the low half and arrivals in the
// high half. It is a global semaphore, not a program-local one: program-local semaphores are
// re-initialized at every launch, which would race a peer one program ahead. Each half is
// consumed by subtracting what this launch waited for. A peer can run at most one call
// ahead (it waits on this chip's arrival), so at most its next readiness is early, and that
// lands after this launch has consumed its own.

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
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    constexpr uint32_t num_candidates = get_named_compile_time_arg_val("num_candidates");
    // The per-row counts of parked candidates, from this semaphore on, each a step of
    // candidates at a time.
    constexpr uint32_t row_sem_id = get_named_compile_time_arg_val("row_sem_id");
    constexpr AttnResProgressFields row_fields{get_named_compile_time_arg_val("row_field_width")};
    constexpr uint32_t row_step = get_named_compile_time_arg_val("row_step");
    // What this core has sent of each row, a candidate at a time: a batch can end mid-step.
    constexpr AttnResProgressFields sent_fields{get_named_compile_time_arg_val("field_width")};
    // The arrivals this core waits for, and as many readinesses: one from the same link's
    // gather core of each peer.
    constexpr uint32_t num_arrivals = ring_size - 1;
    constexpr uint32_t kReady = 1;
    constexpr uint32_t kArrival = 1u << 16;
    constexpr uint32_t num_links = get_named_compile_time_arg_val("num_gather_cores");
    // Pages staged behind one read barrier; the staging buffer holds exactly this many.
    constexpr uint32_t kBatch = get_named_compile_time_arg_val("gather_batch");
    constexpr auto gathered_args = TensorAccessorArgs<0>();

    // The arrival semaphore is the caller's, common to every gather core and patched on a
    // program-cache hit.
    const auto arrival_sem_addr = get_common_arg_val<uint32_t>(0);
    const auto gathered_addr = get_arg_val<uint32_t>(0);
    const auto my_rank = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_headers = get_named_compile_time_arg_val("cb_headers");
    constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
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
    const auto link = get_arg_val<uint32_t>(2);
    size_t arg_idx = 3;
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

    // The peer's copy of the semaphore: same logical core there, one program shape per chip,
    // so the local NOC address names it.
    const uint64_t peer_sem_noc_addr = get_noc_addr(arrival_sem_addr);
    auto* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem_addr);
    const auto signal_peers = [&](uint32_t increment) {
        for (uint32_t r = 0; r < num_routes; ++r) {
            auto& sender = fabric_connections.get(r).sender;
            inc_headers[r]->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{peer_sem_noc_addr, increment});
            sender.wait_for_empty_write_slot();
            sender.send_payload_flush_blocking_from_address((uint32_t)inc_headers[r], packet_header_size_bytes);
        }
    };
    // Consumed by subtraction, not by a store: a peer already signalling for the next call can
    // increment between the load and the store of a plain reset, and that signal is lost.
    const auto consume = [&](uint32_t unit) {
        while ((*sem_ptr / unit & 0xFFFF) < num_arrivals) {
            invalidate_l1_cache();
        }
        noc_semaphore_inc(get_noc_addr(arrival_sem_addr), uint32_t{0} - num_arrivals * unit);
        noc_async_atomic_barrier();
    };
    // Told now, waited on only before the first send, so the round trip overlaps the reduction.
    signal_peers(kReady);
    bool peers_ready = false;

    // Whatever is parked and not yet sent, in batches: read back behind one barrier, then
    // multicast both ways at the same page. A page is this core's when it falls to its link.
    stage_buf.reserve_back(kBatch);
    const uint32_t stage_base = stage_buf.get_write_ptr();
    // Each row's count of candidates sent, a field of num_candidates' bit width (so it cannot
    // carry). The host bounds Ht, so the stack footprint stays small.
    constexpr uint32_t num_row_words = sent_fields.word(Ht - 1) + 1;
    uint32_t sent_words[num_row_words] = {};
    // This core's pages: those of the num_candidates * Ht that fall to its link.
    constexpr uint32_t num_pages = num_candidates * Ht;
    uint32_t pending = num_pages / num_links + (link < num_pages % num_links ? 1 : 0);
    uint32_t batch_pages[kBatch];
    while (pending > 0) {
        uint32_t batch = 0;
        for (uint32_t g = 0; g < Ht && batch < kBatch; ++g) {
            const uint32_t w = row_fields.word(g);
            const uint32_t steps =
                row_fields.count(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(row_sem_id + w)), g);
            const uint32_t parked = steps * row_step < num_candidates ? steps * row_step : num_candidates;
            const uint32_t already_sent = sent_fields.count(sent_words[sent_fields.word(g)], g);
            uint32_t sent = already_sent;
            for (; sent < parked && batch < kBatch; ++sent) {
                const uint32_t index = sent * Ht + g;
                if (index % num_links == link) {
                    batch_pages[batch++] = index * ring_size + my_rank;
                }
            }
            sent_words[sent_fields.word(g)] += (sent - already_sent) * sent_fields.increment(g);
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
        if (!peers_ready) {
            consume(kReady);
            peers_ready = true;
        }
        for (uint32_t i = 0; i < batch; ++i) {
            for (uint32_t r = 0; r < num_routes; ++r) {
                tt::tt_fabric::linear::to_noc_unicast_write(tile_bytes, payload_headers[r], batch_pages[i], gathered);
                perform_payload_send<true, true>(
                    fabric_connections.get(r).sender, stage_base + i * tile_bytes, tile_bytes, payload_headers[r]);
            }
        }
        pending -= batch;
    }

    // A core with no page to send still consumes its peers' readiness, so the count is back
    // at zero for the next call.
    if (!peers_ready) {
        consume(kReady);
    }
    signal_peers(kArrival);
    consume(kArrival);

    fabric_connections.close();
}
