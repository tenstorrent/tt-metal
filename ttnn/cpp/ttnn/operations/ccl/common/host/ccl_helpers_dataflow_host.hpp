// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ccl_helpers_dataflow_host.hpp
 * @brief Host companion to ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp.
 *
 * The host-side half of the multi-device CCL (fabric) dataflow helper: 1-D route
 * computation, fabric packet framing, the fabric-connection runtime-arg append (in
 * the exact layout the kernel-side FabricStreamSender expects), and GlobalSemaphore
 * allocation + the cross-device Synchronize barrier. Mirrors the single-device
 * dataflow-helper precedent (#45698) at the multi-device tier.
 *
 * Header-only (all functions `inline`): the only consumer today is a program factory
 * that already pulls these dependencies; splitting into a compiled .cpp + CMake
 * target is trivial later if the inline footprint grows.
 *
 * @par Authoring a CCL dataflow op — which helper for each step.
 *   These are host-side building blocks called from an op's PROGRAM FACTORY (point_to_point and
 *   all_gather are the consumers today). The pure-computation entries (@c ccl_packet_dims,
 *   @c ccl_dm_route) and @c make_ccl_semaphore are also Python-bound under @c ttnn._ttnn.fabric,
 *   because generated ops assemble their MeshProgramDescriptor from Python host code. A typical
 *   fabric dataflow op builds its writer/reader args in this order:
 *     1. @c ccl_packet_dims(dtype, page_size, num_pages, alignment) — frame pages into fabric
 *        packets (packet size / pages-per-packet / segments); owns the bf16 case + the page regimes.
 *     2. @c ccl_dm_route(mesh_device, sender, receiver, topology) — compute the 1-D {num_hops,
 *        is_forward, neighbor} for a single point-to-point route (owns the fwd/bwd sign reversal +
 *        the ring-vs-line shorter-path choice). The bidirectional (all_gather) case instead pairs the
 *        existing ring-route configuration with @c append_ccl_line_route_ct_args (step 4).
 *     3. @c build_ccl_fabric_rt_args(...) — the fabric-CONNECTION runtime-arg block in the exact
 *        layout the kernel's FabricStreamSender opens (has_forward/has_backward + the connection
 *        block). Place it FIRST in the kernel's runtime args; the kernel consumes it with a cursor
 *        from 0, so neither side hardcodes an offset. Generic across fabric paths.
 *     4. @c append_ccl_line_route_ct_args(...) — bidirectional/all_gather only: append the four
 *        line-ROUTE compile-time args (fwd/bwd × unicast/multicast) in the order the writer reads
 *        them back. Distinct from step 3 — this is the ROUTE, not the connection.
 *     5. @c make_ccl_semaphore(mesh_device) — allocate the cross-device GlobalSemaphore + run the
 *        cache-miss Synchronize barrier; keep the returned handle alive for the workload's lifetime.
 *
 * @par Host helper <-> kernel-side consumer pairing.
 *   | host (this header / ttnn._ttnn.fabric)   | kernel side consumes it via                       |
 *   |-------------------------------------------|---------------------------------------------------|
 *   | @c build_ccl_fabric_rt_args               | @c FabricStreamSender<>(arg_idx, ...) cursor       |
 *   | @c append_ccl_line_route_ct_args          | @c ccl_routing_utils::get_line_*_route_info_from_args |
 *   | @c ccl_packet_dims                        | the writer's packet loop (pages_per_packet, segments) |
 *   | @c ccl_dm_route                           | plain scalars the op passes as its own rt args     |
 *   | @c make_ccl_semaphore                     | @c AtomicIncChannel targets / semaphore waits      |
 *
 * @par EXAMPLE — minimal unidirectional writer (factory side, then kernel side).
 * @code
 *   // ---- program factory ----
 *   auto pkt   = ccl_packet_dims(dtype, page_size, num_pages, l1_alignment);
 *   auto route = ccl_dm_route(mesh_device, my_coord, dst_coord, topology);   // hops/direction/neighbor
 *   auto sem   = make_ccl_semaphore(mesh_device);                             // keep alive with the workload
 *
 *   std::vector<uint32_t> rt_args = build_ccl_fabric_rt_args(                 // fabric block FIRST
 *       my_fabric_id, route.neighbor_id, link_idx, desc, core, route.is_forward);
 *   rt_args.insert(rt_args.end(), {route.num_hops, sem.address(), pkt.pages_per_packet, ...});  // op args AFTER
 *
 *   // ---- kernel (writer) ----
 *   size_t arg_idx = 0;
 *   const bool is_forward = get_arg_val<uint32_t>(0);        // peek: leading has_forward flag
 *   FabricStreamSender<> sender(arg_idx, is_forward, l1_alignment);  // cursor eats the fabric block
 *   const uint32_t num_hops = get_arg_val<uint32_t>(arg_idx++);      // op args resume at the cursor
 * @endcode
 *   Full worked references: the committed example packages under ttnn/ttnn/operations/
 *   (point_to_point is the smallest) and the migrated reduce_scatter_minimal_async factory.
 */

#include <bit>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <tt_stl/assert.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/global_semaphore.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>

namespace ttnn::ccl::dataflow {

using tt::tt_fabric::FabricNodeId;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::DataType;
using tt::tt_metal::GlobalSemaphore;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

// ===========================================================================
// H2 — fabric packet framing
// ===========================================================================

struct PacketDims {
    uint32_t packet_size_bytes;
    uint32_t pages_per_packet;
    uint32_t page_segments;
    uint32_t total_packets;
};

/**
 * @brief Frame `num_pages` pages of `page_size_bytes` into fabric packets.
 *
 * Owns the bfloat16 std::bit_floor special case and the two regimes:
 *   - aligned page <= max packet: pack N pages per packet;
 *   - aligned page  > max packet: split each page into segments.
 */
inline PacketDims ccl_packet_dims(DataType dtype, uint32_t page_size_bytes, uint32_t num_pages, uint32_t alignment) {
    const uint32_t fabric_max_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    const uint32_t max_packet_size_bytes =
        dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    const uint32_t aligned_page_size_bytes = tt::round_up(page_size_bytes, alignment);

    uint32_t num_page_segments, max_num_pages_per_packet, packet_size_bytes, total_packets;
    if (aligned_page_size_bytes <= max_packet_size_bytes) {
        num_page_segments = 1;
        max_num_pages_per_packet = std::min(max_packet_size_bytes / aligned_page_size_bytes, num_pages);
        packet_size_bytes = aligned_page_size_bytes * max_num_pages_per_packet;
        total_packets = tt::div_up(num_pages, max_num_pages_per_packet);
    } else {
        max_num_pages_per_packet = 1;
        num_page_segments = tt::div_up(aligned_page_size_bytes, max_packet_size_bytes);
        packet_size_bytes = max_packet_size_bytes;
        total_packets = num_page_segments * num_pages;
    }

    return {packet_size_bytes, max_num_pages_per_packet, num_page_segments, total_packets};
}

// ===========================================================================
// H1 — 1-D unicast route from two mesh coords + topology
// ===========================================================================

struct DmRoute {
    uint32_t num_hops;
    bool is_forward;
    FabricNodeId neighbor_id;
};

namespace detail {
inline auto fabric_1d_routing_vector(const MeshCoordinate& sender_coord, const MeshCoordinate& receiver_coord) {
    // transmit along row
    if (sender_coord[0] == receiver_coord[0]) {
        constexpr auto dim = 1;
        const int hops = receiver_coord[dim] - sender_coord[dim];
        bool is_fwd = (hops > 0);
        return std::make_tuple(std::abs(hops), is_fwd, dim);
    }
    // transmit along col
    if (sender_coord[1] == receiver_coord[1]) {
        constexpr auto dim = 0;
        const int hops = receiver_coord[dim] - sender_coord[dim];
        bool is_fwd = (hops > 0);
        return std::make_tuple(std::abs(hops), is_fwd, dim);
    }
    TT_THROW("Routing coordinates {} and {} invalid for 1D fabric", sender_coord, receiver_coord);
    return std::make_tuple(0, false, 0);
}
}  // namespace detail

/**
 * @brief Compute {num_hops, is_forward, neighbor FabricNodeId} for a 1-D unicast.
 *
 * Owns the forward/backward SIGN REVERSAL ("fabrics' forward/backward concept is
 * reversed" — returns the negated is_forward) and the ring-vs-line shorter-path
 * choice with WRAP/NONE boundary mode.
 */
inline DmRoute ccl_dm_route(
    const MeshDevice* mesh_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& receiver_coord,
    ttnn::ccl::Topology topology) {
    const auto& mesh_shape = mesh_device->get_view().shape();

    // sign indicates direction, however fabrics' forward/backward concept is reversed
    const auto [line_hops, line_is_forward, dim] = detail::fabric_1d_routing_vector(sender_coord, receiver_coord);

    TT_FATAL(line_hops != 0, "Should not be send/receiving to the same device");

    auto get_neighbor_id = [&sender_coord, &mesh_device, &mesh_shape, dim](
                               bool is_forward, MeshCoordinate::BoundaryMode boundary_mode) {
        const auto neighbor_coord = sender_coord.get_neighbor(mesh_shape, (is_forward ? 1 : -1), dim, boundary_mode);
        TT_FATAL(neighbor_coord.has_value(), "Can't find neighbor for {}", sender_coord);
        return mesh_device->get_fabric_node_id(*neighbor_coord);
    };

    if (topology == ttnn::ccl::Topology::Ring) {
        // fabric_1d_routing_vector returns |hops| + a direction flag; reconstruct the SIGNED line
        // distance, then the other way round the ring is `signed - sign(signed) * ring_size`.
        // (Pre-fix this computed `line_hops + sign(line_hops) * ring_size` over the already-ABSOLUTE
        // line_hops, which is always longer — the wrap branch was unreachable and a Ring route
        // silently degraded to the line route, with the wrong hop count AND direction for wrap
        // pairs. Caught by reduce_scatter's Refinement-1 fabric probe on a (1, 4) ring.)
        const int signed_line_hops = line_is_forward ? line_hops : -line_hops;
        const int ring_hops = signed_line_hops + ((signed_line_hops > 0 ? -1 : 1) * static_cast<int>(mesh_shape[dim]));
        if (std::abs(ring_hops) < std::abs(signed_line_hops)) {
            bool ring_is_forward = (ring_hops > 0);
            const auto next_fabric_id = get_neighbor_id(ring_is_forward, MeshCoordinate::BoundaryMode::WRAP);
            return {static_cast<uint32_t>(std::abs(ring_hops)), !ring_is_forward, next_fabric_id};
        }
    }
    const auto next_fabric_id = get_neighbor_id(line_is_forward, MeshCoordinate::BoundaryMode::NONE);
    return {static_cast<uint32_t>(line_hops), !line_is_forward, next_fabric_id};
}

// ===========================================================================
// H5 — bidirectional line-route compile-time-arg packing (host<->kernel contract)
// ===========================================================================

/**
 * @brief Append the bidirectional line-route block to a writer's compile-time args in the
 *        EXACT order the kernel reads it — forward-unicast, forward-multicast,
 *        backward-unicast, backward-multicast (see ccl_routing_utils::
 *        get_line_{unicast,multicast}_route_info_from_args, e.g. all_gather_async's
 *        minimal_default_writer). Owns the host<->kernel arg-layout contract in one place,
 *        the same way build_ccl_fabric_rt_args owns the connection-arg layout.
 *
 * @note This PACKS; it does NOT compute. The route args are produced by the existing
 *   ring-route abstraction @c ttnn::ccl::get_forward_backward_line_{unicast,mcast}_configuration —
 *   a CCL host helper must not duplicate that route math. For the 1-D point_to_point
 *   single-route case use @c ccl_dm_route above instead; the bidirectional ring case
 *   (all_gather) is topology-specific and does not unify with it host-side — hence two
 *   route surfaces, one packing contract.
 */
// @c UnicastArgs / @c MulticastArgs are the route-arg containers from
// ttnn::ccl::get_forward_backward_line_{unicast,mcast}_configuration (today std::array<uint32_t,2>
// and std::array<uint32_t,6>); templated so any contiguous uint32_t range packs identically.
template <typename UnicastArgs, typename MulticastArgs>
inline void append_ccl_line_route_ct_args(
    std::vector<uint32_t>& ct_args,
    const UnicastArgs& forward_unicast_args,
    const MulticastArgs& forward_multicast_args,
    const UnicastArgs& backward_unicast_args,
    const MulticastArgs& backward_multicast_args) {
    ct_args.insert(ct_args.end(), forward_unicast_args.begin(), forward_unicast_args.end());
    ct_args.insert(ct_args.end(), forward_multicast_args.begin(), forward_multicast_args.end());
    ct_args.insert(ct_args.end(), backward_unicast_args.begin(), backward_unicast_args.end());
    ct_args.insert(ct_args.end(), backward_multicast_args.begin(), backward_multicast_args.end());
}

// ===========================================================================
// H3 — fabric-connection runtime-arg append (kernel-matched layout)
// ===========================================================================

/**
 * @brief Build the fabric-connection runtime-arg BLOCK in the EXACT layout the kernel-side
 *        FabricStreamSender consumes, owning the has-forward / has-backward flag dance:
 *
 *   [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
 *
 * This is the bridge between what the op author knows (two fabric nodes + a direction) and what
 * the fabric needs on the wire — the author never lays out connection args by hand.
 *
 * Place the returned block FIRST in the kernel's runtime args. The kernel then consumes it with a
 * cursor starting at 0 (`size_t arg_idx = 0; FabricStreamSender<> sender(arg_idx, ...)` advances
 * it past the block) and reads the op's own args from the cursor after — so neither side ever
 * hardcodes where the fabric block starts. For a unidirectional sender the leading has_forward
 * flag also equals the send direction, so the kernel peeks `get_arg_val<uint32_t>(0)`.
 */
inline std::vector<uint32_t> build_ccl_fabric_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& neighbor_fabric_node_id,
    uint32_t link_idx,
    ProgramDescriptor& desc,
    const CoreCoord& core,
    bool is_forward) {
    std::vector<uint32_t> rt_args;
    rt_args.push_back(is_forward);  // has_forward
    if (is_forward) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fabric_node_id, neighbor_fabric_node_id, link_idx, desc, core, rt_args);
    }
    rt_args.push_back(!is_forward);  // has_backward
    if (!is_forward) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fabric_node_id, neighbor_fabric_node_id, link_idx, desc, core, rt_args);
    }
    return rt_args;
}

// ===========================================================================
// H4 — cross-device GlobalSemaphore lifecycle
// ===========================================================================

/**
 * @brief Allocate a GlobalSemaphore on the mesh's worker cores and run the cache-miss
 *        cross-device Synchronize barrier. Returns the semaphore; the CALLER must keep
 *        it alive for the cached workload's lifetime (point_to_point parks it in
 *        WorkloadDescriptor::semaphores).
 */
inline GlobalSemaphore make_ccl_semaphore(MeshDevice* mesh_device, uint32_t initial_value = 0) {
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, initial_value);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    return semaphore;
}

}  // namespace ttnn::ccl::dataflow
