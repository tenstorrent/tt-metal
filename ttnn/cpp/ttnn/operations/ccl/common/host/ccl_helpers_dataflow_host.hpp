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
 * Design + per-op mapping: see CCL_DATAFLOW_HELPER_DESIGN.md at the repo root. The
 * bodies of ccl_packet_dims and ccl_dm_route are the proven point_to_point
 * detail::compute_aligned_packet_dims / detail::fabric_1d_routing, moved here so a
 * single documented surface owns them (incl. the bf16 bit_floor and the
 * forward/backward sign-reversal footguns).
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
 * (Moved verbatim from point_to_point detail::compute_aligned_packet_dims.)
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
 * choice with WRAP/NONE boundary mode. (Moved from point_to_point
 * detail::fabric_1d_routing.)
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
        int ring_hops = line_hops + ((line_hops < 0 ? -1 : 1) * mesh_shape[dim]);
        if (std::abs(ring_hops) < std::abs(line_hops)) {
            bool ring_is_forward = (ring_hops > 0);
            const auto next_fabric_id = get_neighbor_id(ring_is_forward, MeshCoordinate::BoundaryMode::WRAP);
            return {static_cast<uint32_t>(std::abs(ring_hops)), !ring_is_forward, next_fabric_id};
        }
    }
    const auto next_fabric_id = get_neighbor_id(line_is_forward, MeshCoordinate::BoundaryMode::NONE);
    return {static_cast<uint32_t>(line_hops), !line_is_forward, next_fabric_id};
}

// Line-MULTICAST route computation (for a ring barrier/broadcast, e.g. all_gather) is
// deliberately omitted: its only consumer is all_gather, which is @skip_for_blackhole
// and thus unverifiable on the available hardware. When all_gather is migrated (with a
// test), add a `mcast_route(...)` here over ccl::get_forward_backward_line_mcast_distance
// alongside the kernel-side multicast route setter — see the ccl_helpers_dataflow.hpp banner.

// ===========================================================================
// H3 — fabric-connection runtime-arg append (kernel-matched layout)
// ===========================================================================

/**
 * @brief Append the fabric-connection runtime args in the EXACT layout the
 *        kernel-side FabricStreamSender consumes, owning the has-forward / has-backward
 *        flag dance (removes the conn_arg_idx overlap footgun).
 *
 * After the call, the block beginning at the pre-call rt_args.size() is:
 *   [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
 * The kernel records that start index as conn_arg_idx; for a unidirectional sender the
 * has_forward flag also equals the send direction, so the kernel can peek it for
 * `is_forward`. This is point_to_point's send_program_factory.cpp:167-175 dance as one call.
 */
inline void append_ccl_fabric_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& neighbor_fabric_node_id,
    uint32_t link_idx,
    ProgramDescriptor& desc,
    const CoreCoord& core,
    std::vector<uint32_t>& rt_args,
    bool is_forward) {
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
}

// ===========================================================================
// H4 — cross-device GlobalSemaphore lifecycle
// ===========================================================================

/**
 * @brief Allocate a GlobalSemaphore on the mesh's worker cores and run the cache-miss
 *        cross-device Synchronize barrier. Returns the semaphore; the CALLER must keep
 *        it alive for the cached workload's lifetime (point_to_point parks it in
 *        WorkloadDescriptor::semaphores). (Moved from point_to_point
 *        create_workload_descriptor.)
 */
inline GlobalSemaphore make_ccl_semaphore(MeshDevice* mesh_device, uint32_t initial_value = 0) {
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    auto semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, initial_value);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    return semaphore;
}

}  // namespace ttnn::ccl::dataflow
