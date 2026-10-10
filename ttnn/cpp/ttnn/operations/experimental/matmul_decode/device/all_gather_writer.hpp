// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <array>
#include <variant>
#include <vector>

#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace ttnn::operations::experimental::matmul_decode {

inline tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs all_gather_named_compile_time_args(
    uint32_t cb_out_index,
    uint32_t M_tiles,
    uint32_t N_tiles_per_core,
    uint32_t ring_index,
    uint32_t ring_size,
    uint32_t out_ready_sem_id,
    uint32_t barrier_sem_id,
    uint32_t start_fwd,
    uint32_t range_fwd,
    uint32_t start_bwd,
    uint32_t range_bwd,
    uint32_t ag_rt_arg_base,
    uint32_t num_shards = 1,
    uint32_t shard_sem_id = 0,
    uint32_t staging_cb_id = 0,
    bool use_mux = false,
    uint32_t mux_num_buffers = 0,
    uint32_t mux_buffer_size = 0,
    uint32_t mux_status_address = 0,
    uint32_t mux_termination_address = 0,
    uint32_t mux_num_clients = 0) {
    return {
        {"cb_out", cb_out_index},
        {"ag_M_tiles", M_tiles},
        {"ag_N_tiles_per_core", N_tiles_per_core},
        {"ag_ring_index", ring_index},
        {"ag_ring_size", ring_size},
        {"ag_out_ready_sem_id", out_ready_sem_id},
        {"ag_barrier_sem_id", barrier_sem_id},
        {"ag_start_fwd", start_fwd},
        {"ag_range_fwd", range_fwd},
        {"ag_start_bwd", start_bwd},
        {"ag_range_bwd", range_bwd},
        {"ag_rt_arg_base", ag_rt_arg_base},
        {"ag_num_shards", num_shards},
        {"ag_shard_sem_id", shard_sem_id},
        {"ag_staging_cb", staging_cb_id},
        {"ag_use_mux", use_mux ? 1u : 0u},
        {"ag_mux_num_buffers", mux_num_buffers},
        {"ag_mux_buffer_size", mux_buffer_size},
        {"ag_mux_status", mux_status_address},
        {"ag_mux_termination", mux_termination_address},
        {"ag_mux_num_clients", mux_num_clients},
    };
}

// Resolve this device's ring index, multicast hop counts, and fabric neighbors.
struct AllGatherFabricRoute {
    uint32_t ring_index = 0;
    uint32_t start_fwd = 0;
    uint32_t range_fwd = 0;
    uint32_t start_bwd = 0;
    uint32_t range_bwd = 0;
    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
};

inline AllGatherFabricRoute make_all_gather_fabric_route(
    const Tensor& tensor_a, const ttnn::MeshCoordinate& sender_coord) {
    AllGatherFabricRoute route;
    const auto topology = ::ttnn::ccl::get_usable_topology(tensor_a, std::nullopt, std::nullopt);
    route.ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(tensor_a, sender_coord, std::nullopt);
    const uint32_t ring_size = ::ttnn::ccl::get_topological_dimension(tensor_a, std::nullopt);
    // Line mcast hop counts only distinguish Linear vs Ring; Mesh/Torus collapse to those.
    const auto line_topology = ::ttnn::ccl::convert_2d_to_1d_topology(topology);
    auto [num_targets_forward, num_targets_backward] =
        ::ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, route.ring_index, line_topology, true);

    auto forward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(tensor_a, sender_coord, 1, topology, std::nullopt);
    auto backward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(tensor_a, sender_coord, -1, topology, std::nullopt);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "matmul_decode all_gather: no fabric neighbor");

    auto* mesh_device = tensor_a.device();
    if (forward_coord.has_value()) {
        route.start_fwd = 1;
        route.range_fwd = num_targets_forward;
        route.dst_nodes.push_back(mesh_device->get_fabric_node_id(forward_coord.value()));
    }
    if (backward_coord.has_value()) {
        route.start_bwd = 1;
        route.range_bwd = num_targets_backward;
        route.dst_nodes.push_back(mesh_device->get_fabric_node_id(backward_coord.value()));
    }
    return route;
}

// Select one valid Ethernet channel for each destination connection. A forwarding link is
// direction-specific: on a line/ring, the link that reaches the forward neighbor need not be
// valid for the backward neighbor. Reusing one link for both connections can leave a multicast
// atomic increment stranded while open_connections still succeeds.
inline std::vector<uint32_t> all_gather_forwarding_links(
    const Tensor& tensor_a,
    const ttnn::MeshCoordinate& sender_coord,
    const AllGatherFabricRoute& route,
    uint32_t core_index) {
    TT_FATAL(!route.dst_nodes.empty(), "matmul_decode all_gather: no fabric destinations");
    const auto sender_node = tensor_a.device()->get_fabric_node_id(sender_coord);
    std::vector<uint32_t> links;
    links.reserve(route.dst_nodes.size());
    for (const auto& dst_node : route.dst_nodes) {
        const auto valid_links = tt::tt_fabric::get_forwarding_link_indices(sender_node, dst_node);
        TT_FATAL(!valid_links.empty(), "matmul_decode all_gather: no forwarding link to destination {}", dst_node);
        links.push_back(valid_links[core_index % valid_links.size()]);
    }
    return links;
}

// Prefix (if any) is the writer kernel's existing runtime args; the all-gather block
// [out_buffer, noc_x, noc_y, num_connections, fabric...] is appended after it.
inline void set_all_gather_writer_runtime_args(
    tt::tt_metal::ProgramDescriptor& desc,
    tt::tt_metal::KernelHandle writer_id,
    const tt::tt_metal::CoreCoord& core,
    tt::tt_metal::IDevice* device,
    const Tensor& tensor_a,
    const ttnn::MeshCoordinate& sender_coord,
    Tensor& output,
    const AllGatherFabricRoute& route,
    const tt::tt_metal::CoreCoord& output_core,
    uint32_t output_tile_offset,
    const std::vector<uint32_t>& link_indices = {},
    const std::vector<uint32_t>& prefix_args = {},
    const std::vector<tt::tt_metal::CoreCoord>& extra_shard_phys = {},
    const std::optional<tt::tt_fabric::FabricMuxConfig>& mux_config = std::nullopt,
    const std::array<tt::tt_metal::CoreCoord, 2>& mux_cores = {},
    const std::array<tt::tt_metal::CoreCoord, 2>& termination_masters = {},
    const std::array<uint32_t, 2>& mux_worker_ids = {},
    const std::array<uint32_t, 2>& mux_client_counts = {}) {
    const auto output_phys = device->worker_core_from_logical_core(output_core);
    const auto worker_phys = device->worker_core_from_logical_core(core);

    if (mux_config.has_value()) {
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.reserve(prefix_args.size() + 3 + 2 * extra_shard_phys.size() + 36);
        for (const auto arg : prefix_args) {
            rt.push_back(arg);
        }
        rt.push_back(output.buffer());
        rt.push_back(static_cast<uint32_t>(output_phys.x));
        rt.push_back(static_cast<uint32_t>(output_phys.y));
        rt.push_back(output_tile_offset);
        rt.push_back(static_cast<uint32_t>(worker_phys.x));
        rt.push_back(static_cast<uint32_t>(worker_phys.y));
        for (const auto& shard_phys : extra_shard_phys) {
            rt.push_back(static_cast<uint32_t>(shard_phys.x));
            rt.push_back(static_cast<uint32_t>(shard_phys.y));
        }
        const auto append_mux_args = [&](uint32_t dir) {
            const bool valid = dir == 0 ? route.range_fwd != 0 : route.range_bwd != 0;
            ttnn::experimental::ccl::append_fabric_mux_connection_rt_args(
                valid,
                device->worker_core_from_logical_core(mux_cores[dir]),
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_config.value(),
                core,
                mux_worker_ids[dir],
                mux_worker_ids[dir] == 0,
                device->worker_core_from_logical_core(termination_masters[dir]),
                desc,
                rt);
            rt.push_back(mux_client_counts[dir]);
        };
        append_mux_args(0);
        append_mux_args(1);
        desc.kernels[writer_id].emplace_runtime_args(core, rt);
    } else {
        std::vector<uint32_t> rt = prefix_args;
        const size_t out_addr_idx = rt.size();
        rt.push_back(0);
        rt.push_back(static_cast<uint32_t>(output_phys.x));
        rt.push_back(static_cast<uint32_t>(output_phys.y));
        rt.push_back(output_tile_offset);
        rt.push_back(static_cast<uint32_t>(worker_phys.x));
        rt.push_back(static_cast<uint32_t>(worker_phys.y));
        for (const auto& shard_phys : extra_shard_phys) {
            rt.push_back(static_cast<uint32_t>(shard_phys.x));
            rt.push_back(static_cast<uint32_t>(shard_phys.y));
        }
        rt.push_back(static_cast<uint32_t>(route.dst_nodes.size()));
        const auto sender_node = tensor_a.device()->get_fabric_node_id(sender_coord);
        tt::tt_metal::KernelHandle writer_id_mut = writer_id;
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args<tt::tt_metal::ProgramDescriptor>(
            sender_node, route.dst_nodes, link_indices, desc, writer_id_mut, core, rt);
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> var;
        var.reserve(rt.size());
        for (size_t i = 0; i < rt.size(); ++i) {
            var.emplace_back(
                i == out_addr_idx ? std::variant<uint32_t, tt::tt_metal::Buffer*>(output.buffer())
                                  : std::variant<uint32_t, tt::tt_metal::Buffer*>(rt[i]));
        }
        desc.kernels[writer_id].emplace_runtime_args(core, var);
    }
}

}  // namespace ttnn::operations::experimental::matmul_decode
