// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_multicast_factory.hpp"

#include <array>

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

////////////////////////////////////////////////////////////////
// Multicast all_gather.
//
// One core per link. Its builder reads this device's slice of the input once into slots whose headers carry
// one route per fabric connection; its pump sends each slot on every connection and copies it into the
// local output. The route sets (and the even-ring alternation) come from the plan; every receiver is covered
// once per route variant.
////////////////////////////////////////////////////////////////

namespace {

constexpr const char* kMulticastBuilderPath =
    "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_builder.cpp";
constexpr const char* kMulticastPumpPath = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_pump.cpp";

using all_gather_plan::kEast;
using all_gather_plan::kNorth;
using all_gather_plan::kSouth;
using all_gather_plan::kWest;

all_gather_program::Kernels create_multicast_program(
    tt::tt_metal::Program& program,
    const AllGatherParams& args,
    const all_gather_plan::Plan& plan,
    const all_gather_program::Workers& workers,
    const ttnn::MeshCoordinate& coord,
    const Tensor& input,
    const Tensor& output) {
    namespace agp = all_gather_program;
    auto* mesh_device = input.device();
    const auto self_node = mesh_device->get_fabric_node_id(coord);
    const uint32_t dev = ::ttnn::ccl::get_linearized_index_from_physical_coord(input, coord, args.cluster_axis);

    // Neighbors by logical direction (axis 1 forward = E, axis 0 forward = S), and the physical direction
    // the fabric reaches each one by.
    std::array<std::optional<ttnn::MeshCoordinate>, 4> neighbor;
    std::array<uint32_t, 4> phys_dir = {kEast, kWest, kNorth, kSouth};
    std::array<bool, 2> axis_ring{};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        if (args.axis_num_devices[axis] <= 1) {
            continue;
        }
        // 2D routes assume a straight axis. A bent 1D axis is planned as unicast; no view bends a 2D axis.
        TT_FATAL(
            !tt::tt_fabric::is_2d_fabric_config(args.fabric_config) || args.axis_is_straight[axis],
            "all_gather 2D multicast needs mesh axis {} wired straight",
            axis);
        const auto topology = args.axis_topology[axis];
        // A 2-device ring is a line: both directions would reach the same neighbor.
        axis_ring[axis] = tt::tt_fabric::is_ring_or_torus(topology) && args.axis_num_devices[axis] > 2;
        const uint32_t fwd = axis == 1 ? kEast : kSouth;
        const uint32_t bwd = axis == 1 ? kWest : kNorth;
        neighbor[fwd] = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, 1, topology, axis);
        neighbor[bwd] = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, -1, topology, axis);
        for (const uint32_t d : {fwd, bwd}) {
            if (!neighbor[d].has_value()) {
                continue;
            }
            const auto phys =
                tt::tt_fabric::get_eth_forwarding_direction(self_node, mesh_device->get_fabric_node_id(*neighbor[d]));
            TT_FATAL(
                phys.has_value() && static_cast<uint32_t>(*phys) <= kSouth,
                "all_gather expected a cardinal (E/W/N/S) fabric direction to a mesh neighbor");
            phys_dir[d] = static_cast<uint32_t>(*phys);
        }
    }

    const std::optional<uint32_t> axis = args.is_true_2d() ? std::nullopt : std::optional<uint32_t>(args.get_1d_axis());
    const auto routes = all_gather_plan::mcast_routes(
        plan,
        args.axis_num_devices,
        {static_cast<uint32_t>(coord[0]), static_cast<uint32_t>(coord[1])},
        axis,
        axis_ring,
        phys_dir);
    const auto num_conns = static_cast<uint32_t>(routes.conns.size());
    TT_FATAL(num_conns > 0, "all_gather multicast found no route to another device");

    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
    std::vector<uint32_t> route_args;
    for (const auto& conn : routes.conns) {
        TT_FATAL(
            neighbor[conn.dir].has_value(), "all_gather multicast route has no neighbor in direction {}", conn.dir);
        dst_nodes.push_back(mesh_device->get_fabric_node_id(*neighbor[conn.dir]));
        agp::append_route_args(route_args, dst_nodes.back(), conn.range_v0, conn.range_v1);
    }

    const auto slot = agp::make_slot_layout(num_conns, plan.geo.P);
    const agp::NamedArgs alternate = {{"alternate", routes.alternates ? 1u : 0u}};
    // Not const: the fabric connection helper takes the kernel handle by reference.
    auto kernels = agp::create_kernels(
        program,
        workers.core_range,
        plan,
        slot,
        agp::KernelSpec{
            .builder_path = kMulticastBuilderPath,
            .pump_path = kMulticastPumpPath,
            .builder_args = alternate,
            .pump_args = alternate},
        input,
        output);

    for (uint32_t link = 0; link < plan.links; ++link) {
        const auto& slice = plan.slices[link];
        const auto& core = workers.cores[link];
        // Every device uses the same core for a link, so the semaphores we target remotely are our own.
        const auto self_xy = mesh_device->worker_core_from_logical_core(core);

        std::vector<uint32_t> builder_args = {
            input.buffer()->address(),   // kBuilderRtInAddr
            output.buffer()->address(),  // kBuilderRtOutAddr
            dev,
            slice.e0,
            slice.ne,
        };
        builder_args.insert(builder_args.end(), route_args.begin(), route_args.end());
        tt::tt_metal::SetRuntimeArgs(program, kernels.builder, core, builder_args);

        std::vector<uint32_t> pump_args = {
            slice.ne * plan.geo.E,
            workers.init_sem.address(),
            workers.sync_sem.address(),
            plan.N - 1,
            all_gather_plan::mcast_done_expect(plan, routes.alternates),
            self_xy.x,
            self_xy.y,
        };
        pump_args.insert(pump_args.end(), route_args.begin(), route_args.end());
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            self_node, dst_nodes, {link}, program, kernels.pump, core, pump_args);
        tt::tt_metal::SetRuntimeArgs(program, kernels.pump, core, pump_args);
    }
    return kernels;
}

}  // namespace

AllGatherMulticastFactory::cached_mesh_workload_t AllGatherMulticastFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    return all_gather_program::build_workload(
        operation_attributes,
        tensor_coords,
        input,
        output_tensor,
        all_gather_plan::AgAlgo::Multicast,
        /*cores_per_link=*/1,
        [&](tt::tt_metal::Program& program, const auto& plan, const auto& workers, const auto& coord) {
            return create_multicast_program(program, operation_attributes, plan, workers, coord, input, output_tensor);
        });
}

void AllGatherMulticastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    all_gather_program::override_runtime_arguments(cached_workload, tensor_args.input_tensor, output_tensor);
}

}  // namespace ttnn::operations::ccl
