// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_multicast_factory.hpp"

#include <array>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "all_gather_plan.hpp"
#include "all_gather_program.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

namespace {

namespace plan_ns = all_gather_plan;
namespace prog = all_gather_program;

////////////////////////////////////////////////////////////////
// Multicast all_gather (1D line/ring or true 2D).
//
// One core per link: its reader reads this device's slice once, writes it into the local output and fills slots;
// its writer multicasts each packet on every connection (one per direction with a neighbor), alternating two route
// variants on an even ring. Init and done ride every (connection, variant), so each device receives V * (N - 1) of
// each, separately counted.
////////////////////////////////////////////////////////////////

tt::tt_metal::ProgramDescriptor build_multicast_program(
    const AllGatherParams& params,
    const ttnn::MeshCoordinate& coord,
    const Tensor& input,
    const Tensor& output,
    const plan_ns::AgPlan& plan,
    uint32_t init_sem,
    uint32_t done_sem) {
    auto* mesh_device = input.device();
    const uint32_t dev = ::ttnn::ccl::get_linearized_index_from_physical_coord(input, coord, params.cluster_axis);
    const auto src_node = mesh_device->get_fabric_node_id(coord);
    const bool fabric_2d = tt::tt_fabric::is_2d_fabric_config(params.fabric_config);

    // Neighbors by logical direction (plan_ns::Dir): axis 1 runs E (forward) / W, axis 0 runs S (forward) / N. A
    // 2-device axis is a line (both neighbors are the same device, one hop).
    std::array<std::optional<MeshCoordinate>, 4> neighbor{};
    std::array<bool, 2> axis_ring{};
    std::array<uint32_t, 4> phys_dir = {plan_ns::kEast, plan_ns::kWest, plan_ns::kNorth, plan_ns::kSouth};
    for (uint32_t a = 0; a < 2; ++a) {
        if (params.axis_num_devices[a] <= 1) {
            continue;
        }
        axis_ring[a] = tt::tt_fabric::is_ring_or_torus(params.axis_topology[a]) && params.axis_num_devices[a] > 2;
        const auto topology = axis_ring[a] ? params.axis_topology[a] : tt::tt_fabric::Topology::Linear;
        const uint32_t fwd = a == 1 ? plan_ns::kEast : plan_ns::kSouth;
        const uint32_t bwd = a == 1 ? plan_ns::kWest : plan_ns::kNorth;
        neighbor[fwd] = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, 1, topology, a);
        neighbor[bwd] = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, -1, topology, a);
    }
    // 2D routes put a connection's hop counts at the neighbor's physical fabric direction, which depends on where this
    // device sits. A 1D route is a single hop count, so the slot does not matter there.
    for (uint32_t d = 0; fabric_2d && d < neighbor.size(); ++d) {
        if (neighbor[d].has_value()) {
            const auto dir =
                tt::tt_fabric::get_eth_forwarding_direction(src_node, mesh_device->get_fabric_node_id(*neighbor[d]));
            TT_FATAL(
                dir.has_value() && static_cast<uint32_t>(*dir) < tt::tt_fabric::eth_chan_directions::Z,
                "all_gather multicast on a 2D fabric needs a cardinal (E/W/N/S) direction to each mesh neighbor");
            phys_dir[d] = static_cast<uint32_t>(*dir);
        }
    }
    const auto routes = plan_ns::mcast_routes(
        plan.N,
        {params.axis_num_devices[0], params.axis_num_devices[1]},
        {coord[0], coord[1]},
        params.cluster_axis,
        axis_ring,
        phys_dir);
    TT_FATAL(!routes.conns.empty(), "all_gather multicast: device {} has no neighbor", dev);
    // A parity check of two derivations: the plan's variant count (hashed) and this device's routes.
    TT_FATAL(
        routes.alternates == (plan.variants == 2),
        "all_gather multicast: route variants disagree with the plan ({} vs {})",
        routes.alternates,
        plan.variants);
    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
    for (const auto& conn : routes.conns) {
        TT_FATAL(neighbor[conn.dir].has_value(), "all_gather multicast: no neighbor in direction {}", conn.dir);
        dst_nodes.push_back(mesh_device->get_fabric_node_id(*neighbor[conn.dir]));
    }

    tt::tt_metal::ProgramDescriptor desc;
    const CoreRangeSet cores = prog::worker_cores(plan);
    desc.cbs.push_back(prog::slot_cb(plan, cores));
    desc.kernels.push_back(prog::reader_kernel(plan, params.force, input, output, cores));
    desc.kernels.push_back(
        prog::writer_kernel(plan, params.force, prog::kMulticastWriterKernel, cores, {{"VARIANTS", plan.variants}}));
    tt::tt_metal::KernelHandle writer_handle = prog::kWriterIdx;  // the connection helper takes it by reference

    for (uint32_t link = 0; link < plan.mcast_cores.size(); ++link) {
        const CoreCoord& core = plan.mcast_cores[link];
        const auto ws = plan_ns::schedule(plan, dev, link, 0, 0);
        prog::add_reader_rt_args(desc.kernels[prog::kReaderIdx], core, input, output, ws, init_sem, done_sem);

        // Init and done go to the same core on every other device (mirror addressing).
        const CoreCoord mirror = mesh_device->worker_core_from_logical_core(core);
        std::vector<uint32_t> writer_rt = {
            ws.last_take,
            ws.init_send ? 1u : 0u,
            init_sem,  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            done_sem,  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            static_cast<uint32_t>(mirror.x),
            static_cast<uint32_t>(mirror.y),
            static_cast<uint32_t>(routes.conns.size()),
        };
        for (const auto& conn : routes.conns) {
            writer_rt.push_back(conn.range_v0);
            writer_rt.push_back(conn.range_v1);
        }
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            src_node,
            dst_nodes,
            {link},
            desc,
            writer_handle,
            core,
            writer_rt,
            fabric_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        desc.kernels[prog::kWriterIdx].runtime_args.emplace_back(core, std::move(writer_rt));
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllGatherMulticastFactory::create_workload_descriptor(
    const AllGatherParams& operation_attributes,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& input = tensor_args.input_tensor;
    auto [plan, workload, init_sem, done_sem] = prog::start_workload(operation_attributes, input, output_tensor);
    log_debug(
        tt::LogOp,
        "all_gather multicast: links={} k={} slot={} B variants={}",
        plan.choice.links,
        plan.choice.k,
        plan.slot_bytes,
        plan.variants);

    workload.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_multicast_program(operation_attributes, coord, input, output_tensor, plan, init_sem, done_sem)});
    }
    return std::move(workload);
}

}  // namespace ttnn::operations::ccl
