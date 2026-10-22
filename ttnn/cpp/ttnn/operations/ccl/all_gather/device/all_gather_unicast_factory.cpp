// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_unicast_factory.hpp"

#include <array>

#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

////////////////////////////////////////////////////////////////
// Store-and-forward all_gather.
//
// Every device relays stripes one hop at a time; a stripe reaches far devices by being re-sent at each hop.
// One core per (link, direction), core 2 * link + dir (dir 0 forward, 1 backward). Its builder walks the
// slice and fills slots (iteration 0 from the input, later ones from what upstream delivered into our
// output, gated on data_valid); its pump sends them one hop and counts what it delivered into the
// downstream builder's data_valid. The schedule (iterations, even-ring split, counts) comes from the plan.
////////////////////////////////////////////////////////////////

namespace {

constexpr const char* kUnicastBuilderPath =
    "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_builder.cpp";
constexpr const char* kUnicastPumpPath = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_pump.cpp";

all_gather_program::Kernels create_unicast_program(
    tt::tt_metal::Program& program,
    const AllGatherParams& args,
    const all_gather_plan::Plan& plan,
    const all_gather_program::Workers& workers,
    const ttnn::MeshCoordinate& coord,
    const Tensor& input,
    const Tensor& output) {
    namespace agp = all_gather_program;
    auto* mesh_device = input.device();

    TT_FATAL(!args.is_true_2d(), "all_gather unicast algorithm does not support true 2D topologies");
    const uint32_t axis = args.get_1d_axis();
    const auto topology = args.axis_topology[axis];
    const uint32_t dev = ::ttnn::ccl::get_linearized_index_from_physical_coord(input, coord, args.cluster_axis);
    const std::array<std::optional<ttnn::MeshCoordinate>, 2> neighbor = {
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, 1, topology, axis),
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, -1, topology, axis)};

    const auto slot = agp::make_slot_layout(1, plan.geo.P);
    // Not const: the fabric connection helper takes the kernel handle by reference.
    auto kernels = agp::create_kernels(
        program,
        workers.core_range,
        plan,
        slot,
        agp::KernelSpec{
            .builder_path = kUnicastBuilderPath,
            .pump_path = kUnicastPumpPath,
            .builder_args = {},
            .pump_args = {{"signal_units", agp::kSignalPackets * plan.geo.P / plan.geo.A}}},
        input,
        output);

    const auto self_node = mesh_device->get_fabric_node_id(coord);
    for (uint32_t link = 0; link < plan.links; ++link) {
        const auto& slice = plan.slices[link];
        for (uint32_t dir = 0; dir < 2; ++dir) {
            const auto s = all_gather_plan::unicast_schedule(plan, dev, link, dir);
            const auto& core = workers.cores[(2 * link) + dir];
            // data_valid goes to our mirror core downstream; init to the downstream's opposite-direction core,
            // the one that sends into our output.
            const auto self_xy = mesh_device->worker_core_from_logical_core(core);
            const auto partner_xy = mesh_device->worker_core_from_logical_core(workers.cores[(2 * link) + 1 - dir]);
            TT_FATAL(s.I == 0 || neighbor[dir].has_value(), "an active all_gather direction must have a neighbor");
            const auto dst = s.I > 0 ? mesh_device->get_fabric_node_id(*neighbor[dir]) : self_node;

            std::vector<uint32_t> builder_args = {
                input.buffer()->address(),   // kBuilderRtInAddr
                output.buffer()->address(),  // kBuilderRtOutAddr
                dev,
                s.stripe_step,
                s.I,
                slice.e0,
                slice.ne,
                s.last_skip,
                s.last_take,
                s.iter_units,
                s.recv_units,
                workers.sync_sem.address(),
            };
            agp::append_route_args(builder_args, dst, 0, 0);
            tt::tt_metal::SetRuntimeArgs(program, kernels.builder, core, builder_args);

            std::vector<uint32_t> pump_args = {
                s.I,
                s.iter_units,
                s.last_units,
                s.local_copy ? 1u : 0u,
                workers.init_sem.address(),
                workers.sync_sem.address(),
                self_xy.x,
                self_xy.y,
                partner_xy.x,
                partner_xy.y,
            };
            agp::append_route_args(pump_args, dst, 0, 0);
            if (s.I > 0) {
                tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                    self_node, {dst}, {link}, program, kernels.pump, core, pump_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, kernels.pump, core, pump_args);
        }
    }
    return kernels;
}

}  // namespace

AllGatherUnicastFactory::cached_mesh_workload_t AllGatherUnicastFactory::create_mesh_workload(
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
        all_gather_plan::AgAlgo::Unicast,
        /*cores_per_link=*/2,
        [&](tt::tt_metal::Program& program, const auto& plan, const auto& workers, const auto& coord) {
            return create_unicast_program(program, operation_attributes, plan, workers, coord, input, output_tensor);
        });
}

void AllGatherUnicastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    all_gather_program::override_runtime_arguments(cached_workload, tensor_args.input_tensor, output_tensor);
}

}  // namespace ttnn::operations::ccl
