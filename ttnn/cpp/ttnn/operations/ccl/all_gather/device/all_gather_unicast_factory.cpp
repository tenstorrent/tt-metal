// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_unicast_factory.hpp"

#include <array>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "all_gather_plan.hpp"
#include "all_gather_program.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::ccl {

namespace {

namespace plan_ns = all_gather_plan;
namespace prog = all_gather_program;

////////////////////////////////////////////////////////////////
// Store-and-forward all_gather (line or ring on one mesh axis; Fabric 1D or 2D).
//
// Per (link, direction) one Mux V2 owns the fabric connection to the neighbor and n workers feed it, one channel
// each. A worker's reader fills slots (iteration 0 from the input, later iterations from the stripe upstream relayed
// into our output, gated on data_valid); its writer encodes each slot's packets and sends them through the mux, then
// signals data_valid on the downstream mirror worker. Core placement, slices, schedules and k all come from the plan,
// so this factory only translates them into kernels and arguments.
////////////////////////////////////////////////////////////////

tt::tt_metal::ProgramDescriptor build_unicast_program(
    const AllGatherParams& params,
    const ttnn::MeshCoordinate& coord,
    const Tensor& input,
    const Tensor& output,
    const plan_ns::AgPlan& plan,
    const tt::tt_fabric::FabricMuxV2Config& mux_config,
    uint32_t init_sem,
    uint32_t data_sem) {
    auto* mesh_device = input.device();
    const uint32_t axis = params.get_1d_axis();
    // A 2-device ring is planned as a line (each neighbor is one hop either way), so look neighbors up as one.
    const auto topology = plan.ring ? params.axis_topology[axis] : tt::tt_fabric::Topology::Linear;
    const uint32_t dev = ::ttnn::ccl::get_linearized_index_from_physical_coord(input, coord, params.cluster_axis);
    const std::array<std::optional<MeshCoordinate>, 2> neighbor = {
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, 1, topology, axis),
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input, coord, -1, topology, axis)};
    const auto src_node = mesh_device->get_fabric_node_id(coord);
    const auto& facts = params.facts;

    tt::tt_metal::ProgramDescriptor desc;
    const CoreRangeSet cores = prog::worker_cores(plan);
    desc.cbs.push_back(prog::slot_cb(plan, cores));
    desc.kernels.push_back(prog::reader_kernel(plan, params.force, input, output, cores));
    desc.kernels.push_back(prog::writer_kernel(
        plan, params.force, prog::kUnicastWriterKernel, cores, {{"MUX_BUFS", plan_ns::mux_buffers(plan.choice.n)}}));

    for (const auto& group : plan.groups) {
        // A mux exists only for a direction with a neighbor; its writers skip their whole body otherwise, so every
        // declared mux channel is opened and closed.
        const bool active = plan_ns::dir_active(plan, dev, group.dir);
        TT_FATAL(
            !active || neighbor[group.dir].has_value(), "all_gather: active direction {} has no neighbor", group.dir);
        const auto dst_node = active ? mesh_device->get_fabric_node_id(*neighbor[group.dir]) : src_node;
        CoreCoord mux_virtual{0, 0};
        if (active) {
            tt::tt_fabric::add_fabric_mux_v2_to_program(
                desc, mux_config, group.mux, src_node, dst_node, group.link, prog::kMuxForwarderNoc);
            mux_virtual = mesh_device->worker_core_from_logical_core(group.mux);
            // The mux map is raw L1 from the allocator base, outside the allocator. A CB over the same bytes makes
            // Metal's CB / L1-buffer clash check refuse a launch whose L1 tensors reach down into it.
            const auto map_bytes = static_cast<uint32_t>(mux_config.get_memory_map_end_address()) - facts.l1_base;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = map_bytes,
                .core_ranges = CoreRangeSet(CoreRange(group.mux)),
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = tt::CBIndex::c_0,
                    .data_format = tt::DataFormat::RawUInt8,
                    .page_size = map_bytes}}},
            });
        }
        // The init credit goes to the neighbor's opposite-direction worker w: the one that writes back into us.
        const auto& partner_group = plan.groups.at((group.link * 2) + (1 - group.dir));

        for (uint32_t w = 0; w < group.workers.size(); ++w) {
            const CoreCoord& core = group.workers[w];
            const auto ws = plan_ns::schedule(plan, dev, group.link, group.dir, w);
            prog::add_reader_rt_args(desc.kernels[prog::kReaderIdx], core, input, output, ws, init_sem, data_sem);

            // Mirror addressing: core coordinates are the same on every device, so our own coordinates name the
            // downstream worker that receives data_valid.
            const CoreCoord mirror = mesh_device->worker_core_from_logical_core(core);
            const CoreCoord partner = mesh_device->worker_core_from_logical_core(partner_group.workers.at(w));
            std::vector<uint32_t> writer_rt = {
                active ? 1u : 0u,
                ws.iters,
                ws.ne,
                ws.last_take,
                ws.cadence_slots,
                ws.init_send ? 1u : 0u,
                init_sem,  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
                data_sem,  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
                static_cast<uint32_t>(partner.x),
                static_cast<uint32_t>(partner.y),
                static_cast<uint32_t>(mirror.x),
                static_cast<uint32_t>(mirror.y),
                static_cast<uint32_t>(dst_node.chip_id),   // 2D packet-header route
                static_cast<uint32_t>(*dst_node.mesh_id),  // 2D packet-header route
            };
            if (active) {
                // Worker w is channel w of its direction's mux; two local semaphores back the connection.
                const uint32_t flow_control_sem =
                    ttnn::experimental::ccl::add_semaphore_descriptor(desc, CoreRangeSet(CoreRange(core)));
                const uint32_t teardown_sem =
                    ttnn::experimental::ccl::add_semaphore_descriptor(desc, CoreRangeSet(CoreRange(core)));
                mux_config.append_client_connection_rt_args(
                    mux_virtual,
                    static_cast<uint8_t>(w),
                    tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
                        .flow_control_sem_id = flow_control_sem, .teardown_sem_id = teardown_sem},
                    writer_rt);
            }
            desc.kernels[prog::kWriterIdx].runtime_args.emplace_back(core, std::move(writer_rt));
        }
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllGatherUnicastFactory::create_workload_descriptor(
    const AllGatherParams& operation_attributes,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& input = tensor_args.input_tensor;
    auto [plan, workload, init_sem, data_sem] = prog::start_workload(operation_attributes, input, output_tensor);
    const AgFacts& facts = operation_attributes.facts;

    // The mux map runs up from the allocator base, below the L1_SMALL floor (legality checked that it fits).
    const uint32_t n = plan.choice.n;
    const tt::tt_fabric::FabricMuxV2Config mux_config(
        *input.device(),
        static_cast<uint8_t>(n),
        static_cast<uint8_t>(plan_ns::mux_buffers(n)),
        facts.channel_bytes,
        facts.l1_base,
        facts.l1_small_floor);
    log_debug(
        tt::LogOp,
        "all_gather unicast: n={} links={} k={} slot={} B; mux map [{:#x}, {:#x}) on each mux core",
        n,
        plan.choice.links,
        plan.choice.k,
        plan.slot_bytes,
        facts.l1_base,
        mux_config.get_memory_map_end_address());

    workload.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_unicast_program(
                 operation_attributes, coord, input, output_tensor, plan, mux_config, init_sem, data_sem)});
    }
    return std::move(workload);
}

}  // namespace ttnn::operations::ccl
