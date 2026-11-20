// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_multicast_factory.hpp"

#include "all_gather_device_operation.hpp"
#include "all_gather_factory_common.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/descriptor_runtime_args.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

namespace {

tt::tt_metal::ProgramDescriptor build_multicast_descriptor_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem,
    const tt::tt_metal::GlobalSemaphore& done_sem,
    uint32_t num_available_cores) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::ProgramDescriptor desc;
    TT_FATAL(input_tensor.is_allocated(), "all_gather multicast requires an allocated input buffer");
    TT_FATAL(output_tensor.is_allocated(), "all_gather multicast requires an allocated output buffer");

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    ////////////////////////////////////////////////////////////////

    auto* mesh_device = input_tensor.device();
    const uint32_t num_devices = operation_attributes.num_devices;

    // Arch-dependent settings: the uncalibrated arch's values, then each calibrated arch's.
    bool writer_on_noc0 = false;  // else the writer is on NOC1; the reader takes the other
    switch (mesh_device->arch()) {
        case tt::ARCH::WORMHOLE_B0:
        case tt::ARCH::BLACKHOLE: writer_on_noc0 = true; break;
        default: break;  // uncalibrated arch: the default reader/writer NOCs
    }

    uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    // Compute hops + neighbors for each mesh axis.
    // Each axis ∈ {0, 1} contributes a forward/backward pair: axis 1 -> (E=fwd, W=bwd),
    // axis 0 -> (S=fwd, N=bwd). In 1D only one axis is active; in 2D both can be.
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(operation_attributes.fabric_config);

    std::optional<MeshCoordinate> e_coord, w_coord, n_coord, s_coord;
    uint32_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
    uint32_t num_active_axes = 0;
    bool axis_load_balance = false;  // an even ring of 4 or more devices

    for (uint32_t axis = 0; axis < 2; ++axis) {
        const uint32_t axis_size = operation_attributes.axis_num_devices[axis];
        const bool is_axis_active = axis_size > 1;
        if (!is_axis_active) {
            continue;
        }
        ++num_active_axes;
        // A true 2D collective cannot fall back to unicast, so factory selection cannot save us
        // here. No mesh view bends a true 2D axis today, so this only guards a future one.
        TT_FATAL(
            !fabric_is_2d || operation_attributes.axis_is_straight[axis],
            "AllGather 2D multicast needs mesh axis {} wired straight, but it turns corners",
            axis);

        const auto axis_topology = operation_attributes.axis_topology[axis];
        const uint32_t axis_index = sender_device_coord[axis];
        // An even ring's antipode is reached on the forward route (route 0; the alternate route reaches it backward).
        auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
            axis_size, axis_index, axis_topology, /*static_alternate=*/false);
        axis_load_balance = tt::tt_fabric::is_ring_or_torus(axis_topology) && axis_size % 2 == 0 && axis_size >= 4;
        auto fwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, 1, axis_topology, axis);
        auto bwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, -1, axis_topology, axis);

        if (axis == 1) {
            e_hops = fwd_hops;
            w_hops = bwd_hops;
            e_coord = fwd_coord;
            w_coord = bwd_coord;
        } else {
            s_hops = fwd_hops;
            n_hops = bwd_hops;
            s_coord = fwd_coord;
            n_coord = bwd_coord;
        }
    }
    TT_FATAL(
        e_coord.has_value() || w_coord.has_value() || n_coord.has_value() || s_coord.has_value(),
        "No neighboring devices");
    // An even ring's forward route reaches N/2 devices and its backward route N/2-1, so one link direction carries
    // more. The writer then alternates packets with a route that swaps the two hop counts. Only on one ring axis of 4
    // or more devices: at 2 the swapped backward route has no hop, and two axes' 4 connections would need more packet
    // headers than the writer's RISC has.
    const bool load_balance_across_alt_routes = axis_load_balance && num_active_axes == 1;
    const uint32_t num_routes = load_balance_across_alt_routes ? 2 : 1;

    // Derive {E, W, N, S} physical directions, which depend on sender's mesh position + topology.
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    auto physical_dir = [&](const MeshCoordinate& neighbor) {
        const auto dir = tt::tt_fabric::get_eth_forwarding_direction(
            *mesh_device, sender_fabric_node_id, mesh_device->get_fabric_node_id(neighbor));
        TT_FATAL(
            dir.has_value() && static_cast<uint32_t>(dir.value()) < tt::tt_fabric::eth_chan_directions::Z,
            "AllGather expected a cardinal (E/W/N/S) fabric forwarding direction to a mesh neighbor");
        return static_cast<uint32_t>(dir.value());
    };
    uint32_t e_dir = 0, w_dir = 0, n_dir = 0, s_dir = 0;
    if (e_coord.has_value()) {
        e_dir = physical_dir(*e_coord);
    }
    if (w_coord.has_value()) {
        w_dir = physical_dir(*w_coord);
    }
    if (n_coord.has_value()) {
        n_dir = physical_dir(*n_coord);
    }
    if (s_coord.has_value()) {
        s_dir = physical_dir(*s_coord);
    }

    const uint32_t packet_size = operation_attributes.packet_size;

    ////////////////////////////////////////////////////////////////
    // Core selection
    ////////////////////////////////////////////////////////////////

    // We allocate one worker core per link, but not per axis.
    // Each worker handles both dirs (forward and backward) and also both axes (N/S and E/W).
    // Known limitation: when the two axes have unequal link counts, the larger axis's extra links
    // go unused. If this is ever a real use-case, we need to allocate separate worker cores per axis.
    const uint32_t links0 = operation_attributes.axis_num_links[0];
    const uint32_t links1 = operation_attributes.axis_num_links[1];
    uint32_t min_num_links = std::min(links0 > 0 ? links0 : links1, links1 > 0 ? links1 : links0);

    // Get worker cores
    uint32_t num_workers_per_link = 1;
    // Shrink core usage to fit available core grid.
    if (min_num_links * num_workers_per_link > num_available_cores) {
        const uint32_t fitted_links = num_available_cores / num_workers_per_link;
        TT_FATAL(
            fitted_links > 0,
            "all_gather needs at least {} worker core(s) but only {} are available; provide a larger sub_core_grid.",
            num_workers_per_link,
            num_available_cores);
        log_warning(
            tt::LogOp,
            "Using {} out of {} Fabric links due to limited {} worker cores. This may lead to performance loss.",
            fitted_links,
            min_num_links,
            num_available_cores);
        min_num_links = fitted_links;
    }
    // Row-major order may put every link's core on one NOC row. Unlike for unicast's link-directions, that is no
    // bottleneck: a multicast core moves far less than a row's NOC bandwidth.
    auto [worker_core_range, worker_cores] = ttnn::ccl::choose_worker_cores(
        min_num_links,
        num_workers_per_link,
        input_tensor.device(),
        operation_attributes.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        operation_attributes.sub_core_grid);
    TT_FATAL(
        worker_cores.size() == static_cast<size_t>(min_num_links) * num_workers_per_link,
        "all_gather needs {} worker cores ({} links x {} cores/link) but only {} are available; provide a larger "
        "sub_core_grid.",
        static_cast<size_t>(min_num_links) * num_workers_per_link,
        min_num_links,
        num_workers_per_link,
        worker_cores.size());

    const auto walk_geometry = compute_walk_geometry(
        input_tensor,
        output_tensor.tensor_spec(),
        operation_attributes.dim_from_end,
        num_devices,
        packet_size,
        /*reads_output=*/false);
    TT_FATAL(
        walk_geometry.has_value(),
        "all_gather multicast cannot cut {} B input pages and {} B output pages on the input's read alignment",
        input_tensor.mesh_buffer().page_size(),
        output_tensor.mesh_buffer().page_size());
    const all_gather_walk::WalkGeo& geo = *walk_geometry;
    const uint32_t unit_bytes = geo.unit_bytes;
    const uint32_t num_elements = num_walk_elements(geo, input_tensor);
    const uint32_t packet_cap = compute_packet_cap(geo, output_tensor);

    // --- CB sizing ---
    // A slot (all_gather_walk::slot_offsets): up to packets_per_slot packets with their records. The writer's headers
    // come from the packet header pool, so the slot holds none.
    const uint32_t packets_per_slot =
        compute_packets_per_slot(geo, packet_cap, min_num_links, /*header_bytes=*/0, input_tensor);
    const uint32_t slot_bytes = all_gather_walk::slot_offsets(packets_per_slot, unit_bytes, 0, packet_cap).slot_bytes;

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // Input CB: slots of records and payload.
    uint32_t cb0_id = tt::CB::c_in0;
    desc.cbs.push_back(slot_cb_descriptor(cb0_id, slot_bytes, worker_core_range));

    // KERNEL CREATION
    const auto writer_noc = writer_on_noc0 ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1;
    const auto reader_noc = writer_on_noc0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    // Reader (no fabric: reads the input, writes the local output, fills the CB)
    std::vector<uint32_t> reader_compile_args;
    append_walk_ct_args(reader_compile_args, geo, packet_cap);
    reader_compile_args.insert(reader_compile_args.end(), {packets_per_slot, cb0_id, slot_bytes});
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer (covers every direction: E-line, W-line, S-rect, N-rect)
    const uint32_t num_connections = (e_hops > 0) + (w_hops > 0) + (s_hops > 0) + (n_hops > 0);
    std::vector<uint32_t> writer_compile_args = {
        unit_bytes,
        packet_cap,
        packets_per_slot,
        cb0_id,
        slot_bytes,
        num_connections,
        static_cast<uint32_t>(load_balance_across_alt_routes),
    };

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = worker_core_range;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = reader_noc};
    desc.kernels.push_back(std::move(reader_kernel_desc));

    // Writer
    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = worker_core_range;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = writer_noc};
    desc.kernels.push_back(std::move(writer_kernel_desc));

    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 1;

    ////////////////////////////////////////////////////////////////
    // Runtime args
    ////////////////////////////////////////////////////////////////

    for (uint32_t link = 0; link < min_num_links; link++) {
        CoreCoord core = worker_cores[link];
        CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

        // Set runtime args
        const auto [slice_start, slice_count] = walk_slice(num_elements, min_num_links, link);

        // Every route reaches every other chip once:
        //   1D: e_hops + w_hops (or n_hops + s_hops) along the active axis = axis_size - 1.
        //   2D: every chip in the mesh outside me is covered by exactly one of the 4 mcast packets.
        // Both equal num_devices - 1. Each other chip sends each worker one init credit (route 0) and one done credit
        // per route.
        const uint32_t num_peers = num_devices - 1;
        const uint32_t num_done_credits = num_routes * num_peers;

        // Slots 0 and 1 are tensor addresses. They are placeholders here; emplace replaces them with Buffer*.
        std::vector<uint32_t> reader_rt_args = {
            0u,                  // input tensor address (Buffer* binding)
            0u,                  // output tensor address (Buffer* binding)
            slice_start,         // this worker's slice start (elements)
            slice_count,         // this worker's slice length (elements)
            device_idx,          // this device's index
            done_sem.address(),  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            num_done_credits,    // done credits to wait for
        };
        emplace_runtime_args_with_leading_buffers(
            desc.kernels[reader_kernel_id], core, {input_tensor.buffer(), output_tensor.buffer()}, reader_rt_args);

        std::vector<uint32_t> writer_rt_args = {
            slice_count,            // this worker's slice length (elements)
            barrier_sem.address(),  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            done_sem.address(),     // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            virtual_core.x,         // semaphore location (core.x)
            virtual_core.y,         // semaphore location (core.y)
            num_peers,              // init credits to wait for
            e_hops,                 // E line
            w_hops,                 // W line
            n_hops,                 // N spine
            s_hops,                 // S spine
            e_dir,
            w_dir,
            n_dir,
            s_dir,
        };
        // Writer connections, in the kernel's order: E-line, W-line (axis 1), then S-rect, N-rect (axis 0).
        std::vector<tt::tt_fabric::FabricNodeId> writer_dst_nodes;
        if (e_hops > 0 && e_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*e_coord));
        }
        if (w_hops > 0 && w_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*w_coord));
        }
        if (s_hops > 0 && s_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*s_coord));
        }
        if (n_hops > 0 && n_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*n_coord));
        }
        TT_FATAL(
            writer_dst_nodes.size() == num_connections, "AllGather multicast: a direction with hops has no neighbor");
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            *mesh_device,
            sender_fabric_node_id,
            writer_dst_nodes,
            {link},
            desc,
            writer_kernel_id,
            core,
            writer_rt_args,
            fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        desc.kernels[writer_kernel_id].runtime_args.emplace_back(core, std::move(writer_rt_args));
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllGatherMulticastFactory::create_workload_descriptor(
    const AllGatherParams& operation_attributes,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = get_available_worker_cores(operation_attributes, tensor_args.input_tensor);
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Kernel needs to wait to receive all remote data before exiting, and in some cases needs to wait
    // for all remote devices to be ready before beginning operation.
    // Since Fabric doesn't provide such capability within kernels, we need to manually sync using global semaphores.
    // Allocate the semaphore in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    const auto sem_buffer_type = ttnn::ccl::prefer_l1_small_buffer_type(*mesh_device);
    ttnn::ccl::warn_if_semaphores_fall_back_to_l1(sem_buffer_type);
    // One semaphore per phase: on a shared one, a fast device's next-launch startup credit could stand in for
    // a slow device's completion credit, and the op would exit before that device's data has landed.
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    const auto& barrier_sem = workload_descriptor.semaphores[0];
    const auto& done_sem = workload_descriptor.semaphores[1];
    log_debug(tt::LogOp, "Semaphore allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    workload_descriptor.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_multicast_descriptor_at(
            operation_attributes,
            coord,
            tensor_args,
            output_tensor,
            barrier_sem,
            done_sem,
            available_cores.num_cores());
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::operations::ccl
