// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_multicast_factory.hpp"

#include "all_gather_device_operation.hpp"
#include "all_gather_test_knobs.hpp"  // TEST ONLY

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
    TT_FATAL(input_tensor.buffer() != nullptr, "all_gather multicast requires an allocated input buffer");
    TT_FATAL(output_tensor.buffer() != nullptr, "all_gather multicast requires an allocated output buffer");

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    ////////////////////////////////////////////////////////////////

    auto* mesh_device = input_tensor.device();
    const uint32_t num_devices = operation_attributes.num_devices;
    uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    // Compute hops + neighbors for each mesh axis.
    // Each axis ∈ {0, 1} contributes a forward/backward pair: axis 1 -> (E=fwd, W=bwd),
    // axis 0 -> (S=fwd, N=bwd). In 1D only one axis is active; in 2D both can be.
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(operation_attributes.fabric_config);

    std::optional<MeshCoordinate> e_coord, w_coord, n_coord, s_coord;
    uint32_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;

    for (uint32_t axis = 0; axis < 2; ++axis) {
        const uint32_t axis_size = operation_attributes.axis_num_devices[axis];
        const bool is_axis_active = axis_size > 1;
        if (!is_axis_active) {
            continue;
        }
        // A true 2D collective cannot fall back to unicast, so factory selection cannot save us
        // here. No mesh view bends a true 2D axis today, so this only guards a future one.
        TT_FATAL(
            !fabric_is_2d || operation_attributes.axis_is_straight[axis],
            "AllGather 2D multicast needs mesh axis {} wired straight, but it turns corners",
            axis);

        const auto axis_topology = operation_attributes.axis_topology[axis];
        const uint32_t axis_index = sender_device_coord[axis];
        auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
            axis_size, axis_index, axis_topology, /*static_alternate=*/false);
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

    // Derive {E, W, N, S} physical directions, which depend on sender's mesh position + topology.
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    auto physical_slot = [&](const MeshCoordinate& neighbor) {
        const auto dir = tt::tt_fabric::get_eth_forwarding_direction(
            sender_fabric_node_id, mesh_device->get_fabric_node_id(neighbor));
        TT_FATAL(
            dir.has_value() && static_cast<uint32_t>(dir.value()) < tt::tt_fabric::eth_chan_directions::Z,
            "AllGather expected a cardinal (E/W/N/S) fabric forwarding direction to a mesh neighbor");
        return static_cast<uint32_t>(dir.value());
    };
    uint32_t e_dir = 0, w_dir = 0, n_dir = 0, s_dir = 0;
    if (e_coord.has_value()) {
        e_dir = physical_slot(*e_coord);
    }
    if (w_coord.has_value()) {
        w_dir = physical_slot(*w_coord);
    }
    if (n_coord.has_value()) {
        n_dir = physical_slot(*n_coord);
    }
    if (s_coord.has_value()) {
        s_dir = physical_slot(*s_coord);
    }

    // One route set: every packet, and the init and done credits, take the same routes (an even ring's antipode on
    // the forward one). The done credit then lands after the data it vouches for.
    // The init barrier waits for remote output tensors to be allocated, and for every other device to have finished
    // the previous launch: a persistent output may still be read there.

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
    // bottleneck: a core moves only ~14 GB/s (BH, 26 MB ring); one core per row is within 1% at 8-26 MB, line and ring.
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

    ////////////////////////////////////////////////////////////////
    // Page indexing
    //
    // The kernels walk this device's stripe of the output (all_gather_walk.hpp) in units of A bytes and pack the
    // runs that are contiguous in the output into packets of up to 4 segments. A walk element is one output page, or
    // one unit when a stripe row is not whole output pages; workers split the elements into contiguous slices.
    ////////////////////////////////////////////////////////////////

    const auto walk_geometry = compute_walk_geometry(
        input_tensor,
        output_tensor.tensor_spec(),
        operation_attributes.dim_from_end,
        num_devices,
        packet_size,
        /*unicast=*/false);
    TT_FATAL(
        walk_geometry.has_value(),
        "all_gather multicast cannot cut {} B input pages and {} B output pages on the input's read alignment",
        input_tensor.buffer()->page_size(),
        output_tensor.buffer()->page_size());
    const all_gather_walk::WalkGeo& geo = *walk_geometry;
    const uint32_t unit_bytes = geo.unit_bytes;
    const uint32_t num_elements = input_tensor.buffer()->num_pages() * geo.in_page_units / geo.element_units;
    const uint32_t packet_cap = compute_packet_cap(geo, output_tensor);

    // --- CB sizing ---
    // One CB page is a slot (all_gather_walk::slot_offsets): up to packets_per_slot packets with their records. The
    // writer's headers come from the packet header pool, so the slot holds none.
    const uint32_t packets_per_slot =
        compute_packets_per_slot(geo, packet_cap, min_num_links, /*header_bytes=*/0, input_tensor);
    const uint32_t cb_page_size = all_gather_walk::slot_offsets(packets_per_slot, unit_bytes, 0, packet_cap).slot_bytes;

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // Input CB. A slot is untyped bytes (records, headers, payload); the format only sizes the page.
    uint32_t cb0_id = tt::CB::c_in0;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = all_gather_walk::kNumSlots * cb_page_size,
        .core_ranges = worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb0_id),
            .data_format = tt::DataFormat::RawUInt8,
            .page_size = cb_page_size,
        }}},
    });

    // KERNEL CREATION
    // NOCs: the writer on NOC0 is 1.13x (line) to 1.42x (ring) faster (BH); the reader takes the other.
    const auto reader_noc = ag_test::env_noc("TT_AG_MC_RD_NOC", tt::tt_metal::NOC::NOC_1);  // TEST ONLY env
    const auto writer_noc = ag_test::env_noc("TT_AG_MC_WR_NOC", tt::tt_metal::NOC::NOC_0);  // TEST ONLY env
    if (ag_test::log_enabled() && device_idx == 0) {  // TEST ONLY; link 0 has the largest slice
        const uint64_t iter_packets = tt::div_up(
            uint64_t{tt::div_up(num_elements, min_num_links)} * geo.element_units * unit_bytes, uint64_t{packet_cap});
        log_info(
            tt::LogOp,
            "AGCFG factory=multicast arch={} topology={} num_devices={} num_links={} k={} packet_cap={} "
            "max_packet_bytes={} unit_bytes={} rd_noc={} wr_noc={} iter_packets={}",
            mesh_device->arch() == tt::ARCH::WORMHOLE_B0 ? "wh" : "bh",
            operation_attributes.is_true_2d() ? "2d"
            : tt::tt_fabric::is_ring_or_torus(operation_attributes.axis_topology[operation_attributes.get_1d_axis()])
                ? "ring"
                : "line",
            num_devices,
            min_num_links,
            packets_per_slot,
            packet_cap,
            geo.max_packet_bytes,
            unit_bytes,
            static_cast<int>(reader_noc),
            static_cast<int>(writer_noc),
            iter_packets);
    }
    // Reader (no fabric: reads the input, writes the local output, fills the CB)
    std::vector<uint32_t> reader_compile_args = {
        unit_bytes,            // cut quantum
        geo.in_page_units,     // input page footprint (units)
        geo.out_page_units,    // output page footprint (units)
        geo.stripe_units,      // units this device contributes to one output row
        num_devices,           // device count (stripe indexing)
        geo.element_units,     // units per walk element
        geo.max_packet_bytes,  // fabric packet payload limit
        geo.dst_cross,         // runs may continue across output pages
        geo.src_cross,         // runs may continue across input pages
        packet_cap,            // packet payload cap
        packets_per_slot,      // packets per CB page
        cb0_id,                // cb id
        cb_page_size,          // cb entry size
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer (covers every direction: E-line, W-line, S-rect, N-rect)
    const uint32_t num_connections = (e_hops > 0) + (w_hops > 0) + (s_hops > 0) + (n_hops > 0);
    std::vector<uint32_t> writer_compile_args = {
        unit_bytes,        // cut quantum
        packet_cap,        // packet payload cap
        packets_per_slot,  // packets per CB page
        cb0_id,            // cb id
        cb_page_size,      // cb entry size
        num_connections,   // num_connections
    };

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = worker_core_range;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.defines = ag_test::diag_defines();  // TEST ONLY
    reader_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = reader_noc};
    desc.kernels.push_back(std::move(reader_kernel_desc));

    // Writer
    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = worker_core_range;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.defines = ag_test::diag_defines();  // TEST ONLY
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
        const uint32_t elements_per_link = num_elements / min_num_links;
        const uint32_t remainder = num_elements % min_num_links;
        const uint32_t slice_start = (link * elements_per_link) + std::min(link, remainder);
        const uint32_t slice_count = elements_per_link + (link < remainder ? 1 : 0);

        // Per-link barrier fan-in = N-1 in every case. Every other chip sends me one atomic_inc:
        //   1D: e_hops + w_hops (or n_hops + s_hops) along the active axis = axis_size - 1.
        //   2D: every chip in the mesh outside me is covered by exactly one of the 4 mcast packets.
        // Both equal num_devices - 1.
        uint32_t barrier_wait_value = num_devices - 1;

        // Slots 0 and 1 are tensor addresses. They are placeholders here; emplace replaces them with Buffer*.
        std::vector<uint32_t> reader_rt_args = {
            0u,                  // input tensor address (Buffer* binding)
            0u,                  // output tensor address (Buffer* binding)
            slice_start,         // this worker's slice start (elements)
            slice_count,         // this worker's slice length (elements)
            device_idx,          // this device's index (its stripe)
            done_sem.address(),  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            barrier_wait_value,  // done credits to wait for
        };
        emplace_runtime_args_with_leading_buffers(
            desc.kernels[reader_kernel_id], core, {input_tensor.buffer(), output_tensor.buffer()}, reader_rt_args);

        std::vector<uint32_t> writer_rt_args = {
            slice_count,            // this worker's slice length (elements)
            barrier_sem.address(),  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            done_sem.address(),     // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
            virtual_core.x,         // semaphore location (core.x)
            virtual_core.y,         // semaphore location (core.y)
            barrier_wait_value,     // barrier counter to wait for
            e_hops,                 // e_hops (E line)
            w_hops,                 // w_hops (W line)
            n_hops,                 // n_hops (N spine)
            s_hops,                 // s_hops (S spine)
            e_dir,                  // e_dir
            w_dir,                  // w_dir
            n_dir,                  // n_dir
            s_dir,                  // s_dir
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
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(operation_attributes.sub_core_grid.value());
    }
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
