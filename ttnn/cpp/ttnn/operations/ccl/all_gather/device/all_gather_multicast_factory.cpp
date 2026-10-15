// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_multicast_factory.hpp"
#include "all_gather_geometry.hpp"
#include "kernels/gather_args.hpp"

#include <algorithm>
#include <array>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

AllGatherMulticastFactory::cached_mesh_workload_t AllGatherMulticastFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    const auto avail =
        available_worker_cores(mesh_device, operation_attributes.subdevice_id, operation_attributes.sub_core_grid);

    // Kernel needs to wait to receive all remote data before exiting, and needs to wait for all remote devices
    // to be ready before beginning operation.
    const auto sems = create_semaphores_and_sync(mesh_device, avail.cores, avail.subdevice_id, 1);
    const auto& barrier_sem = sems[0];

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, output_tensor, barrier_sem, avail.cores.num_cores());
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherMulticastFactory::cached_program_t AllGatherMulticastFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem,
    uint32_t num_available_cores) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};

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
    bool ew_load_balance = false;
    bool ns_load_balance = false;

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

        // A load-balancing technique (alternating between two imbalanced routes) is used
        // in even-sized rings.
        const bool axis_load_balance = tt::tt_fabric::is_ring_or_torus(axis_topology) && (axis_size % 2 == 0);
        if (axis == 1) {
            e_hops = fwd_hops;
            w_hops = bwd_hops;
            e_coord = fwd_coord;
            w_coord = bwd_coord;
            ew_load_balance = axis_load_balance;
        } else {
            s_hops = fwd_hops;
            n_hops = bwd_hops;
            s_coord = fwd_coord;
            n_coord = bwd_coord;
            ns_load_balance = axis_load_balance;
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

    // Kernel alternates between ranges[] and ranges_alt[] hops on every packet send.
    // Enabled if any axis is an even-sized ring.
    const bool load_balance_across_alt_routes = ew_load_balance || ns_load_balance;

    // The kernels always run an init barrier, waiting for the remote output tensors to be allocated. It cannot
    // be skipped even for a persistent output buffer: that buffer may be reused across invocations of this CCL,
    // and the caller may write into it between invocations.

    const auto& tuning = operation_attributes.tuning;  // TEST HOOK (all_gather_tuning.hpp)
    const uint32_t packet_size = tuning.packet_size_override.value_or(operation_attributes.packet_size);

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

    // One worker core per link. Shrink the link count to fit the available core grid.
    constexpr uint32_t num_workers_per_link = 1;
    if (min_num_links > num_available_cores) {
        TT_FATAL(
            num_available_cores > 0,
            "all_gather needs at least 1 worker core but none are available; provide a larger sub_core_grid.");
        log_warning(
            tt::LogOp,
            "Using {} out of {} Fabric links due to limited {} worker cores. This may lead to performance loss.",
            num_available_cores,
            min_num_links,
            num_available_cores);
        min_num_links = num_available_cores;
    }
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
    // Page indexing (glossary and copy modes: all_gather_geometry.hpp, walk terms: kernels/gather_walk.hpp)
    //
    // GatherGeometry is in walk pages and independent of k, so the heuristics below key on it and k cannot
    // perturb one. WalkGeometry (k, stride, xfer, run sources, CB entry) is computed once from it and handed to
    // both kernels as CT args: they share one CB, so they must walk identically.
    ////////////////////////////////////////////////////////////////

    const auto arch = input_tensor.device()->arch();

    // --- Geometry: copy mode, walk pages, stripe (see all_gather_geometry.hpp) ---
    // Multicast never reads the output back, so unlike unicast it has no relay precondition.
    const auto geom =
        GatherGeometry::make(input_tensor, output_tensor.tensor_spec(), operation_attributes.dim_from_end);
    // The heuristics' keys, in walk pages / bytes (k-independent):
    const uint32_t stripe_pages = geom.chunks_per_stripe;  // walk pages per stripe row
    const uint32_t page_bytes = geom.chunk_size;           // walk page bytes

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // --- Per-arch tuning ---
    // Two CB entries is the floor on any arch: the reader runs one entry ahead of the sender, so a single
    // entry deadlocks.
    uint32_t cb_depth = 2;
    uint32_t packets_per_cb_entry = 1;
    uint32_t run_cap_bytes = 0;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        // The defaults above are WH's calibrated values:
        // - cb_depth 2: depth is also the reader's in-flight trid count, but deeper is perf-neutral here, so
        //   keep the L1.
        // - packets_per_cb_entry 1: packing several packets per entry amortises the reader/sender CB
        //   handshake, which is what the store-and-forward writer wants -- but here the reader sends the
        //   packets itself, off transaction ids and with no per-entry flush to stall on, so a one-packet entry
        //   just pipelines finer and wins on a line and on small row-major pages.
        // - No run cap: the best run length lands at the hardware ceiling (7616 B), so any value settable
        //   here is already above it and would only cost payload.
    } else if (arch == tt::ARCH::BLACKHOLE) {
        // Calibrated at 8 devices x 2 links, and at 1, 2 and 4 links. cb_depth and the run cap were both
        // already at their optimum -- every other value measured fell inside the run-to-run spread.
        cb_depth = 3;
        // packets_per_cb_entry: four amortises the reader/sender handshake and is right almost everywhere,
        // but not on a line carrying a long stripe: there the sender is already serialised per hop, and a
        // four-packet entry just delays the first send behind three more packet loads. A ring, sending in
        // both directions at once, wants the amortisation even at a long stripe, so the rule is line-only.
        // The stripe boundary does not move with volume.
        bool any_ring = false;
        for (uint32_t ax = 0; ax < 2; ++ax) {
            any_ring = any_ring || (operation_attributes.axis_num_devices[ax] > 1 &&
                                    tt::tt_fabric::is_ring_or_torus(operation_attributes.axis_topology[ax]));
        }
        // Keyed on walk pages (k-independent).
        const bool long_stripe = stripe_pages >= 32;
        // ...and only while the link count is low. With 4 links the per-hop send is no longer the
        // serialising step, so the amortisation a four-packet entry buys comes back.
        const bool few_links = min_num_links <= 2;
        packets_per_cb_entry = (!any_ring && long_stripe && few_links) ? 1 : 4;  // unicast differs here
        // Past ~8 KB a run gains little packet fill but keeps the walk in one DRAM bank longer. Only pages
        // big enough to still span a few runs are worth capping; below that the packet's own limit wins.
        // Keyed on the walk page (k-independent); the cap itself is bytes, so it bounds a run whatever k is,
        // and choose_k fills min(packet, cap) -- under this cap k stays 1 for pages >= 1 KB.
        run_cap_bytes = page_bytes >= 1024 ? 8192 : 0;
    }

    // TEST HOOK overrides (all_gather_tuning.hpp), after the per-arch defaults.
    run_cap_bytes = tuning.run_cap_bytes.value_or(run_cap_bytes);
    cb_depth = tuning.mcast_cb_depth.value_or(cb_depth);  // parse() enforces >= 2

    // --- Walk and CB sizing (see WalkGeometry::make) ---
    const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
    const auto walk = WalkGeometry::make(
        geom,
        *input_tensor.buffer(),
        *output_tensor.buffer(),
        packet_size,
        run_cap_bytes,
        WalkPolicy::from(tuning),
        packets_per_cb_entry,
        cb_depth,
        max_l1_space);
    const uint32_t cb_page_size = walk.cb_entry_bytes;
    log_debug(
        tt::LogOp,
        "all_gather multicast walk: k {} chunk {} slot {} stride {} xfer {} entry {} in_runs {}/{} out_runs {}/{}",
        walk.k,
        walk.chunk_size,
        walk.slot_bytes,
        walk.stride,
        walk.xfer,
        walk.chunks_per_entry,
        walk.in_runs.whole_page,
        walk.in_runs.cross_page,
        walk.out_runs.whole_page,
        walk.out_runs.cross_page);

    // Input CB
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);

    CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // KERNEL CREATION
    // CT args by name (kernels/gather_args.hpp); every count is in chunks (k included).
    namespace mcr = gather_args::multicast_reader;
    namespace mcw = gather_args::multicast_writer;

    // Reader (covers forward directions E-line + S-rect)
    std::vector<uint32_t> reader_compile_args(mcr::ct::tensor_args);
    reader_compile_args[mcr::ct::in_cpp] = walk.in_cpp;
    reader_compile_args[mcr::ct::chunk_size] = walk.chunk_size;
    reader_compile_args[mcr::ct::out_cpp] = walk.out_cpp;
    reader_compile_args[mcr::ct::cps] = walk.cps;
    reader_compile_args[mcr::ct::num_devices] = num_devices;
    reader_compile_args[mcr::ct::cb_id] = cb0_id;
    reader_compile_args[mcr::ct::cb_depth] = cb_depth;
    reader_compile_args[mcr::ct::cb_page_size] = cb_page_size;
    reader_compile_args[mcr::ct::packet_size] = packet_size;
    reader_compile_args[mcr::ct::alt_routes] = load_balance_across_alt_routes;
    reader_compile_args[mcr::ct::num_connections] = (e_hops > 0) + (s_hops > 0);
    reader_compile_args[mcr::ct::k] = walk.k;
    reader_compile_args[mcr::ct::walk_stride] = walk.stride;
    reader_compile_args[mcr::ct::walk_xfer] = walk.xfer;
    reader_compile_args[mcr::ct::slot_bytes] = walk.slot_bytes;
    reader_compile_args[mcr::ct::in_runs] = run_source_bits(walk.in_runs);
    reader_compile_args[mcr::ct::out_runs] = run_source_bits(walk.out_runs);
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer (covers backward directions W-line + N-rect)
    std::vector<uint32_t> writer_compile_args(mcw::ct::tensor_args);
    writer_compile_args[mcw::ct::chunk_size] = walk.chunk_size;
    writer_compile_args[mcw::ct::out_cpp] = walk.out_cpp;
    writer_compile_args[mcw::ct::cps] = walk.cps;
    writer_compile_args[mcw::ct::num_devices] = num_devices;
    writer_compile_args[mcw::ct::cb_id] = cb0_id;
    writer_compile_args[mcw::ct::cb_page_size] = cb_page_size;
    writer_compile_args[mcw::ct::packet_size] = packet_size;
    writer_compile_args[mcw::ct::alt_routes] = load_balance_across_alt_routes;
    writer_compile_args[mcw::ct::num_connections] = (w_hops > 0) + (n_hops > 0);
    writer_compile_args[mcw::ct::k] = walk.k;
    writer_compile_args[mcw::ct::walk_stride] = walk.stride;
    writer_compile_args[mcw::ct::walk_xfer] = walk.xfer;
    writer_compile_args[mcw::ct::slot_bytes] = walk.slot_bytes;
    writer_compile_args[mcw::ct::out_runs] = run_source_bits(walk.out_runs);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Writer
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_writer.cpp",
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    ////////////////////////////////////////////////////////////////
    // Runtime args
    ////////////////////////////////////////////////////////////////

    for (uint32_t link = 0; link < min_num_links; link++) {
        CoreCoord core = worker_cores[link];
        CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

        // This link's slice of input pages, in chunks (see GatherGeometry::slice_of, WalkGeometry::in_chunks).
        const Slice pages = geom.slice_of(link, min_num_links);
        const Slice slice = walk.in_chunks(pages);
        const uint32_t local_output_start = slice.first_chunk;
        const uint32_t num_worker_output_chunks = slice.num_chunks;
        // One rot for both kernels: they fill and drain one CB, so they must walk in the same order (direction 0;
        // the direction stagger is unicast-only).
        const uint32_t rot = walk.rot(pages, link, min_num_links, 0);

        // Per-link barrier fan-in = N-1 in every case. Every other chip sends me one atomic_inc:
        //   1D: e_hops + w_hops (or n_hops + s_hops) along the active axis = axis_size - 1.
        //   2D: every chip in the mesh outside me is covered by exactly one of the 4 mcast packets.
        // Both equal num_devices - 1.
        uint32_t barrier_wait_value = num_devices - 1;

        std::vector<uint32_t> reader_rt_args(mcr::rt::fabric_args);
        reader_rt_args[mcr::rt::input_addr] = input_tensor.buffer()->address();
        reader_rt_args[mcr::rt::output_addr] = output_tensor.buffer()->address();
        reader_rt_args[mcr::rt::slice_first] = local_output_start;
        reader_rt_args[mcr::rt::slice_chunks] = num_worker_output_chunks;
        reader_rt_args[mcr::rt::device_idx] = device_idx;  // initial stripe
        reader_rt_args[mcr::rt::barrier_sem] = barrier_sem.address();
        reader_rt_args[mcr::rt::barrier_noc_x] = virtual_core.x;
        reader_rt_args[mcr::rt::barrier_noc_y] = virtual_core.y;
        reader_rt_args[mcr::rt::barrier_wait_value] = barrier_wait_value;
        reader_rt_args[mcr::rt::rot] = rot;
        {
            // McastRouteArgs, in kernel read order: line then rect, as (hops, hops_alt, dir).
            const std::array<uint32_t, 12> route = {
                e_hops,                             // line_hops
                e_hops,                             // rect_e_hops
                w_hops,                             // rect_w_hops
                s_hops,                             // rect_spine_hops
                ew_load_balance ? w_hops : e_hops,  // line_hops_alt
                ew_load_balance ? w_hops : e_hops,  // rect_e_hops_alt
                ew_load_balance ? e_hops : w_hops,  // rect_w_hops_alt
                ns_load_balance ? n_hops : s_hops,  // rect_spine_hops_alt
                e_dir,                              // line_dir
                e_dir,                              // rect_e_dir
                w_dir,                              // rect_w_dir
                s_dir,                              // rect_spine_dir
            };
            std::copy(route.begin(), route.end(), reader_rt_args.begin() + mcr::rt::route_args);
        }
        // Reader forward connection info: E-line (axis 1) then S-rect (axis 0).
        std::vector<tt::tt_fabric::FabricNodeId> reader_dst_nodes;
        if (e_hops > 0 && e_coord.has_value()) {
            reader_dst_nodes.push_back(mesh_device->get_fabric_node_id(*e_coord));
        }
        if (s_hops > 0 && s_coord.has_value()) {
            reader_dst_nodes.push_back(mesh_device->get_fabric_node_id(*s_coord));
        }
        if (!reader_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                reader_dst_nodes,
                {link},
                program,
                reader_kernel_id,
                {core},
                reader_rt_args,
                fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        }

        std::vector<uint32_t> writer_rt_args(mcw::rt::fabric_args);
        writer_rt_args[mcw::rt::output_addr] = output_tensor.buffer()->address();
        writer_rt_args[mcw::rt::slice_first] = local_output_start;
        writer_rt_args[mcw::rt::slice_chunks] = num_worker_output_chunks;
        writer_rt_args[mcw::rt::device_idx] = device_idx;
        writer_rt_args[mcw::rt::barrier_sem] = barrier_sem.address();
        writer_rt_args[mcw::rt::barrier_noc_x] = virtual_core.x;
        writer_rt_args[mcw::rt::barrier_noc_y] = virtual_core.y;
        writer_rt_args[mcw::rt::rot] = rot;
        {
            const std::array<uint32_t, 12> route = {
                w_hops,                             // line_hops
                e_hops,                             // rect_e_hops
                w_hops,                             // rect_w_hops
                n_hops,                             // rect_spine_hops
                ew_load_balance ? e_hops : w_hops,  // line_hops_alt
                ew_load_balance ? w_hops : e_hops,  // rect_e_hops_alt
                ew_load_balance ? e_hops : w_hops,  // rect_w_hops_alt
                ns_load_balance ? s_hops : n_hops,  // rect_spine_hops_alt
                w_dir,                              // line_dir
                e_dir,                              // rect_e_dir
                w_dir,                              // rect_w_dir
                n_dir,                              // rect_spine_dir
            };
            std::copy(route.begin(), route.end(), writer_rt_args.begin() + mcw::rt::route_args);
        }
        // Writer backward connections: W-line (axis 1) then N-rect (axis 0).
        std::vector<tt::tt_fabric::FabricNodeId> writer_dst_nodes;
        if (w_hops > 0 && w_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*w_coord));
        }
        if (n_hops > 0 && n_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*n_coord));
        }
        if (!writer_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                writer_dst_nodes,
                {link},
                program,
                writer_kernel_id,
                {core},
                writer_rt_args,
                fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);
    }

    shared_variables_t shared_variables{
        .worker_cores = worker_cores,
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .barrier_sem = barrier_sem,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherMulticastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        const uint32_t barrier_sem_addr = shared_vars.barrier_sem.address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        namespace mcr = gather_args::multicast_reader;
        namespace mcw = gather_args::multicast_writer;
        for (const auto& core : shared_vars.worker_cores) {
            auto& reader_args = reader_args_by_core[core.x][core.y];
            reader_args[mcr::rt::input_addr] = input_addr;
            reader_args[mcr::rt::output_addr] = output_addr;
            reader_args[mcr::rt::barrier_sem] = barrier_sem_addr;
            auto& writer_args = writer_args_by_core[core.x][core.y];
            writer_args[mcw::rt::output_addr] = output_addr;
            writer_args[mcw::rt::barrier_sem] = barrier_sem_addr;
        }
    }
}

}  // namespace ttnn::operations::ccl
