// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_unicast_factory.hpp"
#include "all_gather_geometry.hpp"
#include "all_gather_policy.hpp"
#include "kernels/gather_args.hpp"

#include <tt-metalium/kernel_types.hpp>  // for tt::tt_metal::NOC
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

////////////////////////////////////////////////////////////////
// Store-and-forward AllGather (line/ring over a single mesh axis; for Fabric 1D and 2D)
//
// Every device relays stripes to its neighbor one hop at a time; a shard reaches far devices by being
// re-forwarded at each hop. Forward and backward directions run on separate cores. Per direction: the reader
// (CB producer, no fabric) reads iteration 0 from local input and later iterations from what upstream relayed
// into our output; the writer (CB consumer) unicasts each stripe one hop to the neighbor's output (same
// address on every device). Direction/topology are runtime args, so both kernels compile once and run on all
// cores. Two semaphores: barrier_sem (init handshake) and data_valid_sem (relay gate + completion).
////////////////////////////////////////////////////////////////

AllGatherUnicastFactory::cached_mesh_workload_t AllGatherUnicastFactory::create_mesh_workload(
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
    // to be ready before beginning operation. Two semaphores, created in this order:
    // - barrier_sem: one-shot init handshake ("I'm alive") to the neighbor.
    // - data_valid_sem: chunks upstream has relayed into our output (relay gate + completion).
    const auto sems = create_semaphores_and_sync(mesh_device, avail.cores, avail.subdevice_id, 2);
    const auto& barrier_sem = sems[0];
    const auto& data_valid_sem = sems[1];

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            output_tensor,
            barrier_sem,
            data_valid_sem,
            avail.cores.num_cores());
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherUnicastFactory::cached_program_t AllGatherUnicastFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem,
    const tt::tt_metal::GlobalSemaphore& data_valid_sem,
    uint32_t num_available_cores) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    //
    // Glossary:
    //   relay          -- re-forward a received stripe from upstream one hop to downstream.
    //   slice          -- portion of tensor stripes allocated to this (link, worker)
    //   sink direction -- a direction that forwards nothing (num_iters == 0), e.g. a line endpoint's dead side.
    //   sink stripe    -- a stripe consumed here, not relayed onward (a line endpoint's incoming, or a ring
    //                     antipode).
    //   antipode       -- on a ring, the device N/2 hops away.
    ////////////////////////////////////////////////////////////////

    TT_FATAL(!operation_attributes.is_true_2d(), "all_gather unicast algorithm does not support true 2D topologies");

    const uint32_t axis = operation_attributes.get_1d_axis();
    const auto topology = operation_attributes.axis_topology[axis];
    const bool is_ring = tt::tt_fabric::is_ring_or_torus(topology);

    const uint32_t num_devices = operation_attributes.num_devices;
    const uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    auto fwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, 1, topology, axis);
    auto bwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, -1, topology, axis);

    // Stripes each direction sends from this device (gather_rules.hpp relay_iters; the full per-direction schedule
    // is unicast_schedule, below).
    const uint32_t fwd_iters = relay_iters(num_devices, is_ring, device_idx, /*is_forward=*/true);
    const uint32_t bwd_iters = relay_iters(num_devices, is_ring, device_idx, /*is_forward=*/false);
    TT_FATAL(fwd_iters > 0 || bwd_iters > 0, "device participates in neither direction");

    // The kernels always run an init barrier, waiting for the remote output tensors to be allocated. It cannot
    // be skipped even for a persistent output buffer: that buffer is also an input source (store-and-forward in
    // the relay iterations), and may be reused across invocations of this CCL.

    const uint32_t packet_size = operation_attributes.packet_size;

    ////////////////////////////////////////////////////////////////
    // Page indexing
    //
    // Glossary (shared with the kernels; page, walk page, chunk, stripe and the copy modes are in
    // all_gather_geometry.hpp, the walk terms in kernels/gather_walk.hpp):
    //   chunk id          -- a chunk's index in this device's contribution.
    //   seqno             -- a chunk's position in the emission order. This is what data_valid counts.
    //   segment           -- one scatter-list entry in a packet.
    //
    // GatherGeometry is in walk pages and independent of k, so the policy keys on it and k cannot perturb a
    // tier. WalkGeometry (k, stride, xfer, run sources, CB entry) is computed once from it and handed to both
    // kernels as CT args, so reader and writer cannot disagree.
    ////////////////////////////////////////////////////////////////

    const auto geom =
        GatherGeometry::make(input_tensor, output_tensor.tensor_spec(), operation_attributes.dim_from_end);

    // Relay reads our own output back into the CB, so every chunk's output offset must be NoC-read congruent
    // with its CB slot (gather_rules.hpp relay_read_ok). route_all_gather only picks unicast where that holds
    // or nothing is relayed; this is the precondition for any other way in.
    const bool relays = unicast_relays(operation_attributes);
    geom.require_relay_ok(relays);

    ////////////////////////////////////////////////////////////////
    // Core selection
    //
    // Each link runs two directions: forward (dir 0) and backward (dir 1). With workers_per_dir == 1 each
    // direction is a single core connected directly to its neighbor's ERISC. With > 1 the workers of a direction
    // can't each open a direct connection (an ERISC exposes one worker sender channel per direction), so they
    // share a fabric mux: one mux core per direction per link owns the connection and multiplexes their traffic.
    //
    // One (mux?, workers) group per direction per link, ordered link-major:
    //   group index = (link * num_directions) + dir
    ////////////////////////////////////////////////////////////////

    uint32_t num_links = operation_attributes.axis_num_links[axis];

    // --- Policy (all_gather_policy.cpp) ---
    // At the *requested* link count, so if the core grid later forces links down (which warns) the choices stay
    // the ones for the wider config.
    PolicyInputs policy_in;
    policy_in.arch = input_tensor.device()->arch();
    policy_in.is_ring = is_ring;
    policy_in.num_devices = num_devices;
    policy_in.num_links = num_links;
    policy_in.per_link_bytes = per_link_bytes(output_tensor.tensor_spec(), num_links);
    policy_in.stripe_pages = geom.walk_pages_per_stripe;
    policy_in.page_bytes = geom.walk_page_bytes;
    const UnicastPolicy policy = tune_unicast(policy_in);
    uint32_t workers_per_dir = policy.workers_per_dir;

    // Shrink core usage to fit available core grid. Shrink workers_per_link first, and then shrink num_links.
    auto cores_per_link = [](uint32_t workers) { return 2u * (workers + (workers > 1u ? 1u : 0u)); };
    const uint32_t wanted_workers = workers_per_dir;
    const uint32_t wanted_links = num_links;
    while (workers_per_dir > 1 && num_links * cores_per_link(workers_per_dir) > num_available_cores) {
        --workers_per_dir;
    }
    if (num_links * cores_per_link(workers_per_dir) > num_available_cores) {
        // Even one worker per direction per link does not fit; drop links (workers_per_dir is 1 by now).
        num_links = num_available_cores / cores_per_link(workers_per_dir);
        TT_FATAL(
            num_links > 0,
            "all_gather needs at least {} worker cores but only {} are available; provide a larger sub_core_grid.",
            cores_per_link(workers_per_dir),
            num_available_cores);
    }
    if (workers_per_dir != wanted_workers || num_links != wanted_links) {
        log_warning(
            tt::LogOp,
            "all_gather scaled down from {} links x {} workers/direction to {} links x {} workers/direction to fit "
            "the {} available worker cores. This may lead to performance loss.",
            wanted_links,
            wanted_workers,
            num_links,
            workers_per_dir,
            num_available_cores);
    }

    constexpr uint32_t num_directions = 2;  // 0 = forward, 1 = backward
    const bool use_mux = workers_per_dir > 1;

    const auto core_groups = ttnn::ccl::choose_mux_worker_cores(
        num_links * num_directions,
        workers_per_dir,
        mesh_device,
        operation_attributes.subdevice_id,
        operation_attributes.sub_core_grid);

    // Helpers to index into the core placement (dir: 0 = forward, 1 = backward).
    auto group_at = [&](uint32_t link, uint32_t dir) -> const ttnn::ccl::MuxWorkerGroup& {
        return core_groups[(link * num_directions) + dir];
    };
    auto mux_core = [&](uint32_t link, uint32_t dir) -> const CoreCoord& { return group_at(link, dir).mux.value(); };
    auto worker_core = [&](uint32_t link, uint32_t dir, uint32_t w) -> const CoreCoord& {
        return group_at(link, dir).workers[w];
    };
    auto dir_neighbor = [&](uint32_t dir) { return dir == 0 ? fwd_coord : bwd_coord; };
    auto dir_iters = [&](uint32_t dir) { return dir == 0 ? fwd_iters : bwd_iters; };

    // Reader/writer kernels + CB run on worker cores only; the mux kernels run on their own cores.
    std::vector<CoreCoord> worker_cores;
    worker_cores.reserve(num_links * num_directions * workers_per_dir);
    std::set<CoreRange> worker_core_set;
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t dir = 0; dir < num_directions; ++dir) {
            for (uint32_t w = 0; w < workers_per_dir; ++w) {
                worker_cores.push_back(worker_core(link, dir, w));
                worker_core_set.emplace(worker_core(link, dir, w));
            }
        }
    }
    const CoreRangeSet worker_core_range(worker_core_set);

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // --- Walk and CB sizing (see WalkGeometry::make) ---
    // k, xfer, run sources and the CB entry are chosen once here and passed to both kernels as CT args.
    const auto walk = WalkGeometry::make(geom, policy.shared, input_tensor, output_tensor, packet_size);
    // routing's relay rule was judged at k = 1; this is the chunk and slot the relay actually reads back.
    TT_FATAL(
        !relays || walk.relay_ok(geom),
        "all_gather unicast: the walk's {} B chunks ({} per output page, {} B slot, k={}) cannot be relayed from a {} "
        "B "
        "output page at {} B read alignment",
        walk.chunk_size,
        walk.out_cpp,
        walk.slot_bytes,
        walk.k,
        geom.out_aligned_page,
        geom.out_read_alignment);
    const uint32_t cb_depth = policy.shared.cb_depth;

    // Input and relay CB
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * walk.cb_entry_bytes, {{cb0_id, df}})
            .set_page_size(cb0_id, walk.cb_entry_bytes);
    CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // data_valid_granularity: data_valid is signalled once per this many CB entries so a downstream can start
    // relaying before the whole stripe arrives. Larger = fewer syncs, smaller = finer pipelining. Both divides
    // round up, so a slice gets policy.signals_per_stripe signals -- truncating them lands on more.
    const uint32_t total_slices = num_links * workers_per_dir;
    const uint32_t chunks_per_slice = std::max(1u, (geom.num_walk_pages * walk.k) / total_slices);
    const uint32_t entries_per_slice = std::max(1u, tt::div_up(chunks_per_slice, walk.chunks_per_entry));
    const uint32_t data_valid_granularity = std::max(1u, tt::div_up(entries_per_slice, policy.signals_per_stripe));

    // KERNEL CREATION
    // CT args by name (kernels/gather_args.hpp); every count is in chunks (k included).
    namespace ucr = gather_args::unicast_reader;
    namespace ucw = gather_args::unicast_writer;
    const uint32_t in_runs = run_source_bits(walk.in_runs);
    const uint32_t out_runs = run_source_bits(walk.out_runs);

    // Reader
    std::vector<uint32_t> reader_compile_args(ucr::ct::tensor_args);
    reader_compile_args[ucr::ct::in_cpp] = walk.in_cpp;
    reader_compile_args[ucr::ct::chunk_size] = walk.chunk_size;
    reader_compile_args[ucr::ct::out_cpp] = walk.out_cpp;
    reader_compile_args[ucr::ct::cps] = walk.cps;
    reader_compile_args[ucr::ct::num_devices] = num_devices;
    reader_compile_args[ucr::ct::cb_id] = cb0_id;
    reader_compile_args[ucr::ct::cb_entry_bytes] = walk.cb_entry_bytes;
    reader_compile_args[ucr::ct::k] = walk.k;
    reader_compile_args[ucr::ct::walk_stride] = walk.stride;
    reader_compile_args[ucr::ct::walk_xfer] = walk.xfer;
    reader_compile_args[ucr::ct::slot_bytes] = walk.slot_bytes;
    reader_compile_args[ucr::ct::in_runs] = in_runs;
    reader_compile_args[ucr::ct::out_runs] = out_runs;
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer
    std::vector<uint32_t> writer_compile_args(ucw::ct::tensor_args);
    writer_compile_args[ucw::ct::chunk_size] = walk.chunk_size;
    writer_compile_args[ucw::ct::out_cpp] = walk.out_cpp;
    writer_compile_args[ucw::ct::cps] = walk.cps;
    writer_compile_args[ucw::ct::num_devices] = num_devices;
    writer_compile_args[ucw::ct::cb_id] = cb0_id;
    writer_compile_args[ucw::ct::cb_entry_bytes] = walk.cb_entry_bytes;
    writer_compile_args[ucw::ct::packet_size] = packet_size;
    writer_compile_args[ucw::ct::data_valid_granularity] = data_valid_granularity;
    writer_compile_args[ucw::ct::k] = walk.k;
    writer_compile_args[ucw::ct::walk_stride] = walk.stride;
    writer_compile_args[ucw::ct::walk_xfer] = walk.xfer;
    writer_compile_args[ucw::ct::slot_bytes] = walk.slot_bytes;
    writer_compile_args[ucw::ct::out_runs] = out_runs;
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    std::map<std::string, std::string> writer_defines;
    if (use_mux) {
        writer_defines["USE_WORKER_MUX"] = "1";
    }

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_writer.cpp",
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, writer_defines));

    // Fabric mux
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    // The fabric maximum is also the smallest safe value here: our payload is the max payload, and an
    // undersized slot silently overruns the next one.
    const size_t channel_buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    tt::tt_fabric::FabricMuxV2Config mux_config(
        /*num_channels=*/static_cast<uint8_t>(workers_per_dir),
        /*num_buffers_per_channel=*/policy.mux_slots_per_channel,
        /*channel_buffer_size_bytes=*/channel_buffer_size_bytes,
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    if (use_mux) {
        // A mux owns exactly one downstream connection (dst node + link), so every dir and link needs its own.
        for (uint32_t link = 0; link < num_links; ++link) {
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                // MuxV2 only exits once every channel has been opened and closed, so a mux with a channel nobody
                // connects to hangs the op. Hence mux must be created under the same condition as worker connection.
                if (dir_iters(dir) == 0) {
                    continue;
                }
                const auto dst_node = mesh_device->get_fabric_node_id(*dir_neighbor(dir));
                // Forwarder on NOC 0: our writer pushes into the mux on NOC 1, and sharing one NOC makes
                // them contend.
                tt::tt_fabric::add_fabric_mux_v2_to_program(
                    program,
                    mux_config,
                    mux_core(link, dir),
                    sender_fabric_node_id,
                    dst_node,
                    link,
                    tt::tt_metal::NOC::NOC_0);
            }
        }
    }

    ////////////////////////////////////////////////////////////////
    // Runtime args
    //
    // The page split is per (link, worker) -- num_links * workers_per_dir slices -- and direction-independent,
    // so a (link, worker)'s forward and backward directions relay the same slice. Core assignment is
    // deterministic, so worker w has the same coords on every device: data_valid signals target the mirror
    // worker w on the neighbor.
    ////////////////////////////////////////////////////////////////

    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t w = 0; w < workers_per_dir; ++w) {
            const uint32_t slice_idx = (link * workers_per_dir) + w;
            // This slice of input pages, in chunks (see GatherGeometry::slice_of, WalkGeometry::chunks_of).
            const ChunkSlice slice = walk.chunks_of(geom.slice_of(slice_idx, total_slices));

            // Both directions (dir: 0 = forward, 1 = backward). mirror_core is this core's coords, reused as the
            // data_valid_sem target on the neighbor's mirror core; partner_core is the opposite-direction worker
            // (same index w), the barrier_sem target.
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                const bool is_forward = (dir == 0);
                const CoreCoord core = worker_core(link, dir, w);
                const CoreCoord partner = worker_core(link, 1 - dir, w);
                const CoreCoord mirror_core = mesh_device->worker_core_from_logical_core(core);
                const CoreCoord partner_core = mesh_device->worker_core_from_logical_core(partner);
                const auto neighbor = dir_neighbor(dir);

                // Iterations, local write, data_valid cadence, the even-ring split and the completion count
                // (gather_rules.hpp unicast_schedule; simulated end to end by test_all_gather_schedule.cpp).
                const UnicastSchedule sched =
                    unicast_schedule(num_devices, is_ring, device_idx, is_forward, slice.count);
                // Where the last iteration starts: the host seeks, the kernel never does.
                const WalkPos final_pos =
                    walk_seek(slice.first, slice.count, sched.final_skip, walk.stride, walk.xfer, walk.k);

                std::vector<uint32_t> reader_rt_args(ucr::rt::count);
                reader_rt_args[ucr::rt::input_addr] = input_addr;
                reader_rt_args[ucr::rt::output_addr] = output_addr;
                reader_rt_args[ucr::rt::initial_stripe] = device_idx;
                reader_rt_args[ucr::rt::stripe_step] = sched.stripe_step;
                reader_rt_args[ucr::rt::num_iters] = sched.num_iters;
                reader_rt_args[ucr::rt::total_chunks] = sched.total_chunks;
                reader_rt_args[ucr::rt::slice_first] = slice.first;
                reader_rt_args[ucr::rt::slice_count] = slice.count;
                reader_rt_args[ucr::rt::final_skip] = sched.final_skip;
                reader_rt_args[ucr::rt::final_take] = sched.final_take;
                reader_rt_args[ucr::rt::final_pos_base] = final_pos.base;
                reader_rt_args[ucr::rt::final_pos_lane] = final_pos.lane;
                reader_rt_args[ucr::rt::final_pos_i] = final_pos.i;
                reader_rt_args[ucr::rt::barrier_sem] = barrier_sem.address();
                reader_rt_args[ucr::rt::data_valid_sem] = data_valid_sem.address();
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                // route for Fabric_2D
                const auto route_node =
                    neighbor.has_value() ? mesh_device->get_fabric_node_id(*neighbor) : sender_fabric_node_id;
                std::vector<uint32_t> writer_rt_args(ucw::rt::fabric_args);
                writer_rt_args[ucw::rt::output_addr] = output_addr;
                writer_rt_args[ucw::rt::initial_stripe] = device_idx;
                writer_rt_args[ucw::rt::stripe_step] = sched.stripe_step;
                writer_rt_args[ucw::rt::num_iters] = sched.num_iters;
                writer_rt_args[ucw::rt::slice_first] = slice.first;
                writer_rt_args[ucw::rt::slice_count] = slice.count;
                writer_rt_args[ucw::rt::final_skip] = sched.final_skip;
                writer_rt_args[ucw::rt::final_take] = sched.final_take;
                writer_rt_args[ucw::rt::final_pos_base] = final_pos.base;
                writer_rt_args[ucw::rt::final_pos_lane] = final_pos.lane;
                writer_rt_args[ucw::rt::final_pos_i] = final_pos.i;
                writer_rt_args[ucw::rt::do_local_write] = sched.do_local_write ? 1u : 0u;
                writer_rt_args[ucw::rt::barrier_sem] = barrier_sem.address();
                writer_rt_args[ucw::rt::data_valid_sem] = data_valid_sem.address();
                writer_rt_args[ucw::rt::barrier_noc_x] = partner_core.x;  // neighbor partner core
                writer_rt_args[ucw::rt::barrier_noc_y] = partner_core.y;
                writer_rt_args[ucw::rt::data_valid_noc_x] = mirror_core.x;  // neighbor mirror core
                writer_rt_args[ucw::rt::data_valid_noc_y] = mirror_core.y;
                writer_rt_args[ucw::rt::num_granular_sends] = sched.num_granular;  // leading sends downstream relays
                writer_rt_args[ucw::rt::neighbor_chip_id] = route_node.chip_id;    // packet header 2D route
                writer_rt_args[ucw::rt::neighbor_mesh_id] = *route_node.mesh_id;
                TT_FATAL(sched.num_iters == 0 || neighbor.has_value(), "an active direction must have a neighbor");
                if (sched.num_iters > 0) {
                    if (use_mux) {
                        const CoreCoord mux_vc = mesh_device->worker_core_from_logical_core(mux_core(link, dir));
                        const auto flow_control_sem_id = tt::tt_metal::CreateSemaphore(program, core, 0);
                        const auto teardown_sem_id = tt::tt_metal::CreateSemaphore(program, core, 0);
                        mux_config.append_client_connection_rt_args(
                            mux_vc,
                            /*logical_channel_id=*/static_cast<uint8_t>(w),
                            tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
                                .flow_control_sem_id = flow_control_sem_id,
                                .teardown_sem_id = teardown_sem_id,
                            },
                            writer_rt_args);
                    } else {
                        std::vector<tt::tt_fabric::FabricNodeId> dst = {mesh_device->get_fabric_node_id(*neighbor)};
                        append_routing_plane_connection_manager_rt_args(
                            sender_fabric_node_id,
                            dst,
                            {link},
                            program,
                            writer_kernel_id,
                            {core},
                            writer_rt_args,
                            tt::tt_fabric::FabricApiType::Linear);
                    }
                }
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);
            }
        }
    }

    shared_variables_t shared_variables{
        .worker_cores = worker_cores,
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .barrier_sem = barrier_sem,
        .data_valid_sem = data_valid_sem,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherUnicastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        const uint32_t barrier_addr = shared_vars.barrier_sem.address();
        const uint32_t data_valid_addr = shared_vars.data_valid_sem.address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        namespace ucr = gather_args::unicast_reader;
        namespace ucw = gather_args::unicast_writer;
        for (const auto& core : shared_vars.worker_cores) {
            auto& reader_args = reader_args_by_core[core.x][core.y];
            reader_args[ucr::rt::input_addr] = input_addr;
            reader_args[ucr::rt::output_addr] = output_addr;
            reader_args[ucr::rt::barrier_sem] = barrier_addr;
            reader_args[ucr::rt::data_valid_sem] = data_valid_addr;
            auto& writer_args = writer_args_by_core[core.x][core.y];
            writer_args[ucw::rt::output_addr] = output_addr;
            writer_args[ucw::rt::barrier_sem] = barrier_addr;
            writer_args[ucw::rt::data_valid_sem] = data_valid_addr;
        }
    }
}

}  // namespace ttnn::operations::ccl
