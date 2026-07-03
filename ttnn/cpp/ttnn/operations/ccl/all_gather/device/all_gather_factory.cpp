// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_factory.hpp"

#include <map>
#include <set>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

// ─────────────────────────────────────────────────────────────────────────────
// Tunables (edit here). Hardcoded for now; op-level plumbing + heuristics are future work.
//   NUM_WORKERS_PER_LINK : worker (reader+writer) cores per direction per link. The forward and backward
//                          conveyors each get this many. >1 requires a fabric mux (see below).
//   MUX_NUM_BUFFERS      : L1 buffer slots per worker channel on the fabric mux.
//
// When NUM_WORKERS_PER_LINK == 1 the op takes the original single-conveyor path with a direct fabric
// connection (no mux). When > 1, each direction's workers share a fabric mux core that owns the single
// fabric connection and multiplexes their traffic onto the link.
// ─────────────────────────────────────────────────────────────────────────────
constexpr uint32_t NUM_WORKERS_PER_LINK = 4;
constexpr uint32_t MUX_NUM_BUFFERS = 2;

////////////////////////////////////////////////////////////////////////////////
// Store-and-forward AllGather (Fabric_1D line/ring only).
//
// Every device relays stripes to its immediate neighbor one hop at a time; a shard reaches far devices by
// being re-forwarded at each hop. Forward and backward conveyors run on separate cores. Each conveyor: the
// reader (CB producer, no fabric) reads iteration 0 from this device's own input and later iterations from
// the stripe the upstream neighbor deposited into our output; the writer (CB consumer) unicasts each stripe
// one hop to the neighbor's output (same address on every device).
//
// Direction/topology/load-balance are decided here and passed as runtime args, so the two kernels are
// compiled once and run on both core sets. Two semaphores: data_valid (per-iteration relay gate +
// completion, mirror-core targeted) and ready (init handshake, opposite-core targeted). Readers own every
// wait; writers only send.
////////////////////////////////////////////////////////////////////////////////

AllGatherFactory::cached_mesh_workload_t AllGatherFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(operation_attributes.sub_core_grid.value());
    }
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Neighbor-to-neighbor sync uses two per-core semaphores: data_valid (a stripe has been relayed in) and
    // ready (the init handshake). Allocate in L1_SMALL to avoid fragmenting the larger L1 pool.
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_info(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    auto data_valid_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto ready_sem = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, output_tensor, data_valid_sem, ready_sem);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherFactory::cached_program_t AllGatherFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& data_valid_sem,
    const tt::tt_metal::GlobalSemaphore& ready_sem) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();

    ////////////////////////////////////////////////////////////////
    // Topology (Fabric_1D line/ring only)
    ////////////////////////////////////////////////////////////////

    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());
    TT_FATAL(!fabric_is_2d, "store-and-forward all_gather supports Fabric_1D line/ring only, not Fabric_2D");

    // Exactly one active axis in 1D (the other has num_devices == 1).
    int active_axis = -1;
    for (uint32_t a = 0; a < 2; ++a) {
        if (operation_attributes.axis_num_devices[a] > 1) {
            TT_FATAL(active_axis == -1, "store-and-forward all_gather supports exactly one active axis (1D)");
            active_axis = static_cast<int>(a);
        }
    }
    TT_FATAL(active_axis != -1, "No neighboring devices");
    const uint32_t axis = static_cast<uint32_t>(active_axis);
    const auto topology = operation_attributes.axis_topology[axis];
    const bool is_ring = tt::tt_fabric::is_ring_or_torus(topology);

    const uint32_t num_devices = operation_attributes.num_devices;
    const uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    auto fwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, 1, topology, axis);
    auto bwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, -1, topology, axis);

    // Stripes a conveyor sends from a given device: ring -> N/2; line fwd -> d+1, bwd -> N-d; 0 at a dead
    // endpoint. Also queried for the downstream device below to decide granular vs single data_valid signalling.
    // The rest of each conveyor's schedule (num_recv, stripe_step, do_local_write, half-slice) is derived below.
    auto conveyor_iters = [&](uint32_t idx, bool is_forward) -> uint32_t {
        if (is_ring) {
            return num_devices / 2;
        }
        return is_forward ? (idx + 1 < num_devices ? idx + 1 : 0) : (idx > 0 ? num_devices - idx : 0);
    };
    const uint32_t fwd_iters = conveyor_iters(device_idx, true);
    const uint32_t bwd_iters = conveyor_iters(device_idx, false);
    TT_FATAL(fwd_iters > 0 || bwd_iters > 0, "device participates in neither direction");

    // Even ring: the last iteration sends a contiguous half of the stripe so the two conveyors split the
    // antipode slice (forward first half, backward second half). N==2 rings are remapped to line upstream, so
    // an even ring here has num_iters>=2 and the split never lands on iteration 0.
    const bool ring_even_split = is_ring && (num_devices % 2 == 0) && (num_devices >= 4);
    const bool do_init_barrier = !tensor_args.persistent_output_tensor.has_value();

    const uint32_t num_links = operation_attributes.axis_num_links[axis];

    ////////////////////////////////////////////////////////////////
    // Worker + mux cores.
    //
    // Each link runs two conveyors: forward (dir 0) and backward (dir 1). With NUM_WORKERS_PER_LINK == 1 each
    // conveyor is a single core connecting directly to its neighbor's ERISC. With NUM_WORKERS_PER_LINK > 1 the
    // workers of a direction cannot each open a direct connection (an ERISC exposes one worker sender channel
    // per direction), so they share a fabric mux: one dedicated mux core per direction per link owns the
    // single fabric connection and multiplexes the workers' traffic onto the link.
    //
    // Flat layout from choose_worker_cores(num_links, num_cores_per_link), per link:
    //   [dir 0: (mux?) worker 0 .. worker W-1][dir 1: (mux?) worker 0 .. worker W-1]
    ////////////////////////////////////////////////////////////////

    constexpr uint32_t num_directions = 2;  // 0 = forward, 1 = backward
    const uint32_t workers_per_dir = NUM_WORKERS_PER_LINK;
    const bool use_mux = workers_per_dir > 1;
    const uint32_t mux_per_dir = use_mux ? 1u : 0u;
    const uint32_t cores_per_dir = workers_per_dir + mux_per_dir;
    const uint32_t num_cores_per_link = num_directions * cores_per_dir;

    // all_core_range spans workers + mux; we drive kernels from the worker-only / mux-only subsets built below.
    [[maybe_unused]] auto [all_core_range, all_cores] = ttnn::ccl::choose_worker_cores(
        num_links,
        num_cores_per_link,
        mesh_device,
        operation_attributes.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        operation_attributes.sub_core_grid);
    TT_FATAL(
        all_cores.size() == static_cast<size_t>(num_links) * num_cores_per_link,
        "AllGather needs {} worker cores ({} links x {} cores/link) but only {} are available; reduce "
        "NUM_WORKERS_PER_LINK or provide a larger sub_core_grid.",
        static_cast<size_t>(num_links) * num_cores_per_link,
        num_links,
        num_cores_per_link,
        all_cores.size());

    // Indexing into the flat core vector (dir: 0 = forward, 1 = backward).
    auto core_at = [&](uint32_t link, uint32_t dir, uint32_t idx_in_dir) -> const CoreCoord& {
        return all_cores[(link * num_cores_per_link) + (dir * cores_per_dir) + idx_in_dir];
    };
    auto mux_core = [&](uint32_t link, uint32_t dir) -> const CoreCoord& { return core_at(link, dir, 0); };
    auto worker_core = [&](uint32_t link, uint32_t dir, uint32_t w) -> const CoreCoord& {
        return core_at(link, dir, mux_per_dir + w);
    };
    auto dir_neighbor = [&](uint32_t dir) { return dir == 0 ? fwd_coord : bwd_coord; };
    auto dir_active = [&](uint32_t dir) { return dir_neighbor(dir).has_value(); };

    // Reader/writer kernels + CB run on worker cores only; the mux kernel runs on its own cores (a mux is
    // created only for a direction that has a neighbor).
    std::vector<CoreCoord> worker_cores;
    worker_cores.reserve(static_cast<size_t>(num_links) * num_directions * workers_per_dir);
    std::set<CoreRange> worker_core_set;
    std::set<CoreRange> mux_core_set;
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t dir = 0; dir < num_directions; ++dir) {
            if (use_mux && dir_active(dir)) {
                mux_core_set.emplace(mux_core(link, dir));
            }
            for (uint32_t w = 0; w < workers_per_dir; ++w) {
                worker_cores.push_back(worker_core(link, dir, w));
                worker_core_set.emplace(worker_core(link, dir, w));
            }
        }
    }
    const CoreRangeSet worker_core_range(worker_core_set);
    const CoreRangeSet mux_core_range(mux_core_set);

    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    // Fabric mux config: one full-size channel per worker in a direction. Cheap to construct; only wired up
    // (kernel + args) when use_mux.
    tt::tt_fabric::FabricMuxConfig mux_config(
        /*num_full_size_channels=*/static_cast<uint8_t>(workers_per_dir),
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/static_cast<uint8_t>(MUX_NUM_BUFFERS),
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));

    tt::tt_metal::KernelHandle mux_kernel_id = 0;
    if (use_mux && mux_core_range.num_cores() > 0) {
        mux_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
            mux_core_range,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_config.get_fabric_mux_compile_time_args(),
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    }

    ////////////////////////////////////////////////////////////////
    // Page indexing (unchanged from the multicast op). See the OutputStripeIterator in common.hpp: the kernel
    // derives per-stripe page/byte/jump from {stripe, slice} and the CT geometry, so the host only supplies
    // the geometry constants and each worker's [local_output_start, num_worker_output_chunks) slice.
    //
    //   input page  -- one page of the input tensor.       output page -- one page of the output tensor.
    //   chunk       -- one NOC write = min(input_page, output_page) bytes.
    //   stripe      -- a device's contiguous run of chunks per row.
    //   Copy modes: matched (in==out), concat (out>in, chunks share a page), split (in>out).
    ////////////////////////////////////////////////////////////////

    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();

    auto input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }

    const uint32_t output_chunk_size = std::min(input_page_size, output_page_size);  // NOC write size
    const uint32_t output_chunks_per_page = std::max(1u, output_page_size / input_page_size);
    const uint32_t split_factor = std::max(1u, input_page_size / output_page_size);

    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t num_output_chunks = num_input_pages * split_factor;

    // CB sizing: cb_page_size is a multiple of both input_page_size and output_chunk_size.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    uint32_t cb_page_size = input_page_size * pages_per_packet;
    uint32_t cb_depth = 3;
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        // Pack multiple pages per CB page (fewer reader/writer syncs). Empirical multiplier, L1-clamped.
        const uint32_t ideal_multiplier = (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) ? 4 : 3;
        const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
        const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * cb_page_size), 1u, ideal_multiplier);
        if (multiplier < ideal_multiplier) {
            log_info(
                tt::LogOp,
                "CircularBuffer depth is reduced due to L1 pressure (only {} B available), performance may regress.",
                max_l1_space);
        }
        cb_page_size *= multiplier;
    }

    // Stripe = this device's contiguous run of chunks per row = input pages along [gather_dim..rank-1] * split.
    auto tile = (input_tensor.layout() == Layout::TILE) ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent;
        if (i == rank - 1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tile.get_width();
            } else {
                extent = (input_shape[i] * input_tensor.element_size()) / input_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == rank - 2) {
            extent = input_shape[i] / tile.get_height();
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }
    const uint32_t output_chunks_per_stripe = input_pages_per_stripe * split_factor;
    TT_FATAL(output_chunks_per_stripe > 0, "output_chunks_per_stripe must be > 0");

    ////////////////////////////////////////////////////////////////
    // Circular buffer + kernels (one reader + one writer, shared across both directions).
    ////////////////////////////////////////////////////////////////

    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);
    CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // data_valid is signalled once per this many CB pages so a downstream can start relaying a stripe before
    // the whole stripe has arrived. Tunable: larger = fewer syncs, smaller = finer pipelining.
    constexpr uint32_t data_valid_granularity = 4;

    std::vector<uint32_t> reader_ct = {
        input_page_size,
        output_chunk_size,
        output_chunks_per_page,
        output_chunks_per_stripe,
        num_devices,
        cb0_id,
        cb_page_size,
        do_init_barrier ? 1u : 0u,
        data_valid_granularity,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {
        output_chunk_size,
        output_chunks_per_page,
        output_chunks_per_stripe,
        num_devices,
        cb0_id,
        cb_page_size,
        packet_size,
        do_init_barrier ? 1u : 0u,
        data_valid_granularity,
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct);

    // When multiple workers share a fabric mux, the writer connects through it. Append the mux geometry (after
    // the tensor-accessor args) and flip the kernel onto its USE_WORKER_MUX path.
    std::map<std::string, std::string> writer_defines;
    if (use_mux) {
        ttnn::ccl::fabric_mux_connection_ct_args(
            workers_per_dir, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_config, writer_ct);
        writer_defines["USE_WORKER_MUX"] = "1";
    }

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct));
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/writer.cpp",
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_ct, writer_defines));

    ////////////////////////////////////////////////////////////////
    // Runtime args. The page split is now per (link, worker) -- num_links * workers_per_dir sub-slices -- and
    // is direction-independent, so the forward and backward conveyors of a given (link, worker) relay the same
    // stripe sub-slice. The per-worker/per-device core assignment is deterministic, so worker w's core has the
    // same coords on every device: data_valid signals target the mirror worker w on the neighbor.
    ////////////////////////////////////////////////////////////////

    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    // Mux kernel runtime args: one fabric connection per active direction per link, to that direction's
    // neighbor. The N workers of the direction all feed this one connection.
    if (use_mux) {
        for (uint32_t link = 0; link < num_links; ++link) {
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                if (!dir_active(dir)) {
                    continue;
                }
                const CoreCoord mc = mux_core(link, dir);
                const auto dst_node = mesh_device->get_fabric_node_id(*dir_neighbor(dir));
                auto mux_rt =
                    mux_config.get_fabric_mux_run_time_args(sender_fabric_node_id, dst_node, link, program, mc);
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mc}, mux_rt);
            }
        }
    }

    const uint32_t total_slices = num_links * workers_per_dir;
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t w = 0; w < workers_per_dir; ++w) {
            const uint32_t slice_idx = (link * workers_per_dir) + w;
            const uint32_t input_pages_per_slice = num_input_pages / total_slices;
            const uint32_t remainder = num_input_pages % total_slices;
            const uint32_t input_tile_id_start = (slice_idx * input_pages_per_slice) + std::min(slice_idx, remainder);
            const uint32_t input_tile_id_end =
                ((slice_idx + 1) * input_pages_per_slice) + std::min(slice_idx + 1, remainder);
            const uint32_t local_output_start = (input_tile_id_start * num_output_chunks) / num_input_pages;
            const uint32_t local_output_end = (input_tile_id_end * num_output_chunks) / num_input_pages;
            const uint32_t num_worker_output_chunks = local_output_end - local_output_start;
            const uint32_t half = num_worker_output_chunks / 2;

            // Configure one conveyor (dir: 0 = forward, 1 = backward). own_vc is this core's coords, reused as
            // the data_valid target on the neighbor's mirror core; partner_vc is the opposite-direction worker
            // (same index w), the ready target.
            auto set_conveyor = [&](uint32_t dir) {
                const bool is_forward = (dir == 0);
                const CoreCoord core = worker_core(link, dir, w);
                const CoreCoord partner = worker_core(link, 1 - dir, w);
                const CoreCoord own_vc = mesh_device->worker_core_from_logical_core(core);
                const CoreCoord partner_vc = mesh_device->worker_core_from_logical_core(partner);
                const auto neighbor = dir_neighbor(dir);

                const uint32_t stripe_step = is_forward ? num_devices - 1 : 1;
                const uint32_t num_iters = is_forward ? fwd_iters : bwd_iters;
                const uint32_t num_recv =
                    is_ring ? num_devices / 2 : (is_forward ? device_idx : num_devices - 1 - device_idx);
                const bool do_local_write = is_forward ? (fwd_iters > 0) : (fwd_iters == 0);

                // Signal granularly only for the stripes the downstream will actually relay (its first
                // downstream_iters - 1 received stripes); its antipode stripes get a single completion inc.
                uint32_t num_granular = 0;
                if (num_iters > 0) {
                    const uint32_t downstream_iters =
                        is_ring ? num_devices / 2
                                : conveyor_iters(is_forward ? device_idx + 1 : device_idx - 1, is_forward);
                    num_granular = downstream_iters > 0 ? downstream_iters - 1 : 0;
                }

                uint32_t final_start = local_output_start;
                uint32_t final_count = num_worker_output_chunks;
                if (ring_even_split) {
                    final_start = is_forward ? local_output_start : (local_output_start + half);
                    final_count = is_forward ? half : (num_worker_output_chunks - half);
                }

                std::vector<uint32_t> reader_rt = {
                    input_addr,
                    output_addr,
                    device_idx,
                    stripe_step,
                    num_iters,
                    num_recv,
                    local_output_start,
                    num_worker_output_chunks,
                    final_start,
                    final_count,
                    input_tile_id_start,
                    input_tile_id_end,
                    data_valid_sem.address(),
                    ready_sem.address(),
                };
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt);

                std::vector<uint32_t> writer_rt = {
                    output_addr,
                    device_idx,
                    stripe_step,
                    num_iters,
                    local_output_start,
                    num_worker_output_chunks,
                    final_start,
                    final_count,
                    do_local_write ? 1u : 0u,
                    data_valid_sem.address(),
                    ready_sem.address(),
                    (uint32_t)own_vc.x,
                    (uint32_t)own_vc.y,
                    (uint32_t)partner_vc.x,
                    (uint32_t)partner_vc.y,
                    num_granular,
                };
                if (num_iters > 0 && neighbor.has_value()) {
                    if (use_mux) {
                        // Connect this worker to its channel (== worker index w) on the direction's mux.
                        const CoreCoord mux_vc = mesh_device->worker_core_from_logical_core(mux_core(link, dir));
                        const CoreCoord term_master_vc =
                            mesh_device->worker_core_from_logical_core(worker_core(link, dir, 0));
                        ttnn::ccl::fabric_mux_connection_rt_args(
                            /*mux_connection_valid=*/true,
                            /*is_termination_master=*/w == 0,
                            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                            mux_vc,
                            /*worker_id=*/w,
                            core,
                            mux_config,
                            program,
                            term_master_vc,
                            writer_rt);
                    } else {
                        std::vector<tt::tt_fabric::FabricNodeId> dst = {mesh_device->get_fabric_node_id(*neighbor)};
                        append_routing_plane_connection_manager_rt_args(
                            sender_fabric_node_id,
                            dst,
                            {link},
                            program,
                            writer_kernel_id,
                            {core},
                            writer_rt,
                            tt::tt_fabric::FabricApiType::Linear);
                    }
                }
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt);
            };

            set_conveyor(/*dir=*/0);
            set_conveyor(/*dir=*/1);
        }
    }

    shared_variables_t shared_variables{
        .worker_cores = worker_cores,
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .data_valid_sem = data_valid_sem,
        .ready_sem = ready_sem,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        const uint32_t data_valid_addr = shared_vars.data_valid_sem.address();
        const uint32_t ready_addr = shared_vars.ready_sem.address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        for (const auto& core : shared_vars.worker_cores) {
            // reader: [0]=input_addr, [1]=output_addr, [12]=data_valid, [13]=ready
            auto& r = reader_args_by_core[core.x][core.y];
            r[0] = input_addr;
            r[1] = output_addr;
            r[12] = data_valid_addr;
            r[13] = ready_addr;
            // writer: [0]=output_addr, [9]=data_valid, [10]=ready
            auto& w = writer_args_by_core[core.x][core.y];
            w[0] = output_addr;
            w[9] = data_valid_addr;
            w[10] = ready_addr;
        }
    }
}

}  // namespace ttnn::operations::ccl
