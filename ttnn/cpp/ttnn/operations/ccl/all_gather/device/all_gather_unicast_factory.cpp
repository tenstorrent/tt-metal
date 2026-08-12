// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_unicast_factory.hpp"

#include <tt-metalium/kernel_types.hpp>  // for tt::tt_metal::NOC
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

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
    // Two semaphores:
    // - barrier_sem: one-shot init handshake ("I'm alive") to the neighbor.
    // - data_valid_sem: chunks upstream has relayed into our output (relay gate + completion).
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    auto barrier_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto data_valid_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            output_tensor,
            barrier_sem,
            data_valid_sem,
            available_cores.num_cores());
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

    // Stripes a direction sends from a device: ring -> N/2; line fwd -> d+1, bwd -> N-d; 0 at a dead endpoint.
    // Also queried for the downstream device to choose granular vs single data_valid signalling.
    auto relay_iters = [&](uint32_t idx, bool is_forward) -> uint32_t {
        if (is_ring) {
            return num_devices / 2;
        }
        return is_forward ? (idx + 1 < num_devices ? idx + 1 : 0) : (idx > 0 ? num_devices - idx : 0);
    };
    const uint32_t fwd_iters = relay_iters(device_idx, true);
    const uint32_t bwd_iters = relay_iters(device_idx, false);
    TT_FATAL(fwd_iters > 0 || bwd_iters > 0, "device participates in neither direction");

    // Even-sized ring: for load balancing, the antipode device receives the antipode stripe as halves from both
    // forward and backward directions.
    const bool ring_even_split = is_ring && (num_devices % 2 == 0);
    const bool do_init_barrier = !tensor_args.persistent_output_tensor.has_value();

    const uint32_t packet_size = operation_attributes.packet_size;

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

    // Num worker cores per direction per link. >1 requires an additional fabric mux core to own the fabric
    // connection and multiplex traffic. Values below are per-arch sweep results.
    //
    // The op is transaction-rate bound, not bandwidth bound: measured time tracks the packet count, so what a
    // worker count has to do is keep the link's packet pipeline full. That gives three regimes, and the
    // thresholds below are in bytes *per link* so they carry to other link counts.
    //   tiny   -- the mux is a store-and-forward hop, and below ~320 KB/link its fixed cost outweighs the
    //             throughput it buys, so connect the single worker straight to the ERISC instead.
    //   middle -- one worker cannot keep the link fed; two can, and a third only adds contention.
    //   large  -- only reached on a ring, with pages big enough to fill a packet. There the per-packet payload
    //             is large enough that the downstream drains faster than two workers can issue, so a third
    //             pays. With small pages the packet rate saturates downstream first and the third worker again
    //             only contends, hence the page term. A line never wants a third: measured at 48 MB it costs
    //             1.6%, since its middle links carry more and saturate downstream at two workers already.
    uint32_t num_links = operation_attributes.axis_num_links[axis];
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const auto arch = input_tensor.device()->arch();
    const uint64_t gathered_bytes =
        static_cast<uint64_t>(input_tensor.physical_volume()) * input_tensor.element_size() * num_devices;
    const uint64_t per_link_bytes = gathered_bytes / std::max(1u, num_links);
    uint32_t workers_per_dir = 1;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        // Two workers saturate a link: one cannot keep it fed, and past two they only contend on the NOC.
        // TODO(perf): re-sweep on Wormhole. The Blackhole regimes below came out of a sweep this branch has
        // not had, and two of the three have nothing arch-specific about them: skipping the mux under a size
        // threshold, and sizing the CB page off the stripe. Wormhole's smaller packet (7616 max) and single
        // link should move the thresholds, not remove them. Left at the old flat value until measured.
        workers_per_dir = 2;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        if (per_link_bytes < 320 * 1024) {
            workers_per_dir = 1;
        } else if (is_ring && per_link_bytes >= 16 * 1024 * 1024 && input_page_size >= 2048) {
            workers_per_dir = 3;
        } else {
            workers_per_dir = 2;
        }
    }

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
    // Page indexing
    //
    // Glossary:
    //   input page     -- one page of the input tensor.
    //   output page    -- one page of the output tensor (the real buffer page).
    //   chunk          -- one NOC write = min(input_page, output_page) bytes. An input
    //                     page = split_factor chunks; an output page = output_chunks_per_page
    //                     chunks. The kernel iterator walks chunks.
    //   stripe         -- a run of consecutive chunks this device writes before
    //                     jumping past other devices' contributions.
    //   stripe jump    -- value the kernel adds to output_page_id at the stripe
    //                     boundary.
    //
    // Three copy modes, picked by input vs output page sizes:
    //   matched (in == out): 1 chunk per input page, output_chunks_per_page = 1.
    //   concat  (out > in) : 1 chunk per input page, output_chunks_per_page > 1; each
    //                        chunk lands at a byte offset within a shared output page.
    //   split   (in > out) : split_factor chunks per input page, output_chunks_per_page = 1.
    //
    // Kernel is a dumb chunk iterator. Iteration pattern is:
    //   byte_offset++ within an output page -> chunk++ -> stripe+=jump
    //
    // Host supplies the requisite geometry constants + each worker's slice; the kernel's
    // OutputStripeIterator derives the remaining iterator parameters at compile-time.
    ////////////////////////////////////////////////////////////////

    const auto& input_shape = input_tensor.padded_shape();

    // --- Copy mode ---
    // The kernel always reads whole *aligned* input pages into L1 (required by the input's NoC
    // read alignment, DRAM or L1) but writes at output *content* (unaligned) granularity, so
    // chunk sizing differs by mode:
    //   matched (in == out): 1 chunk per input page, output_chunks_per_page = 1.
    //   concat  (out > in) : 1 chunk per input page, output_chunks_per_page > 1; each chunk
    //                        lands at a byte offset within a shared output page.
    //   split   (in > out) : split_factor chunks per input page, output_chunks_per_page = 1.
    const uint32_t input_unaligned_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_unaligned_page_size = output_tensor.buffer()->page_size();
    // matched/concat write a whole aligned input page (== L1 read stride) into an output slot;
    // split writes output-content-sized pieces to separate output page bases.
    const bool is_split = input_unaligned_page_size > output_unaligned_page_size;
    const uint32_t output_chunk_size = is_split ? output_unaligned_page_size : input_page_size;
    const uint32_t output_chunks_per_page = is_split ? 1u : output_unaligned_page_size / input_unaligned_page_size;
    const uint32_t split_factor = is_split ? input_unaligned_page_size / output_unaligned_page_size : 1u;
    TT_FATAL(
        output_chunks_per_page == 1 || input_page_size == input_unaligned_page_size,
        "concat requires an unpadded input page");  // so slots align to content

    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t num_output_chunks = num_input_pages * split_factor;
    TT_FATAL(
        num_output_chunks / split_factor == num_input_pages,
        "all_gather output chunk count overflowed uint32: {} input pages x split factor {}",
        num_input_pages,
        split_factor);

    ::ttnn::ccl::validate_packet_size(arch, packet_size, output_chunk_size);

    // --- Stripe geometry ---
    // input_pages_per_stripe = num input pages along [gather dim .. last dim] this
    // device contributes per stripe. For a last-dim RM gather this is the *page* count,
    // which handles sharded RM input (> 1 input page per row).
    auto tile_spec = input_tensor.layout() == Layout::TILE ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = operation_attributes.dim_from_end; i < 0; i++) {
        uint32_t extent;
        if (i == -1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tile_spec.get_width();
            } else {
                // This is a page count, so divide by the unaligned page size, not aligned
                extent = (input_shape[i] * input_tensor.element_size()) / input_unaligned_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == -2) {
            extent = input_shape[i] / tile_spec.get_height();
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }

    // Stripe = this device's contiguous run of chunks per row = input_pages_per_stripe
    // * split_factor. Measured in chunks (not output pages) so multi-shard concat works:
    // a stripe's chunks are laid across output pages via the inner byte-offset counter
    // and may straddle pages.
    const uint32_t output_chunks_per_stripe = input_pages_per_stripe * split_factor;
    TT_FATAL(output_chunks_per_stripe > 0, "output_chunks_per_stripe must be > 0");

    // --- CB sizing ---
    // cb_page_size is a multiple of input_page_size, which is itself a multiple of
    // output_chunk_size = min(input, output), so the kernel increments both
    // the cb_read_ptr and cb_write_ptr cleanly.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    uint32_t cb_page_size = input_page_size * pages_per_packet;
    uint32_t cb_depth = 3;
    // Perf hack: pack multiple pages into a single CB page to reduce CB sync frequency between reader and
    // writer. Note this increases effective CB depth. Row-major is safe too: an integer multiplier preserves
    // the multiple-of-input_page_size property above.
    //
    // The multiplier trades sync frequency against pipeline granularity, and a CB page of one to two packets
    // beats the wider ones: bigger coarsens the relay pipeline (a downstream device waits longer for its
    // first batch) without buying back enough sync savings.
    //
    // In the three-worker regime the choice is set by the stripe instead. A short stripe means the chunk
    // iterator jumps to a far-away output page every few chunks, so a wide CB page batches many of those
    // jumps into one burst of scattered writes; a long stripe is contiguous, so a wide CB page is one long
    // run and DRAM likes it. Confirmed by a controlled A/B: the same tensor, layout and page size gathered on
    // dim 2 (one long stripe) wants 2 and on dim 3 (short stripes) wants 1, each by 2-4%.
    //
    // Two workers have their own exception: a narrow band either side of ~1.5 MB gathered where 4 beats 2 by
    // 3-16%. That one resisted explanation -- not tensor size, stripe width or pages per worker; two shapes
    // with identical page counts land on opposite sides of it. Encoded as measured rather than rationalised;
    // if it has to go, 4 everywhere for two workers is the no-regression fallback and costs 2-8% at 1-20 MB.
    // TODO(perf): root-cause the band. Most likely a DRAM access-pattern effect of the page-id stride, so
    // the thing to look at is the bank sequence the stripe jump produces, not another size threshold. A
    // magic size window is a maintenance hazard and should be replaced by whatever actually drives it.
    constexpr uint32_t short_stripe_chunks = 256;
    const uint32_t bh_multiplier = workers_per_dir >= 3   ? (output_chunks_per_stripe < short_stripe_chunks ? 1 : 2)
                                   : workers_per_dir == 1 ? 2
                                   : (per_link_bytes >= 600 * 1024 && per_link_bytes < 1024 * 1024) ? 4
                                                                                                    : 2;
    const uint32_t ideal_multiplier = (arch == tt::ARCH::BLACKHOLE) ? bh_multiplier : 3;
    const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
    const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * cb_page_size), 1u, ideal_multiplier);
    if (multiplier < ideal_multiplier) {
        log_warning(
            tt::LogOp,
            "CircularBuffer depth reduced due to L1 pressure (only {} B available), performance may regress.",
            max_l1_space);
    }
    cb_page_size *= multiplier;

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // Input and relay CB
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);
    CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // data_valid_granularity:
    // data_valid is signalled once per this many CB pages so a downstream can start relaying before the whole
    // stripe arrives. Larger = fewer syncs, smaller = finer pipelining.
    // This is a minor perf knob, below heuristic was determined from extensive test sweeps.
    // Auto-selected to half the per-worker stripe: enough pipelining without the over-signalling that hurts
    // small-page tensors at scale. Kept as a fraction of the stripe so it self-scales with tensor size, links,
    // and workers.
    const uint32_t total_slices = num_links * workers_per_dir;
    const uint32_t outputs_per_cb_page = std::max(1u, cb_page_size / output_chunk_size);
    const uint32_t cb_pages_per_stripe = std::max(1u, (num_output_chunks / total_slices) / outputs_per_cb_page);
    const uint32_t data_valid_granularity = std::max(1u, cb_pages_per_stripe / 2u);

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        input_page_size,           // input tensor page size
        output_chunk_size,         // NOC write size = min(input, output)
        output_chunks_per_page,    // chunks per output page (1 unless concat)
        output_chunks_per_stripe,  // stripe length in chunks
        num_devices,               // device count (stripe indexing)
        cb0_id,                    // cb id
        cb_page_size,              // cb entry size
        do_init_barrier,           // wait for remote output allocation before relaying
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Mux slots per channel. Shared with the writer kernel, which needs it at compile time to fold its slot
    // wrap-around; the two must agree or the mux's flow control breaks.
    // A single buffer stalls the worker on every credit round-trip to the forwarder, so one is never right:
    // it costs 30-60%. Re-swept against the writer's header ring, which had invalidated the previous result:
    // 2 through 16 are all within noise of each other, so the floor is still the only thing that matters.
    constexpr uint8_t num_buffers_per_channel = 2;

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        output_chunk_size,         // NOC write size = min(input, output)
        output_chunks_per_page,    // chunks per output page (1 unless concat)
        output_chunks_per_stripe,  // stripe length in chunks
        num_devices,               // device count (stripe indexing)
        cb0_id,                    // cb id
        cb_page_size,              // cb entry size
        packet_size,               // packet_size
        do_init_barrier,           // send init handshake before relaying
        data_valid_granularity,    // signal data_valid once per this many CB pages
        num_buffers_per_channel,   // mux slots per channel (mux path only)
    };
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
        /*num_buffers_per_channel=*/num_buffers_per_channel,
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
            const uint32_t input_pages_per_slice = num_input_pages / total_slices;
            const uint32_t remainder = num_input_pages % total_slices;
            const uint32_t input_tile_id_start = (slice_idx * input_pages_per_slice) + std::min(slice_idx, remainder);
            const uint32_t input_tile_id_end =
                ((slice_idx + 1) * input_pages_per_slice) + std::min(slice_idx + 1, remainder);
            // Map this slice of input pages to its slice of output chunks. num_output_chunks is
            // num_input_pages * split_factor, so the map is just a scale by split_factor (1 in
            // matched/concat).
            const uint32_t local_output_start = input_tile_id_start * split_factor;
            const uint32_t local_output_end = input_tile_id_end * split_factor;
            const uint32_t num_worker_output_chunks = local_output_end - local_output_start;
            const uint32_t half = num_worker_output_chunks / 2;

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

                const uint32_t stripe_step = is_forward ? num_devices - 1 : 1;
                const uint32_t num_iters = dir_iters(dir);
                const uint32_t num_recv =
                    is_ring ? num_devices / 2 : (is_forward ? device_idx : num_devices - 1 - device_idx);
                const bool do_local_write = is_forward ? (fwd_iters > 0) : (fwd_iters == 0);

                // data_valid sem is granularly incremented when downstream needs to relay the stripe.
                // data_valid sem is incremented just once when downstream is a sink (doesn't need to relay).
                uint32_t num_granular = 0;
                if (num_iters > 0) {
                    const uint32_t downstream_iters =
                        is_ring ? num_devices / 2
                                : relay_iters(is_forward ? device_idx + 1 : device_idx - 1, is_forward);
                    num_granular = downstream_iters > 0 ? downstream_iters - 1 : 0;
                }

                uint32_t final_start = local_output_start;
                uint32_t final_count = num_worker_output_chunks;
                if (ring_even_split) {
                    final_start = is_forward ? local_output_start : (local_output_start + half);
                    final_count = is_forward ? half : (num_worker_output_chunks - half);
                }

                // Chunks the upstream delivers into our output (relayed full stripes + sink). The even-ring
                // antipode arrives as a half, so it contributes final_count instead of a full stripe.
                const uint32_t total_chunks = num_recv * num_worker_output_chunks -
                                              (ring_even_split ? (num_worker_output_chunks - final_count) : 0);

                std::vector<uint32_t> reader_rt_args = {
                    input_addr,                // input tensor address
                    output_addr,               // output tensor address
                    device_idx,                // this device's index (initial stripe)
                    stripe_step,               // stripe index step per iteration
                    num_iters,                 // iterations this direction runs
                    total_chunks,              // chunks upstream delivers (completion wait)
                    local_output_start,        // this worker's slice start (chunks)
                    num_worker_output_chunks,  // this worker's slice length (chunks)
                    final_start,               // last-iteration slice start (even-ring split)
                    final_count,               // last-iteration slice length (even-ring split)
                    input_tile_id_start,       // local data: input page start
                    input_tile_id_end,         // local data: input page end
                    barrier_sem.address(),     // barrier_sem L1 address
                    data_valid_sem.address(),  // data_valid_sem L1 address
                };
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                // route for Fabric_2D
                const auto route_node =
                    neighbor.has_value() ? mesh_device->get_fabric_node_id(*neighbor) : sender_fabric_node_id;
                std::vector<uint32_t> writer_rt_args = {
                    output_addr,                      // output tensor address
                    device_idx,                       // this device's index (initial stripe)
                    stripe_step,                      // stripe index step per iteration
                    num_iters,                        // iterations this direction runs
                    local_output_start,               // this worker's slice start (chunks)
                    num_worker_output_chunks,         // this worker's slice length (chunks)
                    final_start,                      // last-iteration slice start (even-ring split)
                    final_count,                      // last-iteration slice length (even-ring split)
                    do_local_write ? 1u : 0u,         // write local data into local output on iteration 0
                    barrier_sem.address(),            // barrier_sem L1 address
                    data_valid_sem.address(),         // data_valid_sem L1 address
                    (uint32_t)partner_core.x,         // barrier_sem target (neighbor partner core x)
                    (uint32_t)partner_core.y,         // barrier_sem target (neighbor partner core y)
                    (uint32_t)mirror_core.x,          // data_valid_sem target (neighbor mirror core x)
                    (uint32_t)mirror_core.y,          // data_valid_sem target (neighbor mirror core y)
                    num_granular,                     // leading sends the downstream relays
                    (uint32_t)route_node.chip_id,     // neighbor chip id (packet header 2D route)
                    (uint32_t)(*route_node.mesh_id),  // neighbor mesh id (packet header 2D route)
                };
                TT_FATAL(num_iters == 0 || neighbor.has_value(), "an active direction must have a neighbor");
                if (num_iters > 0) {
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
        for (const auto& core : shared_vars.worker_cores) {
            // reader: [0]=input_addr, [1]=output_addr, [12]=barrier_sem, [13]=data_valid_sem
            auto& reader_args = reader_args_by_core[core.x][core.y];
            reader_args[0] = input_addr;
            reader_args[1] = output_addr;
            reader_args[12] = barrier_addr;
            reader_args[13] = data_valid_addr;
            // writer: [0]=output_addr, [9]=barrier_sem, [10]=data_valid_sem
            auto& writer_args = writer_args_by_core[core.x][core.y];
            writer_args[0] = output_addr;
            writer_args[9] = barrier_addr;
            writer_args[10] = data_valid_addr;
        }
    }
}

}  // namespace ttnn::operations::ccl
