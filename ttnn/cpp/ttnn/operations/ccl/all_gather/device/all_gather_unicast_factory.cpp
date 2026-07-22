// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_unicast_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

////////////////////////////////////////////////////////////////
// Store-and-forward AllGather (Fabric1D or direct-neighbor Fabric2D line/ring)
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
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, output_tensor, barrier_sem, data_valid_sem);
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
    const tt::tt_metal::GlobalSemaphore& data_valid_sem) {
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

    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());
    TT_FATAL(
        !fabric_is_2d || operation_attributes.neighbor_unicast_eligible,
        "Fabric2D all_gather neighbor unicast requires a host-proved direct physical line/ring");

    uint32_t active_axis = 0;
    for (uint32_t a = 0; a < 2; ++a) {
        if (operation_attributes.axis_num_devices[a] > 1) {
            active_axis = a;
        }
    }
    const uint32_t axis = operation_attributes.cluster_axis.value_or(active_axis);
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
    // Each link runs two directions: forward (dir 0) and backward (dir 1). With NUM_WORKERS_PER_LINK == 1 each
    // direction is a single core connected directly to its neighbor's ERISC. With > 1 the workers of a direction
    // can't each open a direct connection (an ERISC exposes one worker sender channel per direction), so they
    // share a fabric mux: one mux core per direction per link owns the connection and multiplexes their traffic.
    //
    // Flat layout from choose_worker_cores(num_links, num_cores_per_link), per link:
    //   [dir 0: (mux?) worker 0 .. worker W-1][dir 1: (mux?) worker 0 .. worker W-1]
    ////////////////////////////////////////////////////////////////

    // Num worker cores per direction per link. >1 requires an additional fabric mux core to own the fabric
    // connection and multiplex traffic.
    // This is a major perf knob, below heuristic was determined from extensive test sweeps.
    const uint32_t num_links = operation_attributes.axis_num_links[axis];
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();
    uint32_t workers_per_dir = 1;
    if (input_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0) {
        workers_per_dir = 2;
    } else if (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) {
        // Large steady-state tensors use four workers per direction. For small
        // row pages this also creates one worker per DRAM bank across two links,
        // enabling the bank-owned full-packet schedule below.
        const uint32_t txn_bytes = std::min(input_page_size, output_page_size);  // NOC transaction size
        const uint64_t total_output_bytes =
            (uint64_t)output_tensor.buffer()->num_pages() * output_tensor.buffer()->aligned_page_size();
        const uint64_t per_link_bytes = total_output_bytes / std::max(1u, num_links);
        constexpr uint64_t bw_bound_link_bytes = 4500000ULL;  // gathered bytes/link where fabric link saturates
        if (txn_bytes > 0 && per_link_bytes >= bw_bound_link_bytes) {
            workers_per_dir = 4;
        } else {
            workers_per_dir = 2;
        }
    }

    constexpr uint32_t num_directions = 2;  // 0 = forward, 1 = backward
    const bool use_mux = workers_per_dir > 1;
    const uint32_t mux_per_dir = use_mux ? 1u : 0u;
    const uint32_t cores_per_dir = workers_per_dir + mux_per_dir;
    const uint32_t num_cores_per_link = num_directions * cores_per_dir;

    // all_cores contains workers + mux
    [[maybe_unused]] auto [all_core_range, all_cores] = ttnn::ccl::choose_worker_cores(
        num_links,
        num_cores_per_link,
        mesh_device,
        operation_attributes.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        operation_attributes.sub_core_grid);
    TT_FATAL(
        all_cores.size() == static_cast<size_t>(num_links) * num_cores_per_link,
        "all_gather needs {} worker cores ({} links x {} cores/link) but only {} are available; provide a larger "
        "sub_core_grid.",
        static_cast<size_t>(num_links) * num_cores_per_link,
        num_links,
        num_cores_per_link,
        all_cores.size());

    // Helpers to index into the flat core vector (dir: 0 = forward, 1 = backward).
    auto core_at = [&](uint32_t link, uint32_t dir, uint32_t idx_in_dir) -> const CoreCoord& {
        return all_cores[(link * num_cores_per_link) + (dir * cores_per_dir) + idx_in_dir];
    };
    auto mux_core = [&](uint32_t link, uint32_t dir) -> const CoreCoord& { return core_at(link, dir, 0); };
    auto worker_core = [&](uint32_t link, uint32_t dir, uint32_t w) -> const CoreCoord& {
        return core_at(link, dir, mux_per_dir + w);
    };
    auto dir_neighbor = [&](uint32_t dir) { return dir == 0 ? fwd_coord : bwd_coord; };
    auto dir_active = [&](uint32_t dir) { return dir_neighbor(dir).has_value(); };

    // Reader/writer kernels + CB run on worker cores only; the mux kernel runs on its own cores (created only
    // for a direction that has a neighbor).
    std::vector<CoreCoord> worker_cores;
    worker_cores.reserve(num_links * num_directions * workers_per_dir);
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

    // Fabric mux config
    constexpr uint8_t num_buffers_per_channel = 2;  // hardcoded since no observable impact on performance
    tt::tt_fabric::FabricMuxConfig mux_config(
        /*num_full_size_channels=*/workers_per_dir,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/num_buffers_per_channel,
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

    auto input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }

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

    ::ttnn::ccl::validate_packet_size(input_tensor.device()->arch(), packet_size, output_chunk_size);

    // --- CB sizing ---
    // cb_page_size is a multiple of input_page_size, which is itself a multiple of
    // output_chunk_size = min(input, output), so the kernel increments both
    // the cb_read_ptr and cb_write_ptr cleanly.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    uint32_t cb_page_size = input_page_size * pages_per_packet;
    uint32_t cb_depth = 3;
    // Perf hack: for tile layout, pack multiple pages into a single CB page to reduce CB sync
    // frequency between reader and writer. Note this increases effective CB depth.
    // Don't do this for row-major layout because of all the careful handling of page sizes.
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        // Empirically determined heuristic, works well for all tensor sizes
        const uint32_t ideal_multiplier = (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) ? 4 : 3;
        // Find the largest multiplier in [1, ideal] that fits in available L1
        const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
        const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * cb_page_size), 1u, ideal_multiplier);
        if (multiplier < ideal_multiplier) {
            log_warning(
                tt::LogOp,
                "CircularBuffer depth reduced due to L1 pressure (only {} B available), performance may regress.",
                max_l1_space);
        }
        cb_page_size *= multiplier;
    }

    // --- Stripe geometry ---
    // input_pages_per_stripe = num input pages along [gather_dim .. rank-1] this
    // device contributes per stripe. For RM gather_dim=-1 this is the *page* count,
    // which handles sharded RM input (> 1 input page per row).
    auto tile_spec = input_tensor.layout() == Layout::TILE ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent;
        if (i == rank - 1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tile_spec.get_width();
            } else {
                // This is a page count, so divide by the unaligned page size, not aligned
                extent = (input_shape[i] * input_tensor.element_size()) / input_unaligned_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == rank - 2) {
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

    const uint32_t total_slices = num_links * workers_per_dir;
    const uint32_t num_dram_banks = mesh_device->num_dram_channels();
    const bool bank_owned_schedule =
        input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT && input_tensor.buffer()->is_dram() &&
        output_tensor.buffer()->is_dram() &&
        input_tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        output_tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        output_chunks_per_page == 1 && split_factor == 1 && output_chunks_per_stripe == num_input_pages &&
        total_slices == num_dram_banks && num_input_pages % total_slices == 0;
    const uint32_t slice_step = bank_owned_schedule ? total_slices : 1;

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
        slice_step,                // one means contiguous slices; >1 owns one interleaved DRAM bank
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

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
        slice_step,                // one means scatter packets; >1 enables contiguous full packets
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    // When multiple workers share a fabric mux, the writer connects through it: append the mux geometry (after
    // the tensor-accessor args) and switch the kernel onto its USE_WORKER_MUX path.
    std::map<std::string, std::string> writer_defines;
    if (fabric_is_2d) {
        writer_defines["FABRIC_2D"] = "1";
        writer_defines["API_TYPE_Mesh"] = "1";
    }
    if (use_mux) {
        ttnn::ccl::fabric_mux_connection_ct_args(
            workers_per_dir, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_config, writer_compile_args);
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

    ////////////////////////////////////////////////////////////////
    // Runtime args
    //
    // The page split is per (link, worker) -- num_links * workers_per_dir slices -- and direction-independent,
    // so a (link, worker)'s forward and backward directions relay the same slice. Core assignment is
    // deterministic, so worker w has the same coords on every device: data_valid signals target the mirror
    // worker w on the neighbor.
    ////////////////////////////////////////////////////////////////

    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    // Mux runtime args: one fabric connection per active direction per link, to that direction's neighbor. The
    // direction's workers all feed this one connection.
    if (use_mux) {
        for (uint32_t link = 0; link < num_links; ++link) {
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                if (!dir_active(dir)) {
                    continue;
                }
                const CoreCoord mux_core_coord = mux_core(link, dir);
                const auto dst_node = mesh_device->get_fabric_node_id(*dir_neighbor(dir));
                auto mux_rt_args = mux_config.get_fabric_mux_run_time_args(
                    sender_fabric_node_id, dst_node, link, program, mux_core_coord);
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_core_coord}, mux_rt_args);
            }
        }
    }

    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t w = 0; w < workers_per_dir; ++w) {
            const uint32_t slice_idx = (link * workers_per_dir) + w;
            const uint32_t input_pages_per_slice = num_input_pages / total_slices;
            const uint32_t remainder = num_input_pages % total_slices;
            const uint32_t input_tile_id_start =
                bank_owned_schedule ? slice_idx : (slice_idx * input_pages_per_slice) + std::min(slice_idx, remainder);
            const uint32_t input_tile_id_end =
                bank_owned_schedule ? num_input_pages
                                    : ((slice_idx + 1) * input_pages_per_slice) + std::min(slice_idx + 1, remainder);
            const uint32_t local_output_start =
                bank_owned_schedule
                    ? slice_idx
                    : (static_cast<uint64_t>(input_tile_id_start) * num_output_chunks) / num_input_pages;
            const uint32_t local_output_end =
                bank_owned_schedule ? local_output_start + input_pages_per_slice
                                    : (static_cast<uint64_t>(input_tile_id_end) * num_output_chunks) / num_input_pages;
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
                const auto neighbor_node =
                    neighbor.has_value() ? mesh_device->get_fabric_node_id(*neighbor) : sender_fabric_node_id;
                TT_FATAL(
                    !neighbor.has_value() || !fabric_is_2d ||
                        tt::tt_fabric::are_direct_fabric_neighbors(sender_fabric_node_id, neighbor_node),
                    "Fabric2D neighbor-unicast edge from {} to {} is not one physical hop",
                    sender_fabric_node_id,
                    neighbor_node);

                const uint32_t stripe_step = is_forward ? num_devices - 1 : 1;
                const uint32_t num_iters = is_forward ? fwd_iters : bwd_iters;
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
                    final_start = is_forward ? local_output_start : (local_output_start + half * slice_step);
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

                std::vector<uint32_t> writer_rt_args = {
                    output_addr,               // output tensor address
                    device_idx,                // this device's index (initial stripe)
                    stripe_step,               // stripe index step per iteration
                    num_iters,                 // iterations this direction runs
                    local_output_start,        // this worker's slice start (chunks)
                    num_worker_output_chunks,  // this worker's slice length (chunks)
                    final_start,               // last-iteration slice start (even-ring split)
                    final_count,               // last-iteration slice length (even-ring split)
                    do_local_write ? 1u : 0u,  // write local data into local output on iteration 0
                    barrier_sem.address(),     // barrier_sem L1 address
                    data_valid_sem.address(),  // data_valid_sem L1 address
                    (uint32_t)partner_core.x,  // barrier_sem target (neighbor partner core x)
                    (uint32_t)partner_core.y,  // barrier_sem target (neighbor partner core y)
                    (uint32_t)mirror_core.x,   // data_valid_sem target (neighbor mirror core x)
                    (uint32_t)mirror_core.y,   // data_valid_sem target (neighbor mirror core y)
                    num_granular,              // leading sends the downstream relays
                    static_cast<uint32_t>(neighbor_node.chip_id),
                    static_cast<uint32_t>(*neighbor_node.mesh_id),
                };
                TT_FATAL(num_iters == 0 || neighbor.has_value(), "an active direction must have a neighbor");
                if (num_iters > 0) {
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
                            writer_rt_args);
                    } else {
                        std::vector<tt::tt_fabric::FabricNodeId> dst = {neighbor_node};
                        append_routing_plane_connection_manager_rt_args(
                            sender_fabric_node_id,
                            dst,
                            {link},
                            program,
                            writer_kernel_id,
                            {core},
                            writer_rt_args,
                            fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
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
