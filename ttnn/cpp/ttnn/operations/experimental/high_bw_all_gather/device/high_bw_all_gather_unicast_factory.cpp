// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "high_bw_all_gather_unicast_factory.hpp"
#include "high_bw_all_gather_scheduler.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

using namespace ::ttnn::ccl;

namespace CMAKE_UNIQUE_NAMESPACE {

struct PageGeometry {
    uint32_t input_page_size;
    uint32_t output_chunk_size;
    uint32_t output_chunks_per_page;
    uint32_t split_factor;
    uint32_t num_input_pages;
    uint32_t num_output_chunks;
    uint32_t output_chunks_per_stripe;
    uint32_t input_page_base;
};

enum class ReaderRtArg : std::size_t {
    InputAddress,
    OutputAddress,
    InitialStripe,
    StripeStep,
    NumIters,
    TotalChunks,
    SliceStart,
    SliceCount,
    FinalStart,
    FinalCount,
    InputPageStart,
    InputPageEnd,
    ReadySemaphore,
    DataValidSemaphore,
    OutputChunksPerStripe,
    Count,
};

enum class WriterRtArg : std::size_t {
    OutputAddress,
    InitialStripe,
    StripeStep,
    NumIters,
    SliceStart,
    SliceCount,
    FinalStart,
    FinalCount,
    DoLocalWrite,
    ReadySemaphore,
    ReadyNocX,
    ReadyNocY,
    DataValidSemaphore,
    DataValidNocX,
    DataValidNocY,
    NumGranularSends,
    DataValidGranularity,
    NeighborDeviceId,
    NeighborMeshId,
    OutputChunksPerStripe,
    Count,
};

template <typename Enum>
constexpr std::size_t rt_arg_index(Enum value) {
    return static_cast<std::size_t>(value);
}

PageGeometry derive_page_geometry(
    const Tensor& input_tensor, const Tensor& output_tensor, const HighBwAllGatherParams& operation_attributes) {
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t input_unaligned_page_size = input_tensor.buffer()->page_size();
    const uint32_t output_unaligned_page_size = output_tensor.buffer()->page_size();
    const bool is_split = input_unaligned_page_size > output_unaligned_page_size;
    const uint32_t output_chunk_size = is_split ? output_unaligned_page_size : input_page_size;
    const uint32_t output_chunks_per_page = is_split ? 1u : output_unaligned_page_size / input_unaligned_page_size;
    const uint32_t split_factor = is_split ? input_unaligned_page_size / output_unaligned_page_size : 1u;
    TT_FATAL(
        output_chunks_per_page == 1 || input_page_size == input_unaligned_page_size,
        "concat requires an unpadded input page");

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }

    const bool has_runtime_extent = operation_attributes.gathered_dim_size.has_value();
    const uint32_t active_dim_size = has_runtime_extent
                                         ? *operation_attributes.gathered_dim_size / operation_attributes.num_devices
                                         : input_shape[gather_dim];

    const auto tile_spec =
        input_tensor.layout() == Layout::TILE ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent = i == gather_dim ? active_dim_size : input_shape[i];
        if (i == rank - 1) {
            if (input_tensor.layout() == Layout::TILE) {
                TT_FATAL(
                    extent % tile_spec.get_width() == 0,
                    "high_bw_all_gather active gather extent {} must be tile-width aligned ({})",
                    extent,
                    tile_spec.get_width());
                extent /= tile_spec.get_width();
            } else {
                TT_FATAL(
                    (static_cast<uint64_t>(extent) * input_tensor.element_size()) % input_unaligned_page_size == 0,
                    "high_bw_all_gather active innermost gather extent {} must occupy whole input pages",
                    extent);
                extent = (extent * input_tensor.element_size()) / input_unaligned_page_size;
            }
        } else if (input_tensor.layout() == Layout::TILE && i == rank - 2) {
            TT_FATAL(
                extent % tile_spec.get_height() == 0,
                "high_bw_all_gather active gather extent {} must be tile-height aligned ({})",
                extent,
                tile_spec.get_height());
            extent /= tile_spec.get_height();
        }
        input_pages_per_stripe *= extent;
    }

    // A runtime extent controls how many source pages are transferred, not the placement stride in
    // the preallocated output. Keeping the maximum per-rank stride leaves address-indexed caches
    // (such as sparse MLA's block-cyclic KV cache) at their stable physical offsets.
    uint32_t max_input_pages_per_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent = input_shape[i];
        if (i == rank - 1) {
            extent = input_tensor.layout() == Layout::TILE
                         ? extent / tile_spec.get_width()
                         : (extent * input_tensor.element_size()) / input_unaligned_page_size;
        } else if (input_tensor.layout() == Layout::TILE && i == rank - 2) {
            extent /= tile_spec.get_height();
        }
        max_input_pages_per_stripe *= extent;
    }
    const uint32_t output_chunks_per_stripe = max_input_pages_per_stripe * split_factor;
    TT_FATAL(output_chunks_per_stripe > 0, "output_chunks_per_stripe must be > 0");
    const bool selected_batch = operation_attributes.input_batch_index.has_value();
    const bool selected_or_partial = selected_batch || has_runtime_extent;
    uint32_t input_page_base = 0;
    uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    if (selected_or_partial) {
        // A partial all-gather is sourced from one contiguous batch slot. Its active pages are
        // placed in each rank's fixed worst-case output slot, preserving stable cache offsets.
        for (int32_t i = 1; i < gather_dim; ++i) {
            TT_FATAL(
                input_shape[i] == 1,
                "high_bw_all_gather selected/partial gather requires singleton dimensions between batch and dim; "
                "input shape {}, dim {}",
                input_shape,
                gather_dim);
        }
        if (selected_batch) {
            TT_FATAL(
                input_tensor.buffer()->num_pages() % input_shape[0] == 0,
                "high_bw_all_gather input batch slots must occupy equal page ranges");
            input_page_base =
                *operation_attributes.input_batch_index * (input_tensor.buffer()->num_pages() / input_shape[0]);
        } else {
            TT_FATAL(
                input_shape[0] == 1,
                "high_bw_all_gather gathered_dim_size without input_batch_index requires input batch 1, got {}",
                input_shape[0]);
        }
        num_input_pages = input_pages_per_stripe;
        TT_FATAL(
            input_page_base + num_input_pages <= input_tensor.buffer()->num_pages(),
            "high_bw_all_gather selected range exceeds input allocation");
    }
    const uint32_t num_output_chunks = num_input_pages * split_factor;
    return {
        input_page_size,
        output_chunk_size,
        output_chunks_per_page,
        split_factor,
        num_input_pages,
        num_output_chunks,
        output_chunks_per_stripe,
        input_page_base};
}

uint32_t derive_data_valid_granularity(const PageGeometry& geometry, uint32_t packet_size, uint32_t total_slices) {
    const uint32_t pages_per_packet = std::max(1u, packet_size / geometry.input_page_size);
    const uint32_t cb_page_size = geometry.input_page_size * pages_per_packet;
    const uint32_t outputs_per_cb_page = std::max(1u, cb_page_size / geometry.output_chunk_size);
    const uint32_t cb_pages_per_stripe =
        std::max(1u, (geometry.num_output_chunks / total_slices) / outputs_per_cb_page);
    return std::max(1u, cb_pages_per_stripe / 2u);
}

bool can_use_output_bank_owned_schedule(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const PageGeometry& geometry,
    uint32_t num_links,
    uint32_t workers_per_direction,
    uint32_t num_dram_banks) {
    return input_tensor.buffer()->is_dram() && output_tensor.buffer()->is_dram() &&
           output_tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
           geometry.output_chunks_per_page == 1 && geometry.split_factor == 1 &&
           geometry.output_chunks_per_stripe == geometry.num_input_pages &&
           scheduler::can_partition_workers_by_bank(
               geometry.num_input_pages, num_links, workers_per_direction, num_dram_banks);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

using namespace CMAKE_UNIQUE_NAMESPACE;

////////////////////////////////////////////////////////////////
// Store-and-forward HighBwAllGather (Fabric1D or direct-neighbor Fabric2D line/ring)
//
// Every device relays stripes to its neighbor one hop at a time; a shard reaches far devices by being
// re-forwarded at each hop. Forward and backward directions run on separate cores. Per direction: the reader
// (CB producer, no fabric) reads iteration 0 from local input and later iterations from what upstream relayed
// into our output; the writer (CB consumer) unicasts each stripe one hop to the neighbor's output (same
// address on every device). Direction/topology are runtime args, so both kernels compile once and run on all
// cores. ready_sem protects destination initialization, while data_valid_sem gates relays and tracks completion.
////////////////////////////////////////////////////////////////

HighBwAllGatherUnicastFactory::cached_mesh_workload_t HighBwAllGatherUnicastFactory::create_mesh_workload(
    const HighBwAllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const HighBwAllGatherInputs& tensor_args,
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

    // Keep the startup-readiness and relay/completion semaphores in L1_SMALL when the device reserves it.
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0;
    auto sem_buffer_type = has_l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    auto ready_sem = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto data_valid_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, output_tensor, ready_sem, data_valid_sem);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

HighBwAllGatherUnicastFactory::cached_program_t HighBwAllGatherUnicastFactory::create_at(
    const HighBwAllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const HighBwAllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& ready_sem,
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

    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(operation_attributes.fabric_config);
    TT_FATAL(
        !fabric_is_2d || operation_attributes.neighbor_unicast_eligible,
        "Fabric2D high_bw_all_gather neighbor unicast requires a host-proved direct physical line/ring");

    const bool linearized_mesh_ring = operation_attributes.linearized_mesh_ring;
    const uint32_t axis = operation_attributes.cluster_axis;
    const auto topology =
        linearized_mesh_ring ? tt::tt_fabric::Topology::Ring : operation_attributes.axis_topology[axis];
    const bool is_ring = tt::tt_fabric::is_ring_or_torus(topology);

    const uint32_t num_devices = operation_attributes.num_devices;
    TT_FATAL(
        !is_ring || num_devices > 2,
        "high_bw_all_gather ring schedule requires more than two participating devices; got {}, "
        "linearized_mesh_ring={}, cluster_axis={}",
        num_devices,
        linearized_mesh_ring,
        axis);
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(
        !linearized_mesh_ring ||
            (operation_attributes.mesh_rows == mesh_shape[0] && operation_attributes.mesh_cols == mesh_shape[1]),
        "cached full mesh-ring shape {}x{} does not match live mesh shape {}",
        operation_attributes.mesh_rows,
        operation_attributes.mesh_cols,
        mesh_shape);
    const ttnn::operations::ccl::common::MeshRingPlan mesh_ring_plan{
        .cluster_axis = linearized_mesh_ring ? std::nullopt : std::optional<uint32_t>{axis},
        .full_mesh = linearized_mesh_ring,
        .orientation = operation_attributes.snake_ring_orientation,
        .mesh_rows = linearized_mesh_ring ? operation_attributes.mesh_rows : mesh_shape[0],
        .mesh_cols = linearized_mesh_ring ? operation_attributes.mesh_cols : mesh_shape[1],
        .ring_size = num_devices,
        .num_links = operation_attributes.num_links,
        .topology = topology,
        .fabric_config = operation_attributes.fabric_config,
        .axis_topology = operation_attributes.axis_topology,
        .route_plan_hash = operation_attributes.neighbor_route_plan_hash};
    const auto mesh_ring_position =
        ttnn::operations::ccl::common::get_mesh_ring_position(input_tensor, sender_device_coord, mesh_ring_plan);
    const uint32_t device_idx = mesh_ring_position.transport_rank;
    auto fwd_coord = mesh_ring_position.forward_coord;
    auto bwd_coord = mesh_ring_position.backward_coord;

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
    const uint32_t num_links = operation_attributes.num_links;
    const bool has_runtime_controls =
        operation_attributes.input_batch_index.has_value() || operation_attributes.gathered_dim_size.has_value();
    const auto page_geometry = derive_page_geometry(input_tensor, output_tensor, operation_attributes);
    // gathered_dim_size is deliberately cache-key-independent. Runtime controls reuse the compiled
    // schedule and patch its selected page base and active ranges below from page_geometry.
    const uint32_t input_page_size = page_geometry.input_page_size;
    const uint32_t num_dram_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    TT_FATAL(num_dram_banks > 0, "high_bw_all_gather requires at least one allocator-managed DRAM bank");
    // Size the cached worker topology for the maximum output allocation. Runtime prefix lengths are excluded
    // from the program hash, so every cache hit must reuse this worst-case tier.
    const uint64_t total_output_bytes =
        (uint64_t)output_tensor.buffer()->num_pages() * output_tensor.buffer()->aligned_page_size();
    const uint64_t per_link_bytes = total_output_bytes / std::max(1u, num_links);
    constexpr uint64_t bank_owned_min_link_bytes = 1500000ULL;
    constexpr uint64_t high_parallelism_min_link_bytes = 32000000ULL;
    // Select the worker tier from the maximum per-slot geometry. Runtime prefixes retain the same fixed
    // output-rank stride, and both their source and destination page sequences remain bank-stridable; only
    // the number of active pages changes. This lets one cached program keep the bank-owned fast path while
    // input_batch_index/gathered_dim_size patch its page base and active slice counts at dispatch time.
    auto scheduling_geometry = page_geometry;
    if (has_runtime_controls) {
        scheduling_geometry.num_input_pages = page_geometry.output_chunks_per_stripe;
    }
    const auto can_use_bank_owned = [&](uint32_t workers_per_direction) {
        return can_use_output_bank_owned_schedule(
            input_tensor, output_tensor, scheduling_geometry, num_links, workers_per_direction, num_dram_banks);
    };
    uint32_t workers_per_dir = 1;
    if (input_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0) {
        workers_per_dir = 2;

        // Wormhole has 12 DRAM banks (11 when one is harvested). For sufficiently large messages, use enough
        // workers to retain bank-local packet coalescing instead of forcing these configurations through the
        // scatter fallback. Keep two workers for small messages, where additional core/mux setup is not amortized.
        const uint32_t bank_covering_workers =
            scheduler::workers_per_direction_to_cover_banks(num_links, num_dram_banks);
        if (per_link_bytes >= bank_owned_min_link_bytes && can_use_bank_owned(bank_covering_workers)) {
            workers_per_dir = bank_covering_workers;
        }

        // Large messages benefit from additional in-flight DRAM reads after every bank is covered.
        const uint32_t high_parallelism_workers = std::max(8u, bank_covering_workers);
        if (per_link_bytes >= high_parallelism_min_link_bytes && can_use_bank_owned(high_parallelism_workers)) {
            workers_per_dir = high_parallelism_workers;
        }
    } else if (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) {
        // Measured on Blackhole across every qualified page format and line/ring schedules. Four workers amortize
        // their mux/core overhead once a bank-owned message reaches roughly 1.5 MB/link. Eight workers do not
        // consistently pull ahead of four until roughly 32 MB/link.
        const bool four_workers_bank_owned = can_use_bank_owned(4);
        const bool eight_workers_bank_owned = can_use_bank_owned(8);

        workers_per_dir = 2;
        if (per_link_bytes >= bank_owned_min_link_bytes && four_workers_bank_owned) {
            workers_per_dir = 4;
        }
        if (per_link_bytes >= high_parallelism_min_link_bytes) {
            // Large 2 KB-page shapes still benefit from parallel injection when slice divisibility prevents
            // four-worker bank ownership. Smaller fallback packets did not recover the additional worker/mux overhead.
            const bool use_high_parallelism_fallback = !four_workers_bank_owned && input_page_size >= 2048;
            if (eight_workers_bank_owned || use_high_parallelism_fallback) {
                workers_per_dir = 8;
            }
        }
    }

    // A restricted sub-core grid may not fit the preferred count. Snap to a measured worker tier rather than
    // walking through unqualified intermediate counts; bank ownership is re-evaluated with the selected tier.
    const auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_worker_cores =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_worker_cores = available_worker_cores.intersection(operation_attributes.sub_core_grid.value());
    }
    const uint32_t preferred_workers_per_dir = workers_per_dir;
    const bool preferred_schedule_is_bank_owned = can_use_bank_owned(preferred_workers_per_dir);
    const auto worker_count_fits = [&](uint32_t count) {
        return scheduler::worker_count_fits(
            count, num_links, static_cast<uint32_t>(available_worker_cores.num_cores()));
    };
    if (!worker_count_fits(workers_per_dir)) {
        const uint32_t bank_covering_workers =
            scheduler::workers_per_direction_to_cover_banks(num_links, num_dram_banks);
        const std::array<uint32_t, 4> reduction_tiers =
            input_tensor.device()->arch() == tt::ARCH::WORMHOLE_B0
                ? std::array<uint32_t, 4>{std::max(8u, bank_covering_workers), bank_covering_workers, 2u, 1u}
                : std::array<uint32_t, 4>{8u, 4u, 2u, 1u};
        workers_per_dir = scheduler::select_fitting_worker_count(
            preferred_workers_per_dir,
            num_links,
            static_cast<uint32_t>(available_worker_cores.num_cores()),
            reduction_tiers);
    }
    if (preferred_schedule_is_bank_owned && !can_use_bank_owned(workers_per_dir)) {
        // Do not spend almost as many cores on the scatter fallback when a restricted grid falls just short of
        // covering every bank.
        workers_per_dir = std::min(2u, workers_per_dir);
    }
    TT_FATAL(
        worker_count_fits(workers_per_dir),
        "high_bw_all_gather needs at least {} worker cores for {} link(s), but only {} are available",
        num_links * 2 * (workers_per_dir + (workers_per_dir > 1 ? 1u : 0u)),
        num_links,
        available_worker_cores.num_cores());
    if (workers_per_dir != preferred_workers_per_dir) {
        log_warning(
            tt::LogOp,
            "high_bw_all_gather reduced workers per direction from {} to {} because only {} worker cores are "
            "available; this may disable bank-owned packet coalescing",
            preferred_workers_per_dir,
            workers_per_dir,
            available_worker_cores.num_cores());
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
        "high_bw_all_gather needs {} worker cores ({} links x {} cores/link) but only {} are available; provide a "
        "larger "
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
    if (use_mux) {
        TT_FATAL(
            mux_config.get_memory_map_end_address() <= mesh_device->l1_size_per_core(),
            "high_bw_all_gather Fabric mux requires L1 through address {:#x}, but each Tensix core has only "
            "{:#x} bytes",
            mux_config.get_memory_map_end_address(),
            mesh_device->l1_size_per_core());
    }

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

    // --- Copy mode ---
    // The kernel always reads whole *aligned* input pages into L1 (required by the input's NoC
    // read alignment, DRAM or L1) but writes at output *content* (unaligned) granularity, so
    // chunk sizing differs by mode:
    //   matched (in == out): 1 chunk per input page, output_chunks_per_page = 1.
    //   concat  (out > in) : 1 chunk per input page, output_chunks_per_page > 1; each chunk
    //                        lands at a byte offset within a shared output page.
    //   split   (in > out) : split_factor chunks per input page, output_chunks_per_page = 1.
    const uint32_t output_chunk_size = page_geometry.output_chunk_size;
    const uint32_t output_chunks_per_page = page_geometry.output_chunks_per_page;
    const uint32_t num_input_pages = page_geometry.num_input_pages;
    const uint32_t num_output_chunks = page_geometry.num_output_chunks;
    const uint32_t output_chunks_per_stripe = page_geometry.output_chunks_per_stripe;

    ::ttnn::ccl::validate_packet_size(input_tensor.device()->arch(), packet_size, output_chunk_size);

    // --- CB sizing ---
    // cb_page_size is a multiple of input_page_size, which is itself a multiple of
    // output_chunk_size = min(input, output), so the kernel increments both
    // the cb_read_ptr and cb_write_ptr cleanly.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    const uint32_t cb_page_size = input_page_size * pages_per_packet;
    constexpr uint32_t cb_depth = 3;

    const uint32_t total_slices = num_links * workers_per_dir;
    // Assign logical pages by destination DRAM bank whenever the output is
    // interleaved. TensorAccessor still resolves each source page correctly
    // when the input uses an ND-sharded/block-cyclic DRAM layout. Keeping the
    // destination pages bank-local lets the writer coalesce small pages into a
    // full Fabric packet instead of falling back to four-entry scatter writes.
    // The output stride is already the maximum per-rank slot width. Runtime prefixes patch their active
    // bank-owned page ranges below without changing that stride or the compiled worker topology.
    const bool output_bank_owned_schedule = can_use_bank_owned(workers_per_dir);
    const uint32_t slice_step = output_bank_owned_schedule ? num_dram_banks : 1;
    // Runtime controls change the selected source base and active page count, but every rank retains its
    // maximum output slot. Its stripe width is therefore structural and can stay baked into the iterator,
    // avoiding dynamic divide/modulo on both fixed-shape and selected-prefix paths.
    const uint32_t static_output_chunks_per_stripe = output_chunks_per_stripe;

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
    const uint32_t data_valid_granularity = derive_data_valid_granularity(page_geometry, packet_size, total_slices);

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        input_page_size,         // input tensor page size
        output_chunk_size,       // NOC write size = min(input, output)
        output_chunks_per_page,  // chunks per output page (1 unless concat)
        num_devices,             // device count (stripe indexing)
        cb0_id,                  // cb id
        cb_page_size,            // cb entry size
        slice_step,              // one means contiguous slices; >1 owns one interleaved DRAM bank
        static_output_chunks_per_stripe,
        linearized_mesh_ring,  // translate snake ring indices back to row-major tensor stripe indices
        static_cast<uint32_t>(operation_attributes.snake_ring_orientation),
        operation_attributes.mesh_rows,
        operation_attributes.mesh_cols,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        output_chunk_size,       // NOC write size = min(input, output)
        output_chunks_per_page,  // chunks per output page (1 unless concat)
        num_devices,             // device count (stripe indexing)
        cb0_id,                  // cb id
        cb_page_size,            // cb entry size
        packet_size,             // packet_size
        slice_step,              // one means scatter packets; >1 enables contiguous full packets
        static_output_chunks_per_stripe,
        linearized_mesh_ring,  // translate snake ring indices back to row-major tensor stripe indices
        static_cast<uint32_t>(operation_attributes.snake_ring_orientation),
        operation_attributes.mesh_rows,
        operation_attributes.mesh_cols,
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    // The writer selects its direct-EDM or fabric-mux path with this constexpr argument. Keep the mux geometry
    // present for both variants so the kernel has one stable compile-time-argument layout.
    writer_compile_args.push_back(use_mux);
    std::map<std::string, std::string> writer_defines;
    if (fabric_is_2d) {
        writer_defines["FABRIC_2D"] = "1";
    }
    if (use_mux) {
        ttnn::ccl::fabric_mux_connection_ct_args(
            workers_per_dir, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_config, writer_compile_args);
    } else {
        // Unused by the direct-EDM constexpr branch. One buffer keeps the discarded mux sender type valid.
        writer_compile_args.insert(writer_compile_args.end(), {1, 0, 0, 0, 0});
    }

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/unicast_reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/high_bw_all_gather/device/kernels/unicast_writer.cpp",
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
            uint32_t input_tile_id_start = (slice_idx * input_pages_per_slice) + std::min(slice_idx, remainder);
            uint32_t worker_input_page_count = input_pages_per_slice + (slice_idx < remainder ? 1u : 0u);
            if (output_bank_owned_schedule) {
                const auto bank_owned_slice = scheduler::derive_bank_owned_slice(
                    num_input_pages, num_links, workers_per_dir, num_dram_banks, link, w);
                input_tile_id_start = bank_owned_slice.input_page_start;
                worker_input_page_count = bank_owned_slice.page_count;
            }
            const uint32_t input_tile_id_end =
                output_bank_owned_schedule
                    ? input_tile_id_start + worker_input_page_count * num_dram_banks
                    : ((slice_idx + 1) * input_pages_per_slice) + std::min(slice_idx + 1, remainder);
            const uint32_t local_output_start =
                output_bank_owned_schedule
                    ? input_tile_id_start
                    : (static_cast<uint64_t>(input_tile_id_start) * num_output_chunks) / num_input_pages;
            const uint32_t local_output_end =
                output_bank_owned_schedule
                    ? local_output_start + worker_input_page_count
                    : (static_cast<uint64_t>(input_tile_id_end) * num_output_chunks) / num_input_pages;
            const uint32_t num_worker_output_chunks = local_output_end - local_output_start;
            const uint32_t half = num_worker_output_chunks / 2;

            // Both directions (dir: 0 = forward, 1 = backward). mirror_core is this core's coords, reused as the
            // data_valid_sem target on the neighbor's mirror core; partner_core is the opposite-direction worker
            // targeted by the startup-ready handshake.
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                const bool is_forward = (dir == 0);
                const CoreCoord core = worker_core(link, dir, w);
                const CoreCoord partner = worker_core(link, 1 - dir, w);
                const CoreCoord mirror_core = mesh_device->worker_core_from_logical_core(core);
                const CoreCoord partner_core = mesh_device->worker_core_from_logical_core(partner);
                const auto neighbor = dir_neighbor(dir);
                const auto neighbor_node =
                    neighbor.has_value() ? mesh_device->get_fabric_node_id(*neighbor) : sender_fabric_node_id;

                const uint32_t stripe_step = is_forward ? num_devices - 1 : 1;
                const uint32_t num_iters = is_forward ? fwd_iters : bwd_iters;
                TT_FATAL(
                    dir_active(dir) == (num_iters > 0),
                    "high_bw_all_gather direction {} has inconsistent neighbor and iteration state",
                    dir);
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

                std::vector<uint32_t> reader_rt_args(rt_arg_index(ReaderRtArg::Count));
                reader_rt_args[rt_arg_index(ReaderRtArg::InputAddress)] = input_addr;
                reader_rt_args[rt_arg_index(ReaderRtArg::OutputAddress)] = output_addr;
                reader_rt_args[rt_arg_index(ReaderRtArg::InitialStripe)] = device_idx;
                reader_rt_args[rt_arg_index(ReaderRtArg::StripeStep)] = stripe_step;
                reader_rt_args[rt_arg_index(ReaderRtArg::NumIters)] = num_iters;
                reader_rt_args[rt_arg_index(ReaderRtArg::TotalChunks)] = total_chunks;
                reader_rt_args[rt_arg_index(ReaderRtArg::SliceStart)] = local_output_start;
                reader_rt_args[rt_arg_index(ReaderRtArg::SliceCount)] = num_worker_output_chunks;
                reader_rt_args[rt_arg_index(ReaderRtArg::FinalStart)] = final_start;
                reader_rt_args[rt_arg_index(ReaderRtArg::FinalCount)] = final_count;
                reader_rt_args[rt_arg_index(ReaderRtArg::InputPageStart)] =
                    page_geometry.input_page_base + input_tile_id_start;
                reader_rt_args[rt_arg_index(ReaderRtArg::InputPageEnd)] =
                    page_geometry.input_page_base + input_tile_id_end;
                reader_rt_args[rt_arg_index(ReaderRtArg::ReadySemaphore)] = ready_sem.address();
                reader_rt_args[rt_arg_index(ReaderRtArg::DataValidSemaphore)] = data_valid_sem.address();
                reader_rt_args[rt_arg_index(ReaderRtArg::OutputChunksPerStripe)] = output_chunks_per_stripe;
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                std::vector<uint32_t> writer_rt_args(rt_arg_index(WriterRtArg::Count));
                writer_rt_args[rt_arg_index(WriterRtArg::OutputAddress)] = output_addr;
                writer_rt_args[rt_arg_index(WriterRtArg::InitialStripe)] = device_idx;
                writer_rt_args[rt_arg_index(WriterRtArg::StripeStep)] = stripe_step;
                writer_rt_args[rt_arg_index(WriterRtArg::NumIters)] = num_iters;
                writer_rt_args[rt_arg_index(WriterRtArg::SliceStart)] = local_output_start;
                writer_rt_args[rt_arg_index(WriterRtArg::SliceCount)] = num_worker_output_chunks;
                writer_rt_args[rt_arg_index(WriterRtArg::FinalStart)] = final_start;
                writer_rt_args[rt_arg_index(WriterRtArg::FinalCount)] = final_count;
                writer_rt_args[rt_arg_index(WriterRtArg::DoLocalWrite)] = do_local_write ? 1u : 0u;
                writer_rt_args[rt_arg_index(WriterRtArg::ReadySemaphore)] = ready_sem.address();
                writer_rt_args[rt_arg_index(WriterRtArg::ReadyNocX)] = static_cast<uint32_t>(partner_core.x);
                writer_rt_args[rt_arg_index(WriterRtArg::ReadyNocY)] = static_cast<uint32_t>(partner_core.y);
                writer_rt_args[rt_arg_index(WriterRtArg::DataValidSemaphore)] = data_valid_sem.address();
                writer_rt_args[rt_arg_index(WriterRtArg::DataValidNocX)] = static_cast<uint32_t>(mirror_core.x);
                writer_rt_args[rt_arg_index(WriterRtArg::DataValidNocY)] = static_cast<uint32_t>(mirror_core.y);
                writer_rt_args[rt_arg_index(WriterRtArg::NumGranularSends)] = num_granular;
                writer_rt_args[rt_arg_index(WriterRtArg::DataValidGranularity)] = data_valid_granularity;
                writer_rt_args[rt_arg_index(WriterRtArg::NeighborDeviceId)] =
                    static_cast<uint32_t>(neighbor_node.chip_id);
                writer_rt_args[rt_arg_index(WriterRtArg::NeighborMeshId)] =
                    static_cast<uint32_t>(*neighbor_node.mesh_id);
                writer_rt_args[rt_arg_index(WriterRtArg::OutputChunksPerStripe)] = output_chunks_per_stripe;
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
        .ready_sem = ready_sem,
        .data_valid_sem = data_valid_sem,
        .num_links = num_links,
        .workers_per_direction = workers_per_dir,
        .num_devices = num_devices,
        .device_idx = device_idx,
        .forward_iterations = fwd_iters,
        .backward_iterations = bwd_iters,
        .num_dram_banks = num_dram_banks,
        .is_ring = is_ring,
        .ring_even_split = ring_even_split,
        .output_bank_owned_schedule = output_bank_owned_schedule,
    };

    return {std::move(program), std::move(shared_variables)};
}

void HighBwAllGatherUnicastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const HighBwAllGatherParams& operation_attributes,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();
    const bool has_runtime_controls =
        operation_attributes.input_batch_index.has_value() || operation_attributes.gathered_dim_size.has_value();
    const auto page_geometry = has_runtime_controls
                                   ? derive_page_geometry(tensor_args.input_tensor, output_tensor, operation_attributes)
                                   : PageGeometry{};

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        const uint32_t ready_addr = shared_vars.ready_sem.address();
        const uint32_t data_valid_addr = shared_vars.data_valid_sem.address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        for (const auto& core : shared_vars.worker_cores) {
            auto& reader_args = reader_args_by_core[core.x][core.y];
            reader_args.at(rt_arg_index(ReaderRtArg::InputAddress)) = input_addr;
            reader_args.at(rt_arg_index(ReaderRtArg::OutputAddress)) = output_addr;
            reader_args.at(rt_arg_index(ReaderRtArg::ReadySemaphore)) = ready_addr;
            reader_args.at(rt_arg_index(ReaderRtArg::DataValidSemaphore)) = data_valid_addr;
            auto& writer_args = writer_args_by_core[core.x][core.y];
            writer_args.at(rt_arg_index(WriterRtArg::OutputAddress)) = output_addr;
            writer_args.at(rt_arg_index(WriterRtArg::ReadySemaphore)) = ready_addr;
            writer_args.at(rt_arg_index(WriterRtArg::DataValidSemaphore)) = data_valid_addr;
        }

        if (!has_runtime_controls) {
            continue;
        }

        // Patch only scalar runtime arguments: no program rebuild, worker re-selection, allocation, or
        // tensor view/slice is involved on a cache hit. The compiled schedule may be bank-owned; in that
        // case each worker keeps one output DRAM bank while the selected input base and active count vary.
        const uint32_t total_slices = shared_vars.num_links * shared_vars.workers_per_direction;
        const uint32_t data_valid_granularity =
            derive_data_valid_granularity(page_geometry, operation_attributes.packet_size, total_slices);
        const uint32_t input_pages_per_slice = page_geometry.num_input_pages / total_slices;
        const uint32_t remainder = page_geometry.num_input_pages % total_slices;
        const uint32_t slice_step = shared_vars.output_bank_owned_schedule ? shared_vars.num_dram_banks : 1;
        for (uint32_t link = 0; link < shared_vars.num_links; ++link) {
            for (uint32_t dir = 0; dir < 2; ++dir) {
                const bool is_forward = dir == 0;
                const uint32_t num_recv =
                    shared_vars.is_ring
                        ? shared_vars.num_devices / 2
                        : (is_forward ? shared_vars.device_idx : shared_vars.num_devices - 1 - shared_vars.device_idx);
                for (uint32_t w = 0; w < shared_vars.workers_per_direction; ++w) {
                    const uint32_t slice_idx = link * shared_vars.workers_per_direction + w;
                    uint32_t input_page_start = slice_idx * input_pages_per_slice + std::min(slice_idx, remainder);
                    uint32_t worker_input_page_count = input_pages_per_slice + (slice_idx < remainder ? 1u : 0u);
                    if (shared_vars.output_bank_owned_schedule) {
                        const auto bank_owned_slice = scheduler::derive_bank_owned_slice(
                            page_geometry.num_input_pages,
                            shared_vars.num_links,
                            shared_vars.workers_per_direction,
                            shared_vars.num_dram_banks,
                            link,
                            w);
                        input_page_start = bank_owned_slice.input_page_start;
                        worker_input_page_count = bank_owned_slice.page_count;
                    }
                    const uint32_t input_page_end =
                        shared_vars.output_bank_owned_schedule
                            ? input_page_start + worker_input_page_count * shared_vars.num_dram_banks
                            : (slice_idx + 1) * input_pages_per_slice + std::min(slice_idx + 1, remainder);
                    const uint32_t local_output_start =
                        shared_vars.output_bank_owned_schedule
                            ? input_page_start
                            : (static_cast<uint64_t>(input_page_start) * page_geometry.num_output_chunks) /
                                  page_geometry.num_input_pages;
                    const uint32_t local_output_end =
                        shared_vars.output_bank_owned_schedule
                            ? local_output_start + worker_input_page_count
                            : (static_cast<uint64_t>(input_page_end) * page_geometry.num_output_chunks) /
                                  page_geometry.num_input_pages;
                    const uint32_t slice_count = local_output_end - local_output_start;
                    const uint32_t half = slice_count / 2;
                    const uint32_t final_start =
                        shared_vars.ring_even_split
                            ? (is_forward ? local_output_start : local_output_start + half * slice_step)
                            : local_output_start;
                    const uint32_t final_count =
                        shared_vars.ring_even_split ? (is_forward ? half : slice_count - half) : slice_count;
                    const uint32_t total_chunks =
                        num_recv * slice_count - (shared_vars.ring_even_split ? slice_count - final_count : 0);
                    const auto& core =
                        shared_vars.worker_cores[(link * 2 + dir) * shared_vars.workers_per_direction + w];

                    auto& reader_args = reader_args_by_core[core.x][core.y];
                    reader_args.at(rt_arg_index(ReaderRtArg::TotalChunks)) = total_chunks;
                    reader_args.at(rt_arg_index(ReaderRtArg::SliceStart)) = local_output_start;
                    reader_args.at(rt_arg_index(ReaderRtArg::SliceCount)) = slice_count;
                    reader_args.at(rt_arg_index(ReaderRtArg::FinalStart)) = final_start;
                    reader_args.at(rt_arg_index(ReaderRtArg::FinalCount)) = final_count;
                    reader_args.at(rt_arg_index(ReaderRtArg::InputPageStart)) =
                        page_geometry.input_page_base + input_page_start;
                    reader_args.at(rt_arg_index(ReaderRtArg::InputPageEnd)) =
                        page_geometry.input_page_base + input_page_end;
                    reader_args.at(rt_arg_index(ReaderRtArg::OutputChunksPerStripe)) =
                        page_geometry.output_chunks_per_stripe;

                    auto& writer_args = writer_args_by_core[core.x][core.y];
                    writer_args.at(rt_arg_index(WriterRtArg::SliceStart)) = local_output_start;
                    writer_args.at(rt_arg_index(WriterRtArg::SliceCount)) = slice_count;
                    writer_args.at(rt_arg_index(WriterRtArg::FinalStart)) = final_start;
                    writer_args.at(rt_arg_index(WriterRtArg::FinalCount)) = final_count;
                    writer_args.at(rt_arg_index(WriterRtArg::OutputChunksPerStripe)) =
                        page_geometry.output_chunks_per_stripe;
                    writer_args.at(rt_arg_index(WriterRtArg::DataValidGranularity)) = data_valid_granularity;
                }
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::high_bw_all_gather
