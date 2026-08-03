// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_direct_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include <algorithm>
#include <set>
#include <vector>

namespace ttnn::experimental::prim {

using namespace ::ttnn::ccl;

namespace {
// Prefixed because the ccl ops share a unity-build translation unit with their siblings.
constexpr uint32_t direct_cb_send_id = tt::CBIndex::c_0;    // local input slices queued for the fabric
constexpr uint32_t direct_cb_reduce_id = tt::CBIndex::c_1;  // num_devices blocks per chunk (own + arrivals)
constexpr uint32_t direct_cb_out_id = tt::CBIndex::c_16;    // reduced result (compute -> writer)
}  // namespace

bool reduce_scatter_direct_fabric_supported(
    const ttnn::MeshDevice& mesh_device,
    tt::tt_fabric::FabricConfig fabric_config,
    tt::tt_fabric::Topology axis_topology) {
    if (!::tt::tt_fabric::is_2d_fabric_config(fabric_config)) {
        return true;  // 1D fabric: the case the kernels were written for
    }
    // See the header: on a 2D fabric the ACTIVE AXIS must be a straight wrapping line of the mesh.
    // cluster_axis already confines the collective to one axis, so the mesh's other extent is irrelevant
    // -- a 2x4 with a torus X axis is four independent 4-rings, each exactly the degenerate case the
    // kernels want. What is NOT allowed is a 1xN logical VIEW that snakes across a larger physical grid:
    // there the "ring" mixes X and Y hops, and 2D destination-node routing cannot agree with our
    // hop-count/direction model. A torus axis_topology is precisely the signal that the axis wraps within
    // its own dimension, so it is the whole test.
    (void)mesh_device;
    return ::tt::tt_fabric::is_ring_or_torus(axis_topology);
}

uint32_t reduce_scatter_direct_active_axis(const ReduceScatterMinimalDirectParams& args) {
    uint32_t active_axis = 0;
    for (uint32_t a = 0; a < 2; ++a) {
        if (args.axis_num_devices[a] > 1) {
            active_axis = a;
        }
    }
    return args.cluster_axis.value_or(active_axis);
}

ReduceScatterDirectGeometry reduce_scatter_direct_geometry(
    const ReduceScatterMinimalDirectParams& args, const ttnn::Tensor& input_tensor, bool fp32_dest_acc_en) {
    const uint32_t num_devices = args.num_devices;
    TT_FATAL(input_tensor.layout() == ttnn::TILE_LAYOUT, "reduce_scatter_minimal_direct requires TILE layout");

    const auto padded_shape = input_tensor.padded_shape();
    const uint32_t rank = padded_shape.rank();
    TT_FATAL(rank >= 2, "reduce_scatter_minimal_direct requires a rank >= 2 input, got rank {}", rank);
    TT_FATAL(
        args.dim >= 0 && args.dim < static_cast<int32_t>(rank),
        "scatter dim {} out of range for rank {}",
        args.dim,
        rank);
    const uint32_t dim = static_cast<uint32_t>(args.dim);

    // Size of dim `d` in PAGES: the two innermost dims are counted in tiles, the rest in elements.
    const auto tile = input_tensor.tensor_spec().tile();
    const auto dim_pages = [&](uint32_t d) -> uint32_t {
        if (d == rank - 1) {
            return padded_shape[d] / tile.get_width();
        }
        if (d == rank - 2) {
            return padded_shape[d] / tile.get_height();
        }
        return padded_shape[d];
    };

    // Only the two innermost dims can carry tile padding, and scattering a padded dim would hand the
    // last device a slice of padding rather than data -- the op slices in page space.
    TT_FATAL(
        dim < rank - 2 || padded_shape[dim] == input_tensor.logical_shape()[dim],
        "scatter dim {} is tile-padded (logical {} vs padded {}); slice boundaries would not line up with "
        "the logical tensor",
        args.dim,
        input_tensor.logical_shape()[dim],
        padded_shape[dim]);

    const uint32_t dim_size_pages = dim_pages(dim);
    TT_FATAL(
        dim_size_pages % num_devices == 0,
        "scatter dim {} (padded size {}, {} pages) must split into {} whole-page slices",
        args.dim,
        padded_shape[dim],
        dim_size_pages,
        num_devices);

    uint32_t inner_pages = 1;  // pages under one index of the scatter dim
    for (uint32_t d = dim + 1; d < rank; ++d) {
        inner_pages *= dim_pages(d);
    }

    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    TT_FATAL(
        num_input_pages % num_devices == 0, "input pages {} must divide num_devices {}", num_input_pages, num_devices);

    // Chunk / packet sizing, matching the ring ops' contiguous path (a chunk = tile_granularity tiles
    // stored contiguously, so one contribution chunk is one coalesced fabric write; the granularity is
    // capped by what DST can hold for the reduce). Computed here rather than taken from
    // reduce_scatter_ring_interm_staging_params: that helper's chunk COUNTS are per-4D-channel, so
    // consuming it would drag this op's ND support back through a canonical-4D mapping it does not need.
    const uint32_t single_tile_bytes = input_tensor.buffer()->page_size();
    const uint32_t interm_tiles_per_packet =
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes()) / single_tile_bytes;
    const uint32_t max_dst_size = fp32_dest_acc_en ? 4u : 8u;
    const uint32_t tile_granularity = std::min(4u * std::min(4u, interm_tiles_per_packet), max_dst_size);

    const uint32_t pages_per_slice = num_input_pages / num_devices;
    return ReduceScatterDirectGeometry{
        .single_tile_bytes = single_tile_bytes,
        .tile_granularity = tile_granularity,
        .interm_tiles_per_packet = interm_tiles_per_packet,
        .page_bytes = tile_granularity * single_tile_bytes,
        .pages_per_slice = pages_per_slice,
        .chunks_per_slice = (pages_per_slice + tile_granularity - 1) / tile_granularity,
        .slice_run_pages = (dim_size_pages / num_devices) * inner_pages,
        .stride_pages = dim_size_pages * inner_pages,
    };
}

uint32_t reduce_scatter_direct_num_links(const ReduceScatterMinimalDirectParams& args, uint32_t chunks_per_slice) {
    // Clamped to chunks_per_slice so no worker is handed zero chunks.
    return std::min(std::max(1u, args.num_links), chunks_per_slice);
}

std::pair<CoreRangeSet, std::vector<CoreCoord>> reduce_scatter_direct_worker_cores(
    const ReduceScatterMinimalDirectParams& args, ttnn::MeshDevice* mesh_device, uint32_t chunks_per_slice) {
    const uint32_t num_links = reduce_scatter_direct_num_links(args, chunks_per_slice);
    auto [all_core_range, worker_cores_vec] = ttnn::ccl::choose_worker_cores(
        num_links,
        /*num_cores_per_link=*/1,
        mesh_device,
        args.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        args.sub_core_grid);
    TT_FATAL(
        worker_cores_vec.size() == num_links,
        "reduce_scatter_minimal_direct needs {} worker cores (one per link) but got {}",
        num_links,
        worker_cores_vec.size());

    std::set<CoreRange> worker_core_set;
    for (const auto& c : worker_cores_vec) {
        worker_core_set.emplace(c);
    }
    return {CoreRangeSet(worker_core_set), std::move(worker_cores_vec)};
}

////////////////////////////////////////////////////////////////
// Direct (one-shot) reduce-scatter.
//
// Ring devices 0..N-1. Output slice j (on device j) = sum_k X_k^j. Every device sends slice j straight to
// device j as a single multi-hop fabric unicast (num_hops = ring distance, nearest direction), landing in
// device j's staging slot indexed by the SENDER, with a fused atomic inc on the last packet. Once all N-1
// arrival counters have advanced, the destination reduces the N-1 arrivals together with its own slice and
// writes the output. No device ever relays or accumulates another device's partial: latency is one fabric
// traversal instead of the ring's N/2 store-and-forward steps, at ~2.3x the link traffic.
//
// One worker core per fabric link, owning that link's forward AND backward connection, plus a contiguous
// chunk sub-range of every slice (the only place parallelism exists -- every contribution is one hop).
// Staging is double-buffered by invocation parity so a device one invocation ahead cannot clobber data we
// have not reduced yet; see the reader kernel for the full argument.
////////////////////////////////////////////////////////////////

ReduceScatterMinimalDirectMeshWorkloadFactory::cached_mesh_workload_t
ReduceScatterMinimalDirectMeshWorkloadFactory::create_mesh_workload(
    const ReduceScatterMinimalDirectParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ReduceScatterMinimalDirectInputs& tensor_args,
    tensor_return_value_t& output_tensors) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(operation_attributes.sub_core_grid.value());
    }
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // One arrival counter per SOURCE device (so every counter has exactly one sender -- an absolute wait on
    // it can never be satisfied by a different device that raced ahead), plus the reader's and writer's
    // private invocation counters. Allocate in L1_SMALL when available.
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    std::vector<tt::tt_metal::GlobalSemaphore> arrival_sems;
    arrival_sems.reserve(operation_attributes.num_devices);
    for (uint32_t s = 0; s < operation_attributes.num_devices; ++s) {
        arrival_sems.push_back(
            ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    }
    auto reader_gen_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto writer_gen_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto compute_gen_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    // Start-barrier counter, only consumed when the writer's init sync is compiled in (below).
    auto init_sync_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    // The writer's start barrier is needed only when the op allocates a buffer a peer writes into, since
    // then its address is not pinned across invocations -- see the writer kernel's barrier comment. Both
    // buffers gate it: staging is the address peers actually target, and the output is included because
    // it is the caller's declaration that this call reuses a stable buffer set. Baked in as a compile-time
    // arg so the fully-persistent path pays nothing; compute_program_hash folds the flag in, so the two
    // variants can never share a cached program.
    const bool needs_init_sync =
        !(tensor_args.persistent_output_tensor.has_value() && tensor_args.persistent_staging_tensor.has_value());

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            output_tensors,
            arrival_sems,
            reader_gen_sem,
            writer_gen_sem,
            compute_gen_sem,
            init_sync_sem,
            needs_init_sync);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

ReduceScatterMinimalDirectMeshWorkloadFactory::cached_program_t
ReduceScatterMinimalDirectMeshWorkloadFactory::create_at(
    const ReduceScatterMinimalDirectParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const ReduceScatterMinimalDirectInputs& tensor_args,
    const tensor_return_value_t& output_tensors,
    const std::vector<tt::tt_metal::GlobalSemaphore>& arrival_sems,
    const tt::tt_metal::GlobalSemaphore& reader_gen_sem,
    const tt::tt_metal::GlobalSemaphore& writer_gen_sem,
    const tt::tt_metal::GlobalSemaphore& compute_gen_sem,
    const tt::tt_metal::GlobalSemaphore& init_sync_sem,
    bool needs_init_sync) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensor = output_tensors.at(0);
    const auto& staging = output_tensors.at(1);
    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();

    const uint32_t axis = reduce_scatter_direct_active_axis(operation_attributes);
    const auto topology = operation_attributes.axis_topology[axis];
    TT_FATAL(
        reduce_scatter_direct_fabric_supported(*mesh_device, operation_attributes.fabric_config, topology),
        "reduce_scatter_minimal_direct needs a 1D fabric, or a 2D fabric that collapses to one wrapping "
        "line (torus axis + 1xN/Nx1 mesh); got fabric {} on a {} mesh with axis topology {}",
        operation_attributes.fabric_config,
        mesh_device->shape(),
        topology);
    TT_FATAL(
        tt::tt_fabric::is_ring_or_torus(topology),
        "reduce_scatter_minimal_direct supports Ring topology only (a line would need per-destination "
        "direction clamping)");

    const uint32_t num_devices = operation_attributes.num_devices;
    const uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    auto fwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, 1, topology, axis);
    auto bwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, -1, topology, axis);
    TT_FATAL(fwd_coord.has_value() && bwd_coord.has_value(), "ring must have both neighbors");

    // Scatter geometry in page space -- the scatter dim only shows up as the (run, stride) pair the
    // reader walks a slice with. See ReduceScatterDirectGeometry.
    const bool fp32_dest_acc_en = (input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32);
    const auto geom = reduce_scatter_direct_geometry(operation_attributes, input_tensor, fp32_dest_acc_en);
    const uint32_t pages_per_slice = geom.pages_per_slice;  // tiles per slice (== output tiles)
    const uint32_t slice_run_pages = geom.slice_run_pages;
    const uint32_t stride_pages = geom.stride_pages;
    const uint32_t single_tile_bytes = geom.single_tile_bytes;
    const uint32_t tile_granularity = geom.tile_granularity;                // tiles per chunk
    const uint32_t interm_tiles_per_packet = geom.interm_tiles_per_packet;  // tiles per fabric packet
    const uint32_t chunks_per_slice = geom.chunks_per_slice;
    TT_FATAL(interm_tiles_per_packet >= 1, "a tile must fit in a fabric packet");
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    TT_FATAL(
        geom.page_bytes % dram_alignment == 0,
        "staging page_bytes {} must be a multiple of DRAM alignment {}",
        geom.page_bytes,
        dram_alignment);

    // --- Core selection: one worker core per link. Unlike the store-and-forward ring op there is no
    // per-direction core: a contribution's direction is a property of its DESTINATION, so a single worker
    // owns both of its link's connections and fans out to every destination itself. Placement is
    // deterministic, so worker `link` sits at the same logical coords on every device -- which is what lets
    // a sender target the mirror core's staging slot + arrival counter without any exchange.
    //
    // MEASURED (2026-07-28, rs_8K): splitting the destination list across extra "helper" sender cores on
    // the spare (link, direction) channels made it WORSE (device crit-path 7.7 -> 8.5us, within-iter spread
    // 0.9 -> 1.9us). Every added core pays the per-core fixed cost (kernel launch + fabric connection
    // open/close) and the op's duration is the max over its cores, so shortening the reducer's serial send
    // work by ~3 units does not pay for it. Do not re-add destination-parallel senders without first
    // shrinking that fixed cost.
    const uint32_t num_links = reduce_scatter_direct_num_links(operation_attributes, chunks_per_slice);
    auto [worker_core_range, worker_cores_vec] =
        reduce_scatter_direct_worker_cores(operation_attributes, mesh_device, chunks_per_slice);

    // Chunk partition across the per-link workers: worker l gets base_chunks (+1 for the first `extra`).
    // Only a slice's very last chunk can be partial, so only the last worker's tile_count is ever clipped.
    const uint32_t base_chunks = chunks_per_slice / num_links;
    const uint32_t extra_chunks = chunks_per_slice % num_links;

    // --- Circular buffers (real-tile granularity).
    //
    // ARRIVALS_IN_CB (staging is L1-sharded): cb_reduce IS this core's staging shard, so a sender's fabric
    // packet lands directly in the slot the reducer unpacks from and nothing reads staging on the receive
    // side. That forces the shard's exact geometry -- 2 parity halves x chunks_per_slice chunks x
    // num_devices blocks x tile_granularity tiles -- since the CB read pointer advances one N-block group
    // per chunk and the parity half is reached as a constant tile-index offset on top of it.
    //
    // Otherwise cb_reduce is an ordinary double-buffered CB the reader fills by reading staging over the
    // NoC (the fallback for shapes whose shard will not fit one core).
    const bool arrivals_in_cb = staging.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const uint32_t half_stride_tiles = chunks_per_slice * num_devices * tile_granularity;
    const uint32_t cb_reduce_pages = arrivals_in_cb ? 2 * half_stride_tiles : 2 * num_devices * tile_granularity;
    // The parity offset every kernel applies. Zero off the aliased path: there cb_reduce is an ordinary
    // two-group CB with no parity halves to choose between, and a non-zero offset would push the reader
    // and the reducer clean off the end of it on odd invocations.
    const uint32_t parity_stride_tiles = arrivals_in_cb ? half_stride_tiles : 0u;

    const uint32_t cb_page_size = single_tile_bytes;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    for (auto [cb_id, num_pages] : std::initializer_list<std::pair<uint32_t, uint32_t>>{
             {direct_cb_send_id, 2 * tile_granularity}, {direct_cb_out_id, 2 * tile_granularity}}) {
        tt::tt_metal::CircularBufferConfig cfg =
            tt::tt_metal::CircularBufferConfig(num_pages * cb_page_size, {{cb_id, df}})
                .set_page_size(cb_id, cb_page_size);
        CreateCircularBuffer(program, worker_core_range, cfg);
    }

    tt::tt_metal::CircularBufferConfig reduce_cb_cfg =
        tt::tt_metal::CircularBufferConfig(cb_reduce_pages * cb_page_size, {{direct_cb_reduce_id, df}})
            .set_page_size(direct_cb_reduce_id, cb_page_size);
    if (arrivals_in_cb) {
        TT_FATAL(
            staging.buffer()->aligned_page_size() * staging.buffer()->shard_spec().shape()[0] ==
                cb_reduce_pages * cb_page_size,
            "sharded staging shard ({} rows x {} B) must be exactly the reduce CB ({} tiles x {} B); the "
            "staging spec and the factory disagree on the shard geometry",
            staging.buffer()->shard_spec().shape()[0],
            staging.buffer()->aligned_page_size(),
            cb_reduce_pages,
            cb_page_size);
        reduce_cb_cfg = reduce_cb_cfg.set_globally_allocated_address(*staging.buffer());
    }
    const auto reduce_cb_handle = CreateCircularBuffer(program, worker_core_range, reduce_cb_cfg);

    // --- Kernels ---
    std::vector<uint32_t> reader_ct_args = {
        single_tile_bytes,
        tile_granularity,
        chunks_per_slice,
        pages_per_slice,
        slice_run_pages,
        stride_pages,
        num_devices,
        direct_cb_send_id,
        direct_cb_reduce_id,
        (uint32_t)arrivals_in_cb,
        parity_stride_tiles};
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);  // tiled
    tt::tt_metal::TensorAccessorArgs(staging.buffer()).append_to(reader_ct_args);       // chunk-paged

    std::vector<uint32_t> writer_ct_args = {
        single_tile_bytes,
        tile_granularity,
        chunks_per_slice,
        pages_per_slice,
        num_devices,
        interm_tiles_per_packet,
        direct_cb_send_id,
        direct_cb_out_id,
        (uint32_t)arrivals_in_cb,
        parity_stride_tiles,
        (uint32_t)needs_init_sync};
    tt::tt_metal::TensorAccessorArgs(staging.buffer()).append_to(writer_ct_args);        // chunk-paged
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);  // tiled

    std::vector<uint32_t> compute_ct_args = {
        tile_granularity, num_devices, direct_cb_reduce_id, direct_cb_out_id, parity_stride_tiles};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/"
        "reduce_scatter_minimal_direct_reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/"
        "reduce_scatter_minimal_direct_writer.cpp",
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/"
        "reduce_scatter_minimal_direct_compute.cpp",
        worker_core_range,
        tt::tt_metal::ComputeConfig{.compile_args = compute_ct_args});

    // --- Destination order: farthest ring distance first, since a destination cannot start reducing until
    // its last contribution lands. Direction is whichever way round the ring is shorter (ties -> forward,
    // which also puts an even ring's antipode on the forward link).
    struct Dest {
        uint32_t slice;  // destination device == the input slice it wants
        uint32_t conn;   // 0 = forward connection, 1 = backward
        uint32_t hops;
        // Destination chip/mesh, for 2D fabric. 1D routing takes the distance from the packet header's
        // hop count, but a 2D fabric routes by DESTINATION NODE, so the header carries a route programmed
        // with fabric_set_unicast_route (see the writer). Resolved here because only the host knows the
        // ring's physical layout.
        uint32_t chip_id;
        uint32_t mesh_id;
    };
    std::vector<Dest> dests;
    dests.reserve(num_devices - 1);
    for (uint32_t j = 0; j < num_devices; ++j) {
        if (j == device_idx) {
            continue;
        }
        const uint32_t fwd_hops = (j + num_devices - device_idx) % num_devices;
        const uint32_t bwd_hops = (device_idx + num_devices - j) % num_devices;
        const bool use_fwd = fwd_hops <= bwd_hops;
        const uint32_t hops = use_fwd ? fwd_hops : bwd_hops;
        // Walk `hops` steps in the chosen direction to land on device j's coordinate.
        const auto dest_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, use_fwd ? static_cast<int>(hops) : -static_cast<int>(hops), topology, axis);
        TT_FATAL(dest_coord.has_value(), "ring neighbour {} hops away must exist", hops);
        const auto dest_node = mesh_device->get_fabric_node_id(*dest_coord);
        dests.push_back(Dest{
            j,
            use_fwd ? 0u : 1u,
            hops,
            dest_node.chip_id,
            static_cast<uint32_t>(*dest_node.mesh_id)});
    }
    // On a 2D fabric, WE do not get to choose the direction. The header carries an absolute route from
    // the routing table (fabric_set_unicast_route -> decode_route_to_buffer), and a packet must be handed
    // to the router in that route's first-hop direction; push it into the other connection and it enters a
    // router carrying a foreign route, which hangs. Our ring arithmetic above picks the shorter way round,
    // which for an even ring's antipode (and anywhere the table breaks a tie the other way) can disagree.
    // So on 2D, re-derive conn from the fabric's own answer. Purely a direction fix -- `hops` stays as
    // computed, since 2D ignores it and 1D never reaches this block.
    if (::tt::tt_fabric::is_2d_fabric_config(operation_attributes.fabric_config)) {
        const auto self_node = mesh_device->get_fabric_node_id(sender_device_coord);
        const auto fwd_dir = tt::tt_fabric::pipeline_get_forwarding_direction(
            self_node, mesh_device->get_fabric_node_id(*fwd_coord));
        const auto bwd_dir = tt::tt_fabric::pipeline_get_forwarding_direction(
            self_node, mesh_device->get_fabric_node_id(*bwd_coord));

        // The axis_topology we were handed is not trustworthy on its own: it binds a MESH-VIEW axis index
        // to a fabric dimension (axis 1 -> X, axis 0 -> Y), while a torus fabric config wraps a PHYSICAL
        // dimension. Open this box's 2x4 as 4x2 and the two disagree -- FABRIC_2D_TORUS_Y makes axis 0
        // report Torus, but what it physically wraps is the length-2 dimension, not the 4-ring the
        // collective runs on. The op then believes in a wrap that does not exist and hangs.
        //
        // So verify the wrap for real: a ring neighbour must be reachable in ONE hop (an actual cable in
        // that direction), not by a long way round. Cheap, and it turns that hang into a clear message.
        auto assert_single_hop = [&](const ttnn::MeshCoordinate& neighbor_coord,
                                     const std::optional<tt::tt_fabric::RoutingDirection>& dir,
                                     const char* which) {
            const auto neighbor = mesh_device->get_fabric_node_id(neighbor_coord);
            TT_FATAL(
                dir.has_value(),
                "reduce_scatter_minimal_direct: no fabric route to the {} ring neighbour (chip {})",
                which,
                neighbor.chip_id);
            const auto neighbors = tt::tt_fabric::pipeline_get_chip_neighbors(self_node, *dir);
            const auto it = neighbors.find(*neighbor.mesh_id);
            const bool adjacent =
                it != neighbors.end() && std::find(it->second.begin(), it->second.end(), neighbor.chip_id) !=
                                             it->second.end();
            TT_FATAL(
                adjacent,
                "reduce_scatter_minimal_direct: the {} ring neighbour (chip {}) is not one hop away in "
                "direction {}, so the active axis does not physically wrap even though its topology reports "
                "as a torus. This happens when the mesh VIEW's axis order does not match the fabric "
                "dimension the torus config wraps -- e.g. this 2x4 box opened as 4x2 under TORUS_Y. Open "
                "the mesh so the collective's axis is the dimension the config actually wraps.",
                which,
                neighbor.chip_id,
                static_cast<int>(*dir));
        };
        assert_single_hop(*fwd_coord, fwd_dir, "forward");
        assert_single_hop(*bwd_coord, bwd_dir, "backward");

        for (auto& d : dests) {
            const auto dir = tt::tt_fabric::pipeline_get_forwarding_direction(
                self_node, tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{d.mesh_id}, d.chip_id));
            TT_FATAL(
                dir.has_value(),
                "no fabric route from this device to ring member {} (chip {}) on a 2D fabric",
                d.slice,
                d.chip_id);
            if (fwd_dir.has_value() && *dir == *fwd_dir) {
                d.conn = 0;
            } else if (bwd_dir.has_value() && *dir == *bwd_dir) {
                d.conn = 1;
            } else {
                TT_THROW(
                    "reduce_scatter_minimal_direct: the fabric routes to ring member {} (chip {}) via direction "
                    "{}, which is neither of this device's ring-neighbour directions (fwd {}, bwd {}). The ring "
                    "is not a straight line in one fabric dimension, so destination-node routing cannot follow "
                    "it.",
                    d.slice,
                    d.chip_id,
                    static_cast<int>(*dir),
                    fwd_dir.has_value() ? std::to_string(static_cast<int>(*fwd_dir)) : std::string("none"),
                    bwd_dir.has_value() ? std::to_string(static_cast<int>(*bwd_dir)) : std::string("none"));
            }
        }
    }
    std::stable_sort(dests.begin(), dests.end(), [](const Dest& a, const Dest& b) { return a.hops > b.hops; });
    const bool uses_backward = std::any_of(dests.begin(), dests.end(), [](const Dest& d) { return d.conn == 1; });
    const uint32_t num_connections = uses_backward ? 2u : 1u;

    // Start-barrier multicast ranges, derived from the same destination split so they cannot drift from
    // it: nearest-direction routing gives each direction a set of hops that is contiguous from 1, so a
    // direction's range is just its destination count, and the two together cover every peer exactly once.
    uint32_t mcast_range[2] = {0u, 0u};
    uint32_t max_hops[2] = {0u, 0u};
    for (const auto& d : dests) {
        ++mcast_range[d.conn];
        max_hops[d.conn] = std::max(max_hops[d.conn], d.hops);
    }
    // 1D only: on 2D the barrier sends one unicast per peer (reusing the data path's per-destination
    // routes), so it needs no contiguous hop ranges -- and the direction fix-up above can legitimately
    // split the destinations in a way these would reject.
    if (!::tt::tt_fabric::is_2d_fabric_config(operation_attributes.fabric_config)) {
        for (uint32_t c = 0; c < 2; ++c) {
            TT_FATAL(
                mcast_range[c] == max_hops[c],
                "start-barrier multicast assumes direction {}'s destinations are hops 1..{} with no gaps, but the "
                "{} destinations routed that way reach out to {} hops",
                c,
                mcast_range[c],
                mcast_range[c],
                max_hops[c]);
        }
        TT_FATAL(
            mcast_range[1] == 0 || num_connections == 2,
            "start barrier would multicast backward on a connection that was never opened");
    }

    // --- Runtime args ---
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();
    const uint32_t staging_addr = staging.buffer()->address();

    for (uint32_t link = 0; link < num_links; ++link) {
        const CoreCoord core = worker_cores_vec[link];
        const uint32_t chunk_start = link * base_chunks + std::min(link, extra_chunks);
        const uint32_t chunk_count = base_chunks + (link < extra_chunks ? 1u : 0u);
        const uint32_t tile_start = chunk_start * tile_granularity;
        const uint32_t tile_count = std::min(chunk_count * tile_granularity, pages_per_slice - tile_start);
        // Mirror core: deterministic placement means the peer's worker `link` is at these same coords.
        const CoreCoord peer_core = mesh_device->worker_core_from_logical_core(core);

        std::vector<uint32_t> reader_rt = {
            input_addr,
            staging_addr,
            device_idx,
            chunk_start,
            chunk_count,
            tile_start,
            tile_count,
            reader_gen_sem.address()};
        for (uint32_t s = 0; s < num_devices; ++s) {
            if (s != device_idx) {
                reader_rt.push_back(arrival_sems[s].address());
            }
        }
        for (const auto& d : dests) {
            reader_rt.push_back(d.slice);
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt);

        std::vector<uint32_t> compute_rt = {chunk_count, tile_count, compute_gen_sem.address()};
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, {core}, compute_rt);

        std::vector<uint32_t> writer_rt = {
            staging_addr,
            output_addr,
            device_idx,
            chunk_start,
            chunk_count,
            tile_start,
            tile_count,
            writer_gen_sem.address(),
            arrival_sems[device_idx].address(),  // our source slot's counter, same address on every peer
            (uint32_t)peer_core.x,
            (uint32_t)peer_core.y,
            num_connections,
            init_sync_sem.address(),  // same address on every peer's mirror core
            mcast_range[0],
            mcast_range[1]};
        for (const auto& d : dests) {
            writer_rt.push_back(d.conn);
        }
        for (const auto& d : dests) {
            writer_rt.push_back(d.hops);
        }
        // Our block index in the destination's N-block reduce group. The destination lays its blocks out
        // as [own, then every other device in ascending order], so a source before it shifts up by one.
        // Only the aliased path reads these (off it, the sender indexes staging by source directly).
        for (const auto& d : dests) {
            writer_rt.push_back(device_idx < d.slice ? device_idx + 1 : device_idx);
        }
        // 2D-fabric route targets, ignored by the 1D path. MUST stay last before the fabric connection
        // args -- the writer reads conn, hops, block, chip, mesh in exactly this order.
        for (const auto& d : dests) {
            writer_rt.push_back(d.chip_id);
        }
        for (const auto& d : dests) {
            writer_rt.push_back(d.mesh_id);
        }
        // Fabric connections, appended last: index 0 = forward neighbour, 1 = backward, both on this
        // worker's own routing plane so the links stay independent.
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes = {mesh_device->get_fabric_node_id(*fwd_coord)};
        std::vector<uint32_t> link_indices = {link};
        if (uses_backward) {
            dst_nodes.push_back(mesh_device->get_fabric_node_id(*bwd_coord));
            link_indices.push_back(link);
        }
        append_routing_plane_connection_manager_rt_args(
            sender_fabric_node_id,
            dst_nodes,
            link_indices,
            program,
            writer_kernel_id,
            {core},
            writer_rt,
            tt::tt_fabric::FabricApiType::Linear);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt);
    }

    shared_variables_t shared_variables{
        .worker_cores = worker_cores_vec,
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .arrival_sems = arrival_sems,
        .reader_gen_sem = reader_gen_sem,
        .writer_gen_sem = writer_gen_sem,
        .compute_gen_sem = compute_gen_sem,
        .init_sync_sem = init_sync_sem,
        .reduce_cb_handle = arrivals_in_cb ? std::optional{reduce_cb_handle} : std::nullopt,
    };
    return {std::move(program), std::move(shared_variables)};
}

void ReduceScatterMinimalDirectMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ReduceScatterMinimalDirectParams& /*operation_attributes*/,
    const ReduceScatterMinimalDirectInputs& tensor_args,
    tensor_return_value_t& output_tensors) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensors.at(0).buffer()->address();
    const uint32_t staging_addr = output_tensors.at(1).buffer()->address();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(coordinate_range);
        auto& reader_args_by_core = GetRuntimeArgs(program, sv.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, sv.writer_kernel_id);

        // On the aliased path cb_reduce lives on top of the staging buffer, so it has to follow it.
        if (sv.reduce_cb_handle.has_value()) {
            UpdateDynamicCircularBufferAddress(program, *sv.reduce_cb_handle, *output_tensors.at(1).buffer());
        }

        // Only the tensor addresses move. The semaphores are owned by shared_variables (their addresses are
        // fixed for the cached program's lifetime) and the chunk partition / destination routes are
        // geometry-derived, so both stay valid.
        for (const auto& core : sv.worker_cores) {
            auto& r = reader_args_by_core[core.x][core.y];
            r[0] = input_addr;
            r[1] = staging_addr;
            auto& w = writer_args_by_core[core.x][core.y];
            w[0] = staging_addr;
            w[1] = output_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
