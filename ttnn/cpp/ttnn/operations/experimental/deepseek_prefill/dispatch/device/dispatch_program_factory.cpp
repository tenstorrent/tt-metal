// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_device_operation.hpp"
#include "kernels/dataflow/dispatch_plan.hpp"  // PlanHeader / PlanEntry layout (host sizes the plan CB from these)
#include <algorithm>
#include <array>
#include <utility>
#include <limits>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <ttnn/global_semaphore.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

// ProgramDescriptor-flavored helper.  Pushes a CBDescriptor onto the desc
// instead of calling CreateCircularBuffer.  The FP8 dispatch path allocates its
// buffer as DataType::FP8_E4M3, which datatype_to_dataformat_converter maps
// straight to tt::DataFormat::Fp8_e4m3 — no special-casing needed here.
void create_tensor_cb(
    tt::tt_metal::ProgramDescriptor& desc,
    const CoreRangeSet& core_range_set,
    const ttnn::Tensor& tensor,
    uint32_t buffering_factor,
    tt::CBIndex cb_id,
    const std::string& tensor_name = "tensor") {
    auto page_size = get_page_size(tensor);
    auto num_pages = detail::get_num_pages(tensor);
    auto aligned_page_size = get_aligned_page_size(tensor);
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());

    uint32_t cb_size = buffering_factor * aligned_page_size;

    log_debug(
        tt::LogOp,
        "{} shape: {}, pages: {}, page_size: {}, aligned_page_size: {} buffering_factor: {} cb_id: {} cb_size: {} "
        "cb_dtype: {}",
        tensor_name,
        tensor.logical_shape(),
        num_pages,
        page_size,
        aligned_page_size,
        buffering_factor,
        cb_id,
        cb_size,
        data_format);

    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_size,
        .core_ranges = core_range_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = data_format,
            .page_size = aligned_page_size,
        }}},
    });
}

}  // namespace detail

namespace {

// Pick a routing-plane link index that is VALID for each dispatch-axis neighbor's own forwarding
// direction. get_forwarding_link_indices resolves the forwarding direction first and returns links in
// that direction, so on a ring the wrap-direction neighbor may not share the line direction's valid
// link index. Broadcasting a single {core_link} to every connection would land the wrap connection on
// an EDM plane that never services it -> the worker hangs in open_finish. Indexing core_link into each
// neighbor's own valid-link set keeps the choice valid for that direction while still spreading sender
// cores across links where more than one plane exists. (Shared by the tile and row-major paths below;
// kept file-local because the natural shared home, ccl/common, is outside this op's code ownership.)
std::vector<uint32_t> compute_per_neighbor_forwarding_links(
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    const std::vector<tt::tt_fabric::FabricNodeId>& dst_nodes,
    uint32_t core_link,
    const char* axis_label) {
    std::vector<uint32_t> per_conn_links;
    per_conn_links.reserve(dst_nodes.size());
    for (const auto& dst_node : dst_nodes) {
        const auto links = tt::tt_fabric::get_forwarding_link_indices(src_fabric_node_id, dst_node);
        TT_FATAL(
            !links.empty(), "No forwarding links from {} to {} neighbor {}", src_fabric_node_id, axis_label, dst_node);
        log_debug(
            tt::LogOp,
            "FABRIC_2D {} link select: src={} dst={} dir={} core_link={} valid_links={} -> {}",
            axis_label,
            src_fabric_node_id,
            dst_node,
            tt::tt_fabric::get_eth_forwarding_direction(src_fabric_node_id, dst_node).value(),
            core_link,
            links.size(),
            links[core_link % links.size()]);
        per_conn_links.push_back(links[core_link % links.size()]);
    }
    return per_conn_links;
}

// Unified dispatch path for both TILE and ROW_MAJOR inputs: routing/offsets live on N worker
// cores per sender (num_workers_per_sender, u1..uN) under a baton ring; the sender is fabric-only.
// Tile-layout untilizes tiled input via a compute kernel; row-major reads input rows straight into
// the payload CB and skips compute entirely. Selected internally via is_row_major.
tt::tt_metal::ProgramDescriptor create_dispatch_program(
    const DispatchParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& exit_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    tt::tt_metal::ProgramDescriptor desc;

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.indices_tensor;
    auto offsets_tensor = tensor_args.expert_offsets_tensor;
    auto dispatch_table_tensor = tensor_args.expert_dispatch_table_tensor;

    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);

    // Row-major and tile-layout share this whole worker architecture (baton-ring offsets,
    // plan-based reader/writer, fabric-only sender). The only differences: row-major reads input
    // rows straight into the payload CB instead of untilizing tiled input, so it skips the compute
    // kernel + its signal/output CBs and uses a smaller read batch.
    const bool is_row_major = input_tensor.layout() != tt::tt_metal::Layout::TILE;

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;
    uint32_t num_workers = operation_attributes.num_workers_per_sender;
    TT_FATAL(num_workers >= 1, "num_workers_per_sender must be >= 1; got {}.", num_workers);
    log_debug(
        tt::LogOp,
        "Creating prefill dispatch program ({}) for mesh coordinate: ({}, {}) with topology: {} num_links: {}",
        is_row_major ? "row-major" : "tile",
        mesh_coordinate[0],
        mesh_coordinate[1],
        topology,
        num_links);

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "\nCreating all to all dispatch program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized mesh coord: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord);

    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    constexpr uint32_t MAX_WORKER_CORES = 4;
    uint32_t effective_num_links = std::min(num_links, MAX_WORKER_CORES);
    TT_FATAL(
        subdevice_cores.size() >= effective_num_links,
        "Not enough cores {} for {} links",
        subdevice_cores.size(),
        effective_num_links);

    auto logical_volume = input_tensor.logical_shape().volume();
    auto hidden_size = input_tensor.logical_shape()[-1];
    auto tokens_per_device = logical_volume / hidden_size;

    uint32_t num_cores = effective_num_links;

    // ==================== Core layout: senders + worker cores ====================
    // Collect all cores in the first row (y == subdevice_cores[0].y), sorted by x.
    // Each sender owns num_workers consecutive worker cores (u1..uN): cores are
    // laid out [sender, u1, ..., uN] consecutively along x per sender group.
    uint32_t sender_row_y = subdevice_cores.at(0).y;
    std::vector<CoreCoord> all_row_cores;
    for (const auto& core : subdevice_cores) {
        if (core.y == sender_row_y) {
            all_row_cores.push_back(core);
        }
    }
    std::sort(
        all_row_cores.begin(), all_row_cores.end(), [](const CoreCoord& a, const CoreCoord& b) { return a.x < b.x; });

    uint32_t total_row_cores = static_cast<uint32_t>(all_row_cores.size());
    uint32_t cores_per_sender = 1 + num_workers;  // 1 sender + N worker cores (u1..uN)
    TT_FATAL(
        total_row_cores >= cores_per_sender * num_cores,
        "Same-row has only {} cores for {} senders — need {} cores per sender (>= {} required)",
        total_row_cores,
        num_cores,
        cores_per_sender,
        cores_per_sender * num_cores);

    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(num_cores);
    std::vector<std::vector<CoreCoord>> sender_worker_groups(num_cores);
    std::vector<CoreCoord> all_worker_cores;
    all_worker_cores.reserve(num_cores * num_workers);
    std::vector<uint32_t> worker_sender_map;
    worker_sender_map.reserve(num_cores * num_workers);

    for (uint32_t s = 0; s < num_cores; s++) {
        sender_cores.push_back(all_row_cores[cores_per_sender * s]);
        for (uint32_t u = 0; u < num_workers; u++) {
            CoreCoord uc = all_row_cores[cores_per_sender * s + 1 + u];
            sender_worker_groups[s].push_back(uc);
            all_worker_cores.push_back(uc);
            worker_sender_map.push_back(s);
        }
    }

    uint32_t num_worker_cores = static_cast<uint32_t>(all_worker_cores.size());

    // Build sender_core_grid and worker_core_grid CoreRangeSets
    std::set<CoreRange> sender_ranges_set;
    for (const auto& sc : sender_cores) {
        sender_ranges_set.insert(CoreRange(sc));
    }
    auto sender_core_grid = CoreRangeSet(sender_ranges_set);

    std::set<CoreRange> worker_ranges_set;
    for (const auto& ic : all_worker_cores) {
        worker_ranges_set.insert(CoreRange(ic));
    }
    CoreRangeSet worker_core_grid(worker_ranges_set);

    log_debug(
        tt::LogOp,
        "Dispatch program: num_links: {} num_cores(senders): {} num_worker_cores: {} tokens_per_device: {}",
        num_links,
        num_cores,
        num_worker_cores,
        tokens_per_device);

    // Tile-layout untilizes 32-token batches; row-major reads rows straight into the payload CB
    // 8-at-a-time (read_batch_size==8), double-buffered against the writer drain to keep overlap.
    const uint32_t read_batch_size = is_row_major ? 8u : 32u;

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    uint32_t total_batches = (tokens_per_device + read_batch_size - 1) / read_batch_size;

    // Largest divisor of (hidden_size / 32) that is <= 8.  Mirrors the combine program factory.
    // Avoids the static_assert in llk_pack_untilize when full_ct_dim is not divisible by 8
    const uint32_t full_ct_dim_dispatch = static_cast<uint32_t>(hidden_size) / 32u;
    uint32_t block_ct_dim_dispatch = 8;
    while (full_ct_dim_dispatch % block_ct_dim_dispatch != 0) {
        --block_ct_dim_dispatch;
    }

    // ==================== Semaphores ====================
    // Per-entry credit, each kept on the consumer's L1 (producer NOC-incs):
    //   data_avail  (sender L1, init=0):                worker → sender, "entry ready".
    //   space_avail (worker L1, init=writer_cb_size): sender → worker, "slot freed".
    const uint32_t writer_cb_size = read_batch_size;  // one batch deep (32 tile / 8 row-major)
    uint32_t next_sema_id = 0;
    auto add_sema = [&](const CoreRangeSet& crs, uint32_t init_val = 0) -> uint32_t {
        uint32_t id = next_sema_id++;
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = crs, .initial_value = init_val});
        return id;
    };

    std::vector<uint32_t> data_avail_semaphore_ids(num_workers);
    for (uint32_t s = 0; s < num_workers; s++) {
        data_avail_semaphore_ids[s] = add_sema(sender_core_grid);
    }
    auto space_avail_semaphore_id = add_sema(worker_core_grid, writer_cb_size);
    auto addr_ready_semaphore_id = add_sema(worker_core_grid);
    auto cross_addr_semaphore_id = add_sema(worker_core_grid);
    auto turn_semaphore_id = add_sema(worker_core_grid);

    // ==================== Circular Buffers for worker cores ====================
    // Routing decisions and offsets[] live on the worker core — sender is fabric-only.
    // c_0:
    //   tile-layout: tiled input stripe (reader → compute). buffering_factor MUST be a multiple of
    //     block_ct_dim_dispatch (the reader's per-chunk push size) so untilize blocks never straddle
    //     the CB ring wrap.  2 * block_ct_dim gives double buffering (=16 for the common block_ct_dim=8).
    //   row-major: payload CB the reader fills with input rows and the writer drains (as cb_untilize).
    //     Double-buffered at batch granularity (2 * read_batch_size) so the reader can fetch the next
    //     batch while the writer drains the current one.
    if (is_row_major) {
        // The reader writes input rows here; the writer strides by aligned_output_page_size, so the
        // two page sizes must agree. Row-major requires input dtype == output dtype (bf16 in/out or
        // fp8 in/out), so they always do — the byte-copy path performs no dtype conversion.
        TT_FATAL(
            detail::get_aligned_page_size(input_tensor) == detail::get_aligned_page_size(output_tensor),
            "Row-major dispatch requires matching input/output aligned page sizes ({} vs {}).",
            detail::get_aligned_page_size(input_tensor),
            detail::get_aligned_page_size(output_tensor));
        detail::create_tensor_cb(
            desc,
            worker_core_grid,
            output_tensor,
            /*buffering_factor=*/2 * read_batch_size,
            /*cb_id=*/tt::CBIndex::c_0,
            "rowmajor_payload_scratch");
    } else {
        detail::create_tensor_cb(
            desc,
            worker_core_grid,
            input_tensor,
            /*buffering_factor=*/2 * block_ct_dim_dispatch,
            /*cb_id=*/tt::CBIndex::c_0,
            "worker_input_scratch");
    }
    // c_1: indices scratch (worker reader does per-batch DRAM reads)
    detail::create_tensor_cb(
        desc,
        worker_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "worker_indices_scratch");
    // c_3: offsets (full tensor, mutated in place per batch as the shared running counter).
    // The owner worker core (local_core_id==0) loads expert_offsets[] here once at startup;
    // non-owners leave it uninitialized and pull/push the owner's copy under the baton each
    // batch (see reader_worker_dispatch.cpp). One copy per core — no extra scratch.
    detail::create_tensor_cb(
        desc,
        worker_core_grid,
        offsets_tensor,
        /*buffering_factor=*/detail::get_num_pages(offsets_tensor),
        /*cb_id=*/tt::CBIndex::c_3,
        "worker_offsets_tensor");
    // c_9: dispatch_table (full tensor, loaded once at startup)
    detail::create_tensor_cb(
        desc,
        worker_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),
        /*cb_id=*/tt::CBIndex::c_9,
        "worker_dispatch_table_tensor");
    // c_10 (signal, reader → compute) and c_11 (untilize output, compute → writer) only exist on the
    // tile-layout path. Row-major has no compute kernel — the reader fills c_0 directly and the writer
    // drains it as cb_untilize, so neither CB is allocated.
    if (!is_row_major) {
        // c_10: signal CB (reader → compute)
        {
            uint32_t signal_page_size = l1_alignment;
            constexpr uint32_t signal_buffering = 2;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = signal_buffering * signal_page_size,
                .core_ranges = worker_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                    .data_format = tt::DataFormat::UInt32,
                    .page_size = signal_page_size,
                }}},
            });
        }
        // c_11: untilize output (compute → writer)
        // Double-buffered at batch granularity: two slots of read_batch_size tokens so compute can
        // pack the next batch while the writer is still draining the previous one.
        detail::create_tensor_cb(
            desc,
            worker_core_grid,
            output_tensor,
            /*buffering_factor=*/2 * read_batch_size,
            /*cb_id=*/tt::CBIndex::c_11,
            "worker_untilize_output");
    }
    // c_13: metadata scratch (worker writer builds metadata here before NOC-writing).
    detail::create_tensor_cb(
        desc,
        worker_core_grid,
        metadata_tensor,
        /*buffering_factor=*/(read_batch_size * operation_attributes.num_experts_per_tok) + 1,
        /*cb_id=*/tt::CBIndex::c_13,
        "worker_metadata_scratch");
    // c_15: route_info scratch (16B = l1_alignment). Worker writer builds the 4-u32
    // route_info entry [route, distance, page_idx, dst_chip] here, then NOC-writes the whole
    // block as a single noc_async_write to the sender's c_4 slot (replaces 4× inline_dw).
    // dst_chip is the linearized dest device index used by the sender's 2D fabric route.
    {
        uint32_t route_info_scratch_size = l1_alignment;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = route_info_scratch_size,
            .core_ranges = worker_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_15),
                .data_format = tt::DataFormat::UInt32,
                .page_size = route_info_scratch_size,
            }}},
        });
    }
    // c_14: per-batch route plan (reader RISC → writer RISC, on same worker core).
    // Layout (PlanHeader + PlanEntry[]) is defined in kernels/dataflow/dispatch_plan.hpp:
    // [PlanHeader: 16B][PlanEntry: 48B each] (both alignas(16)). Sized straight from sizeof so the
    // page size always tracks the structs.
    {
        uint32_t max_plan_entries = read_batch_size * operation_attributes.num_experts_per_tok;
        uint32_t plan_page_size = tt::round_up(
            static_cast<uint32_t>(sizeof(PlanHeader) + max_plan_entries * sizeof(PlanEntry)), l1_alignment);
        constexpr uint32_t plan_buffering = 2;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = plan_buffering * plan_page_size,
            .core_ranges = worker_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_14),
                .data_format = tt::DataFormat::UInt32,
                .page_size = plan_page_size,
            }}},
        });
    }

    // ==================== Circular Buffers for SENDER cores ====================
    // Direct-consume pipeline on sender: one writer-CB set (route_info, payload, metadata)
    // per worker. Set s is fed by the worker with local_core_id == s; the sender
    // writer (RISCV_0) polls all N sets round-robin. Worker cores NOC-write per-entry
    // directly into the relevant set (addresses learned via the boot-time handshake).
    //
    // CB index layout: set 0 keeps the legacy c_4/c_5/c_6, set 1 keeps c_16/c_17/c_18, and
    // further sets take consecutive free indices from c_19 up. Large N runs out of CB indices
    // and CB creation asserts — that is the intended "breaks if too many" behaviour.
    std::vector<std::array<uint32_t, 3>> writer_cb_ids(num_workers);  // {route, payload, metadata}
    {
        uint32_t route_info_page_size = l1_alignment;
        uint32_t next_free_cb = static_cast<uint32_t>(tt::CBIndex::c_19);
        for (uint32_t s = 0; s < num_workers; s++) {
            std::array<uint32_t, 3> ids;
            if (s == 0) {
                ids = {
                    static_cast<uint32_t>(tt::CBIndex::c_4),
                    static_cast<uint32_t>(tt::CBIndex::c_5),
                    static_cast<uint32_t>(tt::CBIndex::c_6)};
            } else if (s == 1) {
                ids = {
                    static_cast<uint32_t>(tt::CBIndex::c_16),
                    static_cast<uint32_t>(tt::CBIndex::c_17),
                    static_cast<uint32_t>(tt::CBIndex::c_18)};
            } else {
                ids = {next_free_cb, next_free_cb + 1, next_free_cb + 2};
                next_free_cb += 3;
            }
            writer_cb_ids[s] = ids;

            // route_info CB (UInt32, one L1_ALIGNMENT page per slot).
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = writer_cb_size * route_info_page_size,
                .core_ranges = sender_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(ids[0]),
                    .data_format = tt::DataFormat::UInt32,
                    .page_size = route_info_page_size,
                }}},
            });
            detail::create_tensor_cb(
                desc,
                sender_core_grid,
                output_tensor,
                /*buffering_factor=*/writer_cb_size,
                /*cb_id=*/static_cast<tt::CBIndex>(ids[1]),
                "payload_for_writer_" + std::to_string(s));
            detail::create_tensor_cb(
                desc,
                sender_core_grid,
                metadata_tensor,
                /*buffering_factor=*/writer_cb_size,
                /*cb_id=*/static_cast<tt::CBIndex>(ids[2]),
                "metadata_for_writer_" + std::to_string(s));
        }
    }

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // FABRIC_2D uses the portable RoutingPlaneConnectionManager (per-destination connection +
    // multicast handshake) so dispatch-axis traffic forwards multi-hop; FABRIC_1D keeps the legacy
    // per-direction array connection. INVARIANT: this is_2d_fabric gate (derived from GetFabricConfig())
    // must agree with the kernel's FABRIC_2D #ifdef, which append_routing_plane_connection_manager_rt_args
    // injects based on the control plane's is_2D_routing_enabled(). If the two ever diverge, the host
    // pushes 2D-shaped args while the kernel compiles the 1D #else branch (or vice-versa) and arg
    // parsing corrupts.
    const bool is_2d_fabric = tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());

    // c_8: packet header CB for fabric sends (sender-only)
    if (operation_attributes.num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = packet_header_cb_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                .data_format = tt::DataFormat::UInt8,
                .page_size = packet_header_size_bytes,
            }}},
        });
    }

    // Iterate over every coordinate in the mesh (full coord range derived from
    // the mesh shape) — replaces the legacy `tensor_coords` parameter, which the
    // new descriptor-style entry point no longer threads through.  The fabric
    // defines baked into kernel compile-time args list every device on the mesh.
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_view.shape())) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", ccl::common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", ccl::common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", ccl::common::stringify(directions));

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    log_debug(
        tt::LogOp, "Fabric max packet size: {} bytes, L1 alignment: {} bytes", fabric_max_packet_size, l1_alignment);

    // ==================== Compile-time args for the sender writer kernel ====================
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (9)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_input_id (row-major path only)
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_indices_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_offsets_id
        static_cast<uint32_t>(tt::CBIndex::c_4),  // cb_route_info_id
        static_cast<uint32_t>(tt::CBIndex::c_5),  // cb_payload_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_6),  // cb_metadata_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_7),  // cb_metadata_temp_id
        static_cast<uint32_t>(tt::CBIndex::c_8),  // cb_packet_header_id
        static_cast<uint32_t>(tt::CBIndex::c_9),  // cb_dispatch_table_id

        // Page counts (6)
        detail::get_num_pages(input_tensor),
        detail::get_num_pages(indices_tensor),
        detail::get_num_pages(offsets_tensor),
        detail::get_num_pages(output_tensor),
        detail::get_num_pages(metadata_tensor),
        detail::get_num_pages(dispatch_table_tensor),

        // Page sizes (6)
        detail::get_page_size(input_tensor),
        detail::get_page_size(indices_tensor),
        detail::get_page_size(offsets_tensor),
        detail::get_page_size(output_tensor),
        detail::get_page_size(metadata_tensor),
        detail::get_page_size(dispatch_table_tensor),

        // Operation parameters (7)
        mesh_view.num_devices(),  // num_devices
        (uint32_t)hidden_size,
        operation_attributes.experts_per_chip,
        operation_attributes.num_routed_experts,
        operation_attributes.num_experts_per_tok,
        operation_attributes.metadata_len,
        (uint32_t)tokens_per_device,

        // Mesh information (5)
        src_mesh_id,
        src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        linearized_mesh_coord,

        // Aligned page sizes (6)
        detail::get_aligned_page_size(input_tensor),
        detail::get_aligned_page_size(indices_tensor),
        detail::get_aligned_page_size(offsets_tensor),
        detail::get_aligned_page_size(output_tensor),
        detail::get_aligned_page_size(metadata_tensor),
        detail::get_aligned_page_size(dispatch_table_tensor),

        // Fabric configuration (4)
        (uint32_t)fabric_max_packet_size,
        l1_alignment,
        static_cast<uint32_t>(operation_attributes.num_links),
        static_cast<uint32_t>(topology),

        // Batch configuration (1)
        read_batch_size,

        // Dispatch buffer total token capacity (1) — used by the reader's
        // in-kernel bounds check.
        operation_attributes.max_dispatch_buffer_token_size,
    };

    // Append TensorAccessorArgs for all 6 tensors
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(offsets_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatch_table_tensor.buffer()).append_to(compile_time_args);

    std::map<std::string, std::string> fabric_defines;
    if (operation_attributes.num_links > 0) {
        fabric_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        fabric_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        fabric_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }
    if (operation_attributes.axis.has_value()) {
        fabric_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    // ==================== Sender writer kernel ====================
    // Drains the worker baton-ring CBs and writes tokens and metadata via fabric
    // to their destination chips.
    auto writer_defines = fabric_defines;

    std::vector<uint32_t> writer_compile_time_args = compile_time_args;
    writer_compile_time_args.push_back(writer_cb_size);  // sender writer CB depth
    writer_compile_time_args.push_back(num_workers);     // N: sizes the kernel's per-ring arrays

    tt::tt_metal::KernelDescriptor writer_kd;
    writer_kd.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "writer_sender_dispatch.cpp";
    writer_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kd.core_ranges = sender_core_grid;
    writer_kd.compile_time_args = std::move(writer_compile_time_args);
    writer_kd.defines = {writer_defines.begin(), writer_defines.end()};
    writer_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
    };
    tt::tt_metal::KernelHandle writer_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_kd));

    // ==================== Padding-config scratch CBs (worker cores) ====================
    // The worker reader and writer run on the SAME worker core (separate RISCs), so each needs
    // its own scratch CB index for the [local_real_tokens, pad_side] row. c_16/c_17 are free on
    // worker cores (the c_16-c_18 writer set lives only on sender cores).
    const bool has_padding_config = tensor_args.padding_config.has_value();
    constexpr auto cb_padding_config_reader = tt::CBIndex::c_16;
    constexpr auto cb_padding_config_writer = tt::CBIndex::c_17;
    if (has_padding_config) {
        detail::create_tensor_cb(
            desc,
            worker_core_grid,
            tensor_args.padding_config.value(),
            /*buffering_factor=*/1,
            cb_padding_config_reader,
            "padding_config_reader");
        detail::create_tensor_cb(
            desc,
            worker_core_grid,
            tensor_args.padding_config.value(),
            /*buffering_factor=*/1,
            cb_padding_config_writer,
            "padding_config_writer");
    }

    // ==================== Per-token fp8 scales CB (worker cores) ====================
    // True producer→consumer CB shared across the worker core's two RISCs: the reader fills one
    // scale row per token (parallel to the c_0 row-major payload), the writer drains it and copies
    // each token's scales into the metadata tail. Double-buffered at batch granularity like c_0.
    // c_12 is free on worker cores (the sender writer set lives on sender cores; padding uses c_16/c_17).
    // Gated by the explicit fp8_scaled_input flag (validation guarantees scales_tensor is present).
    const bool fp8_scaled_input = operation_attributes.fp8_scaled_input;
    constexpr auto cb_scales = tt::CBIndex::c_12;
    uint32_t num_scale_words = 0;
    uint32_t aligned_scales_page_size = 0;
    if (fp8_scaled_input) {
        const auto& scales_tensor = tensor_args.scales_tensor.value();
        num_scale_words = scales_tensor.logical_shape()[-1];
        aligned_scales_page_size = detail::get_aligned_page_size(scales_tensor);
        detail::create_tensor_cb(
            desc,
            worker_core_grid,
            scales_tensor,
            /*buffering_factor=*/2 * read_batch_size,
            cb_scales,
            "rowmajor_scales_scratch");
    }

    // ==================== Worker core kernels ====================
    // Reader (RISCV_1): routing decisions, DRAM reads for input/indices/offsets/dispatch_table,
    //                   publishes per-batch route plan to writer via c_14.
    // Writer (RISCV_0): drains c_14 plan, executes local DRAM writes for the local path and direct
    //                   NOC writes into the owning sender's writer-CB set for local_core_id (set s)
    //                   for the cross-device path.  Per-entry handshake via data_avail / space_avail.
    std::vector<tt::tt_metal::KernelHandle> reader_worker_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_worker_kernel_ids;
    reader_worker_kernel_ids.reserve(num_worker_cores);
    writer_worker_kernel_ids.reserve(num_worker_cores);

    // block_ct_dim_dispatch is derived once above (shared with c_0 sizing and the compute kernel).
    for (uint32_t j = 0; j < num_worker_cores; j++) {
        uint32_t s = worker_sender_map[j];
        // Each sender owns num_workers cores; they split batches round-robin by local_core_id
        // (batch i handled by core i % total_workers) and share one offsets[] counter via a baton.
        uint32_t local_core_id = j % num_workers;
        uint32_t total_workers = num_workers;

        // ===== Reader compile args =====
        std::vector<uint32_t> worker_reader_compile_args = {
            static_cast<uint32_t>(tt::CBIndex::c_0),               // 0: cb_input_id
            static_cast<uint32_t>(tt::CBIndex::c_10),              // 1: cb_signal_id
            (uint32_t)hidden_size,                                 // 2
            detail::get_aligned_page_size(input_tensor),           // 3: aligned_input_page_size
            total_batches,                                         // 4
            local_core_id,                                         // 5
            total_workers,                                         // 6
            static_cast<uint32_t>(tt::CBIndex::c_1),               // 7: cb_indices_id
            static_cast<uint32_t>(tt::CBIndex::c_3),               // 8: cb_offsets_id
            static_cast<uint32_t>(tt::CBIndex::c_9),               // 9: cb_dispatch_table_id
            static_cast<uint32_t>(tt::CBIndex::c_14),              // 10: cb_plan_id
            read_batch_size,                                       // 11
            detail::get_aligned_page_size(indices_tensor),         // 12
            detail::get_aligned_page_size(offsets_tensor),         // 13
            detail::get_aligned_page_size(dispatch_table_tensor),  // 14
            detail::get_num_pages(offsets_tensor),                 // 15: offsets_pages
            detail::get_num_pages(dispatch_table_tensor),          // 16: dispatch_table_pages
            operation_attributes.num_experts_per_tok,              // 17
            operation_attributes.num_routed_experts,               // 18: n_routed_experts
            operation_attributes.max_dispatch_buffer_token_size,   // 19
            s,                                                     // 20: dispatch_core_idx
            num_cores,                                             // 21: num_dispatch_cores
            mesh_view.num_devices(),                               // 22: num_devices
            mesh_view.num_rows(),                                  // 23
            mesh_view.num_cols(),                                  // 24
            linearized_mesh_coord,                                 // 25
            static_cast<uint32_t>(topology),                       // 26
            block_ct_dim_dispatch,                                 // 27: must match the compute kernel
        };

        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(worker_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(worker_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(offsets_tensor.buffer()).append_to(worker_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(dispatch_table_tensor.buffer()).append_to(worker_reader_compile_args);

        if (has_padding_config) {
            // padding_config accessor + scratch CB id appended LAST so the existing index layout is unchanged.
            tt::tt_metal::TensorAccessorArgs(tensor_args.padding_config.value().buffer())
                .append_to(worker_reader_compile_args);
            worker_reader_compile_args.push_back(static_cast<uint32_t>(cb_padding_config_reader));
        }
        if (fp8_scaled_input) {
            // scales accessor + CB id + aligned page size appended after padding_config so the
            // kernel's chained TensorAccessorArgs offsets stay consistent (padding before scales).
            tt::tt_metal::TensorAccessorArgs(tensor_args.scales_tensor.value().buffer())
                .append_to(worker_reader_compile_args);
            worker_reader_compile_args.push_back(static_cast<uint32_t>(cb_scales));
            worker_reader_compile_args.push_back(aligned_scales_page_size);
        }

        // ===== Writer compile args =====
        std::vector<uint32_t> worker_writer_compile_args = {
            // Payload CB the writer drains: compute output (c_11) on tile-layout, or the reader-filled
            // row CB (c_0) on row-major. The writer logic is identical either way.
            static_cast<uint32_t>(is_row_major ? tt::CBIndex::c_0 : tt::CBIndex::c_11),  // 0: cb_untilize_id
            read_batch_size,                                                             // 1
            detail::get_aligned_page_size(output_tensor),                                // 2: aligned_output_page_size
            total_batches,                                                               // 3
            local_core_id,                                                               // 4
            total_workers,                                                               // 5
            static_cast<uint32_t>(tt::CBIndex::c_13),                                    // 6: cb_metadata_scratch_id
            detail::get_aligned_page_size(metadata_tensor),              // 7: aligned_metadata_page_size
            static_cast<uint32_t>(tt::CBIndex::c_14),                    // 8: cb_plan_id
            linearized_mesh_coord,                                       // 9
            l1_alignment,                                                // 10: route_info slot stride
            writer_cb_size,                                              // 11: sender writer CB size
            static_cast<uint32_t>(tt::CBIndex::c_15),                    // 12: cb_route_info_scratch_id
            read_batch_size * operation_attributes.num_experts_per_tok,  // 13: meta_scratch_slots
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(worker_writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(worker_writer_compile_args);
        if (has_padding_config) {
            tt::tt_metal::TensorAccessorArgs(tensor_args.padding_config.value().buffer())
                .append_to(worker_writer_compile_args);
            worker_writer_compile_args.push_back(static_cast<uint32_t>(cb_padding_config_writer));
        }
        if (fp8_scaled_input) {
            // The writer reads scales from the CB (no DRAM accessor needed): CB id + aligned page
            // size (to index by token_t) + number of fp32 scale words to copy into the metadata tail.
            worker_writer_compile_args.push_back(static_cast<uint32_t>(cb_scales));
            worker_writer_compile_args.push_back(aligned_scales_page_size);
            worker_writer_compile_args.push_back(num_scale_words);
        }

        auto worker_kernel_defines = fabric_defines;  // carries AXIS define if set
        auto worker_writer_defines = fabric_defines;
        if (has_padding_config) {
            worker_kernel_defines["HAS_PADDING_CONFIG"] = "1";
            worker_writer_defines["HAS_PADDING_CONFIG"] = "1";
        }
        if (fp8_scaled_input) {
            worker_kernel_defines["FP8_SCALED"] = "1";
            worker_writer_defines["FP8_SCALED"] = "1";
        }
        // Reader-only: row-major reads rows straight into c_0 and skips the compute signal/streaming.
        // The writer kernel is layout-agnostic (it just drains cb_untilize), so it needs no define.
        // NOTE: do NOT name this "ROW_MAJOR" — that collides with ShardOrientation::ROW_MAJOR in
        // buffer_types.hpp (pulled in via dataflow_api.h) and breaks the enum at preprocess time.
        if (is_row_major) {
            worker_kernel_defines["ROW_MAJOR_INPUT"] = "1";
        }

        CoreRangeSet single_worker_core({CoreRange(all_worker_cores[j])});
        tt::tt_metal::KernelDescriptor worker_reader_kd;
        worker_reader_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "reader_worker_dispatch.cpp";
        worker_reader_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        worker_reader_kd.core_ranges = single_worker_core;
        worker_reader_kd.compile_time_args = std::move(worker_reader_compile_args);
        worker_reader_kd.defines = {worker_kernel_defines.begin(), worker_kernel_defines.end()};
        worker_reader_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
        };
        reader_worker_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
        desc.kernels.push_back(std::move(worker_reader_kd));

        tt::tt_metal::KernelDescriptor worker_writer_kd;
        worker_writer_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "writer_worker_dispatch.cpp";
        worker_writer_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        worker_writer_kd.core_ranges = single_worker_core;
        worker_writer_kd.compile_time_args = std::move(worker_writer_compile_args);
        worker_writer_kd.defines = {worker_writer_defines.begin(), worker_writer_defines.end()};
        worker_writer_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
        };
        writer_worker_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
        desc.kernels.push_back(std::move(worker_writer_kd));
    }

    // Compute kernel on worker cores — tile-layout only. Row-major performs no pack_untilize:
    // the reader fills the payload CB directly, so no compute kernel is created.
    if (!is_row_major) {
        // block_ct_dim_dispatch is derived once above (shared with the reader kernel).
        tt::tt_metal::KernelDescriptor untilize_compute_kd;
        untilize_compute_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/compute/"
            "untilize_dispatch.cpp";
        untilize_compute_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        untilize_compute_kd.core_ranges = worker_core_grid;
        untilize_compute_kd.compile_time_args = {
            static_cast<uint32_t>(tt::CBIndex::c_10),  // cb_signal_id
            static_cast<uint32_t>(tt::CBIndex::c_11),  // cb_untilize_id
            static_cast<uint32_t>(tt::CBIndex::c_0),   // cb_in_id
            (uint32_t)hidden_size,
            read_batch_size,
            block_ct_dim_dispatch,  // block_ct_dim
        };
        // Blackhole requires the DEST register in 32-bit mode whenever any CB on the core uses an
        // 8-bit float format (Fp8_e4m3). That is the case for fp8 output (c_11) and also for fp8
        // input (c_0), so enable 32-bit DEST whenever either side is fp8.
        const bool any_fp8 =
            operation_attributes.fp8_output || input_tensor.dtype() == tt::tt_metal::DataType::FP8_E4M3;
        untilize_compute_kd.config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = any_fp8,
            // 32-bit DEST halves pack_untilize block capacity: half-sync 32-bit allows only 4
            // tiles, but pack_untilize_block uses block_ct_dim. Full-sync 32-bit restores the
            // full tile budget so the block still fits. Only needed on the FP8 (32-bit) path.
            .dst_full_sync_en = any_fp8,
        };
        desc.kernels.push_back(std::move(untilize_compute_kd));
    }

    // ==================== Pre-compute NOC coordinates ====================
    std::vector<std::pair<uint32_t, uint32_t>> sender_noc_coords;
    for (const auto& sc : sender_cores) {
        auto noc_coord = mesh_device->virtual_core_from_logical_core(sc, tt::CoreType::WORKER);
        sender_noc_coords.emplace_back(noc_coord.x, noc_coord.y);
    }

    // ==================== Runtime args for sender cores ====================
    // Build the base RT arg block as plain uint32_t so the fabric helper (which
    // appends raw uint32_t values) can extend it.  Buffer-address slots are
    // re-pushed below as Buffer* entries when promoting to the
    // KernelDescriptor::RTArgList builder so the framework records
    // BufferBindings for the cache-hit fast path.
    std::vector<uint32_t> base_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        offsets_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        dispatch_table_tensor.buffer()->address(),
        (uint32_t)cross_device_semaphore.address(),
        (uint32_t)init_semaphore.address(),
        0,                            // token_start_idx
        (uint32_t)tokens_per_device,  // token_end_idx
        0,                            // dispatch_core_idx (set per core)
        num_cores,                    // num_dispatch_cores
    };

    // Helper: promote a flat uint32_t RT arg vector into an RTArgList with the
    // first 6 positions converted to Buffer* (so BufferBindings are auto-
    // registered for those slots), preserving all other positions verbatim.
    auto promote_rt_args_with_buffer_bindings = [&](const std::vector<uint32_t>& raw_args) {
        tt::tt_metal::KernelDescriptor::RTArgList args;
        args.reserve(raw_args.size());
        args.push_back(input_tensor.buffer());
        args.push_back(indices_tensor.buffer());
        args.push_back(offsets_tensor.buffer());
        args.push_back(output_tensor.buffer());
        args.push_back(metadata_tensor.buffer());
        args.push_back(dispatch_table_tensor.buffer());
        for (size_t i = 6; i < raw_args.size(); ++i) {
            args.push_back(raw_args[i]);
        }
        return args;
    };

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        // This sender's N workers, ordered by local_core_id: group[s] has local_core_id == s
        // (all_worker_cores is built sender-major, so j = s*num_workers + u → local id u).
        const auto& group = sender_worker_groups[core_idx];

        std::vector<uint32_t> writer_runtime_args = base_runtime_args;
        writer_runtime_args[10] = core_idx;  // dispatch_core_idx

        // Writer-only: exit semaphore address (separate from init_semaphore to avoid
        // init/exit reuse race where a fast peer's exit-inc lands during the post-init
        // set(0) window).
        writer_runtime_args.push_back(
            (uint32_t)exit_semaphore.address());  // smuggled-rta-ok: persistent GlobalSemaphore (created once in
                                                  // TT_CCL, reused) — L1 address stable across program-cache hits

        // ===== Sender writer: handshake + per-entry fabric send + credit =====
        // Shared single-id semaphores (one per-core slot on each worker): pushed once.
        writer_runtime_args.push_back(addr_ready_semaphore_id);
        writer_runtime_args.push_back(cross_addr_semaphore_id);
        writer_runtime_args.push_back(space_avail_semaphore_id);
        // Per-ring group s: {route_cb, payload_cb, metadata_cb, worker_noc_x, worker_noc_y,
        // data_avail_id}. The kernel reads exactly num_workers such groups.
        for (uint32_t s = 0; s < num_workers; s++) {
            auto u_noc = mesh_device->virtual_core_from_logical_core(group[s], tt::CoreType::WORKER);
            writer_runtime_args.push_back(writer_cb_ids[s][0]);
            writer_runtime_args.push_back(writer_cb_ids[s][1]);
            writer_runtime_args.push_back(writer_cb_ids[s][2]);
            writer_runtime_args.push_back((uint32_t)u_noc.x);
            writer_runtime_args.push_back((uint32_t)u_noc.y);
            writer_runtime_args.push_back(data_avail_semaphore_ids[s]);
        }

        if (operation_attributes.num_links > 0) {
            // Dispatch-axis neighbors (each a distinct fabric direction) as fabric nodes.
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }
                dst_nodes.push_back(mesh_device->get_fabric_node_id(neighbor_coordinate));
            }
            const uint32_t core_link = core_idx % num_links;
            if (is_2d_fabric) {
                // Portable RoutingPlaneConnectionManager path: one connection per dispatch-axis neighbor
                // so traffic forwards across MULTIPLE hops (the legacy fixed-link array connection only
                // forwards a single hop, deadlocking multi-hop FABRIC_2D — e.g. the 4-device column of a
                // 4x2 mesh). The writer reads num_connections first, then builds the manager from the
                // appended args.
                //
                // Pick a forwarding link valid for each neighbor's own direction (see
                // compute_per_neighbor_forwarding_links above for why a single broadcast {core_link} hangs).
                const std::vector<uint32_t> per_conn_links =
                    compute_per_neighbor_forwarding_links(src_fabric_node_id, dst_nodes, core_link, "dispatch-axis");
                writer_runtime_args.push_back(static_cast<uint32_t>(dst_nodes.size()));
                tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                    src_fabric_node_id,
                    dst_nodes,
                    per_conn_links,
                    desc,
                    writer_kernel_id,
                    sender_core,
                    writer_runtime_args);
                log_debug(
                    tt::LogOp,
                    "FABRIC_2D dispatch writer (tile): src={} num_connections={} core_link={}",
                    src_fabric_node_id,
                    dst_nodes.size(),
                    core_link);
            } else {
                // Legacy per-direction array connection (FABRIC_1D linear/ring — never deadlocked).
                for (const auto& dst_node : dst_nodes) {
                    tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                        src_fabric_node_id, dst_node, core_link, desc, sender_core, writer_runtime_args);
                }
            }
        }

        desc.kernels[writer_kernel_id].emplace_runtime_args(
            sender_core, promote_rt_args_with_buffer_bindings(writer_runtime_args));
        core_idx++;
    }

    // ==================== Runtime args for worker cores ====================
    // Reader: tensor base addresses + token range (Buffer* slots → BufferBindings on cache hit).
    // Writer: sender NOC coords + addr-handshake + data_avail/space_avail semaphores + output/metadata buffers.
    //   writer-CB set per local_core_id: 0 → c_4/c_5/c_6, 1 → c_16/c_17/c_18, rest → c_19+.
    for (uint32_t j = 0; j < num_worker_cores; j++) {
        uint32_t s = worker_sender_map[j];

        // Reader RT args

        tt::tt_metal::KernelDescriptor::RTArgList worker_reader_rt_args;
        worker_reader_rt_args.push_back(input_tensor.buffer());
        worker_reader_rt_args.push_back(indices_tensor.buffer());
        worker_reader_rt_args.push_back(offsets_tensor.buffer());
        worker_reader_rt_args.push_back(dispatch_table_tensor.buffer());
        worker_reader_rt_args.push_back(0u);                           // token_start_idx
        worker_reader_rt_args.push_back((uint32_t)tokens_per_device);  // token_end_idx
        // Baton-ring offset sync: owner (group[0]) holds the shared offsets[]; the next core
        // in the ring ((local_u_id+1) % num_workers) receives the baton after this batch.
        {
            const auto& group = sender_worker_groups[s];
            uint32_t local_u_id = j % num_workers;
            CoreCoord owner_core = group[0];
            CoreCoord next_core = group[(local_u_id + 1) % num_workers];
            auto owner_noc = mesh_device->virtual_core_from_logical_core(owner_core, tt::CoreType::WORKER);
            auto next_noc = mesh_device->virtual_core_from_logical_core(next_core, tt::CoreType::WORKER);
            worker_reader_rt_args.push_back((uint32_t)owner_noc.x);
            worker_reader_rt_args.push_back((uint32_t)owner_noc.y);
            worker_reader_rt_args.push_back((uint32_t)next_noc.x);
            worker_reader_rt_args.push_back((uint32_t)next_noc.y);
            worker_reader_rt_args.push_back(turn_semaphore_id);
        }
        // padding_config base address appended last (as Buffer* so it refreshes on cache hit).
        if (has_padding_config) {
            worker_reader_rt_args.push_back(tensor_args.padding_config.value().buffer());
        }
        // scales base address appended after padding_config (as Buffer* so it refreshes on cache hit).
        if (fp8_scaled_input) {
            worker_reader_rt_args.push_back(tensor_args.scales_tensor.value().buffer());
        }
        desc.kernels[reader_worker_kernel_ids[j]].emplace_runtime_args(all_worker_cores[j], worker_reader_rt_args);

        // Writer RT args. addr_ready / cross_addr / space_avail are shared single-id semaphores
        // (this worker uses its own per-core slot at each); data_avail is this ring's private
        // id (the sender waits on its matching slot). Arg order unchanged from the kernel's view.
        uint32_t local_u_id = j % num_workers;
        tt::tt_metal::KernelDescriptor::RTArgList worker_writer_rt_args;
        worker_writer_rt_args.push_back(sender_noc_coords[s].first);
        worker_writer_rt_args.push_back(sender_noc_coords[s].second);
        worker_writer_rt_args.push_back(addr_ready_semaphore_id);
        worker_writer_rt_args.push_back(cross_addr_semaphore_id);
        worker_writer_rt_args.push_back(data_avail_semaphore_ids[local_u_id]);
        worker_writer_rt_args.push_back(space_avail_semaphore_id);
        worker_writer_rt_args.push_back(output_tensor.buffer());
        worker_writer_rt_args.push_back(metadata_tensor.buffer());
        // padding_config base address appended last (as Buffer* so it refreshes on cache hit).
        if (has_padding_config) {
            worker_writer_rt_args.push_back(tensor_args.padding_config.value().buffer());
        }
        desc.kernels[writer_worker_kernel_ids[j]].emplace_runtime_args(all_worker_cores[j], worker_writer_rt_args);
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor DispatchProgramFactory::create_workload_descriptor(
    const DispatchParams& operation_attributes,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto* mesh_device = tensor_args.input_tensor.device();

    // Allocate the three cross-device GlobalSemaphores once per workload (cache miss).
    // They live on WorkloadDescriptor.semaphores so the device-side allocations
    // outlive the cached MeshWorkload via the program cache — kernel runtime args
    // bake in their absolute addresses, so per-coord program builds must see the
    // same allocation as every other coord.
    auto sem_buffer_type = operation_attributes.use_l1_small_for_semaphores ? tt::tt_metal::BufferType::L1_SMALL
                                                                            : tt::tt_metal::BufferType::L1;
    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    auto exit_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    // Cross-device barrier: ensure every device has allocated its GlobalSemaphores
    // before any kernel reads them.  Mirrors the previous prepare_resources hook.
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(init_barrier_semaphore);
    workload_descriptor.semaphores.push_back(exit_barrier_semaphore);
    workload_descriptor.semaphores.push_back(final_barrier_semaphore);

    const bool is_tile_layout = tensor_args.input_tensor.layout() == tt::tt_metal::Layout::TILE;
    log_info(tt::LogOp, "Prefill dispatch: input tensor is {} layout", is_tile_layout ? "TILE" : "ROW_MAJOR");

    // Dispatch is mesh-coord-dependent (fabric routing + linearized mesh
    // coordinate are baked into kernel compile-time args), so we cannot
    // replicate one ProgramDescriptor across the whole mesh — every coord
    // gets its own build. Both layouts share create_dispatch_program; it detects
    // row-major vs tile internally (row-major skips the untilize compute kernel).
    for (const auto& coord : tensor_coords.coords()) {
        tt::tt_metal::ProgramDescriptor desc = create_dispatch_program(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            exit_barrier_semaphore,
            final_barrier_semaphore);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
