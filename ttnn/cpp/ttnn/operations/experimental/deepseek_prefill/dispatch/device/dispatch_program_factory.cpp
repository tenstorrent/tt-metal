// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_device_operation.hpp"
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
// instead of calling CreateCircularBuffer.  Mirrors the legacy data_format
// rewrite for the FP8 dispatch path (UINT8 dispatch buffers reinterpret as
// Fp8_e4m3 in the CB until FP8 gets a dedicated dtype).
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
    if (data_format == tt::DataFormat::UInt8) {
        // TODO: remove once FP8 has a dedicated dtype. In this op, UINT8 tensors only appear
        // on the FP8 dispatch path (DRAM is allocated as UINT8 but content is Fp8_e4m3).
        data_format = tt::DataFormat::Fp8_e4m3;
    }

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

// Tile-layout path: TILE inputs, fused untilize across exactly 2 untilize cores per
// sender (u1 and u2); sender is fabric-only.  Per-entry handshake on the writer CBs
// lets each untilize core feed its sender writer in lockstep:
//   u1 (local_core_id=0): drives c_4/c_5/c_6    (sender writer's first  CB set)
//   u2 (local_core_id=1): drives c_16/c_17/c_18 (sender writer's second CB set)
// Sender writer (RISCV_0) consumes both CB sets round-robin and fans out via fabric.
tt::tt_metal::ProgramDescriptor create_at_tile_layout(
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
    auto weights_tensor = tensor_args.weights_tensor;
    auto offsets_tensor = tensor_args.expert_offsets_tensor;
    auto histograms_tensor = tensor_args.expert_histograms_tensor;
    auto dispatch_table_tensor = tensor_args.expert_dispatch_table_tensor;

    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;
    uint32_t num_untilizers = operation_attributes.num_untilizers_per_sender;
    log_debug(
        tt::LogOp,
        "Creating prefill dispatch program (tile layout) for mesh coordinate: ({}, {}) with topology: {} num_links: {}",
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

    // ==================== Core layout: senders + untilize cores ====================
    // Collect all cores in the first row (y == subdevice_cores[0].y), sorted by x.
    // Each sender owns exactly 2 consecutive untilize cores (u1, u2): cores are laid
    // out [sender, u1, u2] consecutively along x per sender group.
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
    uint32_t cores_per_sender = 1 + num_untilizers;  // 1 sender + 2 untilize cores (u1, u2)
    TT_FATAL(
        total_row_cores >= cores_per_sender * num_cores,
        "Same-row has only {} cores for {} senders — need {} cores per sender (>= {} required)",
        total_row_cores,
        num_cores,
        cores_per_sender,
        cores_per_sender * num_cores);

    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(num_cores);
    std::vector<std::vector<CoreCoord>> sender_untilize_groups(num_cores);
    std::vector<CoreCoord> all_untilize_cores;
    all_untilize_cores.reserve(num_cores * num_untilizers);
    std::vector<uint32_t> untilize_sender_map;
    untilize_sender_map.reserve(num_cores * num_untilizers);

    for (uint32_t s = 0; s < num_cores; s++) {
        sender_cores.push_back(all_row_cores[cores_per_sender * s]);
        for (uint32_t u = 0; u < num_untilizers; u++) {
            CoreCoord uc = all_row_cores[cores_per_sender * s + 1 + u];
            sender_untilize_groups[s].push_back(uc);
            all_untilize_cores.push_back(uc);
            untilize_sender_map.push_back(s);
        }
    }

    uint32_t num_untilize_cores = static_cast<uint32_t>(all_untilize_cores.size());

    // Build sender_core_grid and untilize_core_grid CoreRangeSets
    std::set<CoreRange> sender_ranges_set;
    for (const auto& sc : sender_cores) {
        sender_ranges_set.insert(CoreRange(sc));
    }
    auto sender_core_grid = CoreRangeSet(sender_ranges_set);

    std::set<CoreRange> untilize_ranges_set;
    for (const auto& ic : all_untilize_cores) {
        untilize_ranges_set.insert(CoreRange(ic));
    }
    CoreRangeSet untilize_core_grid(untilize_ranges_set);

    // Combined grid for shared semaphores
    std::set<CoreRange> sender_and_untilize_ranges;
    for (const auto& cr : sender_core_grid.ranges()) {
        sender_and_untilize_ranges.insert(cr);
    }
    for (const auto& cr : untilize_core_grid.ranges()) {
        sender_and_untilize_ranges.insert(cr);
    }
    CoreRangeSet sender_and_untilize_grid(sender_and_untilize_ranges);

    log_debug(
        tt::LogOp,
        "Dispatch program: num_links: {} num_cores(senders): {} num_untilize_cores: {} tokens_per_device: {}",
        num_links,
        num_cores,
        num_untilize_cores,
        tokens_per_device);

    constexpr uint32_t read_batch_size = 32;

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    uint32_t total_batches = (tokens_per_device + read_batch_size - 1) / read_batch_size;

    // ==================== Semaphores ====================
    // Per-entry credit, each kept on the consumer's L1 (producer NOC-incs):
    //   data_avail  (sender L1, init=0):                untilize → sender, "entry ready".
    //   space_avail (untilize L1, init=writer_cb_size): sender → untilize, "slot freed".
    //   Seeding space_avail with the CB depth lets untilize prefill all slots cold-start.
    // Boot-time address handshake: sender writes its c_4/c_5/c_6 base addresses into
    // cross_c{4,5,6}_addr scratch slots on untilize, then noc-incs addr_ready.
    // u2 mirrors the same protocol on c_16/c_17/c_18 with a separate semaphore set.
    // SemaphoreDescriptor takes an explicit .id; add_sema feeds it from a monotonic counter.
    constexpr uint32_t writer_cb_size = read_batch_size;  // 32 slots — one batch deep
    uint32_t next_sema_id = 0;
    auto add_sema = [&](const CoreRangeSet& crs, uint32_t init_val = 0) -> uint32_t {
        uint32_t id = next_sema_id++;
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = crs, .initial_value = init_val});
        return id;
    };
    // u1 (drives c_4/c_5/c_6)
    auto data_avail_semaphore_id = add_sema(sender_and_untilize_grid);
    auto space_avail_semaphore_id = add_sema(sender_and_untilize_grid, writer_cb_size);
    auto addr_ready_semaphore_id = add_sema(sender_and_untilize_grid);
    auto cross_c4_addr_semaphore_id = add_sema(sender_and_untilize_grid);
    auto cross_c5_addr_semaphore_id = add_sema(sender_and_untilize_grid);
    auto cross_c6_addr_semaphore_id = add_sema(sender_and_untilize_grid);
    // u2 (drives c_16/c_17/c_18)
    auto data_avail_u2_semaphore_id = add_sema(sender_and_untilize_grid);
    auto space_avail_u2_semaphore_id = add_sema(sender_and_untilize_grid, writer_cb_size);
    auto addr_ready_u2_semaphore_id = add_sema(sender_and_untilize_grid);
    auto cross_c16_addr_semaphore_id = add_sema(sender_and_untilize_grid);
    auto cross_c17_addr_semaphore_id = add_sema(sender_and_untilize_grid);
    auto cross_c18_addr_semaphore_id = add_sema(sender_and_untilize_grid);

    // ==================== Circular Buffers for untilize cores ====================
    // Routing decisions and offsets[] live on the untilize core — sender is fabric-only.
    // c_0: tiled input stripe (reader → compute)
    detail::create_tensor_cb(
        desc,
        untilize_core_grid,
        input_tensor,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_0,
        "untilize_input_scratch");
    // c_1: indices scratch (untilize reader does per-batch DRAM reads)
    detail::create_tensor_cb(
        desc,
        untilize_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "untilize_indices_scratch");
    // c_2: weights scratch
    detail::create_tensor_cb(
        desc,
        untilize_core_grid,
        weights_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_2,
        "untilize_weights_scratch");
    // c_3: offsets (full tensor, loaded once at startup, mutated in place per batch).
    // Sized for 2× num_pages: the lower half holds expert_offsets[] (consumed by both u1 and
    // u2); the upper half is u2-only scratch where the histograms tensor is staged so u2 can
    // compute its right-to-left starting pointers as expert_offsets[e] + histogram[e] in L1.
    detail::create_tensor_cb(
        desc,
        untilize_core_grid,
        offsets_tensor,
        /*buffering_factor=*/2 * detail::get_num_pages(offsets_tensor),
        /*cb_id=*/tt::CBIndex::c_3,
        "untilize_offsets_tensor");
    // c_9: dispatch_table (full tensor, loaded once at startup)
    detail::create_tensor_cb(
        desc,
        untilize_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),
        /*cb_id=*/tt::CBIndex::c_9,
        "untilize_dispatch_table_tensor");
    // c_10: signal CB (reader → compute)
    {
        uint32_t signal_page_size = l1_alignment;
        constexpr uint32_t signal_buffering = 2;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = signal_buffering * signal_page_size,
            .core_ranges = untilize_core_grid,
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
        untilize_core_grid,
        output_tensor,
        /*buffering_factor=*/2 * read_batch_size,
        /*cb_id=*/tt::CBIndex::c_11,
        "untilize_untilize_output");
    // c_13: metadata scratch (untilize writer builds metadata here before NOC-writing).
    // Layout: meta_scratch_slots local-path ring slots followed by 1 cross-device scratch slot.
    //   slots 0..meta_scratch_slots-1: local-path ring (worst-case sized so we never wrap
    //     within a batch; one noc_async_writes_flushed() at batch end covers source reuse)
    //   slot meta_scratch_slots:       cross-device-path scratch (per-entry barrier already
    //     drains the source, so a single slot is enough — kept distinct from the local ring
    //     to avoid clobbering pending local writes when entries interleave)
    detail::create_tensor_cb(
        desc,
        untilize_core_grid,
        metadata_tensor,
        /*buffering_factor=*/(read_batch_size * operation_attributes.num_experts_per_tok) + 1,
        /*cb_id=*/tt::CBIndex::c_13,
        "untilize_metadata_scratch");
    // c_15: route_info scratch (16B = l1_alignment). Untilize writer builds the 4-u32
    // route_info entry [route, distance, page_idx, 0] here, then NOC-writes the whole
    // block as a single noc_async_write to the sender's c_4 slot (replaces 4× inline_dw).
    {
        uint32_t route_info_scratch_size = l1_alignment;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = route_info_scratch_size,
            .core_ranges = untilize_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_15),
                .data_format = tt::DataFormat::UInt32,
                .page_size = route_info_scratch_size,
            }}},
        });
    }
    // c_14: per-batch route plan (reader RISC → writer RISC, on same untilize core).
    // Layout: [entry_count u32][padding to 32B][entries × 8 u32 each]
    {
        constexpr uint32_t plan_entry_u32s = 8;
        uint32_t max_plan_entries = read_batch_size * operation_attributes.num_experts_per_tok;
        uint32_t plan_page_size = tt::round_up(
            32u + max_plan_entries * plan_entry_u32s * static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);
        constexpr uint32_t plan_buffering = 2;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = plan_buffering * plan_page_size,
            .core_ranges = untilize_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_14),
                .data_format = tt::DataFormat::UInt32,
                .page_size = plan_page_size,
            }}},
        });
    }

    // ==================== Circular Buffers for SENDER cores ====================
    // Direct-consume pipeline on sender:
    //   c_4/c_5/c_6    = u1's data CBs (route_info, payload, metadata).
    //   c_16/c_17/c_18 = u2's data CBs (same layout, same depth).
    // Untilize cores NOC-write per-entry directly into the relevant CB set;
    // sender writer (RISCV_0) consumes them round-robin.
    {
        uint32_t route_info_page_size = l1_alignment;

        // u1 set
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = writer_cb_size * route_info_page_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = tt::DataFormat::UInt32,
                .page_size = route_info_page_size,
            }}},
        });

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            output_tensor,
            /*buffering_factor=*/writer_cb_size,
            /*cb_id=*/tt::CBIndex::c_5,
            "payload_for_writer");

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            metadata_tensor,
            /*buffering_factor=*/writer_cb_size,
            /*cb_id=*/tt::CBIndex::c_6,
            "metadata_for_writer");

        // u2 set — identical layout, separate L1 slots.
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = writer_cb_size * route_info_page_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
                .data_format = tt::DataFormat::UInt32,
                .page_size = route_info_page_size,
            }}},
        });

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            output_tensor,
            /*buffering_factor=*/writer_cb_size,
            /*cb_id=*/tt::CBIndex::c_17,
            "payload_for_writer_2");

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            metadata_tensor,
            /*buffering_factor=*/writer_cb_size,
            /*cb_id=*/tt::CBIndex::c_18,
            "metadata_for_writer_2");
    }

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

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

    // ==================== Compile-time args shared by sender writer (and used as a base
    // for untilize kernels' own arg lists where overlap exists) ====================
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (10)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_input_id (row-major path only)
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_indices_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_weights_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_offsets_id
        static_cast<uint32_t>(tt::CBIndex::c_4),  // cb_route_info_id
        static_cast<uint32_t>(tt::CBIndex::c_5),  // cb_payload_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_6),  // cb_metadata_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_7),  // cb_metadata_temp_id
        static_cast<uint32_t>(tt::CBIndex::c_8),  // cb_packet_header_id
        static_cast<uint32_t>(tt::CBIndex::c_9),  // cb_dispatch_table_id

        // Page counts (7)
        detail::get_num_pages(input_tensor),
        detail::get_num_pages(indices_tensor),
        detail::get_num_pages(weights_tensor),
        detail::get_num_pages(offsets_tensor),
        detail::get_num_pages(output_tensor),
        detail::get_num_pages(metadata_tensor),
        detail::get_num_pages(dispatch_table_tensor),

        // Page sizes (7)
        detail::get_page_size(input_tensor),
        detail::get_page_size(indices_tensor),
        detail::get_page_size(weights_tensor),
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

        // Aligned page sizes (7)
        detail::get_aligned_page_size(input_tensor),
        detail::get_aligned_page_size(indices_tensor),
        detail::get_aligned_page_size(weights_tensor),
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

    // Append TensorAccessorArgs for all 7 tensors
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weights_tensor.buffer()).append_to(compile_time_args);
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
    // Tile-layout: no sender reader RISC. The writer (RISCV_0) owns:
    //   * Startup address handshake to untilize (publishes c_4/c_5/c_6 + c_16/c_17/c_18 base L1
    //     addresses, NOC-incs untilize addr_ready / addr_ready_u2).
    //   * Fabric init + per-entry fabric send.
    //   * Per-entry direct credit to untilize space_avail{,_u2} after each fabric send.
    auto writer_defines = fabric_defines;
    writer_defines["IS_TILE_LAYOUT"] = "1";

    std::vector<uint32_t> writer_compile_time_args = compile_time_args;
    writer_compile_time_args.push_back(writer_cb_size);                            // sender writer CB depth
    writer_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_16));  // cb_route_info_2_id
    writer_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_17));  // cb_payload_for_writer_2_id
    writer_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_18));  // cb_metadata_for_writer_2_id

    tt::tt_metal::KernelDescriptor writer_kd;
    writer_kd.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "writer_dispatch.cpp";
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

    // ==================== Untilize core kernels ====================
    // Reader (RISCV_1): routing decisions, DRAM reads for input/indices/weights/offsets/dispatch_table,
    //                   publishes per-batch route plan to writer via c_14.
    // Writer (RISCV_0): drains c_14 plan, executes local DRAM writes for the local path and direct
    //                   NOC writes into the owning sender's c_4/c_5/c_6 (u1) or c_16/c_17/c_18 (u2) for the
    //                   cross-device path.  Per-entry handshake via data_avail / space_avail.
    std::vector<tt::tt_metal::KernelHandle> reader_untilize_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_untilize_kernel_ids;
    reader_untilize_kernel_ids.reserve(num_untilize_cores);
    writer_untilize_kernel_ids.reserve(num_untilize_cores);
    for (uint32_t j = 0; j < num_untilize_cores; j++) {
        uint32_t s = untilize_sender_map[j];
        // Each sender has exactly 2 untilize cores; local_core_id alternates per batch:
        // u1 (core_id=0): even batches, increments offset from start.
        // u2 (core_id=1): odd  batches, decrements offset from end.
        uint32_t local_core_id = j % num_untilizers;
        uint32_t total_workers = num_untilizers;

        // ===== Reader compile args =====
        std::vector<uint32_t> untilize_reader_compile_args = {
            static_cast<uint32_t>(tt::CBIndex::c_0),               // 0: cb_input_id
            static_cast<uint32_t>(tt::CBIndex::c_10),              // 1: cb_signal_id
            (uint32_t)hidden_size,                                 // 2
            detail::get_aligned_page_size(input_tensor),           // 3: aligned_input_page_size
            total_batches,                                         // 4
            local_core_id,                                         // 5
            total_workers,                                         // 6
            static_cast<uint32_t>(tt::CBIndex::c_1),               // 7: cb_indices_id
            static_cast<uint32_t>(tt::CBIndex::c_2),               // 8: cb_weights_id
            static_cast<uint32_t>(tt::CBIndex::c_3),               // 9: cb_offsets_id
            static_cast<uint32_t>(tt::CBIndex::c_9),               // 10: cb_dispatch_table_id
            static_cast<uint32_t>(tt::CBIndex::c_14),              // 11: cb_plan_id
            read_batch_size,                                       // 12
            detail::get_aligned_page_size(indices_tensor),         // 13
            detail::get_aligned_page_size(weights_tensor),         // 14
            detail::get_aligned_page_size(offsets_tensor),         // 15
            detail::get_aligned_page_size(dispatch_table_tensor),  // 16
            detail::get_num_pages(offsets_tensor),                 // 17: offsets_pages
            detail::get_num_pages(dispatch_table_tensor),          // 18: dispatch_table_pages
            operation_attributes.num_experts_per_tok,              // 19
            operation_attributes.num_routed_experts,               // 20: n_routed_experts
            operation_attributes.max_dispatch_buffer_token_size,   // 21
            s,                                                     // 22: dispatch_core_idx
            num_cores,                                             // 23: num_dispatch_cores
            mesh_view.num_devices(),                               // 24: num_devices
            mesh_view.num_rows(),                                  // 25
            mesh_view.num_cols(),                                  // 26
            linearized_mesh_coord,                                 // 27
            static_cast<uint32_t>(topology),                       // 28
        };
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(untilize_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(untilize_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(weights_tensor.buffer()).append_to(untilize_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(offsets_tensor.buffer()).append_to(untilize_reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(dispatch_table_tensor.buffer()).append_to(untilize_reader_compile_args);
        // histograms_args: u2 (core_id=1) additionally loads expert_histograms[] and computes
        // its end-of-region pointers in L1 as expert_offsets[e] + expert_histograms[e].
        tt::tt_metal::TensorAccessorArgs(histograms_tensor.buffer()).append_to(untilize_reader_compile_args);

        // ===== Writer compile args =====
        std::vector<uint32_t> untilize_writer_compile_args = {
            static_cast<uint32_t>(tt::CBIndex::c_11),                    // 0: cb_untilize_id
            read_batch_size,                                             // 1
            detail::get_aligned_page_size(output_tensor),                // 2: aligned_output_page_size
            total_batches,                                               // 3
            local_core_id,                                               // 4
            total_workers,                                               // 5
            static_cast<uint32_t>(tt::CBIndex::c_13),                    // 6: cb_metadata_scratch_id
            detail::get_aligned_page_size(metadata_tensor),              // 7: aligned_metadata_page_size
            static_cast<uint32_t>(tt::CBIndex::c_14),                    // 8: cb_plan_id
            linearized_mesh_coord,                                       // 9
            l1_alignment,                                                // 10: route_info slot stride
            writer_cb_size,                                              // 11: sender writer CB size
            static_cast<uint32_t>(tt::CBIndex::c_15),                    // 12: cb_route_info_scratch_id
            read_batch_size * operation_attributes.num_experts_per_tok,  // 13: meta_scratch_slots
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(untilize_writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(untilize_writer_compile_args);

        auto untilize_kernel_defines = fabric_defines;  // carries AXIS define if set

        CoreRangeSet single_untilize_core({CoreRange(all_untilize_cores[j])});
        tt::tt_metal::KernelDescriptor untilize_reader_kd;
        untilize_reader_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "reader_untilize_dispatch.cpp";
        untilize_reader_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        untilize_reader_kd.core_ranges = single_untilize_core;
        untilize_reader_kd.compile_time_args = std::move(untilize_reader_compile_args);
        untilize_reader_kd.defines = {untilize_kernel_defines.begin(), untilize_kernel_defines.end()};
        untilize_reader_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
        };
        reader_untilize_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
        desc.kernels.push_back(std::move(untilize_reader_kd));

        tt::tt_metal::KernelDescriptor untilize_writer_kd;
        untilize_writer_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "writer_untilize_dispatch.cpp";
        untilize_writer_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        untilize_writer_kd.core_ranges = single_untilize_core;
        untilize_writer_kd.compile_time_args = std::move(untilize_writer_compile_args);
        untilize_writer_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
        };
        writer_untilize_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
        desc.kernels.push_back(std::move(untilize_writer_kd));
    }

    // Compute kernel on untilize cores
    {
        tt::tt_metal::KernelDescriptor untilize_compute_kd;
        untilize_compute_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/compute/"
            "untilize_dispatch.cpp";
        untilize_compute_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        untilize_compute_kd.core_ranges = untilize_core_grid;
        untilize_compute_kd.compile_time_args = {
            static_cast<uint32_t>(tt::CBIndex::c_10),  // cb_signal_id
            static_cast<uint32_t>(tt::CBIndex::c_11),  // cb_untilize_id
            static_cast<uint32_t>(tt::CBIndex::c_0),   // cb_in_id
            (uint32_t)hidden_size,
            read_batch_size,
        };
        untilize_compute_kd.config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            // Blackhole requires the DEST register in 32-bit mode whenever any CB on the core uses
            // an 8-bit float format (Fp8_e4m3). The FP8 dispatch path reinterprets UINT8 CBs as
            // Fp8_e4m3, so fp32_dest_acc_en must be enabled there.
            .fp32_dest_acc_en = operation_attributes.use_fp8_dispatch,
            // 32-bit DEST halves pack_untilize block capacity: half-sync 32-bit allows only 4
            // tiles, but pack_untilize_block uses block_ct_dim=8. Full-sync 32-bit restores the
            // 8-tile budget so the block still fits. Only needed on the FP8 (32-bit) path.
            .dst_full_sync_en = operation_attributes.use_fp8_dispatch,
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
        weights_tensor.buffer()->address(),
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
    // first 7 positions converted to Buffer* (so BufferBindings are auto-
    // registered for those slots), preserving all other positions verbatim.
    auto promote_rt_args_with_buffer_bindings = [&](const std::vector<uint32_t>& raw_args) {
        tt::tt_metal::KernelDescriptor::RTArgList args;
        args.reserve(raw_args.size());
        args.push_back(input_tensor.buffer());
        args.push_back(indices_tensor.buffer());
        args.push_back(weights_tensor.buffer());
        args.push_back(offsets_tensor.buffer());
        args.push_back(output_tensor.buffer());
        args.push_back(metadata_tensor.buffer());
        args.push_back(dispatch_table_tensor.buffer());
        for (size_t i = 7; i < raw_args.size(); ++i) {
            args.push_back(raw_args[i]);
        }
        return args;
    };

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        // Find u1 (local_core_id=0) and u2 (local_core_id=1) untilize cores for this sender.
        uint32_t untilize_u1_j = 0, untilize_u2_j = 0, found_count = 0;
        for (uint32_t j = 0; j < num_untilize_cores; j++) {
            if (untilize_sender_map[j] == core_idx) {
                if (found_count == 0) {
                    untilize_u1_j = j;
                } else if (found_count == 1) {
                    untilize_u2_j = j;
                }
                if (++found_count == 2) {
                    break;
                }
            }
        }
        auto u1_noc =
            mesh_device->virtual_core_from_logical_core(all_untilize_cores[untilize_u1_j], tt::CoreType::WORKER);
        auto u2_noc =
            mesh_device->virtual_core_from_logical_core(all_untilize_cores[untilize_u2_j], tt::CoreType::WORKER);
        uint32_t untilize_noc_x = (uint32_t)u1_noc.x;
        uint32_t untilize_noc_y = (uint32_t)u1_noc.y;

        std::vector<uint32_t> writer_runtime_args = base_runtime_args;
        writer_runtime_args[11] = core_idx;  // dispatch_core_idx

        // Writer-only: exit semaphore address (avoids init/exit reuse race).
        writer_runtime_args.push_back((uint32_t)exit_semaphore.address());

        // ===== Sender writer (tile-layout): handshake + per-entry fabric send + credit =====
        // u1 address handshake + credit semaphores
        writer_runtime_args.push_back(addr_ready_semaphore_id);
        writer_runtime_args.push_back(cross_c4_addr_semaphore_id);
        writer_runtime_args.push_back(cross_c5_addr_semaphore_id);
        writer_runtime_args.push_back(cross_c6_addr_semaphore_id);
        writer_runtime_args.push_back(untilize_noc_x);
        writer_runtime_args.push_back(untilize_noc_y);
        writer_runtime_args.push_back(data_avail_semaphore_id);
        writer_runtime_args.push_back(space_avail_semaphore_id);
        // u2 address handshake + credit semaphores (c_16/c_17/c_18 CB set)
        writer_runtime_args.push_back(addr_ready_u2_semaphore_id);
        writer_runtime_args.push_back(cross_c16_addr_semaphore_id);
        writer_runtime_args.push_back(cross_c17_addr_semaphore_id);
        writer_runtime_args.push_back(cross_c18_addr_semaphore_id);
        writer_runtime_args.push_back((uint32_t)u2_noc.x);
        writer_runtime_args.push_back((uint32_t)u2_noc.y);
        writer_runtime_args.push_back(data_avail_u2_semaphore_id);
        writer_runtime_args.push_back(space_avail_u2_semaphore_id);

        if (operation_attributes.num_links > 0) {
            uint32_t core_link = core_idx % num_links;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }

                log_debug(
                    tt::LogOp,
                    "Connection between mesh coord ({}, {}) and ({}, {}) at core {} link {}",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    core_link);
                // ProgramDescriptor specialization: appends fabric-routing args
                // onto writer_runtime_args and patches desc-side bookkeeping.
                tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    desc,
                    sender_core,
                    writer_runtime_args);
            }
        }

        desc.kernels[writer_kernel_id].emplace_runtime_args(
            sender_core, promote_rt_args_with_buffer_bindings(writer_runtime_args));
        core_idx++;
    }

    // ==================== Runtime args for untilize cores ====================
    // Reader: tensor base addresses + token range (Buffer* slots → BufferBindings on cache hit).
    // Writer: sender NOC coords + addr-handshake + data_avail/space_avail semaphores + output/metadata buffers.
    //   u1 (local_core_id=0) → c_4/c_5/c_6 set; u2 → c_16/c_17/c_18 set.
    for (uint32_t j = 0; j < num_untilize_cores; j++) {
        uint32_t s = untilize_sender_map[j];

        // Reader RT args
        tt::tt_metal::KernelDescriptor::RTArgList untilize_reader_rt_args;
        untilize_reader_rt_args.push_back(input_tensor.buffer());
        untilize_reader_rt_args.push_back(indices_tensor.buffer());
        untilize_reader_rt_args.push_back(weights_tensor.buffer());
        untilize_reader_rt_args.push_back(offsets_tensor.buffer());
        // histograms_tensor is always passed (consumed unconditionally by rt_idx++);
        // u1 reads the address but ignores it; u2 uses it to load expert_histograms[] and
        // compute its end-of-region pointers in L1.
        untilize_reader_rt_args.push_back(histograms_tensor.buffer());
        untilize_reader_rt_args.push_back(dispatch_table_tensor.buffer());
        untilize_reader_rt_args.push_back(0u);                           // token_start_idx
        untilize_reader_rt_args.push_back((uint32_t)tokens_per_device);  // token_end_idx
        desc.kernels[reader_untilize_kernel_ids[j]].emplace_runtime_args(
            all_untilize_cores[j], untilize_reader_rt_args);

        // Writer RT args
        uint32_t local_u_id = j % num_untilizers;
        tt::tt_metal::KernelDescriptor::RTArgList untilize_writer_rt_args;
        untilize_writer_rt_args.push_back(sender_noc_coords[s].first);
        untilize_writer_rt_args.push_back(sender_noc_coords[s].second);
        if (local_u_id == 0) {
            untilize_writer_rt_args.push_back(addr_ready_semaphore_id);
            untilize_writer_rt_args.push_back(cross_c4_addr_semaphore_id);
            untilize_writer_rt_args.push_back(cross_c5_addr_semaphore_id);
            untilize_writer_rt_args.push_back(cross_c6_addr_semaphore_id);
            untilize_writer_rt_args.push_back(data_avail_semaphore_id);
            untilize_writer_rt_args.push_back(space_avail_semaphore_id);
        } else {
            untilize_writer_rt_args.push_back(addr_ready_u2_semaphore_id);
            untilize_writer_rt_args.push_back(cross_c16_addr_semaphore_id);
            untilize_writer_rt_args.push_back(cross_c17_addr_semaphore_id);
            untilize_writer_rt_args.push_back(cross_c18_addr_semaphore_id);
            untilize_writer_rt_args.push_back(data_avail_u2_semaphore_id);
            untilize_writer_rt_args.push_back(space_avail_u2_semaphore_id);
        }
        untilize_writer_rt_args.push_back(output_tensor.buffer());
        untilize_writer_rt_args.push_back(metadata_tensor.buffer());
        desc.kernels[writer_untilize_kernel_ids[j]].emplace_runtime_args(
            all_untilize_cores[j], untilize_writer_rt_args);
    }

    return desc;
}

// Row-major path: ROW_MAJOR inputs, single reader kernel per sender, no untilize cores.
tt::tt_metal::ProgramDescriptor create_at_row_major(
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
    auto weights_tensor = tensor_args.weights_tensor;
    auto offsets_tensor = tensor_args.expert_offsets_tensor;
    auto dispatch_table_tensor = tensor_args.expert_dispatch_table_tensor;

    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;
    log_debug(
        tt::LogOp,
        "Creating prefill dispatch program (row-major) for mesh coordinate: ({}, {}) with topology: {} num_links: {}",
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
    log_debug(
        tt::LogOp,
        "num_links: {}, effective_num_links: {}, tokens_per_device: {}, num_cores: {}",
        num_links,
        effective_num_links,
        tokens_per_device,
        num_cores);
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);
    log_debug(
        tt::LogOp,
        "Selected sender cores for mesh coordinate ({}, {}): {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        sender_cores);

    constexpr uint32_t read_batch_size = 8;  // matches BH DRAM bank count for full bandwidth utilization
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // c_0: input scratch (reader-only, batched DRAM reads)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        input_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_0,
        "input_scratch");
    // c_1: indices scratch (reader-only)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "indices_scratch");
    // c_2: weights scratch (reader-only)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        weights_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_2,
        "weights_scratch");
    // c_3: offsets (reader-only, full tensor)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        offsets_tensor,
        /*buffering_factor=*/detail::get_num_pages(offsets_tensor),
        /*cb_id=*/tt::CBIndex::c_3,
        "offsets_tensor");

    // c_4, c_5, c_6: reader→writer CBs for (route_info, payload, metadata) per remote entry.
    // The reader pushes all three per entry in lockstep, so small buffering (2) suffices
    // for the writer to drain concurrently. No large buffering needed.
    {
        constexpr uint32_t rw_buffering = 2;

        uint32_t route_info_page_size = l1_alignment;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = rw_buffering * route_info_page_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = tt::DataFormat::UInt32,
                .page_size = route_info_page_size,
            }}},
        });

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            input_tensor,
            /*buffering_factor=*/rw_buffering,
            /*cb_id=*/tt::CBIndex::c_5,
            "payload_for_writer");

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            metadata_tensor,
            /*buffering_factor=*/rw_buffering,
            /*cb_id=*/tt::CBIndex::c_6,
            "metadata_for_writer");
    }

    // c_7: metadata_temp (reader-only, for constructing metadata locally)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        metadata_tensor,
        /*buffering_factor=*/1,
        /*cb_id=*/tt::CBIndex::c_7,
        "metadata_temp_buffer");

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // c_8: packet header CB for fabric sends (writer-only)
    if (operation_attributes.num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;  // unicast + metadata
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

    // c_9: dispatch_table (reader-only, full tensor)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),
        /*cb_id=*/tt::CBIndex::c_9,
        "dispatch_table_tensor");

    // Iterate over every coordinate in the mesh (full coord range derived from
    // the mesh shape) — replaces the legacy `tensor_coords` parameter.
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

    // Compile-time args shared by reader and writer
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (10)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_input_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_indices_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_weights_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_offsets_id
        static_cast<uint32_t>(tt::CBIndex::c_4),  // cb_route_info_id
        static_cast<uint32_t>(tt::CBIndex::c_5),  // cb_payload_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_6),  // cb_metadata_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_7),  // cb_metadata_temp_id
        static_cast<uint32_t>(tt::CBIndex::c_8),  // cb_packet_header_id
        static_cast<uint32_t>(tt::CBIndex::c_9),  // cb_dispatch_table_id

        // Page counts (7)
        detail::get_num_pages(input_tensor),
        detail::get_num_pages(indices_tensor),
        detail::get_num_pages(weights_tensor),
        detail::get_num_pages(offsets_tensor),
        detail::get_num_pages(output_tensor),
        detail::get_num_pages(metadata_tensor),
        detail::get_num_pages(dispatch_table_tensor),

        // Page sizes (7)
        detail::get_page_size(input_tensor),
        detail::get_page_size(indices_tensor),
        detail::get_page_size(weights_tensor),
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

        // Aligned page sizes (7)
        detail::get_aligned_page_size(input_tensor),
        detail::get_aligned_page_size(indices_tensor),
        detail::get_aligned_page_size(weights_tensor),
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

    // Append TensorAccessorArgs for all 7 tensors
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weights_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(offsets_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatch_table_tensor.buffer()).append_to(compile_time_args);

    // Both reader and writer get fabric defines so the reader can compute routes
    std::map<std::string, std::string> fabric_defines;
    if (operation_attributes.num_links > 0) {
        fabric_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        fabric_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        fabric_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }
    if (operation_attributes.axis.has_value()) {
        fabric_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    // Single reader kernel shared across all senders.  (Legacy code stored one
    // handle per sender for uniform override_runtime_arguments iteration; the
    // descriptor framework uses BufferBindings on cache hit so the duplication
    // is no longer needed.)
    tt::tt_metal::KernelDescriptor reader_kd;
    reader_kd.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "reader_dispatch.cpp";
    reader_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kd.core_ranges = sender_core_grid;
    reader_kd.compile_time_args = compile_time_args;
    reader_kd.defines = {fabric_defines.begin(), fabric_defines.end()};
    reader_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
    };
    tt::tt_metal::KernelHandle reader_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(reader_kd));

    tt::tt_metal::KernelDescriptor writer_kd;
    writer_kd.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "writer_dispatch.cpp";
    writer_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kd.core_ranges = sender_core_grid;
    writer_kd.compile_time_args = compile_time_args;
    writer_kd.defines = {fabric_defines.begin(), fabric_defines.end()};
    writer_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
    };
    tt::tt_metal::KernelHandle writer_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_kd));

    // Runtime args: all cores process all tokens, experts split round-robin.
    // Build as a flat uint32_t vector so the fabric helper can extend it, then
    // promote to an RTArgList with Buffer* in the first seven slots.
    std::vector<uint32_t> base_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        weights_tensor.buffer()->address(),
        offsets_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        dispatch_table_tensor.buffer()->address(),
        (uint32_t)cross_device_semaphore.address(),
        (uint32_t)init_semaphore.address(),
        0,                            // token_start_idx (all tokens)
        (uint32_t)tokens_per_device,  // token_end_idx (all tokens)
        0,                            // dispatch_core_idx (set per core)
        num_cores,                    // num_dispatch_cores
    };

    auto promote_rt_args_with_buffer_bindings = [&](const std::vector<uint32_t>& raw_args) {
        tt::tt_metal::KernelDescriptor::RTArgList args;
        args.reserve(raw_args.size());
        args.push_back(input_tensor.buffer());
        args.push_back(indices_tensor.buffer());
        args.push_back(weights_tensor.buffer());
        args.push_back(offsets_tensor.buffer());
        args.push_back(output_tensor.buffer());
        args.push_back(metadata_tensor.buffer());
        args.push_back(dispatch_table_tensor.buffer());
        for (size_t i = 7; i < raw_args.size(); ++i) {
            args.push_back(raw_args[i]);
        }
        return args;
    };

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> reader_runtime_args = base_runtime_args;
        std::vector<uint32_t> writer_runtime_args = base_runtime_args;

        reader_runtime_args[11] = core_idx;
        writer_runtime_args[11] = core_idx;

        // Writer-only: exit semaphore address (separate from init_semaphore to avoid
        // init/exit reuse race; mirrors the combine fix).
        writer_runtime_args.push_back((uint32_t)exit_semaphore.address());

        if (operation_attributes.num_links > 0) {
            uint32_t core_link = core_idx % num_links;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }

                log_debug(
                    tt::LogOp,
                    "Connection between mesh coord ({}, {}) and ({}, {}) at core {} link {}",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    core_link);
                tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    desc,
                    sender_core,
                    writer_runtime_args);
            }
        }

        desc.kernels[reader_kernel_id].emplace_runtime_args(
            sender_core, promote_rt_args_with_buffer_bindings(reader_runtime_args));
        desc.kernels[writer_kernel_id].emplace_runtime_args(
            sender_core, promote_rt_args_with_buffer_bindings(writer_runtime_args));
        core_idx++;
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
    if (operation_attributes.use_fp8_dispatch) {
        log_warning(
            tt::LogOp,
            "Prefill dispatch: FP8 path — output buffer is allocated as UINT8 but content is Fp8_e4m3. "
            "CBs reinterpret UINT8 tensors as Fp8_e4m3 (temporary, until FP8 has a dedicated dtype).");
    }

    // Dispatch is mesh-coord-dependent (fabric routing + linearized mesh
    // coordinate are baked into kernel compile-time args), so we cannot
    // replicate one ProgramDescriptor across the whole mesh — every coord
    // gets its own build.
    for (const auto& coord : tensor_coords.coords()) {
        tt::tt_metal::ProgramDescriptor desc = is_tile_layout ? create_at_tile_layout(
                                                                    operation_attributes,
                                                                    coord,
                                                                    tensor_args,
                                                                    tensor_return_value,
                                                                    init_barrier_semaphore,
                                                                    exit_barrier_semaphore,
                                                                    final_barrier_semaphore)
                                                              : create_at_row_major(
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
