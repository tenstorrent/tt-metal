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

// Tile-layout path: TILE inputs, fused untilize across sender + idle cores.
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
    auto dispatch_table_tensor = tensor_args.expert_dispatch_table_tensor;

    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;
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

    // ==================== Core layout: senders + idle groups ====================
    // Collect all cores in the first row (y == subdevice_cores[0].y), sorted by x.
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
    TT_FATAL(
        total_row_cores > num_cores,
        "Same-row has only {} cores for {} senders — need at least one idle core per sender",
        total_row_cores,
        num_cores);

    // Divide total_row_cores into num_cores groups.
    // Within each group: center core = sender, surrounding cores = that sender's idle cores.
    uint32_t base_group_size = total_row_cores / num_cores;
    uint32_t extra_groups = total_row_cores % num_cores;

    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(num_cores);
    std::vector<std::vector<CoreCoord>> sender_idle_groups(num_cores);
    std::vector<CoreCoord> all_idle_cores;
    std::vector<uint32_t> idle_sender_map;

    {
        uint32_t pos = 0;
        for (uint32_t s = 0; s < num_cores; s++) {
            uint32_t group_size = base_group_size + (s >= num_cores - extra_groups ? 1 : 0);
            uint32_t sender_offset = group_size / 2;
            for (uint32_t j = 0; j < group_size; j++, pos++) {
                if (j == sender_offset) {
                    sender_cores.push_back(all_row_cores[pos]);
                } else {
                    sender_idle_groups[s].push_back(all_row_cores[pos]);
                    all_idle_cores.push_back(all_row_cores[pos]);
                    idle_sender_map.push_back(s);
                }
            }
        }
    }

    uint32_t num_idle_cores = static_cast<uint32_t>(all_idle_cores.size());
    TT_FATAL(
        num_idle_cores >= num_cores,
        "Same-row has only {} idle cores for {} senders — need at least one idle core per sender",
        num_idle_cores,
        num_cores);

    // Build sender_core_grid and idle_core_grid CoreRangeSets
    std::set<CoreRange> sender_ranges_set;
    for (const auto& sc : sender_cores) {
        sender_ranges_set.insert(CoreRange(sc));
    }
    auto sender_core_grid = CoreRangeSet(sender_ranges_set);

    std::set<CoreRange> idle_ranges_set;
    for (const auto& ic : all_idle_cores) {
        idle_ranges_set.insert(CoreRange(ic));
    }
    CoreRangeSet idle_core_grid(idle_ranges_set);

    // Combined grid for shared semaphores
    std::set<CoreRange> sender_and_idle_ranges;
    for (const auto& cr : sender_core_grid.ranges()) {
        sender_and_idle_ranges.insert(cr);
    }
    for (const auto& cr : idle_core_grid.ranges()) {
        sender_and_idle_ranges.insert(cr);
    }
    CoreRangeSet sender_and_idle_grid(sender_and_idle_ranges);

    log_debug(
        tt::LogOp,
        "Dispatch program: num_links: {} num_cores(senders): {} num_idle_cores: {} tokens_per_device: {}",
        num_links,
        num_cores,
        num_idle_cores,
        tokens_per_device);

    constexpr uint32_t read_batch_size = 32;

    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    uint32_t total_batches = (tokens_per_device + read_batch_size - 1) / read_batch_size;

    // ==================== Semaphores for inter-core sync ====================
    // ProgramDescriptor semaphores carry an explicit `.id` field — legacy
    // CreateSemaphore() auto-assigned the next available ID per core, so we
    // mirror that with a monotonic counter.  Every semaphore here is created on
    // the same sender_and_idle_grid, so a single counter trivially guarantees
    // per-core uniqueness.
    uint32_t next_sema_id = 0;
    auto add_sema = [&](const CoreRangeSet& crs) -> uint32_t {
        uint32_t id = next_sema_id++;
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = crs, .initial_value = 0});
        return id;
    };
    // One data_ready + one start semaphore per sender (created on sender+idle grid)
    std::vector<uint32_t> data_ready_semaphore_ids;
    std::vector<uint32_t> start_semaphore_ids;
    data_ready_semaphore_ids.reserve(num_cores);
    start_semaphore_ids.reserve(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        data_ready_semaphore_ids.push_back(add_sema(sender_and_idle_grid));
        start_semaphore_ids.push_back(add_sema(sender_and_idle_grid));
    }
    // addr_ready semaphore: sender signals idle cores after writing receive buffer address
    auto addr_ready_semaphore_id = add_sema(sender_and_idle_grid);
    // addr_value semaphore: holds the sender's c_18 L1 address (written via noc_async_write)
    auto addr_value_semaphore_id = add_sema(sender_and_idle_grid);
    // mbox_ready semaphore (per sender): idle cores signal this after NOC-writing their mailbox address.
    // Sender waits for count == num_idle_cores_in_group before reading its own scratch buffer.
    std::vector<uint32_t> mbox_ready_semaphore_ids;
    mbox_ready_semaphore_ids.reserve(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        mbox_ready_semaphore_ids.push_back(add_sema(sender_and_idle_grid));
    }
    // mbox_scratch_addr semaphore: sender writes its route-table scratch base address here and
    // multicasts it to idle cores alongside addr_ready.  Idle cores read it after addr_ready fires,
    // then NOC-write their mailbox L1 address into the sender's scratch slot (core_id * 4 offset).
    auto mbox_scratch_addr_semaphore_id = add_sema(sender_and_idle_grid);

    // ==================== Circular Buffers for IDLE cores ====================
    // c_0: tiled input stripe (reader → compute)
    detail::create_tensor_cb(
        desc,
        idle_core_grid,
        input_tensor,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_0,
        "idle_input_scratch");
    // c_10: signal CB (reader → compute)
    {
        uint32_t signal_page_size = l1_alignment;
        constexpr uint32_t signal_buffering = 2;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = signal_buffering * signal_page_size,
            .core_ranges = idle_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                .data_format = tt::DataFormat::UInt32,
                .page_size = signal_page_size,
            }}},
        });
    }
    // c_11: untilize output (compute → writer)
    // FP8 path: pack_untilize converts BF16 tiles → FP8 row-major; page size is one aligned FP8 row.
    detail::create_tensor_cb(
        desc,
        idle_core_grid,
        output_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_11,
        "idle_untilize_output");
    // c_12: route table mailbox (sender writes before start_semaphore; writer reads after)
    // Layout: [entry_count u32] [entry_0..N × 6 u32s each]
    {
        uint32_t max_route_entries = read_batch_size * operation_attributes.num_experts_per_tok;
        uint32_t mailbox_page_size = tt::round_up(
            static_cast<uint32_t>(sizeof(uint32_t)) + max_route_entries * 6 * static_cast<uint32_t>(sizeof(uint32_t)),
            l1_alignment);
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = mailbox_page_size,
            .core_ranges = idle_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_12),
                .data_format = tt::DataFormat::UInt32,
                .page_size = mailbox_page_size,
            }}},
        });
    }
    // c_13: metadata scratch (idle writer builds metadata here before NOC-writing to DRAM)
    detail::create_tensor_cb(
        desc,
        idle_core_grid,
        metadata_tensor,
        /*buffering_factor=*/1,
        /*cb_id=*/tt::CBIndex::c_13,
        "idle_metadata_scratch");

    // ==================== Circular Buffers for SENDER cores ====================
    // c_0: tiled input stripe for self-untilize (reader → compute on sender)
    // Double-buffered blocks of 8 tiles (compute processes 8 at a time)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        input_tensor,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_0,
        "sender_input_scratch");
    // c_10: signal CB for self-untilize (reader → compute on sender)
    {
        uint32_t signal_page_size = l1_alignment;
        constexpr uint32_t signal_buffering = 2;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = signal_buffering * signal_page_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                .data_format = tt::DataFormat::UInt32,
                .page_size = signal_page_size,
            }}},
        });
    }
    // c_1: indices scratch
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "indices_scratch");
    // c_2: weights scratch
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        weights_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_2,
        "weights_scratch");
    // c_3: offsets (full tensor)
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
                .data_format = tt::DataFormat::UInt8,
                .page_size = route_info_page_size,
            }}},
        });

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            output_tensor,
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
    // c_9: dispatch_table (full tensor)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),
        /*cb_id=*/tt::CBIndex::c_9,
        "dispatch_table_tensor");
    // c_18: receive buffer for untilized data from idle cores (also sender self-untilize output)
    // FP8 path: pack_untilize converts BF16 tiles → FP8 row-major; page size is one aligned FP8 row.
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        output_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_18,
        "receive_untilized");
    // c_19: route table scratch (sender builds local-expert route table here before NOC-writing to idle)
    {
        uint32_t max_route_entries = read_batch_size * operation_attributes.num_experts_per_tok;
        uint32_t mailbox_page_size = tt::round_up(
            static_cast<uint32_t>(sizeof(uint32_t)) + max_route_entries * 6 * static_cast<uint32_t>(sizeof(uint32_t)),
            l1_alignment);
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = mailbox_page_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_19),
                .data_format = tt::DataFormat::UInt32,
                .page_size = mailbox_page_size,
            }}},
        });
    }

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // c_8: packet header CB for fabric sends
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

    // ==================== Compile-time args shared by sender reader and writer ====================
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (10)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_input_id (used by sender reader for self-untilize)
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

    // ==================== Per-sender reader kernels ====================
    // Each sender gets its own reader kernel with per-sender idle core count baked in.
    auto reader_defines = fabric_defines;
    reader_defines["IS_TILE_LAYOUT"] = "1";
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    reader_kernel_ids.reserve(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        uint32_t k_s = static_cast<uint32_t>(sender_idle_groups[s].size());
        uint32_t total_workers = k_s + 1;  // idle cores + sender itself
        auto per_sender_compile_args = compile_time_args;
        per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_18));  // cb_untilize_id (receive buf)
        per_sender_compile_args.push_back(
            detail::get_aligned_page_size(output_tensor));  // aligned_row_major_input_page_size
        per_sender_compile_args.push_back(k_s);             // num_idle_cores
        per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_10));  // cb_signal_id
        per_sender_compile_args.push_back(total_workers);                             // total_workers
        per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_19));  // cb_route_table_scratch_id

        CoreRangeSet single_sender_core({CoreRange(sender_cores[s])});
        tt::tt_metal::KernelDescriptor reader_kd;
        reader_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "reader_dispatch.cpp";
        reader_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        reader_kd.core_ranges = single_sender_core;
        reader_kd.compile_time_args = std::move(per_sender_compile_args);
        reader_kd.defines = {reader_defines.begin(), reader_defines.end()};
        reader_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
        };
        reader_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
        desc.kernels.push_back(std::move(reader_kd));
    }

    // ==================== Sender writer kernel (shared across all senders) ====================
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

    // ==================== Idle core kernels ====================
    // Reader and writer kernels run on the two data-movement RISCs of each idle core
    // so that DRAM reads for the next batch overlap with the NOC write of the previous
    // batch to the owning sender's receive buffer.
    std::vector<tt::tt_metal::KernelHandle> reader_untilize_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_untilize_kernel_ids;
    reader_untilize_kernel_ids.reserve(num_idle_cores);
    writer_untilize_kernel_ids.reserve(num_idle_cores);
    for (uint32_t j = 0; j < num_idle_cores; j++) {
        uint32_t s = idle_sender_map[j];
        uint32_t k_s = static_cast<uint32_t>(sender_idle_groups[s].size());
        // Compute local core_id within this sender's idle group
        uint32_t local_core_id = 0;
        for (uint32_t g = 0; g < k_s; g++) {
            if (sender_idle_groups[s][g] == all_idle_cores[j]) {
                local_core_id = g;
                break;
            }
        }

        uint32_t total_workers = k_s + 1;  // idle cores + sender itself

        // Reader compile args (DRAM-read side)
        std::vector<uint32_t> idle_reader_compile_args = {
            static_cast<uint32_t>(tt::CBIndex::c_0),   // cb_input_id
            static_cast<uint32_t>(tt::CBIndex::c_10),  // cb_signal_id
            (uint32_t)hidden_size,
            detail::get_aligned_page_size(input_tensor),  // aligned_input_page_size
            total_batches,
            local_core_id,  // core_id within sender group
            total_workers,  // batch stride (idle cores + sender)
        };
        // Append TensorAccessorArgs for input tensor only
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(idle_reader_compile_args);

        // Writer compile args (NOC-write side)
        std::vector<uint32_t> idle_writer_compile_args = {
            static_cast<uint32_t>(tt::CBIndex::c_11),  // cb_untilize_id
            read_batch_size,
            detail::get_aligned_page_size(output_tensor),  // aligned_output_page_size
            total_batches,
            local_core_id,
            total_workers,
            // New: direct-DRAM-write support
            static_cast<uint32_t>(tt::CBIndex::c_12),        // cb_mailbox_id
            static_cast<uint32_t>(tt::CBIndex::c_13),        // cb_metadata_scratch_id
            detail::get_aligned_page_size(metadata_tensor),  // aligned_metadata_page_size
            operation_attributes.num_experts_per_tok,
            linearized_mesh_coord,
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(idle_writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(idle_writer_compile_args);

        CoreRangeSet single_idle_core({CoreRange(all_idle_cores[j])});
        tt::tt_metal::KernelDescriptor idle_reader_kd;
        idle_reader_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "reader_untilize_dispatch.cpp";
        idle_reader_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        idle_reader_kd.core_ranges = single_idle_core;
        idle_reader_kd.compile_time_args = std::move(idle_reader_compile_args);
        idle_reader_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
        };
        reader_untilize_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
        desc.kernels.push_back(std::move(idle_reader_kd));

        tt::tt_metal::KernelDescriptor idle_writer_kd;
        idle_writer_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "writer_untilize_dispatch.cpp";
        idle_writer_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        idle_writer_kd.core_ranges = single_idle_core;
        idle_writer_kd.compile_time_args = std::move(idle_writer_compile_args);
        idle_writer_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
        };
        writer_untilize_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
        desc.kernels.push_back(std::move(idle_writer_kd));
    }

    // Compute kernel on idle cores
    {
        tt::tt_metal::KernelDescriptor idle_compute_kd;
        idle_compute_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/compute/"
            "untilize_dispatch.cpp";
        idle_compute_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        idle_compute_kd.core_ranges = idle_core_grid;
        idle_compute_kd.compile_time_args = {
            static_cast<uint32_t>(tt::CBIndex::c_10),  // cb_signal_id
            static_cast<uint32_t>(tt::CBIndex::c_11),  // cb_untilize_id
            static_cast<uint32_t>(tt::CBIndex::c_0),   // cb_in_id
            (uint32_t)hidden_size,
            read_batch_size,
        };
        // Blackhole requires the DEST register in 32-bit mode whenever any CB on the core uses an
        // 8-bit float format (Fp8_e4m3). The FP8 dispatch path makes the untilized output CB
        // Fp8_e4m3, so fp32_dest_acc_en must be enabled. 32-bit DEST halves the pack_untilize block
        // budget (8->4 tiles) under half-sync, but untilize_dispatch packs block_ct_dim=8, so
        // dst_full_sync_en restores the 8-tile budget. Mirrors untilize_combine; only on the FP8 path.
        idle_compute_kd.config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = operation_attributes.use_fp8_dispatch,
            .dst_full_sync_en = operation_attributes.use_fp8_dispatch,
        };
        desc.kernels.push_back(std::move(idle_compute_kd));
    }

    // Compute kernel on sender cores for self-untilize (output goes to c_18)
    {
        tt::tt_metal::KernelDescriptor sender_compute_kd;
        sender_compute_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/compute/"
            "untilize_dispatch.cpp";
        sender_compute_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        sender_compute_kd.core_ranges = sender_core_grid;
        sender_compute_kd.compile_time_args = {
            static_cast<uint32_t>(tt::CBIndex::c_10),  // cb_signal_id
            static_cast<uint32_t>(tt::CBIndex::c_18),  // cb_untilize_id (reuse receive buffer)
            static_cast<uint32_t>(tt::CBIndex::c_0),   // cb_in_id
            (uint32_t)hidden_size,
            read_batch_size,
        };
        // Same FP8 DEST requirement as the idle compute kernel above: Fp8_e4m3 output CB needs
        // 32-bit DEST (fp32_dest_acc_en) + full-sync for the block_ct_dim=8 pack_untilize. FP8 path only.
        sender_compute_kd.config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = operation_attributes.use_fp8_dispatch,
            .dst_full_sync_en = operation_attributes.use_fp8_dispatch,
        };
        desc.kernels.push_back(std::move(sender_compute_kd));
    }

    // ==================== Pre-compute NOC coordinates ====================
    std::vector<std::pair<uint32_t, uint32_t>> sender_noc_coords;
    for (const auto& sc : sender_cores) {
        auto noc_coord = mesh_device->virtual_core_from_logical_core(sc, tt::CoreType::WORKER);
        sender_noc_coords.emplace_back(noc_coord.x, noc_coord.y);
    }

    // Per-sender multicast/idle info
    struct SenderIdleCfg {
        std::vector<std::pair<uint32_t, uint32_t>> idle_noc_coords;
        uint32_t mcast_x_start = 0, mcast_y_start = 0, mcast_x_end = 0, mcast_y_end = 0;
    };
    std::vector<SenderIdleCfg> sender_idle_cfgs(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        bool first = true;
        for (const auto& ic : sender_idle_groups[s]) {
            auto noc = mesh_device->virtual_core_from_logical_core(ic, tt::CoreType::WORKER);
            uint32_t nx = (uint32_t)noc.x, ny = (uint32_t)noc.y;
            sender_idle_cfgs[s].idle_noc_coords.emplace_back(nx, ny);
            if (first) {
                sender_idle_cfgs[s].mcast_x_start = sender_idle_cfgs[s].mcast_x_end = nx;
                sender_idle_cfgs[s].mcast_y_start = sender_idle_cfgs[s].mcast_y_end = ny;
                first = false;
            } else {
                sender_idle_cfgs[s].mcast_x_start = std::min(sender_idle_cfgs[s].mcast_x_start, nx);
                sender_idle_cfgs[s].mcast_x_end = std::max(sender_idle_cfgs[s].mcast_x_end, nx);
                sender_idle_cfgs[s].mcast_y_start = std::min(sender_idle_cfgs[s].mcast_y_start, ny);
                sender_idle_cfgs[s].mcast_y_end = std::max(sender_idle_cfgs[s].mcast_y_end, ny);
            }
        }
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
        std::vector<uint32_t> reader_runtime_args = base_runtime_args;
        std::vector<uint32_t> writer_runtime_args = base_runtime_args;

        reader_runtime_args[11] = core_idx;  // dispatch_core_idx
        writer_runtime_args[11] = core_idx;

        // Writer-only: exit semaphore address (separate from init_semaphore to avoid
        // init/exit reuse race where a fast peer's exit-inc lands during the post-init
        // set(0) window).
        writer_runtime_args.push_back((uint32_t)exit_semaphore.address());

        // Inter-core sync args for reader
        reader_runtime_args.push_back(data_ready_semaphore_ids[core_idx]);
        reader_runtime_args.push_back(start_semaphore_ids[core_idx]);
        reader_runtime_args.push_back(addr_ready_semaphore_id);
        reader_runtime_args.push_back(addr_value_semaphore_id);
        reader_runtime_args.push_back(mbox_ready_semaphore_ids[core_idx]);
        reader_runtime_args.push_back(mbox_scratch_addr_semaphore_id);

        // Pass NOC coords of this sender's idle cores (for per-batch unicast start signal)
        for (const auto& [noc_x, noc_y] : sender_idle_cfgs[core_idx].idle_noc_coords) {
            reader_runtime_args.push_back(noc_x);
            reader_runtime_args.push_back(noc_y);
        }
        // Bounding box for multicast of addr_value / addr_ready to the idle group
        reader_runtime_args.push_back(sender_idle_cfgs[core_idx].mcast_x_start);
        reader_runtime_args.push_back(sender_idle_cfgs[core_idx].mcast_y_start);
        reader_runtime_args.push_back(sender_idle_cfgs[core_idx].mcast_x_end);
        reader_runtime_args.push_back(sender_idle_cfgs[core_idx].mcast_y_end);

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

        desc.kernels[reader_kernel_ids[core_idx]].emplace_runtime_args(
            sender_core, promote_rt_args_with_buffer_bindings(reader_runtime_args));
        desc.kernels[writer_kernel_id].emplace_runtime_args(
            sender_core, promote_rt_args_with_buffer_bindings(writer_runtime_args));
        core_idx++;
    }

    // ==================== Runtime args for idle cores ====================
    // The sender's c_18 L1 address is communicated at runtime: sender uses noc_async_write
    // to copy it to each idle core's addr_value_sem, then signals addr_ready_sem.
    // Reader only needs the input tensor address; writer owns the sender-sync semaphores
    // and NOC coords.
    for (uint32_t j = 0; j < num_idle_cores; j++) {
        uint32_t s = idle_sender_map[j];

        // Idle reader: only RT arg is the input tensor base address — pushed as
        // Buffer* so the framework records a BufferBinding.
        tt::tt_metal::KernelDescriptor::RTArgList idle_reader_rt_args;
        idle_reader_rt_args.push_back(input_tensor.buffer());
        desc.kernels[reader_untilize_kernel_ids[j]].emplace_runtime_args(all_idle_cores[j], idle_reader_rt_args);

        // Idle writer: output_tensor (slot 6) and metadata_tensor (slot 7) are
        // pushed as Buffer*; all other slots are plain uint32_t.
        tt::tt_metal::KernelDescriptor::RTArgList idle_writer_rt_args;
        idle_writer_rt_args.push_back(data_ready_semaphore_ids[s]);
        idle_writer_rt_args.push_back(start_semaphore_ids[s]);
        idle_writer_rt_args.push_back(sender_noc_coords[s].first);
        idle_writer_rt_args.push_back(sender_noc_coords[s].second);
        idle_writer_rt_args.push_back(addr_ready_semaphore_id);
        idle_writer_rt_args.push_back(addr_value_semaphore_id);
        idle_writer_rt_args.push_back(output_tensor.buffer());
        idle_writer_rt_args.push_back(metadata_tensor.buffer());
        idle_writer_rt_args.push_back(mbox_ready_semaphore_ids[s]);
        idle_writer_rt_args.push_back(mbox_scratch_addr_semaphore_id);
        desc.kernels[writer_untilize_kernel_ids[j]].emplace_runtime_args(all_idle_cores[j], idle_writer_rt_args);
    }

    return desc;
}

// Row-major path: ROW_MAJOR inputs, single reader kernel per sender, no idle cores.
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
                .data_format = tt::DataFormat::UInt8,
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
