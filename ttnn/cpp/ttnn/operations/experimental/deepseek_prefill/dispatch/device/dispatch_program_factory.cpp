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
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
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

void create_tensor_cb(
    tt::tt_metal::Program& program,
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

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
}

}  // namespace detail

namespace {

// Tile-layout path: TILE inputs, fused untilize across sender + untilize cores.
ttnn::device_operation::CachedProgram<DispatchSharedVariables> create_at_tile_layout(
    const DispatchParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& exit_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    tt::tt_metal::Program program{};

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

    // ==================== Core layout: senders + untilize cores ====================
    // Collect all cores in the first row (y == subdevice_cores[0].y), sorted by x.
    // Each sender owns exactly one untilize core: cores are paired (sender, untilize)
    // consecutively along x.
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
        total_row_cores >= 2 * num_cores,
        "Same-row has only {} cores for {} senders — need one untilize core per sender (>= {} required)",
        total_row_cores,
        num_cores,
        2 * num_cores);

    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(num_cores);
    std::vector<std::vector<CoreCoord>> sender_untilize_groups(num_cores);
    std::vector<CoreCoord> all_untilize_cores;
    all_untilize_cores.reserve(num_cores);
    std::vector<uint32_t> untilize_sender_map;
    untilize_sender_map.reserve(num_cores);

    for (uint32_t s = 0; s < num_cores; s++) {
        sender_cores.push_back(all_row_cores[2 * s]);
        sender_untilize_groups[s].push_back(all_row_cores[2 * s + 1]);
        all_untilize_cores.push_back(all_row_cores[2 * s + 1]);
        untilize_sender_map.push_back(s);
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
    // Per-entry pipeline (untilize → sender writer CB → fabric):
    //   Sender writer CBs (c_4/c_5/c_6) are writer_cb_size slots deep (one batch worth).
    //   data_avail_semaphore_ids[s] (on sender L1, init=0): untilize NOC-incs per entry
    //       written into the sender slot. Sender writer polls locally.
    //   space_avail_semaphore_ids[s] (on untilize L1, init=writer_cb_size): sender writer
    //       NOC-incs per entry that has been fabric-sent (slot is safe to overwrite). The
    //       initial value seeds untilize so it can fill all slots before the first writer
    //       credit lands.
    // Address handshake (sender writer → untilize):
    //   addr_ready_semaphore_id: signals untilize that writer CB base addresses are valid.
    //   cross_c{4,5,6}_addr_semaphore_id: hold the writer CB (c_4/c_5/c_6) L1 base
    //       addresses (used as 1×u32 scratch on both sides).
    constexpr uint32_t writer_cb_size = read_batch_size;  // 32 slots — one batch deep, per-entry handshake
    std::vector<uint32_t> data_avail_semaphore_ids;
    std::vector<uint32_t> space_avail_semaphore_ids;
    data_avail_semaphore_ids.reserve(num_cores);
    space_avail_semaphore_ids.reserve(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        data_avail_semaphore_ids.push_back(tt::tt_metal::CreateSemaphore(program, sender_and_untilize_grid, 0));
        space_avail_semaphore_ids.push_back(
            tt::tt_metal::CreateSemaphore(program, sender_and_untilize_grid, writer_cb_size));
    }
    auto addr_ready_semaphore_id = tt::tt_metal::CreateSemaphore(program, sender_and_untilize_grid, 0);
    auto cross_c4_addr_semaphore_id = tt::tt_metal::CreateSemaphore(program, sender_and_untilize_grid, 0);
    auto cross_c5_addr_semaphore_id = tt::tt_metal::CreateSemaphore(program, sender_and_untilize_grid, 0);
    auto cross_c6_addr_semaphore_id = tt::tt_metal::CreateSemaphore(program, sender_and_untilize_grid, 0);

    // ==================== Circular Buffers for untilize cores ====================
    // Routing decisions and offsets[] live on the untilize core now — sender is fabric-only.
    // c_0: tiled input stripe (reader → compute)
    detail::create_tensor_cb(
        program,
        untilize_core_grid,
        input_tensor,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_0,
        "untilize_input_scratch");
    // c_1: indices scratch (untilize reader does per-batch DRAM reads)
    detail::create_tensor_cb(
        program,
        untilize_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "untilize_indices_scratch");
    // c_2: weights scratch
    detail::create_tensor_cb(
        program,
        untilize_core_grid,
        weights_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_2,
        "untilize_weights_scratch");
    // c_3: offsets (full tensor, loaded once at startup, mutated in place per batch)
    detail::create_tensor_cb(
        program,
        untilize_core_grid,
        offsets_tensor,
        /*buffering_factor=*/detail::get_num_pages(offsets_tensor),
        /*cb_id=*/tt::CBIndex::c_3,
        "untilize_offsets_tensor");
    // c_9: dispatch_table (full tensor, loaded once at startup)
    detail::create_tensor_cb(
        program,
        untilize_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),
        /*cb_id=*/tt::CBIndex::c_9,
        "untilize_dispatch_table_tensor");
    // c_10: signal CB (reader → compute)
    {
        uint32_t signal_page_size = l1_alignment;
        constexpr uint32_t signal_buffering = 2;
        tt::tt_metal::CircularBufferConfig signal_cb_config =
            tt::tt_metal::CircularBufferConfig(
                signal_buffering * signal_page_size, {{tt::CBIndex::c_10, tt::DataFormat::UInt32}})
                .set_page_size(tt::CBIndex::c_10, signal_page_size);
        tt::tt_metal::CreateCircularBuffer(program, untilize_core_grid, signal_cb_config);
    }
    // c_11: untilize output (compute → writer)
    // Double-buffered at batch granularity: two slots of read_batch_size tokens so compute can
    // pack the next batch while the writer is still draining the previous one.
    detail::create_tensor_cb(
        program,
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
        program,
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
        tt::tt_metal::CircularBufferConfig route_info_scratch_cb_config =
            tt::tt_metal::CircularBufferConfig(route_info_scratch_size, {{tt::CBIndex::c_15, tt::DataFormat::UInt32}})
                .set_page_size(tt::CBIndex::c_15, route_info_scratch_size);
        tt::tt_metal::CreateCircularBuffer(program, untilize_core_grid, route_info_scratch_cb_config);
    }
    // c_14: per-batch route plan (reader RISC → writer RISC, on same untilize core).
    // Layout: [entry_count u32][padding to 32B][entries × 8 u32 each]
    {
        constexpr uint32_t plan_entry_u32s = 8;
        uint32_t max_plan_entries = read_batch_size * operation_attributes.num_experts_per_tok;
        uint32_t plan_page_size = tt::round_up(
            32u + max_plan_entries * plan_entry_u32s * static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);
        constexpr uint32_t plan_buffering = 2;
        tt::tt_metal::CircularBufferConfig plan_cb_config =
            tt::tt_metal::CircularBufferConfig(
                plan_buffering * plan_page_size, {{tt::CBIndex::c_14, tt::DataFormat::UInt32}})
                .set_page_size(tt::CBIndex::c_14, plan_page_size);
        tt::tt_metal::CreateCircularBuffer(program, untilize_core_grid, plan_cb_config);
    }

    // ==================== Circular Buffers for SENDER cores ====================
    // Direct-consume pipeline on sender:
    //   c_4/c_5/c_6 = the only data CBs. Untilize NOC-writes per-entry directly into
    //                 these slots; sender writer (RISCV_0) consumes them via local
    //                 semaphore wait + manual slot math and fabric-sends. Sized
    //                 writer_cb_size = num_sub_batches * sub_batch_slots = 34.
    {
        uint32_t route_info_page_size = l1_alignment;

        tt::tt_metal::CircularBufferConfig writer_route_info_cb_config =
            tt::tt_metal::CircularBufferConfig(
                writer_cb_size * route_info_page_size, {{tt::CBIndex::c_4, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_4, route_info_page_size);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, writer_route_info_cb_config);

        detail::create_tensor_cb(
            program,
            sender_core_grid,
            output_tensor,
            /*buffering_factor=*/writer_cb_size,
            /*cb_id=*/tt::CBIndex::c_5,
            "payload_for_writer");

        detail::create_tensor_cb(
            program,
            sender_core_grid,
            metadata_tensor,
            /*buffering_factor=*/writer_cb_size,
            /*cb_id=*/tt::CBIndex::c_6,
            "metadata_for_writer");
    }

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // c_8: packet header CB for fabric sends
    if (operation_attributes.num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_header_cb_size, {{tt::CBIndex::c_8, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_8, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);
    }

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
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
    //   * Startup address handshake to untilize (publishes c_4/c_5/c_6 base L1 addresses,
    //     NOC-incs untilize addr_ready).
    //   * Fabric init + per-entry fabric send.
    //   * Per-entry direct credit to untilize space_avail after each fabric send.
    auto writer_defines = fabric_defines;
    writer_defines["IS_TILE_LAYOUT"] = "1";

    std::vector<uint32_t> writer_compile_time_args = compile_time_args;
    writer_compile_time_args.push_back(writer_cb_size);  // sender writer CB depth

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "writer_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});

    // ==================== untilize core kernels ====================
    // Reader and writer kernels run on the two data-movement RISCs of each untilize core
    // so that DRAM reads for the next batch overlap with the NOC write of the previous
    // batch to the owning sender's receive buffer.
    std::vector<tt::tt_metal::KernelHandle> reader_untilize_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_untilize_kernel_ids;
    reader_untilize_kernel_ids.reserve(num_untilize_cores);
    writer_untilize_kernel_ids.reserve(num_untilize_cores);
    for (uint32_t j = 0; j < num_untilize_cores; j++) {
        uint32_t s = untilize_sender_map[j];
        // 1:1 sender ↔ untilize pairing — each untilize core processes ALL its sender's batches.
        constexpr uint32_t local_core_id = 0;
        constexpr uint32_t total_workers = 1;

        // ===== Reader compile args =====
        // Routing decision lives here: needs the same tensors and parameters the old sender
        // reader had, plus c_14 (route plan published to writer on the same core).
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

        // ===== Writer compile args =====
        // Data movement only — drains c_14 plan and executes local/cross-device writes.
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
            writer_cb_size,                                              // 11: sender writer CB size (=2)
            static_cast<uint32_t>(tt::CBIndex::c_15),                    // 12: cb_route_info_scratch_id
            read_batch_size * operation_attributes.num_experts_per_tok,  // 13: meta_scratch_slots
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(untilize_writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(untilize_writer_compile_args);

        auto untilize_kernel_defines = fabric_defines;  // carries AXIS define if set

        CoreRangeSet single_untilize_core({CoreRange(all_untilize_cores[j])});
        reader_untilize_kernel_ids.push_back(tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "reader_untilize_dispatch.cpp",
            single_untilize_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
                .compile_args = untilize_reader_compile_args,
                .defines = untilize_kernel_defines}));

        writer_untilize_kernel_ids.push_back(tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
            "writer_untilize_dispatch.cpp",
            single_untilize_core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
                .compile_args = untilize_writer_compile_args}));
    }

    // Compute kernel on untilize cores
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/compute/"
        "untilize_dispatch.cpp",
        untilize_core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = {
                static_cast<uint32_t>(tt::CBIndex::c_10),  // cb_signal_id
                static_cast<uint32_t>(tt::CBIndex::c_11),  // cb_untilize_id
                static_cast<uint32_t>(tt::CBIndex::c_0),   // cb_in_id
                (uint32_t)hidden_size,
                read_batch_size,
            }});

    // ==================== Pre-compute NOC coordinates ====================
    std::vector<std::pair<uint32_t, uint32_t>> sender_noc_coords;
    for (const auto& sc : sender_cores) {
        auto noc_coord = mesh_device->virtual_core_from_logical_core(sc, tt::CoreType::WORKER);
        sender_noc_coords.emplace_back(noc_coord.x, noc_coord.y);
    }

    // Per-sender multicast/untilize info
    struct SenderuntilizeCfg {
        std::vector<std::pair<uint32_t, uint32_t>> untilize_noc_coords;
        uint32_t mcast_x_start = 0, mcast_y_start = 0, mcast_x_end = 0, mcast_y_end = 0;
    };
    std::vector<SenderuntilizeCfg> sender_untilize_cfgs(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        bool first = true;
        for (const auto& ic : sender_untilize_groups[s]) {
            auto noc = mesh_device->virtual_core_from_logical_core(ic, tt::CoreType::WORKER);
            uint32_t nx = (uint32_t)noc.x, ny = (uint32_t)noc.y;
            sender_untilize_cfgs[s].untilize_noc_coords.emplace_back(nx, ny);
            if (first) {
                sender_untilize_cfgs[s].mcast_x_start = sender_untilize_cfgs[s].mcast_x_end = nx;
                sender_untilize_cfgs[s].mcast_y_start = sender_untilize_cfgs[s].mcast_y_end = ny;
                first = false;
            } else {
                sender_untilize_cfgs[s].mcast_x_start = std::min(sender_untilize_cfgs[s].mcast_x_start, nx);
                sender_untilize_cfgs[s].mcast_x_end = std::max(sender_untilize_cfgs[s].mcast_x_end, nx);
                sender_untilize_cfgs[s].mcast_y_start = std::min(sender_untilize_cfgs[s].mcast_y_start, ny);
                sender_untilize_cfgs[s].mcast_y_end = std::max(sender_untilize_cfgs[s].mcast_y_end, ny);
            }
        }
    }

    // ==================== Runtime args for sender cores ====================
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

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        // The j-th untilize core in all_untilize_cores corresponds to sender core_idx via the
        // untilize_sender_map built earlier (1:1). Find it for NOC coord access.
        uint32_t untilize_idx = 0;
        for (uint32_t j = 0; j < num_untilize_cores; j++) {
            if (untilize_sender_map[j] == core_idx) {
                untilize_idx = j;
                break;
            }
        }
        auto untilize_noc =
            mesh_device->virtual_core_from_logical_core(all_untilize_cores[untilize_idx], tt::CoreType::WORKER);
        uint32_t untilize_noc_x = (uint32_t)untilize_noc.x;
        uint32_t untilize_noc_y = (uint32_t)untilize_noc.y;

        std::vector<uint32_t> writer_runtime_args = base_runtime_args;
        writer_runtime_args[11] = core_idx;  // dispatch_core_idx

        // Writer-only: exit semaphore address (avoids init/exit reuse race).
        writer_runtime_args.push_back((uint32_t)exit_semaphore.address());

        // ===== Sender writer (tile-layout): handshake + per-entry fabric send + credit =====
        writer_runtime_args.push_back(addr_ready_semaphore_id);
        writer_runtime_args.push_back(cross_c4_addr_semaphore_id);
        writer_runtime_args.push_back(cross_c5_addr_semaphore_id);
        writer_runtime_args.push_back(cross_c6_addr_semaphore_id);
        writer_runtime_args.push_back(untilize_noc_x);
        writer_runtime_args.push_back(untilize_noc_y);
        writer_runtime_args.push_back(data_avail_semaphore_ids[core_idx]);
        writer_runtime_args.push_back(space_avail_semaphore_ids[core_idx]);

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
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    program,
                    sender_core,
                    writer_runtime_args);
            }
        }

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core, writer_runtime_args);
        core_idx++;
    }

    // ==================== Runtime args for untilize cores ====================
    for (uint32_t j = 0; j < num_untilize_cores; j++) {
        uint32_t s = untilize_sender_map[j];

        // Reader: tensor base addresses + token range
        std::vector<uint32_t> untilize_reader_rt_args = {
            input_tensor.buffer()->address(),
            indices_tensor.buffer()->address(),
            weights_tensor.buffer()->address(),
            offsets_tensor.buffer()->address(),
            dispatch_table_tensor.buffer()->address(),
            0u,                           // token_start_idx
            (uint32_t)tokens_per_device,  // token_end_idx
        };
        tt::tt_metal::SetRuntimeArgs(
            program, reader_untilize_kernel_ids[j], all_untilize_cores[j], untilize_reader_rt_args);

        // Writer: sender NOC coords, address-handshake semaphores, data_avail/space_avail, output/metadata tensors.
        std::vector<uint32_t> untilize_writer_rt_args = {
            sender_noc_coords[s].first,
            sender_noc_coords[s].second,
            addr_ready_semaphore_id,
            cross_c4_addr_semaphore_id,
            cross_c5_addr_semaphore_id,
            cross_c6_addr_semaphore_id,
            data_avail_semaphore_ids[s],
            space_avail_semaphore_ids[s],
            output_tensor.buffer()->address(),
            metadata_tensor.buffer()->address(),
        };
        tt::tt_metal::SetRuntimeArgs(
            program, writer_untilize_kernel_ids[j], all_untilize_cores[j], untilize_writer_rt_args);
    }

    return {
        std::move(program),
        {.reader_kernel_ids = {},
         .writer_kernel_id = writer_kernel_id,
         .reader_untilize_kernel_ids = std::move(reader_untilize_kernel_ids),
         .writer_untilize_kernel_ids = std::move(writer_untilize_kernel_ids),
         .cores = sender_cores,
         .untilize_cores = all_untilize_cores,
         .init_semaphore = init_semaphore,
         .exit_semaphore = exit_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

// Row-major path: ROW_MAJOR inputs, single reader kernel per sender, no untilize cores.
ttnn::device_operation::CachedProgram<DispatchSharedVariables> create_at_row_major(
    const DispatchParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& exit_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    tt::tt_metal::Program program{};

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
        program,
        sender_core_grid,
        input_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_0,
        "input_scratch");
    // c_1: indices scratch (reader-only)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "indices_scratch");
    // c_2: weights scratch (reader-only)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        weights_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_2,
        "weights_scratch");
    // c_3: offsets (reader-only, full tensor)
    detail::create_tensor_cb(
        program,
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
        tt::tt_metal::CircularBufferConfig route_info_cb_config =
            tt::tt_metal::CircularBufferConfig(
                rw_buffering * route_info_page_size, {{tt::CBIndex::c_4, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_4, route_info_page_size);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, route_info_cb_config);

        detail::create_tensor_cb(
            program,
            sender_core_grid,
            input_tensor,
            /*buffering_factor=*/rw_buffering,
            /*cb_id=*/tt::CBIndex::c_5,
            "payload_for_writer");

        detail::create_tensor_cb(
            program,
            sender_core_grid,
            metadata_tensor,
            /*buffering_factor=*/rw_buffering,
            /*cb_id=*/tt::CBIndex::c_6,
            "metadata_for_writer");
    }

    // c_7: metadata_temp (reader-only, for constructing metadata locally)
    detail::create_tensor_cb(
        program,
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

        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_header_cb_size, {{tt::CBIndex::c_8, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_8, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);
    }

    // c_9: dispatch_table (reader-only, full tensor)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),
        /*cb_id=*/tt::CBIndex::c_9,
        "dispatch_table_tensor");

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
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

    // Single reader kernel shared across all senders; store one handle per sender for
    // uniform override_runtime_arguments iteration between tile and row-major paths.
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "reader_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = fabric_defines});
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids(num_cores, reader_kernel_id);

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "writer_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = fabric_defines});

    // Runtime args: all cores process all tokens, experts split round-robin
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
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    program,
                    sender_core,
                    writer_runtime_args);
            }
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core, writer_runtime_args);
        core_idx++;
    }

    return {
        std::move(program),
        {.reader_kernel_ids = std::move(reader_kernel_ids),
         .writer_kernel_id = writer_kernel_id,
         .reader_untilize_kernel_ids = {},
         .writer_untilize_kernel_ids = {},
         .cores = sender_cores,
         .untilize_cores = {},
         .init_semaphore = init_semaphore,
         .exit_semaphore = exit_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

}  // namespace

DispatchProgramFactory::cached_mesh_workload_t DispatchProgramFactory::create_mesh_workload(
    const DispatchParams& operation_attributes,
    const MeshCoordinateRangeSet& tensor_coords,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, DispatchSharedVariables> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();

    auto sem_buffer_type = operation_attributes.use_l1_small_for_semaphores ? tt::tt_metal::BufferType::L1_SMALL
                                                                            : tt::tt_metal::BufferType::L1;
    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    auto exit_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            tensor_coords,
            init_barrier_semaphore,
            exit_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<DispatchSharedVariables> DispatchProgramFactory::create_at(
    const DispatchParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& exit_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    const bool is_tile_layout = tensor_args.input_tensor.layout() == tt::tt_metal::Layout::TILE;
    log_info(tt::LogOp, "Prefill dispatch: input tensor is {} layout", is_tile_layout ? "TILE" : "ROW_MAJOR");
    if (operation_attributes.use_fp8_dispatch) {
        log_warning(
            tt::LogOp,
            "Prefill dispatch: FP8 path — output buffer is allocated as UINT8 but content is Fp8_e4m3. "
            "CBs reinterpret UINT8 tensors as Fp8_e4m3 (temporary, until FP8 has a dedicated dtype).");
    }
    if (is_tile_layout) {
        return create_at_tile_layout(
            operation_attributes,
            mesh_coordinate,
            tensor_args,
            tensor_return_value,
            tensor_coords,
            init_semaphore,
            exit_semaphore,
            cross_device_semaphore);
    }
    return create_at_row_major(
        operation_attributes,
        mesh_coordinate,
        tensor_args,
        tensor_return_value,
        tensor_coords,
        init_semaphore,
        exit_semaphore,
        cross_device_semaphore);
}

void DispatchProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DispatchParams& /*operation_attributes*/,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value) {
    const bool is_tile_layout = tensor_args.input_tensor.layout() == tt::tt_metal::Layout::TILE;
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& writer_kernel_id = shared_variables.writer_kernel_id;
        const auto& cores = shared_variables.cores;

        const auto& output_tensor = tensor_return_value.at(0);
        const auto& metadata_tensor = tensor_return_value.at(1);

        for (size_t s = 0; s < cores.size(); s++) {
            const auto& core = cores[s];
            // Tile-layout has no sender reader RISC (writer drives both handshake and consume),
            // so reader_kernel_ids is empty there.
            if (!shared_variables.reader_kernel_ids.empty()) {
                auto& reader_runtime_args =
                    tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel_ids[s], core);
                reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
                reader_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
                reader_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
                reader_runtime_args.at(3) = tensor_args.expert_offsets_tensor.buffer()->address();
                reader_runtime_args.at(4) = output_tensor.buffer()->address();
                reader_runtime_args.at(5) = metadata_tensor.buffer()->address();
                reader_runtime_args.at(6) = tensor_args.expert_dispatch_table_tensor.buffer()->address();
                reader_runtime_args.at(7) = (uint32_t)shared_variables.cross_device_semaphore.address();
                reader_runtime_args.at(8) = (uint32_t)shared_variables.init_semaphore.address();
            }
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);

            writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
            writer_runtime_args.at(3) = tensor_args.expert_offsets_tensor.buffer()->address();
            writer_runtime_args.at(4) = output_tensor.buffer()->address();
            writer_runtime_args.at(5) = metadata_tensor.buffer()->address();
            writer_runtime_args.at(6) = tensor_args.expert_dispatch_table_tensor.buffer()->address();
            writer_runtime_args.at(7) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(8) = (uint32_t)shared_variables.init_semaphore.address();
            // Index 13 is the writer-only exit_semaphore.address() pushed in create_at_*.
            // base_runtime_args has 13 entries (0..12), then writer push_back's exit_semaphore at 13.
            writer_runtime_args.at(13) = (uint32_t)shared_variables.exit_semaphore.address();
        }

        // untilize cores only exist on the tile-layout path.
        if (is_tile_layout) {
            for (size_t i = 0; i < shared_variables.untilize_cores.size(); i++) {
                auto& untilize_reader_rt_args = tt::tt_metal::GetRuntimeArgs(
                    program, shared_variables.reader_untilize_kernel_ids[i], shared_variables.untilize_cores[i]);
                // Indices 0..4 = input, indices, weights, offsets, dispatch_table addresses.
                untilize_reader_rt_args.at(0) = tensor_args.input_tensor.buffer()->address();
                untilize_reader_rt_args.at(1) = tensor_args.indices_tensor.buffer()->address();
                untilize_reader_rt_args.at(2) = tensor_args.weights_tensor.buffer()->address();
                untilize_reader_rt_args.at(3) = tensor_args.expert_offsets_tensor.buffer()->address();
                untilize_reader_rt_args.at(4) = tensor_args.expert_dispatch_table_tensor.buffer()->address();

                auto& untilize_writer_rt_args = tt::tt_metal::GetRuntimeArgs(
                    program, shared_variables.writer_untilize_kernel_ids[i], shared_variables.untilize_cores[i]);
                // Indices 8, 9 = output, metadata tensor addresses (after the 8 sync args).
                untilize_writer_rt_args.at(8) = output_tensor.buffer()->address();
                untilize_writer_rt_args.at(9) = metadata_tensor.buffer()->address();
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
