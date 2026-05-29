// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_device_operation.hpp"
#include <algorithm>
#include <array>
#include <utility>
#include <limits>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <ttnn/global_semaphore.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine {

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

// ProgramDescriptor-flavored helper.  Mirrors the legacy create_tensor_cb but
// pushes a CBDescriptor onto the desc instead of calling CreateCircularBuffer.
void create_tensor_cb(
    tt::tt_metal::ProgramDescriptor& desc,
    const CoreRangeSet& core_range_set,
    const ttnn::Tensor& tensor,
    uint32_t buffering_factor,
    tt::CBIndex cb_id,
    const std::string& tensor_name = "tensor") {
    auto page_size = get_page_size(tensor);
    auto num_pages = get_num_pages(tensor);
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

// Per-coord ProgramDescriptor builder.  The cross-device GlobalSemaphores are
// allocated once at workload scope in create_workload_descriptor() and passed
// down by const-reference so every per-coord program references the same
// device-side allocation (writer runtime args bake in `init_semaphore.address()`
// / `exit_semaphore.address()` as absolute addresses).
tt::tt_metal::ProgramDescriptor build_program_for_coord(
    const CombineParams& operation_attributes,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& exit_semaphore) {
    tt::tt_metal::ProgramDescriptor desc;

    const auto& dispatched_buffer = tensor_args.dispatched_buffer;
    const auto& dispatched_metadata = tensor_args.dispatched_metadata;
    const auto& expert_token_counts = tensor_args.expert_token_counts;
    const auto& expert_region_offsets = tensor_args.expert_region_offsets;
    const auto& output_tensor = tensor_return_value;
    const bool is_tile_layout = dispatched_buffer.layout() == tt::tt_metal::Layout::TILE;

    auto* mesh_device = dispatched_buffer.device();
    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    const auto& mesh_view = mesh_device->get_view();
    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ccl::common::get_linearized_index(mesh_coordinate, mesh_view);
    uint32_t mesh_rows = mesh_view.num_rows();
    uint32_t mesh_cols = mesh_view.num_cols();

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    log_debug(
        tt::LogOp,
        "Creating prefill combine program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized: {} mesh shape: ({}, {}) topology: {} num_links: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord,
        mesh_rows,
        mesh_cols,
        topology,
        num_links);

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    auto dispatched_shape = dispatched_buffer.logical_shape();
    auto hidden_size = dispatched_shape[-1];
    auto max_dispatch_buffer_token_size = dispatched_shape[-2];

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    // Maximum worker cores: one per fabric link.
    constexpr uint32_t MAX_WORKER_CORES = 4;
    uint32_t effective_num_links = std::min(num_links, MAX_WORKER_CORES);
    TT_FATAL(
        subdevice_cores.size() >= effective_num_links,
        "Not enough cores {} for {} links",
        subdevice_cores.size(),
        effective_num_links);

    uint32_t num_cores = effective_num_links;
    uint32_t experts_per_core_range = tt::div_up(operation_attributes.experts_per_chip, num_cores);

    // Core layout depends on dispatched_buffer layout:
    //   TILE_LAYOUT: sender placed at the start of its idle group so every idle core sits to the
    //     sender's right and can write leftward on NOC1 (the -X NOC, writer default).
    //     Cores are divided into groups, sender placed at group offset 0:
    //     [sender0, idle0_0..idle0_{k0-1}, sender1, idle1_0..idle1_{k1-1}, ...]
    //   ROW_MAJOR: first num_cores cores are senders, remaining are idle (for zero-init only).
    //     [sender0, sender1, idle0, idle1, idle2, ...]
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

    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(num_cores);
    std::vector<std::vector<CoreCoord>> sender_idle_groups(num_cores);
    std::vector<CoreCoord> all_idle_cores;
    std::vector<uint32_t> idle_sender_map;

    if (is_tile_layout) {
        // TILE_LAYOUT: divide into groups, sender at the start of each group
        uint32_t base_group_size = total_row_cores / num_cores;
        uint32_t extra_groups = total_row_cores % num_cores;

        uint32_t pos = 0;
        for (uint32_t s = 0; s < num_cores; s++) {
            uint32_t group_size = base_group_size + (s >= num_cores - extra_groups ? 1 : 0);
            uint32_t sender_offset = 0;

            for (uint32_t j = 0; j < group_size; j++) {
                if (j == sender_offset) {
                    sender_cores.push_back(all_row_cores[pos]);
                } else {
                    sender_idle_groups[s].push_back(all_row_cores[pos]);
                    all_idle_cores.push_back(all_row_cores[pos]);
                    idle_sender_map.push_back(s);
                }
                pos++;
            }
        }
    } else {
        // ROW_MAJOR: first num_cores cores are senders, all remaining are idle
        for (uint32_t s = 0; s < num_cores; s++) {
            sender_cores.push_back(all_row_cores[s]);
        }
        for (uint32_t i = num_cores; i < total_row_cores; i++) {
            // Distribute idle cores round-robin across senders (for zero-init)
            uint32_t s = (i - num_cores) % num_cores;
            sender_idle_groups[s].push_back(all_row_cores[i]);
            all_idle_cores.push_back(all_row_cores[i]);
            idle_sender_map.push_back(s);
        }
    }

    uint32_t num_idle_cores = static_cast<uint32_t>(all_idle_cores.size());
    TT_FATAL(
        num_idle_cores >= num_cores,
        "Same-row has only {} idle cores for {} senders — need at least one idle core per sender",
        num_idle_cores,
        num_cores);
    uint32_t idle_cores_per_sender = num_idle_cores / num_cores;
    uint32_t senders_with_extra_idle = num_idle_cores % num_cores;

    // Build sender_core_grid from selected sender cores
    std::set<CoreRange> sender_ranges_set;
    for (const auto& sc : sender_cores) {
        sender_ranges_set.insert(CoreRange(sc));
    }
    auto sender_core_grid = CoreRangeSet(sender_ranges_set);
    TT_FATAL(sender_cores.size() == num_cores, "Expected {} sender cores, got {}", num_cores, sender_cores.size());

    log_debug(
        tt::LogOp,
        "Combine program: hidden_size: {} num_cores: {} experts_per_core_range: {} "
        "sender_cores: {} num_idle_cores: {} idle_cores_per_sender: {} senders_with_extra_idle: {}",
        hidden_size,
        num_cores,
        experts_per_core_range,
        sender_cores,
        num_idle_cores,
        idle_cores_per_sender,
        senders_with_extra_idle);

    // ProgramDescriptor semaphores carry an explicit `.id` field — legacy
    // CreateSemaphore() auto-assigned the next available ID per core, so we
    // mirror that by maintaining a manual counter.  All semaphore ranges in
    // this function nest (sender_core_grid ⊆ sender_and_idle_grid ⊆
    // worker_core_range_set), so a single monotonic counter is sufficient to
    // guarantee per-core uniqueness across every allocation site.
    uint32_t next_sema_id = 0;
    uint32_t zero_init_semaphore_id = next_sema_id++;
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = zero_init_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = sender_core_grid,
        .initial_value = 0});
    uint32_t zero_init_barrier_semaphore_id = next_sema_id++;
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = zero_init_barrier_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = sender_core_grid,
        .initial_value = 0});

    const uint32_t read_batch_size = is_tile_layout ? dispatched_buffer.tensor_spec().tile().get_height() : 8;

    // c_1: dispatched_metadata scratch (reader-only, batched DRAM reads)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        dispatched_metadata,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "dispatched_metadata_scratch");

    // c_2: expert_token_counts scratch on sender.
    // Sized one extra page larger than the raw counter data so reader_combine can append
    // its receive_buf_addr (get_write_ptr(c_18)) immediately after the counter pages before the
    // multicast, giving idle cores a host-side-free way to discover the sender's receive buffer.
    // Extra space is one full counter_page_size (not l1_alignment) to keep cb_size divisible by page_size.
    {
        uint32_t counter_pages = detail::get_num_pages(expert_token_counts);
        uint32_t counter_page_size = detail::get_aligned_page_size(expert_token_counts);
        auto data_format = tt::tt_metal::datatype_to_dataformat_converter(expert_token_counts.dtype());
        // One extra page holds the single receive_buf_addr (uint32) appended after counter data.
        uint32_t extra_pages = 1;
        uint32_t cb_size = (counter_pages + extra_pages) * counter_page_size;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = cb_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                .data_format = data_format,
                .page_size = counter_page_size,
            }}},
        });
    }
    // c_8: expert_region_offsets (reader-only, full tensor)
    detail::create_tensor_cb(
        desc,
        sender_core_grid,
        expert_region_offsets,
        /*buffering_factor=*/detail::get_num_pages(expert_region_offsets),
        /*cb_id=*/tt::CBIndex::c_8,
        "expert_region_offsets");

    if (is_tile_layout) {
        // c_18: receive buffer for idle-core untilized data written back via NOC (TILE_LAYOUT only).
        // dtype/page-size inherited from output_tensor: BFLOAT16 in the bf16 path,
        // FP8_E4M3 (auto-resolves to tt::DataFormat::Fp8_e4m3) in the fp8 path.
        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            output_tensor,
            /*buffering_factor=*/read_batch_size,
            /*cb_id=*/tt::CBIndex::c_18,
            "untilized_output");
        // c_10: sender-side NOC scratch that receives each idle core's c_9 L1 address during
        //       the startup handshake (idle core writes its get_write_ptr(c_9) to slot
        //       core_id * sizeof(uint32_t)). Sender copies this scratch into a plain local
        //       uint32_t[num_idle_cores_group] array before the batch loop.
        //       Sender's own c_10 L1 offset is appended to the counter multicast trailer so
        //       idle cores know where to unicast their addresses.
        {
            uint32_t max_idle_group_size = 0;
            for (uint32_t s = 0; s < num_cores; s++) {
                max_idle_group_size = std::max(max_idle_group_size, (uint32_t)sender_idle_groups[s].size());
            }
            uint32_t needed_bytes = std::max(max_idle_group_size * (uint32_t)sizeof(uint32_t), l1_alignment);
            uint32_t idle_c9_scratch_size = ((needed_bytes + l1_alignment - 1) / l1_alignment) * l1_alignment;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = idle_c9_scratch_size,
                .core_ranges = sender_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                    .data_format = tt::DataFormat::UInt8,
                    .page_size = idle_c9_scratch_size,
                }}},
            });
        }
    } else {
        // c_0 on sender cores: dispatched_buffer rows for ROW_MAJOR DMA reads
        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            dispatched_buffer,
            /*buffering_factor=*/read_batch_size,
            /*cb_id=*/tt::CBIndex::c_0,
            "dispatched_buffer_sender");
    }

    // c_3: route_info (reader->writer, 4 x uint32_t per entry)
    {
        constexpr uint32_t rw_buffering = 2;

        uint32_t route_info_page_size = l1_alignment;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = rw_buffering * route_info_page_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = tt::DataFormat::UInt8,
                .page_size = route_info_page_size,
            }}},
        });

        detail::create_tensor_cb(
            desc,
            sender_core_grid,
            output_tensor,  // dispatched_buffer and output_tensor have same page size
            /*buffering_factor=*/rw_buffering,
            /*cb_id=*/tt::CBIndex::c_4,
            "output_for_writer");
    }

    // c_5: packet header CB for fabric sends (writer-only)
    if (num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = packet_header_cb_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                .data_format = tt::DataFormat::UInt8,
                .page_size = packet_header_size_bytes,
            }}},
        });
    }

    // Iterate over every coordinate in the mesh (full coord range derived from the
    // mesh shape) — replaces the legacy `tensor_coords` parameter, which the new
    // descriptor-style entry point no longer threads through.  The fabric defines
    // baked into kernel compile-time args list every device on the mesh, so this
    // must remain a full enumeration.
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_view.shape())) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }

    // Compile-time args shared by reader and writer
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (6)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_dispatched_buffer_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_dispatched_metadata_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_experts_tok_counter_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_route_info_id
        static_cast<uint32_t>(tt::CBIndex::c_4),  // cb_output_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_5),  // cb_packet_header_id

        // Page counts (4)
        detail::get_num_pages(dispatched_buffer),
        detail::get_num_pages(dispatched_metadata),
        detail::get_num_pages(expert_token_counts),
        detail::get_num_pages(output_tensor),

        // Page sizes (4)
        detail::get_page_size(dispatched_buffer),
        detail::get_page_size(dispatched_metadata),
        detail::get_page_size(expert_token_counts),
        detail::get_page_size(output_tensor),

        // Operation parameters (5)
        operation_attributes.dispatch_group_size,
        operation_attributes.experts_per_chip,
        operation_attributes.num_experts_per_tok,
        operation_attributes.seq_len_per_chip,

        // Hidden dimension
        (uint32_t)hidden_size,

        // Aligned page sizes (4)
        detail::get_aligned_page_size(dispatched_buffer),
        detail::get_aligned_page_size(dispatched_metadata),
        detail::get_aligned_page_size(expert_token_counts),
        detail::get_aligned_page_size(output_tensor),

        // Mesh information (5)
        src_mesh_id,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        linearized_mesh_coord,

        // Fabric configuration (4)
        (uint32_t)fabric_max_packet_size,
        l1_alignment,
        static_cast<uint32_t>(num_links),
        static_cast<uint32_t>(topology),

        // Batch configuration (1)
        read_batch_size,
    };

    // Compute and append num_dispatch_groups (index 34, after read_batch_size at 33) from tensor dimensions.
    // This decouples the combine kernel from the assumption that mesh_cols == num_dispatch_groups.
    {
        auto counter_shape = expert_token_counts.tensor_spec().logical_shape();
        uint32_t num_routed_experts = counter_shape[-1];
        TT_FATAL(operation_attributes.experts_per_chip > 0, "experts_per_chip must be > 0");
        TT_FATAL(operation_attributes.dispatch_group_size > 0, "dispatch_group_size must be > 0");
        TT_FATAL(num_routed_experts > 0, "num_routed_experts must be > 0");
        uint32_t computed_ndg =
            num_routed_experts / (operation_attributes.experts_per_chip * operation_attributes.dispatch_group_size);
        TT_FATAL(
            computed_ndg > 0 &&
                computed_ndg * operation_attributes.experts_per_chip * operation_attributes.dispatch_group_size ==
                    num_routed_experts,
            "num_dispatch_groups computation failed: routed_experts={} experts_per_chip={} group_size={}",
            num_routed_experts,
            operation_attributes.experts_per_chip,
            operation_attributes.dispatch_group_size);
        compile_time_args.push_back(computed_ndg);

        log_debug(
            tt::LogOp,
            "Combine: num_routed_experts={} computed num_dispatch_groups={}",
            num_routed_experts,
            computed_ndg);
    }

    // Expert region offsets tensor metadata (indices 34-37): CB id, pages, page sizes
    compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_8));
    compile_time_args.push_back(detail::get_num_pages(expert_region_offsets));
    compile_time_args.push_back(detail::get_page_size(expert_region_offsets));
    compile_time_args.push_back(detail::get_aligned_page_size(expert_region_offsets));

    // Dispatch buffer total per-chip capacity (index 38): used by readers as overflow guard.
    compile_time_args.push_back((uint32_t)max_dispatch_buffer_token_size);

    // Append TensorAccessorArgs for all 5 tensors (starting at index 39)
    tt::tt_metal::TensorAccessorArgs(dispatched_buffer.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatched_metadata.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(expert_token_counts.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(expert_region_offsets.buffer()).append_to(compile_time_args);

    // Both reader and writer get fabric defines so the reader can compute routes
    std::map<std::string, std::string> fabric_defines;
    if (num_links > 0) {
        fabric_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        fabric_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        fabric_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }
    if (operation_attributes.axis.has_value()) {
        fabric_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    std::map<std::string, std::string> reader_defines = fabric_defines;
    reader_defines["INIT_ZEROS"] = operation_attributes.init_zeros ? "1" : "0";
    reader_defines["IS_TILE_LAYOUT"] = is_tile_layout ? "1" : "0";

    const bool init_zeros = operation_attributes.init_zeros;
    tt::tt_metal::KernelHandle zero_init_kernel_id = 0;
    std::vector<CoreCoord> zero_init_cores_vec;
    uint32_t zi_done_semaphore_id = 0;
    uint32_t pages_per_core = 0;
    uint32_t remainder_pages = 0;

    // idle_row_cores: all same-row idle cores, ordered by sender group then by x.
    std::vector<CoreCoord>& idle_row_cores = all_idle_cores;

    // Per-sender multicast bounding boxes and idle NOC coordinates (TILE_LAYOUT only).
    // Each sender multicasts only to its own dedicated idle group so both senders
    // can run their multicast in parallel without interfering.
    struct SenderMcastCfg {
        uint32_t mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y;
        std::vector<std::pair<uint32_t, uint32_t>> idle_noc_coords;
    };
    std::vector<SenderMcastCfg> sender_mcast_cfgs(num_cores);
    if (is_tile_layout) {
        // Compute per-sender NOC multicast bounding box (min/max x,y over idle cores)
        // and collect individual idle core NOC coordinates for semaphore signaling.
        for (uint32_t s = 0; s < num_cores; s++) {
            TT_FATAL(!sender_idle_groups[s].empty(), "Sender {} has no idle cores assigned", s);
            auto& cfg = sender_mcast_cfgs[s];

            // Initialize bounding box from the first idle core in the group
            auto first_noc =
                mesh_device->virtual_core_from_logical_core(sender_idle_groups[s][0], tt::CoreType::WORKER);
            cfg.mcast_start_x = cfg.mcast_end_x = first_noc.x;
            cfg.mcast_start_y = cfg.mcast_end_y = first_noc.y;

            // Expand bounding box to cover all idle cores and record each core's NOC coords
            for (const auto& ic : sender_idle_groups[s]) {
                auto noc = mesh_device->virtual_core_from_logical_core(ic, tt::CoreType::WORKER);
                cfg.mcast_start_x = std::min(cfg.mcast_start_x, (uint32_t)noc.x);
                cfg.mcast_end_x = std::max(cfg.mcast_end_x, (uint32_t)noc.x);
                cfg.mcast_start_y = std::min(cfg.mcast_start_y, (uint32_t)noc.y);
                cfg.mcast_end_y = std::max(cfg.mcast_end_y, (uint32_t)noc.y);
                cfg.idle_noc_coords.emplace_back((uint32_t)noc.x, (uint32_t)noc.y);
            }
        }
    }

    // Build idle CoreRangeSet
    std::set<CoreRange> idle_ranges_set;
    for (const auto& core : idle_row_cores) {
        idle_ranges_set.insert(CoreRange(core));
    }
    CoreRangeSet idle_core_grid(idle_ranges_set);

    // TILE_LAYOUT semaphores for sender <-> idle core handshake
    uint32_t counter_ready_semaphore_id = 0;
    std::vector<uint32_t> data_ready_semaphore_ids, start_semaphore_ids;
    if (is_tile_layout) {
        // counter_ready semaphore: created only on sender + idle cores so get_semaphore() returns
        // the same L1 offset on both sides (sender writes to idle's copy, idle waits on its own)
        std::set<CoreRange> sender_and_idle_ranges;
        for (const auto& cr : sender_core_grid.ranges()) {
            sender_and_idle_ranges.insert(cr);
        }
        for (const auto& cr : idle_core_grid.ranges()) {
            sender_and_idle_ranges.insert(cr);
        }
        CoreRangeSet sender_and_idle_grid(sender_and_idle_ranges);
        counter_ready_semaphore_id = next_sema_id++;
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = counter_ready_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = sender_and_idle_grid,
            .initial_value = 0});
        // One data_ready and one start semaphore per sender so each sender's handshake with idle
        // cores is independent — idle core routes to the correct pair via sender_idx = expert / experts_per_sender.
        for (uint32_t s = 0; s < num_cores; s++) {
            uint32_t data_ready_id = next_sema_id++;
            desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = data_ready_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = sender_and_idle_grid,
                .initial_value = 0});
            data_ready_semaphore_ids.push_back(data_ready_id);
            uint32_t start_id = next_sema_id++;
            desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = start_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = sender_and_idle_grid,
                .initial_value = 0});
            start_semaphore_ids.push_back(start_id);
        }
    }

    // cb_factor reduces CB size to fit L1: Wormhole needs 4x reduction
    uint32_t cb_factor;
    {
        const auto arch = mesh_device->arch();
        if (arch == tt::ARCH::BLACKHOLE || !is_tile_layout) {
            cb_factor = 1;
        } else {
            cb_factor = 4;  // Wormhole_B0 and others for TILE_LAYOUT
        }
    }

    if (is_tile_layout) {
        // c_1 on idle cores: receives the expert_token_counts multicast from the owning sender.
        // MUST be allocated BEFORE c_0 so its L1 address matches the sender's c_1 address.
        {
            uint32_t counter_pages = detail::get_num_pages(expert_token_counts);
            uint32_t counter_page_size = detail::get_aligned_page_size(expert_token_counts);
            auto data_format = tt::tt_metal::datatype_to_dataformat_converter(expert_token_counts.dtype());
            uint32_t extra_pages = 1;
            uint32_t cb_size = (counter_pages + extra_pages) * counter_page_size;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = cb_size,
                .core_ranges = idle_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
                    .data_format = data_format,
                    .page_size = counter_page_size,
                }}},
            });
        }
        // c_0 on idle cores: dispatched_buffer tiles
        detail::create_tensor_cb(
            desc,
            idle_core_grid,
            dispatched_buffer,
            /*buffering_factor=*/hidden_size / (32 * cb_factor),
            /*cb_id=*/tt::CBIndex::c_0,
            "dispatched_buffer_idle");
        // c_2 on idle cores: untilized output rows, one full batch (read_batch_size rows).
        // dtype/page-size inherited from output_tensor: BFLOAT16 in the bf16 path,
        // FP8_E4M3 (auto-resolves to tt::DataFormat::Fp8_e4m3) in the fp8 path. The packer
        // selects the correct pack format based on the destination CB's DataFormat.
        detail::create_tensor_cb(
            desc,
            idle_core_grid,
            output_tensor,
            /*buffering_factor=*/read_batch_size,
            /*cb_id=*/tt::CBIndex::c_2,
            "untilize_idle");
        // c_3 on idle cores: 1-page signal CB (reader->compute, mirrors c_17 on sender cores)
        {
            uint32_t signal_page_size = l1_alignment;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = signal_page_size,
                .core_ranges = idle_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                    .data_format = tt::DataFormat::UInt8,
                    .page_size = signal_page_size,
                }}},
            });
        }
        // c_8 on idle cores: 1-page stop-signal CB (compute -> zero_init_writer).
        // Compute pushes 0 per batch (meaning "a batch is ready on c_2") and ROUTE_INFO_SENTINEL
        // when it exits its own loop, so zero_init_writer knows when to stop its send loop.
        {
            uint32_t stop_signal_page_size = l1_alignment;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = stop_signal_page_size,
                .core_ranges = idle_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                    .data_format = tt::DataFormat::UInt8,
                    .page_size = stop_signal_page_size,
                }}},
            });
        }
        // c_9 on idle cores: metadata-batch CB. The owning sender unicasts one read_batch_size
        // worth of metadata here per iteration. Idle cores report this CB's L1 address to their
        // owning sender once at startup (via the idle_c9_addr handshake below) so the sender
        // knows where to unicast. First uint32 of each unicast is a sender-written sentinel:
        // 0xFFFFFFFF → non-local writes exist → idle sends its untilized data back to sender.
        // Any other value → all-local → idle does the local writes itself using this metadata.
        {
            uint32_t metadata_batch_page_size = detail::get_aligned_page_size(dispatched_metadata);
            auto metadata_fmt = tt::tt_metal::datatype_to_dataformat_converter(dispatched_metadata.dtype());
            uint32_t metadata_batch_cb_size = read_batch_size * metadata_batch_page_size;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = metadata_batch_cb_size,
                .core_ranges = idle_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                    .data_format = metadata_fmt,
                    .page_size = metadata_batch_page_size,
                }}},
            });
        }
    }

    // counter_offset mirrors the constexpr calculation in reader_combine.cpp
    uint32_t mesh_row_coord = linearized_mesh_coord / mesh_cols;
    uint32_t mesh_col_coord = linearized_mesh_coord % mesh_cols;
    uint32_t experts_per_dispatch_group = operation_attributes.experts_per_chip * mesh_rows;
    uint32_t counter_offset =
        mesh_col_coord * experts_per_dispatch_group + mesh_row_coord * operation_attributes.experts_per_chip;

    // reader_untilize + compute kernels only needed for TILE_LAYOUT
    std::vector<tt::tt_metal::KernelHandle> reader_untilize_kernel_ids;
    if (is_tile_layout) {
        // Compile-time args layout for reader_untilize (matching reader_untilize.cpp):
        //   0-13: shared base (below, includes max_dispatch_buffer_token_size at 13)
        //   14:   core_id   — local index within sender s's idle group (0..k_s-1)
        //   15:   num_idle_cores — per-sender count k_s (for round-robin batch assignment)
        //   16:   aligned_output_page_size
        //   17:   aligned_experts_tok_counter_page_size
        //   18+:  TensorAccessorArgs for dispatched_buffer (no num_senders — single-sender kernel)
        const uint32_t tile_height = dispatched_buffer.tensor_spec().tile().get_height();
        const uint32_t tile_width = dispatched_buffer.tensor_spec().tile().get_width();
        const std::vector<uint32_t> reader_untilize_compile_time_args_base = {
            static_cast<uint32_t>(tt::CBIndex::c_1),           // 0:  cb_experts_tok_counter_id
            detail::get_num_pages(expert_token_counts),        // 1:  experts_tok_counter_pages
            operation_attributes.experts_per_chip,             // 2:  experts_per_chip
            counter_offset,                                    // 3:  counter_offset
            static_cast<uint32_t>(tt::CBIndex::c_0),           // 4:  cb_dispatched_buffer_id
            static_cast<uint32_t>(tt::CBIndex::c_2),           // 5:  cb_untilize_id
            (uint32_t)hidden_size,                             // 6:  hidden_size
            read_batch_size,                                   // 7:  read_batch_size
            static_cast<uint32_t>(tt::CBIndex::c_3),           // 8:  cb_signal_id
            detail::get_aligned_page_size(dispatched_buffer),  // 9:  aligned_dispatched_buffer_page_size
            cb_factor,                                         // 10: cb_factor
            tile_height,                                       // 11: tile_height
            tile_width,                                        // 12: tile_width
            (uint32_t)max_dispatch_buffer_token_size,          // 13: max_dispatch_buffer_token_size
        };

        // Partitioned idle cores: each sender s owns a dedicated group of k_s idle cores.
        // core_id is LOCAL within the sender's group (0..k_s-1) for round-robin batch assignment.
        // num_idle_cores baked in as k_s so the kernel only considers its own group.
        // No num_senders arg — each kernel is bound to a single sender.
        reader_untilize_kernel_ids.reserve(num_idle_cores);

        uint32_t global_idle_idx = 0;
        for (uint32_t s = 0; s < num_cores; s++) {
            uint32_t k_s = static_cast<uint32_t>(sender_idle_groups[s].size());
            for (uint32_t j = 0; j < k_s; j++, global_idle_idx++) {
                auto per_core_args = reader_untilize_compile_time_args_base;
                per_core_args.push_back(j);    // 14: core_id (local to sender s's group)
                per_core_args.push_back(k_s);  // 15: num_idle_cores (per-sender)
                per_core_args.push_back(detail::get_aligned_page_size(output_tensor));        // 16
                per_core_args.push_back(detail::get_aligned_page_size(expert_token_counts));  // 17
                // 18+: TensorAccessorArgs (no num_senders — single-sender kernel)
                tt::tt_metal::TensorAccessorArgs(dispatched_buffer.buffer()).append_to(per_core_args);

                CoreRangeSet single_idle_core({CoreRange(idle_row_cores[global_idle_idx])});
                tt::tt_metal::KernelDescriptor reader_untilize_kd;
                reader_untilize_kd.kernel_source =
                    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/"
                    "reader_untilize.cpp";
                reader_untilize_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
                reader_untilize_kd.core_ranges = single_idle_core;
                reader_untilize_kd.compile_time_args = std::move(per_core_args);
                reader_untilize_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
                };
                reader_untilize_kernel_ids.push_back(static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size()));
                desc.kernels.push_back(std::move(reader_untilize_kd));
            }
        }
    }

    std::map<std::string, std::string> writer_defines = fabric_defines;
    writer_defines["INIT_ZEROS"] = operation_attributes.init_zeros ? "1" : "0";

    // zero_init_writer kernel is launched whenever either:
    //   (a) init_zeros=True — it does the per-bank zero-init of the output tensor
    //   (b) is_tile_layout — it runs the post-zero-init untilized-data send loop
    //                        (consumes cb_untilize_id, writes back to the sender's c_18).
    // With init_zeros=False on TILE_LAYOUT only (b) applies, so the zero-init CBs/semaphore
    // are skipped and the kernel is compiled with INIT_ZEROS=0.
    const bool create_zi_kernel = init_zeros || is_tile_layout;

    if (init_zeros) {
        uint32_t noc_max_burst_size;
        const auto arch = mesh_device->arch();
        if (arch == tt::ARCH::BLACKHOLE) {
            noc_max_burst_size = 16384;
        } else if (arch == tt::ARCH::WORMHOLE_B0) {
            noc_max_burst_size = 8192;
        } else {
            TT_THROW("Unsupported architecture for zero-init: {}", arch);
        }

        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = noc_max_burst_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_7),
                .data_format = tt::DataFormat::UInt8,
                .page_size = noc_max_burst_size,
            }}},
        });

        uint32_t total_zero_init_cores = num_cores + num_idle_cores;
        uint32_t total_output_pages = detail::get_num_pages(output_tensor);
        pages_per_core = total_output_pages / total_zero_init_cores;
        remainder_pages = total_output_pages % total_zero_init_cores;

        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = noc_max_burst_size,
            .core_ranges = idle_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
                .data_format = tt::DataFormat::UInt8,
                .page_size = noc_max_burst_size,
            }}},
        });

        zi_done_semaphore_id = next_sema_id++;
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = zi_done_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = worker_core_range_set,
            .initial_value = 0});
    }

    if (create_zi_kernel) {
        uint32_t output_aligned_page_size = detail::get_aligned_page_size(output_tensor);
        std::vector<uint32_t> zi_compile_time_args = {
            output_aligned_page_size,
            // num_sender_cores and cb_zero_buffer_id are only referenced inside the
            // INIT_ZEROS-gated zero-init phase in the kernel; pass 0 when init_zeros=False
            // (the c_6 CB is not created in that case so its index is meaningless).
            init_zeros ? num_cores : 0u,
            init_zeros ? static_cast<uint32_t>(tt::CBIndex::c_6) : 0u,
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(zi_compile_time_args);

        // Tile-layout-only compile-time args used by the post-zero-init untilized-data send loop.
        // In ROW_MAJOR the corresponding #if IS_TILE_LAYOUT block is compiled out, so these
        // trailing args are ignored — still pushed unconditionally to keep the kernel object stable.
        zi_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_2));     // cb_untilize_id
        zi_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_8));     // cb_stop_signal_id
        zi_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_1));     // cb_experts_tok_counter_id
        zi_compile_time_args.push_back(detail::get_num_pages(expert_token_counts));  // experts_tok_counter_pages
        zi_compile_time_args.push_back(detail::get_aligned_page_size(expert_token_counts));  // counter page size
        zi_compile_time_args.push_back(read_batch_size);
        zi_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_9));  // cb_metadata_batch_id
        zi_compile_time_args.push_back(operation_attributes.num_experts_per_tok);  // num_experts_per_tok
        zi_compile_time_args.push_back(
            detail::get_aligned_page_size(dispatched_metadata));  // aligned_dispatched_metadata_page_size

        std::map<std::string, std::string> zi_defines;
        zi_defines["IS_TILE_LAYOUT"] = is_tile_layout ? "1" : "0";
        zi_defines["INIT_ZEROS"] = init_zeros ? "1" : "0";

        tt::tt_metal::KernelDescriptor zero_init_kd;
        zero_init_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/"
            "zero_init_writer.cpp";
        zero_init_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        zero_init_kd.core_ranges = idle_core_grid;
        zero_init_kd.compile_time_args = std::move(zi_compile_time_args);
        zero_init_kd.defines = {zi_defines.begin(), zi_defines.end()};
        zero_init_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
        };
        zero_init_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(zero_init_kd));

        zero_init_cores_vec = idle_row_cores;
    }

    // Reader compile-time args base (without num_idle_cores — that is per-sender and appended below).
    std::vector<uint32_t> reader_compile_time_args_base = compile_time_args;
    if (init_zeros) {
        reader_compile_time_args_base.push_back(static_cast<uint32_t>(tt::CBIndex::c_7));  // zi_cb_id
        reader_compile_time_args_base.push_back(num_idle_cores);  // num_total_idle_cores (both layouts need this)
    }
    // num_idle_cores (per-sender k_s) and cb_untilize_id are appended per-sender below (TILE_LAYOUT only).

    // One reader_combine kernel per sender.  For TILE_LAYOUT, k_s (per-sender idle count)
    // is baked in as num_idle_cores so the sender only round-robins across its own dedicated
    // idle cores.  For ROW_MAJOR, no idle-core args are needed.
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    reader_kernel_ids.reserve(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        auto per_sender_compile_args = reader_compile_time_args_base;
        if (is_tile_layout) {
            uint32_t k_s = static_cast<uint32_t>(sender_idle_groups[s].size());
            per_sender_compile_args.push_back(k_s);                                       // num_idle_cores (per-sender)
            per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_18));  // cb_untilize_id
            per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_9));   // cb_metadata_batch_id
            per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_10));  // cb_idle_c9_addr_scratch_id
        }
        CoreRangeSet single_sender_core({CoreRange(sender_cores[s])});
        tt::tt_metal::KernelDescriptor reader_kd;
        reader_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp";
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

    tt::tt_metal::KernelDescriptor writer_kd;
    writer_kd.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp";
    writer_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kd.core_ranges = sender_core_grid;
    writer_kd.compile_time_args = compile_time_args;
    writer_kd.defines = {writer_defines.begin(), writer_defines.end()};
    writer_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
    };
    tt::tt_metal::KernelHandle writer_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_kd));

    // Compute kernel on idle cores that untilizes dispatched_buffer data (TILE_LAYOUT only)
    if (is_tile_layout) {
        const uint32_t full_ct_dim = static_cast<uint32_t>(hidden_size) / 32u;
        uint32_t block_ct_dim = 8;
        while (full_ct_dim % block_ct_dim != 0) {
            --block_ct_dim;
        }

        tt::tt_metal::KernelDescriptor compute_kd;
        compute_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/compute/"
            "untilize_combine.cpp";
        compute_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        compute_kd.core_ranges = idle_core_grid;
        compute_kd.compile_time_args = {
            static_cast<uint32_t>(tt::CBIndex::c_3),  // 0: cb_signal_id (CB used for signaling on the same core that
                                                      //    data is loaded and ready to be untilized)
            static_cast<uint32_t>(tt::CBIndex::c_2),  // 1: cb_untilize_id (untilized dispatched_buffer data)
            static_cast<uint32_t>(tt::CBIndex::c_0),  // 2: cb_in_id (dispatched_buffer data)
            read_batch_size,                          // 3: read_batch_size
            full_ct_dim,                              // 4: full_ct_dim = hidden_size / 32
            block_ct_dim,                             // 5: block_ct_dim = largest divisor of full_ct_dim <= 8
            static_cast<uint32_t>(
                tt::CBIndex::c_8),  // 6: cb_stop_signal_id (compute -> zero_init_writer per-batch/stop)
        };
        compute_kd.config = tt::tt_metal::ComputeConfigDescriptor{};
        desc.kernels.push_back(std::move(compute_kd));
    }

    // Pre-compute NOC coordinates for all sender cores (for inter-core barrier signaling)
    std::vector<std::pair<uint32_t, uint32_t>> sender_noc_coords;
    for (const auto& sc : sender_cores) {
        auto noc_coord = mesh_device->virtual_core_from_logical_core(sc, tt::CoreType::WORKER);
        sender_noc_coords.emplace_back(noc_coord.x, noc_coord.y);
    }

    // Set runtime args for hybrid idle row cores.  Three layouts are possible:
    //   init_zeros && tile_layout: [output_addr, page_start, page_end, zi_done_sem,
    //                               (sender_noc_x, sender_noc_y) * num_cores,
    //                               counter_ready_sem, owning_sender_noc_x, owning_sender_noc_y,
    //                               data_ready_sem, start_sem, local_core_id]
    //   init_zeros && row_major:   [output_addr, page_start, page_end, zi_done_sem,
    //                               (sender_noc_x, sender_noc_y) * num_cores]
    //   !init_zeros && tile_layout:[output_addr, counter_ready_sem, owning_sender_noc_x,
    //                               owning_sender_noc_y, data_ready_sem, start_sem, local_core_id]
    // The kernel guards the zero-init reads with #if INIT_ZEROS so the indices match.
    if (create_zi_kernel) {
        for (uint32_t idle_idx = 0; idle_idx < num_idle_cores; idle_idx++) {
            // Push output_tensor's buffer first as Buffer* so the framework records
            // a BufferBinding for the cache-hit fast path.
            tt::tt_metal::KernelDescriptor::RTArgList zi_runtime_args;
            zi_runtime_args.push_back(output_tensor.buffer());

            if (init_zeros) {
                uint32_t row_idx = num_cores + idle_idx;
                uint32_t page_start = (row_idx * pages_per_core) + std::min(row_idx, remainder_pages);
                uint32_t page_end = page_start + pages_per_core + (row_idx < remainder_pages ? 1 : 0);
                zi_runtime_args.push_back(page_start);
                zi_runtime_args.push_back(page_end);
                zi_runtime_args.push_back(zi_done_semaphore_id);
                // Each idle core signals all sender cores once its zero-init slice is done.
                for (const auto& [noc_x, noc_y] : sender_noc_coords) {
                    zi_runtime_args.push_back(noc_x);
                    zi_runtime_args.push_back(noc_y);
                }
            }

            if (is_tile_layout) {
                uint32_t s = idle_sender_map[idle_idx];
                // core_id = this idle core's local index within sender s's group (0..k_s-1).
                uint32_t local_core_id = 0;
                for (uint32_t j = 0; j < idle_idx; j++) {
                    if (idle_sender_map[j] == s) {
                        local_core_id++;
                    }
                }
                zi_runtime_args.push_back(counter_ready_semaphore_id);
                zi_runtime_args.push_back(sender_noc_coords[s].first);
                zi_runtime_args.push_back(sender_noc_coords[s].second);
                zi_runtime_args.push_back(data_ready_semaphore_ids[s]);
                zi_runtime_args.push_back(start_semaphore_ids[s]);
                zi_runtime_args.push_back(local_core_id);
            }

            desc.kernels[zero_init_kernel_id].emplace_runtime_args(zero_init_cores_vec[idle_idx], zi_runtime_args);
        }
    }

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        uint32_t expert_start = core_idx * experts_per_core_range;
        uint32_t expert_end = std::min((core_idx + 1) * experts_per_core_range, operation_attributes.experts_per_chip);

        // Reader RT args.  Tensor buffer addresses are pushed as Buffer* so the
        // framework records BufferBindings for the O(1) cache-hit fast path.
        tt::tt_metal::KernelDescriptor::RTArgList reader_runtime_args;
        reader_runtime_args.push_back(dispatched_buffer.buffer());
        reader_runtime_args.push_back(dispatched_metadata.buffer());
        reader_runtime_args.push_back(expert_token_counts.buffer());
        reader_runtime_args.push_back(expert_region_offsets.buffer());
        reader_runtime_args.push_back(output_tensor.buffer());
        reader_runtime_args.push_back(zero_init_semaphore_id);
        reader_runtime_args.push_back(zero_init_barrier_semaphore_id);
        reader_runtime_args.push_back(num_cores);
        reader_runtime_args.push_back(expert_start);
        reader_runtime_args.push_back(expert_end);
        if (init_zeros) {
            uint32_t sender_page_start = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);
            uint32_t sender_page_end = sender_page_start + pages_per_core + (core_idx < remainder_pages ? 1 : 0);
            reader_runtime_args.push_back(sender_page_start);
            reader_runtime_args.push_back(sender_page_end);
            reader_runtime_args.push_back(zi_done_semaphore_id);
        }
        if (is_tile_layout) {
            // Multicast targets only this sender's dedicated idle group
            const auto& mcast_cfg = sender_mcast_cfgs[core_idx];
            reader_runtime_args.push_back(counter_ready_semaphore_id);
            reader_runtime_args.push_back(mcast_cfg.mcast_start_x);
            reader_runtime_args.push_back(mcast_cfg.mcast_start_y);
            reader_runtime_args.push_back(mcast_cfg.mcast_end_x);
            reader_runtime_args.push_back(mcast_cfg.mcast_end_y);
            reader_runtime_args.push_back(data_ready_semaphore_ids[core_idx]);
            reader_runtime_args.push_back(start_semaphore_ids[core_idx]);
            // Pass NOC coords of only this sender's k_s dedicated idle cores
            for (const auto& [noc_x, noc_y] : mcast_cfg.idle_noc_coords) {
                reader_runtime_args.push_back(noc_x);
                reader_runtime_args.push_back(noc_y);
            }
        }

        // Writer RT args: build into a plain std::vector<uint32_t> first because
        // append_fabric_connection_rt_args appends raw uint32_t values to a
        // std::vector — then promote to the RTArgList builder (replacing the
        // buffer-address slots with Buffer* entries) before emplace.
        std::vector<uint32_t> writer_runtime_args_raw = {
            dispatched_buffer.buffer()->address(),
            dispatched_metadata.buffer()->address(),
            expert_token_counts.buffer()->address(),
            expert_region_offsets.buffer()->address(),
            output_tensor.buffer()->address(),
            zero_init_semaphore_id,
            (uint32_t)init_semaphore.address(),
            (uint32_t)exit_semaphore.address(),
            zero_init_barrier_semaphore_id,
            num_cores,
            expert_start,
            expert_end,
        };

        // Append NOC coordinates of all cores for inter-core barrier signaling
        for (const auto& [noc_x, noc_y] : sender_noc_coords) {
            writer_runtime_args_raw.push_back(noc_x);
            writer_runtime_args_raw.push_back(noc_y);
        }

        if (num_links > 0) {
            uint32_t core_link = core_idx % num_links;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }

                log_debug(
                    tt::LogOp,
                    "Combine connection: ({}, {}) -> ({}, {}) core {} link {} experts [{}, {})",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    core_link,
                    expert_start,
                    expert_end);

                // ProgramDescriptor specialization: appends fabric-routing args
                // onto writer_runtime_args_raw and patches desc-side bookkeeping.
                tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    desc,
                    sender_core,
                    writer_runtime_args_raw);
            }
        }

        // Promote the writer RT args to the kernel-descriptor builder, replacing
        // the first five positions with Buffer* entries so the framework records
        // BufferBindings for the cache-hit fast path.  All other positions
        // (semaphore IDs, NOC coords, fabric-appended trailers) remain plain
        // uint32_t.
        tt::tt_metal::KernelDescriptor::RTArgList writer_runtime_args;
        writer_runtime_args.reserve(writer_runtime_args_raw.size());
        writer_runtime_args.push_back(dispatched_buffer.buffer());
        writer_runtime_args.push_back(dispatched_metadata.buffer());
        writer_runtime_args.push_back(expert_token_counts.buffer());
        writer_runtime_args.push_back(expert_region_offsets.buffer());
        writer_runtime_args.push_back(output_tensor.buffer());
        for (size_t i = 5; i < writer_runtime_args_raw.size(); ++i) {
            writer_runtime_args.push_back(writer_runtime_args_raw[i]);
        }
        desc.kernels[writer_kernel_id].emplace_runtime_args(sender_core, writer_runtime_args);

        desc.kernels[reader_kernel_ids[core_idx]].emplace_runtime_args(sender_core, reader_runtime_args);
        core_idx++;
    }

    // Set runtime args for idle cores (TILE_LAYOUT only — reader_untilize kernel).
    // Layout: counter_ready_sem, dispatched_buffer_addr, expert_start, expert_end.
    // Sender NOC coords and per-sender data_ready/start semaphores are now consumed by
    // zero_init_writer on the same core (which owns the untilized-data send).
    if (is_tile_layout) {
        for (uint32_t j = 0; j < num_idle_cores; j++) {
            uint32_t s = idle_sender_map[j];
            uint32_t expert_start = s * experts_per_core_range;
            uint32_t expert_end = std::min((s + 1) * experts_per_core_range, operation_attributes.experts_per_chip);
            tt::tt_metal::KernelDescriptor::RTArgList idle_rt_args;
            idle_rt_args.push_back(counter_ready_semaphore_id);
            // dispatched_buffer is pushed as Buffer* so the framework records a
            // BufferBinding for this RT slot (RT layout: [0:counter_ready_sem,
            // 1:dispatched_buffer_addr, 2:expert_start, 3:expert_end]).
            idle_rt_args.push_back(dispatched_buffer.buffer());
            idle_rt_args.push_back(expert_start);
            idle_rt_args.push_back(expert_end);
            desc.kernels[reader_untilize_kernel_ids[j]].emplace_runtime_args(idle_row_cores[j], idle_rt_args);
        }
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor CombineProgramFactory::create_workload_descriptor(
    const CombineParams& operation_attributes,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto* mesh_device = tensor_args.dispatched_buffer.device();

    // Allocate the two cross-device GlobalSemaphores once per workload (cache miss).
    // They live on WorkloadDescriptor.semaphores so the device-side allocations
    // outlive the cached MeshWorkload via the program cache — writer runtime args
    // reference them as absolute addresses.
    auto sem_buffer_type = operation_attributes.use_l1_small_for_semaphores ? tt::tt_metal::BufferType::L1_SMALL
                                                                            : tt::tt_metal::BufferType::L1;
    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    auto exit_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, operation_attributes.worker_core_range_set, 0, sem_buffer_type);
    // Cross-device barrier: ensure every device's GlobalSemaphores have been allocated
    // before any kernel reads them.  Mirrors the previous prepare_resources hook.
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(init_barrier_semaphore);
    workload_descriptor.semaphores.push_back(exit_barrier_semaphore);

    // Combine is mesh-coord-dependent (fabric routing + linearized counter offset
    // are baked into kernel compile-time args), so we cannot replicate one
    // ProgramDescriptor across the whole mesh — every coord gets its own build.
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_program_for_coord(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            coord,
            init_barrier_semaphore,
            exit_barrier_semaphore);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
