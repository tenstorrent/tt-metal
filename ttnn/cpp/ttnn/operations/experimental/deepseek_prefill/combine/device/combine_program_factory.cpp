// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_device_operation.hpp"
#include "combine_connectivity.hpp"  // [debug] host-side topology capture
#include <algorithm>
#include <array>
#include <bitset>
#include <map>
#include <utility>
#include <limits>
#include <tt-metalium/constants.hpp>
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
#include "ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/relay_config.hpp"

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

    // FABRIC_2D uses the portable RoutingPlaneConnectionManager (per-destination connection +
    // multicast handshake) for multi-hop combine-axis forwarding; FABRIC_1D keeps the legacy
    // per-direction array connection. Must match the writer kernel's #ifdef FABRIC_2D gating.
    const bool is_2d_fabric = tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());

    auto dispatched_shape = dispatched_buffer.logical_shape();
    auto hidden_size = dispatched_shape[-1];
    auto max_dispatch_buffer_token_size = dispatched_shape[-2];

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    // Per-untilizer ring depth on the sender's receive_buf — also the initial value of the credits
    // semaphore handed to each untilizer core (TILE_LAYOUT only).  Single source of truth: passed
    // to the reader_combine and writer_untilize kernels as a compile-time arg (they no longer
    // hardcode it).  Must remain a power of 2 (reader_combine masks with SLOTS_PER_UNTILIZER - 1).
    constexpr uint32_t SLOTS_PER_UNTILIZER = 16;
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
    //   TILE_LAYOUT: sender placed at the start of its untilizer group so every untilizer core sits to the
    //     sender's right and can write leftward on NOC1 (the -X NOC, writer default).
    //     Cores are divided into groups, sender placed at group offset 0:
    //     [sender0, untilizer0_0..untilizer0_{k0-1}, sender1, untilizer1_0..untilizer1_{k1-1}, ...]
    //   ROW_MAJOR: first num_cores cores are senders, remaining are untilizer (for output-zeroing only).
    //     [sender0, sender1, untilizer0, untilizer1, untilizer2, ...]
    // Lay the sender/untilizer groups along a 1-D line of worker cores taken from the (sub)device
    // grid.  The subdevice is guaranteed to be a single line — first/last row OR first/last column —
    // and the orientation is inferred here from the grid shape (no extra API flag is needed):
    //   - single-column grid (all cores share x): use the whole column, ordered by y.
    //   - single-row grid, or the default full 2-D worker grid (subdevice_id == None): use the first
    //     row (smallest y), ordered by x — byte-identical to the previous behavior.
    // Everything downstream is geometry-agnostic: it treats `all_row_cores` as an ordered line and
    // addresses cores by absolute NOC coordinates, so a column needs no further changes.
    uint32_t min_x = subdevice_cores.at(0).x, max_x = subdevice_cores.at(0).x;
    uint32_t min_y = subdevice_cores.at(0).y, max_y = subdevice_cores.at(0).y;
    for (const auto& core : subdevice_cores) {
        min_x = std::min(min_x, (uint32_t)core.x);
        max_x = std::max(max_x, (uint32_t)core.x);
        min_y = std::min(min_y, (uint32_t)core.y);
        max_y = std::max(max_y, (uint32_t)core.y);
    }
    // The sender/untilizer pipeline is laid out along a single line of cores, so the worker
    // subdevice must be a single row (first/last row) or single column (first/last column).
    // A subdevice spanning both multiple rows and multiple columns has no defined line layout.
    // The legacy no-subdevice path passes the full device worker grid (also 2-D) and is handled
    // by taking its first row, so that case is exempted.
    if ((max_x > min_x) && (max_y > min_y)) {
        auto compute_grid = mesh_device->compute_with_storage_grid_size();
        const bool is_full_worker_grid = min_x == 0 && min_y == 0 && max_x == compute_grid.x - 1 &&
                                         max_y == compute_grid.y - 1 &&
                                         subdevice_cores.size() == compute_grid.x * compute_grid.y;
        TT_FATAL(
            is_full_worker_grid,
            "Combine requires the worker subdevice to be a single row (first/last row) or single column "
            "(first/last column); a 2-D subdevice core grid spanning x=[{}, {}] y=[{}, {}] is not supported.",
            min_x,
            max_x,
            min_y,
            max_y);
    }
    const bool is_single_column = (min_x == max_x) && (max_y > min_y);

    std::vector<CoreCoord> all_row_cores;
    if (is_single_column) {
        // First/last-column subdevice: every core shares x, so order the line by y.
        all_row_cores = subdevice_cores;
        std::sort(all_row_cores.begin(), all_row_cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
            return a.y < b.y;
        });
    } else {
        // First/last-row subdevice or full 2-D grid: take the first row, ordered by x (legacy path).
        for (const auto& core : subdevice_cores) {
            if (core.y == min_y) {
                all_row_cores.push_back(core);
            }
        }
        std::sort(all_row_cores.begin(), all_row_cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
            return a.x < b.x;
        });
    }

    uint32_t total_row_cores = static_cast<uint32_t>(all_row_cores.size());
    TT_FATAL(
        total_row_cores > num_cores,
        "Worker line has only {} cores for {} senders — need at least one untilizer core per sender",
        total_row_cores,
        num_cores);

    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(num_cores);
    std::vector<std::vector<CoreCoord>> sender_untilizer_groups(num_cores);
    std::vector<CoreCoord> all_untilizer_cores;
    std::vector<uint32_t> untilizer_sender_map;
    std::vector<CoreCoord> relay_cores;  // USE_RELAY: one relay ("R") core per sender battery, left of the sender

    // Both TILE_LAYOUT and ROW_MAJOR route dispatched_buffer through the untilizer pipeline, so
    // both use the same layout: divide the line into per-sender groups with the sender at the
    // start of each group (every untilizer sits to the sender's right).  In ROW_MAJOR the untilizer
    // reader page-copies rows into c_2 instead of untilizing, but the core layout is identical.
    {
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
                    sender_untilizer_groups[s].push_back(all_row_cores[pos]);
                    all_untilizer_cores.push_back(all_row_cores[pos]);
                    untilizer_sender_map.push_back(s);
                }
                pos++;
            }
        }
    }

    // Cap each sender's untilizer group at MAX_UNTILIZERS_PER_SENDER (TILE_LAYOUT only).  Required to
    // stay under the per-core 16-semaphore limit on senders that own one data_ready sem per
    // untilizer (k_s sems on sender) on top of output_init_complete/output_init_barrier/counter_ready/output_init_done
    // + 2 fabric sems for middle chips, totaling 6 + k_s.  Excess untilizers assigned by the
    // initial split above are dropped: their row cores stay in the worker grid but get no
    // untilizer kernels.  k_s[i] = min(k_s[i], MAX_UNTILIZERS_PER_SENDER).
    constexpr uint32_t MAX_UNTILIZERS_PER_SENDER = 4;
    {
        std::vector<CoreCoord> trimmed_all_untilizer_cores;
        std::vector<uint32_t> trimmed_untilizer_sender_map;
        for (uint32_t s = 0; s < num_cores; s++) {
            if (sender_untilizer_groups[s].size() > MAX_UNTILIZERS_PER_SENDER) {
                sender_untilizer_groups[s].resize(MAX_UNTILIZERS_PER_SENDER);
            }
            for (const auto& untilizer : sender_untilizer_groups[s]) {
                trimmed_all_untilizer_cores.push_back(untilizer);
                trimmed_untilizer_sender_map.push_back(s);
            }
        }
        all_untilizer_cores = std::move(trimmed_all_untilizer_cores);
        untilizer_sender_map = std::move(trimmed_untilizer_sender_map);
    }

    uint32_t num_untilizer_cores = static_cast<uint32_t>(all_untilizer_cores.size());
    TT_FATAL(
        num_untilizer_cores >= num_cores,
        "Worker line has only {} untilizer cores for {} senders — need at least one untilizer core per sender",
        num_untilizer_cores,
        num_cores);
    uint32_t untilizer_cores_per_sender = num_untilizer_cores / num_cores;
    uint32_t senders_with_extra_untilizer = num_untilizer_cores % num_cores;

#if USE_RELAY
    // ===================== USE_RELAY — explicit relay-battery placement =====================
    // Discard the default line split above and rebuild placement explicitly so each battery reads
    // R-S-U-U-U-U, with a dedicated relay core prepended to the LEFT of the sender. Coordinates below
    // are PHYSICAL NOC0 (== virtual for unharvested BH tensix); we invert virtual_core_from_logical_core
    // to recover the logical worker cores the rest of the factory addresses. Both batteries sit on the
    // bottom tensix rows (physical y=10, y=11), farthest from the eth cores:
    //   y=10: R@x1, S@x2, U@x3, U@x4, U@x5, U@x6
    //   y=11: R@x2, S@x7, U@x10, U@x11, U@x12, U@x13
    TT_FATAL(num_cores == 2, "USE_RELAY currently supports exactly 2 senders (num_links>=2); got {}", num_cores);
    {
        // physical NOC0 (x,y) -> logical worker CoreCoord, by inverting the logical->virtual map.
        std::map<std::pair<uint32_t, uint32_t>, CoreCoord> phys_to_logical;
        auto cg = mesh_device->compute_with_storage_grid_size();
        for (uint32_t ly = 0; ly < cg.y; ly++) {
            for (uint32_t lx = 0; lx < cg.x; lx++) {
                CoreCoord lc(lx, ly);
                auto v = mesh_device->virtual_core_from_logical_core(lc, tt::CoreType::WORKER);
                phys_to_logical[{(uint32_t)v.x, (uint32_t)v.y}] = lc;
            }
        }
        auto phys_to_logical_core = [&](uint32_t px, uint32_t py) -> CoreCoord {
            auto it = phys_to_logical.find({px, py});
            TT_FATAL(
                it != phys_to_logical.end(),
                "USE_RELAY: requested physical worker core ({}, {}) is not a tensix worker on this device",
                px,
                py);
            return it->second;
        };

        struct RelayBattery {
            uint32_t phys_y;
            uint32_t relay_x;
            uint32_t sender_x;
            std::array<uint32_t, 4> untilizer_x;
        };
        const std::array<RelayBattery, 2> relay_batteries = {{
            {/*phys_y=*/10, /*relay_x=*/1, /*sender_x=*/2, {{3, 4, 5, 6}}},
            {/*phys_y=*/11, /*relay_x=*/2, /*sender_x=*/7, {{10, 11, 12, 13}}},
        }};

        // Clear the default split's assignments and rebuild from the explicit table.
        sender_cores.clear();
        for (auto& g : sender_untilizer_groups) {
            g.clear();
        }
        all_untilizer_cores.clear();
        untilizer_sender_map.clear();
        relay_cores.clear();

        for (uint32_t s = 0; s < num_cores; s++) {
            const auto& b = relay_batteries[s];
            relay_cores.push_back(phys_to_logical_core(b.relay_x, b.phys_y));
            sender_cores.push_back(phys_to_logical_core(b.sender_x, b.phys_y));
            for (uint32_t u = 0; u < b.untilizer_x.size(); u++) {
                CoreCoord uc = phys_to_logical_core(b.untilizer_x[u], b.phys_y);
                sender_untilizer_groups[s].push_back(uc);
                all_untilizer_cores.push_back(uc);
                untilizer_sender_map.push_back(s);
            }
        }
        num_untilizer_cores = static_cast<uint32_t>(all_untilizer_cores.size());
        untilizer_cores_per_sender = num_untilizer_cores / num_cores;
        senders_with_extra_untilizer = num_untilizer_cores % num_cores;
    }
#endif

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
        "sender_cores: {} num_untilizer_cores: {} untilizer_cores_per_sender: {} senders_with_extra_untilizer: {}",
        hidden_size,
        num_cores,
        experts_per_core_range,
        sender_cores,
        num_untilizer_cores,
        untilizer_cores_per_sender,
        senders_with_extra_untilizer);

    // ProgramDescriptor semaphores carry an explicit `.id` field that maps directly
    // to the per-core L1 sem slot (16 slots / core).  Sem core_ranges in this function
    // do NOT all nest (per-pair data_ready and per-untilizer credits are scoped to
    // disjoint subsets), so a global monotonic counter would push ids past 16 even
    // though no individual core needs that many slots.  Emulate the legacy
    // CreateSemaphore() auto-id behaviour by tracking per-core slot usage and
    // picking the lowest id free on every core in a sem's core_range_set; sems
    // scoped to disjoint cores can reuse the same id.
    std::map<CoreCoord, std::bitset<16>> per_core_sema_slots;
    auto add_sema = [&](const CoreRangeSet& crs, uint32_t initial_value = 0) -> uint32_t {
        auto cores = corerange_to_cores(crs);
        for (uint32_t id = 0; id < 16; id++) {
            bool free_everywhere = true;
            for (const auto& c : cores) {
                if (per_core_sema_slots[c].test(id)) {
                    free_everywhere = false;
                    break;
                }
            }
            if (free_everywhere) {
                for (const auto& c : cores) {
                    per_core_sema_slots[c].set(id);
                }
                desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                    .id = id,
                    .core_type = tt::CoreType::WORKER,
                    .core_ranges = crs,
                    .initial_value = static_cast<uint16_t>(initial_value)});
                return id;
            }
        }
        TT_THROW("No free L1 semaphore slot (per-core 16 limit) for the requested core range set");
    };

    uint32_t output_init_complete_semaphore_id = add_sema(sender_core_grid);
    uint32_t output_init_barrier_semaphore_id = add_sema(sender_core_grid);

    // Rows per untilize batch.  Both layouts route through the untilizer pipeline and share the
    // same receive_buf ring / sender polling loop, so ROW_MAJOR uses the same batch size as TILE
    // (tile height = 32) rather than diverging.
    const uint32_t read_batch_size =
        is_tile_layout ? dispatched_buffer.tensor_spec().tile().get_height() : tt::constants::TILE_HEIGHT;

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
    // multicast, giving untilizer cores a host-side-free way to discover the sender's receive buffer.
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

    {
        // c_18: per-sender receive buffer.  Partitioned into k_s 16-row regions, one per untilizer
        // core in this sender's group.  Untilizer i writes to slot j in its region at offset
        //   c_18_base + i * SLOTS_PER_UNTILIZER * aligned_output_page_size + j * aligned_output_page_size
        // Size depends on k_s, so allocate per sender on its single-core CRS.  Both layouts use the
        // untilizer pipeline, so c_18/c_19 are always allocated (no sender-side dispatched_buffer CB).
        for (uint32_t s = 0; s < num_cores; s++) {
            uint32_t k_s = static_cast<uint32_t>(sender_untilizer_groups[s].size());
            CoreRangeSet single_sender_crs({CoreRange(sender_cores[s])});
            detail::create_tensor_cb(
                desc,
                single_sender_crs,
                output_tensor,
                /*buffering_factor=*/k_s * SLOTS_PER_UNTILIZER,
                /*cb_id=*/tt::CBIndex::c_18,
                "receive_buf_sender_" + std::to_string(s));
            // c_19: per-sender metadata ring.  Mirrors c_18's partitioning using the
            // dispatched_metadata page size.  Untilizer i writes routing metadata (dst_chip,
            // dst_token_idx, dst_topk_indice) for each non-local row into slot j at offset
            //   c_19_base + i * SLOTS_PER_UNTILIZER * aligned_dispatched_metadata_page_size + j * ...
            // Sender reads from c_19 instead of DRAM, eliminating metadata DRAM reads on sender.
            detail::create_tensor_cb(
                desc,
                single_sender_crs,
                dispatched_metadata,
                /*buffering_factor=*/k_s * SLOTS_PER_UNTILIZER,
                /*cb_id=*/tt::CBIndex::c_19,
                "metadata_ring_sender_" + std::to_string(s));
        }
    }

    // c_3: merged route_info + output_data (reader -> writer).
    //   Per-page layout: [0..l1_alignment) = route_info (4 x uint32_t)
    //                    [l1_alignment..page_size) = output data
    //   One reserve/push per row replaces the previous c_3 + c_4 pair.
    {
        constexpr uint32_t rw_buffering = 2;

        uint32_t route_info_page_size = l1_alignment;
        uint32_t output_payload_page_size = detail::get_aligned_page_size(output_tensor);
        uint32_t merged_page_size = route_info_page_size + output_payload_page_size;
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = rw_buffering * merged_page_size,
            .core_ranges = sender_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = tt::DataFormat::UInt8,
                .page_size = merged_page_size,
            }}},
        });
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

#if USE_RELAY
    // Relay cores: dedicated single-writer-kernel cores = the combine FABRIC ENDPOINT. Build the core
    // grid + L1 buffers here; the writer kernel is created later (after writer_defines) and its fabric
    // connection + RT args are appended in the RT-arg section.
    CoreRangeSet relay_core_grid;
    tt::tt_metal::KernelHandle relay_writer_kernel_id = 0;
    {
        std::set<CoreRange> relay_ranges_set;
        for (const auto& rc : relay_cores) {
            relay_ranges_set.insert(CoreRange(rc));
        }
        relay_core_grid = CoreRangeSet(relay_ranges_set);
        TT_FATAL(relay_cores.size() == num_cores, "Expected {} relay cores, got {}", num_cores, relay_cores.size());

        // c_24: relay receive ring. RELAY_SLOTS slots, each = route_info (l1_alignment) + one aligned
        // output page (token), mirroring the sender's c_3 layout.
        uint32_t relay_slot_size = l1_alignment + detail::get_aligned_page_size(output_tensor);
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = RELAY_SLOTS * relay_slot_size,
            .core_ranges = relay_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_24),
                .data_format = tt::DataFormat::UInt8,
                .page_size = relay_slot_size,
            }}},
        });

        // c_5: packet-header CB on the relay (2 headers: unicast + handshake), mirroring the sender's c_5.
        // The relay owns the fabric endpoint now, so it owns the header pool.
        if (num_links > 0) {
            constexpr uint32_t num_relay_packet_headers = 2;
            auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = num_relay_packet_headers * packet_header_size_bytes,
                .core_ranges = relay_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                    .data_format = tt::DataFormat::UInt8,
                    .page_size = packet_header_size_bytes,
                }}},
            });
        }
    }
#endif

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
        // CB IDs (5)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_dispatched_buffer_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_dispatched_metadata_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_experts_tok_counter_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_route_info_id (merged route_info + output_data)
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
    tt::tt_metal::KernelHandle writer_untilize_kernel_id = 0;
    std::vector<CoreCoord> writer_untilize_cores_vec;
    uint32_t output_init_done_semaphore_id = 0;
    uint32_t pages_per_core = 0;
    uint32_t remainder_pages = 0;

    // untilizer_row_cores: all same-row untilizer cores, ordered by sender group then by x.
    std::vector<CoreCoord>& untilizer_row_cores = all_untilizer_cores;

    // Per-sender multicast bounding boxes and untilizer NOC coordinates (TILE_LAYOUT only).
    // Each sender multicasts only to its own dedicated untilizer group so both senders
    // can run their multicast in parallel without interfering.
    struct SenderMcastCfg {
        uint32_t mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y;
        std::vector<std::pair<uint32_t, uint32_t>> untilizer_noc_coords;
    };
    std::vector<SenderMcastCfg> sender_mcast_cfgs(num_cores);
    {
        // Compute per-sender NOC multicast bounding box (min/max x,y over untilizer cores)
        // and collect individual untilizer core NOC coordinates for semaphore signaling.
        for (uint32_t s = 0; s < num_cores; s++) {
            TT_FATAL(!sender_untilizer_groups[s].empty(), "Sender {} has no untilizer cores assigned", s);
            auto& cfg = sender_mcast_cfgs[s];

            // Initialize bounding box from the first untilizer core in the group
            auto first_noc =
                mesh_device->virtual_core_from_logical_core(sender_untilizer_groups[s][0], tt::CoreType::WORKER);
            cfg.mcast_start_x = cfg.mcast_end_x = first_noc.x;
            cfg.mcast_start_y = cfg.mcast_end_y = first_noc.y;

            // Expand bounding box to cover all untilizer cores and record each core's NOC coords
            for (const auto& ic : sender_untilizer_groups[s]) {
                auto noc = mesh_device->virtual_core_from_logical_core(ic, tt::CoreType::WORKER);
                cfg.mcast_start_x = std::min(cfg.mcast_start_x, (uint32_t)noc.x);
                cfg.mcast_end_x = std::max(cfg.mcast_end_x, (uint32_t)noc.x);
                cfg.mcast_start_y = std::min(cfg.mcast_start_y, (uint32_t)noc.y);
                cfg.mcast_end_y = std::max(cfg.mcast_end_y, (uint32_t)noc.y);
                cfg.untilizer_noc_coords.emplace_back((uint32_t)noc.x, (uint32_t)noc.y);
            }
        }
    }

    // Build untilizer CoreRangeSet
    std::set<CoreRange> untilizer_ranges_set;
    for (const auto& core : untilizer_row_cores) {
        untilizer_ranges_set.insert(CoreRange(core));
    }
    CoreRangeSet untilizer_core_grid(untilizer_ranges_set);

    // TILE_LAYOUT semaphores for sender <-> untilizer core handshake
    uint32_t counter_ready_semaphore_id = 0;
    std::vector<std::vector<uint32_t>> data_ready_semaphore_ids(num_cores);
    std::vector<std::vector<uint32_t>> credits_semaphore_ids(num_cores);
    {
        // counter_ready semaphore: created only on sender + untilizer cores so get_semaphore() returns
        // the same L1 offset on both sides (sender writes to untilizer's copy, untilizer waits on its own)
        std::set<CoreRange> sender_and_untilizer_ranges;
        for (const auto& cr : sender_core_grid.ranges()) {
            sender_and_untilizer_ranges.insert(cr);
        }
        for (const auto& cr : untilizer_core_grid.ranges()) {
            sender_and_untilizer_ranges.insert(cr);
        }
        CoreRangeSet sender_and_untilizer_grid(sender_and_untilizer_ranges);
        counter_ready_semaphore_id = add_sema(sender_and_untilizer_grid);
        // Per (sender, untilizer) pair:
        //   data_ready (init 0, scoped to {sender, untilizer}): untilizer ++ after each row write;
        //                  sender atomically dec(-1) per row consumed.  Count = rows in flight
        //                  in untilizer's ring.  Pair-scoped so sender can use get_semaphore(id)
        //                  for fast local waits AND untilizer can NOC-inc to sender's copy.
        //   credits    (init 0, scoped to {untilizer only}): sender ++ untilizer's L1 via NOC each
        //                  time it frees a row slot.  Untilizer's kernel-side local_credits
        //                  already starts at SLOTS_PER_UNTILIZER to cover the initially-empty
        //                  ring; the sem must start at 0 so untilizer doesn't double-count the
        //                  initial credits when it later sucks the sem (otherwise it would
        //                  overwrite live slots).  Allocated on untilizer-only CRS so it does
        //                  not consume a sem slot on sender (which is at the 16/core limit
        //                  for senders with k_s≈7); sender still derives the L1 offset via
        //                  the uniform `get_semaphore(id)` formula and addresses it remotely.
        // Per-pair sems are scoped tightly so they don't burn one of the 16 per-core slots
        // on cores that don't need them.
        for (uint32_t s = 0; s < num_cores; s++) {
            uint32_t k_s = static_cast<uint32_t>(sender_untilizer_groups[s].size());
            for (uint32_t c = 0; c < k_s; c++) {
                CoreRangeSet pair_grid(
                    std::set<CoreRange>{CoreRange(sender_cores[s]), CoreRange(sender_untilizer_groups[s][c])});
                CoreRangeSet untilizer_only_grid(std::set<CoreRange>{CoreRange(sender_untilizer_groups[s][c])});
                data_ready_semaphore_ids[s].push_back(add_sema(pair_grid));
                credits_semaphore_ids[s].push_back(add_sema(untilizer_only_grid));
            }
        }
    }

#if USE_RELAY
    // sender->relay flow-control semaphores (1:1 sender:relay), scoped to each {sender, relay} pair so
    // the id is valid on both cores:
    //   relay_data_ready (lives on relay, init 0): sender ++ after writing a token slot; relay waits
    //                     on a monotonic `consumed` counter.
    //   relay_credits    (lives on sender, init 0): relay ++ when it frees a slot; sender keeps a local
    //                     credit counter (seeded with the full ring budget of RELAY_SLOTS) and sucks this
    //                     sem when it runs out. MUST init 0 -- it carries slot RETURNS, not the initial
    //                     budget; seeding it to RELAY_SLOTS as well would double the sender's write budget
    //                     to 2*RELAY_SLOTS, letting it lap the ring and overwrite unconsumed slots.
    //   relay_buf_addr   (lives on sender, init 0): relay publishes get_write_ptr(c_24) here so the sender
    //                     learns where to NOC-write tokens; the sender spins until it is non-zero.
    std::vector<uint32_t> relay_data_ready_sem_ids(num_cores);
    std::vector<uint32_t> relay_credits_sem_ids(num_cores);
    std::vector<uint32_t> relay_buf_addr_sem_ids(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        CoreRangeSet pair_grid(std::set<CoreRange>{CoreRange(sender_cores[s]), CoreRange(relay_cores[s])});
        relay_data_ready_sem_ids[s] = add_sema(pair_grid, 0);
        relay_credits_sem_ids[s] = add_sema(pair_grid, 0);
        relay_buf_addr_sem_ids[s] = add_sema(pair_grid, 0);
    }
#endif

    // Largest divisor of (hidden_size / 32) that is <= 8.  Reader_untilize pushes tiles into
    // cb_dispatched_buffer (c_0) in chunks of this size, and the untilize compute kernel consumes
    // the same chunk size — so c_0 only needs to hold 2 * block_ct_dim pages for double-buffering.
    const uint32_t full_ct_dim = static_cast<uint32_t>(hidden_size) / 32u;
    uint32_t block_ct_dim = 8;
    while (full_ct_dim % block_ct_dim != 0) {
        --block_ct_dim;
    }

    {
        // c_1 on untilizer cores: receives the expert_token_counts multicast from the owning sender.
        // MUST be allocated BEFORE c_0/c_2 so its L1 address matches the sender's c_1 address.
        // Both layouts use the untilizer pipeline, so these CBs are always allocated; only c_0
        // (tile-input for the compute kernel) is TILE-only.
        {
            uint32_t counter_pages = detail::get_num_pages(expert_token_counts);
            uint32_t counter_page_size = detail::get_aligned_page_size(expert_token_counts);
            auto data_format = tt::tt_metal::datatype_to_dataformat_converter(expert_token_counts.dtype());
            uint32_t extra_pages = 1;
            uint32_t cb_size = (counter_pages + extra_pages) * counter_page_size;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = cb_size,
                .core_ranges = untilizer_core_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
                    .data_format = data_format,
                    .page_size = counter_page_size,
                }}},
            });
        }
        // c_0 on untilizer cores: dispatched_buffer tiles, sized for double-buffered block_ct_dim
        // chunks.  TILE_LAYOUT only — the compute kernel reads c_0 and untilizes into c_2.  In
        // ROW_MAJOR there is no compute kernel and reader_untilize writes rows straight into c_2,
        // so c_0 is not allocated.
        if (is_tile_layout) {
            detail::create_tensor_cb(
                desc,
                untilizer_core_grid,
                dispatched_buffer,
                /*buffering_factor=*/2 * block_ct_dim,
                /*cb_id=*/tt::CBIndex::c_0,
                "dispatched_buffer_untilizer");
        }
        // c_2 on untilizer cores: untilized (TILE) or row-copied (ROW_MAJOR) output rows,
        // double-buffered (2 × read_batch_size).  Lets the producer fill batch N+1 into the
        // second half while writer_untilize is still routing batch N out of the first half.
        detail::create_tensor_cb(
            desc,
            untilizer_core_grid,
            output_tensor,
            /*buffering_factor=*/2 * read_batch_size,
            /*cb_id=*/tt::CBIndex::c_2,
            "untilize");
        // c_9 on untilizer cores: metadata-batch CB. reader_untilize on this core reads the
        // per-batch metadata pages from DRAM and pushes them here; writer_untilize pops
        // batch_count pages each iteration and decides the per-batch path locally (sender
        // no longer writes to this CB).  Double-buffered (2 × read_batch_size) so
        // reader_untilize can stage batch N+1's metadata while writer_untilize is still
        // consuming batch N — matches the double-buffered untilize CB (c_2).
        {
            uint32_t metadata_batch_page_size = detail::get_aligned_page_size(dispatched_metadata);
            auto metadata_fmt = tt::tt_metal::datatype_to_dataformat_converter(dispatched_metadata.dtype());
            uint32_t metadata_batch_cb_size = 2 * read_batch_size * metadata_batch_page_size;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = metadata_batch_cb_size,
                .core_ranges = untilizer_core_grid,
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

    // reader_untilize runs on untilizer cores for BOTH layouts: TILE reads tiles into c_0 (for the
    // compute kernel), ROW_MAJOR reads rows straight into c_2 (no compute).  The compute kernel
    // itself is created TILE-only further below.
    std::vector<tt::tt_metal::KernelHandle> reader_untilize_kernel_ids;
    {
        // Compile-time args layout for reader_untilize (matching reader_untilize.cpp):
        //   0-11: shared base (below, includes max_dispatch_buffer_token_size at 11)
        //   12:   aligned_output_page_size
        //   13:   aligned_experts_tok_counter_page_size
        //   14:   cb_metadata_batch_id — CB this kernel pushes per-batch metadata pages into
        //   15:   aligned_dispatched_metadata_page_size
        //   16:   block_ct_dim — tiles per chunk pushed to cb_dispatched_buffer_id (matches the
        //                       compute kernel's per-block consumption)
        //   17+:  TensorAccessorArgs for dispatched_buffer, then TensorAccessorArgs for
        //         dispatched_metadata (no num_senders — single-sender kernel)
        // ROW_MAJOR dispatched_buffer has no tile spec; use the hardware tile dims so the
        // tile-aligned per-expert region math (start_token / tiles_per_batch) in reader_untilize
        // matches the host's expert_region_offsets (cumsum of ceil(count, 32)*32) in both layouts.
        const uint32_t tile_height =
            is_tile_layout ? dispatched_buffer.tensor_spec().tile().get_height() : tt::constants::TILE_HEIGHT;
        const uint32_t tile_width =
            is_tile_layout ? dispatched_buffer.tensor_spec().tile().get_width() : tt::constants::TILE_WIDTH;
        const std::vector<uint32_t> reader_untilize_compile_time_args_base = {
            static_cast<uint32_t>(tt::CBIndex::c_1),           // 0:  cb_experts_tok_counter_id
            detail::get_num_pages(expert_token_counts),        // 1:  experts_tok_counter_pages
            operation_attributes.experts_per_chip,             // 2:  experts_per_chip
            counter_offset,                                    // 3:  counter_offset
            static_cast<uint32_t>(tt::CBIndex::c_0),           // 4:  cb_dispatched_buffer_id
            static_cast<uint32_t>(tt::CBIndex::c_2),           // 5:  cb_untilize_id
            (uint32_t)hidden_size,                             // 6:  hidden_size
            read_batch_size,                                   // 7:  read_batch_size
            detail::get_aligned_page_size(dispatched_buffer),  // 8:  aligned_dispatched_buffer_page_size
            tile_height,                                       // 9:  tile_height
            tile_width,                                        // 10: tile_width
            (uint32_t)max_dispatch_buffer_token_size,          // 11: max_dispatch_buffer_token_size
        };

        // Partitioned untilizer cores: each sender s owns a dedicated group of k_s untilizer cores.
        // The global round-robin (untilizer_global_pos / total_untilizers, passed as runtime args)
        // drives batch assignment, so the kernel no longer needs core_id / num_untilizer_cores.
        // No num_senders arg — each kernel is bound to a single sender.
        reader_untilize_kernel_ids.reserve(num_untilizer_cores);

        uint32_t global_untilizer_idx = 0;
        for (uint32_t s = 0; s < num_cores; s++) {
            uint32_t k_s = static_cast<uint32_t>(sender_untilizer_groups[s].size());
            for (uint32_t j = 0; j < k_s; j++, global_untilizer_idx++) {
                auto per_core_args = reader_untilize_compile_time_args_base;
                per_core_args.push_back(detail::get_aligned_page_size(output_tensor));        // 12
                per_core_args.push_back(detail::get_aligned_page_size(expert_token_counts));  // 13
                per_core_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_9));  // 14: cb_metadata_batch_id
                per_core_args.push_back(detail::get_aligned_page_size(dispatched_metadata));  // 15
                per_core_args.push_back(block_ct_dim);                                        // 16: block_ct_dim
                // 17: cb_counter_total_pages = full page capacity of c_1 on the untilizer
                // (counter pages + trailer page). Passed so reader_untilize can reserve / push /
                // wait the entire CB, not just the counter slice.
                per_core_args.push_back(detail::get_num_pages(expert_token_counts) + 1);  // cb_counter_total_pages
                // 18+: TensorAccessorArgs for dispatched_buffer + dispatched_metadata
                tt::tt_metal::TensorAccessorArgs(dispatched_buffer.buffer()).append_to(per_core_args);
                tt::tt_metal::TensorAccessorArgs(dispatched_metadata.buffer()).append_to(per_core_args);

                CoreRangeSet single_untilizer_core({CoreRange(untilizer_row_cores[global_untilizer_idx])});
                tt::tt_metal::KernelDescriptor reader_untilize_kd;
                reader_untilize_kd.kernel_source =
                    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/"
                    "reader_untilize.cpp";
                reader_untilize_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
                reader_untilize_kd.core_ranges = single_untilizer_core;
                reader_untilize_kd.compile_time_args = std::move(per_core_args);
                reader_untilize_kd.defines = {{"IS_TILE_LAYOUT", is_tile_layout ? "1" : "0"}};
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

    // writer_untilize runs on untilizer cores for BOTH layouts now: it consumes cb_untilize_id
    // (c_2 — produced by the compute kernel in TILE, or directly by reader_untilize in ROW_MAJOR)
    // and forwards each row to the owning sender's receive_buf (c_18) / metadata ring (c_19).  It
    // also performs the optional per-bank output-zeroing when init_zeros=True (INIT_ZEROS define).
    const bool create_writer_untilize_kernel = true;

    if (init_zeros) {
        uint32_t noc_max_burst_size;
        const auto arch = mesh_device->arch();
        if (arch == tt::ARCH::BLACKHOLE) {
            noc_max_burst_size = 16384;
        } else if (arch == tt::ARCH::WORMHOLE_B0) {
            noc_max_burst_size = 8192;
        } else {
            TT_THROW("Unsupported architecture for output-zeroing: {}", arch);
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

        uint32_t total_init_cores = num_cores + num_untilizer_cores;
        uint32_t total_output_pages = detail::get_num_pages(output_tensor);
        pages_per_core = total_output_pages / total_init_cores;
        remainder_pages = total_output_pages % total_init_cores;

        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = noc_max_burst_size,
            .core_ranges = untilizer_core_grid,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
                .data_format = tt::DataFormat::UInt8,
                .page_size = noc_max_burst_size,
            }}},
        });

        output_init_done_semaphore_id = add_sema(worker_core_range_set);
    }

    if (create_writer_untilize_kernel) {
        uint32_t output_aligned_page_size = detail::get_aligned_page_size(output_tensor);
        std::vector<uint32_t> writer_untilize_compile_time_args = {
            output_aligned_page_size,
            // num_sender_cores and cb_zero_buffer_id are only referenced inside the
            // INIT_ZEROS-gated output-zeroing phase in the kernel; pass 0 when init_zeros=False
            // (the c_6 CB is not created in that case so its index is meaningless).
            init_zeros ? num_cores : 0u,
            init_zeros ? static_cast<uint32_t>(tt::CBIndex::c_6) : 0u,
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_untilize_compile_time_args);

        // Compile-time args for the untilized-data send loop, which now runs for BOTH layouts
        // (writer_untilize forwards cb_untilize_id rows to the sender regardless of layout).
        writer_untilize_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_2));  // cb_untilize_id
        writer_untilize_compile_time_args.push_back(
            static_cast<uint32_t>(tt::CBIndex::c_1));  // cb_experts_tok_counter_id
        writer_untilize_compile_time_args.push_back(
            detail::get_num_pages(expert_token_counts));  // experts_tok_counter_pages
        writer_untilize_compile_time_args.push_back(
            detail::get_aligned_page_size(expert_token_counts));  // counter page size
        writer_untilize_compile_time_args.push_back(read_batch_size);
        writer_untilize_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_9));   // cb_metadata_batch_id
        writer_untilize_compile_time_args.push_back(operation_attributes.num_experts_per_tok);  // num_experts_per_tok
        writer_untilize_compile_time_args.push_back(
            detail::get_aligned_page_size(dispatched_metadata));             // aligned_dispatched_metadata_page_size
        writer_untilize_compile_time_args.push_back(linearized_mesh_coord);  // linearized_mesh_coord
        writer_untilize_compile_time_args.push_back(operation_attributes.experts_per_chip);  // experts_per_chip
        writer_untilize_compile_time_args.push_back(counter_offset);                         // counter_offset
        writer_untilize_compile_time_args.push_back(
            (uint32_t)max_dispatch_buffer_token_size);             // max_dispatch_buffer_token_size
        writer_untilize_compile_time_args.push_back(full_ct_dim);  // full_ct_dim (= hidden_size / 32)
        // cb_counter_total_pages = full page capacity of c_1 on the untilizer (counter pages +
        // trailer page). Used so writer_untilize cb_wait_fronts the entire CB.
        writer_untilize_compile_time_args.push_back(
            detail::get_num_pages(expert_token_counts) + 1);  // cb_counter_total_pages
        writer_untilize_compile_time_args.push_back(
            SLOTS_PER_UNTILIZER);  // per-untilizer ring depth on the sender's receive_buf

        std::map<std::string, std::string> writer_untilize_defines;
        writer_untilize_defines["IS_TILE_LAYOUT"] = is_tile_layout ? "1" : "0";
        writer_untilize_defines["INIT_ZEROS"] = init_zeros ? "1" : "0";

        tt::tt_metal::KernelDescriptor writer_untilize_kd;
        writer_untilize_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/"
            "writer_untilize.cpp";
        writer_untilize_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        writer_untilize_kd.core_ranges = untilizer_core_grid;
        writer_untilize_kd.compile_time_args = std::move(writer_untilize_compile_time_args);
        writer_untilize_kd.defines = {writer_untilize_defines.begin(), writer_untilize_defines.end()};
        writer_untilize_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
        };
        writer_untilize_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(writer_untilize_kd));

        writer_untilize_cores_vec = untilizer_row_cores;
    }

    // Reader compile-time args base (without num_untilizer_cores — that is per-sender and appended below).
    std::vector<uint32_t> reader_compile_time_args_base = compile_time_args;
    if (init_zeros) {
        reader_compile_time_args_base.push_back(static_cast<uint32_t>(tt::CBIndex::c_7));  // zero-buffer CB id (c_7)
        reader_compile_time_args_base.push_back(
            num_untilizer_cores);  // num_total_untilizer_cores (both layouts need this)
    }
    // num_untilizer_cores (per-sender k_s) and the c_18/c_19 ring CBs are appended per-sender below.

    // One reader_combine kernel per sender.  k_s (per-sender untilizer count) is baked in as
    // num_untilizer_cores so the sender only round-robins across its own dedicated untilizer
    // cores.  Both layouts use the untilizer-fed sender path, so these args are always appended.
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    reader_kernel_ids.reserve(num_cores);
    for (uint32_t s = 0; s < num_cores; s++) {
        auto per_sender_compile_args = reader_compile_time_args_base;
        {
            uint32_t k_s = static_cast<uint32_t>(sender_untilizer_groups[s].size());
            per_sender_compile_args.push_back(k_s);  // num_untilizer_cores (per-sender)
            per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_18));  // cb_untilize_id
            per_sender_compile_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_19));  // cb_metadata_buf_id
            per_sender_compile_args.push_back(SLOTS_PER_UNTILIZER);                       // per-untilizer ring depth
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

#if USE_RELAY
    // Relay writer kernel = the fabric endpoint (opens eth connections, runs the init/exit handshake,
    // drains its c_24 ring, sends tokens to fabric). Single kernel on its core -> free to use NOC_0. Gets
    // the same fabric defines as writer_combine (DEST_CHIP_ID / DEST_MESH_ID / DIRECTIONS / AXIS). num_chips
    // mirrors writer_combine's compile arg 13 (== dispatch_group_size).
    {
        std::vector<uint32_t> relay_compile_time_args = {
            detail::get_num_pages(output_tensor),          // 0:  output_pages
            detail::get_aligned_page_size(output_tensor),  // 1:  aligned_output_page_size
            (uint32_t)fabric_max_packet_size,              // 2:  fabric_max_packet_size
            l1_alignment,                                  // 3:  l1_alignment
            static_cast<uint32_t>(num_links),              // 4:  num_links
            static_cast<uint32_t>(topology),               // 5:  topology
            mesh_rows,                                     // 6:  mesh_rows
            mesh_cols,                                     // 7:  mesh_cols
            linearized_mesh_coord,                         // 8:  linearized_mesh_coord
            src_chip_id,                                   // 9:  src_chip_id
            src_mesh_id,                                   // 10: src_mesh_id
            operation_attributes.dispatch_group_size,      // 11: num_chips (== writer_combine arg 13)
            static_cast<uint32_t>(tt::CBIndex::c_5),       // 12: cb_packet_header_id
            static_cast<uint32_t>(tt::CBIndex::c_24),      // 13: cb_relay_buf
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(relay_compile_time_args);  // 14+

        tt::tt_metal::KernelDescriptor relay_writer_kd;
        relay_writer_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/"
            "writer_relay.cpp";
        relay_writer_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        relay_writer_kd.core_ranges = relay_core_grid;
        relay_writer_kd.compile_time_args = std::move(relay_compile_time_args);
        relay_writer_kd.defines = {writer_defines.begin(), writer_defines.end()};
        relay_writer_kd.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
        };
        relay_writer_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(relay_writer_kd));
    }
#endif

    // Compute kernel on untilizer cores that untilizes dispatched_buffer data (TILE_LAYOUT only).
    // Compile-time args are shared across all untilizer cores; per-sender values (core_id,
    // num_untilizer_cores, expert range) are passed via SetRuntimeArgs below.  Initialized to 0
    // so the compiler can prove definite-initialization for the !is_tile_layout case (the
    // SetRuntimeArgs call below is guarded by the same is_tile_layout flag).
    tt::tt_metal::KernelHandle untilize_compute_kernel_id = 0;
    if (is_tile_layout) {
        tt::tt_metal::KernelDescriptor compute_kd;
        compute_kd.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/compute/"
            "untilize_combine.cpp";
        compute_kd.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        compute_kd.core_ranges = untilizer_core_grid;
        compute_kd.compile_time_args = {
            static_cast<uint32_t>(tt::CBIndex::c_2),     // 0: cb_untilize_id
            static_cast<uint32_t>(tt::CBIndex::c_0),     // 1: cb_in_id
            static_cast<uint32_t>(tt::CBIndex::c_1),     // 2: cb_experts_tok_counter_id
            detail::get_num_pages(expert_token_counts),  // 3: experts_tok_counter_pages
            operation_attributes.experts_per_chip,       // 4: experts_per_chip
            counter_offset,                              // 5: counter_offset
            (uint32_t)max_dispatch_buffer_token_size,    // 6: max_dispatch_buffer_token_size
            read_batch_size,                             // 7: read_batch_size
            full_ct_dim,                                 // 8: full_ct_dim = hidden_size / 32
            block_ct_dim,                                // 9: block_ct_dim
            // 10: cb_counter_total_pages = full page capacity of c_1 on the untilizer
            // (counter pages + trailer page). Used so this kernel cb_wait_fronts the entire CB.
            detail::get_num_pages(expert_token_counts) + 1,
        };
        compute_kd.config = tt::tt_metal::ComputeConfigDescriptor{
            // Blackhole requires the DEST register in 32-bit mode whenever any CB on the core uses
            // an 8-bit float format (Fp8_e4m3). The FP8 combine path produces an Fp8_e4m3 output
            // CB (c_2), so fp32_dest_acc_en must be enabled there.
            .fp32_dest_acc_en = operation_attributes.use_fp8_combine,
            // 32-bit DEST halves pack_untilize block capacity: half-sync 32-bit allows only 4
            // tiles, but pack_untilize_block uses block_ct_dim (up to 8). Full-sync 32-bit restores
            // the 8-tile budget so the block still fits. Only needed on the FP8 (32-bit) path.
            .dst_full_sync_en = operation_attributes.use_fp8_combine,
        };
        untilize_compute_kernel_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(compute_kd));
    }

    // Pre-compute NOC coordinates for all sender cores (for inter-core barrier signaling)
    std::vector<std::pair<uint32_t, uint32_t>> sender_noc_coords;
    for (const auto& sc : sender_cores) {
        auto noc_coord = mesh_device->virtual_core_from_logical_core(sc, tt::CoreType::WORKER);
        sender_noc_coords.emplace_back(noc_coord.x, noc_coord.y);
    }

    // Global interleaved position of each untilizer core in rank-major / sender-minor order
    // [S0.U0, S1.U0, S0.U1, S1.U1, …]: every untilizer core processes batches global_pos, +G,
    // +2G, … of every expert (G = num_untilizer_cores, the total untilizer-core count).  Spreading
    // consecutive batches of an expert across senders keeps either sender's forwarder from getting
    // a monopoly of local (or remote) rows regardless of how dispatch clustered them; each sender's
    // batch share is proportional to its untilizer-core count.  Indexed by untilizer idx so both
    // the writer_untilize loop and the reader_untilize/compute loop below agree on the same value.
    std::vector<uint32_t> untilizer_global_pos(num_untilizer_cores, 0);
    {
        uint32_t max_k = 0;
        for (uint32_t s = 0; s < num_cores; s++) {
            max_k = std::max(max_k, static_cast<uint32_t>(sender_untilizer_groups[s].size()));
        }
        std::vector<std::vector<uint32_t>> pos_by_sender_rank(num_cores);
        for (uint32_t s = 0; s < num_cores; s++) {
            pos_by_sender_rank[s].resize(sender_untilizer_groups[s].size());
        }
        uint32_t pos = 0;
        for (uint32_t r = 0; r < max_k; r++) {
            for (uint32_t s = 0; s < num_cores; s++) {
                if (r < sender_untilizer_groups[s].size()) {
                    pos_by_sender_rank[s][r] = pos++;
                }
            }
        }
        std::vector<uint32_t> rank_counter(num_cores, 0);
        for (uint32_t idx = 0; idx < num_untilizer_cores; idx++) {
            uint32_t s = untilizer_sender_map[idx];
            uint32_t r = rank_counter[s]++;
            untilizer_global_pos[idx] = pos_by_sender_rank[s][r];
        }
    }

    // Set runtime args for hybrid untilizer row cores.  Three layouts are possible:
    //   init_zeros && tile_layout: [output_addr, page_start, page_end, output_init_done_sem,
    //                               (sender_noc_x, sender_noc_y) * num_cores,
    //                               counter_ready_sem, owning_sender_noc_x, owning_sender_noc_y,
    //                               data_ready_sem, start_sem, local_core_id]
    //   init_zeros && row_major:   [output_addr, page_start, page_end, output_init_done_sem,
    //                               (sender_noc_x, sender_noc_y) * num_cores]
    //   !init_zeros && tile_layout:[output_addr, counter_ready_sem, owning_sender_noc_x,
    //                               owning_sender_noc_y, data_ready_sem, start_sem, local_core_id]
    // The kernel guards the output-zeroing reads with #if INIT_ZEROS so the indices match.
    if (create_writer_untilize_kernel) {
        for (uint32_t untilizer_idx = 0; untilizer_idx < num_untilizer_cores; untilizer_idx++) {
            // Push output_tensor's buffer first as Buffer* so the framework records
            // a BufferBinding for the cache-hit fast path.
            tt::tt_metal::KernelDescriptor::RTArgList writer_untilize_runtime_args;
            writer_untilize_runtime_args.push_back(output_tensor.buffer());

            if (init_zeros) {
                uint32_t row_idx = num_cores + untilizer_idx;
                uint32_t page_start = (row_idx * pages_per_core) + std::min(row_idx, remainder_pages);
                uint32_t page_end = page_start + pages_per_core + (row_idx < remainder_pages ? 1 : 0);
                writer_untilize_runtime_args.push_back(page_start);
                writer_untilize_runtime_args.push_back(page_end);
                writer_untilize_runtime_args.push_back(output_init_done_semaphore_id);
                // Each untilizer core signals all sender cores once its output-zeroing slice is done.
                for (const auto& [noc_x, noc_y] : sender_noc_coords) {
                    writer_untilize_runtime_args.push_back(noc_x);
                    writer_untilize_runtime_args.push_back(noc_y);
                }
            }

            // Append owning-sender info so writer_untilize can run its send loop.  Both layouts
            // forward c_2 rows to the sender, so these args are always pushed.
            {
                uint32_t s = untilizer_sender_map[untilizer_idx];
                // Every sender now processes EVERY expert (full range); the per-expert work is
                // split across senders by data instead — sender s handles batch-chunk s of each
                // expert (see sender_idx / num_senders below).
                uint32_t expert_start = 0;
                uint32_t expert_end = operation_attributes.experts_per_chip;
                // core_id = this untilizer core's local index within sender s's group (0..k_s-1).
                uint32_t local_core_id = 0;
                for (uint32_t j = 0; j < untilizer_idx; j++) {
                    if (untilizer_sender_map[j] == s) {
                        local_core_id++;
                    }
                }
                writer_untilize_runtime_args.push_back(counter_ready_semaphore_id);
                writer_untilize_runtime_args.push_back(sender_noc_coords[s].first);
                writer_untilize_runtime_args.push_back(sender_noc_coords[s].second);
                writer_untilize_runtime_args.push_back(data_ready_semaphore_ids[s][local_core_id]);
                writer_untilize_runtime_args.push_back(credits_semaphore_ids[s][local_core_id]);
                writer_untilize_runtime_args.push_back(local_core_id);
                writer_untilize_runtime_args.push_back(expert_start);  // expert_start_idx
                writer_untilize_runtime_args.push_back(expert_end);    // expert_end_idx
                writer_untilize_runtime_args.push_back(untilizer_global_pos[untilizer_idx]);  // global batch start
                writer_untilize_runtime_args.push_back(num_untilizer_cores);                  // global batch stride (G)
            }

            desc.kernels[writer_untilize_kernel_id].emplace_runtime_args(
                writer_untilize_cores_vec[untilizer_idx], writer_untilize_runtime_args);
        }
    }

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        // Sender kernels (reader_combine / writer_combine) are expert-range agnostic — they only
        // poll their untilizer group's receive_buf and fabric-forward whatever rows arrive — so
        // they get the full expert range now that the per-expert split is by data, not by expert.
        uint32_t expert_start = 0;
        uint32_t expert_end = operation_attributes.experts_per_chip;

        // Reader RT args.  Tensor buffer addresses are pushed as Buffer* so the
        // framework records BufferBindings for the O(1) cache-hit fast path.
        tt::tt_metal::KernelDescriptor::RTArgList reader_runtime_args;
        reader_runtime_args.push_back(dispatched_buffer.buffer());
        reader_runtime_args.push_back(dispatched_metadata.buffer());
        reader_runtime_args.push_back(expert_token_counts.buffer());
        reader_runtime_args.push_back(expert_region_offsets.buffer());
        reader_runtime_args.push_back(output_tensor.buffer());
        reader_runtime_args.push_back(output_init_complete_semaphore_id);
        reader_runtime_args.push_back(output_init_barrier_semaphore_id);
        reader_runtime_args.push_back(num_cores);
        reader_runtime_args.push_back(expert_start);
        reader_runtime_args.push_back(expert_end);
        if (init_zeros) {
            uint32_t sender_page_start = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);
            uint32_t sender_page_end = sender_page_start + pages_per_core + (core_idx < remainder_pages ? 1 : 0);
            reader_runtime_args.push_back(sender_page_start);
            reader_runtime_args.push_back(sender_page_end);
            reader_runtime_args.push_back(output_init_done_semaphore_id);
        }
        {
            // Multicast targets only this sender's dedicated untilizer group
            const auto& mcast_cfg = sender_mcast_cfgs[core_idx];
            reader_runtime_args.push_back(counter_ready_semaphore_id);
            reader_runtime_args.push_back(mcast_cfg.mcast_start_x);
            reader_runtime_args.push_back(mcast_cfg.mcast_start_y);
            reader_runtime_args.push_back(mcast_cfg.mcast_end_x);
            reader_runtime_args.push_back(mcast_cfg.mcast_end_y);
            // Per untilizer core: (data_ready_sem_id, credits_sem_id, untilizer_noc_x, untilizer_noc_y).
            // The kernel reconstructs parallel arrays of per-untilizer sem pointers / NOC addrs.
            const auto& per_sender_data_ready = data_ready_semaphore_ids[core_idx];
            const auto& per_sender_credits = credits_semaphore_ids[core_idx];
            for (uint32_t c = 0; c < mcast_cfg.untilizer_noc_coords.size(); c++) {
                reader_runtime_args.push_back(per_sender_data_ready[c]);
                reader_runtime_args.push_back(per_sender_credits[c]);
                reader_runtime_args.push_back(mcast_cfg.untilizer_noc_coords[c].first);
                reader_runtime_args.push_back(mcast_cfg.untilizer_noc_coords[c].second);
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
            output_init_complete_semaphore_id,
            (uint32_t)init_semaphore.address(),
            (uint32_t)exit_semaphore.address(),
            output_init_barrier_semaphore_id,
            num_cores,
            expert_start,
            expert_end,
        };

        // Append NOC coordinates of all cores for inter-core barrier signaling
        for (const auto& [noc_x, noc_y] : sender_noc_coords) {
            writer_runtime_args_raw.push_back(noc_x);
            writer_runtime_args_raw.push_back(noc_y);
        }

#if USE_RELAY
        // sender->relay pipe args (owning relay NOC coords + the 3 flow-control sem ids), appended right
        // after the barrier NOC coords; the sender reads them in its USE_RELAY producer block. No
        // fabric-connection args follow (the sender is not the fabric endpoint), so these are the last of
        // the sender's RT args. NOTE: the combine_sender_index arg is NOT appended in this build -- the relay
        // owns the flow-log window -- and writer_combine skips reading it under USE_RELAY (kept in sync).
        {
            auto relay_v = mesh_device->virtual_core_from_logical_core(relay_cores[core_idx], tt::CoreType::WORKER);
            writer_runtime_args_raw.push_back((uint32_t)relay_v.x);
            writer_runtime_args_raw.push_back((uint32_t)relay_v.y);
            writer_runtime_args_raw.push_back(relay_data_ready_sem_ids[core_idx]);
            writer_runtime_args_raw.push_back(relay_credits_sem_ids[core_idx]);
            writer_runtime_args_raw.push_back(relay_buf_addr_sem_ids[core_idx]);
        }
#else
        // [debug] per-sender index, used by writer_combine to build the eRISC combine marker value
        // (100 + chip*10 + index) written into the connected eth router's telemetry scratch[0] to open/close
        // the flow-log window. Read unconditionally by the kernel (before the fabric-connection args), so
        // append it here for all cores.
        writer_runtime_args_raw.push_back(core_idx);
#endif

        // USE_RELAY: the relay is the fabric endpoint, so the sender opens no fabric connection and gets
        // no fabric-connection RT args (writer_combine returns before it would read them). The relay's
        // connection is appended in the relay RT-arg loop below.
        if (num_links > 0 && !USE_RELAY) {
            // Combine-axis neighbors (each a distinct fabric direction) as fabric nodes.
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }
                dst_nodes.push_back(mesh_device->get_fabric_node_id(neighbor_coordinate));
            }
            const uint32_t core_link = core_idx % num_links;
            if (is_2d_fabric) {
                // Portable RoutingPlaneConnectionManager path: one connection per combine-axis neighbor
                // so traffic forwards across MULTIPLE hops (the legacy fixed-link array connection only
                // forwards a single hop, deadlocking multi-hop FABRIC_2D — e.g. the 4-device column of a
                // 4x2 mesh). The writer reads num_connections first, then builds the manager from the
                // appended args. {core_link} (= core_idx % num_links) is one link index applied to all of
                // this sender core's connections, spreading sender cores across links (matches the
                // FABRIC_1D path & broadcast).
                writer_runtime_args_raw.push_back(static_cast<uint32_t>(dst_nodes.size()));
                tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                    src_fabric_node_id,
                    dst_nodes,
                    {core_link},
                    desc,
                    writer_kernel_id,
                    sender_core,
                    writer_runtime_args_raw);
                log_debug(
                    tt::LogOp,
                    "FABRIC_2D combine writer: src={} num_connections={} core_link={}",
                    src_fabric_node_id,
                    dst_nodes.size(),
                    core_link);
            } else {
                // Legacy per-direction array connection (FABRIC_1D linear/ring — never deadlocked).
                for (const auto& dst_node : dst_nodes) {
                    tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                        src_fabric_node_id, dst_node, core_link, desc, sender_core, writer_runtime_args_raw);
                }
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

#if USE_RELAY
    // Relay RT args = the fabric endpoint's args. Order the relay kernel reads:
    //   output_addr, init_sem_addr, exit_sem_addr, sender pipe args (sender NOC x/y + 3 sem ids),
    //   combine_window_index, then the fabric-connection args (FABRIC_2D: num_connections +
    //   RoutingPlaneConnectionManager args; FABRIC_1D: per-target unicast). output_addr is promoted to a
    //   Buffer* below for the cache-hit fast path. Matches the sender's connection-append branch above.
    for (uint32_t s = 0; s < num_cores; s++) {
        const CoreCoord& relay_core = relay_cores[s];
        std::vector<uint32_t> relay_rt_raw = {
            output_tensor.buffer()->address(),   // 0: output_addr (promoted to Buffer* below)
            (uint32_t)init_semaphore.address(),  // 1: init_semaphore_address
            (uint32_t)exit_semaphore.address(),  // 2: exit_semaphore_address
        };
        // 3..7: sender->relay pipe args (owning sender NOC coords + the 3 flow-control sem ids). Read by
        // the relay BEFORE open_direction_connections, so they must precede the fabric-connection args.
        auto sender_v = mesh_device->virtual_core_from_logical_core(sender_cores[s], tt::CoreType::WORKER);
        relay_rt_raw.push_back((uint32_t)sender_v.x);
        relay_rt_raw.push_back((uint32_t)sender_v.y);
        relay_rt_raw.push_back(relay_data_ready_sem_ids[s]);
        relay_rt_raw.push_back(relay_credits_sem_ids[s]);
        relay_rt_raw.push_back(relay_buf_addr_sem_ids[s]);
        // 8: [debug] per-relay index (== sender index) for the eth-router detailed flow-log window id
        // (100 + chip*10 + index). The relay is the fabric endpoint under USE_RELAY, so it -- not the sender --
        // opens/closes the [rxlog]/[txlog] window. Read before open_direction_connections (see writer_relay).
        relay_rt_raw.push_back(s);

        if (num_links > 0) {
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }
                dst_nodes.push_back(mesh_device->get_fabric_node_id(neighbor_coordinate));
            }
            const uint32_t core_link = s % num_links;
            if (is_2d_fabric) {
                // Portable RoutingPlaneConnectionManager path (mirrors the sender branch above): one
                // connection per combine-axis neighbor so traffic forwards across MULTIPLE hops. The relay
                // reads num_connections first, then builds the manager from the appended args.
                relay_rt_raw.push_back(static_cast<uint32_t>(dst_nodes.size()));
                tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                    src_fabric_node_id, dst_nodes, {core_link}, desc, relay_writer_kernel_id, relay_core, relay_rt_raw);
                log_debug(
                    tt::LogOp,
                    "FABRIC_2D combine relay: src={} num_connections={} core_link={}",
                    src_fabric_node_id,
                    dst_nodes.size(),
                    core_link);
            } else {
                // Legacy per-direction array connection (FABRIC_1D linear/ring).
                for (const auto& dst_node : dst_nodes) {
                    tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                        src_fabric_node_id, dst_node, core_link, desc, relay_core, relay_rt_raw);
                }
            }
        }

        // Promote: output_addr slot -> Buffer*, everything else stays uint32.
        tt::tt_metal::KernelDescriptor::RTArgList relay_rt_args;
        relay_rt_args.reserve(relay_rt_raw.size());
        relay_rt_args.push_back(output_tensor.buffer());
        for (size_t i = 1; i < relay_rt_raw.size(); ++i) {
            relay_rt_args.push_back(relay_rt_raw[i]);
        }
        desc.kernels[relay_writer_kernel_id].emplace_runtime_args(relay_core, relay_rt_args);
    }
#endif

    // Set runtime args for untilizer cores (both layouts — reader_untilize kernel).
    // Layout: counter_ready_sem, dispatched_buffer_addr, expert_start, expert_end,
    //         dispatched_metadata_addr.
    // Sender NOC coords and per-sender data_ready/start semaphores are now consumed by
    // writer_untilize on the same core (which owns the untilized-data send).  The compute
    // kernel exists only in TILE_LAYOUT, so its runtime args are set under that guard below.
    {
        for (uint32_t j = 0; j < num_untilizer_cores; j++) {
            uint32_t s = untilizer_sender_map[j];
            uint32_t k_s = static_cast<uint32_t>(sender_untilizer_groups[s].size());
            // Full expert range on every untilizer core; the per-expert work is split across
            // senders by data — sender s's group handles batch-chunk s of each expert (sender_idx /
            // num_senders below), then round-robins those batches across its k_s untilizer cores.
            uint32_t expert_start = 0;
            uint32_t expert_end = operation_attributes.experts_per_chip;
            // local_core_id: this untilizer's index within sender s's group, found by counting prior
            // untilizer_idxs that map to the same sender (untilizer_row_cores is grouped by sender).
            uint32_t local_core_id = 0;
            for (uint32_t k = 0; k < j; k++) {
                if (untilizer_sender_map[k] == s) {
                    local_core_id++;
                }
            }
            // Reader_untilize RT args (7):
            //   [0]: counter_ready_sem, [1]: dispatched_buffer*, [2]: expert_start,
            //   [3]: expert_end,        [4]: dispatched_metadata*, [5]: sender_idx, [6]: num_senders.
            // Buffers pushed as Buffer* so the framework records BufferBindings for the
            // cache-hit fast path.
            tt::tt_metal::KernelDescriptor::RTArgList untilizer_rt_args;
            untilizer_rt_args.push_back(counter_ready_semaphore_id);
            untilizer_rt_args.push_back(dispatched_buffer.buffer());
            untilizer_rt_args.push_back(expert_start);
            untilizer_rt_args.push_back(expert_end);
            untilizer_rt_args.push_back(dispatched_metadata.buffer());
            untilizer_rt_args.push_back(untilizer_global_pos[j]);  // global batch start
            untilizer_rt_args.push_back(num_untilizer_cores);      // global batch stride (G)
            desc.kernels[reader_untilize_kernel_ids[j]].emplace_runtime_args(
                untilizer_row_cores[j], untilizer_rt_args);

            // Compute kernel walks the same expert/batch iteration as reader_untilize and
            // writer_untilize (no per-batch signal CB).  Per-sender k_s + local_core_id drive
            // round-robin batch assignment within the group; sender_idx / num_senders select this
            // group's data chunk within each expert.  TILE_LAYOUT only — ROW_MAJOR has no compute
            // kernel (reader_untilize writes rows straight into c_2).
            if (is_tile_layout) {
                tt::tt_metal::KernelDescriptor::RTArgList compute_rt_args;
                compute_rt_args.push_back(expert_start);
                compute_rt_args.push_back(expert_end);
                compute_rt_args.push_back(local_core_id);
                compute_rt_args.push_back(k_s);
                compute_rt_args.push_back(untilizer_global_pos[j]);  // global batch start
                compute_rt_args.push_back(num_untilizer_cores);      // global batch stride (G)
                desc.kernels[untilize_compute_kernel_id].emplace_runtime_args(untilizer_row_cores[j], compute_rt_args);
            }
        }
    }

    // [debug] Build this device's NoC-connectivity descriptors for the host-side dump (ttnn ...
    // dump_combine_connectivity). The dump is role-agnostic: we just hand it one CoreDesc per core we placed,
    // describing the on-device NoC write each makes to the next core (and, for eth cores, the fabric-cable far
    // end). Ids are unique within this device; every downstream_id refers to a core in this same list.
    {
        std::vector<CoreDesc> cores;
        int32_t next_id = 0;
        // NoC each tensix writer uses (writer_untilize / writer_combine): preferred_noc_for_dram_write. Under
        // USE_RELAY the relay ('R') cores are appended below with their own NoC (NOC_0, the write-to-eth NOC).
        const int32_t worker_noc =
            static_cast<int32_t>(tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()));

        // Combine-axis neighbors (fabric nodes), excluding self, in the same order as the fabric-connection loop.
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
        for (const auto& neighbor_coordinate : neighbors) {
            if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                continue;
            }
            dst_nodes.push_back(mesh_device->get_fabric_node_id(neighbor_coordinate));
        }

        // --- Eth ('E') cores: one per (combine neighbor, routing plane). get_link_eth_info(src, dst, link)
        // selects the eth channel candidate_eth_chans[link] whose routing plane IS that index -- so the combine's
        // "link" and the fabric routing plane are the SAME number, and we key everything by plane. Coord, plane
        // and the eth->eth forwarding NoC come from the fabric; the fabric-cable far end is that neighbor. ---
        std::map<uint32_t, std::vector<size_t>> eth_idx_by_plane;  // plane (== link) -> indices into `cores`
        for (size_t ni = 0; ni < dst_nodes.size(); ++ni) {
            for (uint32_t plane = 0; plane < num_links; ++plane) {
                try {
                    const auto info = tt::tt_fabric::get_link_eth_info(src_fabric_node_id, dst_nodes[ni], plane);
                    CoreDesc e;
                    e.type = 'E';
                    e.core_type = tt::CoreType::ETH;
                    e.id = next_id++;
                    e.coord = info.eth_core_logical;  // LOGICAL: matches the [rxlog]/[txlog] file naming
                    e.noc0_physical = info.eth_core_noc0;  // fabric-supplied PHYSICAL NOC0 (eth can't be translated)
                    e.downstream_noc = static_cast<int32_t>(info.forwarding_noc);  // fabric-supplied eth->eth NoC
                    e.fabric_dst_mesh = static_cast<int32_t>(*dst_nodes[ni].mesh_id);
                    e.fabric_dst_dev = static_cast<int32_t>(dst_nodes[ni].chip_id);
                    e.routing_plane = static_cast<int32_t>(info.routing_plane);  // == plane
                    eth_idx_by_plane[info.routing_plane].push_back(cores.size());
                    cores.push_back(e);
                } catch (...) {
                    // No forwarding direction / fewer eth channels than links -> that plane is unavailable; skip.
                }
            }
        }
        // eth->eth on-device NoC forwarding: within a plane the two direction-eths forward to each other (the
        // device receives from one combine neighbor and forwards over NoC to the eth facing the other). For
        // exactly two eths on a plane, point each at the other; a line end (one eth) has no on-device forward.
        for (auto& [plane, idxs] : eth_idx_by_plane) {
            if (idxs.size() == 2) {
                cores[idxs[0]].downstream_ids = {cores[idxs[1]].id};
                cores[idxs[1]].downstream_ids = {cores[idxs[0]].id};
            }
        }

        // --- Relay ('R') cores (USE_RELAY only): the fabric endpoint. relay s connects on plane (s % num_links)
        // and fans out to the eth toward EACH combine neighbor on that plane, writing to eth over NOC_0. The
        // sender ('S') no longer touches eth -- it writes tokens to its relay, which forwards them to fabric. ---
        [[maybe_unused]] std::vector<int32_t> relay_id(relay_cores.size(), -1);
#if USE_RELAY
        for (uint32_t r = 0; r < relay_cores.size(); ++r) {
            CoreDesc rd;
            rd.type = 'R';
            rd.id = next_id++;
            rd.coord = relay_cores[r];
            const uint32_t plane = num_links > 0 ? r % num_links : 0;
            auto it = eth_idx_by_plane.find(plane);
            if (it != eth_idx_by_plane.end()) {
                for (size_t idx : it->second) {
                    rd.downstream_ids.push_back(cores[idx].id);
                }
            }
            rd.downstream_noc = 0;  // relay->eth on NOC_0 (writer_relay is the single kernel on its core)
            relay_id[r] = rd.id;
            cores.push_back(rd);
        }
#endif

        // --- Sender ('S') cores: without a relay the sender connects on plane (core_idx % num_links) and fans
        // out to the eth toward EACH combine neighbor on that plane. Under USE_RELAY it instead writes to its
        // single owning relay (which is the fabric endpoint). Either way its NoC is worker_noc. ---
        std::vector<int32_t> sender_id(sender_cores.size(), -1);
        for (uint32_t s = 0; s < sender_cores.size(); ++s) {
            CoreDesc sd;
            sd.type = 'S';
            sd.id = next_id++;
            sd.coord = sender_cores[s];
#if USE_RELAY
            if (s < relay_id.size() && relay_id[s] >= 0) {
                sd.downstream_ids.push_back(relay_id[s]);
            }
#else
            const uint32_t plane = num_links > 0 ? s % num_links : 0;
            auto it = eth_idx_by_plane.find(plane);
            if (it != eth_idx_by_plane.end()) {
                for (size_t idx : it->second) {
                    sd.downstream_ids.push_back(cores[idx].id);
                }
            }
#endif
            sd.downstream_noc = worker_noc;
            sender_id[s] = sd.id;
            cores.push_back(sd);
        }

        // --- Untilizer ('U') cores: each writes to the single sender it feeds (untilizer_sender_map). ---
        for (uint32_t u = 0; u < all_untilizer_cores.size(); ++u) {
            CoreDesc ud;
            ud.type = 'U';
            ud.id = next_id++;
            ud.coord = all_untilizer_cores[u];
            const uint32_t s = u < untilizer_sender_map.size() ? untilizer_sender_map[u] : 0;
            if (s < sender_id.size() && sender_id[s] >= 0) {
                ud.downstream_ids = {sender_id[s]};
            }
            ud.downstream_noc = worker_noc;
            cores.push_back(ud);
        }

        record_combine_connectivity(mesh_device, src_fabric_node_id, std::move(cores));
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
