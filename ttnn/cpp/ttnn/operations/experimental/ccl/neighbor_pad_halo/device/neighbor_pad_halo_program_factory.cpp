// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_halo_program_factory.hpp"

// NP fabric includes
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>  // FabricMuxConfig, get_tt_fabric_channel_buffer_size_bytes
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <tt-metalium/hal.hpp>

#include <optional>
#include <ranges>
#include <sstream>
#include <type_traits>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// ============================================================================
// create_mesh_workload — iterate over mesh coords and call create_at()
// ============================================================================
NpHaloMeshWorkloadFactory::cached_mesh_workload_t NpHaloMeshWorkloadFactory::create_mesh_workload(
    const NpHaloParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const NpHaloInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Synchronize before dispatching programs.
    auto* mesh_device = tensor_args.input_tensor.device();
    {
        tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    }

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = create_at(operation_attributes, mesh_coord, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = cached_program.shared_variables;
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

// ============================================================================
// create_at — builds the H-fabric and W-fabric halo-exchange kernels (and, in
//             padded-output mode, the concurrent interior-copy scatter).
// ============================================================================
NpHaloMeshWorkloadFactory::cached_program_t NpHaloMeshWorkloadFactory::create_at(
    const NpHaloParams& op,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const NpHaloInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)tensor_return_value;  // output IS the pre-allocated tensor_args.halo_buffer
    auto* mesh_device = tensor_args.input_tensor.device();

    Program program{};

    // Padded-output fused mode: mux worker/mux core ranges (defined in the mux branches below, in cols>=1)
    // that the concurrent interior-copy scatter must avoid. Empty in non-mux paths (subtract is a no-op).
    CoreRangeSet occ_hw_workers, occ_hmux, occ_w_workers, occ_wmux;

    // =========================================================================
    // PART 1: H-DIM FABRIC HALO EXCHANGE (H is index 2 in BTHWC)
    // =========================================================================

    // Use MeshCoordinates to find forward and backward devices along H axis
    uint32_t device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, op.np_cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, ttnn::ccl::Topology::Linear, op.np_cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1, ttnn::ccl::Topology::Linear, op.np_cluster_axis);

    // Tensor info
    const auto& input_tensor_shape = tensor_args.input_tensor.padded_shape();
    Buffer* input_buffer = tensor_args.input_tensor.buffer();
    Buffer* halo_buffer = tensor_args.halo_buffer.buffer();

    // H-dim is index 2 in BTHWC layout (B=0, T=1, H=2, W=3, C=4)
    constexpr uint32_t np_dim = 2;
    uint32_t page_size = input_buffer->aligned_page_size();

    // num_sticks_per_halo_dim = W_dev (product of dims between H and C)
    uint32_t num_sticks_per_halo_dim = 1;
    for (size_t d = np_dim + 1; d < input_tensor_shape.size() - 1; d++) {
        num_sticks_per_halo_dim *= input_tensor_shape[d];
    }
    uint32_t input_halo_dim_size = input_tensor_shape[np_dim];  // H_dev

    // Padded-input mode: the input tensor is [.., H+2pH, W+2pW, C], so the values derived from its shape
    // above are the PADDED W/H. Keep them as the reader's input row-stride / frame-rows (for the strided
    // interior read), but reduce the halo-geometry dims to the INTERIOR (H_dev, W_dev) — the compact
    // buffer, writer, and W-section sizing are all about the interior halo, unchanged from contiguous input.
    const bool padded_input = op.input_pad_h > 0 || op.input_pad_w > 0;
    const uint32_t reader_row_stride = num_sticks_per_halo_dim;  // padded W (input tensor W)
    const uint32_t reader_frame_rows = input_halo_dim_size;      // padded H (input tensor H)
    if (padded_input) {
        num_sticks_per_halo_dim -= 2 * op.input_pad_w;  // -> interior W_dev
        input_halo_dim_size -= 2 * op.input_pad_h;      // -> interior H_dev
    }

    // Total H-halo rows the compact buffer holds per frame (top + bottom).
    uint32_t output_halo_dim_size = op.np_padding_h + op.np_padding_h;

    // outer_dim_size = B * T (all dims before H)
    uint32_t outer_dim_size = 1;
    for (size_t d = 0; d < np_dim; d++) {
        outer_dim_size *= input_tensor_shape[d];
    }

    bool is_first_device = !backward_coord.has_value();
    bool is_last_device = !forward_coord.has_value();
    bool is_padding_zeros = op.padding_mode == "zeros";
    // This op is always 2D (H+W padding) — validate() enforces np_pad_dim2.has_value(). is_2d is kept
    // as a named constant for readability where the W-phase setup is gated.
    TT_FATAL(op.np_pad_dim2.has_value(), "neighbor_pad_halo requires 2D padding (H+W).");
    const bool is_2d = true;

    // Fabric-mux buffers per channel. 4 is the measured sweet spot on BH-LB 2x4 (+~3pp vs 1; 8 regresses
    // on L1 pressure). PCC-invariant (channel buffering doesn't touch data). TT_NP_NUM_BUFFERS overrides.
    const uint8_t np_num_buffers = std::getenv("TT_NP_NUM_BUFFERS")
                                       ? static_cast<uint8_t>(std::max(1, atoi(std::getenv("TT_NP_NUM_BUFFERS"))))
                                       : 4;

    // H corner-first (PCC-neutral): the H-writer sends the W-boundary corner sticks to the neighbor's L1
    // recv buffer + raises the recv sem BEFORE the bulk middle row, so the neighbor's H recv-wait clears
    // after ~2 sticks instead of the full row. Requires padding==1 (the kernel's corner-first path).
    const uint32_t use_corner_first = (op.np_padding_h == 1) ? 1u : 0u;

    // Compact H-section rows are exactly W_dev wide: W-padding lives in a separate W-section, so
    // H rows carry no extra columns and the H and W sections stay independent.
    uint32_t output_num_sticks_per_halo_dim = num_sticks_per_halo_dim;
    uint32_t writer_stick_start_id = 0;
    uint32_t writer_num_sticks_to_read = num_sticks_per_halo_dim;

    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t num_links = static_cast<uint32_t>(op.np_num_links);
    uint32_t pad2_num_links = static_cast<uint32_t>(op.np_pad2_num_links);

    // H->W ordering is an upfront barrier (the W-reader waits the H-writers' Phase-2 barrier signal —
    // see np_phase2_w_reader), not per-batch progress signalling.

    // H-send bank-major coalescing. The halo-only op's upfront H->W barrier makes the corner-first L1
    // path unnecessary, so an eligible H exchange uses the simpler straight-to-DRAM path
    // (use_l1_intermediate=0, whole H-halo row -> neighbor H-section DRAM; corners land in DRAM where the
    // W-reader reads them) AND ships each row bank-major coalesced (h_coalesce_n same-bank sticks per 4KB
    // packet). Eligible when padding_h==1: the row's w=j,j+8,... land same-bank/next-offset at
    // base_row+j (valid for any base_row / W_dev, reader gathers in the same order). BH: 8 DRAM banks.
    // Coalesce a bank's sticks into one fabric packet up to the fabric's actual max payload (not a fixed
    // 4KB): default BH is 4352 B, and an 8K FabricRouterConfig raises it so each packet carries 2x the
    // sticks. Cap the stick count so the send CBs stay bounded.
    const uint32_t fabric_max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    const uint32_t max_coalesce_sticks = std::max(1u, std::min(64u, fabric_max_payload / page_size));
    // Zeros only: the bank-major coalesced path handles the interior + zero fill, not replicate's
    // edge-outward slice replication, so replicate stays on the per-stick direct path.
    uint32_t h_coalesce_n = 0;
    // The bank-major gather assumes each of the 8 DRAM banks receives the same number of row sticks; when
    // W_dev is not a multiple of 8 the trailing banks get fewer, and the writer's un-scatter mis-places
    // them, scrambling the H-halo row. Keep coalescing only for 8-aligned W_dev; else use the direct path.
    constexpr uint32_t np_num_dram_banks = 8;  // BH DRAM bank count (matches NP_NUM_DRAM_BANKS in kernels)
    if (op.np_padding_h == 1 && is_padding_zeros && (num_sticks_per_halo_dim % np_num_dram_banks == 0)) {
        h_coalesce_n = max_coalesce_sticks;
        if (h_coalesce_n < 2) {
            h_coalesce_n = 0;
        }
    }
    const uint32_t h_use_l1 = (h_coalesce_n > 0) ? 0u : 1u;  // straight-to-DRAM when coalescing

    constexpr uint32_t MAX_PAD2_NUM_LINKS = 4;
    uint32_t total_fabric_cores = (num_links * 2) + (pad2_num_links * 2);
    // Fabric cores live in the first column (y-axis), so bound against compute_grid_size.y.
    if (total_fabric_cores > compute_grid_size.y) {
        uint32_t max_total = compute_grid_size.y;
        uint32_t h_cores = num_links * 2;
        uint32_t available_for_w = (max_total > h_cores) ? (max_total - h_cores) : 0;
        pad2_num_links = available_for_w / 2;
        if (pad2_num_links == 0) {
            pad2_num_links = 1;
            num_links = (max_total - 2) / 2;
        }
    }

    uint32_t num_h_fabric_cores = num_links * 2;
    uint32_t num_w_fabric_cores = pad2_num_links * 2;
    TT_FATAL(
        pad2_num_links <= MAX_PAD2_NUM_LINKS,
        "pad2_num_links ({}) exceeds maximum supported ({})",
        pad2_num_links,
        MAX_PAD2_NUM_LINKS);

    // Fabric-mux worker count, auto-selected by W data-moved per (link,direction) like all_gather's
    // default_workers: one worker caps at ~1.4 GB/s/link; multiple workers feeding the eth link through
    // the mux reach the ~12.5 GB/s Linear ceiling. all_gather uses >256KB=>4, 4KB..256KB=>2, else 1.
    // Capped at 2 here: the mux + 8-aligned split is validated at 2 workers (12.8 GB/s, PCC-exact);
    // 4-worker mux is not yet validated. Zeros-only: the replicate-mode edge-outward mux path is unfixed,
    // so replicate stays on the direct single-worker path. TT_NP_W_WORKERS overrides (matches all_gather's
    // optional num_workers_per_direction).
    uint32_t num_w_workers = 1;
    {
        const uint32_t w_h_total_est = input_halo_dim_size + 2 * op.np_padding_h;
        const uint64_t w_rows_est = static_cast<uint64_t>(outer_dim_size) * w_h_total_est;  // W interior rows/dir
        const uint64_t w_bytes_per_link_dir =
            (pad2_num_links > 0) ? (w_rows_est * op.np_pad2_left * page_size / pad2_num_links) : 0;
        uint32_t heuristic = 1;
        if (w_bytes_per_link_dir > 256u * 1024u) {
            heuristic = 4;
        } else if (w_bytes_per_link_dir > 4u * 1024u) {
            heuristic = 2;
        }
        // Auto-engage the W mux from the per-link byte heuristic, zeros only: the mux path does not do
        // replicate's edge-outward slice replication, so replicate stays on the direct per-stick path.
        // TT_NP_W_WORKERS overrides for tuning.
        if (is_padding_zeros) {
            num_w_workers = heuristic;
        }
        if (const char* e = std::getenv("TT_NP_W_WORKERS")) {
            num_w_workers = std::max(1, atoi(e));
        }
        // Clamp to available 8-row bank-units per link so no worker gets 0 rows (the 8-aligned split
        // gives worker0 zero rows when rows_per_link < 8*num_w_workers -> degenerate/incorrect).
        const uint32_t w_units_per_link = (pad2_num_links > 0) ? (w_rows_est / pad2_num_links / 8u) : 0u;
        num_w_workers = std::max(1u, std::min<uint32_t>(num_w_workers, std::max(1u, w_units_per_link)));
        log_debug(
            tt::LogOp,
            "np_halo mux: W bytes/(link,dir)={}, heuristic={}, num_w_workers={}",
            w_bytes_per_link_dir,
            heuristic,
            num_w_workers);
    }

    // H fabric cores occupy column 0, rows [0, num_h_fabric_cores). W fabric cores
    // follow in the same column, leaving cols [1, grid.x) as a clean rectangular grid
    // for the interior-copy scatter — instead of the L-shape a first-row layout would produce.
    CoreCoord np_core_grid(1, num_h_fabric_cores);
    auto
        [num_np_cores,
         np_worker_core_ranges,
         np_core_group_1,
         np_core_group_2,
         np_dims_per_core_group_1,
         np_dims_per_core_group_2] = split_work_to_cores(np_core_grid, outer_dim_size * 2);

    // L1 scratch CB for NP fabric transfer
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_sticks_to_write_per_packet = 1;
    uint32_t np_cb_num_pages = 2 * num_sticks_to_write_per_packet;
    tt::DataFormat df = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    uint32_t sender_cb_index = tt::CB::c_in0;
    CircularBufferConfig cb_sender_config =
        CircularBufferConfig(np_cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
            .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, np_worker_core_ranges, cb_sender_config);

    // Dedicated H-send CB: the H-reader batches a full halo row (num_sticks_per_halo_dim sticks) per
    // cb_reserve so the row reads coalesce into one barrier (the per-stick path was latency-bound,
    // ~18k read+barrier pairs). Sized exactly 2 rows so a row reserve never wraps mid-batch (double
    // buffered: reader fills row N+1 while the H-writer drains row N per stick). Kept separate from
    // the sender CB because the per-stick recv/is_first pushes there would desync the row ring.
    // H cores only; W keeps the per-stick c_in0 path.
    uint32_t hsend_cb_index = tt::CB::c_in2;
    uint32_t hsend_cb_num_pages = 2 * num_sticks_per_halo_dim;
    CircularBufferConfig cb_hsend_config =
        CircularBufferConfig(hsend_cb_num_pages * l1_scratch_cb_page_size_bytes, {{hsend_cb_index, df}})
            .set_page_size(hsend_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, np_worker_core_ranges, cb_hsend_config);

    // L1 receive buffer for 2D padding: fabric-delivered H halo corner sticks arrive here.
    // Corners-only optimization: only W-boundary sticks (pad2_left + pad2_right per row) go
    // to L1; non-corner sticks go directly to neighbor DRAM via fabric.
    // Buffer must hold ALL outer_dims' corner sticks (no per-outer_dim reuse) because the
    // fabric pipeline can deliver data for outer_dim N+1 before the reader finishes
    // copying outer_dim N.
    uint32_t recv_cb_index = tt::CB::c_in1;
    uint32_t corner_sticks_per_row = std::min(op.np_pad2_left + op.np_pad2_right, num_sticks_per_halo_dim);
    if (is_2d) {
        uint32_t max_padding = op.np_padding_h;  // symmetric H padding; matches main's max(left,right)
        uint32_t max_outer_dims_per_core = np_dims_per_core_group_1;
        uint32_t recv_total_sticks = max_outer_dims_per_core * max_padding * corner_sticks_per_row;
        uint32_t recv_buf_size = recv_total_sticks * page_size;
        if (recv_buf_size > 0) {
            CircularBufferConfig recv_cb_config =
                CircularBufferConfig(recv_buf_size, {{recv_cb_index, df}}).set_page_size(recv_cb_index, page_size);
            CreateCircularBuffer(program, np_worker_core_ranges, recv_cb_config);
        }
    }

    // Phase 2 W-axis setup (for 2D padding)
    std::vector<CoreCoord> w_fabric_logical_cores;
    std::vector<CoreCoord> w_fabric_virtual_cores;
    CoreRangeSet w_fabric_core_range;
    bool is_first_w_device = true;
    bool is_last_w_device = true;
    uint32_t w_forward_device_offset = 0;
    uint32_t w_backward_device_offset = 0;
    std::optional<MeshCoordinate> w_forward_coord;
    std::optional<MeshCoordinate> w_backward_coord;
    uint32_t w_outer_dim_size = 0;
    uint32_t w_rows_per_link = 0;
    uint32_t w_extra_rows = 0;
    uint32_t w_section_wleft_base = 0;
    uint32_t w_section_wright_base = 0;
    uint32_t w_coalesce_n = 0;  // W-send bank-major coalesce factor (0 = per-stick); set in the is_2d block

    if (is_2d) {
        w_forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, 1, ttnn::ccl::Topology::Linear, op.np_pad2_cluster_axis);
        w_backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, -1, ttnn::ccl::Topology::Linear, op.np_pad2_cluster_axis);

        is_first_w_device = !w_backward_coord.has_value();
        is_last_w_device = !w_forward_coord.has_value();
        if (w_forward_coord.has_value()) {
            w_forward_device_offset = 1;
        }
        if (w_backward_coord.has_value()) {
            w_backward_device_offset = 1;
        }

        for (uint32_t i = 0; i < num_w_fabric_cores; i++) {
            CoreCoord wc = {0, num_h_fabric_cores + i};
            w_fabric_logical_cores.push_back(wc);
            w_fabric_virtual_cores.push_back(mesh_device->worker_core_from_logical_core(wc));
        }
        w_fabric_core_range =
            CoreRangeSet(CoreRange({0, num_h_fabric_cores}, {0, num_h_fabric_cores + num_w_fabric_cores - 1}));

        // W exchange covers all H_dev + 2*ph rows per T (the H-halo is already in place by then).
        uint32_t h_total = input_halo_dim_size + 2 * op.np_padding_h;
        w_outer_dim_size = outer_dim_size * h_total;

        // W-section base offsets in the compact halo buffer:
        // Layout: [H-top | H-bot | W-left | W-right]
        uint32_t h_section_sticks = outer_dim_size * 2 * op.np_padding_h * num_sticks_per_halo_dim;
        w_section_wleft_base = h_section_sticks;
        w_section_wright_base = w_section_wleft_base + outer_dim_size * op.np_pad2_left * h_total;

        w_rows_per_link = w_outer_dim_size / pad2_num_links;
        w_extra_rows = w_outer_dim_size % pad2_num_links;

        // W-send bank-major coalescing (BH: 8 interleaved DRAM banks). Sticks base+r, base+r+8, ...,
        // base+r+8*(N-1) land on bank (base+r)%8 at consecutive page offsets, so N ship as ONE N*page
        // fabric write to get_noc_addr(base+r) — valid for ANY base (the N sticks differ by 8, hence same
        // bank / next offset), and the reader gathers rel=r,r+8,... in the same order the writer sends. So
        // only pw==1 is required; base/row-count alignment is NOT needed.
        // Zeros only: the coalesced reader/writer don't do replicate's edge-outward slice replication, so
        // replicate stays on the per-stick direct path (matches h_coalesce_n).
        const bool w_coalesce_ok = (op.np_pad2_left == 1) && (op.np_pad2_right == 1) && is_padding_zeros;
        if (w_coalesce_ok) {
            w_coalesce_n = max_coalesce_sticks;  // sticks per fabric-max-payload packet
            if (w_coalesce_n < 2) {
                w_coalesce_n = 0;
            }
        }
        // W send CB (c_in0 on W cores): deep enough to double-buffer a coalesce group (per-stick uses 2).
        uint32_t w_cb_num_pages = (w_coalesce_n > 0) ? (2 * w_coalesce_n) : np_cb_num_pages;
        CircularBufferConfig cb_w_sender_config =
            CircularBufferConfig(w_cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
                .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
        CreateCircularBuffer(program, w_fabric_core_range, cb_w_sender_config);
    }

    // Mux gating + W worker/mux core coords, computed HERE (before the H-writer setup) so the H->W
    // barrier signal can target the relocated mux W-reader cores. TT_NP_W_MUX forces the mux path at any
    // worker count (bring-up).
    const bool w_force_mux = std::getenv("TT_NP_W_MUX") != nullptr;
    // Uniform mux across ALL W devices (edges too) so the recv-sem targeting is consistent along the whole
    // W chain (mixed mux/standard breaks edge<->middle recv). Edge devices' no-send direction is handled
    // by has_neighbor/has_send_neighbor gating in the kernels + no mux kernel for that (link,dir).
    const bool use_w_mux = is_2d && ((num_w_workers > 1) || w_force_mux) && (w_coalesce_n > 0);
    std::vector<CoreCoord> mux_worker_logical, mux_worker_virtual, mux_core_logical;
    if (use_w_mux) {
        for (uint32_t s = 0; s < pad2_num_links * 2; s++) {
            mux_core_logical.push_back(CoreCoord{1, s});
            for (uint32_t wk = 0; wk < num_w_workers; wk++) {
                CoreCoord wc{2 + wk, s};
                mux_worker_logical.push_back(wc);
                mux_worker_virtual.push_back(mesh_device->worker_core_from_logical_core(wc));
            }
        }
    }

    // Compute H fabric unicast and multicast route configurations
    auto [h_unicast_forward_args, h_unicast_backward_args] =
        ::ttnn::ccl::get_forward_backward_line_unicast_configuration(
            mesh_coordinate, forward_coord, backward_coord, mesh_device);

    auto [num_targets_forward, num_targets_backward] =
        ::ttnn::ccl::get_forward_backward_line_mcast_distance(op.np_ring_size, device_index, op.np_topology, false);
    auto [h_mcast_forward_args, h_mcast_backward_args] = ::ttnn::ccl::get_forward_backward_line_mcast_configuration(
        mesh_coordinate, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    uint32_t num_directions = 2;

    // -------------------------------------------------------------------------
    // PART 2: no conv consumer. There are no progress-sem signal targets, so reader_noc_coords stays
    // empty (the H/W writers loop over 0 targets → no per-batch signalling).
    // -------------------------------------------------------------------------
    std::vector<std::pair<uint32_t, uint32_t>> reader_noc_coords;

    // H-mux: mirror the W-mux (N workers per (link,dir) feed the H-axis eth link through a mux). Gated on
    // TT_NP_H_MUX during bring-up; auto num_h_workers by heuristic (capped 2), zeros-only. H cores placed
    // in columns after the W-mux block.
    // Auto-engage like W-mux: 2 H workers for zeros+coalesce (validated 8/8 PCC). Replicate stays on the
    // direct path (pre-existing W-mux edge-outward bug). TT_NP_H_MUX forces on; TT_NP_H_WORKERS overrides.
    const bool h_force_mux = std::getenv("TT_NP_H_MUX") != nullptr;
    // H bytes per (link,dir): 2 workers only when large enough to be bandwidth-bound (mirrors W's 4KB
    // gate) and each worker gets >=1 frame — tiny shapes stay on the correct direct path.
    const uint64_t h_bytes_per_link = (num_links > 0) ? (static_cast<uint64_t>(outer_dim_size) * op.np_padding_h *
                                                         num_sticks_per_halo_dim * page_size / num_links)
                                                      : 0;
    const uint32_t h_frames_per_link = (num_links > 0) ? (outer_dim_size / num_links) : 0;
    // Auto-engage H workers by per-(link,dir) bytes, mirroring the W heuristic: 4 for the large,
    // bandwidth-bound shapes (>256KB/link) — measured to cut the s4-class wall (H exchange is on the op's
    // critical path, e.g. s4 469us at H4 vs 560us at H2) — and 2 for the >4KB mid band; tiny shapes stay on
    // the direct path. Requires >=1 frame/worker. Zeros only (replicate uses the direct path, like W).
    // W4+H4 is PCC byte-exact on all 2x4 prod shapes. TT_NP_H_WORKERS overrides for tuning.
    uint32_t num_h_workers = 1;
    if (is_padding_zeros && (h_frames_per_link >= 2u)) {
        if (h_bytes_per_link > 256u * 1024u) {
            num_h_workers = 4;
        } else if (h_bytes_per_link > 4u * 1024u) {
            num_h_workers = 2;
        }
    }
    if (const char* e = std::getenv("TT_NP_H_WORKERS")) {
        num_h_workers = std::max(1, atoi(e));
    }
    const bool use_h_mux = is_2d && (h_coalesce_n > 0) && (num_h_workers > 1 || h_force_mux);
    std::vector<CoreCoord> hmux_worker_logical, hmux_worker_virtual, hmux_core_logical;
    if (use_h_mux) {
        const uint32_t h_mux_col = use_w_mux ? (2u + num_w_workers) : 1u;
        for (uint32_t s = 0; s < num_links * num_directions; s++) {
            hmux_core_logical.push_back(CoreCoord{h_mux_col, s});
            for (uint32_t wk = 0; wk < num_h_workers; wk++) {
                CoreCoord wc{h_mux_col + 1 + wk, s};
                hmux_worker_logical.push_back(wc);
                hmux_worker_virtual.push_back(mesh_device->worker_core_from_logical_core(wc));
            }
        }
    }
    KernelHandle h_reader_kernel_id = 0;
    KernelHandle h_writer_kernel_id = 0;

    if (!use_h_mux) {
        // -------------------------------------------------------------------------
        // NP H-fabric reader kernel
        // -------------------------------------------------------------------------
        auto h_reader_kernel_config = ReaderDataMovementConfig{};
        h_reader_kernel_config.compile_args = {
            sender_cb_index,   // cb_output_id
            is_padding_zeros,  // is_padding_zeros
            page_size};        // stick_size
        TensorAccessorArgs(*input_buffer).append_to(h_reader_kernel_config.compile_args);
        h_reader_kernel_config.compile_args.push_back(h_use_l1);        // use_l1_intermediate (0 when H-coalescing)
        h_reader_kernel_config.compile_args.push_back(recv_cb_index);   // recv_cb_id
        h_reader_kernel_config.compile_args.push_back(hsend_cb_index);  // send_cb_id (batched H send)
        h_reader_kernel_config.compile_args.push_back(h_coalesce_n);    // H-send bank-major coalesce factor (0=off)
        h_reader_kernel_config.compile_args.push_back(0);  // H_SIGNAL_W_RECV: direct path, np_writer signals
        h_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/"
            "np_h_reader.cpp",
            np_worker_core_ranges,
            h_reader_kernel_config);
        SetCommonRuntimeArgs(
            program,
            h_reader_kernel_id,
            {input_buffer->address(), halo_buffer->address(), op.h_neighbor_semaphore.address()});

        // -------------------------------------------------------------------------
        // NP H-fabric writer kernel
        // -------------------------------------------------------------------------
        auto h_writer_kernel_config = WriterDataMovementConfig{};
        h_writer_kernel_config.compile_args = {
            sender_cb_index,   // cb_output_id
            is_padding_zeros,  // is_padding_zeros
            page_size};        // stick_size
        TensorAccessorArgs(*halo_buffer).append_to(h_writer_kernel_config.compile_args);
        h_writer_kernel_config.compile_args.push_back(h_use_l1);         // use_l1_intermediate (0 when H-coalescing)
        h_writer_kernel_config.compile_args.push_back(recv_cb_index);    // recv_cb_id
        h_writer_kernel_config.compile_args.push_back(h_use_l1);         // handle_incoming_writes (0 when H-coalescing)
        h_writer_kernel_config.compile_args.push_back(0);                // is_w_fabric_writer (false for H)
        h_writer_kernel_config.compile_args.push_back(op.np_ring_size);  // ring_size
        h_writer_kernel_config.compile_args.push_back(hsend_cb_index);   // send_cb_id (batched H send)
        h_writer_kernel_config.compile_args.push_back(use_corner_first);  // H-writer corner-first gate
        h_writer_kernel_config.compile_args.push_back(h_coalesce_n);  // coalesce factor (H-writer uses the H branch)
        h_writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/"
            "np_writer.cpp",
            np_worker_core_ranges,
            h_writer_kernel_config);
        {
            std::vector<uint32_t> h_writer_crta = {
                input_buffer->address(),
                halo_buffer->address(),
                op.h_neighbor_semaphore.address(),
                op.barrier_semaphore.address(),
                // CRTA[4]: number of per-batch consumer cores to signal (0 in this op)
                static_cast<uint32_t>(reader_noc_coords.size()),
                // CRTA[5+]: interleaved (x, y) NOC coords for each such core
            };
            for (const auto& [x, y] : reader_noc_coords) {
                h_writer_crta.push_back(x);
                h_writer_crta.push_back(y);
            }
            SetCommonRuntimeArgs(program, h_writer_kernel_id, h_writer_crta);
        }

        // Set per-core runtime args for H fabric cores
        uint32_t link_offset_start_id = 0;
        uint32_t writer_link_offset_start_id = 0;
        for (uint32_t link = 0; link < num_links; link++) {
            uint32_t link_dims_to_read = 0;

            for (uint32_t direction = 0; direction < num_directions; direction++) {
                CoreCoord core = {0, link * num_directions + direction};
                CoreCoord opposite_core = {0, (link * num_directions) + (1 - direction)};
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                CoreCoord virtual_opposite_core = mesh_device->worker_core_from_logical_core(opposite_core);
                if (np_core_group_1.contains(core)) {
                    link_dims_to_read = np_dims_per_core_group_1;
                } else {
                    link_dims_to_read = np_dims_per_core_group_2;
                }

                // Reader runtime args. Padded input: frame base/stride use padded dims (reader_frame_rows *
                // reader_row_stride), stick_start skips the pH/pW border, row stride is padded W, but the
                // edge-row count and sticks-to-read stay interior (input_halo_dim_size / num_sticks_per_halo_dim).
                const uint32_t rd_frame_base = padded_input ? (link_offset_start_id / num_sticks_per_halo_dim) *
                                                                  reader_frame_rows * reader_row_stride
                                                            : link_offset_start_id * input_halo_dim_size;
                std::vector<uint32_t> reader_rt_args = {
                    rd_frame_base,  // outer_dim_offset_start_id
                    padded_input ? (op.input_pad_h * reader_row_stride + op.input_pad_w) : 0u,  // stick_start_id
                    input_halo_dim_size,                                         // input_halo_dim_size (interior)
                    link_dims_to_read,                                           // outer_dim_size
                    op.np_padding_h,                                             // padding (symmetric)
                    num_sticks_per_halo_dim,                                     // num_sticks_to_read (interior)
                    padded_input ? reader_row_stride : num_sticks_per_halo_dim,  // num_sticks_per_halo_dim (stride)
                    corner_sticks_per_row,                                       // num_l1_recv_sticks_per_row
                    padded_input ? reader_frame_rows : input_halo_dim_size};     // input_frame_rows (frame stride)
                reader_rt_args.push_back(direction ? is_last_device : is_first_device);  // is_first_chip
                reader_rt_args.push_back(direction ? is_first_device : is_last_device);  // is_last_chip
                reader_rt_args.push_back(direction);                                     // direction
                SetRuntimeArgs(program, h_reader_kernel_id, {core}, reader_rt_args);

                // Writer runtime args
                uint32_t h_writer_num_sticks_per_halo_dim = output_num_sticks_per_halo_dim;
                uint32_t h_writer_stick_start = writer_stick_start_id;
                uint32_t h_writer_num_sticks_to_read = writer_num_sticks_to_read;

                std::vector<uint32_t> writer_rt_args = {
                    writer_link_offset_start_id * output_halo_dim_size,  // outer_dim_offset_start_id
                    h_writer_stick_start,                                // stick_start_id
                    input_halo_dim_size,                                 // input_halo_dim_size
                    output_halo_dim_size,                                // output_halo_dim_size
                    link_dims_to_read,                                   // outer_dim_size
                    op.np_padding_h,                                     // padding (symmetric)
                    op.np_pad2_left,                   // padding_left (W-axis, for L1 corner detection)
                    h_writer_num_sticks_to_read,       // num_sticks_to_read
                    h_writer_num_sticks_per_halo_dim,  // num_sticks_per_halo_dim
                    virtual_core.x,                    // neighbor_sem_noc0_x
                    virtual_core.y,                    // neighbor_sem_noc0_y
                    true,                              // use_barrier_semaphore
                    virtual_opposite_core.x,           // barrier_sem_noc0_x
                    virtual_opposite_core.y};          // barrier_sem_noc0_y
                // Phase 2 signal targets (W fabric reader cores). Mux path: signal the relocated mux
                // worker-reader cores (each waits the H->W barrier) instead of the standard column-0 W cores.
                // Must match MAX_PHASE2_SIGNAL_TARGETS in np_writer.cpp.
                constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 32;
                const std::vector<CoreCoord>& w_sig_cores = use_w_mux ? mux_worker_virtual : w_fabric_virtual_cores;
                const uint32_t n_w_sig = static_cast<uint32_t>(w_sig_cores.size());
                // Every H writer signals every W reader; if this ever exceeds the array the readers past
                // the cap never see the H->W barrier and deadlock, so fail loudly instead.
                TT_FATAL(
                    n_w_sig <= MAX_PHASE2_SIGNAL_TARGETS,
                    "neighbor_pad_halo: {} W reader cores exceeds MAX_PHASE2_SIGNAL_TARGETS ({})",
                    n_w_sig,
                    MAX_PHASE2_SIGNAL_TARGETS);
                writer_rt_args.push_back(n_w_sig);
                for (uint32_t s = 0; s < MAX_PHASE2_SIGNAL_TARGETS; s++) {
                    if (s < n_w_sig) {
                        writer_rt_args.push_back(w_sig_cores[s].x);
                        writer_rt_args.push_back(w_sig_cores[s].y);
                    } else {
                        writer_rt_args.push_back(0);
                        writer_rt_args.push_back(0);
                    }
                }
                writer_rt_args.push_back(direction ? is_last_device : is_first_device);  // is_first_chip
                writer_rt_args.push_back(direction ? is_first_device : is_last_device);  // is_last_chip
                writer_rt_args.push_back(direction);                                     // direction
                // Unicast route args
                const auto& h_unicast_args = direction ? h_unicast_backward_args : h_unicast_forward_args;
                writer_rt_args.insert(writer_rt_args.end(), h_unicast_args.begin(), h_unicast_args.end());
                // Barrier multicast route info
                const auto& h_mcast_args = direction ? h_mcast_backward_args : h_mcast_forward_args;
                writer_rt_args.insert(writer_rt_args.end(), h_mcast_args.begin(), h_mcast_args.end());
                // Fabric connection args
                if (direction) {
                    writer_rt_args.push_back(false);
                    writer_rt_args.push_back(backward_coord.has_value());
                    if (backward_coord.has_value()) {
                        const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                        const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
                    }
                } else {
                    writer_rt_args.push_back(forward_coord.has_value());
                    if (forward_coord.has_value()) {
                        const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                        const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
                    }
                    writer_rt_args.push_back(false);
                }
                // Override the H-writer rt_args to index into the compact halo buffer's H-sections.
                {
                    uint32_t link_t_start = (output_num_sticks_per_halo_dim > 0)
                                                ? (writer_link_offset_start_id / output_num_sticks_per_halo_dim)
                                                : 0u;
                    uint32_t top_halo_total = outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim;
                    uint32_t h_top_link_start = link_t_start * op.np_padding_h * num_sticks_per_halo_dim;
                    uint32_t h_bot_link_start =
                        top_halo_total + link_t_start * op.np_padding_h * num_sticks_per_halo_dim;
                    writer_rt_args[0] = direction ? h_bot_link_start : h_top_link_start;  // per-link offset
                    writer_rt_args[1] = 0;                        // stick_start_id (no W-offset in compact)
                    writer_rt_args[3] = op.np_padding_h;          // output_halo_dim_size (compact)
                    writer_rt_args[8] = num_sticks_per_halo_dim;  // stride = W_dev, not padded W
                }
                // No per-batch progress args (no per-batch consumer).
                SetRuntimeArgs(program, h_writer_kernel_id, {core}, writer_rt_args);
            }
            link_offset_start_id += (link_dims_to_read * num_sticks_per_halo_dim);
            writer_link_offset_start_id += (link_dims_to_read * output_num_sticks_per_halo_dim);
        }
    }  // end if(!use_h_mux)

    if (use_h_mux) {
        using tt::tt_fabric::FabricMuxChannelType;
        const uint32_t l1_base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        const size_t mux_buf_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
        auto hmux_cfg = tt::tt_fabric::FabricMuxConfig(
            static_cast<uint8_t>(num_h_workers), 0, np_num_buffers, 0, mux_buf_size, l1_base);

        const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/";
        std::set<CoreRange> hw_crs, hm_crs;
        for (const auto& c : hmux_worker_logical) {
            hw_crs.insert(CoreRange(c));
        }
        // Mux kernel only for (link,dir) that send (edge outward dirs don't).
        for (uint32_t s = 0; s < hmux_core_logical.size(); s++) {
            const bool sends = (s % num_directions == 0) ? !is_last_device : !is_first_device;
            if (sends) {
                hm_crs.insert(CoreRange(hmux_core_logical[s]));
            }
        }
        CoreRangeSet hw_crset(hw_crs);
        CoreRangeSet hm_crset(hm_crs.empty() ? std::set<CoreRange>{CoreRange({0, 0})} : hm_crs);
        occ_hw_workers = hw_crset;
        occ_hmux = hm_crset;

        // Send CB (hsend) + sender CB (c_in0) on the H worker cores. hsend MUST be a whole multiple of the
        // row size (num_sticks_per_halo_dim): the H reader reserves a full row at once and the writer reads
        // it via get_read_ptr + m*stick, so a row must never straddle the CB wrap. 4 rows for pipelining.
        {
            CircularBufferConfig cb_hs(
                4u * num_sticks_per_halo_dim * l1_scratch_cb_page_size_bytes, {{hsend_cb_index, df}});
            cb_hs.set_page_size(hsend_cb_index, l1_scratch_cb_page_size_bytes);
            CreateCircularBuffer(program, hw_crset, cb_hs);
            CircularBufferConfig cb_s0(np_cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}});
            cb_s0.set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
            CreateCircularBuffer(program, hw_crset, cb_s0);
        }

        // H reader on worker cores (same CT layout as the standard H reader).
        std::vector<uint32_t> hr_ct = {sender_cb_index, is_padding_zeros, page_size};
        TensorAccessorArgs(*input_buffer).append_to(hr_ct);
        hr_ct.push_back(h_use_l1);
        hr_ct.push_back(recv_cb_index);
        hr_ct.push_back(hsend_cb_index);
        hr_ct.push_back(h_coalesce_n);
        hr_ct.push_back(1);  // H_SIGNAL_W_RECV: this reader signals the H->W barrier after its recv drains
        auto hr_cfg = ReaderDataMovementConfig{};
        hr_cfg.compile_args = hr_ct;
        h_reader_kernel_id = CreateKernel(program, kdir + "np_h_reader.cpp", hw_crset, hr_cfg);
        SetCommonRuntimeArgs(
            program,
            h_reader_kernel_id,
            {input_buffer->address(),
             halo_buffer->address(),
             op.h_neighbor_semaphore.address(),
             op.barrier_semaphore.address()});

        // H-mux writer on worker cores. CT: is_padding_zeros, c_in0 (is_first local pad), hsend, stick.
        std::vector<uint32_t> hw_ct = {is_padding_zeros, sender_cb_index, hsend_cb_index, page_size};
        TensorAccessorArgs(*halo_buffer).append_to(hw_ct);
        hw_ct.push_back(h_coalesce_n);
        ttnn::ccl::fabric_mux_connection_ct_args(
            num_h_workers, FabricMuxChannelType::FULL_SIZE_CHANNEL, hmux_cfg, hw_ct);
        auto hw_cfg = WriterDataMovementConfig{};
        hw_cfg.compile_args = hw_ct;
        h_writer_kernel_id = CreateKernel(program, kdir + "np_h_mux_writer.cpp", hw_crset, hw_cfg);
        SetCommonRuntimeArgs(
            program,
            h_writer_kernel_id,
            {input_buffer->address(),
             halo_buffer->address(),
             op.h_neighbor_semaphore.address(),
             op.barrier_semaphore.address()});

        auto hmux_kernel_id = CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
            hm_crset,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = hmux_cfg.get_fabric_mux_compile_time_args(),
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

        // W-reader barrier targets (H workers signal these so the W reader clears its H->W wait).
        const std::vector<CoreCoord>& w_sig = use_w_mux ? mux_worker_virtual : w_fabric_virtual_cores;
        const uint32_t top_halo_total = outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim;
        const uint32_t frames_per_link = outer_dim_size / num_links;
        const uint32_t frames_extra = outer_dim_size % num_links;

        for (uint32_t link = 0; link < num_links; link++) {
            const uint32_t link_f0 = link * frames_per_link + std::min(link, frames_extra);
            const uint32_t link_fc = frames_per_link + (link < frames_extra ? 1u : 0u);
            const uint32_t per_wk = link_fc / num_h_workers;
            const uint32_t extra_wk = link_fc % num_h_workers;
            for (uint32_t dir = 0; dir < num_directions; dir++) {
                const uint32_t s = link * num_directions + dir;
                CoreCoord mux_lc = hmux_core_logical[s];
                CoreCoord mux_vc = mesh_device->worker_core_from_logical_core(mux_lc);
                const bool sends = (dir == 0) ? !is_last_device : !is_first_device;
                if (sends) {
                    const auto src_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                    const auto dst_id =
                        mesh_device->get_fabric_node_id(dir ? backward_coord.value() : forward_coord.value());
                    auto mux_rt = hmux_cfg.get_fabric_mux_run_time_args(src_id, dst_id, link, program, {mux_lc});
                    SetRuntimeArgs(program, hmux_kernel_id, {mux_lc}, mux_rt);
                }
                CoreCoord term_master_vc =
                    mesh_device->worker_core_from_logical_core(hmux_worker_logical[s * num_h_workers + 0]);
                const uint32_t h_section_base = dir ? top_halo_total : 0u;
                const uint32_t h_dev_off = (dir ? backward_coord.has_value() : forward_coord.has_value()) ? 1u : 0u;
                for (uint32_t wk = 0; wk < num_h_workers; wk++) {
                    CoreCoord wc = hmux_worker_logical[s * num_h_workers + wk];
                    CoreCoord wc_vc = hmux_worker_virtual[s * num_h_workers + wk];
                    const uint32_t wk_f0 = link_f0 + wk * per_wk + std::min(wk, extra_wk);
                    const uint32_t wk_fc = per_wk + (wk < extra_wk ? 1u : 0u);
                    // Reader RT args (same layout as the standard H reader). wk_f0 is in FRAME units here.
                    std::vector<uint32_t> r_rt = {
                        padded_input
                            ? wk_f0 * reader_frame_rows * reader_row_stride
                            : wk_f0 * num_sticks_per_halo_dim * input_halo_dim_size,  // outer_dim_offset_start_id
                        padded_input ? (op.input_pad_h * reader_row_stride + op.input_pad_w) : 0u,  // stick_start_id
                        input_halo_dim_size,
                        wk_fc,  // outer_dim_size (frames this worker owns)
                        op.np_padding_h,
                        num_sticks_per_halo_dim,
                        padded_input ? reader_row_stride : num_sticks_per_halo_dim,
                        corner_sticks_per_row,
                        padded_input ? reader_frame_rows : input_halo_dim_size};  // input_frame_rows
                    r_rt.push_back(dir ? is_last_device : is_first_device);
                    r_rt.push_back(dir ? is_first_device : is_last_device);
                    r_rt.push_back(dir);
                    // H->W barrier targets: this reader signals these W-reader cores after its recv drains.
                    // Must match MAX_W_BAR_TARGETS in np_h_reader.cpp; overflow silently strands the W
                    // readers past the cap at the barrier, so fail loudly instead.
                    constexpr uint32_t MAX_W_BARRIER_TARGETS = 32;
                    TT_FATAL(
                        w_sig.size() <= MAX_W_BARRIER_TARGETS,
                        "neighbor_pad_halo: {} W reader cores exceeds MAX_W_BARRIER_TARGETS ({})",
                        w_sig.size(),
                        MAX_W_BARRIER_TARGETS);
                    r_rt.push_back(static_cast<uint32_t>(w_sig.size()));
                    for (uint32_t t = 0; t < MAX_W_BARRIER_TARGETS; t++) {
                        r_rt.push_back(t < w_sig.size() ? w_sig[t].x : 0u);
                        r_rt.push_back(t < w_sig.size() ? w_sig[t].y : 0u);
                    }
                    SetRuntimeArgs(program, h_reader_kernel_id, {wc}, r_rt);
                    // Writer RT args (np_h_mux_writer layout).
                    std::vector<uint32_t> w_rt = {
                        h_section_base + wk_f0 * op.np_padding_h * num_sticks_per_halo_dim,  // h_base
                        wk_fc,                                                               // outer_dim_count
                        op.np_padding_h,
                        num_sticks_per_halo_dim,  // num_sticks_to_read (W_dev)
                        num_sticks_per_halo_dim,  // num_sticks_per_halo_dim
                        wc_vc.x,
                        wc_vc.y,  // recv-sem target = same-dir worker
                        // is_first/is_last direction-adjusted (match np_h_reader + np_writer).
                        static_cast<uint32_t>(dir ? is_last_device : is_first_device),
                        static_cast<uint32_t>(dir ? is_first_device : is_last_device),
                        dir,
                        0u,
                        h_dev_off};  // route mesh, distance-in-hops
                    ttnn::ccl::fabric_mux_connection_rt_args(
                        sends,
                        /*is_termination_master=*/wk == 0,
                        FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        mux_vc,
                        wk,
                        wc,
                        hmux_cfg,
                        program,
                        term_master_vc,
                        w_rt,
                        std::nullopt);
                    SetRuntimeArgs(program, h_writer_kernel_id, {wc}, w_rt);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Fused padded-output border fold: the W-readers write the padded BORDER for their own rows after they
    // have observed the compact data (recv sem + H->W barrier), so the read is visibility-safe (no
    // cross-core fabric race). dir==0 cores write W-left + H-top/H-bot (pad rows); dir==1 write W-right.
    // The interior is written concurrently by the free-core scatter above. common args (SCATTER_BORDER):
    // [3]=padded_addr, [4]=w_section_wleft_base, [5]=w_section_wright_base, [6]=np_pad2_right.
    // -------------------------------------------------------------------------
    const bool scatter_border = op.output_padded && tensor_args.padded_output.has_value() && compute_grid_size.x > 1;
    Buffer* padded_buf_w = scatter_border ? tensor_args.padded_output.value().buffer() : nullptr;
    constexpr uint32_t w_scatter_scratch_cb = tt::CB::c_in1;  // private L1 scratch for the border scatter
    // Logical-mask offsets: a stick is zeroed when its GLOBAL content index (device offset + local index)
    // is >= logical_h/logical_w. Mirrors neighbor_pad_async's fused masking.
    const uint32_t mask_device_h_offset = device_index * input_halo_dim_size;
    const uint32_t mask_device_w_offset =
        op.np_pad2_cluster_axis.has_value()
            ? ::ttnn::ccl::get_linearized_index_from_physical_coord(
                  tensor_args.input_tensor, mesh_coordinate, op.np_pad2_cluster_axis.value()) *
                  num_sticks_per_halo_dim
            : 0u;
    auto w_border_common = [&](std::vector<uint32_t>& crta) {
        if (scatter_border) {
            crta.push_back(padded_buf_w->address());
            crta.push_back(w_section_wleft_base);
            crta.push_back(w_section_wright_base);
            crta.push_back(static_cast<uint32_t>(op.np_pad2_right));
            crta.push_back(op.logical_h);
            crta.push_back(mask_device_h_offset);
            crta.push_back(op.logical_w);
            crta.push_back(mask_device_w_offset);
        }
    };
    auto w_border_scratch_cb = [&](const CoreRangeSet& cores) {
        if (scatter_border) {
            constexpr uint32_t scratch_pages = 16;  // must match BATCH in np_phase2_w_reader border scatter
            CircularBufferConfig cfg(scratch_pages * page_size, {{w_scatter_scratch_cb, df}});
            cfg.set_page_size(w_scatter_scratch_cb, page_size);
            CreateCircularBuffer(program, cores, cfg);
        }
    };

    // -------------------------------------------------------------------------
    // W fabric kernels for 2D padding (Phase 2)
    // -------------------------------------------------------------------------
    KernelHandle w_reader_kernel_id = 0;
    KernelHandle w_writer_kernel_id = 0;
    if (is_2d) {
        const auto& mesh_view_w = mesh_device->get_view();
        uint32_t w_ring_size = (op.np_pad2_cluster_axis.value() == 0) ? mesh_view_w.num_rows() : mesh_view_w.num_cols();
        uint32_t w_device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, op.np_pad2_cluster_axis);
        auto [w_num_targets_forward, w_num_targets_backward] =
            ::ttnn::ccl::get_forward_backward_line_mcast_distance(w_ring_size, w_device_index, op.np_topology, false);
        auto [w_mcast_forward_args, w_mcast_backward_args] = ::ttnn::ccl::get_forward_backward_line_mcast_configuration(
            mesh_coordinate,
            w_forward_coord,
            w_backward_coord,
            w_num_targets_forward,
            w_num_targets_backward,
            mesh_device);

        // ---------------------------------------------------------------------
        // FABRIC-MUX W path (middle W devices, coalesced): N workers per (link,dir) feed a mux core so
        // the eth link is saturated (reach ~12.5 GB/s/link vs ~1.4 single-worker). Gated on num_w_workers
        // > 1 + middle device + coalesce-eligible; else falls through to the standard 1-worker path.
        // Uses the shipping tt_fabric_mux kernel + ccl::fabric_mux_connection_{ct,rt}_args helpers.
        // use_w_mux + mux core coords are computed above (before the H-writer barrier signal setup).
        if (use_w_mux) {
            using tt::tt_fabric::FabricMuxChannelType;
            const uint32_t l1_base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
            const size_t mux_buf_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
            // num_buffers per channel: 4 (np_num_buffers) is the measured 2x4 sweet spot (+~3pp); the mux
            // still fits its full-size channels in L1 at 4. PCC-invariant.
            auto mux_cfg = tt::tt_fabric::FabricMuxConfig(
                /*num_full_size_channels=*/static_cast<uint8_t>(num_w_workers),
                /*num_header_only_channels=*/0,
                /*num_buffers_full_size_channel=*/np_num_buffers,
                /*num_buffers_header_only_channel=*/0,
                mux_buf_size,
                l1_base);

            // Core layout (halo-only op: cols>=1 are free). Per (link,dir): 1 mux core (col 1) + N worker
            // cores (cols 2+). link/dir index = w_link*2 + w_dir.

            // W reader + mux writer kernels on the worker cores.
            std::vector<uint32_t> mux_reader_ct = {sender_cb_index, is_padding_zeros, page_size};
            TensorAccessorArgs(*halo_buffer).append_to(mux_reader_ct);
            TensorAccessorArgs(*input_buffer).append_to(mux_reader_ct);
            mux_reader_ct.push_back(w_coalesce_n);
            mux_reader_ct.push_back(1);  // W_MUX_MODE: coalesce for edge devices too
            auto mux_reader_cfg = ReaderDataMovementConfig{};
            mux_reader_cfg.compile_args = mux_reader_ct;

            std::vector<uint32_t> mux_writer_ct = {sender_cb_index, page_size};
            TensorAccessorArgs(*halo_buffer).append_to(mux_writer_ct);
            mux_writer_ct.push_back(w_coalesce_n);
            ttnn::ccl::fabric_mux_connection_ct_args(
                num_w_workers, FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_cfg, mux_writer_ct);
            auto mux_writer_cfg = WriterDataMovementConfig{};
            mux_writer_cfg.compile_args = mux_writer_ct;

            const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/";
            // Reuse the hoisted core lists (flat: [link*2+dir][worker] for workers, [link*2+dir] for mux).
            const std::vector<CoreCoord>& mux_worker_cores = mux_worker_logical;
            const std::vector<CoreCoord>& mux_mux_cores = mux_core_logical;
            std::set<CoreRange> worker_crs, mux_crs;
            for (const auto& c : mux_worker_cores) {
                worker_crs.insert(CoreRange(c));
            }
            // Only create a mux kernel for (link,dir) that actually SEND (have a send-neighbor); the mux
            // kernel blocks until its worker connects + signals terminate, so an unused mux would hang.
            // dir=0 (forward) sends iff !is_last_w_device; dir=1 (backward) sends iff !is_first_w_device.
            for (uint32_t s = 0; s < mux_mux_cores.size(); s++) {
                const uint32_t w_dir_s = s % 2;
                const bool sends = w_dir_s == 0 ? !is_last_w_device : !is_first_w_device;
                if (sends) {
                    mux_crs.insert(CoreRange(mux_mux_cores[s]));
                }
            }
            CoreRangeSet worker_crset(worker_crs);
            CoreRangeSet mux_crset(mux_crs.empty() ? std::set<CoreRange>{CoreRange({0, 0})} : mux_crs);
            occ_w_workers = worker_crset;
            occ_wmux = mux_crset;

            // Send CB (c_in0) on the mux WORKER cores — the base cb_w_sender_config was created only on
            // the standard column-0 W cores, so the relocated workers had no CB (reader/writer would
            // block on cb_reserve/cb_wait_front). Sized to hold coalesce groups deep for pipelining.
            {
                const uint32_t w_mux_cb_pages = 16 * w_coalesce_n;
                CircularBufferConfig cb_mux_w(w_mux_cb_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}});
                cb_mux_w.set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
                CreateCircularBuffer(program, worker_crset, cb_mux_w);
            }

            mux_reader_ct.push_back(scatter_border ? 1u : 0u);  // SCATTER_BORDER (ct_after_src + 2)
            mux_reader_ct.push_back(w_scatter_scratch_cb);      // SCATTER_SCRATCH_CB (ct_after_src + 3)
            // Padded accessor at +4 must always exist (constructed unconditionally); filler when unused.
            TensorAccessorArgs(scatter_border ? *padded_buf_w : *halo_buffer).append_to(mux_reader_ct);
            if (scatter_border) {
                w_border_scratch_cb(worker_crset);
            }
            mux_reader_cfg.compile_args = mux_reader_ct;
            w_reader_kernel_id = CreateKernel(program, kdir + "np_phase2_w_reader.cpp", worker_crset, mux_reader_cfg);
            w_writer_kernel_id = CreateKernel(program, kdir + "np_w_mux_writer.cpp", worker_crset, mux_writer_cfg);
            // Reader common args (CRTA[0]=halo addr, [1]=barrier_sem, [2]=w_neighbor_sem) — MUST be set on
            // the mux reader too, else barrier_sem_addr is unset and the H->W barrier wait never completes.
            // Border fold appends [3]=padded_addr, [4]=wleft_base, [5]=wright_base, [6]=np_pad2_right.
            std::vector<uint32_t> mux_reader_crta = {
                halo_buffer->address(), op.barrier_semaphore.address(), op.w_neighbor_semaphore.address()};
            w_border_common(mux_reader_crta);
            SetCommonRuntimeArgs(program, w_reader_kernel_id, mux_reader_crta);
            SetCommonRuntimeArgs(
                program,
                w_writer_kernel_id,
                {halo_buffer->address(),
                 halo_buffer->address(),
                 op.w_neighbor_semaphore.address(),
                 op.barrier_semaphore.address(),
                 input_halo_dim_size,
                 static_cast<uint32_t>(op.np_padding_h)});

            auto mux_kernel_id = CreateKernel(
                program,
                "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                mux_crset,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = mux_cfg.get_fabric_mux_compile_time_args(),
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

            for (uint32_t w_link = 0; w_link < pad2_num_links; w_link++) {
                for (uint32_t w_dir = 0; w_dir < 2; w_dir++) {
                    const uint32_t s = w_link * 2 + w_dir;
                    const bool dir_has_neighbor = w_dir ? w_backward_coord.has_value() : w_forward_coord.has_value();
                    // Mux kernel RT args (one mux core per (link,dir)); only meaningful if the dir has a neighbor.
                    CoreCoord mux_lc = mux_mux_cores[s];
                    CoreCoord mux_vc = mesh_device->worker_core_from_logical_core(mux_lc);
                    if (dir_has_neighbor) {
                        const auto src_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                        const auto dst_id =
                            mesh_device->get_fabric_node_id(w_dir ? w_backward_coord.value() : w_forward_coord.value());
                        auto mux_rt = mux_cfg.get_fabric_mux_run_time_args(src_id, dst_id, w_link, program, {mux_lc});
                        SetRuntimeArgs(program, mux_kernel_id, {mux_lc}, mux_rt);
                    }
                    // Termination master = worker 0 of this (link,dir) group.
                    CoreCoord term_master_lc = mux_worker_cores[s * num_w_workers + 0];
                    CoreCoord term_master_vc = mesh_device->worker_core_from_logical_core(term_master_lc);
                    // Split this (link,dir)'s W rows across workers.
                    const uint32_t rows_this_link = w_rows_per_link + (w_link < w_extra_rows ? 1 : 0);
                    const uint32_t base_link_start = (w_link * w_rows_per_link) + std::min(w_link, w_extra_rows);
                    // 8-aligned per-worker split: each worker's start is a whole number of 8-row (bank)
                    // units so its coalesce base stays bank-consistent with the receiver's read; the LAST
                    // worker absorbs the non-8 remainder (its coalesce handles the partial tail group).
                    // Even (non-8) splits corrupt the W-section at ~0.996 PCC on non-aligned shapes.
                    constexpr uint32_t BANKS = 8;
                    const uint32_t units = rows_this_link / BANKS;
                    const uint32_t units_per_wk = units / num_w_workers;
                    const uint32_t pw = w_dir ? op.np_pad2_right : op.np_pad2_left;
                    const uint32_t section_base = w_dir ? w_section_wright_base : w_section_wleft_base;
                    // recv-sem target = the SAME-direction worker core (matches the standard W writer,
                    // which sets neighbor_sem_noc0 = w_virtual_core[w_link*2+w_dir]). The receiving reader
                    // on the neighbor waits at the same (device-independent) core coords.
                    for (uint32_t wk = 0; wk < num_w_workers; wk++) {
                        CoreCoord wc = mux_worker_cores[s * num_w_workers + wk];
                        CoreCoord wc_vc = mux_worker_virtual[s * num_w_workers + wk];
                        const uint32_t wk_start = base_link_start + wk * units_per_wk * BANKS;
                        const bool last_wk = (wk == num_w_workers - 1);
                        const uint32_t wk_count =
                            last_wk ? (rows_this_link - wk * units_per_wk * BANKS) : (units_per_wk * BANKS);
                        // Reader RT args (same layout as the standard W reader). barrier_count = number of
                        // H workers that signal the H->W barrier (H-mux: links*dirs*workers; else H cores).
                        const uint32_t h_signal_count =
                            use_h_mux ? (num_links * num_directions * num_h_workers) : num_h_fabric_cores;
                        std::vector<uint32_t> r_rt = {
                            wk_count,
                            wk_start,
                            pw,
                            h_signal_count,
                            op.np_pad2_left,
                            num_sticks_per_halo_dim,
                            static_cast<uint32_t>(w_dir ? is_last_w_device : is_first_w_device),
                            static_cast<uint32_t>(w_dir ? is_first_w_device : is_last_w_device),
                            w_dir,
                            input_buffer->address(),
                            input_halo_dim_size,
                            op.np_padding_h,
                            outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim,
                            op.input_pad_h,
                            op.input_pad_w};
                        SetRuntimeArgs(program, w_reader_kernel_id, {wc}, r_rt);
                        // Writer RT args: per-core (base,rows,sem xy,dir,route) then mux conn (0..16).
                        // sem targets = the neighbor's reader worker core (NOC coords are device-independent),
                        // NOT the mux core: the recv-sem + startup-barrier incs land on the reader that waits them.
                        std::vector<uint32_t> w_rt = {
                            section_base + wk_start * pw,
                            wk_count,
                            wc_vc.x,
                            wc_vc.y,
                            static_cast<uint32_t>(is_first_w_device),
                            static_cast<uint32_t>(is_last_w_device),
                            w_dir,
                            0u,
                            static_cast<uint32_t>(w_dir ? w_backward_device_offset : w_forward_device_offset)};
                        ttnn::ccl::fabric_mux_connection_rt_args(
                            dir_has_neighbor,
                            /*is_termination_master=*/wk == 0,
                            FabricMuxChannelType::FULL_SIZE_CHANNEL,
                            mux_vc,
                            wk,
                            wc,
                            mux_cfg,
                            program,
                            term_master_vc,
                            w_rt,
                            std::nullopt);
                        SetRuntimeArgs(program, w_writer_kernel_id, {wc}, w_rt);
                    }
                }
            }
            w_fabric_core_range = worker_crset;
        } else {
            // W reader kernel (non-mux, single-worker path).
            auto w_reader_kernel_config = ReaderDataMovementConfig{};
            w_reader_kernel_config.compile_args = {sender_cb_index, is_padding_zeros, page_size};
            TensorAccessorArgs(*halo_buffer).append_to(w_reader_kernel_config.compile_args);
            TensorAccessorArgs(*input_buffer).append_to(w_reader_kernel_config.compile_args);
            w_reader_kernel_config.compile_args.push_back(w_coalesce_n);  // W-send bank-major coalesce factor (0=off)
            w_reader_kernel_config.compile_args.push_back(0);             // W_MUX_MODE off (standard 1-worker path)
            w_reader_kernel_config.compile_args.push_back(scatter_border ? 1u : 0u);  // SCATTER_BORDER
            w_reader_kernel_config.compile_args.push_back(w_scatter_scratch_cb);      // SCATTER_SCRATCH_CB
            // Padded accessor at +4 must always exist (constructed unconditionally); filler when unused.
            TensorAccessorArgs(scatter_border ? *padded_buf_w : *halo_buffer)
                .append_to(w_reader_kernel_config.compile_args);
            if (scatter_border) {
                w_border_scratch_cb(w_fabric_core_range);
            }
            w_reader_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/"
                "np_phase2_w_reader.cpp",
                w_fabric_core_range,
                w_reader_kernel_config);
            {
                std::vector<uint32_t> w_reader_crta = {
                    halo_buffer->address(),
                    op.barrier_semaphore.address(),
                    op.w_neighbor_semaphore.address(),
                };
                // No per-batch H-region sems. H->W ordering is the upfront barrier the W-reader waits on
                // (op.barrier_semaphore, CRTA[1]). Border fold appends
                // [3]=padded_addr, [4]=wleft_base, [5]=wright_base, [6]=np_pad2_right.
                w_border_common(w_reader_crta);
                SetCommonRuntimeArgs(program, w_reader_kernel_id, w_reader_crta);
            }

            // W writer kernel
            auto w_writer_kernel_config = WriterDataMovementConfig{};
            w_writer_kernel_config.compile_args = {sender_cb_index, is_padding_zeros, page_size};
            TensorAccessorArgs(*halo_buffer).append_to(w_writer_kernel_config.compile_args);
            w_writer_kernel_config.compile_args.push_back(0);            // use_l1_intermediate
            w_writer_kernel_config.compile_args.push_back(0);            // recv_cb_id
            w_writer_kernel_config.compile_args.push_back(0);            // handle_incoming_writes
            w_writer_kernel_config.compile_args.push_back(1);            // is_w_fabric_writer
            w_writer_kernel_config.compile_args.push_back(w_ring_size);  // ring_size
            w_writer_kernel_config.compile_args.push_back(
                sender_cb_index);  // send_cb_id: W keeps the c_in0 per-stick path
            w_writer_kernel_config.compile_args.push_back(
                use_corner_first);  // unused on W writer; keeps arg layout aligned
            w_writer_kernel_config.compile_args.push_back(w_coalesce_n);  // W-send bank-major coalesce factor (0=off)
            w_writer_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/"
                "np_writer.cpp",
                w_fabric_core_range,
                w_writer_kernel_config);
            SetCommonRuntimeArgs(
                program,
                w_writer_kernel_id,
                {halo_buffer->address(),
                 halo_buffer->address(),
                 op.w_neighbor_semaphore.address(),
                 op.h_neighbor_semaphore.address(),
                 input_halo_dim_size,
                 static_cast<uint32_t>(op.np_padding_h)});  // [4],[5]: W-writer per-batch two-pass reorder dims

            // Per-core W fabric runtime args. Split this direction's W rows across links evenly.
            for (uint32_t w_link = 0; w_link < pad2_num_links; w_link++) {
                const uint32_t w_link_start = (w_link * w_rows_per_link) + std::min(w_link, w_extra_rows);
                const uint32_t w_link_count = w_rows_per_link + (w_link < w_extra_rows ? 1 : 0);

                for (uint32_t w_direction = 0; w_direction < 2; w_direction++) {
                    uint32_t w_core_idx = (w_link * 2) + w_direction;
                    CoreCoord w_core = w_fabric_logical_cores[w_core_idx];
                    CoreCoord w_virtual_core = w_fabric_virtual_cores[w_core_idx];

                    // barrier_count = number of H workers that signal the H->W barrier (H-mux:
                    // links*dirs*workers; else H fabric cores).
                    uint32_t barrier_count =
                        use_h_mux ? (num_links * num_directions * num_h_workers) : num_h_fabric_cores;

                    // W reader runtime args
                    std::vector<uint32_t> w_reader_rt_args = {
                        w_link_count,
                        w_link_start,
                        w_direction ? op.np_pad2_right : op.np_pad2_left,
                        barrier_count,
                        op.np_pad2_left,
                        num_sticks_per_halo_dim};
                    w_reader_rt_args.push_back(w_direction ? is_last_w_device : is_first_w_device);
                    w_reader_rt_args.push_back(w_direction ? is_first_w_device : is_last_w_device);
                    w_reader_rt_args.push_back(w_direction);
                    // Input buffer address for the W-edge interior row reads.
                    w_reader_rt_args.push_back(input_buffer->address());
                    w_reader_rt_args.push_back(input_halo_dim_size);
                    w_reader_rt_args.push_back(op.np_padding_h);
                    // h_halo_hbot_base
                    w_reader_rt_args.push_back(outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim);
                    // Padded-input strides for the W-edge interior reads (0 = contiguous).
                    w_reader_rt_args.push_back(op.input_pad_h);
                    w_reader_rt_args.push_back(op.input_pad_w);
                    SetRuntimeArgs(program, w_reader_kernel_id, {w_core}, w_reader_rt_args);

                    // W writer runtime args — addresses the W-section of the compact buffer.
                    // Direction 0 (forward): writes W-left on receiver.
                    // Direction 1 (backward): writes W-right on receiver.
                    uint32_t w_pad = w_direction ? op.np_pad2_right : op.np_pad2_left;
                    uint32_t w_base = w_direction ? w_section_wright_base : w_section_wleft_base;
                    std::vector<uint32_t> w_writer_rt_args = {
                        w_base + w_link_start * w_pad,
                        0,
                        num_sticks_per_halo_dim,
                        w_pad,
                        w_link_count,
                        w_pad,
                        0,
                        1,
                        1,
                        w_virtual_core.x,
                        w_virtual_core.y,
                        true,
                        w_fabric_virtual_cores[(w_link * 2) + (1 - w_direction)].x,
                        w_fabric_virtual_cores[(w_link * 2) + (1 - w_direction)].y};
                    // No Phase 2 signal targets (W writer doesn't signal the H->W barrier), but the
                    // placeholder count must match np_writer.cpp's MAX_PHASE2_SIGNAL_TARGETS read.
                    constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 32;
                    w_writer_rt_args.push_back(0);
                    for (uint32_t s = 0; s < MAX_PHASE2_SIGNAL_TARGETS * 2; s++) {
                        w_writer_rt_args.push_back(0);
                    }
                    w_writer_rt_args.push_back(w_direction ? is_last_w_device : is_first_w_device);
                    w_writer_rt_args.push_back(w_direction ? is_first_w_device : is_last_w_device);
                    w_writer_rt_args.push_back(w_direction);
                    // W unicast route args
                    uint32_t w_device_offset = w_direction ? w_backward_device_offset : w_forward_device_offset;
                    w_writer_rt_args.push_back(0);                // dst_mesh_id (unused for 1D)
                    w_writer_rt_args.push_back(w_device_offset);  // distance_in_hops
                    // W barrier multicast route info
                    const auto& w_mcast_args = w_direction ? w_mcast_backward_args : w_mcast_forward_args;
                    w_writer_rt_args.insert(w_writer_rt_args.end(), w_mcast_args.begin(), w_mcast_args.end());
                    // Fabric connection args
                    if (w_direction) {
                        w_writer_rt_args.push_back(false);
                        w_writer_rt_args.push_back(w_backward_coord.has_value());
                        if (w_backward_coord.has_value()) {
                            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(w_backward_coord.value());
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_fabric_node_id, dst_fabric_node_id, w_link, program, {w_core}, w_writer_rt_args);
                        }
                    } else {
                        w_writer_rt_args.push_back(w_forward_coord.has_value());
                        if (w_forward_coord.has_value()) {
                            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(w_forward_coord.value());
                            tt::tt_fabric::append_fabric_connection_rt_args(
                                src_fabric_node_id, dst_fabric_node_id, w_link, program, {w_core}, w_writer_rt_args);
                        }
                        w_writer_rt_args.push_back(false);
                    }
                    // Override the W-writer rt_args to index into the compact halo buffer's W-sections.
                    {
                        const uint32_t h_total = input_halo_dim_size + 2 * op.np_padding_h;
                        const uint32_t wleft_base = outer_dim_size * 2u * op.np_padding_h * num_sticks_per_halo_dim;
                        const uint32_t wright_base = wleft_base + outer_dim_size * op.np_pad2_left * h_total;
                        const uint32_t pw_this_dir = w_direction ? op.np_pad2_right : op.np_pad2_left;
                        const uint32_t section_base = w_direction ? wright_base : wleft_base;
                        w_writer_rt_args[0] = section_base + w_link_start * pw_this_dir;
                        w_writer_rt_args[3] = pw_this_dir;
                        w_writer_rt_args[6] = 0;
                        w_writer_rt_args[7] = 1;
                        w_writer_rt_args[8] = 1;
                    }
                    SetRuntimeArgs(program, w_writer_kernel_id, {w_core}, w_writer_rt_args);
                }
            }
        }  // end else (standard 1-worker W path)
    }

    // ------------------------------------------------------------------------
    // Padded-output fused mode: copy the INTERIOR (input -> padded output) on the free cores (columns
    // x>=1; fabric uses column 0) CONCURRENTLY with the fabric exchange. The interior has no dependency
    // on the exchange, so no barrier is needed — it just overlaps the fabric transport instead of being
    // serialized after it (the old two-op flow). The caller fills the border afterward via
    // halo_scatter(border_only), which reads the compact buffer this op wrote (op-level dependency).
    // ------------------------------------------------------------------------
    tt::tt_metal::KernelHandle scatter_kernel_id = 0;
    CoreRangeSet scatter_core_range;
    bool has_scatter = false;
    if (op.output_padded && tensor_args.padded_output.has_value() && compute_grid_size.x > 1) {
        Buffer* padded_buf = tensor_args.padded_output.value().buffer();
        const uint32_t Hd_i = input_halo_dim_size;      // interior H (input is unpadded in repack mode)
        const uint32_t Wd_i = num_sticks_per_halo_dim;  // interior W
        const uint32_t Hp_i = Hd_i + 2 * op.np_padding_h;
        const uint32_t Wp_i = Wd_i + 2 * op.np_padding_w;
        const uint32_t n_int = outer_dim_size * Hd_i * Wd_i;
        // Free cores = cols [1, grid.x) minus the fabric workers/mux. The INTERIOR copy runs here,
        // concurrent with the exchange (no dependency). The BORDER is written by the W-readers themselves
        // (visibility-safe: each observed its compact rows before writing) — see the SCATTER_BORDER path.
        const CoreRangeSet cols_ge1(CoreRange({1, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
        const CoreRangeSet scatter_cores = cols_ge1.subtract(np_worker_core_ranges)
                                               .subtract(occ_hw_workers)
                                               .subtract(occ_hmux)
                                               .subtract(occ_w_workers)
                                               .subtract(occ_wmux);
        const uint32_t num_scatter = scatter_cores.num_cores();
        if (n_int > 0 && num_scatter > 0) {
            const uint32_t per_core = (n_int + num_scatter - 1) / num_scatter;

            constexpr uint32_t sc_cb_id = tt::CBIndex::c_0;
            constexpr uint32_t sc_pages = 8;
            tt::DataFormat sc_df = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
            CircularBufferConfig sc_cb_cfg =
                CircularBufferConfig(sc_pages * page_size, {{sc_cb_id, sc_df}}).set_page_size(sc_cb_id, page_size);
            CreateCircularBuffer(program, scatter_cores, sc_cb_cfg);

            std::vector<uint32_t> sc_ct = {
                page_size,
                outer_dim_size,
                Hp_i,
                Wp_i,
                Hd_i,
                Wd_i,
                op.np_padding_h,
                op.np_padding_w,
                sc_cb_id,
                sc_pages,
                /*border_only=*/0u};
            TensorAccessorArgs(*input_buffer).append_to(sc_ct);
            TensorAccessorArgs(*halo_buffer).append_to(sc_ct);
            TensorAccessorArgs(*padded_buf).append_to(sc_ct);
            auto sc_cfg = WriterDataMovementConfig{};
            sc_cfg.compile_args = sc_ct;
            scatter_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/"
                "np_fused_scatter_writer.cpp",
                scatter_cores,
                sc_cfg);
            scatter_core_range = scatter_cores;
            has_scatter = true;

            uint32_t assigned = 0;
            for (const auto& cr : scatter_cores.ranges()) {
                for (const auto& c : cr) {
                    const uint32_t start = assigned;
                    const uint32_t count = (start >= n_int) ? 0u : std::min(per_core, n_int - start);
                    // Interior-only range [0,n_int): no exchange dependency, overlaps the fabric transport.
                    // The border is written by the W-readers (num_readers=0 => this kernel never waits).
                    SetRuntimeArgs(
                        program,
                        scatter_kernel_id,
                        c,
                        {input_buffer->address(),
                         halo_buffer->address(),
                         padded_buf->address(),
                         start,
                         count,
                         /*compact_ready=*/0u,
                         /*num_readers=*/0u,
                         op.logical_h,
                         mask_device_h_offset,
                         op.logical_w,
                         mask_device_w_offset});
                    assigned += count;
                }
            }
        }
    }

    return cached_program_t{
        std::move(program),
        NpHaloSharedVariables{
            .np_artifacts = NpHaloArtifacts{
                h_reader_kernel_id,
                h_writer_kernel_id,
                w_reader_kernel_id,
                w_writer_kernel_id,
                /*has_w_fabric=*/true,
                w_fabric_core_range,
                has_scatter,
                scatter_kernel_id,
                scatter_core_range}}};
}

// ============================================================================
// override_runtime_arguments — refresh per-dispatch addresses for the NP kernels
// ============================================================================
void NpHaloMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const NpHaloParams& op,
    const NpHaloInputs& tensor_args,
    Tensor& tensor_return_value) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t halo_buffer_addr = tensor_args.halo_buffer.buffer()->address();
    (void)tensor_return_value;  // output IS the halo_buffer
    const uint32_t h_sem_addr = op.h_neighbor_semaphore.address();
    const uint32_t barrier_sem_addr = op.barrier_semaphore.address();
    const uint32_t w_sem_addr = op.w_neighbor_semaphore.address();

    for (auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(coordinate_range);

        // --- NP H-fabric reader CRTA ---
        // CRTA[0] = input_addr, CRTA[1] = halo_buffer_addr, CRTA[2] = h_sem_addr
        auto& hr = GetCommonRuntimeArgs(program, shared_vars.np_artifacts.h_reader_kernel_id);
        hr[0] = input_addr;
        hr[1] = halo_buffer_addr;
        hr[2] = h_sem_addr;
        // Mux path (H_SIGNAL_W_RECV): np_h_reader signals the H->W barrier from CRTA[3]. It ping-pongs
        // per dispatch, so refresh it or a cache hit signals the prior barrier while the W-reader waits
        // on the current one -> deadlock. Direct path has only 3 CRTA (np_writer owns the barrier).
        if (hr.size() > 3) {
            hr[3] = barrier_sem_addr;
        }

        // --- NP H-fabric writer CRTA ---
        // CRTA[0] = input_addr, CRTA[1] = halo_buffer_addr, CRTA[2] = h_sem_addr,
        // CRTA[3] = barrier_sem_addr, CRTA[4] = num_reader_cores (static), CRTA[5+] = NOC coords (static)
        auto& hw = GetCommonRuntimeArgs(program, shared_vars.np_artifacts.h_writer_kernel_id);
        hw[0] = input_addr;
        hw[1] = halo_buffer_addr;
        hw[2] = h_sem_addr;
        hw[3] = barrier_sem_addr;
        // hw[4+] = num_reader_cores and NOC coords — static, set once in create_at()

        // --- NP W-fabric kernels (only when 2D) ---
        if (shared_vars.np_artifacts.has_w_fabric) {
            auto& wr = GetCommonRuntimeArgs(program, shared_vars.np_artifacts.w_reader_kernel_id);
            wr[0] = halo_buffer_addr;
            wr[1] = barrier_sem_addr;
            wr[2] = w_sem_addr;
            // wr[3+] = num_reader_cores and NOC coords — static, set once in create_at()
            // Border-fold path: wr[3] = padded_output addr (ping-pong buffer, changes per dispatch);
            // wr[4..6] (wleft/wright bases, pad2_right) are static.
            if (tensor_args.padded_output.has_value() && wr.size() > 6) {
                wr[3] = tensor_args.padded_output.value().buffer()->address();
            }

            // Per-core RTA[9] of the W-reader holds input_buffer->address() (set in
            // create_at from the first dispatch's input). On subsequent dispatches the
            // input tensor may be at a different DRAM address, so refresh RTA[9] here
            // or the W-reader will pull halo sticks from a stale/garbage DRAM region
            // and fabric-write garbage into the neighbor's halo buffer.
            auto& w_reader_args_by_core = GetRuntimeArgs(program, shared_vars.np_artifacts.w_reader_kernel_id);
            for (const auto& core_range : shared_vars.np_artifacts.fabric_core_range.ranges()) {
                for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                    for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                        auto& w_reader_args = w_reader_args_by_core[x][y];
                        if (w_reader_args.size() > 9) {
                            w_reader_args[9] = input_addr;
                        }
                    }
                }
            }

            auto& ww = GetCommonRuntimeArgs(program, shared_vars.np_artifacts.w_writer_kernel_id);
            ww[0] = halo_buffer_addr;
            ww[1] = halo_buffer_addr;
            ww[2] = w_sem_addr;
            ww[3] = h_sem_addr;
        }

        // --- Padded-output fused interior-copy scatter kernel: refresh per-core [input, compact, padded]. ---
        if (shared_vars.np_artifacts.has_scatter && tensor_args.padded_output.has_value()) {
            const uint32_t padded_addr = tensor_args.padded_output.value().buffer()->address();
            auto& sc_args_by_core = GetRuntimeArgs(program, shared_vars.np_artifacts.scatter_kernel_id);
            for (const auto& cr : shared_vars.np_artifacts.scatter_core_range.ranges()) {
                for (uint32_t x = cr.start_coord.x; x <= cr.end_coord.x; ++x) {
                    for (uint32_t y = cr.start_coord.y; y <= cr.end_coord.y; ++y) {
                        auto& a = sc_args_by_core[x][y];
                        if (a.size() >= 3) {
                            a[0] = input_addr;
                            a[1] = halo_buffer_addr;
                            a[2] = padded_addr;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
