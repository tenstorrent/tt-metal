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
// create_at — the heart of the fusion: NP fabric kernels + conv3d kernels in
//             one program, sharing a progress semaphore for pipelining.
// ============================================================================
NpHaloMeshWorkloadFactory::cached_program_t NpHaloMeshWorkloadFactory::create_at(
    const NpHaloParams& op,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const NpHaloInputs& tensor_args,
    Tensor& tensor_return_value) {
    (void)tensor_return_value;  // output IS the pre-allocated tensor_args.halo_buffer
    auto* mesh_device = tensor_args.input_tensor.device();

    Program program{};

    // =========================================================================
    // PART 1: NP FABRIC KERNELS (fabric_only H-dim path, dim=1 for BTHWC)
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

    // In fabric_only mode output is a compact halo buffer
    uint32_t output_halo_dim_size = op.np_padding_h + op.np_padding_h;  // padding_left + padding_right

    // outer_dim_size = B * T (all dims before H)
    uint32_t outer_dim_size = 1;
    for (size_t d = 0; d < np_dim; d++) {
        outer_dim_size *= input_tensor_shape[d];
    }

    bool is_first_device = !backward_coord.has_value();
    bool is_last_device = !forward_coord.has_value();
    bool is_padding_zeros = op.padding_mode == "zeros";
    // The fused op is always 2D (H+W padding) — validate() enforces np_pad_dim2.has_value(). The
    // H-only (1D) path is removed; is_2d is kept as a named constant for readability where the W-phase
    // setup is gated.
    TT_FATAL(op.np_pad_dim2.has_value(), "NpConv3d: fused op requires 2D padding (H+W).");
    const bool is_2d = true;

    // H corner-first (PCC-neutral): the H-writer sends the W-boundary corner sticks to the neighbor's L1
    // recv buffer + raises the recv sem BEFORE the bulk middle row, so the neighbor's H recv-wait clears
    // after ~2 sticks instead of the full row. Requires padding==1 (the kernel's corner-first path).
    // W two-pass is the progress>0 conv gate; the halo op uses its own interior-first reorder (below)
    // instead, so keep this OFF here.
    const uint32_t use_corner_first = (op.np_padding_h == 1) ? 1u : 0u;
    const uint32_t use_w_two_pass = 0u;

    // For the compact halo buffer, H-section rows are exactly W_dev wide.
    // W-padding is handled in a separate W-section, so no extra columns in H rows.
    // (The standalone NP factory widens rows to W+pad for padded-tensor output,
    //  but the compact buffer layout keeps H and W sections independent.)
    uint32_t output_num_sticks_per_halo_dim = num_sticks_per_halo_dim;
    uint32_t writer_stick_start_id = 0;
    uint32_t writer_num_sticks_to_read = num_sticks_per_halo_dim;

    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t num_links = static_cast<uint32_t>(op.np_num_links);
    uint32_t pad2_num_links = static_cast<uint32_t>(op.np_pad2_num_links);

    // Halo-only op: no conv, so no per-batch progress signalling. progress_t_batch_size == 0 compiles
    // out every per-batch region-sem path in the NP kernels; H->W ordering is instead an upfront barrier
    // (the W-reader waits the H-writers' Phase-2 barrier signal — see np_phase2_w_reader).
    const uint32_t progress_t_batch_size = 0;

    // H-send bank-major coalescing. The halo-only op's upfront H->W barrier makes the corner-first L1
    // path unnecessary, so an eligible H exchange uses the simpler straight-to-DRAM path
    // (use_l1_intermediate=0, whole H-halo row -> neighbor H-section DRAM; corners land in DRAM where the
    // W-reader reads them) AND ships each row bank-major coalesced (h_coalesce_n same-bank sticks per 4KB
    // packet). Eligible when padding_h==1 and W_dev (num_sticks_per_halo_dim) is 8-aligned so every row
    // base is 8-aligned. BH: 8 interleaved DRAM banks.
    uint32_t h_coalesce_n = 0;
    if (op.np_padding_h == 1 && (num_sticks_per_halo_dim % 8u == 0)) {
        h_coalesce_n = std::min(16u, 4096u / page_size);
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

    // Fabric-mux worker count (reach link BW; see PLAN_NP_FABRIC_MUX_IMPL.md). One worker per link
    // caps at ~1.4 GB/s; the eth link (~12.5 GB/s Linear) is only saturated by multiple workers feeding
    // it through a mux core. Worker count follows tt-metal's all_gather heuristic by W data-moved per
    // (link,direction): >256KB=>4, 4KB..256KB=>2, else 1. Env override TT_NP_W_WORKERS for bring-up.
    uint32_t num_w_workers = 1;
    {
        const uint32_t w_h_total_est = input_halo_dim_size + 2 * op.np_padding_h;
        const uint64_t w_rows_est = static_cast<uint64_t>(outer_dim_size) * w_h_total_est;  // W interior rows/dir
        const uint64_t w_bytes_per_link_dir =
            (pad2_num_links > 0) ? (w_rows_est * op.np_pad2_left * page_size / pad2_num_links) : 0;
        if (w_bytes_per_link_dir > 256u * 1024u) {
            num_w_workers = 4;
        } else if (w_bytes_per_link_dir > 4u * 1024u) {
            num_w_workers = 2;
        }
        if (const char* e = std::getenv("TT_NP_W_WORKERS")) {
            num_w_workers = std::max(1, atoi(e));
        }
        log_debug(
            tt::LogOp, "np_halo mux: W bytes/(link,dir)={}, num_w_workers={}", w_bytes_per_link_dir, num_w_workers);
    }

    // H fabric cores occupy column 0, rows [0, num_h_fabric_cores). W fabric cores
    // follow in the same column. This gives conv3d a clean rectangular grid
    // (cols [1, grid.x)) instead of the L-shape produced by a first-row layout.
    CoreCoord np_core_grid(1, num_h_fabric_cores);
    auto
        [num_np_cores,
         np_worker_core_ranges,
         np_core_group_1,
         np_core_group_2,
         np_dims_per_core_group_1,
         np_dims_per_core_group_2] = split_work_to_cores(np_core_grid, outer_dim_size * 2);

    // Inc B: batch-align H partition so each (direction, link) owns whole T-batches (per-(HT/HB,link)
    // sems need it). Link l owns frames [l*h_dims_per_link, min((l+1)*h_dims_per_link, outer_dim_size)).
    const uint32_t h_pb = progress_t_batch_size;
    const uint32_t h_total_batches = (h_pb > 0) ? ((outer_dim_size + h_pb - 1) / h_pb) : 0;
    const uint32_t h_batches_per_link = (h_total_batches > 0) ? ((h_total_batches + num_links - 1) / num_links) : 0;
    const uint32_t h_dims_per_link = h_batches_per_link * h_pb;

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
        uint32_t max_outer_dims_per_core = std::max(np_dims_per_core_group_1, h_dims_per_link);
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

        // fabric_only: W exchange covers all H_dev + 2*ph rows per T
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
        // base+r+8*(N-1) are contiguous on bank (base+r)%8, so N of them ship as ONE N*page fabric
        // write straight to the neighbor's interleaved DRAM (no L1-recv, no receiver changes). Eligible
        // only when every W-core's base stick-id is 8-aligned (so rel%8 == bank) and pw==1, on the
        // halo-only (progress==0) path. Middle devices only (edge devices keep the per-stick path).
        constexpr uint32_t NP_NUM_DRAM_BANKS = 8;
        const bool w_coalesce_ok = (progress_t_batch_size == 0) && (op.np_pad2_left == 1) && (op.np_pad2_right == 1) &&
                                   (w_section_wleft_base % NP_NUM_DRAM_BANKS == 0) &&
                                   (w_section_wright_base % NP_NUM_DRAM_BANKS == 0) &&
                                   (w_rows_per_link % NP_NUM_DRAM_BANKS == 0) && (w_extra_rows == 0);
        if (w_coalesce_ok) {
            w_coalesce_n = std::min(16u, 4096u / page_size);  // sticks per <=4KB packet
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
    auto h_reader_kernel_id = CreateKernel(
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
    // Per-batch progress-sem granularity is always passed; 0 disables the per-batch path.
    h_writer_kernel_config.compile_args.push_back(progress_t_batch_size);
    h_writer_kernel_config.compile_args.push_back(hsend_cb_index);    // send_cb_id (batched H send)
    h_writer_kernel_config.compile_args.push_back(use_w_two_pass);    // unused on H writer; keeps arg layout aligned
    h_writer_kernel_config.compile_args.push_back(use_corner_first);  // H-writer corner-first gate
    h_writer_kernel_config.compile_args.push_back(h_coalesce_n);      // coalesce factor (H-writer uses the H branch)
    auto h_writer_kernel_id = CreateKernel(
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
            // CRTA[4]: number of conv3d reader cores to signal
            static_cast<uint32_t>(reader_noc_coords.size()),
            // CRTA[5+]: interleaved (x, y) NOC coords for each reader core
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
            if (h_pb > 0) {
                const uint32_t b_start = link * h_dims_per_link;
                const uint32_t b_end = std::min((link + 1) * h_dims_per_link, outer_dim_size);
                link_dims_to_read = (b_end > b_start) ? (b_end - b_start) : 0u;
            }

            // Reader runtime args
            std::vector<uint32_t> reader_rt_args = {
                link_offset_start_id * input_halo_dim_size,                          // outer_dim_offset_start_id
                0,                                                                   // stick_start_id
                input_halo_dim_size,                                                 // input_halo_dim_size
                link_dims_to_read,                                                   // outer_dim_size
                op.np_padding_h,                                                     // padding (symmetric)
                num_sticks_per_halo_dim,                                             // num_sticks_to_read
                num_sticks_per_halo_dim,                                             // num_sticks_per_halo_dim
                corner_sticks_per_row};                                              // num_l1_recv_sticks_per_row
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
                op.np_pad2_left,                                     // padding_left (W-axis, for L1 corner detection)
                h_writer_num_sticks_to_read,                         // num_sticks_to_read
                h_writer_num_sticks_per_halo_dim,                    // num_sticks_per_halo_dim
                virtual_core.x,                                      // neighbor_sem_noc0_x
                virtual_core.y,                                      // neighbor_sem_noc0_y
                true,                                                // use_barrier_semaphore
                virtual_opposite_core.x,                             // barrier_sem_noc0_x
                virtual_opposite_core.y};                            // barrier_sem_noc0_y
            // Phase 2 signal targets (W fabric reader cores)
            constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 8;
            writer_rt_args.push_back(num_w_fabric_cores);
            for (uint32_t s = 0; s < MAX_PHASE2_SIGNAL_TARGETS; s++) {
                if (s < num_w_fabric_cores) {
                    writer_rt_args.push_back(w_fabric_virtual_cores[s].x);
                    writer_rt_args.push_back(w_fabric_virtual_cores[s].y);
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
            // fabric_only mode: override writer rt_args to index into the compact halo buffer
            {
                uint32_t link_t_start = (output_num_sticks_per_halo_dim > 0)
                                            ? (writer_link_offset_start_id / output_num_sticks_per_halo_dim)
                                            : 0u;
                uint32_t top_halo_total = outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim;
                uint32_t h_top_link_start = link_t_start * op.np_padding_h * num_sticks_per_halo_dim;
                uint32_t h_bot_link_start = top_halo_total + link_t_start * op.np_padding_h * num_sticks_per_halo_dim;
                writer_rt_args[0] = direction ? h_bot_link_start : h_top_link_start;  // per-link offset
                writer_rt_args[1] = 0;                        // stick_start_id (no W-offset in compact)
                writer_rt_args[3] = op.np_padding_h;          // output_halo_dim_size (compact)
                writer_rt_args[8] = num_sticks_per_halo_dim;  // stride = W_dev, not padded W
            }
            // No per-batch progress args (progress_t_batch_size == 0, no conv consumer).
            SetRuntimeArgs(program, h_writer_kernel_id, {core}, writer_rt_args);
        }
        link_offset_start_id += (link_dims_to_read * num_sticks_per_halo_dim);
        writer_link_offset_start_id += (link_dims_to_read * output_num_sticks_per_halo_dim);
    }

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
        const bool use_w_mux =
            (num_w_workers > 1) && !is_first_w_device && !is_last_w_device && (w_coalesce_n > 0);
        if (use_w_mux) {
            using tt::tt_fabric::FabricMuxChannelType;
            const uint32_t l1_base =
                mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
            const size_t mux_buf_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
            auto mux_cfg = tt::tt_fabric::FabricMuxConfig(
                /*num_full_size_channels=*/static_cast<uint8_t>(num_w_workers),
                /*num_header_only_channels=*/0,
                /*num_buffers_full_size_channel=*/8,
                /*num_buffers_header_only_channel=*/0,
                mux_buf_size,
                l1_base);

            // Core layout (halo-only op: cols>=1 are free). Per (link,dir): 1 mux core (col 1) + N worker
            // cores (cols 2+). link/dir index = w_link*2 + w_dir.

            // W reader + mux writer kernels on the worker cores.
            std::vector<uint32_t> mux_reader_ct = {sender_cb_index, is_padding_zeros, page_size};
            TensorAccessorArgs(*halo_buffer).append_to(mux_reader_ct);
            TensorAccessorArgs(*input_buffer).append_to(mux_reader_ct);
            mux_reader_ct.push_back(progress_t_batch_size);
            mux_reader_ct.push_back(use_w_two_pass);
            mux_reader_ct.push_back(w_coalesce_n);
            auto mux_reader_cfg = ReaderDataMovementConfig{};
            mux_reader_cfg.compile_args = mux_reader_ct;

            std::vector<uint32_t> mux_writer_ct = {sender_cb_index, page_size};
            TensorAccessorArgs(*halo_buffer).append_to(mux_writer_ct);
            mux_writer_ct.push_back(w_coalesce_n);
            ttnn::ccl::fabric_mux_connection_ct_args(
                num_w_workers, FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_cfg, mux_writer_ct);
            auto mux_writer_cfg = WriterDataMovementConfig{};
            mux_writer_cfg.compile_args = mux_writer_ct;

            const std::string kdir =
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/";
            // Gather all worker + mux cores into ranges for kernel creation.
            std::vector<CoreCoord> mux_worker_cores;   // flat: [link*2+dir][worker]
            std::vector<CoreCoord> mux_mux_cores;      // flat: [link*2+dir]
            for (uint32_t s = 0; s < pad2_num_links * 2; s++) {
                mux_mux_cores.push_back(CoreCoord{1, s});
                for (uint32_t wk = 0; wk < num_w_workers; wk++) {
                    mux_worker_cores.push_back(CoreCoord{2 + wk, s});
                }
            }
            std::set<CoreRange> worker_crs, mux_crs;
            for (const auto& c : mux_worker_cores) worker_crs.insert(CoreRange(c));
            for (const auto& c : mux_mux_cores) mux_crs.insert(CoreRange(c));
            CoreRangeSet worker_crset(worker_crs), mux_crset(mux_crs);

            w_reader_kernel_id = CreateKernel(program, kdir + "np_phase2_w_reader.cpp", worker_crset, mux_reader_cfg);
            w_writer_kernel_id = CreateKernel(program, kdir + "np_w_mux_writer.cpp", worker_crset, mux_writer_cfg);
            SetCommonRuntimeArgs(
                program,
                w_writer_kernel_id,
                {halo_buffer->address(), halo_buffer->address(), op.w_neighbor_semaphore.address(),
                 op.barrier_semaphore.address(), input_halo_dim_size, static_cast<uint32_t>(op.np_padding_h)});

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
                        const auto dst_id = mesh_device->get_fabric_node_id(
                            w_dir ? w_backward_coord.value() : w_forward_coord.value());
                        auto mux_rt = mux_cfg.get_fabric_mux_run_time_args(src_id, dst_id, w_link, program, {mux_lc});
                        SetRuntimeArgs(program, mux_kernel_id, {mux_lc}, mux_rt);
                    }
                    // Termination master = worker 0 of this (link,dir) group.
                    CoreCoord term_master_lc = mux_worker_cores[s * num_w_workers + 0];
                    CoreCoord term_master_vc = mesh_device->worker_core_from_logical_core(term_master_lc);
                    // Split this (link,dir)'s W rows across workers.
                    const uint32_t rows_this_link = w_rows_per_link + (w_link < w_extra_rows ? 1 : 0);
                    const uint32_t base_link_start = (w_link * w_rows_per_link) + std::min(w_link, w_extra_rows);
                    const uint32_t per_wk = rows_this_link / num_w_workers;
                    const uint32_t extra_wk = rows_this_link % num_w_workers;
                    const uint32_t pw = w_dir ? op.np_pad2_right : op.np_pad2_left;
                    const uint32_t section_base = w_dir ? w_section_wright_base : w_section_wleft_base;
                    for (uint32_t wk = 0; wk < num_w_workers; wk++) {
                        CoreCoord wc = mux_worker_cores[s * num_w_workers + wk];
                        const uint32_t wk_start = base_link_start + wk * per_wk + std::min(wk, extra_wk);
                        const uint32_t wk_count = per_wk + (wk < extra_wk ? 1 : 0);
                        // Reader RT args (same layout as the standard W reader).
                        std::vector<uint32_t> r_rt = {
                            wk_count, wk_start, pw, num_h_fabric_cores, output_num_sticks_per_halo_dim,
                            op.np_pad2_left, num_sticks_per_halo_dim,
                            static_cast<uint32_t>(w_dir ? is_last_w_device : is_first_w_device),
                            static_cast<uint32_t>(w_dir ? is_first_w_device : is_last_w_device),
                            w_dir, input_buffer->address(), input_halo_dim_size, op.np_padding_h,
                            outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim, 0u};
                        SetRuntimeArgs(program, w_reader_kernel_id, {wc}, r_rt);
                        // Writer RT args: per-core (base,rows,sem xy,dir,route) then mux conn (0..16).
                        std::vector<uint32_t> w_rt = {
                            section_base + wk_start * pw, wk_count,
                            mux_vc.x, mux_vc.y, mux_vc.x, mux_vc.y,
                            static_cast<uint32_t>(is_first_w_device), static_cast<uint32_t>(is_last_w_device), w_dir,
                            0u, static_cast<uint32_t>(w_dir ? w_backward_device_offset : w_forward_device_offset)};
                        ttnn::ccl::fabric_mux_connection_rt_args(
                            dir_has_neighbor, /*is_termination_master=*/wk == 0,
                            FabricMuxChannelType::FULL_SIZE_CHANNEL, mux_vc, wk, wc, mux_cfg, program,
                            term_master_vc, w_rt, std::nullopt);
                        SetRuntimeArgs(program, w_writer_kernel_id, {wc}, w_rt);
                    }
                }
            }
            w_fabric_core_range = worker_crset;
        } else {
        // W reader kernel — fused-owned copy: always fabric-only, always per-batch
        // progress-sem signalling.
        auto w_reader_kernel_config = ReaderDataMovementConfig{};
        w_reader_kernel_config.compile_args = {sender_cb_index, is_padding_zeros, page_size};
        TensorAccessorArgs(*halo_buffer).append_to(w_reader_kernel_config.compile_args);
        TensorAccessorArgs(*input_buffer).append_to(w_reader_kernel_config.compile_args);
        // Per-batch signal granularity (matches conv3d reader's progress_t_batch_size CT arg).
        w_reader_kernel_config.compile_args.push_back(progress_t_batch_size);
        w_reader_kernel_config.compile_args.push_back(use_w_two_pass);  // global two-pass gate (lockstep w/ writer)
        w_reader_kernel_config.compile_args.push_back(w_coalesce_n);    // W-send bank-major coalesce factor (0=off)
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
            // No per-batch H-region sems (progress_t_batch_size == 0). H->W ordering is the upfront
            // barrier the W-reader waits on (op.barrier_semaphore, CRTA[1]).
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
        // progress_t_batch_size: W-writer doesn't per-batch-signal in 2D (gated by
        // num_phase2_signal_targets > 0 at runtime), but the CT arg must be present to
        // match np_writer.cpp's arg layout.
        w_writer_kernel_config.compile_args.push_back(progress_t_batch_size);
        w_writer_kernel_config.compile_args.push_back(sender_cb_index);  // send_cb_id: W keeps the c_in0 per-stick path
        w_writer_kernel_config.compile_args.push_back(use_w_two_pass);   // global two-pass gate (lockstep w/ reader)
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

        // Per-core W fabric runtime args.
        // When pipelining (progress_t_batch_size>0), align each link to whole T-batches so the
        // per-(region,link) progress sem counts batches 1:1 and the conv3d reader can map a batch to
        // its owning link by integer division (no per-link boundary table).
        const uint32_t w_h_total = input_halo_dim_size + 2 * op.np_padding_h;
        const uint32_t w_sticks_per_batch = progress_t_batch_size * w_h_total;
        const uint32_t total_w_batches =
            w_sticks_per_batch > 0 ? ((w_outer_dim_size + w_sticks_per_batch - 1) / w_sticks_per_batch) : 0;
        const uint32_t w_batches_per_link =
            (total_w_batches > 0) ? ((total_w_batches + pad2_num_links - 1) / pad2_num_links) : 0;
        for (uint32_t w_link = 0; w_link < pad2_num_links; w_link++) {
            uint32_t w_link_start, w_link_count;
            if (progress_t_batch_size > 0) {
                const uint32_t b_start = w_link * w_batches_per_link;
                const uint32_t b_end = std::min((w_link + 1) * w_batches_per_link, total_w_batches);
                w_link_start = std::min(b_start * w_sticks_per_batch, w_outer_dim_size);
                w_link_count = std::min(b_end * w_sticks_per_batch, w_outer_dim_size) - w_link_start;
            } else {
                w_link_start = (w_link * w_rows_per_link) + std::min(w_link, w_extra_rows);
                w_link_count = w_rows_per_link + (w_link < w_extra_rows ? 1 : 0);
            }

            for (uint32_t w_direction = 0; w_direction < 2; w_direction++) {
                uint32_t w_core_idx = (w_link * 2) + w_direction;
                CoreCoord w_core = w_fabric_logical_cores[w_core_idx];
                CoreCoord w_virtual_core = w_fabric_virtual_cores[w_core_idx];

                uint32_t barrier_count = num_h_fabric_cores;  // fabric_only: no local_copy writers

                // W reader runtime args
                std::vector<uint32_t> w_reader_rt_args = {
                    w_link_count,
                    w_link_start,
                    w_direction ? op.np_pad2_right : op.np_pad2_left,
                    barrier_count,
                    output_num_sticks_per_halo_dim,
                    op.np_pad2_left,
                    num_sticks_per_halo_dim};
                w_reader_rt_args.push_back(w_direction ? is_last_w_device : is_first_w_device);
                w_reader_rt_args.push_back(w_direction ? is_first_w_device : is_last_w_device);
                w_reader_rt_args.push_back(w_direction);
                // fabric_only: pass input buffer address for interior row reads
                w_reader_rt_args.push_back(input_buffer->address());
                w_reader_rt_args.push_back(input_halo_dim_size);
                w_reader_rt_args.push_back(op.np_padding_h);
                // h_halo_hbot_base
                w_reader_rt_args.push_back(outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim);
                // No conv W-edge consumer: push 0 for the (unused) per-batch region progress sem addr
                // to keep the W-reader's per-core arg layout stable.
                w_reader_rt_args.push_back(0u);
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
                // No Phase 2 signal targets
                constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 8;
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
                // fabric_only: override W writer rt_args to index into compact halo buffer W section
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

    return cached_program_t{
        std::move(program),
        NpHaloSharedVariables{
            .np_artifacts = NpHaloArtifacts{
                h_reader_kernel_id,
                h_writer_kernel_id,
                w_reader_kernel_id,
                w_writer_kernel_id,
                /*has_w_fabric=*/true,
                w_fabric_core_range}}};
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

            // Per-core RTA[10] of the W-reader holds input_buffer->address() (set in
            // create_at from the first dispatch's input). On subsequent dispatches the
            // input tensor may be at a different DRAM address, so refresh RTA[10] here
            // or the W-reader will pull halo sticks from a stale/garbage DRAM region
            // and fabric-write garbage into the neighbor's halo buffer.
            auto& w_reader_args_by_core = GetRuntimeArgs(program, shared_vars.np_artifacts.w_reader_kernel_id);
            for (const auto& core_range : shared_vars.np_artifacts.fabric_core_range.ranges()) {
                for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; ++x) {
                    for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; ++y) {
                        auto& w_reader_args = w_reader_args_by_core[x][y];
                        if (w_reader_args.size() > 10) {
                            w_reader_args[10] = input_addr;
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
    }
}

}  // namespace ttnn::experimental::prim
