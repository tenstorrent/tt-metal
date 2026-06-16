// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_conv3d_program_factory.hpp"

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
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

// Conv3d includes
#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation_types.hpp"
#include "ttnn/operations/experimental/conv3d/device/kernels/conv3d_gather_tuning.hpp"
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
NpConv3dMeshWorkloadFactory::cached_mesh_workload_t NpConv3dMeshWorkloadFactory::create_mesh_workload(
    const NpConv3dParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const NpConv3dInputs& tensor_args,
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
NpConv3dMeshWorkloadFactory::cached_program_t NpConv3dMeshWorkloadFactory::create_at(
    const NpConv3dParams& op,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const NpConv3dInputs& tensor_args,
    Tensor& tensor_return_value) {
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

    // Per-T progress batch size: the fused op always pipelines per T-out block, so it equals the base
    // blocking's T_out_block (the Python wrapper used to copy it into a dedicated field).
    const uint32_t progress_t_batch_size = op.conv_config.T_out_block;
    // Link stride of the region-progress sem table. The Python wrapper allocates 4*np_num_links sems
    // and lays them out [region*np_num_links + link], so the table stride is the op's NP num_links —
    // the un-clamped param, independent of the fabric-core clamp applied to the local num_links below
    // (which is only the active-link loop bound, not the table layout).
    const uint32_t region_progress_num_links = static_cast<uint32_t>(op.np_num_links);

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

        CreateCircularBuffer(program, w_fabric_core_range, cb_sender_config);

        w_rows_per_link = w_outer_dim_size / pad2_num_links;
        w_extra_rows = w_outer_dim_size % pad2_num_links;
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
    // PART 2: PROGRESS SEMAPHORE — allocated in this program on conv3d cores
    // -------------------------------------------------------------------------
    // We need the conv3d core range to place the progress semaphore and know
    // the reader NOC coordinates for the H writer CRTA. We compute it by
    // subtracting the NP fabric core range from the full grid.
    CoreRangeSet np_fabric_cores = np_worker_core_ranges;
    if (is_2d) {
        np_fabric_cores = np_fabric_cores.merge(w_fabric_core_range);
    }

    const auto& conv_config = op.conv_config;
    CoreRangeSet conv3d_core_range;
    {
        CoreCoord grid = conv_config.compute_with_storage_grid_size;
        CoreRangeSet full_grid(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        conv3d_core_range = full_grid.subtract(np_fabric_cores);
    }

    // Collect conv3d reader core NOC coords (the NP writers signal each reader's per-(region,link)
    // progress sem at these coords after each T-batch).
    std::vector<std::pair<uint32_t, uint32_t>> reader_noc_coords;
    if (progress_t_batch_size > 0) {
        for (const auto& core : corerange_to_cores(conv3d_core_range, std::nullopt, true)) {
            auto noc = mesh_device->worker_core_from_logical_core(core);
            reader_noc_coords.emplace_back(noc.x, noc.y);
        }
    }

    // -------------------------------------------------------------------------
    // NP H-fabric reader kernel
    // -------------------------------------------------------------------------
    auto h_reader_kernel_config = ReaderDataMovementConfig{};
    h_reader_kernel_config.compile_args = {
        sender_cb_index,   // cb_output_id
        is_padding_zeros,  // is_padding_zeros
        page_size};        // stick_size
    TensorAccessorArgs(*input_buffer).append_to(h_reader_kernel_config.compile_args);
    h_reader_kernel_config.compile_args.push_back(1);               // use_l1_intermediate (always 2D)
    h_reader_kernel_config.compile_args.push_back(recv_cb_index);   // recv_cb_id
    h_reader_kernel_config.compile_args.push_back(hsend_cb_index);  // send_cb_id (batched H send)
    auto h_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/"
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
    h_writer_kernel_config.compile_args.push_back(1);                // use_l1_intermediate (always 2D)
    h_writer_kernel_config.compile_args.push_back(recv_cb_index);    // recv_cb_id
    h_writer_kernel_config.compile_args.push_back(1);                // handle_incoming_writes (always 2D)
    h_writer_kernel_config.compile_args.push_back(0);                // is_w_fabric_writer (false for H)
    h_writer_kernel_config.compile_args.push_back(op.np_ring_size);  // ring_size
    // Per-batch progress-sem granularity is always passed; 0 disables the per-batch path.
    h_writer_kernel_config.compile_args.push_back(progress_t_batch_size);
    h_writer_kernel_config.compile_args.push_back(hsend_cb_index);  // send_cb_id (batched H send)
    auto h_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/"
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
    uint32_t link_t_offset = 0;
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
            if (progress_t_batch_size > 0) {
                writer_rt_args.push_back(link_t_offset);
                // Inc B: this (direction,link)'s H-region sem (H-top=region 0 dir 0, H-bot=region 1 dir 1),
                // signalled per-batch from handle_incoming_writes.
                const uint32_t h_region_idx = direction * region_progress_num_links + link;
                writer_rt_args.push_back(conv_config.region_progress_sem_addr[h_region_idx]);
            }
            SetRuntimeArgs(program, h_writer_kernel_id, {core}, writer_rt_args);
        }
        link_t_offset += link_dims_to_read;
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

        // W reader kernel — fused-owned copy: always fabric-only, always per-batch
        // progress-sem signalling.
        auto w_reader_kernel_config = ReaderDataMovementConfig{};
        w_reader_kernel_config.compile_args = {sender_cb_index, is_padding_zeros, page_size};
        TensorAccessorArgs(*halo_buffer).append_to(w_reader_kernel_config.compile_args);
        TensorAccessorArgs(*input_buffer).append_to(w_reader_kernel_config.compile_args);
        // Per-batch signal granularity (matches conv3d reader's progress_t_batch_size CT arg).
        w_reader_kernel_config.compile_args.push_back(progress_t_batch_size);
        w_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/"
            "np_phase2_w_reader.cpp",
            w_fabric_core_range,
            w_reader_kernel_config);
        {
            std::vector<uint32_t> w_reader_crta = {
                halo_buffer->address(),
                op.barrier_semaphore.address(),
                op.w_neighbor_semaphore.address(),
            };
            if (progress_t_batch_size > 0) {
                w_reader_crta.push_back(static_cast<uint32_t>(reader_noc_coords.size()));
                for (const auto& [x, y] : reader_noc_coords) {
                    w_reader_crta.push_back(x);
                    w_reader_crta.push_back(y);
                }
                // H-region sems (all H-links) + params, for the per-batch corner H-gate that
                // replaces the upfront barrier. Layout after coords: HT[0..3], HB[0..3], h_bpl,
                // num_h_links, h_total_batches. Slots >= num_links are 0 (unused).
                // A no-H-neighbor side is a zero-pad boundary with no producer (the sender-direction
                // H-writer skips handle_incoming_writes), so its sem never increments — pass addr 0 so
                // the W-reader's hsem==0 skip applies. Mirrors the consumer's have_htop/have_hbot.
                const uint32_t rstride = region_progress_num_links;
                const bool have_htop = !is_first_device;
                const bool have_hbot = !is_last_device;
                for (uint32_t l = 0; l < 4u; l++) {
                    const bool ok = have_htop && l < num_links;
                    w_reader_crta.push_back(ok ? conv_config.region_progress_sem_addr[0 * rstride + l] : 0u);
                }
                for (uint32_t l = 0; l < 4u; l++) {
                    const bool ok = have_hbot && l < num_links;
                    w_reader_crta.push_back(ok ? conv_config.region_progress_sem_addr[1 * rstride + l] : 0u);
                }
                w_reader_crta.push_back(h_batches_per_link);
                w_reader_crta.push_back(num_links);
                w_reader_crta.push_back(h_total_batches);
            }
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
        w_writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/"
            "np_writer.cpp",
            w_fabric_core_range,
            w_writer_kernel_config);
        SetCommonRuntimeArgs(
            program,
            w_writer_kernel_id,
            {halo_buffer->address(),
             halo_buffer->address(),
             op.w_neighbor_semaphore.address(),
             op.h_neighbor_semaphore.address()});

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
                // Per-batch region progress sem for this (W-direction, link): W-left=region 2,
                // W-right=region 3. Table stride is region_progress_num_links (= num_links), so the
                // count is 4*num_links (8 on 2x4). conv3d W-edge tiles for this link's batches poll it.
                const uint32_t w_region_idx = (2u + w_direction) * region_progress_num_links + w_link;
                w_reader_rt_args.push_back(conv_config.region_progress_sem_addr[w_region_idx]);
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
    }

    // =========================================================================
    // PART 3: CONV3D KERNELS (halo buffer always enabled, per-T pipelining at T_out_block granularity)
    // =========================================================================

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& output_tensor = tensor_return_value;

    const auto& compute_kernel_config = op.compute_kernel_config;

    auto& core_grid = conv3d_core_range;
    auto conv3d_num_cores = core_grid.num_cores();

    auto input_tensor_shape_logical = input_tensor.logical_shape();
    uint32_t N = input_tensor_shape_logical[0];
    uint32_t T_in = input_tensor_shape_logical[1];
    uint32_t H_in = input_tensor_shape_logical[2];
    uint32_t W_in = input_tensor_shape_logical[3];
    uint32_t C_in = input_tensor_shape_logical[4];

    // Compact halo-buffer geometry — derived from the input shape and the op's NP padding (the conv3d
    // reader indexes the buffer with these). H/W halo padding equals the per-side NP padding; the
    // outer dim is B*T (outer_dim_size, computed above); the buffered H spans the per-device H plus
    // both H halo sides; W rows are W_dev wide.
    const uint32_t h_halo_padding_h = op.np_padding_h;
    const uint32_t h_halo_padding_w = op.np_padding_w;
    const uint32_t h_halo_outer_dim_size = outer_dim_size;
    const uint32_t h_halo_H = H_in + 2 * h_halo_padding_h;
    const uint32_t h_halo_W = W_in;

    // Inflate effective padding with halo buffer H/W contributions.
    // The fused op always uses the halo buffer path.
    std::array<uint32_t, 3> effective_padding = op.padding;
    effective_padding[1] += h_halo_padding_h;
    effective_padding[2] += h_halo_padding_w;

    auto [T_out, H_out, W_out] =
        detail::compute_output_dims(T_in, H_in, W_in, effective_padding, op.stride, op.kernel_size, op.dilation);

    uint32_t C_out = op.output_channels;
    uint32_t padded_C_out = tt::round_up(C_out, tt::constants::TILE_WIDTH);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto dtype_bytes = input_tensor.element_size();
    auto tile_size = tt::tile_size(data_format);

    bool use_bias = bias_tensor.has_value();

    [[maybe_unused]] auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), compute_kernel_config);

    uint32_t C_out_block = conv_config.C_out_block > 0 ? conv_config.C_out_block : padded_C_out;
    uint32_t C_in_block = conv_config.C_in_block > 0 ? conv_config.C_in_block : C_in;

    uint32_t patch_size = op.kernel_size[0] * op.kernel_size[1] * op.kernel_size[2] * C_in_block;
    uint32_t padded_patch_size = tt::round_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t num_patches = conv_config.T_out_block * conv_config.H_out_block * conv_config.W_out_block;

    uint32_t C_in_num_blocks = tt::div_up(C_in, C_in_block);
    TT_FATAL(C_in_num_blocks * C_in_block == C_in, "C_in_num_blocks * C_in_block must equal C_in");
    uint32_t C_out_num_blocks = tt::div_up(padded_C_out, C_out_block);
    TT_FATAL(
        C_out_num_blocks * C_out_block == padded_C_out,
        "C_out_num_blocks * C_out_block must equal padded_C_out ({}). Got C_out_num_blocks={}, C_out_block={}.",
        padded_C_out,
        C_out_num_blocks,
        C_out_block);

    uint32_t matmul_M_t = tt::div_up(num_patches, tt::constants::TILE_HEIGHT);
    uint32_t matmul_K_t = tt::div_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t matmul_N_t = tt::div_up(C_out_block, tt::constants::TILE_WIDTH);
    uint32_t num_patches_tile_padded = tt::round_up(num_patches, tt::constants::TILE_HEIGHT);

    uint32_t patch_size_bytes = patch_size * dtype_bytes;
    uint32_t padded_patch_size_bytes = padded_patch_size * dtype_bytes;
    uint32_t patch_pad_bytes = padded_patch_size_bytes - patch_size_bytes;
    uint32_t C_out_block_bytes = C_out_block * dtype_bytes;
    uint32_t C_in_block_bytes = C_in_block * dtype_bytes;

    // Create conv3d circular buffers on conv3d_core_range
    uint32_t next_cb_index = tt::CBIndex::c_0;

    uint32_t vol2col_rm_pages = (num_patches % tt::constants::TILE_HEIGHT == 0)
                                    ? std::min(num_patches, (uint32_t)tt::constants::TILE_HEIGHT)
                                    : std::min(num_patches, 2 * tt::constants::TILE_HEIGHT);
    uint32_t cb_vol2col_rm_id = next_cb_index++;
    tt::tt_metal::create_cb(
        cb_vol2col_rm_id, program, core_grid, padded_patch_size_bytes, vol2col_rm_pages, data_format);

    uint32_t cb_vol2col_tiled_id = next_cb_index++;
    tt::tt_metal::create_cb(cb_vol2col_tiled_id, program, core_grid, tile_size, matmul_K_t, data_format);

    uint32_t cb_weight_tiled_id = next_cb_index++;
    tt::tt_metal::create_cb(cb_weight_tiled_id, program, core_grid, tile_size, matmul_K_t * matmul_N_t, data_format);

    bool use_fp32_partials = fp32_dest_acc_en && C_in_num_blocks > 1;
    auto partial_data_format = use_fp32_partials ? tt::DataFormat::Float32 : data_format;
    auto partial_tile_size = tt::tile_size(partial_data_format);

    uint32_t cb_matmul_interm_tiled_id = next_cb_index++;
    tt::tt_metal::create_cb(
        cb_matmul_interm_tiled_id, program, core_grid, partial_tile_size, matmul_M_t * matmul_N_t, partial_data_format);

    uint32_t cb_matmul_result_rm_id = next_cb_index++;
    tt::tt_metal::create_cb(
        cb_matmul_result_rm_id, program, core_grid, tile_size, matmul_M_t * matmul_N_t, data_format);

    uint32_t cb_zero_tiled_id = 32;
    if (use_fp32_partials) {
        cb_zero_tiled_id = next_cb_index++;
        tt::tt_metal::create_cb(cb_zero_tiled_id, program, core_grid, tile_size, 1, data_format);
    }

    uint32_t cb_reduction_tiled_id = 32;
    uint32_t cb_worker_ack_back_id = 32;
    if (C_in_num_blocks > 1) {
        cb_reduction_tiled_id = next_cb_index++;
        tt::tt_metal::create_cb(
            cb_reduction_tiled_id, program, core_grid, partial_tile_size, matmul_M_t * matmul_N_t, partial_data_format);
        cb_worker_ack_back_id = next_cb_index++;
        tt::tt_metal::create_cb(cb_worker_ack_back_id, program, core_grid, tile_size, 1, data_format);
    }

    uint32_t cb_bias_tiled_id = 32;
    if (use_bias) {
        cb_bias_tiled_id = next_cb_index++;
        tt::tt_metal::create_cb(cb_bias_tiled_id, program, core_grid, tile_size, matmul_N_t, data_format);
    }

    bool is_conv3d_padding_zeros = op.padding_mode == "zeros";

    uint32_t in_row_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_row_size_bytes = output_tensor.buffer()->aligned_page_size();

    // --- Upstream conv3d gather tuning (re-based from #43541 / #44418) ---
    // DRAM-read staging: when DRAM pages are read-aligned but the split C-in slice is not,
    // stage small reads through an aligned scratch CB so every read is alignment-safe.
    const uint32_t device_num_dram_banks = static_cast<uint32_t>(input_tensor.device()->num_dram_channels());
    TT_FATAL(device_num_dram_banks > 0, "Device must report at least one DRAM channel");
    const bool input_is_dram_interleaved =
        input_tensor.buffer()->is_dram() &&
        input_tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        !input_tensor.buffer()->buffer_distribution_spec().has_value();
    const uint32_t dram_read_alignment = tt::tt_metal::hal::get_dram_alignment();
    const bool input_pages_are_dram_read_aligned = in_row_size_bytes % dram_read_alignment == 0;
    const bool c_in_slice_is_dram_read_aligned = C_in_block_bytes % dram_read_alignment == 0;
    const bool enable_dram_read_staging =
        input_is_dram_interleaved && input_pages_are_dram_read_aligned && !c_in_slice_is_dram_read_aligned;
    const uint32_t max_staged_dram_window_bytes =
        tt::round_up(C_in_block_bytes + dram_read_alignment - 1, dram_read_alignment);
    const uint32_t dram_read_scratch_page_bytes =
        enable_dram_read_staging ? max_staged_dram_window_bytes + dram_read_alignment : 0;
    uint32_t cb_dram_read_scratch_id = 32;  // Invalid; set below if DRAM read staging is needed
    if (enable_dram_read_staging) {
        cb_dram_read_scratch_id = next_cb_index++;
        tt::tt_metal::create_cb(
            cb_dram_read_scratch_id, program, core_grid, dram_read_scratch_page_bytes, 1, data_format);
    }

    // L1 prefetch buffer
    constexpr uint32_t L1_KERNEL_CODE_RESERVE = 200 * 1024;
    constexpr uint32_t L1_PREFETCH_HARD_CAP = 500 * 1024;
    const uint32_t l1_usable_for_cbs = tt::tt_metal::hal::get_max_worker_l1_unreserved_size() - L1_KERNEL_CODE_RESERVE;

    uint32_t other_cbs_bytes = (padded_patch_size_bytes * vol2col_rm_pages) + (tile_size * matmul_K_t) +
                               (tile_size * matmul_K_t * matmul_N_t) + (partial_tile_size * matmul_M_t * matmul_N_t) +
                               (tile_size * matmul_M_t * matmul_N_t);
    if (enable_dram_read_staging) {
        other_cbs_bytes += dram_read_scratch_page_bytes;
    }
    if (C_in_num_blocks > 1) {
        other_cbs_bytes += partial_tile_size * matmul_M_t * matmul_N_t;
        other_cbs_bytes += tile_size;
    }
    if (use_fp32_partials) {
        other_cbs_bytes += tile_size;
    }
    if (use_bias) {
        other_cbs_bytes += tile_size * matmul_N_t;
    }
    uint32_t l1_prefetch_max_bytes =
        (other_cbs_bytes < l1_usable_for_cbs) ? std::min(l1_usable_for_cbs - other_cbs_bytes, L1_PREFETCH_HARD_CAP) : 0;

    const uint32_t kT = op.kernel_size[0];
    const uint32_t kH = op.kernel_size[1];
    const uint32_t kW = op.kernel_size[2];

    // Coalesced bank-major gather candidate: DRAM interleaved, single C-in block, row-aligned,
    // and enough W columns that each DRAM bank gets multiple pages to amortize the L1 reorder.
    const uint32_t W_shard_full_for_coalesce = (conv_config.W_out_block - 1) * op.stride[2] + kW;
    const uint32_t coalesced_min_w_shard = 2 * device_num_dram_banks;
    const bool coalesced_shard_reads_candidate = input_is_dram_interleaved && C_in_num_blocks == 1 &&
                                                 C_in_block_bytes == in_row_size_bytes &&
                                                 W_shard_full_for_coalesce >= coalesced_min_w_shard;

    uint32_t cb_input_shard_id = 32;
    uint32_t T_shard_max = 0;
    uint32_t H_shard_max = 0;
    uint32_t W_shard_max = 0;
    bool enable_coalesced_shard_reads = false;
    uint32_t coalesced_scratch_rows = 0;

    const bool has_spatial_reuse = (kT > 1 || kH > 1 || kW > 1);
    const bool has_no_dilation = (op.dilation[0] == 1 && op.dilation[1] == 1 && op.dilation[2] == 1);

    if (has_spatial_reuse && has_no_dilation) {
        T_shard_max = (conv_config.T_out_block - 1) * op.stride[0] + kT;
        H_shard_max = (conv_config.H_out_block - 1) * op.stride[1] + kH;
        W_shard_max = (conv_config.W_out_block - 1) * op.stride[2] + kW;
        uint32_t shard_positions_max = T_shard_max * H_shard_max * W_shard_max;
        uint32_t shard_bytes = shard_positions_max * C_in_block_bytes;
        uint32_t shard_rows_max = T_shard_max * H_shard_max;
        uint32_t coalesced_scratch_pages_per_row = W_shard_max;
        uint32_t coalesced_scratch_row_bytes = coalesced_scratch_pages_per_row * C_in_block_bytes;
        uint32_t coalesced_scratch_rows_fit = (coalesced_scratch_row_bytes > 0 && shard_bytes < l1_prefetch_max_bytes)
                                                  ? (l1_prefetch_max_bytes - shard_bytes) / coalesced_scratch_row_bytes
                                                  : 0;
        uint32_t coalesced_scratch_rows_candidate =
            coalesced_shard_reads_candidate ? std::min(shard_rows_max, coalesced_scratch_rows_fit) : 0;
        uint32_t coalesced_scratch_rows_min =
            coalesced_shard_reads_candidate ? std::min(shard_rows_max, device_num_dram_banks) : 0;
        uint32_t coalesced_scratch_positions = coalesced_scratch_rows_candidate * coalesced_scratch_pages_per_row;
        uint32_t shard_bytes_with_coalesced_scratch =
            (shard_positions_max + coalesced_scratch_positions) * C_in_block_bytes;

        if (shard_bytes <= l1_prefetch_max_bytes) {
            enable_coalesced_shard_reads = coalesced_shard_reads_candidate &&
                                           coalesced_scratch_rows_candidate >= coalesced_scratch_rows_min &&
                                           shard_bytes_with_coalesced_scratch <= l1_prefetch_max_bytes;
            coalesced_scratch_rows = enable_coalesced_shard_reads ? coalesced_scratch_rows_candidate : 0;
            const uint32_t shard_positions_alloc =
                shard_positions_max + coalesced_scratch_rows * coalesced_scratch_pages_per_row;
            cb_input_shard_id = next_cb_index++;
            tt::tt_metal::create_cb(
                cb_input_shard_id, program, core_grid, C_in_block_bytes, shard_positions_alloc, data_format);
        } else {
            T_shard_max = 0;
            H_shard_max = 0;
            W_shard_max = 0;
        }
    }

    // Trid-ring depth classifier for the NP fused gather.  The upstream conv3d intensity gate
    // (bytes/matmul-tile >= 128) classifies the NP production shapes as compute-bound and disables
    // the ring, but ablation shows the fused conv is gather-bound on every halo_last shape (skipping
    // the vol2col gather drops device FW 24-39%).  The discriminator that actually predicts a win is
    // the per-row inner-gather burst (T_shard * W_shard): a moderate burst (~18) is latency-bound and
    // the depth-8 ring hides the per-read latency (s1/s2/s3_res: -3 to -4% FW), while a large burst
    // (>= kNpGatherBurstCap) is already NOC-bandwidth-saturated so the ring's drain barriers only add
    // overhead (s0/s4_res: neutral-to-worse).  Scratch-backed reader modes issue larger/serialized
    // reads, so the ring is allowed only outside them.
    const uint32_t reader_T_shard = (conv_config.T_out_block - 1) * op.stride[0] + kT;
    const uint32_t reader_W_shard = (conv_config.W_out_block - 1) * op.stride[2] + kW;
    const uint32_t inner_gather_burst = reader_T_shard * reader_W_shard;
    constexpr uint32_t kNpGatherBurstCap = 64;  // above this the gather is bandwidth-bound; ring off
    const bool gather_trid_ring_allowed = !enable_coalesced_shard_reads && !enable_dram_read_staging;
    uint32_t gather_trids = 0;
    if (gather_trid_ring_allowed && inner_gather_burst >= conv3d_gather_tuning::kGatherInnerBurstCutoff &&
        inner_gather_burst < kNpGatherBurstCap) {
        gather_trids = conv3d_gather_tuning::kGatherTridDepthHigh;
    }

    // Conv3d intra-reduction semaphore (dual-purpose: worker done count / valid bit)
    auto conv3d_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, 0);

    std::vector<uint32_t> reader_compile_time_args = {
        cb_vol2col_rm_id,
        N,
        T_in,
        H_in,
        W_in,
        C_in,
        T_out,
        H_out,
        W_out,
        C_out,
        effective_padding[0],
        effective_padding[1],
        effective_padding[2],
        op.kernel_size[0],
        op.kernel_size[1],
        op.kernel_size[2],
        conv_config.T_out_block,
        conv_config.H_out_block,
        conv_config.W_out_block,
        C_out_num_blocks,
        in_row_size_bytes,
        C_in_block_bytes,
        out_row_size_bytes,
        is_conv3d_padding_zeros,
        conv3d_semaphore_id,
        op.stride[0],
        op.stride[1],
        op.stride[2],
        op.dilation[0],
        op.dilation[1],
        op.dilation[2],
        cb_input_shard_id,
        T_shard_max,
        H_shard_max,
        W_shard_max,
        patch_pad_bytes,
        // Upstream gather tuning (indices 36-41), re-based from conv3d #43541/#44418.
        gather_trids,
        (uint32_t)enable_coalesced_shard_reads,
        coalesced_scratch_rows,
        cb_dram_read_scratch_id,
        (uint32_t)enable_dram_read_staging,
        dram_read_alignment};
    // Per-T-batch pipelining CT arg (index 42).  The reader's RT input_progress_signal_count = the
    // number of receiving W directions (2 on a W-middle device, 1 on an edge), which is how many times
    // each global T-batch is signalled — once per receiving direction, by the link core that owns
    // that batch's rows.  It is INDEPENDENT of pad2_num_links: the per-direction link cores split
    // the T-batches by row, so adding links does not add signals per batch.  Scaling by
    // pad2_num_links over-counts and deadlocks the conv3d reader's last t-block for num_links>1
    // (it waits for a threshold the W-readers never reach).  An edge value (no direction count)
    // would under-count on a middle device and release the reader before both halos land (W seam).
    const uint32_t num_signaling_w_dirs = (is_first_w_device ? 0u : 1u) + (is_last_w_device ? 0u : 1u);
    const uint32_t progress_signal_count = num_signaling_w_dirs;
    reader_compile_time_args.push_back(progress_t_batch_size);            // [42]
    reader_compile_time_args.push_back((uint32_t)conv_config.halo_last);  // [43]
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    auto conv3d_reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/"
        "conv3d_reader_vol2col.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Matmul parameters
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t in0_block_w = matmul_K_t;
    const uint32_t out_subblock_w = std::min(matmul_N_t, dst_size);
    TT_FATAL(matmul_N_t % out_subblock_w == 0, "matmul_N_t must be divisible by out_subblock_w");
    const uint32_t out_subblock_h = 1;
    const uint32_t in0_num_subblocks = 1;
    const uint32_t in1_num_subblocks = matmul_N_t / out_subblock_w;

    std::vector<uint32_t> compute_compile_time_args = {
        cb_vol2col_rm_id,
        cb_vol2col_tiled_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_matmul_result_rm_id,
        cb_reduction_tiled_id,
        cb_worker_ack_back_id,
        N,
        num_patches,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        (uint32_t)use_bias,
        T_out,
        H_out,
        W_out,
        conv_config.T_out_block,
        conv_config.H_out_block,
        conv_config.W_out_block,
        C_out_num_blocks,
        in0_num_subblocks,
        in1_num_subblocks,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        conv3d_semaphore_id,
        (uint32_t)use_fp32_partials,
        cb_zero_tiled_id};

    auto conv3d_compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/"
        "conv3d_compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        cb_matmul_result_rm_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_reduction_tiled_id,
        cb_worker_ack_back_id,
        N,
        T_out,
        H_out,
        W_out,
        conv_config.T_out_block,
        conv_config.H_out_block,
        conv_config.W_out_block,
        C_out_num_blocks,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        num_patches_tile_padded,
        out_row_size_bytes,
        C_out_block_bytes,
        (uint32_t)use_bias,
        conv3d_semaphore_id,
        cb_zero_tiled_id};
    writer_compile_time_args.push_back((uint32_t)conv_config.halo_last);  // [23], before the accessors
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_tensor.has_value() ? bias_tensor.value().buffer() : nullptr)
        .append_to(writer_compile_time_args);

    auto conv3d_writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/"
        "conv3d_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();
    uint32_t weight_addr = weight_tensor.buffer()->address();
    uint32_t bias_addr = bias_tensor.has_value() ? bias_tensor.value().buffer()->address() : 0;
    uint32_t halo_buffer_addr = halo_buffer->address();

    // Parallelism computation (same as conv3d_program_factory.cpp)
    uint32_t T_out_blocks = tt::div_up(T_out, conv_config.T_out_block);
    uint32_t H_out_blocks = tt::div_up(H_out, conv_config.H_out_block);
    uint32_t W_out_blocks = tt::div_up(W_out, conv_config.W_out_block);

    uint32_t c_in_parallel_factor = std::min(C_in_num_blocks, (uint32_t)conv3d_num_cores);
    uint32_t cores_per_output = std::max(1u, (uint32_t)(conv3d_num_cores / c_in_parallel_factor));
    uint32_t c_out_parallel_factor = std::min(C_out_num_blocks, cores_per_output);
    uint32_t remaining_parallel = cores_per_output / c_out_parallel_factor;

    uint32_t t_out_parallel_factor, h_out_parallel_factor, w_out_parallel_factor;
    if (conv_config.force_spatial_parallel) {
        // Every core owns the full t-range (t_out_per_core == T_out_blocks), so the reader's
        // per-t-block halo wait ramps and interior (h,w) tiles skip halo. Split the budget across
        // H and W to fill the grid: greedily maxing H leaves a non-divisible remainder for W
        // (51/34=1 on s4 → 68/102 cores), so pick the (h,w) that wastes the fewest cores.
        t_out_parallel_factor = 1u;
        uint32_t best_h = 1u, best_w = 1u, best_fill = 0u;
        for (uint32_t h = std::min(H_out_blocks, remaining_parallel); h >= 1u; --h) {
            uint32_t w = std::min(W_out_blocks, remaining_parallel / h);
            if (h * w > best_fill) {
                best_fill = h * w;
                best_h = h;
                best_w = w;
            }
        }
        h_out_parallel_factor = best_h;
        w_out_parallel_factor = best_w;
    } else {
        t_out_parallel_factor = std::min(T_out_blocks, remaining_parallel);
        remaining_parallel = remaining_parallel / t_out_parallel_factor;
        h_out_parallel_factor = std::min(H_out_blocks, remaining_parallel);
        remaining_parallel = remaining_parallel / h_out_parallel_factor;
        w_out_parallel_factor = std::min(W_out_blocks, remaining_parallel);
    }

    uint32_t total_output_parallel =
        c_out_parallel_factor * t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor;

    TT_FATAL(
        c_in_parallel_factor * total_output_parallel <= conv3d_num_cores,
        "NpConv3d: Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        c_in_parallel_factor * total_output_parallel,
        conv3d_num_cores);

    if (conv_config.force_spatial_parallel &&
        c_in_parallel_factor * total_output_parallel < (conv3d_num_cores * 3u) / 4u) {
        log_warning(
            tt::LogOp,
            "NpConv3d force_spatial_parallel under-fills grid ({} of {} cores); shape may prefer temporal fill.",
            c_in_parallel_factor * total_output_parallel,
            conv3d_num_cores);
    }

    const uint32_t c_in_per_core = tt::div_up(C_in_num_blocks, c_in_parallel_factor);
    TT_FATAL(
        c_in_per_core == 1,
        "NpConv3d: Each core must handle exactly 1 C_in block, but got c_in_per_core={}.",
        c_in_per_core);

    const uint32_t c_out_per_core = tt::div_up(C_out_num_blocks, c_out_parallel_factor);
    const uint32_t t_out_per_core = tt::div_up(T_out_blocks, t_out_parallel_factor);
    const uint32_t h_out_per_core = tt::div_up(H_out_blocks, h_out_parallel_factor);
    const uint32_t w_out_per_core = tt::div_up(W_out_blocks, w_out_parallel_factor);

    std::vector<std::vector<uint32_t>> reduction_groups(total_output_parallel);
    std::vector<std::vector<uint32_t>> reader_args_per_core(conv3d_num_cores);
    std::vector<std::vector<uint32_t>> compute_args_per_core(conv3d_num_cores);
    std::vector<std::vector<uint32_t>> writer_args_per_core(conv3d_num_cores);
    std::vector<uint32_t> reducer_core_ids(total_output_parallel, UINT32_MAX);
    std::vector<std::vector<uint32_t>> worker_core_ids(total_output_parallel);
    std::vector<uint32_t> reducer_core_physical_xs(total_output_parallel);
    std::vector<uint32_t> reducer_core_physical_ys(total_output_parallel);
    std::vector<std::vector<uint32_t>> worker_core_physical_xs(total_output_parallel);
    std::vector<std::vector<uint32_t>> worker_core_physical_ys(total_output_parallel);

    auto conv3d_cores = corerange_to_cores(core_grid, conv3d_num_cores, true);
    auto* device = input_tensor.device();

    // W-region progress params for the conv3d reader (2D pipelined path): a W-edge tile maps the
    // batch it needs to the owning link via integer division by reader_w_batches_per_link, then polls
    // that (W-left/W-right, link) sem. Mirrors the batch-aligned W partition above.
    const uint32_t reader_w_h_total = input_halo_dim_size + 2 * op.np_padding_h;
    const uint32_t reader_w_sticks_per_batch = progress_t_batch_size * reader_w_h_total;
    const uint32_t reader_total_w_batches =
        (reader_w_sticks_per_batch > 0)
            ? ((outer_dim_size * reader_w_h_total + reader_w_sticks_per_batch - 1) / reader_w_sticks_per_batch)
            : 0;
    const uint32_t reader_pad2_num_links = pad2_num_links;
    const uint32_t reader_w_batches_per_link =
        (reader_total_w_batches > 0) ? ((reader_total_w_batches + pad2_num_links - 1) / pad2_num_links) : 0;

    for (uint32_t core_id = 0; core_id < conv3d_num_cores; ++core_id) {
        CoreCoord core = conv3d_cores.at(core_id);

        uint32_t output_idx = core_id % total_output_parallel;
        uint32_t c_in_idx = core_id / total_output_parallel;

        uint32_t c_out_idx = output_idx / (t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor);
        uint32_t remaining = output_idx % (t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor);
        uint32_t t_out_idx = remaining / (h_out_parallel_factor * w_out_parallel_factor);
        remaining = remaining % (h_out_parallel_factor * w_out_parallel_factor);
        uint32_t h_out_idx = remaining / w_out_parallel_factor;
        uint32_t w_out_idx = remaining % w_out_parallel_factor;

        uint32_t reduction_group_id = output_idx;

        uint32_t c_in_block_start = c_in_idx * c_in_per_core;
        uint32_t c_in_block_end = std::min(c_in_block_start + c_in_per_core, C_in_num_blocks);
        uint32_t c_out_block_start = c_out_idx * c_out_per_core;
        uint32_t c_out_block_end = std::min(c_out_block_start + c_out_per_core, C_out_num_blocks);
        uint32_t t_out_block_start = t_out_idx * t_out_per_core;
        uint32_t t_out_block_end = std::min(t_out_block_start + t_out_per_core, T_out_blocks);
        uint32_t h_out_block_start = h_out_idx * h_out_per_core;
        uint32_t h_out_block_end = std::min(h_out_block_start + h_out_per_core, H_out_blocks);
        uint32_t w_out_block_start = w_out_idx * w_out_per_core;
        uint32_t w_out_block_end = std::min(w_out_block_start + w_out_per_core, W_out_blocks);

        uint32_t t_out_start = t_out_block_start * conv_config.T_out_block;
        uint32_t t_out_end = std::min(t_out_block_end * conv_config.T_out_block, T_out);
        uint32_t h_out_start = h_out_block_start * conv_config.H_out_block;
        uint32_t h_out_end = std::min(h_out_block_end * conv_config.H_out_block, H_out);
        uint32_t w_out_start = w_out_block_start * conv_config.W_out_block;
        uint32_t w_out_end = std::min(w_out_block_end * conv_config.W_out_block, W_out);

        bool has_work = (c_in_block_end > c_in_block_start) && (c_out_block_end > c_out_block_start) &&
                        (t_out_end > t_out_start) && (h_out_end > h_out_start) && (w_out_end > w_out_start);

        bool is_reducer = has_work && c_in_idx == 0;

        if (has_work) {
            reduction_groups[reduction_group_id].push_back(core_id);
            if (is_reducer) {
                reducer_core_ids[reduction_group_id] = core_id;
                auto reducer_core_physical = device->worker_core_from_logical_core(core);
                reducer_core_physical_xs[reduction_group_id] = (uint32_t)reducer_core_physical.x;
                reducer_core_physical_ys[reduction_group_id] = (uint32_t)reducer_core_physical.y;
            } else {
                worker_core_ids[reduction_group_id].push_back(core_id);
                auto worker_core_physical = device->worker_core_from_logical_core(core);
                worker_core_physical_xs[reduction_group_id].push_back((uint32_t)worker_core_physical.x);
                worker_core_physical_ys[reduction_group_id].push_back((uint32_t)worker_core_physical.y);
            }
        }

        // Reader args[0..10] match conv3d_program_factory.cpp layout exactly; [11..] are fused-only:
        //   [0]  input_addr
        //   [1]  c_in_block_start
        //   [2]  c_in_block_end
        //   [3]  c_out_block_start
        //   [4]  c_out_block_end
        //   [5]  t_out_start
        //   [6]  t_out_end
        //   [7]  h_out_start
        //   [8]  h_out_end
        //   [9]  w_out_start
        //   [10] w_out_end
        //   [11] input_progress_signal_count (per-batch count = num receiving W directions)
        //   [12] h_halo_buffer_addr
        //   [13] h_halo_outer_dim_size
        //   [14] h_halo_H
        //   [15] h_halo_W
        //   [16] h_halo_padding_h
        //   [17] h_halo_padding_w
        reader_args_per_core[core_id] = {
            input_addr,
            c_in_block_start,
            c_in_block_end,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end,
            // [11]: input_progress_signal_count — number of W-readers signalling this
            //   conv3d reader per batch; scales with the device's real W-neighbor count
            //   (see progress_signal_count above), not the edge-only num_w_fabric_cores/2.
            progress_signal_count,
            // [12]: halo buffer DRAM address
            halo_buffer_addr,
            h_halo_outer_dim_size,
            h_halo_H,
            h_halo_W,
            h_halo_padding_h,
            h_halo_padding_w,
        };
        // [18] pad2_num_links, [19] w_batches_per_link, [20..23] W-left sems, [24..27] W-right sems.
        // A W side with no real neighbor (mesh edge) has zero-pad halo and NO producer (the W-reader
        // for that direction is the sender, gated off from signalling), so pass addr 0 and the
        // consumer skips it — else it would wait a sem that never increments and deadlock. WL is
        // produced unless is_first_w_device (no left neighbor); WR unless is_last_w_device.
        const bool have_wleft = !is_first_w_device;
        const bool have_wright = !is_last_w_device;
        const uint32_t region_stride = region_progress_num_links;  // = num_links; table stride
        reader_args_per_core[core_id].push_back(reader_pad2_num_links);
        reader_args_per_core[core_id].push_back(reader_w_batches_per_link);
        for (uint32_t l = 0; l < 4u; l++) {
            const bool valid = have_wleft && l < region_stride;
            reader_args_per_core[core_id].push_back(
                valid ? conv_config.region_progress_sem_addr[2 * region_stride + l] : 0u);
        }
        for (uint32_t l = 0; l < 4u; l++) {
            const bool valid = have_wright && l < region_stride;
            reader_args_per_core[core_id].push_back(
                valid ? conv_config.region_progress_sem_addr[3 * region_stride + l] : 0u);
        }
        reader_args_per_core[core_id].push_back(reader_total_w_batches);  // [28] W threshold cap
        // [29..39]: H-region sems (all H-links) + params for H-edge tiles. HT produced unless
        // is_first_device (no top neighbor); HB unless is_last_device. [29..32] HT, [33..36] HB,
        // [37] h_bpl, [38] num_h_links, [39] h_total_batches.
        const bool have_htop = !is_first_device;
        const bool have_hbot = !is_last_device;
        for (uint32_t l = 0; l < 4u; l++) {
            const bool ok = have_htop && l < num_links;
            reader_args_per_core[core_id].push_back(
                ok ? conv_config.region_progress_sem_addr[0 * region_stride + l] : 0u);
        }
        for (uint32_t l = 0; l < 4u; l++) {
            const bool ok = have_hbot && l < num_links;
            reader_args_per_core[core_id].push_back(
                ok ? conv_config.region_progress_sem_addr[1 * region_stride + l] : 0u);
        }
        reader_args_per_core[core_id].push_back(h_batches_per_link);
        reader_args_per_core[core_id].push_back(num_links);
        reader_args_per_core[core_id].push_back(h_total_batches);

        compute_args_per_core[core_id] = {
            c_in_block_start,
            c_in_block_end,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end,
            (uint32_t)is_reducer};

        writer_args_per_core[core_id] = {
            out_addr,
            weight_addr,
            bias_addr,
            c_in_block_start,
            c_in_block_end,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end,
            (uint32_t)is_reducer};
    }

    // Second loop: set runtime args with reducer and worker information
    for (uint32_t core_id = 0; core_id < conv3d_num_cores; ++core_id) {
        CoreCoord core = conv3d_cores.at(core_id);
        uint32_t output_idx = core_id % total_output_parallel;
        uint32_t reduction_group_id = output_idx;

        auto& reader_args = reader_args_per_core[core_id];
        auto& compute_args = compute_args_per_core[core_id];
        auto& writer_args = writer_args_per_core[core_id];

        uint32_t num_workers = worker_core_ids[reduction_group_id].size();
        compute_args.push_back(num_workers);
        writer_args.push_back(num_workers);

        if (num_workers > 0) {
            writer_args.push_back(reducer_core_physical_xs[reduction_group_id]);
            writer_args.push_back(reducer_core_physical_ys[reduction_group_id]);
            writer_args.insert(
                writer_args.end(),
                worker_core_physical_xs[reduction_group_id].begin(),
                worker_core_physical_xs[reduction_group_id].end());
            writer_args.insert(
                writer_args.end(),
                worker_core_physical_ys[reduction_group_id].begin(),
                worker_core_physical_ys[reduction_group_id].end());
        }

        SetRuntimeArgs(program, conv3d_reader_kernels_id, core, reader_args);
        SetRuntimeArgs(program, conv3d_compute_kernels_id, core, compute_args);
        SetRuntimeArgs(program, conv3d_writer_kernels_id, core, writer_args);
    }

    return cached_program_t{
        std::move(program),
        NpConv3dSharedVariables{
            .np_artifacts =
                NpFabricOnlyArtifacts{
                    h_reader_kernel_id,
                    h_writer_kernel_id,
                    w_reader_kernel_id,
                    w_writer_kernel_id,
                    /*has_w_fabric=*/true,
                    w_fabric_core_range},
            .conv3d_num_cores = conv3d_num_cores,
            .conv3d_cores = conv3d_cores,
            .conv3d_reader_kernel_id = conv3d_reader_kernels_id,
            .conv3d_writer_kernel_id = conv3d_writer_kernels_id,
            .conv3d_compute_kernel_id = conv3d_compute_kernels_id}};
}

// ============================================================================
// override_runtime_arguments — update addresses for NP kernels + conv3d kernels
// ============================================================================
void NpConv3dMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const NpConv3dParams& op,
    const NpConv3dInputs& tensor_args,
    Tensor& tensor_return_value) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t halo_buffer_addr = tensor_args.halo_buffer.buffer()->address();
    const uint32_t output_addr = tensor_return_value.buffer()->address();
    const uint32_t weight_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t bias_addr =
        tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
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

        // --- Conv3d reader: update input_addr (args[0]) and halo_buffer_addr (args[12]) ---
        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.conv3d_reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.conv3d_writer_kernel_id);

        for (uint32_t i = 0; i < shared_vars.conv3d_num_cores; ++i) {
            CoreCoord core = shared_vars.conv3d_cores.at(i);
            auto& reader_args = reader_args_by_core[core.x][core.y];
            auto& writer_args = writer_args_by_core[core.x][core.y];

            reader_args[0] = input_addr;
            // args[12] = halo buffer DRAM address — changes per call (ping-pong buffer). args[13..17]
            // are the halo geometry, which is derived from the (hash-pinned) input shape in create_at
            // and is constant for a cached program, so it is not refreshed here.
            reader_args[12] = halo_buffer_addr;

            writer_args[0] = output_addr;
            writer_args[1] = weight_addr;
            writer_args[2] = bias_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
