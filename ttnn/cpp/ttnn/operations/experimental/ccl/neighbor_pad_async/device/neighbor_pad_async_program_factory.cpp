// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "neighbor_pad_async_program_factory.hpp"

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
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <optional>
#include <ranges>
#include <sstream>
#include <type_traits>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

// ---------------------------------------------------------------------------
// Helper: build_np_fabric_only_program_artifacts
//
// Adds H fabric (and optionally W fabric) kernels to an existing program.
// This is the NP equivalent of build_all_gather_async_minimal_default_program_artifacts().
// ---------------------------------------------------------------------------
NpFabricOnlyArtifacts build_np_fabric_only_program_artifacts(
    Program& program,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::optional<ttnn::MeshCoordinate>& forward_coord,
    const std::optional<ttnn::MeshCoordinate>& backward_coord,
    uint32_t device_index,
    Buffer* input_buffer,
    Buffer* halo_buffer,
    // NP params
    uint32_t np_dim,
    uint32_t padding_left,
    uint32_t padding_right,
    const std::string& padding_mode,
    uint32_t ring_size,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& h_neighbor_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    uint32_t num_links,
    // Shape-derived params
    uint32_t input_halo_dim_size,
    uint32_t num_sticks_per_halo_dim,
    uint32_t outer_dim_size,
    uint32_t output_halo_dim_size,
    bool fabric_only,
    tt::DataFormat df,
    // Optional 2D W-axis params (pad_dim2 != nullopt enables 2D mode)
    std::optional<uint32_t> pad_dim2,
    uint32_t pad2_left,
    uint32_t pad2_right,
    std::optional<uint32_t> pad2_cluster_axis,
    uint32_t pad2_num_links,
    const GlobalSemaphore& w_neighbor_semaphore,
    // Pre-computed W-axis topology (ignored when pad_dim2 == nullopt)
    const std::optional<MeshCoordinate>& w_forward_coord_in,
    const std::optional<MeshCoordinate>& w_backward_coord_in,
    uint32_t w_device_index_in,
    // Progress semaphore for T-batch pipelining: H-writer signals per T-batch, W-reader signals once at end.
    uint32_t progress_sem_addr,
    const std::vector<std::pair<uint32_t, uint32_t>>& reader_noc_coords,
    uint32_t progress_t_batch_size) {
    // Use the buffer's aligned page size (architecture-specific: 32B on WH, 64B on BH).
    uint32_t page_size = input_buffer->aligned_page_size();

    // Device position from physical neighbor availability
    bool is_first_device = !backward_coord.has_value();
    bool is_last_device = !forward_coord.has_value();
    uint32_t forward_device_offset = is_last_device ? 0 : 1;
    uint32_t backward_device_offset = is_first_device ? 0 : 1;

    log_trace(
        tt::LogOp,
        "NeighborPad H-fabric: mesh_coord=({},{}), device_index={}, src_node_id={}, "
        "fwd_offset={}, bwd_offset={}, is_first={}, is_last={}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        device_index,
        mesh_device->get_fabric_node_id(mesh_coordinate),
        forward_device_offset,
        backward_device_offset,
        is_first_device,
        is_last_device);
    if (forward_coord.has_value()) {
        log_trace(
            tt::LogOp,
            "  forward_coord=({},{}), fwd_node_id={}",
            (*forward_coord)[0],
            (*forward_coord)[1],
            mesh_device->get_fabric_node_id(forward_coord.value()));
    }
    if (backward_coord.has_value()) {
        log_trace(
            tt::LogOp,
            "  backward_coord=({},{}), bwd_node_id={}",
            (*backward_coord)[0],
            (*backward_coord)[1],
            mesh_device->get_fabric_node_id(backward_coord.value()));
    }

    bool is_padding_zeros = padding_mode == "zeros";
    const bool is_2d = pad_dim2.has_value();

    // For 2D padding: compute secondary dimension metrics
    uint32_t output_num_sticks_per_halo_dim = num_sticks_per_halo_dim;  // default: same as input
    uint32_t writer_stick_start_id = 0;
    uint32_t writer_num_sticks_to_read = num_sticks_per_halo_dim;
    if (is_2d) {
        output_num_sticks_per_halo_dim = num_sticks_per_halo_dim + pad2_left + pad2_right;
        writer_stick_start_id = pad2_left;
    }

    // W-writer signals conv3d readers once after completing all W-halo work.
    // progress_sem_addr != 0 enables this signaling; reader_noc_coords is the target list.

    // Get worker cores
    constexpr uint32_t MAX_PAD2_NUM_LINKS = 4;

    // Cap num_links and pad2_num_links so total fabric cores fit within the device compute grid width.
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t capped_num_links = num_links;
    uint32_t capped_pad2_num_links = pad2_num_links;
    uint32_t total_fabric_cores = (capped_num_links * 2) + (is_2d ? capped_pad2_num_links * 2 : 0);
    if (total_fabric_cores > compute_grid_size.x) {
        uint32_t max_total = compute_grid_size.x;
        uint32_t h_cores = capped_num_links * 2;
        if (is_2d) {
            uint32_t available_for_w = (max_total > h_cores) ? (max_total - h_cores) : 0;
            capped_pad2_num_links = available_for_w / 2;
            if (capped_pad2_num_links == 0) {
                capped_pad2_num_links = 1;
                capped_num_links = (max_total - 2) / 2;
            }
        } else {
            capped_num_links = max_total / 2;
        }
        log_warning(
            tt::LogOp,
            "neighbor_pad_async: Capped num_links from {} to {} and pad2_num_links from {} to {} "
            "to fit device compute grid width {}",
            num_links,
            capped_num_links,
            pad2_num_links,
            capped_pad2_num_links,
            compute_grid_size.x);
    }

    uint32_t num_h_fabric_cores = capped_num_links * 2;
    uint32_t num_w_fabric_cores = is_2d ? (capped_pad2_num_links * 2) : 0;
    TT_FATAL(
        capped_pad2_num_links <= MAX_PAD2_NUM_LINKS,
        "pad2_num_links ({}) exceeds maximum supported ({}). Kernel Phase 2 signal target arrays are sized for {}.",
        capped_pad2_num_links,
        MAX_PAD2_NUM_LINKS,
        MAX_PAD2_NUM_LINKS * 2);
    CoreCoord core_grid(num_h_fabric_cores, 1);
    auto [num_cores, worker_core_ranges, core_group_1, core_group_2, dims_per_core_group_1, dims_per_core_group_2] =
        (np_dim > 0) ? split_work_to_cores(core_grid, outer_dim_size * 2)
                     : split_work_to_cores(core_grid, num_sticks_per_halo_dim * 2);

    // L1 Scratch CB Creation
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_sticks_to_write_per_packet = 1;
    uint32_t cb_num_pages = 2 * num_sticks_to_write_per_packet;  // double buffering

    // CBs for transferring data between reader and writer
    uint32_t sender_cb_index = tt::CB::c_in0;
    CircularBufferConfig cb_sender_config =
        CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
            .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, worker_core_ranges, cb_sender_config);

    // L1 receive buffer for 2D padding: fabric-delivered H halo corner sticks arrive here.
    // Corners-only: only W-boundary sticks (pad2_left + pad2_right per row) go to L1.
    // Non-corner sticks go directly to compact buffer DRAM via fabric.
    // NOTE: On BH, fabric DRAM writes do not provide ordering guarantees relative to
    // subsequent NOC reads from a different source (fabric router vs BRISC NOC). As a result,
    // non-corner H-halo sticks (pages 1..W-2) may not be committed before conv3d reads them,
    // causing a PCC gap (~0.05%) between fused and old-halo paths. A correct fix would route
    // all sticks through L1, but this requires per-link semaphores to avoid a semaphore race
    // condition with num_links=2. Tracked as a known limitation.
    uint32_t recv_cb_index = tt::CB::c_in1;
    uint32_t corner_sticks_per_row = is_2d ? std::min(pad2_left + pad2_right, num_sticks_per_halo_dim) : 0;
    if (is_2d) {
        uint32_t max_padding = std::max(padding_left, padding_right);
        uint32_t max_outer_dims_per_core = dims_per_core_group_1;
        uint32_t recv_total_sticks = max_outer_dims_per_core * max_padding * corner_sticks_per_row;
        uint32_t recv_buf_size = recv_total_sticks * page_size;
        if (recv_buf_size > 0) {
            CircularBufferConfig recv_cb_config =
                CircularBufferConfig(recv_buf_size, {{recv_cb_index, df}}).set_page_size(recv_cb_index, page_size);
            CreateCircularBuffer(program, worker_core_ranges, recv_cb_config);
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

    if (is_2d) {
        // W-axis device topology (pre-computed by caller)
        w_forward_coord = w_forward_coord_in;
        w_backward_coord = w_backward_coord_in;

        is_first_w_device = !w_backward_coord.has_value();
        is_last_w_device = !w_forward_coord.has_value();
        if (w_forward_coord.has_value()) {
            w_forward_device_offset = 1;
        }
        if (w_backward_coord.has_value()) {
            w_backward_device_offset = 1;
        }

        log_trace(
            tt::LogOp,
            "NeighborPad W-fabric: mesh_coord=({},{}), "
            "w_fwd_offset={}, w_bwd_offset={}, is_first_w={}, is_last_w={}",
            mesh_coordinate[0],
            mesh_coordinate[1],
            w_forward_device_offset,
            w_backward_device_offset,
            is_first_w_device,
            is_last_w_device);

        // W fabric core coordinates (placed after H fabric cores in first row)
        for (uint32_t i = 0; i < num_w_fabric_cores; i++) {
            CoreCoord wc = {num_h_fabric_cores + i, 0};
            w_fabric_logical_cores.push_back(wc);
            w_fabric_virtual_cores.push_back(mesh_device->worker_core_from_logical_core(wc));
        }
        w_fabric_core_range =
            CoreRangeSet(CoreRange({num_h_fabric_cores, 0}, {num_h_fabric_cores + num_w_fabric_cores - 1, 0}));

        w_outer_dim_size = fabric_only ? outer_dim_size * (input_halo_dim_size + 2 * padding_left)
                                       : outer_dim_size * output_halo_dim_size;

        // CB on W fabric cores (sender CB)
        CreateCircularBuffer(program, w_fabric_core_range, cb_sender_config);

        w_rows_per_link = w_outer_dim_size / capped_pad2_num_links;
        w_extra_rows = w_outer_dim_size % capped_pad2_num_links;

        // W-halo L1 recv buffer: fabric writes W-halo to L1 instead of DRAM to guarantee
        // ordering (BH DRAM writes from fabric are not ordered with local NOC reads).
        // The sender writes to the neighbor's L1 recv_buf; the receiver (phase2_w_reader)
        // drains from L1 to DRAM locally with noc_async_write_barrier() before signaling
        // the progress semaphore, guaranteeing DRAM commit before conv3d reads.
        if (fabric_only) {
            uint32_t max_w_sticks_per_link = w_rows_per_link + (w_extra_rows > 0 ? 1 : 0);
            uint32_t max_w_padding = std::max(pad2_left, pad2_right);
            uint32_t w_recv_buf_size = max_w_sticks_per_link * max_w_padding * page_size;
            if (w_recv_buf_size > 0) {
                CircularBufferConfig w_recv_cb_config = CircularBufferConfig(w_recv_buf_size, {{recv_cb_index, df}})
                                                            .set_page_size(recv_cb_index, page_size);
                CreateCircularBuffer(program, w_fabric_core_range, w_recv_cb_config);
            }
        }
    }

    // Compute H fabric unicast route configuration (for compile-time args)
    auto [h_unicast_forward_args, h_unicast_backward_args] =
        ::ttnn::ccl::get_forward_backward_line_unicast_configuration(
            mesh_coordinate, forward_coord, backward_coord, mesh_device);

    // Compute H fabric multicast barrier route configuration
    auto [num_targets_forward, num_targets_backward] =
        ::ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, device_index, topology, false);
    auto [h_mcast_forward_args, h_mcast_backward_args] = ::ttnn::ccl::get_forward_backward_line_mcast_configuration(
        mesh_coordinate, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    // KERNEL CREATION
    uint32_t num_directions = 2;

    // Create consolidated H fabric reader kernel (uniform compile args across all H cores)
    auto h_reader_kernel_config = ReaderDataMovementConfig{};
    h_reader_kernel_config.compile_args = {
        sender_cb_index,   // cb_output_id
        is_padding_zeros,  // is_padding_zeros
        page_size};        // stick_size
    TensorAccessorArgs(*input_buffer).append_to(h_reader_kernel_config.compile_args);
    h_reader_kernel_config.compile_args.push_back(is_2d ? 1 : 0);              // use_l1_intermediate
    h_reader_kernel_config.compile_args.push_back(is_2d ? recv_cb_index : 0);  // recv_cb_id
    auto h_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
        "minimal_default_reader.cpp",
        worker_core_ranges,
        h_reader_kernel_config);
    SetCommonRuntimeArgs(
        program, h_reader_kernel_id, {input_buffer->address(), halo_buffer->address(), h_neighbor_semaphore.address()});

    // Create consolidated H fabric writer kernel (uniform compile args across all H cores)
    auto h_writer_kernel_config = WriterDataMovementConfig{};
    h_writer_kernel_config.compile_args = {
        sender_cb_index,   // cb_output_id
        is_padding_zeros,  // is_padding_zeros
        page_size};        // stick_size
    TensorAccessorArgs(*halo_buffer).append_to(h_writer_kernel_config.compile_args);
    h_writer_kernel_config.compile_args.push_back(is_2d ? 1 : 0);              // use_l1_intermediate
    h_writer_kernel_config.compile_args.push_back(is_2d ? recv_cb_index : 0);  // recv_cb_id
    h_writer_kernel_config.compile_args.push_back(is_2d ? 1 : 0);              // handle_incoming_writes
    h_writer_kernel_config.compile_args.push_back(0);                          // is_w_fabric_writer (false for H)
    h_writer_kernel_config.compile_args.push_back(ring_size);                  // ring_size
    // H-writer signals conv3d readers per T-batch when progress semaphore is active.
    // W-reader additionally signals once at end (after W-halo completes). The extra W-reader signal
    // is harmless since Conv3d uses noc_semaphore_wait_min (>= threshold, not == threshold).
    if (progress_sem_addr != 0) {
        h_writer_kernel_config.defines["NP_PROGRESS_SEM"] = "1";
    }
    h_writer_kernel_config.compile_args.push_back(progress_t_batch_size);  // progress_t_batch_size (0 if no pipelining)
    auto h_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
        "minimal_default_writer.cpp",
        worker_core_ranges,
        h_writer_kernel_config);
    {
        // Build H writer common runtime args.
        // H-writer signals conv3d readers per T-batch when pipelining is enabled (NP_PROGRESS_SEM defined).
        // W-reader handles final W-halo ordering signal; H-writer handles per-T-batch progress.
        std::vector<uint32_t> h_writer_crta = {
            input_buffer->address(),
            halo_buffer->address(),
            h_neighbor_semaphore.address(),
            barrier_semaphore.address(),
            progress_sem_addr,                                // progress_sem (0 if no pipelining)
            static_cast<uint32_t>(reader_noc_coords.size()),  // num_reader_cores
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
    for (uint32_t link = 0; link < capped_num_links; link++) {
        uint32_t link_dims_to_read = 0;

        for (uint32_t direction = 0; direction < num_directions; direction++) {
            CoreCoord core = {link * num_directions + direction, 0};
            CoreCoord opposite_core = {(link * num_directions) + (1 - direction), 0};
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
            CoreCoord virtual_opposite_core = mesh_device->worker_core_from_logical_core(opposite_core);
            if (core_group_1.contains(core)) {
                link_dims_to_read = dims_per_core_group_1;
            } else {
                link_dims_to_read = dims_per_core_group_2;
            }

            // Reader runtime args
            std::vector<uint32_t> reader_rt_args = {
                (np_dim > 0) ? link_offset_start_id * input_halo_dim_size : outer_dim_size - 1,
                (np_dim == 0) ? link_offset_start_id : 0,
                input_halo_dim_size,
                (np_dim > 0) ? link_dims_to_read : outer_dim_size,
                direction ? padding_right : padding_left,
                (np_dim == 0) ? link_dims_to_read : num_sticks_per_halo_dim,
                num_sticks_per_halo_dim,
                corner_sticks_per_row};
            reader_rt_args.push_back(direction ? is_last_device : is_first_device);
            reader_rt_args.push_back(direction ? is_first_device : is_last_device);
            reader_rt_args.push_back(direction);
            SetRuntimeArgs(program, h_reader_kernel_id, {core}, reader_rt_args);

            // For 2D case, H fabric writer uses output row width and W offset
            uint32_t h_writer_num_sticks_per_halo_dim =
                is_2d ? output_num_sticks_per_halo_dim : num_sticks_per_halo_dim;
            uint32_t h_writer_stick_start = (np_dim == 0) ? link_offset_start_id : writer_stick_start_id;
            uint32_t h_writer_num_sticks_to_read = (np_dim == 0) ? link_dims_to_read : writer_num_sticks_to_read;

            // Writer runtime args
            std::vector<uint32_t> writer_rt_args = {
                (np_dim > 0) ? writer_link_offset_start_id * output_halo_dim_size : outer_dim_size - 1,
                h_writer_stick_start,
                input_halo_dim_size,
                output_halo_dim_size,
                (np_dim > 0) ? link_dims_to_read : outer_dim_size,
                direction ? padding_right : padding_left,
                padding_left,
                h_writer_num_sticks_to_read,
                h_writer_num_sticks_per_halo_dim,
                virtual_core.x,
                virtual_core.y,
                true,
                virtual_opposite_core.x,
                virtual_opposite_core.y};
            // Phase 2 signal targets: H-writers signal W-reader cores after all H work is done.
            // W-reader waits for this barrier before reading H-halo corners from the compact buffer.
            // This guarantees H-halo is fully committed (fabric.close() has run on all H-chips).
            constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 8;
            const uint32_t phase2_targets = is_2d ? num_w_fabric_cores : 0;
            writer_rt_args.push_back(phase2_targets);
            for (uint32_t s = 0; s < MAX_PHASE2_SIGNAL_TARGETS; s++) {
                if (s < phase2_targets) {
                    writer_rt_args.push_back(w_fabric_virtual_cores[s].x);
                    writer_rt_args.push_back(w_fabric_virtual_cores[s].y);
                } else {
                    writer_rt_args.push_back(0);
                    writer_rt_args.push_back(0);
                }
            }
            // Per-core direction and routing args
            writer_rt_args.push_back(direction ? is_last_device : is_first_device);
            writer_rt_args.push_back(direction ? is_first_device : is_last_device);
            writer_rt_args.push_back(direction);
            const auto& h_unicast_args = direction ? h_unicast_backward_args : h_unicast_forward_args;
            writer_rt_args.insert(writer_rt_args.end(), h_unicast_args.begin(), h_unicast_args.end());
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
            // In fabric_only mode: override writer rt_args to index into the compact halo buffer.
            // In fabric_only mode: only corner sticks (pad2_left + pad2_right per row) go through
            // L1→CB→local DRAM to guarantee ordering. Non-corner sticks go directly to DRAM via fabric
            // (BH ordering limitation: tracked as known issue). All-L1 routing would require per-link
            // semaphores to avoid a race with num_links=2 — deferred to future work.
            //   stick_start_id=0: no W-offset in compact buffer
            //   num_sticks_per_halo_dim=W: compact buffer row width (not W+pad)
            //   padding_left kept at pad2_left: gives pad2_left_sticks=1, pad2_right_sticks=1 (corners only)
            if (fabric_only) {
                // Compact buffer layout: [H-top: outer_dim_size×ph×W | H-bot: same | W-left | W-right]
                // outer_dim_size (function param) = total T-slices per device.
                // Each link handles link_dims_to_read T-slices starting at its own T-offset.
                // link_t_start: number of T-slices handled by links before this one.
                // writer_link_offset_start_id counts in output-row units (×output_num_sticks_per_halo_dim each),
                // so dividing recovers the T-slice count.
                uint32_t link_t_start = (output_num_sticks_per_halo_dim > 0)
                                            ? (writer_link_offset_start_id / output_num_sticks_per_halo_dim)
                                            : 0u;
                uint32_t top_halo_total = outer_dim_size * padding_left * num_sticks_per_halo_dim;
                uint32_t h_top_link_start = link_t_start * padding_left * num_sticks_per_halo_dim;
                uint32_t h_bot_link_start = top_halo_total + link_t_start * padding_right * num_sticks_per_halo_dim;
                uint32_t padding_this_dir = direction ? padding_right : padding_left;

                writer_rt_args[0] = direction ? h_bot_link_start : h_top_link_start;  // outer_dim_offset_start_id
                writer_rt_args[1] = 0;                                                // stick_start_id: 0 (no W-offset)
                writer_rt_args[3] = padding_this_dir;                                 // output_halo_dim_size (compact)
                // [6] padding_left: keep at pad2_left for corner count (pad2_left_sticks=1, pad2_right_sticks=1)
                writer_rt_args[8] = num_sticks_per_halo_dim;  // compact row width = W
            }
            SetRuntimeArgs(program, h_writer_kernel_id, {core}, writer_rt_args);
        }
        if (np_dim > 0) {
            link_offset_start_id += (link_dims_to_read * num_sticks_per_halo_dim);
            writer_link_offset_start_id += (link_dims_to_read * output_num_sticks_per_halo_dim);
        } else {
            link_offset_start_id += link_dims_to_read;
            writer_link_offset_start_id += link_dims_to_read;
        }
    }

    // Phase 2: W fabric kernel creation (for 2D padding)
    KernelHandle w_reader_kernel_id = 0;
    KernelHandle w_writer_kernel_id = 0;
    if (is_2d) {
        // barrier_count: how many Phase 2 signals W-reader waits for before starting.
        // barrier_count = num_h_fabric_cores: W-reader waits for all H-writers to complete
        // before starting W-halo exchange. Phase 2 barrier guarantees H-halo is fully
        // committed to the compact buffer (fabric.close() has run on all H-chips).
        uint32_t barrier_count = num_h_fabric_cores;
        log_trace(
            tt::LogOp,
            "NeighborPad2D (helper): barrier_count={} (h_writers={}), "
            "w_outer_dim_size={}, is_first_h={}, is_last_h={}, is_first_w={}, is_last_w={}, "
            "output_row_width={}, num_interior_sticks={}, pad2_left={}",
            barrier_count,
            num_h_fabric_cores,
            w_outer_dim_size,
            is_first_device,
            is_last_device,
            is_first_w_device,
            is_last_w_device,
            output_num_sticks_per_halo_dim,
            num_sticks_per_halo_dim,
            pad2_left);

        // W-axis startup barrier (w_device_index pre-computed by caller)
        const auto& mesh_view_w = mesh_device->get_view();
        uint32_t w_ring_size = (pad2_cluster_axis.value() == 0) ? mesh_view_w.num_rows() : mesh_view_w.num_cols();
        uint32_t w_device_index = w_device_index_in;
        auto [w_num_targets_forward, w_num_targets_backward] =
            ::ttnn::ccl::get_forward_backward_line_mcast_distance(w_ring_size, w_device_index, topology, false);
        auto [w_mcast_forward_args, w_mcast_backward_args] = ::ttnn::ccl::get_forward_backward_line_mcast_configuration(
            mesh_coordinate,
            w_forward_coord,
            w_backward_coord,
            w_num_targets_forward,
            w_num_targets_backward,
            mesh_device);

        // Create consolidated W fabric reader kernel
        auto w_reader_kernel_config = ReaderDataMovementConfig{};
        w_reader_kernel_config.compile_args = {sender_cb_index, is_padding_zeros, page_size};
        TensorAccessorArgs(*halo_buffer).append_to(w_reader_kernel_config.compile_args);
        TensorAccessorArgs(*input_buffer).append_to(w_reader_kernel_config.compile_args);
        // NP_W_HALO_L1: W-halo sticks go through L1 on receive side (ordering fix).
        // recv_cb_id compile arg is added after src TensorAccessorArgs.
        if (fabric_only) {
            w_reader_kernel_config.compile_args.push_back(recv_cb_index);  // recv_cb_id
            w_reader_kernel_config.defines["NP_W_HALO_L1"] = "1";
        }
        // W-reader signals conv3d readers after w_nbr_sem wait (NP_PROGRESS_SEM).
        // Must define BEFORE CreateKernel so the kernel compiles with it.
        if (progress_sem_addr != 0) {
            w_reader_kernel_config.defines["NP_PROGRESS_SEM"] = "1";
        }
        w_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
            "phase2_w_reader.cpp",
            w_fabric_core_range,
            w_reader_kernel_config);
        {
            // W-reader CRTA: [0]=halo_buf, [1]=barrier_sem, [2]=w_neighbor_sem
            // With NP_PROGRESS_SEM: [3]=progress_sem_addr, [4]=num_reader_cores, [5+]=NOC coords.
            // W-reader signals conv3d after w_nbr_sem wait guarantees W-halo is committed.
            std::vector<uint32_t> w_reader_crta = {
                halo_buffer->address(), barrier_semaphore.address(), w_neighbor_semaphore.address()};
            if (progress_sem_addr != 0) {
                w_reader_crta.push_back(progress_sem_addr);
                w_reader_crta.push_back(static_cast<uint32_t>(reader_noc_coords.size()));
                for (const auto& [x, y] : reader_noc_coords) {
                    w_reader_crta.push_back(x);
                    w_reader_crta.push_back(y);
                }
            }
            SetCommonRuntimeArgs(program, w_reader_kernel_id, w_reader_crta);
        }
        // Create consolidated W fabric writer kernel
        auto w_writer_kernel_config = WriterDataMovementConfig{};
        w_writer_kernel_config.compile_args = {sender_cb_index, is_padding_zeros, page_size};
        TensorAccessorArgs(*halo_buffer).append_to(w_writer_kernel_config.compile_args);
        // use_l1_intermediate=1 in fabric_only: W-halo sticks go to neighbor's L1 (not DRAM)
        // to guarantee BH DRAM ordering (see W recv_cb comment above).
        w_writer_kernel_config.compile_args.push_back(fabric_only ? 1 : 0);              // use_l1_intermediate
        w_writer_kernel_config.compile_args.push_back(fabric_only ? recv_cb_index : 0);  // recv_cb_id
        w_writer_kernel_config.compile_args.push_back(0);  // handle_incoming_writes (data goes direct to DRAM)
        w_writer_kernel_config.compile_args.push_back(1);  // is_w_fabric_writer
        w_writer_kernel_config.compile_args.push_back(w_ring_size);  // ring_size
        // W-writer no longer signals conv3d (W-reader handles this after w_nbr_sem wait).
        w_writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
            "minimal_default_writer.cpp",
            w_fabric_core_range,
            w_writer_kernel_config);
        {
            std::vector<uint32_t> w_writer_crta = {
                halo_buffer->address(),
                halo_buffer->address(),
                w_neighbor_semaphore.address(),
                h_neighbor_semaphore.address(),
            };
            SetCommonRuntimeArgs(program, w_writer_kernel_id, w_writer_crta);
        }

        // Set per-core runtime args for W fabric cores
        for (uint32_t w_link = 0; w_link < capped_pad2_num_links; w_link++) {
            uint32_t w_link_start = (w_link * w_rows_per_link) + std::min(w_link, w_extra_rows);
            uint32_t w_link_count = w_rows_per_link + (w_link < w_extra_rows ? 1 : 0);
            log_trace(
                tt::LogOp,
                "NeighborPad2D W-link {}: start={}, count={} (of {})",
                w_link,
                w_link_start,
                w_link_count,
                w_outer_dim_size);

            for (uint32_t w_direction = 0; w_direction < 2; w_direction++) {
                uint32_t w_core_idx = (w_link * 2) + w_direction;
                CoreCoord w_core = w_fabric_logical_cores[w_core_idx];
                CoreCoord w_virtual_core = w_fabric_virtual_cores[w_core_idx];

                // W reader runtime args
                std::vector<uint32_t> w_reader_rt_args = {
                    w_link_count,
                    w_link_start,
                    w_direction ? pad2_right : pad2_left,
                    barrier_count,
                    output_num_sticks_per_halo_dim,
                    pad2_left,
                    num_sticks_per_halo_dim};
                w_reader_rt_args.push_back(w_direction ? is_last_w_device : is_first_w_device);
                w_reader_rt_args.push_back(w_direction ? is_first_w_device : is_last_w_device);
                w_reader_rt_args.push_back(w_direction);
                w_reader_rt_args.push_back(fabric_only ? input_buffer->address() : 0u);
                w_reader_rt_args.push_back(input_halo_dim_size);
                w_reader_rt_args.push_back(fabric_only ? padding_left : 0u);
                w_reader_rt_args.push_back(fabric_only ? outer_dim_size * padding_left * num_sticks_per_halo_dim : 0u);
                // [14] recv_dram_offset: base DRAM page for received W-halo sticks (NP_W_HALO_L1).
                // Equals the SENDER's outer_dim_offset_start_id, which is section_base + link_start.
                // Computed after w_writer_outer_dim_start (same value for pw_this_dir=1).
                // Set to 0 for non-fabric_only (NP_W_HALO_L1 not active).
                // NOTE: pushed AFTER w_writer_outer_dim_start is computed below.

                // W writer runtime args.
                // In fabric_only mode (compact halo buffer): the W-halo sections are at
                //   W-left base  = 2 * outer_dim_size * padding_left * num_sticks_per_halo_dim
                //   W-right base = W-left base + outer_dim_size * pad2_left * w_halo_H
                // Each W outer_dim occupies exactly 1 consecutive page → stride=1.
                // In non-fabric_only mode: existing addressing relative to output tensor page 0.
                uint32_t w_writer_outer_dim_start;
                uint32_t w_writer_output_halo_dim_size;
                uint32_t w_writer_num_sticks_per_halo_dim;
                if (fabric_only) {
                    // W-halo section offsets in the compact buffer
                    uint32_t w_halo_H = input_halo_dim_size + 2 * padding_left;  // H_dev + 2*ph
                    uint32_t h_wleft_base = 2u * outer_dim_size * padding_left * num_sticks_per_halo_dim;
                    uint32_t h_wright_base = h_wleft_base + outer_dim_size * pad2_left * w_halo_H;
                    uint32_t w_section_base = w_direction ? h_wright_base : h_wleft_base;
                    // Each outer_dim (= one T*H_extended pair) occupies exactly 1 consecutive page.
                    // Link offset: w_link_start outer_dims already handled before this link.
                    w_writer_outer_dim_start = w_section_base + w_link_start;
                    w_writer_output_halo_dim_size = 1;  // stride = 1*1 = 1 page per outer_dim
                    w_writer_num_sticks_per_halo_dim = 1;
                } else {
                    w_writer_outer_dim_start = w_link_start * output_num_sticks_per_halo_dim;
                    w_writer_output_halo_dim_size = output_num_sticks_per_halo_dim;
                    w_writer_num_sticks_per_halo_dim = 1;
                }
                // Now that w_writer_outer_dim_start is known, finalize and set W reader RT args.
                // recv_dram_offset = base DRAM page for received W-halo sticks = w_writer_outer_dim_start
                // (valid for pad2_left=pad2_right=1; generalized version matches the override below).
                w_reader_rt_args.push_back(fabric_only ? w_writer_outer_dim_start : 0u);
                SetRuntimeArgs(program, w_reader_kernel_id, {w_core}, w_reader_rt_args);

                std::vector<uint32_t> w_writer_rt_args = {
                    w_writer_outer_dim_start,
                    0,
                    num_sticks_per_halo_dim,
                    w_writer_output_halo_dim_size,
                    w_link_count,
                    w_direction ? pad2_right : pad2_left,
                    pad2_left,
                    1,
                    w_writer_num_sticks_per_halo_dim,
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
                uint32_t w_device_offset = w_direction ? w_backward_device_offset : w_forward_device_offset;
                w_writer_rt_args.push_back(0);
                w_writer_rt_args.push_back(w_device_offset);
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
                // In fabric_only mode: override W writer rt_args for compact halo buffer
                if (fabric_only) {
                    const uint32_t h_total = input_halo_dim_size + 2 * padding_left;
                    const uint32_t wleft_base = outer_dim_size * 2u * padding_left * num_sticks_per_halo_dim;
                    const uint32_t wright_base = wleft_base + outer_dim_size * pad2_left * h_total;
                    const uint32_t pw_this_dir = w_direction ? pad2_right : pad2_left;
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

    // Build the combined fabric core range for the caller
    CoreRangeSet fabric_core_range = worker_core_ranges;
    if (is_2d) {
        fabric_core_range = fabric_core_range.merge(w_fabric_core_range);
    }

    return NpFabricOnlyArtifacts{
        .h_reader_kernel_id = h_reader_kernel_id,
        .h_writer_kernel_id = h_writer_kernel_id,
        .w_reader_kernel_id = w_reader_kernel_id,
        .w_writer_kernel_id = w_writer_kernel_id,
        .has_w_fabric = is_2d,
        .fabric_core_range = fabric_core_range};
}

// ---------------------------------------------------------------------------
// MeshWorkloadFactory methods
// ---------------------------------------------------------------------------

NeighborPadAsyncMeshWorkloadFactory::cached_mesh_workload_t NeighborPadAsyncMeshWorkloadFactory::create_mesh_workload(
    const NeighborPadAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const NeighborPadAsyncInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Synchronize before dispatching neighbor_pad programs.
    // In fabric_only mode with sub_device_id: sync only the fabric CQ (CQ1) to avoid
    // blocking the conv3d CQ (CQ0) and allow true concurrent execution.
    auto* mesh_device = tensor_args.input_tensor.device();
    {
        ttnn::SmallVector<tt::tt_metal::SubDeviceId> sync_sds;
        std::optional<uint8_t> sync_cq = std::nullopt;
        if (operation_attributes.sub_device_id.has_value()) {
            sync_sds.push_back(*operation_attributes.sub_device_id);
            // Only sync this CQ — don't block other CQs running concurrently
            sync_cq = GetCurrentCommandQueueIdForThread();
        }
        tt::tt_metal::distributed::Synchronize(mesh_device, sync_cq, sync_sds);
    }

    // Create programs for each coordinate in tensor_coords
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

void NeighborPadAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const NeighborPadAsyncParams& operation_attributes,
    const NeighborPadAsyncInputs& tensor_args,
    Tensor& tensor_return_value) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = tensor_return_value.buffer()->address();
    const uint32_t h_sem_addr = operation_attributes.h_neighbor_semaphore.address();
    const uint32_t barrier_sem_addr = operation_attributes.barrier_semaphore.address();
    const uint32_t w_sem_addr = operation_attributes.w_neighbor_semaphore.address();

    for (auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(coordinate_range);

        // All addresses are uniform across cores → use Common Runtime Args (multicast, no per-core loops).
        auto& hr = GetCommonRuntimeArgs(program, shared_vars.h_reader_kernel_id);
        hr[0] = input_addr;
        hr[1] = output_addr;
        hr[2] = h_sem_addr;

        auto& hw = GetCommonRuntimeArgs(program, shared_vars.h_writer_kernel_id);
        hw[0] = input_addr;
        hw[1] = output_addr;
        hw[2] = h_sem_addr;
        hw[3] = barrier_sem_addr;
        // hw[4] = progress_sem_addr (ping-pong, update each call) — only if NP_PROGRESS_SEM
        // hw[5] = num_reader_cores, hw[6+] = NOC coords — static, set once in create()
        if (operation_attributes.progress_semaphore.has_value() && hw.size() > 4) {
            hw[4] = operation_attributes.progress_semaphore->address();
        }

        if (shared_vars.has_local_copy) {
            auto& lr = GetCommonRuntimeArgs(program, shared_vars.local_reader_kernel_id);
            lr[0] = input_addr;
            lr[1] = output_addr;

            auto& lw = GetCommonRuntimeArgs(program, shared_vars.local_writer_kernel_id);
            lw[0] = input_addr;
            lw[1] = output_addr;
            lw[2] = barrier_sem_addr;
        }

        if (shared_vars.has_w_fabric) {
            auto& wr = GetCommonRuntimeArgs(program, shared_vars.w_reader_kernel_id);
            wr[0] = output_addr;
            wr[1] = barrier_sem_addr;
            wr[2] = w_sem_addr;
            // wr[3] = progress_sem_addr (ping-pong, update each call) — only if NP_PROGRESS_SEM
            if (operation_attributes.progress_semaphore.has_value() && wr.size() > 3) {
                wr[3] = operation_attributes.progress_semaphore->address();
            }
            // wr[4] = num_reader_cores, wr[5+] = NOC coords — static, set once in create()

            auto& ww = GetCommonRuntimeArgs(program, shared_vars.w_writer_kernel_id);
            ww[0] = output_addr;
            ww[1] = output_addr;
            ww[2] = w_sem_addr;
            ww[3] = h_sem_addr;
            // W-writer no longer signals conv3d — W-reader handles this.
        }
    }
}

// Fused 2D NeighborPad Algorithm (single op, two phases):
//
// Input: [B,T,H,W,C] fractured across 2D mesh (H across rows, W across columns)
// Output: [B,T,H+2pH,W+2pW,C]
//
// Phase 0 — Startup barrier (multicast sync across all devices):
//   H fabric writers multicast barrier with all H-axis devices (same column).
//   W fabric writers multicast barrier with all W-axis devices (same row).
//   Together these transitively synchronize all devices, ensuring the previous
//   dispatch has completed before any new fabric data is sent.
//
// Phase 1 — Interior copy + H halo exchange (all ~120 cores active):
//   Local copy cores: read input sticks → write to output DRAM at (h+pH, w+pW) offset.
//   H fabric writer (BRISC): self-pad zeros/replicate to output DRAM for H pad rows,
//     send H boundary data to neighbor via fabric.
//   H fabric reader (NCRISC): receive H halo from fabric → L1 → output DRAM.
//   All Phase 1 cores (local copy writers, H fabric writers, and H fabric readers)
//     signal Phase 2 barrier on completion.
//
// Phase 2 — W halo exchange (2–8 W fabric cores, i.e. 2 × pad2_num_links):
//   W reader: waits on Phase 2 barrier, then reads W boundary sticks from output DRAM
//     (safe because Phase 1 calls noc_async_write_barrier() before signaling the barrier).
//     Sends to neighbor via fabric or self-pads. Receives from neighbor → L1 → CB.
//   W writer: pops from CB, writes self-pad and incoming W padding to output DRAM,
//     sends W boundary data to neighbor via fabric.
NeighborPadAsyncMeshWorkloadFactory::cached_program_t NeighborPadAsyncMeshWorkloadFactory::create_at(
    const NeighborPadAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const NeighborPadAsyncInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* mesh_device = tensor_args.input_tensor.device();

    // Use MeshCoordinates to find forward and backward devices
    uint32_t device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, ttnn::ccl::Topology::Linear, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1, ttnn::ccl::Topology::Linear, operation_attributes.cluster_axis);

    // Program creation
    Program program{};

    // Tensor Info
    const auto& input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto& output_tensor_shape = tensor_return_value.padded_shape();
    Buffer* input_buffer = tensor_args.input_tensor.buffer();
    Buffer* output_buffer = tensor_return_value.buffer();

    // Compute shape-derived values
    uint32_t page_size = input_buffer->aligned_page_size();
    uint32_t num_sticks_per_halo_dim = 1;
    for (size_t d = operation_attributes.dim + 1; d < input_tensor_shape.size() - 1; d++) {
        num_sticks_per_halo_dim *= input_tensor_shape[d];
    }
    uint32_t input_halo_dim_size = input_tensor_shape[operation_attributes.dim];
    uint32_t output_halo_dim_size = operation_attributes.fabric_only
                                        ? (operation_attributes.padding_left + operation_attributes.padding_right)
                                        : output_tensor_shape[operation_attributes.dim];
    uint32_t outer_dim_size = 1;
    for (size_t d = 0; d < operation_attributes.dim; d++) {
        outer_dim_size *= input_tensor_shape[d];
    }

    const bool is_2d = operation_attributes.pad_dim2.has_value();
    tt::DataFormat df = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    // Cap num_links (need to replicate capping logic here to know num_h_fabric_cores for local copy)
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t num_links = operation_attributes.num_links;
    uint32_t pad2_num_links = operation_attributes.pad2_num_links;
    uint32_t total_fabric_cores = (num_links * 2) + (is_2d ? pad2_num_links * 2 : 0);
    if (total_fabric_cores > compute_grid_size.x) {
        uint32_t max_total = compute_grid_size.x;
        uint32_t h_cores = num_links * 2;
        if (is_2d) {
            uint32_t available_for_w = (max_total > h_cores) ? (max_total - h_cores) : 0;
            pad2_num_links = available_for_w / 2;
            if (pad2_num_links == 0) {
                pad2_num_links = 1;
                num_links = (max_total - 2) / 2;
            }
        } else {
            num_links = max_total / 2;
        }
    }

    // Pre-compute W-axis topology for 2D case
    std::optional<MeshCoordinate> w_forward_coord;
    std::optional<MeshCoordinate> w_backward_coord;
    uint32_t w_device_index = 0;
    if (is_2d) {
        w_forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coordinate,
            1,
            ttnn::ccl::Topology::Linear,
            operation_attributes.pad2_cluster_axis);
        w_backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coordinate,
            -1,
            ttnn::ccl::Topology::Linear,
            operation_attributes.pad2_cluster_axis);
        w_device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, operation_attributes.pad2_cluster_axis);
    }

    // Compute progress semaphore address and reader NOC coords
    uint32_t progress_sem_addr =
        operation_attributes.progress_semaphore.has_value() ? operation_attributes.progress_semaphore->address() : 0u;
    std::vector<std::pair<uint32_t, uint32_t>> reader_noc_coords;
    if (operation_attributes.progress_semaphore.has_value() && operation_attributes.progress_t_batch_size > 0) {
        uint32_t num_h_fabric_cores = num_links * 2;
        CoreCoord core_grid(num_h_fabric_cores, 1);
        auto [nc, wcr, cg1, cg2, dpcg1, dpcg2] = (operation_attributes.dim > 0)
                                                     ? split_work_to_cores(core_grid, outer_dim_size * 2)
                                                     : split_work_to_cores(core_grid, num_sticks_per_halo_dim * 2);
        CoreRangeSet fabric_worker_core_ranges = wcr;
        CoreRangeSet full_grid(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
        CoreRangeSet conv3d_cores = full_grid.subtract(fabric_worker_core_ranges);
        for (const auto& core : corerange_to_cores(conv3d_cores, std::nullopt, true)) {
            auto noc = mesh_device->worker_core_from_logical_core(core);
            reader_noc_coords.emplace_back(noc.x, noc.y);
        }
        log_warning(
            tt::LogOp,
            "NP_BASELINE: h_fabric={} conv3d_reader_cores={} reader_noc_count={}",
            fabric_worker_core_ranges.str(),
            conv3d_cores.str(),
            reader_noc_coords.size());
    }

    // Call the helper (adds fabric kernels to program)
    auto np_artifacts = build_np_fabric_only_program_artifacts(
        program,
        mesh_device,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        device_index,
        input_buffer,
        output_buffer,
        // NP params
        operation_attributes.dim,
        operation_attributes.padding_left,
        operation_attributes.padding_right,
        operation_attributes.padding_mode,
        operation_attributes.ring_size,
        operation_attributes.topology,
        operation_attributes.h_neighbor_semaphore,
        operation_attributes.barrier_semaphore,
        num_links,
        // Shape-derived params
        input_halo_dim_size,
        num_sticks_per_halo_dim,
        outer_dim_size,
        output_halo_dim_size,
        operation_attributes.fabric_only,
        df,
        // 2D params
        operation_attributes.pad_dim2,
        operation_attributes.pad2_left,
        operation_attributes.pad2_right,
        operation_attributes.pad2_cluster_axis,
        pad2_num_links,
        operation_attributes.w_neighbor_semaphore,
        // W-axis topology
        w_forward_coord,
        w_backward_coord,
        w_device_index,
        // Progress sem: H-writer signals per T-batch, W-reader signals once at end for W-halo ordering.
        progress_sem_addr,
        reader_noc_coords,
        operation_attributes.progress_t_batch_size);

    // Local copy workers on cores not used by fabric: AllCores - FabricCores
    // In fabric_only mode, local_copy is skipped entirely — conv3d reads interior from original tensor.
    std::vector<CoreCoord> local_copy_core_coords;
    KernelHandle local_reader_kernel_id = 0;
    KernelHandle local_writer_kernel_id = 0;
    bool has_local_copy = false;
    if (!operation_attributes.fabric_only) {
        // For 2D: compute output row width metrics for local copy
        uint32_t output_num_sticks_per_halo_dim = num_sticks_per_halo_dim;
        uint32_t writer_stick_start_id = 0;
        uint32_t writer_num_sticks_to_read = num_sticks_per_halo_dim;
        if (is_2d) {
            output_num_sticks_per_halo_dim =
                num_sticks_per_halo_dim + operation_attributes.pad2_left + operation_attributes.pad2_right;
            writer_stick_start_id = operation_attributes.pad2_left;
        }

        uint32_t num_h_fabric_cores = num_links * 2;
        uint32_t num_w_fabric_cores = is_2d ? (pad2_num_links * 2) : 0;

        // Reconstruct W fabric virtual cores for Phase 2 signal targets
        std::vector<CoreCoord> w_fabric_virtual_cores;
        if (is_2d) {
            for (uint32_t i = 0; i < num_w_fabric_cores; i++) {
                CoreCoord wc = {num_h_fabric_cores + i, 0};
                w_fabric_virtual_cores.push_back(mesh_device->worker_core_from_logical_core(wc));
            }
        }

        CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
        CoreRangeSet local_copy_cores = all_cores.subtract(np_artifacts.fabric_core_range);

        if (!local_copy_cores.empty()) {
            has_local_copy = true;

            uint32_t l1_scratch_cb_page_size_bytes = page_size;
            uint32_t cb_num_pages = 2;
            uint32_t sender_cb_index = tt::CB::c_in0;
            CircularBufferConfig cb_sender_config =
                CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
                    .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);

            // CB on all local-copy cores
            CreateCircularBuffer(program, local_copy_cores, cb_sender_config);

            // Create consolidated local copy reader kernel (uniform compile args)
            auto local_reader_cfg = ReaderDataMovementConfig{};
            local_reader_cfg.compile_args = {sender_cb_index, page_size};
            TensorAccessorArgs(*input_buffer).append_to(local_reader_cfg.compile_args);
            local_reader_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/local_copy_reader.cpp",
                local_copy_cores,
                local_reader_cfg);
            SetCommonRuntimeArgs(
                program,
                local_reader_kernel_id,
                {input_buffer->address(),
                 output_buffer->address(),
                 0u,
                 input_halo_dim_size,
                 num_sticks_per_halo_dim,
                 num_sticks_per_halo_dim});

            // Create consolidated local copy writer kernel (uniform compile args)
            auto local_writer_cfg = WriterDataMovementConfig{};
            local_writer_cfg.compile_args = {sender_cb_index, page_size};
            TensorAccessorArgs(*output_buffer).append_to(local_writer_cfg.compile_args);
            local_writer_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/local_copy_writer.cpp",
                local_copy_cores,
                local_writer_cfg);
            // Local writer CRTAs
            std::vector<uint32_t> local_writer_crta = {
                input_buffer->address(),
                output_buffer->address(),
                operation_attributes.barrier_semaphore.address(),
                writer_stick_start_id,
                input_halo_dim_size,
                output_halo_dim_size,
                operation_attributes.padding_left,
                writer_num_sticks_to_read,
                output_num_sticks_per_halo_dim,
                is_2d ? num_w_fabric_cores : 0u};
            constexpr uint32_t LOCAL_MAX_PHASE2_SIGNAL_TARGETS = 8;
            for (uint32_t s = 0; s < LOCAL_MAX_PHASE2_SIGNAL_TARGETS; s++) {
                if (is_2d && s < num_w_fabric_cores) {
                    local_writer_crta.push_back(w_fabric_virtual_cores[s].x);
                    local_writer_crta.push_back(w_fabric_virtual_cores[s].y);
                } else {
                    local_writer_crta.push_back(0);
                    local_writer_crta.push_back(0);
                }
            }
            SetCommonRuntimeArgs(program, local_writer_kernel_id, local_writer_crta);

            // Distribute work evenly across local-copy cores and set per-core runtime args
            std::vector<CoreCoord> local_cores = corerange_to_cores(local_copy_cores, std::nullopt, /*row_wise=*/true);
            const uint32_t num_local_cores = local_cores.size();
            const uint32_t total_units =
                (operation_attributes.dim > 0) ? (outer_dim_size * input_halo_dim_size) : input_halo_dim_size;
            const uint32_t base = (num_local_cores == 0) ? 0 : (total_units / num_local_cores);
            const uint32_t rem = (num_local_cores == 0) ? 0 : (total_units % num_local_cores);

            uint32_t unit_offset = 0;
            for (uint32_t i = 0; i < num_local_cores; ++i) {
                const uint32_t units_for_core = base + (i < rem ? 1u : 0u);
                if (units_for_core == 0) {
                    continue;
                }

                const CoreCoord& logical_core = local_cores[i];
                local_copy_core_coords.push_back(logical_core);

                SetRuntimeArgs(program, local_reader_kernel_id, {logical_core}, {unit_offset, units_for_core});
                SetRuntimeArgs(program, local_writer_kernel_id, {logical_core}, {unit_offset, units_for_core});

                unit_offset += units_for_core;
            }
        }
    }  // end !fabric_only local_copy block

    // For the non-fabric_only 2D case, the W fabric barrier_count must include local copy writers.
    // The helper set barrier_count = num_h_fabric_cores (fabric_only assumption). When local copy
    // workers exist, we need to update the W reader rt_args[3] (barrier_count) on each W core.
    if (is_2d && has_local_copy) {
        uint32_t num_h_fabric_cores = num_links * 2;
        uint32_t corrected_barrier_count = num_h_fabric_cores + static_cast<uint32_t>(local_copy_core_coords.size());
        for (uint32_t w_link = 0; w_link < pad2_num_links; w_link++) {
            for (uint32_t w_direction = 0; w_direction < 2; w_direction++) {
                uint32_t w_core_idx = (w_link * 2) + w_direction;
                CoreCoord w_core = {num_h_fabric_cores + w_core_idx, 0};
                auto& w_reader_rt = GetRuntimeArgs(program, np_artifacts.w_reader_kernel_id, w_core);
                w_reader_rt[3] = corrected_barrier_count;
            }
        }
    }

    return cached_program_t(
        std::move(program),
        NeighborPadAsyncSharedVariables{
            .h_reader_kernel_id = np_artifacts.h_reader_kernel_id,
            .h_writer_kernel_id = np_artifacts.h_writer_kernel_id,
            .local_reader_kernel_id = local_reader_kernel_id,
            .local_writer_kernel_id = local_writer_kernel_id,
            .w_reader_kernel_id = np_artifacts.w_reader_kernel_id,
            .w_writer_kernel_id = np_artifacts.w_writer_kernel_id,
            .has_local_copy = has_local_copy,
            .has_w_fabric = np_artifacts.has_w_fabric});
}

}  // namespace ttnn::experimental::prim
