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

    // H-dim is index 1 in BTHWC layout (dim=1)
    constexpr uint32_t np_dim = 1;
    uint32_t page_size = input_buffer->aligned_page_size();

    // num_sticks_per_halo_dim: sticks per H row = W * (all dims after H, before C)
    // In BTHWC: after dim 1 (H) we have W (dim 2), then C (last dim excluded = sticks).
    // So num_sticks_per_halo_dim = W_dev
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
    [[maybe_unused]] uint32_t forward_device_offset = forward_coord.has_value() ? 1u : 0u;
    [[maybe_unused]] uint32_t backward_device_offset = backward_coord.has_value() ? 1u : 0u;

    bool is_padding_zeros = op.padding_mode == "zeros";
    const bool is_2d = op.np_pad_dim2.has_value();

    // For 2D padding W-axis setup
    uint32_t output_num_sticks_per_halo_dim = num_sticks_per_halo_dim;
    uint32_t writer_stick_start_id = 0;
    uint32_t writer_num_sticks_to_read = num_sticks_per_halo_dim;
    if (is_2d) {
        output_num_sticks_per_halo_dim = num_sticks_per_halo_dim + op.np_pad2_left + op.np_pad2_right;
        writer_stick_start_id = op.np_pad2_left;
    }

    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t num_links = static_cast<uint32_t>(op.np_num_links);
    uint32_t pad2_num_links = static_cast<uint32_t>(op.np_pad2_num_links);

    constexpr uint32_t MAX_PAD2_NUM_LINKS = 4;
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

    uint32_t num_h_fabric_cores = num_links * 2;
    uint32_t num_w_fabric_cores = is_2d ? (pad2_num_links * 2) : 0;
    TT_FATAL(
        pad2_num_links <= MAX_PAD2_NUM_LINKS,
        "pad2_num_links ({}) exceeds maximum supported ({})",
        pad2_num_links,
        MAX_PAD2_NUM_LINKS);

    CoreCoord np_core_grid(num_h_fabric_cores, 1);
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
            CoreCoord wc = {num_h_fabric_cores + i, 0};
            w_fabric_logical_cores.push_back(wc);
            w_fabric_virtual_cores.push_back(mesh_device->worker_core_from_logical_core(wc));
        }
        w_fabric_core_range =
            CoreRangeSet(CoreRange({num_h_fabric_cores, 0}, {num_h_fabric_cores + num_w_fabric_cores - 1, 0}));

        // fabric_only: W exchange covers all H_dev + 2*ph rows per T
        w_outer_dim_size = outer_dim_size * (input_halo_dim_size + 2 * op.np_padding_h);

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

    // Progress semaphore address: provided by the caller via conv_config (same as standalone conv3d).
    // Python resets this semaphore to 0 before each dispatch via reset_global_semaphore_value().
    // The NP writer signals each conv3d reader core at this L1 address via NOC atomic after each T-batch.
    uint32_t progress_sem_l1_addr = conv_config.input_progress_sem_addr;

    // Collect conv3d reader core NOC coords (for H writer progress-sem signaling)
    std::vector<std::pair<uint32_t, uint32_t>> reader_noc_coords;
    if (conv_config.input_progress_t_batch_size > 0) {
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
    h_reader_kernel_config.compile_args.push_back(is_2d ? 1 : 0);  // use_l1_intermediate
    h_reader_kernel_config.compile_args.push_back(0);              // recv_cb_id (unused in H-only)
    auto h_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
        "minimal_default_reader.cpp",
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
    h_writer_kernel_config.compile_args.push_back(is_2d ? 1 : 0);    // use_l1_intermediate
    h_writer_kernel_config.compile_args.push_back(0);                // recv_cb_id (unused)
    h_writer_kernel_config.compile_args.push_back(is_2d ? 1 : 0);    // handle_incoming_writes
    h_writer_kernel_config.compile_args.push_back(0);                // is_w_fabric_writer (false for H)
    h_writer_kernel_config.compile_args.push_back(op.np_ring_size);  // ring_size
    if (conv_config.input_progress_t_batch_size > 0) {
        h_writer_kernel_config.defines["NP_PROGRESS_SEM"] = "1";
        h_writer_kernel_config.compile_args.push_back(conv_config.input_progress_t_batch_size);
    }
    auto h_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
        "minimal_default_writer.cpp",
        np_worker_core_ranges,
        h_writer_kernel_config);
    {
        std::vector<uint32_t> h_writer_crta = {
            input_buffer->address(),
            halo_buffer->address(),
            op.h_neighbor_semaphore.address(),
            op.barrier_semaphore.address(),
            (conv_config.input_progress_t_batch_size > 0) ? progress_sem_l1_addr : 0u,
            // CRTA[5]: number of conv3d reader cores to signal
            static_cast<uint32_t>(reader_noc_coords.size()),
            // CRTA[6+]: interleaved (x, y) NOC coords for each reader core
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
            CoreCoord core = {link * num_directions + direction, 0};
            CoreCoord opposite_core = {(link * num_directions) + (1 - direction), 0};
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
            CoreCoord virtual_opposite_core = mesh_device->worker_core_from_logical_core(opposite_core);
            if (np_core_group_1.contains(core)) {
                link_dims_to_read = np_dims_per_core_group_1;
            } else {
                link_dims_to_read = np_dims_per_core_group_2;
            }

            // Reader runtime args
            std::vector<uint32_t> reader_rt_args = {
                link_offset_start_id * input_halo_dim_size,     // outer_dim_offset_start_id
                0,                                              // stick_start_id
                input_halo_dim_size,                            // input_halo_dim_size
                link_dims_to_read,                              // outer_dim_size
                direction ? op.np_padding_h : op.np_padding_h,  // padding (same both sides for symmetric)
                num_sticks_per_halo_dim,                        // num_sticks_to_read
                num_sticks_per_halo_dim,                        // num_sticks_per_halo_dim
                0u};                                            // num_l1_recv_sticks_per_row (no corners in H-only)
            reader_rt_args.push_back(direction ? is_last_device : is_first_device);  // is_first_chip
            reader_rt_args.push_back(direction ? is_first_device : is_last_device);  // is_last_chip
            reader_rt_args.push_back(direction);                                     // direction
            SetRuntimeArgs(program, h_reader_kernel_id, {core}, reader_rt_args);

            // Writer runtime args
            uint32_t h_writer_num_sticks_per_halo_dim =
                is_2d ? output_num_sticks_per_halo_dim : num_sticks_per_halo_dim;
            uint32_t h_writer_stick_start = writer_stick_start_id;
            uint32_t h_writer_num_sticks_to_read = writer_num_sticks_to_read;

            std::vector<uint32_t> writer_rt_args = {
                writer_link_offset_start_id * output_halo_dim_size,  // outer_dim_offset_start_id
                h_writer_stick_start,                                // stick_start_id
                input_halo_dim_size,                                 // input_halo_dim_size
                output_halo_dim_size,                                // output_halo_dim_size
                link_dims_to_read,                                   // outer_dim_size
                direction ? op.np_padding_h : op.np_padding_h,       // padding
                0u,                                                  // padding_left (no W-offset in compact)
                h_writer_num_sticks_to_read,                         // num_sticks_to_read
                h_writer_num_sticks_per_halo_dim,                    // num_sticks_per_halo_dim
                virtual_core.x,                                      // neighbor_sem_noc0_x
                virtual_core.y,                                      // neighbor_sem_noc0_y
                true,                                                // use_barrier_semaphore
                virtual_opposite_core.x,                             // barrier_sem_noc0_x
                virtual_opposite_core.y};                            // barrier_sem_noc0_y
            // Phase 2 signal targets (W fabric reader cores)
            constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 8;
            writer_rt_args.push_back(is_2d ? num_w_fabric_cores : 0);
            for (uint32_t s = 0; s < MAX_PHASE2_SIGNAL_TARGETS; s++) {
                if (is_2d && s < num_w_fabric_cores) {
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
                uint32_t top_halo_sticks = outer_dim_size * op.np_padding_h * num_sticks_per_halo_dim;
                uint32_t padding_this_dir = direction ? op.np_padding_h : op.np_padding_h;
                writer_rt_args[0] = direction ? top_halo_sticks : 0;  // outer_dim_offset_start_id
                writer_rt_args[3] = padding_this_dir;                 // output_halo_dim_size (compact)
                writer_rt_args[6] = 0;                                // padding_left (no W-offset in compact)
                // arg[40] = num_phase2_signal_targets — already set correctly above
                // (is_2d ? num_w_fabric_cores : 0 at index 14+2*8=30... wait, let's recalculate)
                // The 14 fixed args (indices 0-13) + is_2d targets at index 14:
                // [0..13] = 14 fixed args, [14] = num_targets, [15..30] = 8 pairs
                // writer_rt_args[14] = 0 when !is_2d (already set correctly)
            }
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

        // W reader kernel
        auto w_reader_kernel_config = ReaderDataMovementConfig{};
        w_reader_kernel_config.compile_args = {sender_cb_index, is_padding_zeros, page_size};
        TensorAccessorArgs(*halo_buffer).append_to(w_reader_kernel_config.compile_args);
        TensorAccessorArgs(*input_buffer).append_to(w_reader_kernel_config.compile_args);
        if (conv_config.input_progress_t_batch_size > 0) {
            w_reader_kernel_config.defines["NP_PROGRESS_SEM"] = "1";
        }
        w_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
            "phase2_w_reader.cpp",
            w_fabric_core_range,
            w_reader_kernel_config);
        {
            std::vector<uint32_t> w_reader_crta = {
                halo_buffer->address(),
                op.barrier_semaphore.address(),
                op.w_neighbor_semaphore.address(),
                (conv_config.input_progress_t_batch_size > 0) ? progress_sem_l1_addr : 0u,
                static_cast<uint32_t>(reader_noc_coords.size()),
            };
            for (const auto& [x, y] : reader_noc_coords) {
                w_reader_crta.push_back(x);
                w_reader_crta.push_back(y);
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
        w_writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
            "minimal_default_writer.cpp",
            w_fabric_core_range,
            w_writer_kernel_config);
        SetCommonRuntimeArgs(
            program,
            w_writer_kernel_id,
            {halo_buffer->address(),
             halo_buffer->address(),
             op.w_neighbor_semaphore.address(),
             op.h_neighbor_semaphore.address()});

        // Per-core W fabric runtime args
        for (uint32_t w_link = 0; w_link < pad2_num_links; w_link++) {
            uint32_t w_link_start = (w_link * w_rows_per_link) + std::min(w_link, w_extra_rows);
            uint32_t w_link_count = w_rows_per_link + (w_link < w_extra_rows ? 1 : 0);

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
                // progress_t_batch_size (read by W reader when NP_PROGRESS_SEM is defined)
                if (conv_config.input_progress_t_batch_size > 0) {
                    w_reader_rt_args.push_back(conv_config.input_progress_t_batch_size);
                }
                SetRuntimeArgs(program, w_reader_kernel_id, {w_core}, w_reader_rt_args);

                // W writer runtime args
                std::vector<uint32_t> w_writer_rt_args = {
                    w_link_start * output_num_sticks_per_halo_dim,
                    0,
                    num_sticks_per_halo_dim,
                    output_num_sticks_per_halo_dim,
                    w_link_count,
                    w_direction ? op.np_pad2_right : op.np_pad2_left,
                    op.np_pad2_left,
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
    // PART 3: CONV3D KERNELS (use_h_halo_buffer=true, input_progress_t_batch_size from config)
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

    // Inflate effective padding with halo buffer H/W contributions
    std::array<uint32_t, 3> effective_padding = op.padding;
    if (conv_config.use_h_halo_buffer) {
        effective_padding[1] += conv_config.h_halo_padding_h;
        effective_padding[2] += conv_config.h_halo_padding_w;
    }

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

    // L1 prefetch buffer
    constexpr uint32_t L1_KERNEL_CODE_RESERVE = 200 * 1024;
    constexpr uint32_t L1_PREFETCH_HARD_CAP = 500 * 1024;
    const uint32_t l1_usable_for_cbs = tt::tt_metal::hal::get_max_worker_l1_unreserved_size() - L1_KERNEL_CODE_RESERVE;

    uint32_t other_cbs_bytes = (padded_patch_size_bytes * vol2col_rm_pages) + (tile_size * matmul_K_t) +
                               (tile_size * matmul_K_t * matmul_N_t) + (partial_tile_size * matmul_M_t * matmul_N_t) +
                               (tile_size * matmul_M_t * matmul_N_t);
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

    uint32_t cb_input_shard_id = 32;
    uint32_t T_shard_max = 0;
    uint32_t H_shard_max = 0;
    uint32_t W_shard_max = 0;

    const bool has_spatial_reuse = (kT > 1 || kH > 1 || kW > 1);
    const bool has_no_dilation = (op.dilation[0] == 1 && op.dilation[1] == 1 && op.dilation[2] == 1);

    if (has_spatial_reuse && has_no_dilation) {
        T_shard_max = (conv_config.T_out_block - 1) * op.stride[0] + kT;
        H_shard_max = (conv_config.H_out_block - 1) * op.stride[1] + kH;
        W_shard_max = (conv_config.W_out_block - 1) * op.stride[2] + kW;
        uint32_t shard_positions_max = T_shard_max * H_shard_max * W_shard_max;
        uint32_t shard_bytes = shard_positions_max * C_in_block_bytes;

        if (shard_bytes <= l1_prefetch_max_bytes) {
            cb_input_shard_id = next_cb_index++;
            tt::tt_metal::create_cb(
                cb_input_shard_id, program, core_grid, C_in_block_bytes, shard_positions_max, data_format);
        } else {
            T_shard_max = 0;
            H_shard_max = 0;
            W_shard_max = 0;
        }
    }

    uint32_t in_row_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_row_size_bytes = output_tensor.buffer()->aligned_page_size();

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
        patch_pad_bytes};
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    // Ablation defines
    std::map<std::string, std::string> ablation_defines;
    {
        const char* ablate_env = std::getenv("CONV3D_ABLATE");
        if (ablate_env != nullptr) {
            std::string ablate_str(ablate_env);
            if (ablate_str == "tilize") {
                ablation_defines["ABLATE_TILIZE"] = "1";
            } else if (ablate_str == "dm") {
                ablation_defines["ABLATE_DM"] = "1";
            } else if (ablate_str == "tilize_dm") {
                ablation_defines["ABLATE_TILIZE"] = "1";
                ablation_defines["ABLATE_DM"] = "1";
            } else if (ablate_str == "profile") {
                ablation_defines["PROFILE_ZONES"] = "1";
            }
        }
    }
    if (conv_config.input_progress_t_batch_size > 0) {
        ablation_defines["CONV3D_INPUT_PROGRESS_SEM"] = "1";
    }
    if (conv_config.use_h_halo_buffer) {
        ablation_defines["CONV3D_H_HALO"] = "1";
    }

    auto conv3d_reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, ablation_defines));

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
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = ablation_defines});

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
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_tensor.has_value() ? bias_tensor.value().buffer() : nullptr)
        .append_to(writer_compile_time_args);

    auto conv3d_writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, ablation_defines));

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
    uint32_t t_out_parallel_factor = std::min(T_out_blocks, remaining_parallel);
    remaining_parallel = remaining_parallel / t_out_parallel_factor;
    uint32_t h_out_parallel_factor = std::min(H_out_blocks, remaining_parallel);
    remaining_parallel = remaining_parallel / h_out_parallel_factor;
    uint32_t w_out_parallel_factor = std::min(W_out_blocks, remaining_parallel);

    uint32_t total_output_parallel =
        c_out_parallel_factor * t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor;

    TT_FATAL(
        c_in_parallel_factor * total_output_parallel <= conv3d_num_cores,
        "NpConv3d: Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        c_in_parallel_factor * total_output_parallel,
        conv3d_num_cores);

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

        // Reader args[0..18] match conv3d_program_factory.cpp layout exactly:
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
        //   [11] input_progress_sem_addr (L1 addr of progress sem, 0 if disabled)
        //   [12] input_progress_t_batch_size
        //   [13] h_halo_buffer_addr
        //   [14] h_halo_outer_dim_size
        //   [15] h_halo_H
        //   [16] h_halo_W
        //   [17] h_halo_padding_h
        //   [18] h_halo_padding_w
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
            // [11]: progress semaphore L1 address (static: same per-call since it is L1, not GlobalSemaphore)
            progress_sem_l1_addr,
            conv_config.input_progress_t_batch_size,
            // [13]: halo buffer DRAM address
            halo_buffer_addr,
            conv_config.h_halo_outer_dim_size,
            conv_config.h_halo_H,
            conv_config.h_halo_W,
            conv_config.h_halo_padding_h,
            conv_config.h_halo_padding_w,
        };

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
                    is_2d,
                    is_2d ? w_fabric_core_range : CoreRangeSet{}},
            .conv3d_num_cores = conv3d_num_cores,
            .conv3d_cores = conv3d_cores,
            .conv3d_reader_kernel_id = conv3d_reader_kernels_id,
            .conv3d_writer_kernel_id = conv3d_writer_kernels_id,
            .conv3d_compute_kernel_id = conv3d_compute_kernels_id,
            .progress_sem_l1_addr = progress_sem_l1_addr}};
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
        // CRTA[3] = barrier_sem_addr, CRTA[4] = progress_sem_l1_addr (static),
        // CRTA[5] = num_reader_cores (static), CRTA[6+] = NOC coords (static)
        auto& hw = GetCommonRuntimeArgs(program, shared_vars.np_artifacts.h_writer_kernel_id);
        hw[0] = input_addr;
        hw[1] = halo_buffer_addr;
        hw[2] = h_sem_addr;
        hw[3] = barrier_sem_addr;
        // hw[4] = progress_sem_l1_addr — ping-pong semaphore changes each call
        hw[4] = op.conv_config.input_progress_sem_addr;
        // hw[5+] = num_reader_cores and NOC coords — static, set once in create_at()

        // --- NP W-fabric kernels (only when 2D) ---
        if (shared_vars.np_artifacts.has_w_fabric) {
            auto& wr = GetCommonRuntimeArgs(program, shared_vars.np_artifacts.w_reader_kernel_id);
            wr[0] = halo_buffer_addr;
            wr[1] = barrier_sem_addr;
            wr[2] = w_sem_addr;
            // wr[3] = progress_sem_l1_addr — ping-pong semaphore, update each call
            if (op.conv_config.input_progress_t_batch_size > 0) {
                wr[3] = op.conv_config.input_progress_sem_addr;
            }

            auto& ww = GetCommonRuntimeArgs(program, shared_vars.np_artifacts.w_writer_kernel_id);
            ww[0] = halo_buffer_addr;
            ww[1] = halo_buffer_addr;
            ww[2] = w_sem_addr;
            ww[3] = h_sem_addr;
        }

        // --- Conv3d reader: update input_addr (args[0]) and halo_buffer_addr (args[13]) ---
        // progress_sem_l1_addr (args[11]) is static (L1 address does not change)
        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.conv3d_reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.conv3d_writer_kernel_id);

        for (uint32_t i = 0; i < shared_vars.conv3d_num_cores; ++i) {
            CoreCoord core = shared_vars.conv3d_cores.at(i);
            auto& reader_args = reader_args_by_core[core.x][core.y];
            auto& writer_args = writer_args_by_core[core.x][core.y];

            reader_args[0] = input_addr;
            // args[11] = progress_sem_l1_addr — ping-pong semaphore, update each call
            if (op.conv_config.input_progress_t_batch_size > 0) {
                reader_args[11] = op.conv_config.input_progress_sem_addr;
            }
            // args[13] = halo_buffer_addr
            if (op.conv_config.use_h_halo_buffer) {
                reader_args[13] = halo_buffer_addr;
                reader_args[14] = op.conv_config.h_halo_outer_dim_size;
                reader_args[15] = op.conv_config.h_halo_H;
                reader_args[16] = op.conv_config.h_halo_W;
                reader_args[17] = op.conv_config.h_halo_padding_h;
                reader_args[18] = op.conv_config.h_halo_padding_w;
            }

            writer_args[0] = output_addr;
            writer_args[1] = weight_addr;
            writer_args[2] = bias_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
