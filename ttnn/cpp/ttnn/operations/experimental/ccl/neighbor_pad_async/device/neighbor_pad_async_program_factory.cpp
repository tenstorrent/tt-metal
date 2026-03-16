// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

NeighborPadAsyncMeshWorkloadFactory::cached_mesh_workload_t NeighborPadAsyncMeshWorkloadFactory::create_mesh_workload(
    const NeighborPadAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const NeighborPadAsyncInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Synchronize all devices before dispatching neighbor_pad programs.
    // This ensures all previous fabric-initiated writes (from prior ops) have completed.
    auto* mesh_device = tensor_args.input_tensor.device();
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    // Create programs for each coordinate in tensor_coords
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = create_at(operation_attributes, mesh_coord, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
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

            auto& ww = GetCommonRuntimeArgs(program, shared_vars.w_writer_kernel_id);
            ww[0] = output_addr;
            ww[1] = output_addr;
            ww[2] = w_sem_addr;
            // Use h_neighbor_semaphore (not barrier_semaphore) — W reader on same core uses
            // barrier_semaphore for Phase 2 barrier, so they must use different addresses.
            ww[3] = h_sem_addr;
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
    // This is safe on bigmesh where remote devices might not exist on this rank
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

    // Get OP Config, topology config
    // Use the buffer's aligned page size (architecture-specific: 32B on WH, 64B on BH).
    // The interleaved address generator spaces pages at aligned_page_size intervals,
    // so NOC transfers must use this size to avoid sub-minimum or misaligned reads.
    uint32_t page_size = input_buffer->aligned_page_size();
    uint32_t num_sticks_per_halo_dim = 1;
    for (size_t d = operation_attributes.dim + 1; d < input_tensor_shape.size() - 1; d++) {
        num_sticks_per_halo_dim *= input_tensor_shape[d];
    }
    uint32_t input_halo_dim_size = input_tensor_shape[operation_attributes.dim];
    uint32_t output_halo_dim_size = output_tensor_shape[operation_attributes.dim];
    uint32_t outer_dim_size = 1;
    for (size_t d = 0; d < operation_attributes.dim; d++) {
        outer_dim_size *= input_tensor_shape[d];
    }

    bool is_first_device = true;
    bool is_last_device = true;
    uint32_t forward_device_offset = 0;
    uint32_t backward_device_offset = 0;

    if (operation_attributes.secondary_cluster_axis.has_value()) {
        // secondary_cluster_axis==1, devices on row
        // secondary_mesh_shape(0) == number of rows, (1) is number of cols
        uint32_t secondary_cluster_axis_val = operation_attributes.secondary_cluster_axis.value();
        uint32_t row_index = device_index / operation_attributes.secondary_mesh_shape.value().at(1);
        uint32_t col_index = device_index % operation_attributes.secondary_mesh_shape.value().at(1);
        if (secondary_cluster_axis_val) {
            // row
            if (col_index != 0) {
                is_first_device = false;
                backward_device_offset = 1;
            }
            if (col_index != operation_attributes.secondary_mesh_shape.value().at(1) - 1) {
                is_last_device = false;
                forward_device_offset = 1;
            }
        } else {
            // column
            if (row_index != 0) {
                is_first_device = false;
                backward_device_offset = operation_attributes.secondary_mesh_shape.value().at(1);
            }
            if (row_index != (operation_attributes.secondary_mesh_shape.value().at(0) - 1)) {
                is_last_device = false;
                forward_device_offset = operation_attributes.secondary_mesh_shape.value().at(1);
            }
        }
    } else {
        is_first_device = !backward_coord.has_value();
        is_last_device = !forward_coord.has_value();
        if (!is_first_device) {
            backward_device_offset = 1;
        }
        if (!is_last_device) {
            forward_device_offset = 1;
        }
    }

    log_trace(
        tt::LogOp,
        "NeighborPad H-fabric: mesh_coord=({},{}), device_index={}, src_node_id={}, "
        "fwd_offset={}, bwd_offset={}, is_first={}, is_last={}, cluster_axis={}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        device_index,
        mesh_device->get_fabric_node_id(mesh_coordinate),
        forward_device_offset,
        backward_device_offset,
        is_first_device,
        is_last_device,
        operation_attributes.cluster_axis);
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

    bool is_padding_zeros = operation_attributes.padding_mode == "zeros";
    const bool is_2d = operation_attributes.pad_dim2.has_value();

    // For 2D padding: compute secondary dimension metrics
    uint32_t output_num_sticks_per_halo_dim = num_sticks_per_halo_dim;  // default: same as input
    uint32_t writer_stick_start_id = 0;
    uint32_t writer_num_sticks_to_read = num_sticks_per_halo_dim;
    if (is_2d) {
        // The output has extra W padding, so its row width is wider
        output_num_sticks_per_halo_dim =
            num_sticks_per_halo_dim + operation_attributes.pad2_left + operation_attributes.pad2_right;
        writer_stick_start_id = operation_attributes.pad2_left;
    }

    // Get worker cores
    constexpr uint32_t MAX_PAD2_NUM_LINKS = 4;  // kernel arrays sized for pad2_num_links * 2 = 8 targets

    // Cap num_links and pad2_num_links so total fabric cores fit within the device compute grid width.
    // WH has 8x8 logical grid vs BH's 14x10, so large link counts can exceed the grid.
    auto compute_grid_size = mesh_device->compute_with_storage_grid_size();
    uint32_t num_links = operation_attributes.num_links;
    uint32_t pad2_num_links = operation_attributes.pad2_num_links;
    uint32_t total_fabric_cores = (num_links * 2) + (is_2d ? pad2_num_links * 2 : 0);
    if (total_fabric_cores > compute_grid_size.x) {
        // Reduce pad2_num_links first (W-fabric), then num_links (H-fabric) if still too many
        uint32_t max_total = compute_grid_size.x;
        uint32_t h_cores = num_links * 2;
        if (is_2d) {
            uint32_t available_for_w = (max_total > h_cores) ? (max_total - h_cores) : 0;
            pad2_num_links = available_for_w / 2;
            if (pad2_num_links == 0) {
                // H-fabric alone exceeds grid, reduce num_links too
                pad2_num_links = 1;
                num_links = (max_total - 2) / 2;  // reserve 2 cores for 1 W link
            }
        } else {
            num_links = max_total / 2;
        }
        log_warning(
            tt::LogOp,
            "neighbor_pad_async: Capped num_links from {} to {} and pad2_num_links from {} to {} "
            "to fit device compute grid width {}",
            operation_attributes.num_links,
            num_links,
            operation_attributes.pad2_num_links,
            pad2_num_links,
            compute_grid_size.x);
    }

    uint32_t num_h_fabric_cores = num_links * 2;
    uint32_t num_w_fabric_cores = is_2d ? (pad2_num_links * 2) : 0;
    TT_FATAL(
        pad2_num_links <= MAX_PAD2_NUM_LINKS,
        "pad2_num_links ({}) exceeds maximum supported ({}). Kernel Phase 2 signal target arrays are sized for {}.",
        pad2_num_links,
        MAX_PAD2_NUM_LINKS,
        MAX_PAD2_NUM_LINKS * 2);
    CoreCoord core_grid(num_h_fabric_cores, 1);
    auto [num_cores, worker_core_ranges, core_group_1, core_group_2, dims_per_core_group_1, dims_per_core_group_2] =
        (operation_attributes.dim > 0) ? split_work_to_cores(core_grid, outer_dim_size * 2)
                                       : split_work_to_cores(core_grid, num_sticks_per_halo_dim * 2);

    // L1 Scratch CB Creation
    uint32_t l1_scratch_cb_page_size_bytes = page_size;

    uint32_t num_sticks_to_write_per_packet = 1;
    uint32_t cb_num_pages = 3 * num_sticks_to_write_per_packet;  // triple buffering
    tt::DataFormat df = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    // CBs for transferring data between reader and writer
    uint32_t sender_cb_index = tt::CB::c_in0;
    CircularBufferConfig cb_sender_config =
        CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
            .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, worker_core_ranges, cb_sender_config);

    // L1 receive buffer for 2D padding: fabric-delivered H halo corner sticks arrive here.
    // Corners-only optimization: only W-boundary sticks (pad2_left + pad2_right per row) go
    // to L1; non-corner sticks go directly to neighbor DRAM via fabric.
    // Buffer must hold ALL outer_dims' corner sticks (no per-outer_dim reuse) because the
    // fabric pipeline can deliver data for outer_dim N+1 before the reader finishes
    // copying outer_dim N.
    uint32_t recv_cb_index = tt::CB::c_in1;
    uint32_t corner_sticks_per_row =
        is_2d ? std::min(operation_attributes.pad2_left + operation_attributes.pad2_right, num_sticks_per_halo_dim) : 0;
    if (is_2d) {
        uint32_t max_padding = std::max(operation_attributes.padding_left, operation_attributes.padding_right);
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
        // W-axis device topology
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

        is_first_w_device = !w_backward_coord.has_value();
        is_last_w_device = !w_forward_coord.has_value();
        // W neighbors are physically adjacent (same row, adjacent columns) = 1 physical hop.
        // The fabric chain chip_id difference can be much larger (e.g., 4 on a 4x8 mesh),
        // but the EDM routing uses physical hops along the selected direction, NOT chain hops.
        // Using the chain distance would cause packets to overshoot the target.
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

        // Phase 2 processes all rows of the H-padded output tensor
        w_outer_dim_size = outer_dim_size * output_halo_dim_size;

        // CB and recv buffer on W fabric cores
        CreateCircularBuffer(program, w_fabric_core_range, cb_sender_config);

        // W recv: no L1 recv buffer needed. The W writer sends padding sticks directly to
        // the neighbor's output DRAM (same pattern as 1D H). The W reader just waits for
        // the completion semaphore. DRAM write ordering is guaranteed by synchronize_device()
        // before the next op dispatch.
        w_rows_per_link = w_outer_dim_size / pad2_num_links;
        w_extra_rows = w_outer_dim_size % pad2_num_links;
    }

    // Compute H fabric unicast route configuration (for compile-time args)
    auto [h_unicast_forward_args, h_unicast_backward_args] =
        ::ttnn::ccl::get_forward_backward_line_unicast_configuration(
            mesh_coordinate, forward_coord, backward_coord, mesh_device);

    // Compute H fabric multicast barrier route configuration
    auto [num_targets_forward, num_targets_backward] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
        operation_attributes.ring_size, device_index, operation_attributes.topology, false);
    auto [h_mcast_forward_args, h_mcast_backward_args] = ::ttnn::ccl::get_forward_backward_line_mcast_configuration(
        mesh_coordinate, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    // KERNEL CREATION — Consolidated: 6 kernels per device
    // Each logical role (H reader, H writer, local reader, local writer, W reader, W writer)
    // uses a single kernel on a CoreRangeSet. Per-core variation (direction, routing) is
    // handled via runtime args instead of separate compile-time-arg kernels.
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
        program,
        h_reader_kernel_id,
        {input_buffer->address(), output_buffer->address(), operation_attributes.h_neighbor_semaphore.address()});

    // Create consolidated H fabric writer kernel (uniform compile args across all H cores)
    auto h_writer_kernel_config = WriterDataMovementConfig{};
    h_writer_kernel_config.compile_args = {
        sender_cb_index,   // cb_output_id
        is_padding_zeros,  // is_padding_zeros
        page_size};        // stick_size
    TensorAccessorArgs(*output_buffer).append_to(h_writer_kernel_config.compile_args);
    h_writer_kernel_config.compile_args.push_back(is_2d ? 1 : 0);                   // use_l1_intermediate
    h_writer_kernel_config.compile_args.push_back(is_2d ? recv_cb_index : 0);       // recv_cb_id
    h_writer_kernel_config.compile_args.push_back(is_2d ? 1 : 0);                   // handle_incoming_writes
    h_writer_kernel_config.compile_args.push_back(0);                               // is_w_fabric_writer (false for H)
    h_writer_kernel_config.compile_args.push_back(operation_attributes.ring_size);  // ring_size
    auto h_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
        "minimal_default_writer.cpp",
        worker_core_ranges,
        h_writer_kernel_config);
    SetCommonRuntimeArgs(
        program,
        h_writer_kernel_id,
        {input_buffer->address(),
         output_buffer->address(),
         operation_attributes.h_neighbor_semaphore.address(),
         operation_attributes.barrier_semaphore.address()});

    // Set per-core runtime args for H fabric cores
    uint32_t link_offset_start_id = 0;
    uint32_t writer_link_offset_start_id = 0;  // separate offset for writer (uses output row width)
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t link_dims_to_read = 0;

        // direction 0 means pad left (top), 1 means pad right (bottom)
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

            // Reader runtime args (addresses in CRTAs, not here)
            std::vector<uint32_t> reader_rt_args = {
                (operation_attributes.dim > 0) ? link_offset_start_id * input_halo_dim_size
                                               : outer_dim_size - 1,                  // outer_dim_offset_start_id
                (operation_attributes.dim == 0) ? link_offset_start_id : 0,           // stick_start_id
                input_halo_dim_size,                                                  // input_halo_dim_size
                (operation_attributes.dim > 0) ? link_dims_to_read : outer_dim_size,  // outer_dim_size
                direction ? operation_attributes.padding_right : operation_attributes.padding_left,  // padding
                (operation_attributes.dim == 0) ? link_dims_to_read : num_sticks_per_halo_dim,  // num_sticks_to_read
                num_sticks_per_halo_dim,  // num_sticks_per_halo_dim
                corner_sticks_per_row};   // num_l1_recv_sticks_per_row (corners-only for 2D)
            // Per-core direction args (moved from compile-time for kernel consolidation)
            reader_rt_args.push_back(direction ? is_last_device : is_first_device);  // is_first_chip
            reader_rt_args.push_back(direction ? is_first_device : is_last_device);  // is_last_chip
            reader_rt_args.push_back(direction);                                     // direction
            SetRuntimeArgs(program, h_reader_kernel_id, {core}, reader_rt_args);

            // For 2D case, H fabric writer uses output row width and W offset
            uint32_t h_writer_num_sticks_per_halo_dim =
                is_2d ? output_num_sticks_per_halo_dim : num_sticks_per_halo_dim;
            uint32_t h_writer_stick_start =
                (operation_attributes.dim == 0) ? link_offset_start_id : writer_stick_start_id;
            uint32_t h_writer_num_sticks_to_read =
                (operation_attributes.dim == 0) ? link_dims_to_read : writer_num_sticks_to_read;

            // Writer runtime args (addresses in CRTAs, not here)
            std::vector<uint32_t> writer_rt_args = {
                (operation_attributes.dim > 0) ? writer_link_offset_start_id * output_halo_dim_size
                                               : outer_dim_size - 1,                  // outer_dim_offset_start_id
                h_writer_stick_start,                                                 // stick_start_id
                input_halo_dim_size,                                                  // input_halo_dim_size
                output_halo_dim_size,                                                 // output_halo_dim_size
                (operation_attributes.dim > 0) ? link_dims_to_read : outer_dim_size,  // outer_dim_size
                direction ? operation_attributes.padding_right : operation_attributes.padding_left,  // padding
                operation_attributes.padding_left,                                                   // padding left
                h_writer_num_sticks_to_read,       // num_sticks_to_read
                h_writer_num_sticks_per_halo_dim,  // num_sticks_per_halo_dim
                virtual_core.x,                    // neighbor_sem_noc0_x
                virtual_core.y,                    // neighbor_sem_noc0_y
                true,                              // use_barrier_semaphore
                virtual_opposite_core.x,           // barrier_sem_noc0_x
                virtual_opposite_core.y};          // barrier_sem_noc0_y
            // Phase 2 signal targets (W fabric reader cores for 2D padding)
            // Max targets = pad2_num_links * 2 directions (up to 8 W fabric cores)
            // sem_addr omitted — kernel reads barrier_sem from CRTA[3]
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
            // Per-core direction and routing args (moved from compile-time for kernel consolidation)
            writer_rt_args.push_back(direction ? is_last_device : is_first_device);  // is_first_chip
            writer_rt_args.push_back(direction ? is_first_device : is_last_device);  // is_last_chip
            writer_rt_args.push_back(direction);                                     // direction
            // Unicast route args: select forward or backward based on direction
            const auto& h_unicast_args = direction ? h_unicast_backward_args : h_unicast_forward_args;
            writer_rt_args.insert(writer_rt_args.end(), h_unicast_args.begin(), h_unicast_args.end());
            // Barrier multicast route info (6 args) for full-mesh startup barrier
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
            SetRuntimeArgs(program, h_writer_kernel_id, {core}, writer_rt_args);
        }
        if (operation_attributes.dim > 0) {
            link_offset_start_id += (link_dims_to_read * num_sticks_per_halo_dim);
            // Writer offset uses output row width (wider for 2D due to W padding)
            writer_link_offset_start_id += (link_dims_to_read * output_num_sticks_per_halo_dim);
        } else {
            link_offset_start_id += link_dims_to_read;
            writer_link_offset_start_id += link_dims_to_read;
        }
    }

    // Local copy workers on cores not used by fabric: AllCores - FabricCores
    std::vector<CoreCoord> local_copy_core_coords;
    KernelHandle local_reader_kernel_id = 0;
    KernelHandle local_writer_kernel_id = 0;
    bool has_local_copy = false;
    {
        CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
        CoreRangeSet fabric_cores = worker_core_ranges;
        if (is_2d) {
            fabric_cores = fabric_cores.merge(w_fabric_core_range);
        }
        CoreRangeSet local_copy_cores = all_cores.subtract(fabric_cores);

        if (!local_copy_cores.empty()) {
            has_local_copy = true;
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
                {input_buffer->address(),    // CRTA[0]
                 output_buffer->address(),   // CRTA[1] (unused by reader, reserved for consistency)
                 0u,                         // CRTA[2]: stick_start_id (always 0)
                 input_halo_dim_size,        // CRTA[3]
                 num_sticks_per_halo_dim,    // CRTA[4]: num_sticks_to_read
                 num_sticks_per_halo_dim});  // CRTA[5]: num_sticks_per_halo_dim

            // Create consolidated local copy writer kernel (uniform compile args)
            auto local_writer_cfg = WriterDataMovementConfig{};
            local_writer_cfg.compile_args = {sender_cb_index, page_size};
            TensorAccessorArgs(*output_buffer).append_to(local_writer_cfg.compile_args);
            local_writer_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/local_copy_writer.cpp",
                local_copy_cores,
                local_writer_cfg);
            // Local writer CRTAs: addresses (changing) + shape/config params + Phase 2 targets (static)
            std::vector<uint32_t> local_writer_crta = {
                input_buffer->address(),                           // CRTA[0] (unused by writer, reserved)
                output_buffer->address(),                          // CRTA[1]
                operation_attributes.barrier_semaphore.address(),  // CRTA[2]
                writer_stick_start_id,                             // CRTA[3]
                input_halo_dim_size,                               // CRTA[4]
                output_halo_dim_size,                              // CRTA[5]
                operation_attributes.padding_left,                 // CRTA[6]
                writer_num_sticks_to_read,                         // CRTA[7]
                output_num_sticks_per_halo_dim,                    // CRTA[8]
                is_2d ? num_w_fabric_cores : 0u};                  // CRTA[9]: num_phase2_signal_targets
            // Phase 2 signal targets (x,y) × 8 — CRTA[10..25]
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

                // Per-core unique args: only work distribution (shape/config/targets all in CRTAs)
                SetRuntimeArgs(program, local_reader_kernel_id, {logical_core}, {unit_offset, units_for_core});
                SetRuntimeArgs(program, local_writer_kernel_id, {logical_core}, {unit_offset, units_for_core});

                unit_offset += units_for_core;
            }
        }
    }

    // Phase 2: W fabric kernel creation (for 2D padding)
    KernelHandle w_reader_kernel_id = 0;
    KernelHandle w_writer_kernel_id = 0;
    if (is_2d) {
        // Each H fabric writer and local copy writer signals Phase 2 exactly once,
        // after ALL their work is complete (main loop + handle_incoming_writes).
        uint32_t barrier_count = num_h_fabric_cores + static_cast<uint32_t>(local_copy_core_coords.size());
        log_trace(
            tt::LogOp,
            "NeighborPad2D: barrier_count={} (h_writers={} local_writers={}), "
            "w_outer_dim_size={}, is_first_h={}, is_last_h={}, is_first_w={}, is_last_w={}, "
            "output_row_width={}, num_interior_sticks={}, pad2_left={}",
            barrier_count,
            num_h_fabric_cores,
            local_copy_core_coords.size(),
            w_outer_dim_size,
            is_first_device,
            is_last_device,
            is_first_w_device,
            is_last_w_device,
            output_num_sticks_per_halo_dim,
            num_sticks_per_halo_dim,
            operation_attributes.pad2_left);

        // W-axis startup barrier: compute ring size, device index, and multicast routes.
        // W writers use this barrier to synchronize with all W-axis devices (same row),
        // ensuring the previous dispatch has completed before sending new fabric data.
        const auto& mesh_view_w = mesh_device->get_view();
        uint32_t w_ring_size =
            (operation_attributes.pad2_cluster_axis.value() == 0) ? mesh_view_w.num_rows() : mesh_view_w.num_cols();
        uint32_t w_device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, operation_attributes.pad2_cluster_axis);
        auto [w_num_targets_forward, w_num_targets_backward] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
            w_ring_size, w_device_index, operation_attributes.topology, false);
        auto [w_mcast_forward_args, w_mcast_backward_args] = ::ttnn::ccl::get_forward_backward_line_mcast_configuration(
            mesh_coordinate,
            w_forward_coord,
            w_backward_coord,
            w_num_targets_forward,
            w_num_targets_backward,
            mesh_device);

        // Create consolidated W fabric reader kernel (uniform compile args across all W cores)
        auto w_reader_kernel_config = ReaderDataMovementConfig{};
        w_reader_kernel_config.compile_args = {
            sender_cb_index,   // cb_output_id
            is_padding_zeros,  // is_padding_zeros
            page_size};        // stick_size
        TensorAccessorArgs(*output_buffer).append_to(w_reader_kernel_config.compile_args);
        w_reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
            "phase2_w_reader.cpp",
            w_fabric_core_range,
            w_reader_kernel_config);
        SetCommonRuntimeArgs(
            program,
            w_reader_kernel_id,
            {output_buffer->address(),
             operation_attributes.barrier_semaphore.address(),
             operation_attributes.w_neighbor_semaphore.address()});

        // Create consolidated W fabric writer kernel (uniform compile args across all W cores)
        auto w_writer_kernel_config = WriterDataMovementConfig{};
        w_writer_kernel_config.compile_args = {
            sender_cb_index,   // cb_output_id
            is_padding_zeros,  // is_padding_zeros
            page_size};        // stick_size
        TensorAccessorArgs(*output_buffer).append_to(w_writer_kernel_config.compile_args);
        w_writer_kernel_config.compile_args.push_back(0);  // use_l1_intermediate (direct-to-DRAM for W)
        w_writer_kernel_config.compile_args.push_back(0);  // recv_cb_id (unused)
        w_writer_kernel_config.compile_args.push_back(0);  // handle_incoming_writes (data goes direct to DRAM)
        w_writer_kernel_config.compile_args.push_back(1);              // is_w_fabric_writer (W writer: true)
        w_writer_kernel_config.compile_args.push_back(w_ring_size);    // ring_size
        w_writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
            "minimal_default_writer.cpp",
            w_fabric_core_range,
            w_writer_kernel_config);
        SetCommonRuntimeArgs(
            program,
            w_writer_kernel_id,
            {output_buffer->address(),
             output_buffer->address(),
             operation_attributes.w_neighbor_semaphore.address(),
             // Use h_neighbor_semaphore (not barrier_semaphore) for W startup barrier:
             // W reader (NCRISC) on the same core uses barrier_semaphore for Phase 2 barrier.
             operation_attributes.h_neighbor_semaphore.address()});

        // Set per-core runtime args for W fabric cores
        for (uint32_t w_link = 0; w_link < pad2_num_links; w_link++) {
            // Per-link work distribution: split w_outer_dim_size rows across pad2_num_links
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

                // W reader runtime args (addresses in CRTAs, not here)
                std::vector<uint32_t> w_reader_rt_args = {
                    w_link_count,  // outer_dim_size (per-link)
                    w_link_start,  // outer_dim_start (per-link)
                    w_direction ? operation_attributes.pad2_right : operation_attributes.pad2_left,  // padding
                    barrier_count,
                    output_num_sticks_per_halo_dim,  // output_row_width (W + 2*pW)
                    operation_attributes.pad2_left,  // pad2_left
                    num_sticks_per_halo_dim};        // num_interior_sticks (W)
                // Per-core direction args (moved from compile-time for kernel consolidation)
                w_reader_rt_args.push_back(w_direction ? is_last_w_device : is_first_w_device);  // is_first_chip
                w_reader_rt_args.push_back(w_direction ? is_first_w_device : is_last_w_device);  // is_last_chip
                w_reader_rt_args.push_back(w_direction);                                         // direction
                SetRuntimeArgs(program, w_reader_kernel_id, {w_core}, w_reader_rt_args);

                // W writer runtime args (addresses in CRTAs, not here)
                std::vector<uint32_t> w_writer_rt_args = {
                    w_link_start * output_num_sticks_per_halo_dim,  // outer_dim_offset_start_id (per-link)
                    0,                                              // stick_start_id
                    num_sticks_per_halo_dim,                        // input_halo_dim_size (unused by writer)
                    output_num_sticks_per_halo_dim,                 // output_halo_dim_size = W'
                    w_link_count,                                   // outer_dim_size (per-link)
                    w_direction ? operation_attributes.pad2_right : operation_attributes.pad2_left,  // padding
                    operation_attributes.pad2_left,                                                  // padding_left
                    1,                 // num_sticks_to_read
                    1,                 // num_sticks_per_halo_dim
                    w_virtual_core.x,  // neighbor_sem_noc0_x
                    w_virtual_core.y,  // neighbor_sem_noc0_y
                    true,              // use_barrier_semaphore (W writers: W-axis startup barrier)
                    w_fabric_virtual_cores[(w_link * 2) + (1 - w_direction)].x,   // barrier_sem_noc0_x (opp W dir)
                    w_fabric_virtual_cores[(w_link * 2) + (1 - w_direction)].y};  // barrier_sem_noc0_y (opp W dir)
                // No Phase 2 signal targets (W writers ARE Phase 2)
                // sem_addr omitted — kernel reads barrier_sem from CRTA[3]
                constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 8;
                w_writer_rt_args.push_back(0);
                for (uint32_t s = 0; s < MAX_PHASE2_SIGNAL_TARGETS * 2; s++) {
                    w_writer_rt_args.push_back(0);
                }
                // Per-core direction and routing args (moved from compile-time for kernel consolidation)
                w_writer_rt_args.push_back(w_direction ? is_last_w_device : is_first_w_device);  // is_first_chip
                w_writer_rt_args.push_back(w_direction ? is_first_w_device : is_last_w_device);  // is_last_chip
                w_writer_rt_args.push_back(w_direction);                                         // direction
                // W fabric unicast route args: manually constructed with actual hop distances
                uint32_t w_device_offset = w_direction ? w_backward_device_offset : w_forward_device_offset;
                w_writer_rt_args.push_back(0);                // dst_mesh_id (unused for 1D)
                w_writer_rt_args.push_back(w_device_offset);  // distance_in_hops
                // W barrier multicast route info (6 args)
                const auto& w_mcast_args = w_direction ? w_mcast_backward_args : w_mcast_forward_args;
                w_writer_rt_args.insert(w_writer_rt_args.end(), w_mcast_args.begin(), w_mcast_args.end());
                // Fabric connection args: W neighbors are physically adjacent (1 hop via E/W
                // ethernet), so append_fabric_connection_rt_args correctly finds them.
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
                SetRuntimeArgs(program, w_writer_kernel_id, {w_core}, w_writer_rt_args);
            }
        }
    }

    return cached_program_t(
        std::move(program),
        NeighborPadAsyncSharedVariables{
            .h_reader_kernel_id = h_reader_kernel_id,
            .h_writer_kernel_id = h_writer_kernel_id,
            .local_reader_kernel_id = local_reader_kernel_id,
            .local_writer_kernel_id = local_writer_kernel_id,
            .w_reader_kernel_id = w_reader_kernel_id,
            .w_writer_kernel_id = w_writer_kernel_id,
            .has_local_copy = has_local_copy,
            .has_w_fabric = is_2d});
}

}  // namespace ttnn::experimental::prim
