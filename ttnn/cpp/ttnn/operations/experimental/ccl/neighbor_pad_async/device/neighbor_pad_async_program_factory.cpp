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
    // Update runtime arguments for each program in the workload
    for (auto& [coordinate_range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(coordinate_range);

        const auto& input = tensor_args.input_tensor;
        const auto& output = tensor_return_value;

        // Update readers/writers
        uint32_t core_idx = 0;
        for (uint32_t link = 0; link < shared_vars.num_links; link++) {
            // direction 0 means pad left (top), 1 means pad right (bottom)
            for (uint32_t direction = 0; direction < shared_vars.num_directions; direction++) {
                CoreCoord core = {(link * shared_vars.num_directions) + direction, 0};
                auto& reader_runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_ids[core_idx]);
                auto& writer_runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_ids[core_idx]);

                // reader
                auto& worker_reader_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_runtime_args[0] = input.buffer()->address();
                worker_reader_runtime_args[1] = output.buffer()->address();
                worker_reader_runtime_args[9] = operation_attributes.h_neighbor_semaphore.address();

                // writer
                auto& worker_writer_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_runtime_args[0] = input.buffer()->address();
                worker_writer_runtime_args[1] = output.buffer()->address();
                worker_writer_runtime_args[13] = operation_attributes.h_neighbor_semaphore.address();
                worker_writer_runtime_args[17] = operation_attributes.barrier_semaphore.address();

                core_idx++;
            }
        }
        // Local copy workers (addresses only)
        for (size_t i = 0; i < shared_vars.local_reader_kernel_ids.size(); ++i) {
            CoreCoord core = shared_vars.local_copy_core_coords[i];
            auto& reader_runtime_args = GetRuntimeArgs(program, shared_vars.local_reader_kernel_ids[i]);
            auto& writer_runtime_args = GetRuntimeArgs(program, shared_vars.local_writer_kernel_ids[i]);

            auto& worker_reader_runtime_args = reader_runtime_args[core.x][core.y];
            worker_reader_runtime_args[0] = input.buffer()->address();
            worker_reader_runtime_args[1] = output.buffer()->address();

            auto& worker_writer_runtime_args = writer_runtime_args[core.x][core.y];
            worker_writer_runtime_args[0] = input.buffer()->address();
            worker_writer_runtime_args[1] = output.buffer()->address();
        }

        // W fabric workers (Phase 2, for 2D padding)
        for (size_t i = 0; i < shared_vars.w_reader_kernel_ids.size(); ++i) {
            CoreCoord core = shared_vars.w_fabric_core_coords[i];

            // W reader
            auto& reader_runtime_args = GetRuntimeArgs(program, shared_vars.w_reader_kernel_ids[i]);
            auto& w_reader_args = reader_runtime_args[core.x][core.y];
            w_reader_args[2] = operation_attributes.barrier_semaphore.address();
            w_reader_args[4] = operation_attributes.w_neighbor_semaphore.address();
            w_reader_args[5] = output.buffer()->address();

            // W writer
            auto& writer_runtime_args = GetRuntimeArgs(program, shared_vars.w_writer_kernel_ids[i]);
            auto& w_writer_args = writer_runtime_args[core.x][core.y];
            w_writer_args[0] = output.buffer()->address();
            w_writer_args[1] = output.buffer()->address();
            w_writer_args[13] = operation_attributes.w_neighbor_semaphore.address();
        }
    }
}

// Fused 2D NeighborPad Algorithm (single op, two phases):
//
// Input: [B,T,H,W,C] fractured across 2D mesh (H across rows, W across columns)
// Output: [B,T,H+2pH,W+2pW,C]
//
// Phase 1 — Interior copy + H halo exchange (all ~120 cores active):
//   Local copy cores: read input sticks → write to output DRAM at (h+pH, w+pW) offset.
//   H fabric writer (BRISC): self-pad zeros/replicate to output DRAM for H pad rows.
//   H fabric reader (NCRISC): receive H halo from fabric → L1 → output DRAM.
//   All Phase 1 cores signal Phase 2 barrier on completion.
//
// Phase 2 — W halo exchange (2-4 W fabric cores only):
//   W reader: reads W boundary sticks from output DRAM (safe because Phase 1 calls
//     noc_async_write_barrier() before signaling the barrier semaphore).
//     Sends to neighbor via fabric or self-pads. Receives from neighbor → L1 → output DRAM.
//   W writer: writes self-pad to output DRAM, sends W boundary data via fabric.
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
    // InterleavedAddrGen spaces pages at aligned_page_size intervals, so NOC transfers
    // must use this size to avoid sub-minimum or misaligned reads.
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
        uint32_t secondary_cluster_axis_val = operation_attributes.secondary_cluster_axis.value_or((uint32_t)0);
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

    // Debug logging: device topology for H fabric
    log_info(
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
        log_info(
            tt::LogOp,
            "  forward_coord=({},{}), fwd_node_id={}",
            (*forward_coord)[0],
            (*forward_coord)[1],
            mesh_device->get_fabric_node_id(forward_coord.value()));
    }
    if (backward_coord.has_value()) {
        log_info(
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
        writer_num_sticks_to_read = num_sticks_per_halo_dim;  // still read original W sticks per row
    }

    // Get worker cores
    uint32_t num_h_fabric_cores = operation_attributes.num_links * 2;
    uint32_t num_w_fabric_cores = is_2d ? (operation_attributes.pad2_num_links * 2) : 0;
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

    // L1 receive buffer for 2D padding: fabric-delivered H halo data arrives here
    // instead of going directly to DRAM, so the reader can copy it with proper barriers.
    // Buffer must hold ALL outer_dims' sticks (no per-outer_dim reuse) because the
    // fabric pipeline can deliver data for outer_dim N+1 before the reader finishes
    // copying outer_dim N.
    uint32_t recv_cb_index = tt::CB::c_in1;
    if (is_2d) {
        uint32_t max_padding = std::max(operation_attributes.padding_left, operation_attributes.padding_right);
        uint32_t max_outer_dims_per_core = dims_per_core_group_1;
        uint32_t recv_total_sticks = max_outer_dims_per_core * max_padding * writer_num_sticks_to_read;
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
    uint32_t max_w_padding = 0;

    if (is_2d) {
        // W-axis device topology
        w_forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coordinate,
            1,
            ttnn::ccl::Topology::Linear,
            operation_attributes.pad2_cluster_axis.value());
        w_backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor,
            mesh_coordinate,
            -1,
            ttnn::ccl::Topology::Linear,
            operation_attributes.pad2_cluster_axis.value());

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

        log_info(
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
        max_w_padding = std::max(operation_attributes.pad2_left, operation_attributes.pad2_right);

        // CB and recv buffer on W fabric cores
        CreateCircularBuffer(program, w_fabric_core_range, cb_sender_config);

        // L1 recv buffer on W fabric cores: fabric-delivered W padding data arrives here
        // instead of going directly to DRAM. Buffer holds ALL rows (no reuse).
        uint32_t w_recv_total_sticks = w_outer_dim_size * max_w_padding;
        uint32_t w_recv_buf_size = w_recv_total_sticks * page_size;
        if (w_recv_buf_size > 0) {
            CircularBufferConfig w_recv_cb_config =
                CircularBufferConfig(w_recv_buf_size, {{recv_cb_index, df}}).set_page_size(recv_cb_index, page_size);
            CreateCircularBuffer(program, w_fabric_core_range, w_recv_cb_config);
        }
    }

    // Compute H fabric unicast route configuration (for compile-time args)
    auto [h_unicast_forward_args, h_unicast_backward_args] =
        ::ttnn::ccl::get_forward_backward_line_unicast_configuration(
            ttnn::ccl::Topology::Linear, mesh_coordinate, forward_coord, backward_coord, mesh_device);

    // KERNEL CREATION
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    uint32_t num_directions = 2;
    uint32_t link_offset_start_id = 0;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
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

            // Reader
            auto reader_kernel_config = ReaderDataMovementConfig{};
            // When direction == 0, first_device is leftmost, when direction == 1, first_device is rightmost
            reader_kernel_config.compile_args = {
                direction ? is_last_device : is_first_device,
                direction ? is_first_device : is_last_device,
                sender_cb_index,  // cb_forward_id
                direction,
                is_padding_zeros,
                page_size};
            TensorAccessorArgs(*input_buffer).append_to(reader_kernel_config.compile_args);
            reader_kernel_config.compile_args.push_back(is_2d ? 1 : 0);              // use_l1_intermediate
            reader_kernel_config.compile_args.push_back(is_2d ? recv_cb_index : 0);  // recv_cb_id
            auto worker_reader_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
                "minimal_default_reader.cpp",
                {core},
                reader_kernel_config);
            reader_kernel_ids.push_back(worker_reader_kernel_id);

            std::vector<uint32_t> reader_rt_args = {
                tensor_args.input_tensor.buffer()->address(),  // input_tensor_address
                tensor_return_value.buffer()->address(),       // output_tensor_address
                (operation_attributes.dim > 0) ? link_offset_start_id * input_halo_dim_size
                                               : outer_dim_size - 1,  // link_offset_start_id
                (operation_attributes.dim == 0) ? link_offset_start_id : 0,
                input_halo_dim_size,                                                  // input_halo_dim_size
                (operation_attributes.dim > 0) ? link_dims_to_read : outer_dim_size,  // outer_dim_size
                direction ? operation_attributes.padding_right : operation_attributes.padding_left,  // padding
                (operation_attributes.dim == 0) ? link_dims_to_read : num_sticks_per_halo_dim,  // num_sticks_to_read
                num_sticks_per_halo_dim,  // num_sticks_per_halo_dim
                operation_attributes.h_neighbor_semaphore.address()};
            SetRuntimeArgs(program, worker_reader_kernel_id, {core}, reader_rt_args);

            // Writer
            auto writer_kernel_config = WriterDataMovementConfig{};
            writer_kernel_config.compile_args = {
                direction ? is_last_device : is_first_device,
                direction ? is_first_device : is_last_device,
                sender_cb_index,  // cb_forward_id
                direction,
                is_padding_zeros,
                page_size};
            TensorAccessorArgs(*output_buffer).append_to(writer_kernel_config.compile_args);
            writer_kernel_config.compile_args.push_back(is_2d ? 1 : 0);              // use_l1_intermediate
            writer_kernel_config.compile_args.push_back(is_2d ? recv_cb_index : 0);  // recv_cb_id
            writer_kernel_config.compile_args.push_back(
                is_2d ? 1 : 0);  // handle_incoming_writes (H writer: yes for 2D)
            writer_kernel_config.compile_args.push_back(0);  // is_w_fabric_writer (H writer: false)
            // Unicast route args: select forward or backward based on direction
            const auto& h_unicast_args = direction ? h_unicast_backward_args : h_unicast_forward_args;
            writer_kernel_config.compile_args.insert(
                writer_kernel_config.compile_args.end(), h_unicast_args.begin(), h_unicast_args.end());
            auto worker_writer_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
                "minimal_default_writer.cpp",
                {core},
                writer_kernel_config);
            writer_kernel_ids.push_back(worker_writer_kernel_id);

            // For 2D case, H fabric writer uses output row width and W offset
            uint32_t h_writer_num_sticks_per_halo_dim =
                is_2d ? output_num_sticks_per_halo_dim : num_sticks_per_halo_dim;
            uint32_t h_writer_stick_start =
                (operation_attributes.dim == 0) ? link_offset_start_id : writer_stick_start_id;
            uint32_t h_writer_num_sticks_to_read =
                (operation_attributes.dim == 0) ? link_dims_to_read : writer_num_sticks_to_read;

            std::vector<uint32_t> writer_rt_args = {
                tensor_args.input_tensor.buffer()->address(),  // input_tensor_address
                tensor_return_value.buffer()->address(),       // output_tensor_address
                (operation_attributes.dim > 0) ? link_offset_start_id * output_halo_dim_size
                                               : outer_dim_size - 1,                  // link_offset_start_id
                h_writer_stick_start,                                                 // stick_start_id
                input_halo_dim_size,                                                  // input_halo_dim_size
                output_halo_dim_size,                                                 // output_halo_dim_size
                (operation_attributes.dim > 0) ? link_dims_to_read : outer_dim_size,  // outer_dim_size
                direction ? operation_attributes.padding_right : operation_attributes.padding_left,  // padding
                operation_attributes.padding_left,                                                   // padding left
                h_writer_num_sticks_to_read,                          // num_sticks_to_read
                h_writer_num_sticks_per_halo_dim,                     // num_sticks_per_halo_dim
                virtual_core.x,                                       // neighbor_sem_noc0_x
                virtual_core.y,                                       // neighbor_sem_noc0_y
                operation_attributes.h_neighbor_semaphore.address(),  // neighbor_sem_bank_addr
                true,                                                 // use_barrier_semaphore
                virtual_opposite_core.x,                              // barrier_sem_noc0_x
                virtual_opposite_core.y,                              // barrier_sem_noc0_y
                operation_attributes.barrier_semaphore.address()};
            // Phase 2 signal targets (W fabric reader cores for 2D padding)
            writer_rt_args.push_back(is_2d ? num_w_fabric_cores : 0);
            for (uint32_t s = 0; s < 2; s++) {
                if (is_2d && s < num_w_fabric_cores) {
                    writer_rt_args.push_back(w_fabric_virtual_cores[s].x);
                    writer_rt_args.push_back(w_fabric_virtual_cores[s].y);
                    writer_rt_args.push_back(operation_attributes.barrier_semaphore.address());
                } else {
                    writer_rt_args.push_back(0);
                    writer_rt_args.push_back(0);
                    writer_rt_args.push_back(0);
                }
            }
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
            SetRuntimeArgs(program, worker_writer_kernel_id, {core}, writer_rt_args);
        }
        if (operation_attributes.dim > 0) {
            link_offset_start_id += (link_dims_to_read * num_sticks_per_halo_dim);
        } else {
            link_offset_start_id += link_dims_to_read;
        }
    }

    // Local copy workers on cores not used by fabric: AllCores - FabricCores
    std::vector<KernelHandle> local_reader_kernel_ids;
    std::vector<KernelHandle> local_writer_kernel_ids;
    std::vector<CoreCoord> local_copy_core_coords;
    {
        auto compute_grid = mesh_device->compute_with_storage_grid_size();
        CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid.x - 1, compute_grid.y - 1}));
        CoreRangeSet fabric_cores = worker_core_ranges;
        if (is_2d) {
            fabric_cores = fabric_cores.merge(w_fabric_core_range);
        }
        CoreRangeSet local_copy_cores = all_cores.subtract(fabric_cores);

        if (!local_copy_cores.empty()) {
            // CB on all local-copy cores
            CreateCircularBuffer(program, local_copy_cores, cb_sender_config);

            // Distribute work evenly across local-copy cores
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

                // Local copy reader (no fabric)
                auto local_reader_cfg = ReaderDataMovementConfig{};
                local_reader_cfg.compile_args = {sender_cb_index, page_size};
                TensorAccessorArgs(*input_buffer).append_to(local_reader_cfg.compile_args);
                auto local_reader_kernel_id = CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/local_copy_reader.cpp",
                    {logical_core},
                    local_reader_cfg);
                local_reader_kernel_ids.push_back(local_reader_kernel_id);

                // Reader runtime args
                const uint32_t reader_total_rows_start = unit_offset;
                const uint32_t reader_stick_start_id = 0;
                const uint32_t reader_rows_count = units_for_core;
                const uint32_t reader_num_sticks_to_read = num_sticks_per_halo_dim;

                std::vector<uint32_t> local_reader_rt_args = {
                    tensor_args.input_tensor.buffer()->address(),  // input_tensor_address
                    tensor_return_value.buffer()->address(),       // output_tensor_address (unused)
                    reader_total_rows_start,
                    reader_stick_start_id,
                    input_halo_dim_size,
                    reader_rows_count,
                    reader_num_sticks_to_read,
                    num_sticks_per_halo_dim,
                };
                SetRuntimeArgs(program, local_reader_kernel_id, {logical_core}, local_reader_rt_args);

                // Local copy writer (no fabric)
                auto local_writer_cfg = WriterDataMovementConfig{};
                local_writer_cfg.compile_args = {sender_cb_index, page_size};
                TensorAccessorArgs(*output_buffer).append_to(local_writer_cfg.compile_args);
                auto local_writer_kernel_id = CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/local_copy_writer.cpp",
                    {logical_core},
                    local_writer_cfg);
                local_writer_kernel_ids.push_back(local_writer_kernel_id);

                // Writer runtime args
                const uint32_t lw_total_rows_start = unit_offset;
                const uint32_t lw_rows_count = units_for_core;

                std::vector<uint32_t> local_writer_rt_args = {
                    tensor_args.input_tensor.buffer()->address(),  // input_tensor_address (unused by writer)
                    tensor_return_value.buffer()->address(),       // output_tensor_address
                    lw_total_rows_start,
                    writer_stick_start_id,  // pad2_left for 2D, 0 for 1D
                    input_halo_dim_size,
                    output_halo_dim_size,
                    lw_rows_count,
                    operation_attributes.padding_left,
                    writer_num_sticks_to_read,        // original W sticks per row
                    output_num_sticks_per_halo_dim};  // W+2pW for 2D, W for 1D
                // Phase 2 signal targets (W fabric reader cores for 2D padding)
                local_writer_rt_args.push_back(is_2d ? num_w_fabric_cores : 0);
                for (uint32_t s = 0; s < 2; s++) {
                    if (is_2d && s < num_w_fabric_cores) {
                        local_writer_rt_args.push_back(w_fabric_virtual_cores[s].x);
                        local_writer_rt_args.push_back(w_fabric_virtual_cores[s].y);
                        local_writer_rt_args.push_back(operation_attributes.barrier_semaphore.address());
                    } else {
                        local_writer_rt_args.push_back(0);
                        local_writer_rt_args.push_back(0);
                        local_writer_rt_args.push_back(0);
                    }
                }
                SetRuntimeArgs(program, local_writer_kernel_id, {logical_core}, local_writer_rt_args);

                unit_offset += units_for_core;
            }
        }
    }

    // Phase 2: W fabric kernel creation (for 2D padding)
    std::vector<KernelHandle> w_reader_kernel_ids;
    std::vector<KernelHandle> w_writer_kernel_ids;
    if (is_2d) {
        // Each H fabric writer and local copy writer signals Phase 2 exactly once,
        // after ALL their work is complete (main loop + handle_incoming_writes).
        uint32_t barrier_count = static_cast<uint32_t>(writer_kernel_ids.size() + local_writer_kernel_ids.size());
        log_info(
            tt::LogOp,
            "NeighborPad2D: barrier_count={} (h_writers={} local_writers={}), "
            "w_outer_dim_size={}, is_first_h={}, is_last_h={}, is_first_w={}, is_last_w={}, "
            "output_row_width={}, num_interior_sticks={}, pad2_left={}",
            barrier_count,
            writer_kernel_ids.size(),
            local_writer_kernel_ids.size(),
            w_outer_dim_size,
            is_first_device,
            is_last_device,
            is_first_w_device,
            is_last_w_device,
            output_num_sticks_per_halo_dim,
            num_sticks_per_halo_dim,
            operation_attributes.pad2_left);

        for (uint32_t w_link = 0; w_link < operation_attributes.pad2_num_links; w_link++) {
            for (uint32_t w_direction = 0; w_direction < 2; w_direction++) {
                uint32_t w_core_idx = w_link * 2 + w_direction;
                CoreCoord w_core = w_fabric_logical_cores[w_core_idx];
                CoreCoord w_virtual_core = w_fabric_virtual_cores[w_core_idx];
                // CoreCoord w_opposite_virtual_core = w_fabric_virtual_cores[w_link * 2 + (1 - w_direction)];

                // Phase 2 W reader kernel — reads boundary sticks from output DRAM.
                auto w_reader_kernel_config = ReaderDataMovementConfig{};
                w_reader_kernel_config.compile_args = {
                    w_direction ? is_last_w_device : is_first_w_device,  // is_first_chip
                    w_direction ? is_first_w_device : is_last_w_device,  // is_last_chip
                    sender_cb_index,                                     // cb_output_id
                    w_direction,                                         // direction
                    is_padding_zeros,
                    page_size};  // stick_size
                TensorAccessorArgs(*output_buffer).append_to(w_reader_kernel_config.compile_args);
                w_reader_kernel_config.compile_args.push_back(recv_cb_index);  // recv_cb_id
                auto w_reader_kernel_id = CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
                    "phase2_w_reader.cpp",
                    {w_core},
                    w_reader_kernel_config);
                w_reader_kernel_ids.push_back(w_reader_kernel_id);

                std::vector<uint32_t> w_reader_rt_args = {
                    w_outer_dim_size,                                                                // outer_dim_size
                    w_direction ? operation_attributes.pad2_right : operation_attributes.pad2_left,  // padding
                    operation_attributes.barrier_semaphore.address(),                                // barrier_sem_addr
                    barrier_count,
                    operation_attributes.w_neighbor_semaphore.address(),
                    tensor_return_value.buffer()->address(),  // output_tensor_address
                    output_num_sticks_per_halo_dim,           // output_row_width (W + 2*pW)
                    operation_attributes.pad2_left,           // pad2_left
                    num_sticks_per_halo_dim};                 // num_interior_sticks (W)
                SetRuntimeArgs(program, w_reader_kernel_id, {w_core}, w_reader_rt_args);

                // Phase 2 W writer kernel (reuses minimal_default_writer)
                auto w_writer_kernel_config = WriterDataMovementConfig{};
                w_writer_kernel_config.compile_args = {
                    w_direction ? is_last_w_device : is_first_w_device,
                    w_direction ? is_first_w_device : is_last_w_device,
                    sender_cb_index,
                    w_direction,
                    is_padding_zeros,
                    page_size};
                TensorAccessorArgs(*output_buffer).append_to(w_writer_kernel_config.compile_args);
                w_writer_kernel_config.compile_args.push_back(1);              // use_l1_intermediate
                w_writer_kernel_config.compile_args.push_back(recv_cb_index);  // recv_cb_id
                w_writer_kernel_config.compile_args.push_back(1);              // handle_incoming_writes (W writer: yes)
                w_writer_kernel_config.compile_args.push_back(1);              // is_w_fabric_writer (W writer: true)
                // W fabric unicast route args: manually constructed with actual hop distances
                // (standard get_forward_backward_line_unicast_configuration hardcodes distance=1 for 1D)
                uint32_t w_device_offset = w_direction ? w_backward_device_offset : w_forward_device_offset;
                w_writer_kernel_config.compile_args.push_back(0);                // dst_mesh_id (unused for 1D)
                w_writer_kernel_config.compile_args.push_back(w_device_offset);  // distance_in_hops
                auto w_writer_kernel_id = CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
                    "minimal_default_writer.cpp",
                    {w_core},
                    w_writer_kernel_config);
                w_writer_kernel_ids.push_back(w_writer_kernel_id);

                std::vector<uint32_t> w_writer_rt_args = {
                    tensor_return_value.buffer()->address(),  // input_tensor_address (unused)
                    tensor_return_value.buffer()->address(),  // output_tensor_address
                    0,                                        // outer_dim_offset_start_id
                    0,                                        // stick_start_id
                    num_sticks_per_halo_dim,                  // input_halo_dim_size (unused by writer)
                    output_num_sticks_per_halo_dim,           // output_halo_dim_size = W'
                    w_outer_dim_size,                         // outer_dim_size = B*T*(H+2pH)
                    w_direction ? operation_attributes.pad2_right : operation_attributes.pad2_left,  // padding
                    operation_attributes.pad2_left,                                                  // padding_left
                    1,                 // num_sticks_to_read
                    1,                 // num_sticks_per_halo_dim
                    w_virtual_core.x,  // neighbor_sem_noc0_x
                    w_virtual_core.y,  // neighbor_sem_noc0_y
                    operation_attributes.w_neighbor_semaphore.address(),
                    false,  // use_barrier_semaphore (W writers: no startup barrier)
                    0,      // barrier_sem_noc0_x (unused)
                    0,      // barrier_sem_noc0_y (unused)
                    0};     // barrier_sem (unused)
                // No Phase 3 signal targets
                w_writer_rt_args.push_back(0);
                for (uint32_t s = 0; s < 6; s++) {
                    w_writer_rt_args.push_back(0);
                }
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
            .reader_kernel_ids = std::move(reader_kernel_ids),
            .writer_kernel_ids = std::move(writer_kernel_ids),
            .local_reader_kernel_ids = std::move(local_reader_kernel_ids),
            .local_writer_kernel_ids = std::move(local_writer_kernel_ids),
            .local_copy_core_coords = std::move(local_copy_core_coords),
            .num_links = operation_attributes.num_links,
            .num_directions = num_directions,
            .w_reader_kernel_ids = std::move(w_reader_kernel_ids),
            .w_writer_kernel_ids = std::move(w_writer_kernel_ids),
            .w_fabric_core_coords = std::move(w_fabric_logical_cores),
            .num_w_links = is_2d ? operation_attributes.pad2_num_links : 0});
}

}  // namespace ttnn::experimental::prim
