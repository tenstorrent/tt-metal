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
#include "ttnn/tensor/tensor_impl.hpp"
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

namespace ttnn::operations::experimental::ccl::neighbor_pad {

NeighborPadAsyncMeshWorkloadFactory::cached_mesh_workload_t NeighborPadAsyncMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
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
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
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
                worker_reader_runtime_args[9] = operation_attributes.final_semaphore.address();

                // writer
                auto& worker_writer_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_runtime_args[0] = input.buffer()->address();
                worker_writer_runtime_args[1] = output.buffer()->address();
                worker_writer_runtime_args[13] = operation_attributes.final_semaphore.address();
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
    }
}

NeighborPadAsyncMeshWorkloadFactory::cached_program_t NeighborPadAsyncMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto* mesh_device = tensor_args.input_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.input_tensor.device();
    std::vector<IDevice*> devices_to_use = {};
    const auto& mesh_view = tensor_args.input_tensor.device()->get_view();
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    devices_to_use = (operation_attributes.cluster_axis == 0) ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                                              : mesh_view.get_devices_on_row(mesh_coordinate[0]);
    uint32_t target_ring_size = devices_to_use.size();

    // cluster_axis
    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < target_ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            }
            if (i != target_ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            }
        }
    }

    // Program creation
    Program program{};

    // Tensor Info
    const auto& input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto& output_tensor_shape = tensor_return_value.padded_shape();
    Buffer* input_buffer = tensor_args.input_tensor.buffer();
    Buffer* output_buffer = tensor_return_value.buffer();

    // Get OP Config, topology config
    uint32_t page_size = tensor_args.input_tensor.buffer()->page_size();
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
        is_first_device = !backward_device.has_value();
        is_last_device = !forward_device.has_value();
        if (!is_first_device) {
            backward_device_offset = 1;
        }
        if (!is_last_device) {
            forward_device_offset = 1;
        }
    }

    bool is_padding_zeros = operation_attributes.padding_mode == "zeros";

    // Get worker cores
    CoreCoord core_grid(operation_attributes.num_links * 2, 1);
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
                num_sticks_per_halo_dim,                        // num_sticks_per_halo_dim
                operation_attributes.final_semaphore.address()  // out_ready_sem_bank_addr (absolute address)
            };
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
            auto worker_writer_kernel_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
                "minimal_default_writer.cpp",
                {core},
                writer_kernel_config);
            writer_kernel_ids.push_back(worker_writer_kernel_id);

            std::vector<uint32_t> writer_rt_args = {
                tensor_args.input_tensor.buffer()->address(),  // input_tensor_address
                tensor_return_value.buffer()->address(),       // output_tensor_address
                (operation_attributes.dim > 0) ? link_offset_start_id * output_halo_dim_size
                                               : outer_dim_size - 1,  // link_offset_start_id
                (operation_attributes.dim == 0) ? link_offset_start_id : 0,
                input_halo_dim_size,                                                  // input_halo_dim_size
                output_halo_dim_size,                                                 // output_halo_dim_size
                (operation_attributes.dim > 0) ? link_dims_to_read : outer_dim_size,  // outer_dim_size
                direction ? operation_attributes.padding_right : operation_attributes.padding_left,  // padding
                operation_attributes.padding_left,                                                   // padding left
                (operation_attributes.dim == 0) ? link_dims_to_read : num_sticks_per_halo_dim,  // num_sticks_to_read
                num_sticks_per_halo_dim,                         // num_sticks_per_halo_dim
                virtual_core.x,                                  // out_ready_sem_noc0_x
                virtual_core.y,                                  // out_ready_sem_noc0_y
                operation_attributes.final_semaphore.address(),  // out_ready_sem_bank_addr (absolute address)
                true,                                            // use_barrier_semaphore
                virtual_opposite_core.x,                         // barrier_sem_noc0_x
                virtual_opposite_core.y,                         // barrier_sem_noc0_y
                operation_attributes.barrier_semaphore.address(),
                direction ? backward_device_offset : forward_device_offset,
                direction ? backward_device_offset : forward_device_offset};
            if (direction) {
                writer_rt_args.push_back(false);
                writer_rt_args.push_back(backward_device.has_value());
                if (backward_device.has_value()) {
                    const auto src_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
                    const auto dst_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
                }
            } else {
                writer_rt_args.push_back(forward_device.has_value());

                if (forward_device.has_value()) {
                    const auto src_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
                    const auto dst_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
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
        auto compute_grid = target_device->compute_with_storage_grid_size();
        CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid.x - 1, compute_grid.y - 1}));
        CoreRangeSet fabric_cores = worker_core_ranges;
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
                const uint32_t writer_total_rows_start = unit_offset;
                const uint32_t writer_stick_start_id = 0;
                const uint32_t writer_rows_count = units_for_core;
                const uint32_t writer_num_sticks_to_read = num_sticks_per_halo_dim;

                std::vector<uint32_t> local_writer_rt_args = {
                    tensor_args.input_tensor.buffer()->address(),  // input_tensor_address (unused by writer)
                    tensor_return_value.buffer()->address(),       // output_tensor_address
                    writer_total_rows_start,
                    writer_stick_start_id,
                    input_halo_dim_size,
                    output_halo_dim_size,
                    writer_rows_count,
                    operation_attributes.padding_left,
                    writer_num_sticks_to_read,
                    num_sticks_per_halo_dim};
                SetRuntimeArgs(program, local_writer_kernel_id, {logical_core}, local_writer_rt_args);

                unit_offset += units_for_core;
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
            .num_directions = num_directions});
}

}  // namespace ttnn::operations::experimental::ccl::neighbor_pad
