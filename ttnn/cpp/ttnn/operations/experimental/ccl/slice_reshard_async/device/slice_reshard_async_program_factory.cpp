// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <ranges>
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

SliceReshardAsyncProgramFactory::cached_mesh_workload_t SliceReshardAsyncProgramFactory::create_mesh_workload(
    const SliceReshardAsyncParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Tensor& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(args, coord, tensor_args, tensor_return_value);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_vars.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

SliceReshardAsyncProgramFactory::cached_program_t SliceReshardAsyncProgramFactory::create_at(
    const SliceReshardAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const Tensor& tensor_args,
    Tensor& tensor_return_value) {
    const ttnn::Tensor& input_tensor = tensor_args;
    const ttnn::Tensor& output_tensor = tensor_return_value;

    tt::tt_metal::Program program{};

    auto* mesh_device = input_tensor.device();
    IDevice* sender_device = mesh_device ? mesh_device->get_device(mesh_coord) : input_tensor.device();
    std::vector<IDevice*> devices_to_use = {};
    const auto& mesh_view = input_tensor.device()->get_view();
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    devices_to_use = (args.cluster_axis == 0) ? mesh_view.get_devices_on_column(mesh_coord[1])
                                              : mesh_view.get_devices_on_row(mesh_coord[0]);
    uint32_t ring_size = devices_to_use.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t ring_index = 0;  // Initialize ring (device) index
    for (uint32_t i = 0; i < ring_size; ++i) {
        if (devices_to_use.at(i) == sender_device) {
            ring_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            }
            if (i != ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            }
        }
    }

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* output_buffer = output_tensor.buffer();

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    uint32_t num_sticks_per_outer_dim = input_tensor_shape[1] * input_tensor_shape[2];
    uint32_t input_outer_dim_size = input_tensor_shape[0];
    uint32_t output_outer_dim_size = output_tensor_shape[0];
    bool is_first_device = !backward_device.has_value();
    bool is_last_device = !forward_device.has_value();
    // output coords for this device, in the input space
    uint32_t global_output_outer_dim_start = args.output_dim_offset + (output_outer_dim_size * ring_index);
    uint32_t global_output_outer_dim_end = args.output_dim_offset + (output_outer_dim_size * (ring_index + 1)) - 1;
    // input coords for this device, in the input space
    uint32_t global_input_outer_dim_start = input_outer_dim_size * ring_index;
    uint32_t global_input_outer_dim_end = (input_outer_dim_size * (ring_index + 1)) - 1;

    int32_t backward_device_end = (int32_t)global_input_outer_dim_start - 1;
    uint32_t outer_dims_from_backward = std::max(backward_device_end - (int32_t)global_output_outer_dim_start + 1, 0);
    int32_t forward_device_start = global_input_outer_dim_end + 1;
    uint32_t outer_dims_from_forward = std::max((int32_t)global_output_outer_dim_end - forward_device_start + 1, 0);
    uint32_t outer_dims_to_keep_start =
        std::max((int32_t)global_output_outer_dim_start - (int32_t)global_input_outer_dim_start, 0);
    uint32_t outer_dims_to_keep_end = std::min(
        outer_dims_to_keep_start - outer_dims_from_backward + output_outer_dim_size - 1, input_outer_dim_size - 1);
    int32_t backward_device_output_end =
        std::max((int32_t)global_output_outer_dim_start - 1, (int32_t)args.output_dim_offset - 1);
    uint32_t outer_dims_to_backward =
        is_first_device ? 0 : std::max(backward_device_output_end - (int32_t)global_input_outer_dim_start + 1, 0);
    int32_t forward_device_output_start =
        std::min(global_output_outer_dim_end + 1, output_outer_dim_size * ring_size - 1);
    uint32_t outer_dims_to_forward =
        is_last_device ? 0 : std::max((int32_t)global_input_outer_dim_end - forward_device_output_start + 1, 0);

    // Get worker cores
    CoreCoord core_grid(args.num_links * 2, 1);
    auto
        [num_cores,
         worker_core_ranges,
         core_group_1,
         core_group_2,
         num_sticks_per_core_group_1,
         num_sticks_per_core_group_2] = tt::tt_metal::split_work_to_cores(core_grid, num_sticks_per_outer_dim * 2);

    // L1 Scratch CB Creation
    uint32_t l1_scratch_cb_page_size_bytes = page_size;

    uint32_t num_sticks_to_write_per_packet = 1;
    uint32_t cb_num_pages = 3 * num_sticks_to_write_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // CBs for transferring data between reader and writer
    uint32_t sender_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_cb_index, df}})
            .set_page_size(sender_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, worker_core_ranges, cb_sender_config);

    // KERNEL CREATION
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    uint32_t num_directions = 2;
    uint32_t stick_start_id = 0;
    for (uint32_t link = 0; link < args.num_links; link++) {
        uint32_t num_sticks_to_read = 0;
        for (uint32_t direction = 0; direction < num_directions; direction++) {
            CoreCoord core = {(link * num_directions) + direction, 0};
            CoreCoord opposite_core = {(link * num_directions) + (1 - direction), 0};
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
            CoreCoord virtual_opposite_core = mesh_device->worker_core_from_logical_core(opposite_core);
            if (core_group_1.contains(core)) {
                num_sticks_to_read = num_sticks_per_core_group_1;
            } else {
                num_sticks_to_read = num_sticks_per_core_group_2;
            }

            // Reader
            auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
            reader_kernel_config.compile_args = {
                direction ? is_first_device : is_last_device,
                direction ? is_last_device : is_first_device,
                sender_cb_index,  // cb_forward_id
                direction,
                page_size,
            };
            TensorAccessorArgs(*input_buffer).append_to(reader_kernel_config.compile_args);
            auto worker_reader_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                "minimal_default_reader.cpp",
                {core},
                reader_kernel_config);
            reader_kernel_ids.push_back(worker_reader_kernel_id);

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),                            // input_tensor_address
                stick_start_id,                                              // stick_start_id
                num_sticks_to_read,                                          // num_sticks_to_read
                input_outer_dim_size,                                        // input_outer_dim_size
                direction ? outer_dims_to_forward : outer_dims_to_backward,  // outer_dims_to_forward
                outer_dims_from_forward,                                     // outer_dims_from_forward
                outer_dims_to_keep_start,                                    // outer_dims_to_keep
                outer_dims_to_keep_end,                                      // outer_dims_to_keep
                num_sticks_per_outer_dim,                                    // num_sticks_per_outer_dim
                args.final_semaphore.address()  // out_ready_sem_bank_addr (absolute address)
            };
            tt::tt_metal::SetRuntimeArgs(program, worker_reader_kernel_id, {core}, reader_rt_args);

            // Writer
            auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
            writer_kernel_config.compile_args = {
                direction ? is_first_device : is_last_device,
                direction ? is_last_device : is_first_device,
                sender_cb_index,  // cb_forward_id
                direction,
            };
            TensorAccessorArgs(*output_buffer).append_to(writer_kernel_config.compile_args);
            auto worker_writer_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                "minimal_default_writer.cpp",
                {core},
                writer_kernel_config);
            writer_kernel_ids.push_back(worker_writer_kernel_id);

            std::vector<uint32_t> writer_rt_args = {
                input_tensor.buffer()->address(),                                // input_tensor_address
                output_tensor.buffer()->address(),                               // output_tensor_address
                page_size,                                                       // stick_size
                stick_start_id,                                                  // stick_start_id
                num_sticks_to_read,                                              // num_sticks_to_read
                output_outer_dim_size,                                           // output_outer_dim_size
                direction ? outer_dims_to_forward : outer_dims_to_backward,      // outer_dims_to_forward
                outer_dims_to_keep_start,                                        // outer_dims_to_keep
                outer_dims_to_keep_end,                                          // outer_dims_to_keep
                direction ? outer_dims_from_backward : outer_dims_from_forward,  // outer_dims_to_receive
                outer_dims_from_forward,                                         // outer_dims_from_forward
                num_sticks_per_outer_dim,                                        // num_sticks_per_outer_dim
                virtual_core.x,                                                  // out_ready_sem_noc0_x
                virtual_core.y,                                                  // out_ready_sem_noc0_y
                args.final_semaphore.address(),  // out_ready_sem_bank_addr (absolute address)
                true,                            // use_barrier_semaphore
                virtual_opposite_core.x,         // barrier_sem_noc0_x
                virtual_opposite_core.y,         // barrier_sem_noc0_y
                args.barrier_semaphore.address(),
            };
            if (direction) {
                writer_rt_args.push_back(forward_device.has_value());
                if (forward_device.has_value()) {
                    const auto src_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                    const auto dst_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
                }
                writer_rt_args.push_back(false);
            } else {
                writer_rt_args.push_back(false);
                writer_rt_args.push_back(backward_device.has_value());
                if (backward_device.has_value()) {
                    const auto src_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
                    const auto dst_fabric_node_id =
                        tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        src_fabric_node_id, dst_fabric_node_id, link, program, {core}, writer_rt_args);
                }
            }
            tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel_id, {core}, writer_rt_args);
        }
        stick_start_id += num_sticks_to_read;
    }

    return cached_program_t{std::move(program), {num_directions, reader_kernel_ids, writer_kernel_ids}};
}

void SliceReshardAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SliceReshardAsyncParams& args,
    const Tensor& tensor_args,
    Tensor& tensor_return_value) {
    const ttnn::Tensor& output_tensor = tensor_return_value;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        uint32_t core_idx = 0;
        for (uint32_t link = 0; link < args.num_links; link++) {
            for (uint32_t direction = 0; direction < shared_vars.num_directions; direction++) {
                CoreCoord core = {(link * shared_vars.num_directions) + direction, 0};
                auto& reader_runtime_args = GetRuntimeArgs(program, shared_vars.reader_kernel_ids[core_idx]);
                auto& writer_runtime_args = GetRuntimeArgs(program, shared_vars.writer_kernel_ids[core_idx]);

                // reader
                auto& worker_reader_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_runtime_args[0] = tensor_args.buffer()->address();
                worker_reader_runtime_args[9] = args.final_semaphore.address();

                // writer
                auto& worker_writer_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_runtime_args[0] = tensor_args.buffer()->address();
                worker_writer_runtime_args[1] = output_tensor.buffer()->address();
                worker_writer_runtime_args[14] = args.final_semaphore.address();
                worker_writer_runtime_args[18] = args.barrier_semaphore.address();

                core_idx++;
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
