// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_op.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_async/device/neighbor_pad_async_program.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks neighbor_pad_async_minimal(
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t padding_left,
    const uint32_t padding_right,
    const std::string& padding_mode,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const uint32_t num_links,
    ccl::Topology topology,
    const uint32_t ring_size,
    const uint32_t ring_index,
    std::optional<uint32_t> secondary_cluster_axis,
    const std::optional<std::vector<uint32_t>>& secondary_mesh_shape) {
    tt::tt_metal::Program program{};

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* output_buffer = output_tensor.buffer();

    auto mesh_device = input_tensor.device();

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    uint32_t num_sticks_per_halo_dim = 1;
    for (int d = dim + 1; d < input_tensor_shape.size() - 1; d++) {
        num_sticks_per_halo_dim *= input_tensor_shape[d];
    }
    uint32_t input_halo_dim_size = input_tensor_shape[dim];
    uint32_t output_halo_dim_size = output_tensor_shape[dim];
    uint32_t outer_dim_size = 1;
    for (int d = 0; d < dim; d++) {
        outer_dim_size *= input_tensor_shape[d];
    }

    bool is_first_device = true;
    bool is_last_device = true;
    uint32_t forward_device_offset = 0;
    uint32_t backward_device_offset = 0;

    if (secondary_cluster_axis.has_value()) {
        // secondary_cluster_axis==1, devices on row
        // secondary_mesh_shape(0) == number of rows, (1) is number of cols
        uint32_t secondary_cluster_axis_val = secondary_cluster_axis.value_or((uint32_t)0);
        uint32_t row_index = ring_index / secondary_mesh_shape.value().at(1);
        uint32_t col_index = ring_index % secondary_mesh_shape.value().at(1);
        if (secondary_cluster_axis_val) {
            // row
            if (col_index != 0) {
                is_first_device = false;
                backward_device_offset = 1;
            }
            if (col_index != secondary_mesh_shape.value().at(1) - 1) {
                is_last_device = false;
                forward_device_offset = 1;
            }
        } else {
            // column
            if (row_index != 0) {
                is_first_device = false;
                backward_device_offset = secondary_mesh_shape.value().at(1);
            }
            if (row_index != (secondary_mesh_shape.value().at(0) - 1)) {
                is_last_device = false;
                forward_device_offset = secondary_mesh_shape.value().at(1);
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

    bool is_padding_zeros = padding_mode == "zeros";

    // Get worker cores
    CoreCoord core_grid(num_links * 2, 1);
    auto [num_cores, worker_core_ranges, core_group_1, core_group_2, dims_per_core_group_1, dims_per_core_group_2] =
        (dim > 0) ? tt::tt_metal::split_work_to_cores(core_grid, outer_dim_size * 2)
                  : tt::tt_metal::split_work_to_cores(core_grid, num_sticks_per_halo_dim * 2);

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
    uint32_t link_offset_start_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t link_dims_to_read = 0;

        // direction 0 means pad left (top), 1 means pad right (bottom)
        for (uint32_t direction = 0; direction < num_directions; direction++) {
            CoreCoord core = {link * num_directions + direction, 0};
            CoreCoord opposite_core = {link * num_directions + (1 - direction), 0};
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
            CoreCoord virtual_opposite_core = mesh_device->worker_core_from_logical_core(opposite_core);
            if (core_group_1.contains(core)) {
                link_dims_to_read = dims_per_core_group_1;
            } else {
                link_dims_to_read = dims_per_core_group_2;
            }

            // Reader
            auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
            // When direction == 0, first_device is leftmost, when direction == 1, first_device is rightmost
            reader_kernel_config.compile_args = {
                direction ? is_last_device : is_first_device,
                direction ? is_first_device : is_last_device,
                sender_cb_index,  // cb_forward_id
                direction,
                is_padding_zeros,
                page_size};
            TensorAccessorArgs(*input_buffer).append_to(reader_kernel_config.compile_args);
            auto worker_reader_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
                "minimal_default_reader.cpp",
                {core},
                reader_kernel_config);
            reader_kernel_ids.push_back(worker_reader_kernel_id);

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),                                             // input_tensor_address
                output_tensor.buffer()->address(),                                            // output_tensor_address
                (dim > 0) ? link_offset_start_id * input_halo_dim_size : outer_dim_size - 1,  // link_offset_start_id
                (dim == 0) ? link_offset_start_id : 0,
                input_halo_dim_size,                                       // input_halo_dim_size
                (dim > 0) ? link_dims_to_read : outer_dim_size,            // outer_dim_size
                direction ? padding_right : padding_left,                  // padding
                (dim == 0) ? link_dims_to_read : num_sticks_per_halo_dim,  // num_sticks_to_read
                num_sticks_per_halo_dim,                                   // num_sticks_per_halo_dim
                final_semaphore.address()                                  // out_ready_sem_bank_addr (absolute address)
            };
            tt::tt_metal::SetRuntimeArgs(program, worker_reader_kernel_id, {core}, reader_rt_args);

            // Writer
            auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
            writer_kernel_config.compile_args = {
                direction ? is_last_device : is_first_device,
                direction ? is_first_device : is_last_device,
                sender_cb_index,  // cb_forward_id
                direction,
                is_padding_zeros,
                page_size};
            TensorAccessorArgs(*output_buffer).append_to(writer_kernel_config.compile_args);
            auto worker_writer_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/operations/experimental/ccl/neighbor_pad_async/device/kernels/"
                "minimal_default_writer.cpp",
                {core},
                writer_kernel_config);
            writer_kernel_ids.push_back(worker_writer_kernel_id);

            std::vector<uint32_t> writer_rt_args = {
                input_tensor.buffer()->address(),                                              // input_tensor_address
                output_tensor.buffer()->address(),                                             // output_tensor_address
                (dim > 0) ? link_offset_start_id * output_halo_dim_size : outer_dim_size - 1,  // link_offset_start_id
                (dim == 0) ? link_offset_start_id : 0,
                input_halo_dim_size,                                       // input_halo_dim_size
                output_halo_dim_size,                                      // output_halo_dim_size
                (dim > 0) ? link_dims_to_read : outer_dim_size,            // outer_dim_size
                direction ? padding_right : padding_left,                  // padding
                padding_left,                                              // padding left
                (dim == 0) ? link_dims_to_read : num_sticks_per_halo_dim,  // num_sticks_to_read
                num_sticks_per_halo_dim,                                   // num_sticks_per_halo_dim
                virtual_core.x,                                            // out_ready_sem_noc0_x
                virtual_core.y,                                            // out_ready_sem_noc0_y
                final_semaphore.address(),                                 // out_ready_sem_bank_addr (absolute address)
                true,                                                      // use_barrier_semaphore
                virtual_opposite_core.x,                                   // barrier_sem_noc0_x
                virtual_opposite_core.y,                                   // barrier_sem_noc0_y
                barrier_semaphore.address(),
                direction ? backward_device_offset : forward_device_offset,
                direction ? backward_device_offset : forward_device_offset};
            if (direction) {
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
            } else {
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
            }
            tt::tt_metal::SetRuntimeArgs(program, worker_writer_kernel_id, {core}, writer_rt_args);
        }
        if (dim > 0) {
            link_offset_start_id += (link_dims_to_read * num_sticks_per_halo_dim);
        } else {
            link_offset_start_id += link_dims_to_read;
        }
    }

    auto override_runtime_arguments_callback =
        [num_links, num_directions, reader_kernel_ids, writer_kernel_ids](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            // update readers/writers
            uint32_t core_idx = 0;
            for (uint32_t link = 0; link < num_links; link++) {
                // direction 0 means pad left (top), 1 means pad right (bottom)
                for (uint32_t direction = 0; direction < num_directions; direction++) {
                    CoreCoord core = {link * num_directions + direction, 0};
                    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

                    auto out_ready_semaphore = static_cast<const ttnn::NeighborPadAsync*>(operation)->final_semaphore;
                    auto barrier_semaphore = static_cast<const ttnn::NeighborPadAsync*>(operation)->barrier_semaphore;
                    // reader
                    auto& worker_reader_runtime_args = reader_runtime_args[core.x][core.y];
                    worker_reader_runtime_args[0] = input.buffer()->address();
                    worker_reader_runtime_args[1] = output.buffer()->address();
                    worker_reader_runtime_args[9] = out_ready_semaphore.address();
                    // writer
                    auto& worker_writer_runtime_args = writer_runtime_args[core.x][core.y];
                    worker_writer_runtime_args[0] = input.buffer()->address();
                    worker_writer_runtime_args[1] = output.buffer()->address();
                    worker_writer_runtime_args[13] = out_ready_semaphore.address();
                    worker_writer_runtime_args[17] = barrier_semaphore.address();

                    core_idx++;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
