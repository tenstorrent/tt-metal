// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_op.hpp"
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
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
#include <cstdint>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks slice_reshard_async_minimal(
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t output_dim_offset,
    const uint32_t output_dim_shape,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const uint32_t num_links,
    ccl::Topology topology,
    const uint32_t ring_size,
    const uint32_t ring_index) {
    tt::tt_metal::Program program{};

    // Tensor Info
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* output_buffer = output_tensor.buffer();

    auto mesh_device = input_tensor.device();

    // Get OP Config, topology config
    uint32_t page_size = input_tensor.buffer()->page_size();
    uint32_t num_sticks_per_outer_dim = input_tensor_shape[1] * input_tensor_shape[2];
    uint32_t input_outer_dim_size = input_tensor_shape[0];
    uint32_t output_outer_dim_size = output_tensor_shape[0];
    bool is_first_device = !backward_device.has_value();
    bool is_last_device = !forward_device.has_value();
    uint32_t global_output_outer_dim_start = output_dim_offset + output_outer_dim_size * ring_index;
    uint32_t global_output_outer_dim_end = output_dim_offset + output_outer_dim_size * (ring_index + 1) - 1;
    uint32_t global_input_outer_dim_start = input_outer_dim_size * ring_index;
    uint32_t global_input_outer_dim_end = input_outer_dim_size * (ring_index + 1) - 1;

    int32_t backward_device_end = std::max((int32_t)global_input_outer_dim_start - 1, 0);
    uint32_t outer_dims_from_backward = std::max(backward_device_end - (int32_t)global_output_outer_dim_start + 1, 0);
    int32_t forward_device_start = std::min(global_input_outer_dim_end + 1, input_outer_dim_size * ring_size - 1);
    uint32_t outer_dims_from_forward = std::max((int32_t)global_output_outer_dim_end - forward_device_start + 1, 0);
    uint32_t outer_dims_to_keep_start =
        std::max((int32_t)global_output_outer_dim_start - (int32_t)global_input_outer_dim_start, 0);
    uint32_t outer_dims_to_keep_end = std::min(
        outer_dims_to_keep_start - outer_dims_from_backward + output_outer_dim_size - 1, global_input_outer_dim_end);
    int32_t backward_device_output_end =
        std::max((int32_t)global_output_outer_dim_start - 1, (int32_t)output_dim_offset - 1);
    uint32_t outer_dims_to_backward =
        is_first_device ? 0 : std::max(backward_device_output_end - (int32_t)global_input_outer_dim_start + 1, 0);
    int32_t forward_device_output_start =
        std::min(global_output_outer_dim_end + 1, output_outer_dim_size * ring_size - 1);
    uint32_t outer_dims_to_forward =
        is_last_device ? 0 : std::max((int32_t)global_input_outer_dim_end - forward_device_output_start + 1, 0);

    /****TODO BARRIER SEMAPHORE****/
    // auto [unicast_forward_args, unicast_backward_args] =
    //     ccl::get_forward_backward_line_unicast_configuration(topology, sender_device, forward_device,
    //     backward_device);
    // auto [barrier_mcast_forward_args, barrier_mcast_backward_args] =
    // ccl::get_forward_backward_line_mcast_configuration(
    //     topology,
    //     sender_device,
    //     forward_device,
    //     backward_device,
    //     topology == ccl::Topology::Linear ? num_targets_forward : ring_size - 1,
    //     topology == ccl::Topology::Linear ? num_targets_backward : ring_size - 1);
    /*******/

    // Get worker cores
    CoreCoord core_grid(num_links * 2, 1);
    auto
        [num_cores,
         worker_core_ranges,
         core_group_1,
         core_group_2,
         num_sticks_per_core_group_1,
         num_sticks_per_core_group_2] = tt::tt_metal::split_work_to_cores(core_grid, num_sticks_per_outer_dim * 2);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
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

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 2;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, worker_core_ranges, cb_reserved_packet_header_config);

    // KERNEL CREATION
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    uint32_t num_directions = 2;
    uint32_t stick_start_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_sticks_to_read = 0;
        for (uint32_t direction = 0; direction < num_directions; direction++) {
            CoreCoord core = {link * num_directions + direction, 0};
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
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
            };
            TensorAccessorArgs(*input_buffer).append_to(reader_kernel_config.compile_args);
            TensorAccessorArgs(*output_buffer).append_to(reader_kernel_config.compile_args);
            auto worker_reader_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                "minimal_default_reader.cpp",
                {core},
                reader_kernel_config);
            reader_kernel_ids.push_back(worker_reader_kernel_id);

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),                            // input_tensor_address
                output_tensor.buffer()->address(),                           // output_tensor_address
                page_size,                                                   // stick_size
                stick_start_id,                                              // stick_start_id
                num_sticks_to_read,                                          // num_sticks_to_read
                input_outer_dim_size,                                        // input_outer_dim_size
                direction ? outer_dims_to_forward : outer_dims_to_backward,  // outer_dims_to_forward
                outer_dims_to_keep_start,                                    // outer_dims_to_keep
                outer_dims_to_keep_end,                                      // outer_dims_to_keep
                num_sticks_per_outer_dim,                                    // num_sticks_per_outer_dim
                final_semaphore.address()  // out_ready_sem_bank_addr (absolute address)
            };
            tt::tt_metal::SetRuntimeArgs(program, worker_reader_kernel_id, {core}, reader_rt_args);

            // Writer
            auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
            writer_kernel_config.compile_args = {
                direction ? is_first_device : is_last_device,
                direction ? is_last_device : is_first_device,
                sender_cb_index,                  // cb_forward_id
                reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
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
                num_sticks_per_outer_dim,                                        // num_sticks_per_outer_dim
                virtual_core.x,                                                  // out_ready_sem_noc0_x
                virtual_core.y,                                                  // out_ready_sem_noc0_y
                final_semaphore.address()  // out_ready_sem_bank_addr (absolute address)
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

    auto override_runtime_arguments_callback =
        [num_links, num_directions, reader_kernel_ids, writer_kernel_ids](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            // auto barrier_semaphore = static_cast<const ttnn::AllGatherAsync*>(operation)->barrier_semaphore;
            // update readers/writers
            uint32_t core_idx = 0;
            for (uint32_t link = 0; link < num_links; link++) {
                for (uint32_t direction = 0; direction < num_directions; direction++) {
                    CoreCoord core = {link * num_directions + direction, 0};
                    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

                    auto out_ready_semaphore = static_cast<const ttnn::SliceReshardAsync*>(operation)->final_semaphore;
                    // reader
                    auto& worker_reader_runtime_args = reader_runtime_args[core.x][core.y];
                    worker_reader_runtime_args[0] = input.buffer()->address();
                    worker_reader_runtime_args[1] = output.buffer()->address();
                    worker_reader_runtime_args[10] = out_ready_semaphore.address();
                    // writer
                    auto& worker_writer_runtime_args = writer_runtime_args[core.x][core.y];
                    worker_writer_runtime_args[0] = input.buffer()->address();
                    worker_writer_runtime_args[1] = output.buffer()->address();
                    worker_writer_runtime_args[13] = out_ready_semaphore.address();

                    // if (barrier_semaphore.has_value()) {
                    // 	worker_writer_sender_runtime_args[16] = barrier_semaphore.value().address();
                    // }

                    core_idx++;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
