// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async_generic/device/all_to_all_async_generic_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tuple>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;
ttnn::Shape get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttnn::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); i++) {
        uint32_t dim = 0;
        if (i == shape.rank() - 1) {
            dim = shape[i] / tile_shape[1];
        } else if (i == shape.rank() - 2) {
            dim = shape[i] / tile_shape[0];
        } else {
            dim = shape[i];
        }
        tiled_shape.push_back(dim);
    }
    auto res = ttnn::Shape(tiled_shape);
    return res;
}

tt::tt_metal::operation::ProgramWithCallbacks all_to_all_async_generic_program(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    std::optional<MeshCoordinate> target_device,
    std::optional<MeshCoordinate> forward_coord,
    std::optional<MeshCoordinate> backward_coord,
    const uint32_t in_dim,
    const uint32_t out_dim,
    const uint32_t num_links,
    const uint32_t num_devices,
    const uint32_t device_index,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& final_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};
    MeshDevice* device = input_tensor.device();

    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);

    const size_t num_senders_per_link = 1;
    uint32_t num_receivers_per_link = 1;
    auto topology_type = topology == ttnn::ccl::Topology::Ring ? "RING" : "LINEAR";

    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_senders_per_link, device, sub_device_id);

    // Get OP Config, topology config
    const auto [total_workers_core_range, total_workers_cores] =
        choose_worker_cores(num_links, (num_senders_per_link + num_receivers_per_link), device, sub_device_id);

    // Create CB
    const uint32_t page_size = op_config.get_page_size();
    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    const uint32_t number_pages_per_packet = 2;
    const uint32_t cb_size = (packet_size / page_size) * page_size * number_pages_per_packet;  // round_down
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(cb_size, {{tt::CB::c_in0, data_format}})
                              .set_page_size(tt::CB::c_in0, page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Create CB for fabric
    const auto reserved_packet_header_CB_index = tt::CB::c_in4;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t num_packet_headers_storable = 4;
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    const auto input_shape = get_tiled_shape(input_tensor);
    uint32_t src_out_dims = 1;
    uint32_t src_in_dims = 1;
    for (uint32_t i = 0; i < out_dim; ++i) {
        src_out_dims *= input_shape[i];
    }

    for (uint32_t i = out_dim + 1; i < input_shape.size(); ++i) {
        src_in_dims *= input_shape[i];
    }
    const auto output_shape = get_tiled_shape(output_tensor);
    uint32_t dst_out_dims = 1;
    uint32_t dst_in_dims = 1;
    for (uint32_t i = 0; i < in_dim; ++i) {
        dst_out_dims *= output_shape[i];
    }

    const uint32_t reader_has_extra_half_tile =
        out_dim == input_shape.size() - 2 && output_tensor.logical_shape()[out_dim] % 32 == 16;
    const uint32_t writer_has_extra_half_tile =
        in_dim == input_shape.size() - 2 && input_tensor.logical_shape()[in_dim] % 32 == 16;
    for (uint32_t i = in_dim + 1; i < output_shape.size(); ++i) {
        dst_in_dims *= output_shape[i];
    }

    auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_kernel_config.defines.emplace("TOPOLOGY", topology_type);
    sender_reader_kernel_config.compile_args = {
        tt::CB::c_in0,                        // cb0_id
        page_size,                            // tensor0_page_size
        device_index,                         // device_index
        num_devices,                          // num_devices
        src_out_dims,                         // outer_dims_size
        input_shape[out_dim],                 // split_dim_size
        src_in_dims,                          // inner_dims_size
        input_shape[input_shape.size() - 1],  // last_dim_sizes
        number_pages_per_packet,              // number_pages_per_packet
        reader_has_extra_half_tile,           // has_reader_tail
        writer_has_extra_half_tile,           // has_writer_tail
    };

    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_kernel_config.compile_args);

    auto sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
        "all_to_all_sender_reader.cpp",
        sender_worker_core_range,
        sender_reader_kernel_config);

    auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_writer_kernel_config.defines.emplace("TOPOLOGY", topology_type);
    sender_writer_kernel_config.compile_args = {
        tt::CB::c_in0,               // cb0_id
        device_index,                // device_index
        num_devices,                 // num_devices
        dst_out_dims,                // outer_dims_size
        output_shape[in_dim],        // concat_dim_size
        dst_in_dims,                 // inner_dims_size
        number_pages_per_packet,     // number_pages_per_packet
        writer_has_extra_half_tile,  // has_writer_tail
        page_size,                   // intermediate_page_size
        reserved_packet_header_CB_index,
    };

    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_kernel_config.compile_args);

    auto sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
        "all_to_all_sender_writer.cpp",
        sender_worker_core_range,
        sender_writer_kernel_config);

    std::vector<uint32_t> sender_reader_rt_args = {input_tensor.buffer()->address()};
    tt::tt_metal::SetRuntimeArgs(program, sender_reader_kernel_id, sender_worker_core_range, sender_reader_rt_args);

    auto drain_sync_core = device->worker_core_from_logical_core({sender_worker_cores[0]});
    std::vector<uint32_t> sender_writer_rt_args = {
        output_tensor.buffer()->address(),
        init_semaphore.address(),
        final_semaphore.address(),
        drain_sync_core.x,
        drain_sync_core.y};
    sender_writer_rt_args.push_back(forward_coord.has_value());

    if (forward_coord.has_value()) {
        const auto sender_device_fabric_node_id = device->get_fabric_node_id(target_device.value());
        const auto forward_device_fabric_node_id = device->get_fabric_node_id(forward_coord.value());
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_device_fabric_node_id,
            forward_device_fabric_node_id,
            0,
            program,
            {sender_worker_cores[0]},
            sender_writer_rt_args);
    }

    sender_writer_rt_args.push_back(backward_coord.has_value());
    if (backward_coord.has_value()) {
        const auto sender_device_fabric_node_id = device->get_fabric_node_id(target_device.value());
        const auto backward_device_fabric_node_id = device->get_fabric_node_id(backward_coord.value());
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_device_fabric_node_id,
            backward_device_fabric_node_id,
            0,
            program,
            {sender_worker_cores[0]},
            sender_writer_rt_args);
    }
    tt::tt_metal::SetRuntimeArgs(program, sender_writer_kernel_id, sender_worker_core_range, sender_writer_rt_args);

    auto override_runtime_arguments_callback =
        [sender_reader_kernel_id, sender_writer_kernel_id, sender_worker_cores, init_semaphore, final_semaphore](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            auto& sender_reader_runtime_args = GetRuntimeArgs(program, sender_reader_kernel_id);
            auto& sender_writer_runtime_args = GetRuntimeArgs(program, sender_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                auto& worker_sender_reader_runtime_args = sender_reader_runtime_args[core.x][core.y];
                auto& worker_sender_writer_runtime_args = sender_writer_runtime_args[core.x][core.y];
                worker_sender_reader_runtime_args[0] = input.buffer()->address();
                worker_sender_writer_runtime_args[0] = output.buffer()->address();
                worker_sender_writer_runtime_args[1] = init_semaphore.address();
                worker_sender_writer_runtime_args[2] = final_semaphore.address();
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
