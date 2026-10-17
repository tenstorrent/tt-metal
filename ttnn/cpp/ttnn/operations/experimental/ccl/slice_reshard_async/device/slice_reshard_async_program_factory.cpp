// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include <tt-metalium/program_descriptors.hpp>
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

namespace ttnn::experimental::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

tt::tt_metal::ProgramDescriptor build_slice_reshard_program_descriptor(
    const SliceReshardAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const Tensor& input_tensor,
    const Tensor& output_tensor) {
    using namespace tt::tt_metal;
    namespace dyn = slice_reshard_async_dynamic;

    ProgramDescriptor desc;

    auto* mesh_device = input_tensor.device();
    IDevice* sender_device = mesh_device ? mesh_device->get_device(mesh_coord) : input_tensor.device();
    std::vector<IDevice*> devices_to_use = {};
    const auto& mesh_view = input_tensor.device()->get_view();
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    devices_to_use = (args.cluster_axis == 0) ? mesh_view.get_devices_on_column(mesh_coord[1])
                                              : mesh_view.get_devices_on_row(mesh_coord[0]);
    const auto fabric_node_ids = (args.cluster_axis == 0) ? mesh_view.get_fabric_node_ids_on_column(mesh_coord[1])
                                                          : mesh_view.get_fabric_node_ids_on_row(mesh_coord[0]);
    uint32_t ring_size = devices_to_use.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    uint32_t ring_index = 0;  // Initialize ring (device) index
    for (uint32_t i = 0; i < ring_size; ++i) {
        if (devices_to_use.at(i) == sender_device) {
            ring_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
                backward_fabric_node_id = fabric_node_ids.at(i - 1);
            }
            if (i != ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
                forward_fabric_node_id = fabric_node_ids.at(i + 1);
            }
        }
    }

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    Buffer* input_buffer = input_tensor.buffer();
    Buffer* output_buffer = output_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "slice_reshard_async: input tensor buffer is null");
    TT_FATAL(output_buffer != nullptr, "slice_reshard_async: output tensor buffer is null");

    // Get OP Config, topology config
    const uint32_t page_size = static_cast<uint32_t>(input_buffer->page_size());
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
    CoreCoord core_grid(args.num_links * dyn::kNumDirections, 1);
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
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = worker_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });

    const auto final_semaphore_address = static_cast<uint32_t>(args.final_semaphore.address());
    const auto barrier_semaphore_address = static_cast<uint32_t>(args.barrier_semaphore.address());

    // KERNEL CREATION
    uint32_t stick_start_id = 0;
    for (uint32_t link = 0; link < args.num_links; link++) {
        uint32_t num_sticks_to_read = 0;
        for (uint32_t direction = 0; direction < dyn::kNumDirections; direction++) {
            const uint32_t core_idx = (link * dyn::kNumDirections) + direction;
            CoreCoord core = {core_idx, 0};
            CoreCoord opposite_core = {(link * dyn::kNumDirections) + (1 - direction), 0};
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
            CoreCoord virtual_opposite_core = mesh_device->worker_core_from_logical_core(opposite_core);
            if (core_group_1.contains(core)) {
                num_sticks_to_read = num_sticks_per_core_group_1;
            } else {
                num_sticks_to_read = num_sticks_per_core_group_2;
            }

            // Reader
            KernelDescriptor reader_desc;
            reader_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                "minimal_default_reader.cpp";
            reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            reader_desc.core_ranges = CoreRangeSet(CoreRange(core));
            reader_desc.compile_time_args = {
                direction ? is_first_device : is_last_device,
                direction ? is_last_device : is_first_device,
                sender_cb_index,  // cb_forward_id
                direction,
                page_size,
            };
            TensorAccessorArgs(*input_buffer).append_to(reader_desc.compile_time_args);
            reader_desc.named_compile_time_args = {{"cb_forward_id", sender_cb_index}};
            reader_desc.config = ReaderConfigDescriptor{};

            KernelDescriptor::RTArgList reader_rt_args;
            reader_rt_args.push_back(input_buffer);          // input_tensor_address
            reader_rt_args.push_back(stick_start_id);        // stick_start_id
            reader_rt_args.push_back(num_sticks_to_read);    // num_sticks_to_read
            reader_rt_args.push_back(input_outer_dim_size);  // input_outer_dim_size
            reader_rt_args.push_back(
                direction ? outer_dims_to_forward : outer_dims_to_backward);  // outer_dims_to_forward
            reader_rt_args.push_back(outer_dims_from_forward);                // outer_dims_from_forward
            reader_rt_args.push_back(outer_dims_to_keep_start);               // outer_dims_to_keep
            reader_rt_args.push_back(outer_dims_to_keep_end);                 // outer_dims_to_keep
            reader_rt_args.push_back(num_sticks_per_outer_dim);               // num_sticks_per_outer_dim
            // out_ready_sem_bank_addr (absolute address)
            reader_rt_args.push_back(final_semaphore_address);  // smuggled-rta-ok: caller-supplied GlobalSemaphore,
                                                                // re-applied via override_runtime_arguments
            reader_desc.emplace_runtime_args(core, reader_rt_args);

            TT_FATAL(
                desc.kernels.size() == dyn::reader_kernel_idx(core_idx),
                "slice_reshard_async: reader kernel for core {} pushed at index {}, expected {}",
                core_idx,
                desc.kernels.size(),
                dyn::reader_kernel_idx(core_idx));
            desc.kernels.push_back(std::move(reader_desc));

            // Writer
            KernelDescriptor writer_desc;
            writer_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                "minimal_default_writer.cpp";
            writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            writer_desc.core_ranges = CoreRangeSet(CoreRange(core));
            writer_desc.compile_time_args = {
                direction ? is_first_device : is_last_device,
                direction ? is_last_device : is_first_device,
                sender_cb_index,  // cb_forward_id
                direction,
            };
            TensorAccessorArgs(*output_buffer).append_to(writer_desc.compile_time_args);
            writer_desc.named_compile_time_args = {{"cb_forward_id", sender_cb_index}};
            writer_desc.config = WriterConfigDescriptor{};

            KernelDescriptor::RTArgList writer_rt_args;
            writer_rt_args.push_back(input_buffer);           // input_tensor_address
            writer_rt_args.push_back(output_buffer);          // output_tensor_address
            writer_rt_args.push_back(page_size);              // stick_size
            writer_rt_args.push_back(stick_start_id);         // stick_start_id
            writer_rt_args.push_back(num_sticks_to_read);     // num_sticks_to_read
            writer_rt_args.push_back(output_outer_dim_size);  // output_outer_dim_size
            writer_rt_args.push_back(
                direction ? outer_dims_to_forward : outer_dims_to_backward);  // outer_dims_to_forward
            writer_rt_args.push_back(outer_dims_to_keep_start);               // outer_dims_to_keep
            writer_rt_args.push_back(outer_dims_to_keep_end);                 // outer_dims_to_keep
            writer_rt_args.push_back(
                direction ? outer_dims_from_backward : outer_dims_from_forward);  // outer_dims_to_receive
            writer_rt_args.push_back(outer_dims_from_forward);                    // outer_dims_from_forward
            writer_rt_args.push_back(num_sticks_per_outer_dim);                   // num_sticks_per_outer_dim
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_core.x));      // out_ready_sem_noc0_x
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_core.y));      // out_ready_sem_noc0_y
            // out_ready_sem_bank_addr (absolute address)
            writer_rt_args.push_back(final_semaphore_address);  // smuggled-rta-ok: caller-supplied GlobalSemaphore,
                                                                // re-applied via override_runtime_arguments
            writer_rt_args.push_back(1u);                       // use_barrier_semaphore
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_opposite_core.x));  // barrier_sem_noc0_x
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_opposite_core.y));  // barrier_sem_noc0_y
            writer_rt_args.push_back(barrier_semaphore_address);  // smuggled-rta-ok: caller-supplied GlobalSemaphore,
                                                                  // re-applied via override_runtime_arguments

            std::vector<uint32_t> fabric_rt_args;
            if (direction) {
                fabric_rt_args.push_back(forward_fabric_node_id.has_value());
                if (forward_fabric_node_id.has_value()) {
                    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        src_fabric_node_id, forward_fabric_node_id.value(), link, desc, core, fabric_rt_args);
                }
                fabric_rt_args.push_back(false);
            } else {
                fabric_rt_args.push_back(false);
                fabric_rt_args.push_back(backward_fabric_node_id.has_value());
                if (backward_fabric_node_id.has_value()) {
                    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        src_fabric_node_id, backward_fabric_node_id.value(), link, desc, core, fabric_rt_args);
                }
            }
            writer_rt_args.append(fabric_rt_args);
            writer_desc.emplace_runtime_args(core, writer_rt_args);

            TT_FATAL(
                desc.kernels.size() == dyn::writer_kernel_idx(core_idx),
                "slice_reshard_async: writer kernel for core {} pushed at index {}, expected {}",
                core_idx,
                desc.kernels.size(),
                dyn::writer_kernel_idx(core_idx));
            desc.kernels.push_back(std::move(writer_desc));
        }
        stick_start_id += num_sticks_to_read;
    }

    return desc;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::WorkloadDescriptor SliceReshardAsyncProgramFactory::create_workload_descriptor(
    const SliceReshardAsyncParams& args,
    const Tensor& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    const auto coords = tensor_coords.coords();
    workload_descriptor.programs.reserve(coords.size());
    // Ring index and forward/backward neighbors depend on the coordinate, so each coord gets its own program.
    for (const auto& coord : coords) {
        auto desc = CMAKE_UNIQUE_NAMESPACE::build_slice_reshard_program_descriptor(
            args, coord, tensor_args, tensor_return_value);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

void SliceReshardAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SliceReshardAsyncParams& args,
    const Tensor& /*tensor_args*/,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    namespace dyn = slice_reshard_async_dynamic;

    // final_semaphore / barrier_semaphore are excluded from
    // SliceReshardAsyncDeviceOperation::compute_program_hash, so they are re-applied here on every hit.
    const auto final_semaphore_address = static_cast<uint32_t>(args.final_semaphore.address());
    const auto barrier_semaphore_address = static_cast<uint32_t>(args.barrier_semaphore.address());

    const uint32_t num_cores = args.num_links * dyn::kNumDirections;
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        const tt::tt_metal::CoreCoord core = {core_idx, 0};
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, dyn::reader_kernel_idx(core_idx), core);
        reader_runtime_args[dyn::kReaderOutReadySemArg] = final_semaphore_address;

        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, dyn::writer_kernel_idx(core_idx), core);
        writer_runtime_args[dyn::kWriterOutReadySemArg] = final_semaphore_address;
        writer_runtime_args[dyn::kWriterBarrierSemArg] = barrier_semaphore_address;
    }
}

}  // namespace ttnn::experimental::prim
