// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/slice_reshard_async/device/slice_reshard_async_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/math.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

// Builds the ProgramDescriptor for one coord.  ring_index, forward/backward
// fabric node ids, and per-direction outer-dim splits vary with the coord;
// the rest mirrors the legacy create_at body verbatim.
ProgramDescriptor build_program_descriptor(
    const SliceReshardAsyncParams& args,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::MeshCoordinate& mesh_coord) {
    auto* mesh_device = input_tensor.device();
    IDevice* sender_device = mesh_device ? mesh_device->get_device(mesh_coord) : input_tensor.device();
    std::vector<IDevice*> devices_to_use{};
    const auto& mesh_view = input_tensor.device()->get_view();
    devices_to_use = (args.cluster_axis == 0) ? mesh_view.get_devices_on_column(mesh_coord[1])
                                              : mesh_view.get_devices_on_row(mesh_coord[0]);
    const auto fabric_node_ids = (args.cluster_axis == 0) ? mesh_view.get_fabric_node_ids_on_column(mesh_coord[1])
                                                          : mesh_view.get_fabric_node_ids_on_row(mesh_coord[0]);
    uint32_t ring_size = devices_to_use.size();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    uint32_t ring_index = 0;
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

    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* output_buffer = output_tensor.buffer();

    uint32_t page_size = input_buffer->page_size();
    uint32_t num_sticks_per_outer_dim = input_tensor_shape[1] * input_tensor_shape[2];
    uint32_t input_outer_dim_size = input_tensor_shape[0];
    uint32_t output_outer_dim_size = output_tensor_shape[0];
    bool is_first_device = !backward_device.has_value();
    bool is_last_device = !forward_device.has_value();
    uint32_t global_output_outer_dim_start = args.output_dim_offset + (output_outer_dim_size * ring_index);
    uint32_t global_output_outer_dim_end = args.output_dim_offset + (output_outer_dim_size * (ring_index + 1)) - 1;
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

    CoreCoord core_grid(args.num_links * 2, 1);
    auto
        [num_cores,
         worker_core_ranges,
         core_group_1,
         core_group_2,
         num_sticks_per_core_group_1,
         num_sticks_per_core_group_2] = tt::tt_metal::split_work_to_cores(core_grid, num_sticks_per_outer_dim * 2);

    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_sticks_to_write_per_packet = 1;
    uint32_t cb_num_pages = 3 * num_sticks_to_write_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    ProgramDescriptor desc;

    uint32_t sender_cb_index = tt::CB::c_in0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = worker_core_ranges,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes}},
    });

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
            std::vector<uint32_t> reader_ct_args = {
                direction ? is_first_device : is_last_device,
                direction ? is_last_device : is_first_device,
                sender_cb_index,
                direction,
                page_size,
            };
            TensorAccessorArgs(*input_buffer).append_to(reader_ct_args);

            KernelDescriptor reader_kernel_desc;
            reader_kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                "minimal_default_reader.cpp";
            reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            reader_kernel_desc.core_ranges = CoreRangeSet(CoreRange(core, core));
            reader_kernel_desc.compile_time_args = std::move(reader_ct_args);
            reader_kernel_desc.config = ReaderConfigDescriptor{};
            desc.kernels.push_back(std::move(reader_kernel_desc));
            const KernelHandle reader_kernel_id = desc.kernels.size() - 1;

            KernelDescriptor::RTArgList reader_rt_args;
            reader_rt_args.push_back(input_tensor.buffer());  // Buffer* binding
            reader_rt_args.push_back(stick_start_id);
            reader_rt_args.push_back(num_sticks_to_read);
            reader_rt_args.push_back(input_outer_dim_size);
            reader_rt_args.push_back(direction ? outer_dims_to_forward : outer_dims_to_backward);
            reader_rt_args.push_back(outer_dims_from_forward);
            reader_rt_args.push_back(outer_dims_to_keep_start);
            reader_rt_args.push_back(outer_dims_to_keep_end);
            reader_rt_args.push_back(num_sticks_per_outer_dim);
            reader_rt_args.push_back(args.final_semaphore.address());  // workload-scoped semaphore
            desc.kernels[reader_kernel_id].emplace_runtime_args(core, reader_rt_args);

            // Writer
            std::vector<uint32_t> writer_ct_args = {
                direction ? is_first_device : is_last_device,
                direction ? is_last_device : is_first_device,
                sender_cb_index,
                direction,
            };
            TensorAccessorArgs(*output_buffer).append_to(writer_ct_args);

            KernelDescriptor writer_kernel_desc;
            writer_kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async/device/kernels/"
                "minimal_default_writer.cpp";
            writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            writer_kernel_desc.core_ranges = CoreRangeSet(CoreRange(core, core));
            writer_kernel_desc.compile_time_args = std::move(writer_ct_args);
            writer_kernel_desc.config = WriterConfigDescriptor{};
            desc.kernels.push_back(std::move(writer_kernel_desc));
            const KernelHandle writer_kernel_id = desc.kernels.size() - 1;

            // Build fabric extras (and trailing has_value flags) in a raw vector,
            // then assemble the final RTArgList so input/output tensors get
            // BufferBinding-patched on cache hit.
            std::vector<uint32_t> writer_tail;
            if (direction) {
                writer_tail.push_back(forward_fabric_node_id.has_value());
                if (forward_fabric_node_id.has_value()) {
                    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);
                    tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                        src_fabric_node_id, forward_fabric_node_id.value(), link, desc, core, writer_tail);
                }
                writer_tail.push_back(false);
            } else {
                writer_tail.push_back(false);
                writer_tail.push_back(backward_fabric_node_id.has_value());
                if (backward_fabric_node_id.has_value()) {
                    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);
                    tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                        src_fabric_node_id, backward_fabric_node_id.value(), link, desc, core, writer_tail);
                }
            }

            KernelDescriptor::RTArgList writer_rt_args;
            writer_rt_args.push_back(input_tensor.buffer());   // Buffer* binding
            writer_rt_args.push_back(output_tensor.buffer());  // Buffer* binding
            writer_rt_args.push_back(page_size);
            writer_rt_args.push_back(stick_start_id);
            writer_rt_args.push_back(num_sticks_to_read);
            writer_rt_args.push_back(output_outer_dim_size);
            writer_rt_args.push_back(direction ? outer_dims_to_forward : outer_dims_to_backward);
            writer_rt_args.push_back(outer_dims_to_keep_start);
            writer_rt_args.push_back(outer_dims_to_keep_end);
            writer_rt_args.push_back(direction ? outer_dims_from_backward : outer_dims_from_forward);
            writer_rt_args.push_back(outer_dims_from_forward);
            writer_rt_args.push_back(num_sticks_per_outer_dim);
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_core.x));
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_core.y));
            writer_rt_args.push_back(args.final_semaphore.address());  // workload-scoped semaphore
            writer_rt_args.push_back(1u);                              // true (matches prior literal)
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_opposite_core.x));
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_opposite_core.y));
            writer_rt_args.push_back(args.barrier_semaphore.address());
            writer_rt_args.append(writer_tail);
            desc.kernels[writer_kernel_id].emplace_runtime_args(core, writer_rt_args);
        }
        stick_start_id += num_sticks_to_read;
    }

    return desc;
}

}  // namespace

WorkloadDescriptor SliceReshardAsyncProgramFactory::create_workload_descriptor(
    const SliceReshardAsyncParams& args,
    const Tensor& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());

    for (const auto& coord : coords) {
        ProgramDescriptor desc = build_program_descriptor(args, tensor_args, tensor_return_value, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return wd;
}

}  // namespace ttnn::experimental::prim
