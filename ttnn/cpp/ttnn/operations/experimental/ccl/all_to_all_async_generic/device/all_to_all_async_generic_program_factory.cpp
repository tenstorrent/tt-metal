// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_program_factory.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
ttnn::Shape get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttsl::SmallVector<uint32_t> tiled_shape;
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
    return ttnn::Shape(tiled_shape);
}

// Builds the ProgramDescriptor for one coord.  device_index, forward/backward
// neighbours, and per-link work distribution vary with the coord; the rest
// mirrors the legacy create_at body verbatim.
ProgramDescriptor build_program_descriptor(
    const AllToAllAsyncGenericParams& operation_attributes,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const GlobalSemaphore& init_barrier_semaphore,
    const GlobalSemaphore& final_barrier_semaphore) {
    log_debug(tt::LogOp, "DEBUG: create_at is called");

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor,
        mesh_coordinate,
        -1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);

    TT_FATAL(device_index < operation_attributes.num_devices, "DEBUG: device_index: {}", device_index);

    MeshDevice* device = tensor_args.input_tensor.device();

    std::vector<Tensor> input_tensors = {tensor_args.input_tensor};
    std::vector<Tensor> output_tensors = {tensor_return_value};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, operation_attributes.topology);

    const bool is_ring = operation_attributes.topology == ttnn::ccl::Topology::Ring;
    const size_t num_senders_per_link = (is_ring && operation_attributes.num_devices % 2 == 0) ? 2 : 1;
    const auto* topology_type = is_ring ? "RING" : "LINEAR";

    const auto [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links, num_senders_per_link, device, operation_attributes.sub_device_id);

    ProgramDescriptor desc;

    // Create CB
    const uint32_t page_size = op_config.get_page_size();
    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    const uint32_t number_pages_per_packet = 2;
    const uint32_t cb_size = (packet_size / page_size) * page_size * number_pages_per_packet;  // round_down
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_size,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CB::c_in0),
            .data_format = data_format,
            .page_size = number_pages_per_packet * page_size}},
    });

    // Create CB for fabric
    const auto reserved_packet_header_CB_index = tt::CB::c_in4;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t num_packet_headers_storable = 4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes}},
    });

    const auto input_shape = get_tiled_shape(tensor_args.input_tensor);
    uint32_t src_in_dims = 1;

    for (uint32_t i = operation_attributes.out_dim + 1; i < input_shape.size(); ++i) {
        src_in_dims *= input_shape[i];
    }
    const auto output_shape = get_tiled_shape(tensor_return_value);
    uint32_t dst_out_dims = 1;
    uint32_t dst_in_dims = 1;
    for (uint32_t i = 0; i < operation_attributes.in_dim; ++i) {
        dst_out_dims *= output_shape[i];
    }

    const uint32_t reader_has_extra_half_tile =
        operation_attributes.out_dim == input_shape.size() - 2 &&
        tensor_return_value.logical_shape()[operation_attributes.out_dim] % 32 == 16;
    const uint32_t writer_has_extra_half_tile =
        operation_attributes.in_dim == input_shape.size() - 2 &&
        tensor_args.input_tensor.logical_shape()[operation_attributes.in_dim] % 32 == 16;
    for (uint32_t i = operation_attributes.in_dim + 1; i < output_shape.size(); ++i) {
        dst_in_dims *= output_shape[i];
    }

    const uint32_t concat_num_half_tiles =
        output_shape[operation_attributes.in_dim] * 2 / operation_attributes.num_devices;
    const uint32_t concat_num_tiles = (concat_num_half_tiles + 1) / 2;
    const uint32_t num_blocks = dst_out_dims * dst_in_dims * concat_num_tiles;

    const uint32_t num_blocks_devices = num_senders_per_link;
    const uint32_t num_cores_per_blocks = operation_attributes.num_links;
    const uint32_t blocks_per_core = num_blocks / num_cores_per_blocks;

    std::vector<uint32_t> sender_reader_compile_args = {
        tt::CB::c_in0,                              // cb0_id
        page_size,                                  // tensor0_page_size
        device_index,                               // device_index
        operation_attributes.num_devices,           // num_devices
        input_shape[operation_attributes.out_dim],  // split_dim_size
        src_in_dims,                                // inner_dims_size
        input_shape[input_shape.size() - 1],        // last_dim_sizes
        reader_has_extra_half_tile,                 // has_reader_tail
        writer_has_extra_half_tile,                 // has_writer_tail
        concat_num_tiles,                           // concat_num_tiles
        dst_in_dims                                 // dst_inner_dims_size
    };

    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer()).append_to(sender_reader_compile_args);

    KernelDescriptor sender_reader_kernel_desc;
    sender_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
        "all_to_all_sender_reader.cpp";
    sender_reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_reader_kernel_desc.core_ranges = sender_worker_core_range;
    sender_reader_kernel_desc.compile_time_args = std::move(sender_reader_compile_args);
    sender_reader_kernel_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(sender_reader_kernel_desc));
    const KernelHandle sender_reader_kernel_id = desc.kernels.size() - 1;

    std::vector<int32_t> device_offsets[2];
    std::vector<std::vector<int32_t>> block_starts[2], block_ends[2];
    for (int i = 0; i < 2; ++i) {
        block_starts[i].resize(operation_attributes.num_links);
        block_ends[i].resize(operation_attributes.num_links);
    }
    // splitting device blocks for Ring topology, starting from the farthest device to ensure better load balance
    const uint32_t num_splitted_devices = 1;
    if (is_ring) {
        for (int d = operation_attributes.num_devices - 1 + num_splitted_devices; d >= 0; --d) {
            int distance = (d + 1) / 2;
            int device_offset = (d % 2 == 0) ? distance : -distance;
            if (num_senders_per_link == 1) {
                device_offsets[0].push_back(device_offset);
            } else {
                device_offsets[d % 2].push_back(device_offset);
            }
        }
    } else {
        // Linear topology
        for (uint32_t i = 0; i < operation_attributes.num_devices; ++i) {
            device_offsets[0].push_back(i - device_index);
        }
    }
    uint32_t semaphore_sent = 0;
    for (int l = 0; l < operation_attributes.num_links; ++l) {
        uint32_t current_start_block = l * blocks_per_core;
        uint32_t current_end_block = (l + 1) * blocks_per_core;
        if (l == operation_attributes.num_links - 1) {
            current_end_block = num_blocks;
        }
        for (int c = 0; c < num_senders_per_link; ++c) {
            for (int d = 0; d < device_offsets[c].size(); ++d) {
                semaphore_sent++;
                block_starts[c][l].push_back(current_start_block);
                block_ends[c][l].push_back(current_end_block);
            }
        }
        if (is_ring) {
            for (int i = 0; i < num_splitted_devices; ++i) {
                uint32_t split = (block_ends[0][l][i] + block_starts[0][l][i]) / 2;
                block_ends[0][l][i] = split;
                block_starts[1][l][num_splitted_devices - 1 - i] = split;
            }
        }
    }

    std::vector<uint32_t> sender_writer_compile_args = {
        tt::CB::c_in0,                              // cb0_id
        device_index,                               // device_index
        operation_attributes.num_devices,           // num_devices
        output_shape[operation_attributes.in_dim],  // concat_dim_size
        dst_in_dims,                                // inner_dims_size
        writer_has_extra_half_tile,                 // has_writer_tail
        page_size,                                  // intermediate_page_size
        reserved_packet_header_CB_index,            // reserved_packet_header_cb_id
        semaphore_sent,                             // semaphore_expected_value
        concat_num_tiles,                           // concat_num_tiles
        (concat_num_half_tiles * device_index) / 2  // full_block_offset
    };

    tt::tt_metal::TensorAccessorArgs(tensor_return_value.buffer()).append_to(sender_writer_compile_args);

    KernelDescriptor sender_writer_kernel_desc;
    sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
        "all_to_all_sender_writer.cpp";
    sender_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_writer_kernel_desc.core_ranges = sender_worker_core_range;
    sender_writer_kernel_desc.compile_time_args = std::move(sender_writer_compile_args);
    sender_writer_kernel_desc.defines = {{"TOPOLOGY", topology_type}};
    sender_writer_kernel_desc.config = WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(sender_writer_kernel_desc));
    const KernelHandle sender_writer_kernel_id = desc.kernels.size() - 1;

    CoreRange sender_box = sender_worker_core_range.bounding_box();
    // Swap start and end coord
    const uint32_t mcast_dest_noc_start_x = device->worker_core_from_logical_core(sender_box.end_coord).x;
    const uint32_t mcast_dest_noc_end_x = device->worker_core_from_logical_core(sender_box.start_coord).x;
    const uint32_t mcast_dest_noc_start_y = device->worker_core_from_logical_core(sender_box.end_coord).y;
    const uint32_t mcast_dest_noc_end_y = device->worker_core_from_logical_core(sender_box.start_coord).y;
    const uint32_t mcast_size = sender_box.size();

    auto drain_sync_core = device->worker_core_from_logical_core(sender_worker_cores[0]);

    for (uint32_t core_id = 0; core_id < sender_worker_cores.size(); ++core_id) {
        const auto& core = sender_worker_cores[core_id];
        // Reader: input_tensor is a tensor buffer → binding.
        KernelDescriptor::RTArgList sender_reader_rt_args;
        sender_reader_rt_args.push_back(tensor_args.input_tensor.buffer());  // binding
        sender_reader_rt_args.push_back(static_cast<uint32_t>(device_offsets[core_id % num_blocks_devices].size()));
        for (uint32_t i = 0; i < device_offsets[core_id % num_blocks_devices].size(); ++i) {
            sender_reader_rt_args.push_back(device_offsets[core_id % num_blocks_devices][i]);
            sender_reader_rt_args.push_back(
                block_starts[core_id % num_blocks_devices][core_id / num_blocks_devices][i]);
            sender_reader_rt_args.push_back(block_ends[core_id % num_blocks_devices][core_id / num_blocks_devices][i]);
        }
        desc.kernels[sender_reader_kernel_id].emplace_runtime_args(core, sender_reader_rt_args);

        // Writer: build the raw vector (incl. fabric extras) first, then graft
        // tensor_return_value buffer as a binding at position [0] in an RTArgList.
        bool with_forward =
            (num_senders_per_link == 1 || (core_id % num_blocks_devices == 0)) && forward_coord.has_value();
        bool with_backward =
            (num_senders_per_link == 1 || (core_id % num_blocks_devices == 1)) && backward_coord.has_value();
        std::vector<uint32_t> sender_writer_tail;
        // Block descriptors for this core
        for (uint32_t i = 0; i < device_offsets[core_id % num_blocks_devices].size(); ++i) {
            sender_writer_tail.push_back(device_offsets[core_id % num_blocks_devices][i]);
            sender_writer_tail.push_back(block_starts[core_id % num_blocks_devices][core_id / num_blocks_devices][i]);
            sender_writer_tail.push_back(block_ends[core_id % num_blocks_devices][core_id / num_blocks_devices][i]);
        }
        sender_writer_tail.push_back(with_forward);
        if (with_forward) {
            const auto sender_device_fabric_node_id = device->get_fabric_node_id(mesh_coordinate);
            const auto forward_device_fabric_node_id = device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                sender_device_fabric_node_id,
                forward_device_fabric_node_id,
                core_id / num_senders_per_link,
                desc,
                core,
                sender_writer_tail);
        }
        sender_writer_tail.push_back(with_backward);
        if (with_backward) {
            const auto sender_device_fabric_node_id = device->get_fabric_node_id(mesh_coordinate);
            const auto backward_device_fabric_node_id = device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                sender_device_fabric_node_id,
                backward_device_fabric_node_id,
                core_id / num_senders_per_link,
                desc,
                core,
                sender_writer_tail);
        }

        KernelDescriptor::RTArgList sender_writer_rt_args;
        sender_writer_rt_args.push_back(tensor_return_value.buffer());  // binding
        sender_writer_rt_args.push_back(init_barrier_semaphore.address());
        sender_writer_rt_args.push_back(final_barrier_semaphore.address());
        sender_writer_rt_args.push_back(static_cast<uint32_t>(core_id % num_blocks_devices));
        sender_writer_rt_args.push_back(static_cast<uint32_t>(core_id / num_blocks_devices));
        sender_writer_rt_args.push_back(mcast_dest_noc_start_x);
        sender_writer_rt_args.push_back(mcast_dest_noc_start_y);
        sender_writer_rt_args.push_back(mcast_dest_noc_end_x);
        sender_writer_rt_args.push_back(mcast_dest_noc_end_y);
        sender_writer_rt_args.push_back(mcast_size);
        sender_writer_rt_args.push_back(drain_sync_core.x);
        sender_writer_rt_args.push_back(drain_sync_core.y);
        sender_writer_rt_args.push_back(static_cast<uint32_t>(device_offsets[core_id % num_blocks_devices].size()));
        sender_writer_rt_args.append(sender_writer_tail);
        desc.kernels[sender_writer_kernel_id].emplace_runtime_args(core, sender_writer_rt_args);
    }

    return desc;
}

}  // namespace

WorkloadDescriptor AllToAllAsyncGenericProgram::create_workload_descriptor(
    const AllToAllAsyncGenericParams& operation_attributes,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sub_device_id = operation_attributes.sub_device_id;
    auto subdevice = sub_device_id.has_value() ? *sub_device_id : mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice);
    auto subdevices = {subdevice};

    // Allocate workload-scoped GlobalSemaphores once at cache miss and park
    // them in wd.semaphores so they outlive the cached MeshWorkload.
    WorkloadDescriptor wd;
    wd.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    wd.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    const auto& init_barrier_semaphore = wd.semaphores[0];
    const auto& final_barrier_semaphore = wd.semaphores[1];

    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());

    for (const auto& coord : coords) {
        ProgramDescriptor desc = build_program_descriptor(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            coord,
            init_barrier_semaphore,
            final_barrier_semaphore);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return wd;
}

}  // namespace ttnn::experimental::prim
