// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/deepseek_reduce_scatter_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/deepseek_reduce_scatter_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
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

using namespace tt::constants;
using namespace tt::tt_metal;

// Import types from the new TMP pattern
using ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail::DeepseekReduceScatterProgramArtifacts;

namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail {

DeepseekReduceScatterProgramArtifacts build_deepseek_reduce_scatter_program_artifacts(
    tt::tt_metal::Program& program,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& intermediate_tensor,
    const ttnn::Tensor& output_tensor,
    const ttnn::MeshCoordinate& sender_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    uint32_t ring_index,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multidevice_semaphores,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore,
    uint32_t num_links,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord core_grid_offset) {
    auto* mesh_device = input_tensor.device();

    const uint32_t ring_size = 8;
    const uint32_t num_directions_per_link = 2;
    const uint32_t num_cores_per_link = num_directions_per_link;

    const auto [all_core_range, all_cores] =
        ttnn::ccl::choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);

    std::vector<CoreRange> sender_worker_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& worker_core = all_cores[core_id++];
            sender_worker_core_ranges.emplace_back(worker_core);
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);

    // Tensor Info
    const auto& input_tensor_shape = input_tensor.padded_shape();
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);

    const uint32_t input_tensor_B = input_tensor_shape[-4];
    const uint32_t input_tensor_C = input_tensor_shape[-3];
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    const uint32_t slice_B = input_tensor_B;
    const uint32_t slice_C = input_tensor_C;
    const uint32_t slice_Ht = input_tensor_Ht;
    const uint32_t slice_Wt = input_tensor_Wt / ring_size;

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;
    const uint32_t input_batch_num_pages = input_tensor_num_pages / input_tensor_B;
    const uint32_t output_batch_num_pages = output_tensor_num_pages / slice_B;
    const uint32_t input_channel_num_pages = input_batch_num_pages / input_tensor_C;
    const uint32_t output_channel_num_pages = output_batch_num_pages / slice_C;

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

    // L1 Scratch CB Creation
    uint32_t page_size = input_tensor.buffer()->page_size();
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    uint32_t tile_granularity = num_tiles_to_write_per_packet < 4 ? 4 * num_tiles_to_write_per_packet : 8;
    uint32_t cb_num_pages = 3 * tile_granularity;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

    bool output_is_sharded = output_tensor.is_sharded();
    std::map<std::string, std::string> writer_compute_defines;
    if (output_is_sharded) {
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

    std::vector<uint32_t> sender_reader_compile_args = {
        ring_index,               // my_chip_id
        ring_size,                // ring_size
        input_cb_index,           // cb_input_id
        intermediate_cb_index,    // cb_intermediate_id
        reader_output_cb_index,   // cb_reader_output_id
        tile_granularity,         // tile_granularity
        page_size,                // page_size
        input_batch_num_pages,    // input_batch_num_pages
        input_channel_num_pages,  // input_channel_num_pages
        input_tensor_B,           // input_tensor_B
        input_tensor_Wt,          // input_tensor_Wt
        slice_C,                  // slice_C
        slice_Ht,                 // slice_Ht
        slice_Wt,                 // slice_Wt
    };

    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_reader_compile_args);

    std::string sender_reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/kernels/"
        "deepseek_reduce_scatter_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_reader_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args));

    // Writer
    std::vector<uint32_t> sender_writer_compile_args = {
        ring_index,                     // my_chip_id
        ring_size,                      // ring_size
        compute_output_cb_index,        // cb_compute_output_id
        reader_output_cb_index,         // cb_reader_output_id
        tile_granularity,               // packet_size_in_pages
        page_size,                      // page_size
        num_tiles_to_write_per_packet,  // num_tiles_to_write_per_packet
        output_batch_num_pages,         // output_batch_num_pages
        input_channel_num_pages,        // input_channel_num_pages
        output_channel_num_pages,       // output_channel_num_pages
        input_tensor_B,                 // input_tensor_B
        input_tensor_Wt,                //         input_tensor_Wt
        slice_C,                        // slice_C
        slice_Ht,                       // slice_Ht
        slice_Wt,                       // slice_Wt
    };

    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_writer_compile_args);
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }

    std::string sender_writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/kernels/"
        "deepseek_reduce_scatter_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        sender_writer_kernel_path,
        sender_worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args, writer_compute_defines));

    // Reduce kernel
    auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    sender_reduce_kernel_config.compile_args = {
        input_cb_index,           //         input_cb_id
        intermediate_cb_index,    // intermediate_cb
        compute_output_cb_index,  // output_cb
        tile_granularity,         // tile_granularity
        ring_size,                // ring_size
        input_tensor_B,           // input_tensor_B
        slice_C,                  // slice_C
    };

    std::string sender_reduce_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/kernels/deepseek_reduction.cpp";

    auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
        program, sender_reduce_kernel_path, sender_worker_core_range_set, sender_reduce_kernel_config);

    auto worker_core_iter = sender_worker_core_range_set.ranges().cbegin();
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            auto core = *((worker_core_iter++)->begin());
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

            uint32_t worker_id = link;
            uint32_t num_workers = num_links;

            uint32_t start_tiles_read = worker_id * output_channel_num_pages / num_workers;
            uint32_t start_tiles_to_read = (worker_id + 1) * output_channel_num_pages / num_workers;

            uint32_t start_pages_read_in_row = start_tiles_read % slice_Wt;
            uint32_t start_row_offset = start_tiles_read / slice_Wt * input_tensor_Wt;

            uint32_t chunks_per_sync_val = 1;
            log_trace(tt::LogOp, "DEBUG: chunks_per_sync_val: {}", chunks_per_sync_val);

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),          // input_tensor_address
                intermediate_tensor.buffer()->address(),   // intermediate_tensor_address
                multidevice_semaphores.at(dir).address(),  // out_ready_semaphore
                dir,                                       // direction
                chunks_per_sync_val,                       // chunks_per_sync
                start_tiles_read,                          // start_tiles_read
                start_tiles_to_read,                       // start_tiles_to_read
                start_pages_read_in_row,                   // start_pages_read_in_row
                start_row_offset,                          // start_row_offset (unused by dim0 kernel)
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

            // Writer RT args
            std::vector<uint32_t> writer_rt_args = {
                intermediate_tensor.buffer()->address(),                       // intermediate_tensor_address
                output_tensor.buffer()->address(),                             // output_tensor_address
                virtual_core.x,                                                // out_ready_sem_noc0_x
                virtual_core.y,                                                // out_ready_sem_noc0_y
                multidevice_semaphores.at(dir).address(),                      // out_ready_fwd_semaphore
                multidevice_semaphores.at(num_directions_per_link).address(),  // batch_ready_semaphore
                barrier_semaphore.address(),                                   // barrier_sem
                dir,                                                           // direction
                chunks_per_sync_val,                                           // chunks_per_sync
                start_pages_read_in_row,  // start_pages_read_in_row (unused by dim0 kernel)
                start_row_offset,         // start_row_offset (unused by dim0 kernel)
                start_tiles_read,         // start_tiles_read
                start_tiles_to_read,      // tiles_to_read
            };

            if (output_is_sharded) {
                shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
            }

            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_coord);
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            uint32_t num_connections = 1;
            dst_nodes.reserve(num_connections);
            if (dir == 0) {
                // backward
                const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                dst_nodes.push_back(backward_coord_fabric_node_id);
            } else {
                // forward
                const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                dst_nodes.push_back(forward_coord_fabric_node_id);
            }
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id, dst_nodes, {link}, program, writer_kernel_id, {core}, writer_rt_args);

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);

            std::vector<uint32_t> reduce_rt_args = {
                start_tiles_read,     // start_tiles_read
                start_tiles_to_read,  // start_tiles_to_read
                dir};                 // dir
            tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, reduce_rt_args);
        }
    }

    return {reader_kernel_id, writer_kernel_id, all_cores, num_directions_per_link};
}

void deepseek_reduce_scatter_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_directions_per_link,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multidevice_semaphores,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore,
    uint32_t num_links,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& intermediate_tensor,
    const ttnn::Tensor& output_tensor) {
    // update senders
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            CoreCoord core = all_cores[link * num_directions_per_link + dir];
            std::vector<std::vector<RuntimeArgsData>> reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
            std::vector<std::vector<RuntimeArgsData>> writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

            // sender reader
            auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input_tensor.buffer()->address();
            worker_reader_sender_runtime_args[1] = intermediate_tensor.buffer()->address();
            worker_reader_sender_runtime_args[2] = multidevice_semaphores.at(dir).address();
            // sender writer
            auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
            worker_writer_sender_runtime_args[0] = intermediate_tensor.buffer()->address();
            worker_writer_sender_runtime_args[1] = output_tensor.buffer()->address();
            worker_writer_sender_runtime_args[4] = multidevice_semaphores.at(dir).address();
            worker_writer_sender_runtime_args[5] = multidevice_semaphores.at(num_directions_per_link).address();
            worker_writer_sender_runtime_args[7] = barrier_semaphore.address();
        }
    }
}

// Mesh Workload Factory implementations
DeepseekReduceScatterMeshWorkloadFactory::cached_mesh_workload_t
DeepseekReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto sub_device_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto sub_device_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // 3 semaphores used for within op synchronizations
    std::vector<tt::tt_metal::GlobalSemaphore> multidevice_semaphores = {
        ttnn::global_semaphore::create_global_semaphore(mesh_device, sub_device_core_range_set, 0),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, sub_device_core_range_set, 0),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, sub_device_core_range_set, 0),
    };

    // 1 barrier semaphore used to ensure that all the buffers are allocated
    tt::tt_metal::GlobalSemaphore barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, sub_device_core_range_set, 0);

    ttnn::SmallVector<tt::tt_metal::SubDeviceId> sub_device_ids = {sd_id};
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, sub_device_ids);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes, coord, tensor_args, tensor_return_value, multidevice_semaphores, barrier_semaphore);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return {std::move(mesh_workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<DeepseekReduceScatterMeshWorkloadFactory::shared_variables_t>
DeepseekReduceScatterMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::vector<tt::tt_metal::GlobalSemaphore>& multidevice_semaphores,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& intermediate_tensor = tensor_return_value.at(0);
    const ttnn::Tensor& output_tensor = tensor_return_value.at(1);

    std::optional<uint32_t> cluster_axis = operation_attributes.cluster_axis;

    const std::optional<MeshCoordinate> forward_coordinate = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, tt::tt_fabric::Topology::Ring, cluster_axis);
    const std::optional<MeshCoordinate> backward_coordinate = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, tt::tt_fabric::Topology::Ring, cluster_axis);
    TT_FATAL(
        forward_coordinate.has_value() || backward_coordinate.has_value(),
        "DEBUG: forward_coord or backward_coord is null");

    uint32_t device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, cluster_axis);
    log_debug(tt::LogOp, "Device index for {} is {}", mesh_coordinate, device_index);

    auto sub_device_id = operation_attributes.sub_device_id;
    auto* mesh_device = tensor_args.input_tensor.device();
    auto sd_id = sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto sub_device_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
    auto bbox = sub_device_core_range_set.bounding_box();
    auto first_coord = bbox.start_coord;

    tt::tt_metal::Program program{};
    auto deepseek_reduce_scatter_program_artifacts = build_deepseek_reduce_scatter_program_artifacts(
        program,
        input_tensor,
        intermediate_tensor,
        output_tensor,
        mesh_coordinate,
        forward_coordinate,
        backward_coordinate,
        device_index,
        multidevice_semaphores,
        barrier_semaphore,
        operation_attributes.num_links,
        operation_attributes.sub_device_id,
        first_coord);

    shared_variables_t shared_vars{
        .multidevice_semaphores = multidevice_semaphores,
        .barrier_semaphore = barrier_semaphore,
        .program_artifacts = deepseek_reduce_scatter_program_artifacts};

    return {std::move(program), std::move(shared_vars)};
}

void DeepseekReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& intermediate_tensor = tensor_return_value.at(0);
    const ttnn::Tensor& output_tensor = tensor_return_value.at(1);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        deepseek_reduce_scatter_helper_override_runtime_arguments(
            program,
            shared_vars.program_artifacts.reader_kernel_id,
            shared_vars.program_artifacts.writer_kernel_id,
            shared_vars.program_artifacts.all_cores,
            shared_vars.program_artifacts.num_directions_per_link,
            shared_vars.multidevice_semaphores,
            shared_vars.barrier_semaphore,
            operation_attributes.num_links,
            input_tensor,
            intermediate_tensor,
            output_tensor);
    }
}

}  // namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail
