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
#include <tt-metalium/tt_align.hpp>
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

DeviceAddr calculate_bank_size_spread(
    DeviceAddr size_bytes, DeviceAddr page_size_bytes, uint32_t num_banks, uint32_t alignment_bytes) {
    TT_ASSERT(
        page_size_bytes == 0 ? size_bytes == 0 : size_bytes % page_size_bytes == 0,
        "Page size {} should be divisible by buffer size {}",
        page_size_bytes,
        size_bytes);
    DeviceAddr num_pages = page_size_bytes == 0 ? 0 : size_bytes / page_size_bytes;
    DeviceAddr num_equally_distributed_pages = num_pages == 0 ? 0 : 1 + ((num_pages - 1) / num_banks);
    return num_equally_distributed_pages * tt::round_up(page_size_bytes, static_cast<DeviceAddr>(alignment_bytes));
}

// NOTE: shadow_global_buffer (set in set_globally_allocated_address_and_total_size) seems only to be used for
// operator==
tt::tt_metal::CircularBufferConfig create_sub_tensor_cb_config(
    uint32_t cb_idx,
    uint32_t slice_idx,
    const ttnn::Tensor& input_tensor,
    distributed::MeshDevice* mesh_device,
    uint32_t cb_num_pages,
    tt::DataFormat cb_data_format,
    CoreRangeSet shard_core_range_set) {
    uint32_t page_size = input_tensor.buffer()->page_size();

    const NdShardSpec& input_nd_shard_spec = input_tensor.nd_shard_spec().value();
    const Shape& input_nd_shard_shape = input_nd_shard_spec.shard_shape;

    const uint32_t input_tensor_nd_shard_shape_B = input_nd_shard_shape[0];
    const uint32_t input_tensor_nd_shard_shape_C = input_nd_shard_shape[1];
    const uint32_t input_tensor_nd_shard_shape_Ht = input_nd_shard_shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_nd_shard_shape_Wt = input_nd_shard_shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t shard_spec_num_pages = input_tensor_nd_shard_shape_B * input_tensor_nd_shard_shape_C *
                                          input_tensor_nd_shard_shape_Ht * input_tensor_nd_shard_shape_Wt;

    uint32_t alignment = mesh_device->allocator()->get_alignment(input_tensor.memory_config().buffer_type());
    uint32_t num_dev_pages = shard_spec_num_pages * shard_core_range_set.size();
    DeviceAddr aligned_page_size = tt::align(page_size, alignment);
    DeviceAddr aligned_size = num_dev_pages * aligned_page_size;

    uint32_t num_banks = shard_core_range_set.size();
    DeviceAddr max_size = calculate_bank_size_spread(aligned_size, aligned_page_size, num_banks, alignment);

    uint32_t buffer_size = num_dev_pages * aligned_page_size;

    tt::tt_metal::CircularBufferConfig circular_buffer_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * page_size, {{cb_idx, cb_data_format}})
            .set_page_size(cb_idx, page_size);

    uint32_t sub_tensor_offset = slice_idx * shard_spec_num_pages;
    uint32_t start_address = input_tensor.buffer()->address() + sub_tensor_offset;
    circular_buffer_config = tt::tt_metal::CircularBufferConfig(
        circular_buffer_config.total_size(),             // no changes from init
        start_address,                                   // set_globally_allocated_address_and_total_size
        circular_buffer_config.data_formats(),           // set_config(data_format_spec), no changes from init
        circular_buffer_config.page_sizes(),             // set_page_size(cb_id, page_size),  no changes from init
        circular_buffer_config.tiles(),                  // no changes from init (never set in init)
        circular_buffer_config.buffer_indices(),         // set_config(data_format_spec), no changes from init
        circular_buffer_config.local_buffer_indices(),   // set_config(data_format_spec), no changes from init
        circular_buffer_config.remote_buffer_indices(),  // no changes from init (never set in init)
        /* bool dynamic_cb - true */ true,               // set_globally_allocated_address_and_total_size
                                                         /* uint32_t max_size - buffer.aligned_size_per_bank() */
        max_size,                                        // set_globally_allocated_address_and_total_size
        /* uint32_t buffer_size - buffer.aligned_size() */ buffer_size  // set_globally_allocated_address_and_total_size
    );

    return circular_buffer_config;
}

DeepseekReduceScatterProgramArtifacts build_deepseek_reduce_scatter_program_artifacts(
    tt::tt_metal::Program& program,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& intermediate_tensor,
    const ttnn::Tensor& output_tensor,
    const ttnn::MeshCoordinate& sender_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    uint32_t ring_index,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& barrier_semaphores,
    uint32_t num_links,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord core_grid_offset) {
    auto* mesh_device = input_tensor.device();

    // hardcoded constants
    const uint32_t ring_size = 8;

    // choose cores
    // TODO: (GR) need to add an extra dummy core when moving to 4 links and the proper shape
    const NdShardSpec& input_nd_shard_spec = input_tensor.nd_shard_spec().value();
    CoreRangeSet worker_core_range_set = input_nd_shard_spec.grid;
    std::vector<CoreCoord> worker_cores = corerange_to_cores(worker_core_range_set);

    // tensor info
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

    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t slice_Wt = input_tensor_Wt / ring_size;

    const uint32_t input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t output_tensor_num_pages = input_tensor_num_pages / ring_size;

    // L1 Scratch CB Creation
    const uint32_t page_size = input_tensor.buffer()->page_size();
    const uint32_t tile_granularity = 2;  // NOTE: writer kernel hardcoded to always use scatter_write with 2 tiles
    const uint32_t cb_num_pages = 2 * tile_granularity;  // TODO: (GR) double buffering (enough to hold all tiles for a
                                                         // given slice), test/check if we should use more
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_slice_0_cb_id = tt::CBIndex::c_0;
    uint32_t input_slice_1_cb_id = tt::CBIndex::c_1;
    uint32_t input_slice_2_cb_id = tt::CBIndex::c_2;
    uint32_t input_slice_3_cb_id = tt::CBIndex::c_3;
    uint32_t input_slice_4_cb_id = tt::CBIndex::c_4;
    uint32_t input_slice_5_cb_id = tt::CBIndex::c_5;
    uint32_t input_slice_6_cb_id = tt::CBIndex::c_6;
    uint32_t input_slice_7_cb_id = tt::CBIndex::c_7;
    uint32_t intermediate_cb_id = tt::CBIndex::c_8;
    uint32_t compute_cb_id = tt::CBIndex::c_9;

    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * page_size, {{intermediate_cb_id, df}})
            .set_page_size(intermediate_cb_id, page_size);
    CreateCircularBuffer(program, worker_core_range_set, cb_intermediate_config);

    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * page_size, {{compute_cb_id, df}})
            .set_page_size(compute_cb_id, page_size);
    CreateCircularBuffer(program, worker_core_range_set, cb_compute_output_config);

    tt::tt_metal::CircularBufferConfig input_slice_0_cb_config = create_sub_tensor_cb_config(
        input_slice_0_cb_id,
        /* slice_idx */ 0,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_0_cb_config);  // SEG FAULTING HERE

    tt::tt_metal::CircularBufferConfig input_slice_1_cb_config = create_sub_tensor_cb_config(
        input_slice_1_cb_id,
        /* slice_idx */ 1,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_1_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_2_cb_config = create_sub_tensor_cb_config(
        input_slice_2_cb_id,
        /* slice_idx */ 2,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_2_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_3_cb_config = create_sub_tensor_cb_config(
        input_slice_3_cb_id,
        /* slice_idx */ 3,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_3_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_4_cb_config = create_sub_tensor_cb_config(
        input_slice_4_cb_id,
        /* slice_idx */ 4,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_4_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_5_cb_config = create_sub_tensor_cb_config(
        input_slice_5_cb_id,
        /* slice_idx */ 5,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_5_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_6_cb_config = create_sub_tensor_cb_config(
        input_slice_6_cb_id,
        /* slice_idx */ 6,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_6_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_7_cb_config = create_sub_tensor_cb_config(
        input_slice_7_cb_id,
        /* slice_idx */ 7,
        input_tensor,
        mesh_device,
        cb_num_pages,
        df,
        /* shard_core_range_set */ worker_core_range_set);
    CreateCircularBuffer(program, worker_core_range_set, input_slice_7_cb_config);

    // handle output sharded tensors using ShardedAddrGen
    bool output_is_sharded = output_tensor.is_sharded();
    std::map<std::string, std::string> writer_compute_defines;
    if (output_is_sharded) {
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }

    // reader
    std::vector<uint32_t> reader_ct_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        page_size,         // page_size
        tile_granularity,  // tile_granularity
        input_tensor_Wt,   // input_tensor_Wt
        slice_Wt,          // slice_Wt
        input_slice_0_cb_id,
        input_slice_1_cb_id,
        input_slice_2_cb_id,
        input_slice_3_cb_id,
        input_slice_4_cb_id,
        input_slice_5_cb_id,
        input_slice_6_cb_id,
        input_slice_7_cb_id,
        intermediate_cb_id,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(reader_ct_args);

    std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/kernels/"
        "deepseek_reduce_scatter_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, worker_core_range_set, tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // writer
    std::vector<uint32_t> writer_ct_args = {
        ring_index,        // my_chip_id
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        page_size,         // page_size
        tile_granularity,  // tile_granularity
        input_tensor_Wt,   // input_tensor_Wt
        slice_Wt,          // slice_Wt
        input_slice_0_cb_id,
        input_slice_1_cb_id,
        input_slice_2_cb_id,
        input_slice_3_cb_id,
        input_slice_4_cb_id,
        input_slice_5_cb_id,
        input_slice_6_cb_id,
        input_slice_7_cb_id,
        compute_cb_id,
    };
    tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(writer_ct_args);
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, writer_ct_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);
    }

    std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/kernels/"
        "deepseek_reduce_scatter_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        writer_kernel_path,
        worker_core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args, writer_compute_defines));

    // reduce
    auto reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    reduce_kernel_config.compile_args = {
        ring_size,         // ring_size
        tile_granularity,  // tile_granularity
        input_slice_0_cb_id,
        input_slice_1_cb_id,
        input_slice_2_cb_id,
        input_slice_3_cb_id,
        input_slice_4_cb_id,
        input_slice_5_cb_id,
        input_slice_6_cb_id,
        input_slice_7_cb_id,
        intermediate_cb_id,
        compute_cb_id,
    };

    std::string reduce_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_reduce_scatter/device/kernels/deepseek_reduction.cpp";

    auto reduce_kernel_id =
        tt::tt_metal::CreateKernel(program, reduce_kernel_path, worker_core_range_set, reduce_kernel_config);

    // runtime args
    const uint32_t num_directions_per_link = 2;
    auto worker_core_iter = worker_core_range_set.ranges().cbegin();
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t direction = 0; direction < num_directions_per_link; direction++) {
            auto core = *((worker_core_iter++)->begin());
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

            // DO NOT ALTERNATE DIRECTIONS version
            // will break for certain tensors since processing in batches of 2 tiles (scatter_write)
            uint32_t num_workers = num_directions_per_link * num_links;
            uint32_t tiles_per_worker = tt::div_up(output_tensor_num_pages, num_workers);
            uint32_t start_tiles_read = tiles_per_worker * ((link * num_directions_per_link) + direction);
            uint32_t start_tiles_to_read = tiles_per_worker * ((link * num_directions_per_link) + direction + 1);
            start_tiles_to_read = std::min(start_tiles_to_read, output_tensor_num_pages);
            /*
             * NOTE
             * - need to create kernels even if worker not processing tiles
             * - required for pre and post op barrier/sync
             * - min so that we don't try to process non-existent tiles
             */

            // OLD version
            // uint32_t start_tiles_read = link * output_tensor_num_pages / num_links;
            // uint32_t start_tiles_to_read = (link + 1) * output_tensor_num_pages / num_links;

            uint32_t start_pages_read_in_row = start_tiles_read % slice_Wt;
            uint32_t start_row_offset = start_tiles_read / slice_Wt * input_tensor_Wt;

            // reader
            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),         // input_tensor_address
                intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                op_semaphore.address(),                   // op_semaphore
                direction,                                // direction
                start_tiles_read,                         // start_tiles_read
                start_tiles_to_read,                      // start_tiles_to_read
                start_pages_read_in_row,                  // start_pages_read_in_row
                start_row_offset,                         // start_row_offset
            };

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

            // writer
            std::vector<uint32_t> writer_rt_args = {
                intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                output_tensor.buffer()->address(),        // output_tensor_address
                virtual_core.x,                           // semaphore_noc0_x
                virtual_core.y,                           // semaphore_noc0_y
                op_semaphore.address(),                   // op_semaphore
                barrier_semaphores.at(0).address(),       // pre_op_barrier_semaphore
                barrier_semaphores.at(1).address(),       // post_op_barrier_semaphore
                direction,                                // direction
                start_tiles_read,                         // start_tiles_read
                start_tiles_to_read,                      // tiles_to_read
                start_pages_read_in_row,                  // start_pages_read_in_row
                start_row_offset,                         // start_row_offset
            };
            if (output_is_sharded) {
                shard_builder::extend_sharding_run_time_args(output_tensor, writer_rt_args);
            }

            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_coord);
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            uint32_t num_connections = 1;
            dst_nodes.reserve(num_connections);
            if (direction == 0) {
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

            // reduce
            std::vector<uint32_t> reduce_rt_args = {
                start_tiles_read,     // start_tiles_read
                start_tiles_to_read,  // start_tiles_to_read
                direction};           // direction
            tt::tt_metal::SetRuntimeArgs(program, reduce_kernel_id, {core}, reduce_rt_args);
        }
    }

    return {reader_kernel_id, writer_kernel_id, worker_cores, num_directions_per_link};
}

void deepseek_reduce_scatter_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t num_directions_per_link,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& barrier_semaphores,
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

            // reader
            auto& reader_rt_args = reader_runtime_args[core.x][core.y];
            reader_rt_args[0] = input_tensor.buffer()->address();
            reader_rt_args[1] = intermediate_tensor.buffer()->address();
            reader_rt_args[2] = op_semaphore.address();

            // writer
            auto& writer_rt_args = writer_runtime_args[core.x][core.y];
            writer_rt_args[0] = intermediate_tensor.buffer()->address();
            writer_rt_args[1] = output_tensor.buffer()->address();
            writer_rt_args[4] = op_semaphore.address();
            writer_rt_args[5] = barrier_semaphores.at(0).address();
            writer_rt_args[6] = barrier_semaphores.at(1).address();
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

    // 1 semaphore used for within op synchronizations
    tt::tt_metal::GlobalSemaphore op_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, sub_device_core_range_set, 0);

    // 2 barrier semaphores used for pre/post op synchronization
    // pre: remote tensors are allocated, post: all incoming data received
    std::vector<tt::tt_metal::GlobalSemaphore> barrier_semaphores = {
        ttnn::global_semaphore::create_global_semaphore(mesh_device, sub_device_core_range_set, 0),
        ttnn::global_semaphore::create_global_semaphore(mesh_device, sub_device_core_range_set, 0),
    };

    ttnn::SmallVector<tt::tt_metal::SubDeviceId> sub_device_ids = {sd_id};
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, sub_device_ids);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, tensor_return_value, op_semaphore, barrier_semaphores);
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
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const std::vector<tt::tt_metal::GlobalSemaphore>& barrier_semaphores) {
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
        op_semaphore,
        barrier_semaphores,
        operation_attributes.num_links,
        operation_attributes.sub_device_id,
        first_coord);

    shared_variables_t shared_vars{
        .op_semaphore = op_semaphore,
        .barrier_semaphores = barrier_semaphores,
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
            shared_vars.op_semaphore,
            shared_vars.barrier_semaphores,
            operation_attributes.num_links,
            input_tensor,
            intermediate_tensor,
            output_tensor);
    }
}

}  // namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail
