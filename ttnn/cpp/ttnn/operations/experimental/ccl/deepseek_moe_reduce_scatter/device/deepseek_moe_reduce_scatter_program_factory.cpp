// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_program_factory.hpp"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/math.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

using ttnn::experimental::prim::DeepseekMoEReduceScatterProgramArtifacts;

namespace {

CoreCoord choose_additional_core(
    ttnn::MeshDevice* mesh_device, const std::vector<CoreCoord>& cores_already_selected, uint32_t clamped_num_links) {
    /*
     * - optimal core to use as the additional core (when necessary), so that each used link has both a forward and
     * backward worker
     * - respective core is only optimal when the optimal shard grid is used for the input tensors
     */
    constexpr std::array optimal_supplemental_core_per_link = {
        CoreCoord(2, 5),
        CoreCoord(3, 5),
        CoreCoord(6, 5),
        CoreCoord(7, 5),
    };

    // try optimal core first
    CoreCoord optimal_supplemental_core = optimal_supplemental_core_per_link.at(clamped_num_links - 1);
    if (std::find(cores_already_selected.begin(), cores_already_selected.end(), optimal_supplemental_core) ==
        cores_already_selected.end()) {
        return optimal_supplemental_core;
    }

    // try to find any other available core
    auto available_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0));
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                CoreCoord core = CoreCoord(x, y);
                if (std::find(cores_already_selected.begin(), cores_already_selected.end(), core) ==
                    cores_already_selected.end()) {
                    return core;
                }
            }
        }
    }

    TT_FATAL(false, "deepseek_moe_reduce_scatter requires an even number of worker cores");
}

std::tuple<uint32_t, CoreRangeSet, std::vector<CoreCoord>> get_cores(
    ttnn::MeshDevice* mesh_device,
    const NdShardSpec& input_nd_shard_spec,
    uint32_t num_shards,
    uint32_t num_directions_per_link) {
    uint32_t clamped_num_links = tt::div_up(num_shards, num_directions_per_link);

    std::vector<CoreCoord> worker_cores = corerange_to_cores(
        input_nd_shard_spec.grid, num_shards, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    TT_FATAL(
        worker_cores.size() == num_shards,
        "deepseek_moe_reduce_scatter requires each shard to be located on a different core");

    // always need a forward and backward core for each link being used (for in op synchronization), even if the forward
    // worker isn't being used for data transfer due to an odd number of shards
    if (num_shards % 2 != 0) {
        worker_cores.emplace_back(choose_additional_core(mesh_device, worker_cores, clamped_num_links));
    }

    std::vector<CoreRange> worker_core_ranges;
    worker_core_ranges.reserve(worker_cores.size());
    for (const CoreCoord& worker_core : worker_cores) {
        worker_core_ranges.emplace_back(worker_core);
    }
    CoreRangeSet worker_core_range_set = CoreRangeSet(worker_core_ranges);

    return {clamped_num_links, worker_core_range_set, worker_cores};
}

DeepseekMoEReduceScatterProgramArtifacts build_deepseek_moe_reduce_scatter_program_artifacts(
    tt::tt_metal::Program& program,
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::vector<ttnn::Tensor>& intermediate_slice_tensors,
    const ttnn::Tensor& output_tensor,
    const ttnn::MeshCoordinate& sender_coord,
    const std::optional<ttnn::MeshCoordinate>& forward_coord,
    const std::optional<ttnn::MeshCoordinate>& backward_coord,
    uint32_t ring_index,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const tt::tt_metal::GlobalSemaphore& pre_op_barrier_semaphore,
    uint32_t num_links) {
    auto* mesh_device = input_tensors.at(0).device();

    // hardcoded constants
    const uint32_t ring_size = 8;
    const uint32_t num_directions_per_link = 2;
    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    // tensor details
    const NdShardSpec& input_nd_shard_spec = input_tensors.at(0).nd_shard_spec().value();
    const uint32_t num_pages_per_shard = input_nd_shard_spec.shard_shape.volume() / num_tile_elements;
    const uint32_t num_shards = input_tensors.at(0).logical_volume() / (num_tile_elements * num_pages_per_shard);
    const uint32_t num_pages_per_slice = input_tensors.at(0).buffer()->num_pages();
    const uint32_t page_size = input_tensors.at(0).buffer()->page_size();

    // choose cores
    const auto [clamped_num_links, worker_core_range_set, worker_cores] =
        get_cores(mesh_device, input_nd_shard_spec, num_shards, num_directions_per_link);
    TT_FATAL(clamped_num_links <= num_links, "{} links available, but {} requested", num_links, clamped_num_links);

    // NOTE: writer kernel hardcoded to always use scatter_write with 2 tiles
    const uint32_t tile_granularity = 2;

    // L1 scratch CB creation
    const uint32_t compute_input_cb_num_pages = num_pages_per_shard;   // entire shard
    const uint32_t compute_ouput_cb_num_pages = 2 * tile_granularity;  // double buffer

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype());

    uint32_t input_slice_0_cb_id = tt::CBIndex::c_0;
    uint32_t input_slice_1_cb_id = tt::CBIndex::c_1;
    uint32_t input_slice_2_cb_id = tt::CBIndex::c_2;
    uint32_t input_slice_3_cb_id = tt::CBIndex::c_3;
    uint32_t input_slice_4_cb_id = tt::CBIndex::c_4;
    uint32_t input_slice_5_cb_id = tt::CBIndex::c_5;
    uint32_t input_slice_6_cb_id = tt::CBIndex::c_6;
    uint32_t input_slice_7_cb_id = tt::CBIndex::c_7;

    uint32_t intermediate_slice_0_cb_id = tt::CBIndex::c_8;
    uint32_t intermediate_slice_1_cb_id = tt::CBIndex::c_9;
    uint32_t intermediate_slice_2_cb_id = tt::CBIndex::c_10;
    uint32_t intermediate_slice_3_cb_id = tt::CBIndex::c_11;
    uint32_t intermediate_slice_4_cb_id = tt::CBIndex::c_12;
    uint32_t intermediate_slice_5_cb_id = tt::CBIndex::c_13;
    uint32_t intermediate_slice_6_cb_id = tt::CBIndex::c_14;
    uint32_t intermediate_slice_7_cb_id = tt::CBIndex::c_15;

    uint32_t compute_cb_id = tt::CBIndex::c_16;

    // input CBs
    tt::tt_metal::CircularBufferConfig input_slice_0_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_0_cb_id, data_format}})
            .set_page_size(input_slice_0_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(0).buffer());
    tt::tt_metal::CBHandle input_slice_0_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_0_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_1_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_1_cb_id, data_format}})
            .set_page_size(input_slice_1_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(1).buffer());
    tt::tt_metal::CBHandle input_slice_1_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_1_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_2_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_2_cb_id, data_format}})
            .set_page_size(input_slice_2_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(2).buffer());
    tt::tt_metal::CBHandle input_slice_2_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_2_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_3_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_3_cb_id, data_format}})
            .set_page_size(input_slice_3_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(3).buffer());
    tt::tt_metal::CBHandle input_slice_3_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_3_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_4_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_4_cb_id, data_format}})
            .set_page_size(input_slice_4_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(4).buffer());
    tt::tt_metal::CBHandle input_slice_4_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_4_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_5_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_5_cb_id, data_format}})
            .set_page_size(input_slice_5_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(5).buffer());
    tt::tt_metal::CBHandle input_slice_5_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_5_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_6_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_6_cb_id, data_format}})
            .set_page_size(input_slice_6_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(6).buffer());
    tt::tt_metal::CBHandle input_slice_6_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_6_cb_config);

    tt::tt_metal::CircularBufferConfig input_slice_7_cb_config =
        tt::tt_metal::CircularBufferConfig(compute_input_cb_num_pages * page_size, {{input_slice_7_cb_id, data_format}})
            .set_page_size(input_slice_7_cb_id, page_size)
            .set_globally_allocated_address(*input_tensors.at(7).buffer());
    tt::tt_metal::CBHandle input_slice_7_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, input_slice_7_cb_config);

    std::vector<tt::tt_metal::CBHandle> input_cb_handles = {
        input_slice_0_cb_handle,
        input_slice_1_cb_handle,
        input_slice_2_cb_handle,
        input_slice_3_cb_handle,
        input_slice_4_cb_handle,
        input_slice_5_cb_handle,
        input_slice_6_cb_handle,
        input_slice_7_cb_handle,
    };

    // intermediate CBs
    tt::tt_metal::CircularBufferConfig intermediate_slice_0_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_0_cb_id, data_format}})
            .set_page_size(intermediate_slice_0_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(0).buffer());
    tt::tt_metal::CBHandle intermediate_slice_0_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_0_cb_config);

    tt::tt_metal::CircularBufferConfig intermediate_slice_1_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_1_cb_id, data_format}})
            .set_page_size(intermediate_slice_1_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(1).buffer());
    tt::tt_metal::CBHandle intermediate_slice_1_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_1_cb_config);

    tt::tt_metal::CircularBufferConfig intermediate_slice_2_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_2_cb_id, data_format}})
            .set_page_size(intermediate_slice_2_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(2).buffer());
    tt::tt_metal::CBHandle intermediate_slice_2_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_2_cb_config);

    tt::tt_metal::CircularBufferConfig intermediate_slice_3_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_3_cb_id, data_format}})
            .set_page_size(intermediate_slice_3_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(3).buffer());
    tt::tt_metal::CBHandle intermediate_slice_3_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_3_cb_config);

    tt::tt_metal::CircularBufferConfig intermediate_slice_4_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_4_cb_id, data_format}})
            .set_page_size(intermediate_slice_4_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(4).buffer());
    tt::tt_metal::CBHandle intermediate_slice_4_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_4_cb_config);

    tt::tt_metal::CircularBufferConfig intermediate_slice_5_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_5_cb_id, data_format}})
            .set_page_size(intermediate_slice_5_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(5).buffer());
    tt::tt_metal::CBHandle intermediate_slice_5_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_5_cb_config);

    tt::tt_metal::CircularBufferConfig intermediate_slice_6_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_6_cb_id, data_format}})
            .set_page_size(intermediate_slice_6_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(6).buffer());
    tt::tt_metal::CBHandle intermediate_slice_6_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_6_cb_config);

    tt::tt_metal::CircularBufferConfig intermediate_slice_7_cb_config =
        tt::tt_metal::CircularBufferConfig(
            compute_input_cb_num_pages * page_size, {{intermediate_slice_7_cb_id, data_format}})
            .set_page_size(intermediate_slice_7_cb_id, page_size)
            .set_globally_allocated_address(*intermediate_slice_tensors.at(7).buffer());
    tt::tt_metal::CBHandle intermediate_slice_7_cb_handle =
        CreateCircularBuffer(program, worker_core_range_set, intermediate_slice_7_cb_config);

    std::vector<tt::tt_metal::CBHandle> intermediate_cb_handles = {
        intermediate_slice_0_cb_handle,
        intermediate_slice_1_cb_handle,
        intermediate_slice_2_cb_handle,
        intermediate_slice_3_cb_handle,
        intermediate_slice_4_cb_handle,
        intermediate_slice_5_cb_handle,
        intermediate_slice_6_cb_handle,
        intermediate_slice_7_cb_handle,
    };

    // compute CB
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(compute_ouput_cb_num_pages * page_size, {{compute_cb_id, data_format}})
            .set_page_size(compute_cb_id, page_size);
    CreateCircularBuffer(program, worker_core_range_set, cb_compute_output_config);

    // reader
    std::vector<uint32_t> reader_ct_args = {
        ring_index,        // my_chip_id
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
        intermediate_slice_0_cb_id,
        intermediate_slice_1_cb_id,
        intermediate_slice_2_cb_id,
        intermediate_slice_3_cb_id,
        intermediate_slice_4_cb_id,
        intermediate_slice_5_cb_id,
        intermediate_slice_6_cb_id,
        intermediate_slice_7_cb_id,
    };

    std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_reader.cpp";

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, worker_core_range_set, tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    // writer
    std::vector<uint32_t> writer_ct_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        page_size,         // page_size
        tile_granularity,  // tile_granularity
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
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(0).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(1).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(2).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(3).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(4).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(5).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(6).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(7).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);

    std::string writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_writer.cpp";

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, worker_core_range_set, tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    // reduce
    auto reduce_kernel_config = tt::tt_metal::ComputeConfig{};
    reduce_kernel_config.compile_args = {
        ring_index,        // my_chip_id
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
        intermediate_slice_0_cb_id,
        intermediate_slice_1_cb_id,
        intermediate_slice_2_cb_id,
        intermediate_slice_3_cb_id,
        intermediate_slice_4_cb_id,
        intermediate_slice_5_cb_id,
        intermediate_slice_6_cb_id,
        intermediate_slice_7_cb_id,
        compute_cb_id,
    };

    std::string reduce_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_reduction.cpp";

    auto reduce_kernel_id =
        tt::tt_metal::CreateKernel(program, reduce_kernel_path, worker_core_range_set, reduce_kernel_config);

    // runtime args
    for (uint32_t link = 0; link < clamped_num_links; link++) {
        for (uint32_t direction = 0; direction < num_directions_per_link; direction++) {
            uint32_t worker_id = (link * num_directions_per_link) + direction;
            uint32_t opposite_direction_worker_id =
                (link * num_directions_per_link) + ((direction + 1) % num_directions_per_link);

            CoreCoord core = worker_cores[worker_id];
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

            CoreCoord opposite_direction_core = worker_cores[opposite_direction_worker_id];
            CoreCoord opposition_direction_virtual_core =
                mesh_device->worker_core_from_logical_core(opposite_direction_core);

            /*
             * NOTE
             * - need to create kernels even if worker not processing tiles, required for pre and post op barrier/sync
             * - min so that we don't try to process non-existent tiles on that dummy worker
             */
            uint32_t start_tiles_read = num_pages_per_shard * worker_id;
            uint32_t start_tiles_to_read = num_pages_per_shard * (worker_id + 1);
            start_tiles_to_read = std::min(start_tiles_to_read, num_pages_per_slice);

            // reader
            std::vector<uint32_t> reader_rt_args = {
                op_semaphore.address(),  // op_semaphore
                direction,               // direction
                start_tiles_read,        // start_tiles_read
                start_tiles_to_read,     // start_tiles_to_read
            };

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

            // writer
            std::vector<uint32_t> writer_rt_args = {
                intermediate_slice_tensors.at(0).mesh_buffer()->address(),  // intermediate_slice_0_address
                intermediate_slice_tensors.at(1).mesh_buffer()->address(),  // intermediate_slice_1_address
                intermediate_slice_tensors.at(2).mesh_buffer()->address(),  // intermediate_slice_2_address
                intermediate_slice_tensors.at(3).mesh_buffer()->address(),  // intermediate_slice_3_address
                intermediate_slice_tensors.at(4).mesh_buffer()->address(),  // intermediate_slice_4_address
                intermediate_slice_tensors.at(5).mesh_buffer()->address(),  // intermediate_slice_5_address
                intermediate_slice_tensors.at(6).mesh_buffer()->address(),  // intermediate_slice_6_address
                intermediate_slice_tensors.at(7).mesh_buffer()->address(),  // intermediate_slice_7_address
                output_tensor.mesh_buffer()->address(),                     // output_address
                virtual_core.x,                                             // op_semaphore_noc0_x
                virtual_core.y,                                             // op_semaphore_noc0_y
                op_semaphore.address(),                                     // op_semaphore
                opposition_direction_virtual_core.x,                        // pre_op_barrier_semaphore_noc0_x
                opposition_direction_virtual_core.y,                        // pre_op_barrier_semaphore_noc0_y
                pre_op_barrier_semaphore.address(),                         // pre_op_barrier_semaphore
                direction,                                                  // direction
                start_tiles_read,                                           // start_tiles_read
                start_tiles_to_read,                                        // tiles_to_read
            };

            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_coord);
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            dst_nodes.reserve(1);
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

    return {
        reader_kernel_id,
        writer_kernel_id,
        worker_cores,
        clamped_num_links,
        num_directions_per_link,
        input_cb_handles,
        intermediate_cb_handles};
}

void deepseek_moe_reduce_scatter_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    const tt::tt_metal::KernelHandle reader_kernel_id,
    const tt::tt_metal::KernelHandle writer_kernel_id,
    const std::vector<tt::tt_metal::CoreCoord>& all_cores,
    uint32_t clamped_num_links,
    uint32_t num_directions_per_link,
    const std::vector<tt::tt_metal::CBHandle>& input_cb_handles,
    const std::vector<tt::tt_metal::CBHandle>& intermediate_cb_handles,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const tt::tt_metal::GlobalSemaphore& pre_op_barrier_semaphore,
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::vector<ttnn::Tensor>& intermediate_slice_tensors,
    const ttnn::Tensor& output_tensor) {
    // input CB handles
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(0), *input_tensors.at(0).buffer());
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(1), *input_tensors.at(1).buffer());
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(2), *input_tensors.at(2).buffer());
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(3), *input_tensors.at(3).buffer());
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(4), *input_tensors.at(4).buffer());
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(5), *input_tensors.at(5).buffer());
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(6), *input_tensors.at(6).buffer());
    UpdateDynamicCircularBufferAddress(program, input_cb_handles.at(7), *input_tensors.at(7).buffer());

    // intermediate CB handles
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(0), *intermediate_slice_tensors.at(0).buffer());
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(1), *intermediate_slice_tensors.at(1).buffer());
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(2), *intermediate_slice_tensors.at(2).buffer());
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(3), *intermediate_slice_tensors.at(3).buffer());
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(4), *intermediate_slice_tensors.at(4).buffer());
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(5), *intermediate_slice_tensors.at(5).buffer());
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(6), *intermediate_slice_tensors.at(6).buffer());
    UpdateDynamicCircularBufferAddress(
        program, intermediate_cb_handles.at(7), *intermediate_slice_tensors.at(7).buffer());

    // update senders
    for (uint32_t link = 0; link < clamped_num_links; link++) {
        for (uint32_t direction = 0; direction < num_directions_per_link; direction++) {
            uint32_t worker_id = (link * num_directions_per_link) + direction;
            CoreCoord core = all_cores[worker_id];
            std::vector<std::vector<RuntimeArgsData>> reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
            std::vector<std::vector<RuntimeArgsData>> writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

            // reader
            auto& reader_rt_args = reader_runtime_args[core.x][core.y];
            reader_rt_args[0] = op_semaphore.address();

            // writer
            auto& writer_rt_args = writer_runtime_args[core.x][core.y];
            writer_rt_args[0] = intermediate_slice_tensors.at(0).mesh_buffer()->address();
            writer_rt_args[1] = intermediate_slice_tensors.at(1).mesh_buffer()->address();
            writer_rt_args[2] = intermediate_slice_tensors.at(2).mesh_buffer()->address();
            writer_rt_args[3] = intermediate_slice_tensors.at(3).mesh_buffer()->address();
            writer_rt_args[4] = intermediate_slice_tensors.at(4).mesh_buffer()->address();
            writer_rt_args[5] = intermediate_slice_tensors.at(5).mesh_buffer()->address();
            writer_rt_args[6] = intermediate_slice_tensors.at(6).mesh_buffer()->address();
            writer_rt_args[7] = intermediate_slice_tensors.at(7).mesh_buffer()->address();
            writer_rt_args[8] = output_tensor.mesh_buffer()->address();
            writer_rt_args[11] = op_semaphore.address();
            writer_rt_args[14] = pre_op_barrier_semaphore.address();
        }
    }
}

}  // namespace

namespace ttnn::experimental::prim {

DeepseekMoEReduceScatterMeshWorkloadFactory::cached_mesh_workload_t
DeepseekMoEReduceScatterMeshWorkloadFactory::create_mesh_workload(
    const DeepseekMoEReduceScatterParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const DeepseekMoEReduceScatterInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensors.at(0).device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    // 1 semaphore used for within op synchronizations
    tt::tt_metal::GlobalSemaphore op_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);

    // 1 semaphore used for pre op synchronization to ensure intermediate/output tensors are allocated
    tt::tt_metal::GlobalSemaphore pre_op_semaphore_barrier =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);

    ttnn::SmallVector<tt::tt_metal::SubDeviceId> sub_device_ids = {sd_id};
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, sub_device_ids);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes, coord, tensor_args, tensor_return_value, op_semaphore, pre_op_semaphore_barrier);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return {std::move(mesh_workload), std::move(shared_variables)};
}

ttnn::device_operation::CachedProgram<DeepseekMoEReduceScatterMeshWorkloadFactory::shared_variables_t>
DeepseekMoEReduceScatterMeshWorkloadFactory::create_at(
    const DeepseekMoEReduceScatterParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const DeepseekMoEReduceScatterInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const tt::tt_metal::GlobalSemaphore& pre_op_barrier_semaphore) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    const std::vector<ttnn::Tensor> intermediate_slice_tensors(
        tensor_return_value.begin(), tensor_return_value.end() - 1);  // first 8 are intermediate tensors
    const ttnn::Tensor& output_tensor = tensor_return_value.back();   // last is the output tensor

    std::optional<uint32_t> cluster_axis = operation_attributes.cluster_axis;

    const std::optional<ttnn::MeshCoordinate> forward_coordinate =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensors.at(0), mesh_coordinate, 1, tt::tt_fabric::Topology::Ring, cluster_axis);
    const std::optional<ttnn::MeshCoordinate> backward_coordinate =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensors.at(0), mesh_coordinate, -1, tt::tt_fabric::Topology::Ring, cluster_axis);
    TT_FATAL(
        forward_coordinate.has_value() && backward_coordinate.has_value(),
        "DEBUG: forward_coord or backward_coord is null");

    uint32_t device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensors.at(0), mesh_coordinate, cluster_axis);
    log_debug(tt::LogOp, "Device index for {} is {}", mesh_coordinate, device_index);

    tt::tt_metal::Program program{};
    auto deepseek_moe_reduce_scatter_program_artifacts = build_deepseek_moe_reduce_scatter_program_artifacts(
        program,
        input_tensors,
        intermediate_slice_tensors,
        output_tensor,
        mesh_coordinate,
        forward_coordinate,
        backward_coordinate,
        device_index,
        op_semaphore,
        pre_op_barrier_semaphore,
        operation_attributes.num_links);

    shared_variables_t shared_vars{
        .op_semaphore = op_semaphore,
        .pre_op_barrier_semaphore = pre_op_barrier_semaphore,
        .program_artifacts = deepseek_moe_reduce_scatter_program_artifacts};

    return {std::move(program), std::move(shared_vars)};
}

void DeepseekMoEReduceScatterMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DeepseekMoEReduceScatterParams&,
    const DeepseekMoEReduceScatterInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    const std::vector<ttnn::Tensor> intermediate_slice_tensors(
        tensor_return_value.begin(), tensor_return_value.end() - 1);  // first 8 are intermediate tensors
    const ttnn::Tensor& output_tensor = tensor_return_value.back();   // last is the output tensor

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        deepseek_moe_reduce_scatter_helper_override_runtime_arguments(
            program,
            shared_vars.program_artifacts.reader_kernel_id,
            shared_vars.program_artifacts.writer_kernel_id,
            shared_vars.program_artifacts.all_cores,
            shared_vars.program_artifacts.clamped_num_links,
            shared_vars.program_artifacts.num_directions_per_link,
            shared_vars.program_artifacts.input_cb_handles,
            shared_vars.program_artifacts.intermediate_cb_handles,
            shared_vars.op_semaphore,
            shared_vars.pre_op_barrier_semaphore,
            input_tensors,
            intermediate_slice_tensors,
            output_tensor);
    }
}

}  // namespace ttnn::experimental::prim
