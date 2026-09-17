// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "indexed_fused_update_cache_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim::indexed_fused_update_cache {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts IndexedFusedUpdateCacheProgramFactory::create_program_artifacts(
    const IndexedFusedUpdateCacheParams&, const IndexedFusedUpdateCacheInputs& args, IndexedFusedUpdateCacheResult&) {
    const auto& cache1 = args.cache_tensor1;
    const auto& cache2 = args.cache_tensor2;
    const auto& input1 = args.input_tensor1;
    const auto& input2 = args.input_tensor2;
    const auto& positions = args.physical_update_idxs_tensor;

    const uint32_t num_heads = cache1.logical_shape()[1];
    const uint32_t cache_page_rows = cache1.logical_shape()[2];
    const uint32_t total_cache_rows = cache1.logical_shape()[0] * cache_page_rows;
    const uint32_t width_tiles = cache1.logical_shape()[3] / tt::constants::TILE_WIDTH;
    const uint32_t source_rows = input1.logical_shape()[2];
    const uint32_t source_height_tiles = input1.padded_shape()[2] / tt::constants::TILE_HEIGHT;
    const uint32_t worker_count = num_heads * width_tiles;

    // Work is partitioned over disjoint (head, width_tile) workers, not cache pages. Every worker
    // walks the positions but writes only its own tile column, so num_cores may exceed the page count.
    const auto grid = cache1.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min(worker_count, static_cast<uint32_t>(grid.x * grid.y));
    const auto all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);

    const auto cache_data_format = datatype_to_dataformat_converter(cache1.dtype());
    const auto positions_data_format = datatype_to_dataformat_converter(positions.dtype());
    const uint32_t bytes_per_element = cache1.element_size();
    const uint32_t tile_bytes = cache1.tensor_spec().tile().get_tile_size(cache_data_format);
    const uint32_t positions_page_bytes = positions.buffer()->aligned_page_size();
    constexpr uint32_t scratch_buffer_depth = 2;

    const DFBSpecName SCRATCH{"scratch"};
    const DFBSpecName POSITIONS_DFB{"positions"};
    const KernelSpecName UPDATE{"update"};
    const TensorParamName CACHE1{"cache1"};
    const TensorParamName CACHE2{"cache2"};
    const TensorParamName INPUT1{"input1"};
    const TensorParamName INPUT2{"input2"};
    const TensorParamName POSITIONS{"positions"};

    KernelSpec update{
        .unique_id = UPDATE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/indexed_page_cache/device/kernels/dataflow/"
            "indexed_fused_update_cache.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SCRATCH,
                 .accessor_name = "scratch",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = SCRATCH,
                 .accessor_name = "scratch",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = POSITIONS_DFB,
                 .accessor_name = "positions",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = POSITIONS_DFB,
                 .accessor_name = "positions",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             }},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = CACHE1, .accessor_name = "cache1"},
             TensorBinding{.tensor_parameter_name = CACHE2, .accessor_name = "cache2"},
             TensorBinding{.tensor_parameter_name = INPUT1, .accessor_name = "input1"},
             TensorBinding{.tensor_parameter_name = INPUT2, .accessor_name = "input2"},
             TensorBinding{.tensor_parameter_name = POSITIONS, .accessor_name = "positions"}},
        .compile_time_args =
            {{"num_heads", num_heads},
             {"width_tiles", width_tiles},
             {"source_height_tiles", source_height_tiles},
             {"cache_page_rows", cache_page_rows},
             {"total_cache_rows", total_cache_rows},
             {"worker_count", worker_count},
             {"bytes_per_element", bytes_per_element},
             {"scratch_buffer_depth", scratch_buffer_depth}},
        .runtime_arg_schema = {.runtime_arg_names = {"source_rows", "worker_start", "worker_stride"}},
        .hw_config = ttnn::create_reader_datamovement_config(cache1.device()->arch()),
    };

    ProgramSpec spec{
        .name = "indexed_fused_update_cache",
        .kernels = {update},
        .dataflow_buffers =
            {DataflowBufferSpec{
                 .unique_id = SCRATCH,
                 .entry_size = tile_bytes,
                 .num_entries = scratch_buffer_depth,
                 .data_format_metadata = cache_data_format,
             },
             DataflowBufferSpec{
                 .unique_id = POSITIONS_DFB,
                 .entry_size = positions_page_bytes,
                 .num_entries = 1,
                 .data_format_metadata = positions_data_format,
             }},
        .tensor_parameters =
            {TensorParameter{.unique_id = CACHE1, .spec = cache1.tensor_spec()},
             TensorParameter{.unique_id = CACHE2, .spec = cache2.tensor_spec()},
             TensorParameter{.unique_id = INPUT1, .spec = input1.tensor_spec()},
             TensorParameter{.unique_id = INPUT2, .spec = input2.tensor_spec()},
             TensorParameter{.unique_id = POSITIONS, .spec = positions.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {UPDATE}, .target_nodes = all_cores}},
    };

    KernelRunArgs update_run_args{.kernel = UPDATE};
    for (uint32_t worker_start = 0; worker_start < cores.size(); ++worker_start) {
        AddRuntimeArgsForNode(
            update_run_args.runtime_arg_values,
            cores[worker_start],
            {{"source_rows", source_rows}, {"worker_start", worker_start}, {"worker_stride", num_cores}});
    }

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(update_run_args)},
        .tensor_args =
            {{CACHE1, TensorArgument{cache1.mesh_tensor()}},
             {CACHE2, TensorArgument{cache2.mesh_tensor()}},
             {INPUT1, TensorArgument{input1.mesh_tensor()}},
             {INPUT2, TensorArgument{input2.mesh_tensor()}},
             {POSITIONS, TensorArgument{positions.mesh_tensor()}}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim::indexed_fused_update_cache
