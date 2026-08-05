// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "tt-metalium/host_api.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts NdReshardCopyPagesFactory::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();

    auto input_nd_shard_spec = input.memory_config().nd_shard_spec().value();

    auto aligned_page_size = input_buffer->aligned_page_size();

    // Create grid + cores
    auto grid_size = input.device()->compute_with_storage_grid_size();
    auto grid = CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))});
    auto cores = corerange_to_cores(grid, std::nullopt, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    // Create Dataflow Buffer (staging FIFO: reader produces a page, writer consumes it)
    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    constexpr uint32_t num_tiles_in_cb = 1;  // TODO: Try double buffering

    // Metal 2.0 resource names.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName STAGING{"staging"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    DataflowBufferSpec staging_dfb{
        .unique_id = STAGING,
        .entry_size = aligned_page_size,
        .num_entries = num_tiles_in_cb,
        .data_format_metadata = data_format,
    };

    // Create kernels. The buffer base address is delivered by the TensorBinding (no address RTA/CRTA);
    // aligned_page_size stays a named CTA (the byte count the kernel hands to the NoC transfer, not a
    // TensorAccessor 3rd argument).
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = STAGING, .accessor_name = "staging", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"page_size", aligned_page_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_page", "end_page"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = STAGING, .accessor_name = "staging", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"page_size", aligned_page_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_page", "end_page"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };

    // Per-core unique runtime args: [start_page, end_page]. Keep the legacy node-first loop; the
    // AddRuntimeArgsForNode helper transposes it into the name-first table ProgramRunArgs expects.
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    uint32_t start_page = 0;
    uint32_t num_dev_pages =
        static_cast<uint32_t>(input_buffer->buffer_distribution_spec()->tensor_shape_in_pages().volume());
    uint32_t n_pages_per_core = num_dev_pages / static_cast<uint32_t>(cores.size());
    uint32_t remainder = num_dev_pages % static_cast<uint32_t>(cores.size());

    for (const auto& core : cores) {
        uint32_t num_pages_for_core = n_pages_per_core;
        if (remainder > 0) {
            num_pages_for_core++;
            remainder--;
        }
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_page", start_page}, {"end_page", start_page + num_pages_for_core}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"start_page", start_page}, {"end_page", start_page + num_pages_for_core}});
        start_page += num_pages_for_core;
    }

    ProgramSpec spec{
        .name = "nd_reshard_copy_pages",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(staging_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input.mesh_tensor().tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output.mesh_tensor().tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = grid,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args.insert({INPUT, input.mesh_tensor()});
    run_args.tensor_args.insert({OUTPUT, output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
