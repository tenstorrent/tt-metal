// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts NdReshardCopyPagesFactory::create_program_spec(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();

    auto input_nd_shard_spec = input.memory_config().nd_shard_spec().value();

    uint32_t aligned_page_size = static_cast<uint32_t>(input_buffer->aligned_page_size());

    // Create grid + cores
    auto grid_size = input.device()->compute_with_storage_grid_size();
    auto grid = CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))});
    auto cores = corerange_to_cores(grid, std::nullopt, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    // Create Dataflow Buffer (was Circular Buffer)
    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    constexpr uint32_t num_tiles_in_cb = 1;  // TODO: Try double buffering

    const TensorParamName kInput{"input"};
    const TensorParamName kOutput{"output"};
    const DFBSpecName kCbIn0{"cb_in0"};
    const KernelSpecName kReader{"reader"};
    const KernelSpecName kWriter{"writer"};

    ProgramSpec spec;
    spec.name = "nd_reshard_copy_pages";

    // Tensor parameters: input (read by reader) and output (written by writer).
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kInput, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = kOutput, .spec = output.tensor_spec()});

    // Dataflow buffer: reader produces a page, writer consumes it.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = kCbIn0,
        .entry_size = aligned_page_size,
        .num_entries = num_tiles_in_cb,
        .data_format_metadata = data_format,
    });

    // Reader kernel: reads pages from the input tensor, pushes them into the DFB.
    KernelSpec reader;
    reader.unique_id = kReader;
    reader.source = std::filesystem::path(
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/"
        "nd_reshard_copy_pages_reader.cpp");
    reader.compile_time_args = {{"page_size", aligned_page_size}};
    reader.runtime_arg_schema.runtime_arg_names = {"start_page", "end_page"};
    reader.dfb_bindings = {DFBBinding{
        .dfb_spec_name = kCbIn0,
        .accessor_name = "cb_in0",
        .endpoint_type = DFBEndpointType::PRODUCER,
    }};
    reader.tensor_bindings = {TensorBinding{.tensor_parameter_name = kInput, .accessor_name = "input"}};
    reader.hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER};

    // Writer kernel: waits on pages from the DFB, writes them to the output tensor.
    KernelSpec writer;
    writer.unique_id = kWriter;
    writer.source = std::filesystem::path(
        "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/"
        "nd_reshard_copy_pages_writer.cpp");
    writer.compile_time_args = {{"page_size", aligned_page_size}};
    writer.runtime_arg_schema.runtime_arg_names = {"start_page", "end_page"};
    writer.dfb_bindings = {DFBBinding{
        .dfb_spec_name = kCbIn0,
        .accessor_name = "cb_in0",
        .endpoint_type = DFBEndpointType::CONSUMER,
    }};
    writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = kOutput, .accessor_name = "output"}};
    writer.hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER};

    spec.kernels.push_back(std::move(reader));
    spec.kernels.push_back(std::move(writer));

    // Single WorkUnitSpec hosting both kernels on the full grid: the DFB's producer (reader)
    // and consumer (writer) share the same nodes (local-DFB invariant).
    spec.work_units.push_back(WorkUnitSpec{
        .name = "wu",
        .kernels = {kReader, kWriter},
        .target_nodes = grid,
    });

    // Run args.
    ProgramRunArgs run_args;

    KernelRunArgs reader_run;
    reader_run.kernel = kReader;
    KernelRunArgs writer_run;
    writer_run.kernel = kWriter;

    // Per-core unique runtime args: [start_page, end_page]
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
        uint32_t end_page = start_page + num_pages_for_core;
        reader_run.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"start_page", start_page}, {"end_page", end_page}}});
        writer_run.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"start_page", start_page}, {"end_page", end_page}}});
        start_page += num_pages_for_core;
    }

    run_args.kernel_run_args.push_back(std::move(reader_run));
    run_args.kernel_run_args.push_back(std::move(writer_run));

    // Tensor arguments: reference the MeshTensors reachable from the factory io tensors
    // (matched back by pointer identity by the framework adapter).
    run_args.tensor_args = {
        {kInput, input.mesh_tensor()},
        {kOutput, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
