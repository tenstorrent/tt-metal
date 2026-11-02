// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp"

#include <filesystem>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "tt-metalium/host_api.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// (Names are prefixed to avoid unity-build collisions with the sibling reshard factories.)
constexpr const char* kCPReaderKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp";
constexpr const char* kCPWriterKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp";

// Resource / parameter names referenced by the kernel sources (tensor:: / dfb:: accessors).
constexpr const char* kCPInputTensorParam = "input";
constexpr const char* kCPOutputTensorParam = "output";
constexpr const char* kCPPageDfbName = "page";

constexpr const char* kCPReaderKernel = "reader";
constexpr const char* kCPWriterKernel = "writer";

}  // namespace

ttnn::device_operation::ProgramArtifacts NdReshardCopyPagesFactory::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();

    auto input_nd_shard_spec = input.memory_config().nd_shard_spec().value();

    const uint32_t aligned_page_size = static_cast<uint32_t>(input_buffer->aligned_page_size());

    // Create grid + cores
    auto grid_size = input.device()->compute_with_storage_grid_size();
    auto grid = CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))});
    auto cores = corerange_to_cores(grid, std::nullopt, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    constexpr uint32_t num_tiles_in_dfb = 1;  // TODO: Try double buffering

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "nd_reshard_copy_pages";

    const KernelSpec::CompileTimeArgs compile_time_args = {
        {"page_size", aligned_page_size},
    };

    spec.kernels = {
        KernelSpec{
            .unique_id = KernelSpecName{kCPReaderKernel},
            .source = std::filesystem::path(kCPReaderKernelPath),
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{kCPPageDfbName},
                .accessor_name = kCPPageDfbName,
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = TensorParamName{kCPInputTensorParam}, .accessor_name = kCPInputTensorParam}},
            .compile_time_args = compile_time_args,
            .runtime_arg_schema = {.runtime_arg_names = {"start_page", "end_page"}},
            .hw_config = ttnn::create_reader_datamovement_config(),
        },
        KernelSpec{
            .unique_id = KernelSpecName{kCPWriterKernel},
            .source = std::filesystem::path(kCPWriterKernelPath),
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = DFBSpecName{kCPPageDfbName},
                .accessor_name = kCPPageDfbName,
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = TensorParamName{kCPOutputTensorParam}, .accessor_name = kCPOutputTensorParam}},
            .compile_time_args = compile_time_args,
            .runtime_arg_schema = {.runtime_arg_names = {"start_page", "end_page"}},
            .hw_config = ttnn::create_writer_datamovement_config(),
        },
    };

    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = DFBSpecName{kCPPageDfbName},
        .entry_size = aligned_page_size,
        .num_entries = num_tiles_in_dfb,
        .data_format_metadata = data_format,
    }};

    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kCPInputTensorParam}, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kCPOutputTensorParam}, .spec = output.tensor_spec()},
    };

    spec.work_units = {WorkUnitSpec{
        .name = "nd_reshard_copy_pages_work_unit",
        .kernels = {KernelSpecName{kCPReaderKernel}, KernelSpecName{kCPWriterKernel}},
        .target_nodes = grid,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    // Per-core unique runtime args: [start_page, end_page]
    KernelRunArgs reader_run_args{.kernel = KernelSpecName{kCPReaderKernel}};
    KernelRunArgs writer_run_args{.kernel = KernelSpecName{kCPWriterKernel}};

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
            reader_run_args.runtime_arg_values,
            core,
            {{"start_page", start_page}, {"end_page", start_page + num_pages_for_core}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"start_page", start_page}, {"end_page", start_page + num_pages_for_core}});
        start_page += num_pages_for_core;
    }

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_params.tensor_args = {
        {TensorParamName{kCPInputTensorParam}, TensorArgument{input.mesh_tensor()}},
        {TensorParamName{kCPOutputTensorParam}, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
