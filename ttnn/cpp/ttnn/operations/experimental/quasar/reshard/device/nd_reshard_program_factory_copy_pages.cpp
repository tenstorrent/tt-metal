// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/reshard/device/nd_reshard_program_factory_copy_pages.hpp"

#include <filesystem>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// DFB / tensor / kernel names for the copy-pages factory.
// Prefixed with the factory role to avoid unity-build anon-namespace collisions
// with the other reshard factories.
const DFBSpecName COPY_PAGES_CB{"copy_pages_cb"};
const TensorParamName COPY_PAGES_INPUT{"copy_pages_input"};
const TensorParamName COPY_PAGES_OUTPUT{"copy_pages_output"};
const KernelSpecName COPY_PAGES_READER{"copy_pages_reader"};
const KernelSpecName COPY_PAGES_WRITER{"copy_pages_writer"};

}  // namespace

ttnn::device_operation::ProgramArtifacts NdReshardCopyPagesFactory::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    const auto& input_mt = input.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    auto* input_buffer = input.buffer();

    auto input_nd_shard_spec = input.memory_config().nd_shard_spec().value();

    auto aligned_page_size = input_buffer->aligned_page_size();

    // Create grid + cores
    auto grid_size = input.device()->compute_with_storage_grid_size();
    auto grid = CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1))});
    auto cores = corerange_to_cores(grid, std::nullopt, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    // DFB entry/page format
    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    constexpr uint32_t num_tiles_in_cb = 1;  // TODO: Try double buffering

    // Dataflow buffer (real FIFO: reader produces a page, writer consumes it).
    DataflowBufferSpec copy_pages_dfb{
        .unique_id = COPY_PAGES_CB,
        .entry_size = aligned_page_size,
        .num_entries = num_tiles_in_cb,
        .data_format_metadata = data_format,
    };

    // Tensor parameters (replace the legacy TensorAccessorArgs CTA plumbing +
    // Buffer-base CRTA bindings).
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = COPY_PAGES_INPUT, .spec = input_mt.tensor_spec()},
        TensorParameter{.unique_id = COPY_PAGES_OUTPUT, .spec = output_mt.tensor_spec()},
    };

    // Kernels
    KernelSpec reader{
        .unique_id = COPY_PAGES_READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = COPY_PAGES_CB, .accessor_name = "cb", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = COPY_PAGES_INPUT, .accessor_name = "src"},
            },
        .compile_time_args =
            {
                {"page_size", aligned_page_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_page", "end_page"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };

    KernelSpec writer{
        .unique_id = COPY_PAGES_WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = COPY_PAGES_CB, .accessor_name = "cb", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = COPY_PAGES_OUTPUT, .accessor_name = "dst"},
            },
        .compile_time_args =
            {
                {"page_size", aligned_page_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_page", "end_page"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };

    // Work split: per-core page ranges.
    uint32_t num_dev_pages =
        static_cast<uint32_t>(input_buffer->buffer_distribution_spec()->tensor_shape_in_pages().volume());
    uint32_t n_pages_per_core = num_dev_pages / static_cast<uint32_t>(cores.size());
    uint32_t remainder = num_dev_pages % static_cast<uint32_t>(cores.size());

    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = COPY_PAGES_READER};
    ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = COPY_PAGES_WRITER};

    uint32_t start_page = 0;
    for (const auto& core : cores) {
        uint32_t num_pages_for_core = n_pages_per_core;
        if (remainder > 0) {
            num_pages_for_core++;
            remainder--;
        }
        const uint32_t end_page = start_page + num_pages_for_core;
        reader_run_args.runtime_arg_values.push_back(ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs{
            .node = core, .args = {{"start_page", start_page}, {"end_page", end_page}}});
        writer_run_args.runtime_arg_values.push_back(ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs{
            .node = core, .args = {{"start_page", start_page}, {"end_page", end_page}}});
        start_page = end_page;
    }

    ProgramSpec spec{
        .name = "nd_reshard_copy_pages",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(copy_pages_dfb)},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{.name = "wu", .kernels = {COPY_PAGES_READER, COPY_PAGES_WRITER}, .target_nodes = grid},
            },
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {COPY_PAGES_INPUT, input_mt},
                {COPY_PAGES_OUTPUT, output_mt},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
