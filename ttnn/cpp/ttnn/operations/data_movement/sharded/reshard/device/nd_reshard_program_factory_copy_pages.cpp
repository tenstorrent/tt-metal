// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

constexpr const char* kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_reader.cpp";
constexpr const char* kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/kernels/nd_reshard_copy_pages_writer.cpp";

// Resource / parameter names referenced by the kernel sources (ta::/dfb:: accessors).
constexpr const char* kInputTensorParam = "input";
constexpr const char* kOutputTensorParam = "output";
constexpr const char* kDfbName = "reshard_dfb";

}  // namespace

ttnn::device_operation::ProgramArtifacts NdReshardCopyPagesFactory::create_program_artifacts(
    const ReshardParams& /*operation_attributes*/, const ReshardInputs& tensor_args, Tensor& output_tensor) {
    const auto& input = tensor_args.input;
    auto& output = output_tensor;

    auto* input_buffer = input.buffer();

    const auto input_nd_shard_spec = input.memory_config().nd_shard_spec().value();
    const uint32_t aligned_page_size = input_buffer->aligned_page_size();

    // Grid + cores: mirror the legacy factory's full compute grid + page split.
    const auto grid_size = input.device()->compute_with_storage_grid_size();
    const CoreRange grid_range(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1));
    const CoreRangeSet grid({grid_range});
    const bool row_wise = input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(grid, std::nullopt, row_wise);

    // DFB: a small software FIFO carrying one page per entry between reader (producer)
    // and writer (consumer). Double-buffered (legacy used a single-entry CB; 2 entries
    // is the minimal pipelined depth used by the Metal 2.0 reference flows).
    constexpr uint32_t num_dfb_entries = 2;

    // ------------------------------------------------------------------
    // ProgramSpec (immutable)
    // ------------------------------------------------------------------
    ProgramSpec spec;
    spec.name = "nd_reshard_copy_pages";

    // Reader (producer): input tensor -> DFB, addressed via TensorAccessor(ta::input).
    KernelSpec reader{
        .unique_id = KernelSpecName{"reader"},
        .source = std::filesystem::path(kReaderKernelPath),
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };
    reader.dfb_bindings.push_back(ProducerOf(DFBSpecName{kDfbName}, kDfbName));
    reader.tensor_bindings.push_back(
        TensorBinding{.tensor_parameter_name = TensorParamName{kInputTensorParam}, .accessor_name = kInputTensorParam});
    // Per-node positional varargs: [0]=start_page, [1]=end_page.
    reader.advanced_options.num_runtime_varargs = 2;

    // Writer (consumer): DFB -> output tensor, addressed via TensorAccessor(ta::output).
    KernelSpec writer{
        .unique_id = KernelSpecName{"writer"},
        .source = std::filesystem::path(kWriterKernelPath),
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };
    writer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{kDfbName}, kDfbName));
    writer.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = TensorParamName{kOutputTensorParam}, .accessor_name = kOutputTensorParam});
    writer.advanced_options.num_runtime_varargs = 2;

    DataflowBufferSpec dfb{
        .unique_id = DFBSpecName{kDfbName},
        .entry_size = aligned_page_size,
        .num_entries = num_dfb_entries,
        .data_format_metadata = datatype_to_dataformat_converter(input.dtype()),
    };

    spec.kernels = {reader, writer};
    spec.dataflow_buffers = {dfb};
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TensorParamName{kInputTensorParam}, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = TensorParamName{kOutputTensorParam}, .spec = output.tensor_spec()},
    };
    spec.work_units = {WorkUnitSpec{
        .name = "reshard_work_unit",
        .kernels = {KernelSpecName{"reader"}, KernelSpecName{"writer"}},
        .target_nodes = grid,
    }};

    // ------------------------------------------------------------------
    // ProgramRunArgs (mutable)
    // ------------------------------------------------------------------
    const uint32_t num_dev_pages =
        static_cast<uint32_t>(input_buffer->buffer_distribution_spec()->tensor_shape_in_pages().volume());
    const uint32_t n_pages_per_core = num_dev_pages / static_cast<uint32_t>(cores.size());
    uint32_t remainder = num_dev_pages % static_cast<uint32_t>(cores.size());

    KernelRunArgs reader_run_args{.kernel = KernelSpecName{"reader"}};
    KernelRunArgs writer_run_args{.kernel = KernelSpecName{"writer"}};

    uint32_t start_page = 0;
    for (const auto& core : cores) {
        uint32_t num_pages_for_core = n_pages_per_core;
        if (remainder > 0) {
            num_pages_for_core++;
            remainder--;
        }
        const uint32_t end_page = start_page + num_pages_for_core;
        const NodeCoord node{core.x, core.y};
        reader_run_args.advanced_options.runtime_varargs.emplace(
            node, AdvancedKernelRunArgs::Varargs{start_page, end_page});
        writer_run_args.advanced_options.runtime_varargs.emplace(
            node, AdvancedKernelRunArgs::Varargs{start_page, end_page});
        start_page = end_page;
    }

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_params.tensor_args = {
        {TensorParamName{kInputTensorParam}, TensorArgument{input.mesh_tensor()}},
        {TensorParamName{kOutputTensorParam}, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
