// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_program_factory.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include <algorithm>
#include <vector>

namespace ttnn::experimental::prim {

using namespace tt::constants;

namespace {
// Names of the program's kernels, dataflow buffers and tensor parameters, shared by
// create_program_artifacts and override_runtime_arguments. Nested so they cannot collide with the
// convert_to_hwc factory's names in the unity build (both ops' factories compile into one
// translation unit).
namespace chw {
const tt::tt_metal::experimental::KernelSpecName READER{"reader"};
const tt::tt_metal::experimental::KernelSpecName WRITER{"writer"};
const tt::tt_metal::experimental::KernelSpecName COMPUTE{"compute"};
const tt::tt_metal::experimental::DFBSpecName IN{"in"};
const tt::tt_metal::experimental::DFBSpecName OUT{"out"};
const tt::tt_metal::experimental::DFBSpecName IN_TRANSPOSE{"in_transpose"};
const tt::tt_metal::experimental::TensorParamName INPUT{"input"};
const tt::tt_metal::experimental::TensorParamName OUTPUT{"output"};
}  // namespace chw

// Helper function to set runtime arguments for reader, writer, and compute kernels
void set_runtime_args_for_all_kernels(
    tt::tt_metal::experimental::KernelRunArgs& reader_run_args,
    tt::tt_metal::experimental::KernelRunArgs& writer_run_args,
    tt::tt_metal::experimental::KernelRunArgs& compute_run_args,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    uint32_t total_tiles_per_core) {
    using tt::tt_metal::experimental::AddRuntimeArgsForNode;
    std::for_each(cores.cbegin(), cores.cend(), [&](const tt::tt_metal::CoreCoord& core) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"total_tiles", total_tiles_per_core}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"total_tiles", total_tiles_per_core}});
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"total_tiles", total_tiles_per_core}});
    });
}
}  // namespace

ttnn::device_operation::ProgramArtifacts ConvertToCHWProgramFactory::create_program_artifacts(
    const ConvertToCHWParams& /*operation_attributes*/, const Tensor& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::tt_metal::experimental;
    using namespace chw;

    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    const auto& input_shape = a.logical_shape();
    const auto input_core_grid = a.shard_spec()->grid;
    const auto input_cores = corerange_to_cores(
        input_core_grid, std::nullopt, a.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    const auto output_shard_shape = output.shard_spec()->shape;

    const auto HW = input_shape[2];
    const auto C = input_shape[3];

    log_debug(tt::LogType::LogOp, "Running op with HW={}, C={}, shard_shape={}", HW, C, a.shard_spec()->shape);

    TT_FATAL(C <= TILE_HEIGHT, "C must not exceed 32");
    TT_FATAL(
        tt::div_up(HW, a.shard_spec()->shape[0]) == input_cores.size(),
        "Mismatch between core grid and input/shard shapes");

    const uint32_t total_tiles = HW / TILE_HEIGHT;  // assume C < 32
    const uint32_t total_tiles_per_core = tt::div_up(total_tiles, input_cores.size());

    log_debug(tt::LogType::LogOp, "Processing {} tiles per core ({} total tiles)", total_tiles_per_core, total_tiles);

    auto* input_buffer = a.buffer();
    auto* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr, "Input buffer must be allocated on device");
    TT_FATAL(output_buffer != nullptr, "Output buffer must be allocated on device");

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    ProgramSpec spec{.name = "convert_to_chw"};

    spec.tensor_parameters.push_back(TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()});

    // The input and output buffers are built on the io tensors' own sharded L1 memory rather than on
    // Program-lifetime storage; override_runtime_arguments re-points them on a cache hit.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN,
        .entry_size = input_tile_size,
        .num_entries = total_tiles_per_core,
        .data_format_metadata = input_format,
        .borrowed_from = INPUT,
    });

    const tt::DataFormat output_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t element_size = tt::datum_size(output_format);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT,
        .entry_size = output_shard_shape[1] * element_size,
        .num_entries = output_shard_shape[0],
        .data_format_metadata = output_format,
        .borrowed_from = OUTPUT,
    });

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN_TRANSPOSE,
        .entry_size = intermediary_tile_size,
        .num_entries = 16,
        .data_format_metadata = intermediary_format,
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/reader_convert_to_chw.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"total_tiles"}},
        .hw_config = create_reader_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/writer_convert_to_chw.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN_TRANSPOSE,
                    .accessor_name = "in_transpose",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // The writer is the output buffer's only user: it fills the resident output shard and
                // nothing drains it, so it is bound as both producer and consumer.
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args = {{"C", C}},
        .runtime_arg_schema = {.runtime_arg_names = {"total_tiles"}},
        .hw_config = create_writer_datamovement_config(),
    });

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/convert_to_chw.cpp",
        // A ComputeConfigDescriptor with no opt_level builds at O3, while a KernelSpec defaults to O2,
        // so the level is stated explicitly.
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = IN_TRANSPOSE,
                    .accessor_name = "transpose_in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"total_tiles"}},
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = tt::tt_metal::Precision::Precise,
                .enable_32_bit_dest = false,
                .double_buffer_dest = true,
            },
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = input_core_grid,
    });

    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    // Set initial runtime args
    set_runtime_args_for_all_kernels(
        reader_run_args, writer_run_args, compute_run_args, input_cores, total_tiles_per_core);

    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args.emplace(INPUT, input_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs ConvertToCHWProgramFactory::override_runtime_arguments(
    const ConvertToCHWParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace tt::tt_metal::experimental;
    using namespace chw;

    // buffers are not provided so we take them from output/input tensors
    auto* input_buffer = tensor_args.buffer();
    auto* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr, "Input buffer must be allocated on device");
    TT_FATAL(output_buffer != nullptr, "Output buffer must be allocated on device");

    // Tile count is keyed by the tensor spec; only the sharded buffer addresses move, and both ride on
    // the borrowed dataflow buffers (in, out), which re-resolve their L1 address from these tensor
    // arguments.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(INPUT, tensor_args.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());
    return run_args;
}

}  // namespace ttnn::experimental::prim
