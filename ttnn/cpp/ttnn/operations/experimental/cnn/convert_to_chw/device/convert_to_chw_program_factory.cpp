// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_program_factory.hpp"
#include <limits>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::MathFidelity;
using tt::tt_metal::Precision;
using tt::tt_metal::experimental::ComputeGen1Config;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

ttnn::device_operation::ProgramArtifacts ConvertToCHWProgramFactory::create_program_artifacts(
    const ConvertToCHWParams& /*operation_attributes*/, const Tensor& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    TT_FATAL(
        a.device()->arch() != tt::ARCH::QUASAR,
        "ConvertToCHW uses a Gen1 plain-CB self-loop for its output writer and is not supported on Quasar");

    const auto& input_shape = a.logical_shape();
    const auto input_core_grid = a.shard_spec()->grid;
    const uint32_t num_cores = input_core_grid.num_cores();

    const auto output_shard_shape = output.shard_spec()->shape;

    const auto HW = input_shape[2];
    const auto C = input_shape[3];

    log_debug(tt::LogType::LogOp, "Running op with HW={}, C={}, shard_shape={}", HW, C, a.shard_spec()->shape);

    TT_FATAL(
        C <= tt::constants::TILE_HEIGHT, "C must not exceed TILE_HEIGHT ({}), was {}", tt::constants::TILE_HEIGHT, C);
    TT_FATAL(
        tt::div_up(HW, a.shard_spec()->shape[0]) == num_cores, "Mismatch between core grid and input/shard shapes");

    const uint32_t total_tiles = HW / tt::constants::TILE_HEIGHT;  // C <= TILE_HEIGHT: one tile column.
    const uint32_t total_tiles_per_core = tt::div_up(total_tiles, num_cores);
    TT_FATAL(
        total_tiles_per_core <= std::numeric_limits<uint16_t>::max(),
        "Tiles per core ({}) exceed the device-side DataflowBuffer transaction-count limit (uint16: {})",
        total_tiles_per_core,
        std::numeric_limits<uint16_t>::max());

    log_debug(tt::LogType::LogOp, "Processing {} tiles per core ({} total tiles)", total_tiles_per_core, total_tiles);

    const tt::DataFormat input_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_tile_size = tt::tile_size(input_format);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    const tt::DataFormat output_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t element_size = tt::datum_size(output_format);

    constexpr uint32_t compute_batch_size = 8;  // Must match BATCH_SIZE in the compute and writer kernels.
    constexpr uint32_t transpose_dfb_entries = 2 * compute_batch_size;
    log_debug(
        tt::LogType::LogOp,
        "ConvertToCHW DFB geometry: input={}x{}B, output={}x{}B, transpose={}x{}B",
        total_tiles_per_core,
        input_tile_size,
        output_shard_shape[0],
        output_shard_shape[1] * element_size,
        transpose_dfb_entries,
        intermediary_tile_size);

    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName TRANSPOSE_DFB{"transpose"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};
    const KernelSpecName WRITER{"writer"};

    const DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_tile_size,
        .num_entries = total_tiles_per_core,
        .data_format_metadata = input_format,
        .borrowed_from = INPUT,
    };
    // Retain two compute batches for double buffering.
    const DataflowBufferSpec transpose_dfb{
        .unique_id = TRANSPOSE_DFB,
        .entry_size = intermediary_tile_size,
        .num_entries = transpose_dfb_entries,
        .data_format_metadata = intermediary_format,
    };
    const DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_shard_shape[1] * element_size,
        .num_entries = output_shard_shape[0],
        .data_format_metadata = output_format,
        .borrowed_from = OUTPUT,
    };

    const TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    const KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/reader_convert_to_chw.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.common_runtime_arg_names = {"tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };
    const KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/convert_to_chw.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSE_DFB,
                 .accessor_name = "transpose",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.common_runtime_arg_names = {"tiles_per_core"}},
        // Legacy fp32_dest_acc_en, dst_full_sync_en, and bfp8_pack_precise were false; Metalium 2.0 defaults
        // preserve those settings.
        .hw_config =
            ComputeGen1Config{
                .fpu_math_fidelity = MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,
            },
    };
    const KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw/device/kernels/writer_convert_to_chw.cpp",
        // The Gen1 plain-CB output writer binds OUTPUT_DFB as both producer and consumer (a self-loop).
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = TRANSPOSE_DFB,
                 .accessor_name = "transpose",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = {{"channels", C}},
        .runtime_arg_schema = {.common_runtime_arg_names = {"tiles_per_core"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    KernelRunArgs reader_run{.kernel = READER, .common_runtime_arg_values = {{"tiles_per_core", total_tiles_per_core}}};
    KernelRunArgs compute_run{
        .kernel = COMPUTE, .common_runtime_arg_values = {{"tiles_per_core", total_tiles_per_core}}};
    KernelRunArgs writer_run{.kernel = WRITER, .common_runtime_arg_values = {{"tiles_per_core", total_tiles_per_core}}};

    ProgramSpec spec{
        .name = "convert_to_chw",
        .kernels = {reader, compute, writer},
        // Preserve the legacy physical CB-slot order: input=c0, output=c1, transpose=c2.
        .dataflow_buffers = {input_dfb, output_dfb, transpose_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "convert_to_chw",
            .kernels = {READER, COMPUTE, WRITER},
            .target_nodes = input_core_grid,
        }},
    };
    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run), std::move(compute_run), std::move(writer_run)},
        .tensor_args = {{INPUT, a.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
