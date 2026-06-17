// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_width_sharded_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreWidthShardedProgramFactory::create_program_artifacts(
    const TilizeParams& /*operation_attributes*/, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;

    const auto& input_tensor = input.mesh_tensor();
    const auto& output_tensor = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName SRC_DFB{"src"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / TILE_HW;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / TILE_WIDTH;
    const CoreRangeSet& all_cores = shard_spec.grid;

    // ------------------------------------------------------------------------
    // Borrowed DataflowBuffers (legacy CBs c_0 / c_16 allocated onto the input/output buffers).
    // ------------------------------------------------------------------------
    DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT_TENSOR,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_tensor.tensor_spec()};

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "reader_unary_sharded_metal2.cpp"},
        .dfb_bindings = {ProducerOf(SRC_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_sharded_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_modes.insert({SRC_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_compute_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(SRC_DFB, "in"), ProducerOf(OUT_DFB, "out")},
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles_per_shard / num_tiles_per_row},
             {"per_core_block_tile_cnt", num_tiles_per_row}},
        .hw_config =
            ComputeHardwareConfig{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = unpack_to_dest_modes,
            },
    };

    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    for (const auto& core : corerange_to_cores(all_cores)) {
        const NodeCoord node = core;
        reader_run.runtime_arg_values.push_back({node, {{"num_tiles_per_core", num_tiles_per_shard}}});
        writer_run.runtime_arg_values.push_back({node, {{"num_units", num_tiles_per_shard}}});
    }

    WorkUnitSpec wu{
        .name = "tilize_multi_core_width_sharded",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "tilize_multi_core_width_sharded",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor)}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
