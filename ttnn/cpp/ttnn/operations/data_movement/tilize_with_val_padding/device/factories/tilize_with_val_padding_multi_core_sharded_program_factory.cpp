// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_sharded_program_factory.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingMultiCoreShardedFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    auto pad_value = operation_attributes.pad_value;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName SRC_SHARD_DFB{"src_shard"};  // legacy c_1: sharded input (borrowed)
    const DFBSpecName IN_DFB{"in"};                // legacy c_0: tilize input stream (intermediate)
    const DFBSpecName PAD_DFB{"pad"};              // legacy c_2: reader-local padding-row scratch
    const DFBSpecName OUT_DFB{"out"};              // legacy c_16: sharded output (borrowed)
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    auto input_shard_spec = a.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();
    auto all_cores = output_shard_spec.grid;

    uint32_t num_batches = output.physical_volume() / (output.padded_shape()[-2] * output.padded_shape()[-1]);

    uint32_t num_input_rows = input_shard_spec.shape[0];
    uint32_t input_shard_width_bytes = input_shard_spec.shape[1] * a.element_size();
    uint32_t ntiles_per_core = output_shard_spec.shape[0] * output_shard_spec.shape[1] / TILE_HW;
    uint32_t ntiles_per_batch = ntiles_per_core / num_batches;
    uint32_t ntiles_per_block = output_shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = output_shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_padded_rows = output.padded_shape()[-2] - a.padded_shape()[-2];

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t packed_pad_value = detail::get_packed_value(a, pad_value);

    // ------------------------------------------------------------------------
    // DataflowBuffers. Borrowed DFBs (src_shard / out) map the legacy globally-allocated sharded CBs
    // onto the input/output tensor memory; the in/pad DFBs are reader-local intermediates.
    // ------------------------------------------------------------------------
    DataflowBufferSpec src_shard_dfb_spec{
        .unique_id = SRC_SHARD_DFB,
        .entry_size = input_shard_width_bytes,
        .num_entries = num_input_rows,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = INPUT_TENSOR,
    };
    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = ntiles_per_batch * 2,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec pad_dfb_spec{
        .unique_id = PAD_DFB,
        .entry_size = input_shard_width_bytes,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = ntiles_per_core,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
                "reader_unary_pad_height_width_sharded.cpp"},
        .dfb_bindings =
            {ProducerOf(SRC_SHARD_DFB, "src_shard"),
             ConsumerOf(SRC_SHARD_DFB, "src_shard"),
             ProducerOf(IN_DFB, "in"),
             ProducerOf(PAD_DFB, "pad"),
             ConsumerOf(PAD_DFB, "pad")},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_input_rows",
                  "input_width_bytes",
                  "input_block_size",
                  "num_padded_tiles_per_batch",
                  "num_padded_rows",
                  "num_batches",
                  "packed_pad_value"}},
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
        unpack_to_dest_modes.insert({IN_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_compute_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(IN_DFB, "in"), ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"per_core_block_cnt", nblocks_per_core}, {"per_core_block_tile_cnt", ntiles_per_block}},
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
        reader_run.runtime_arg_values.push_back(
            {node,
             {{"num_input_rows", num_input_rows},
              {"input_width_bytes", input_shard_width_bytes},
              {"input_block_size", (num_input_rows / num_batches) * input_shard_width_bytes},
              {"num_padded_tiles_per_batch", ntiles_per_batch},
              {"num_padded_rows", num_padded_rows},
              {"num_batches", num_batches},
              {"packed_pad_value", packed_pad_value}}});
        writer_run.runtime_arg_values.push_back({node, {{"num_units", ntiles_per_core}}});
    }

    WorkUnitSpec wu{
        .name = "tilize_with_val_padding_multi_core_sharded",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "tilize_with_val_padding_multi_core_sharded",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src_shard_dfb_spec, in_dfb_spec, pad_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_mesh)}}, {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
