// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_sharded_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize/device/tilize_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreShardedProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const uint32_t tile_width = operation_attributes.tile.get_width();
    const uint32_t tile_hw = operation_attributes.tile.get_tile_hw();
    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = operation_attributes.tile.get_tile_size(input_data_format);
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = operation_attributes.tile.get_tile_size(output_data_format);
    bool fp32_llk_acc = input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B ||
                        input.dtype() == DataType::UINT8;

    auto shard_spec = input.shard_spec().value();
    uint32_t num_tiles_per_shard = shard_spec.shape[0] * shard_spec.shape[1] / tile_hw;
    uint32_t num_tiles_per_row = shard_spec.shape[1] / tile_width;
    const CoreRangeSet& all_cores = shard_spec.grid;

    const bool output_is_interleaved = output.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;

    // ---- Metal 2.0 spec resource names (function-local) ----
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    auto* device = input.device();

    // Sharded input DFB — borrowed from the input shard buffer for zero-copy read.
    DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = input_data_format,
        .tile_format_metadata = operation_attributes.tile,
        .borrowed_from = INPUT,
    };

    // Output DFB:
    //   Sharded output  → borrowed from the output shard buffer (zero-copy write); full shard size.
    //   Interleaved output → local DFB sized to one tile-row; writer drains it row-by-row via
    //     TensorAccessor, so the full shard does not need to fit in L1 at once.
    DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_is_interleaved ? num_tiles_per_row : num_tiles_per_shard,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = operation_attributes.tile,
    };
    if (!output_is_interleaved) {
        output_dfb.borrowed_from = OUTPUT;
    }

    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // Reader: sharded unary — the input DFB is borrowed memory, so the reader only handshakes.
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Writer: interleaved scatter (TensorAccessor) or sharded in-place (handshake only).
    KernelSpec writer;
    if (output_is_interleaved) {
        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                "writer_unary_interleaved_start_id_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = OUTPUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = OUTPUT,
                .accessor_name = "dst",
            }},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    } else {
        writer = KernelSpec{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = OUTPUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    }

    ComputeGen1Config compute_cfg;
    compute_cfg.enable_32_bit_dest = fp32_llk_acc;
    if (fp32_llk_acc && input.dtype() != DataType::UINT8) {
        compute_cfg.unpack_modes.emplace(INPUT_DFB, UnpackMode::UnpackToDest);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", num_tiles_per_shard / num_tiles_per_row},
             {"per_core_block_tile_cnt", num_tiles_per_row}},
        .hw_config = ComputeHardwareConfig{compute_cfg},
    };

    ProgramSpec spec{
        .name = "tilize_multi_core_sharded",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {input_dfb, output_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "tilize_sharded",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = all_cores,
        }},
    };

    // ---- Run args ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};

    if (output_is_interleaved) {
        // HEIGHT_SHARDED with ROW_MAJOR orientation: each core's shard maps to a contiguous
        // tile range in the output, so start_id = i * num_tiles_per_shard.
        const auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
        uint32_t tile_start_id = 0;
        for (const auto& core : cores) {
            AddRuntimeArgsForNode(reader_ra.runtime_arg_values, core, {{"num_tiles_per_core", num_tiles_per_shard}});
            AddRuntimeArgsForNode(
                writer_ra.runtime_arg_values, core, {{"num_pages", num_tiles_per_shard}, {"start_id", tile_start_id}});
            tile_start_id += num_tiles_per_shard;
        }
    } else {
        for (const auto& core : corerange_to_cores(all_cores)) {
            AddRuntimeArgsForNode(reader_ra.runtime_arg_values, core, {{"num_tiles_per_core", num_tiles_per_shard}});
            AddRuntimeArgsForNode(writer_ra.runtime_arg_values, core, {{"num_units", num_tiles_per_shard}});
        }
    }

    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs TilizeMultiCoreShardedProgramFactory::override_runtime_arguments(
    const TilizeParams& /*operation_attributes*/,
    const TilizeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The borrowed input DFB and the output binding (borrowed shard buffer, or interleaved
    // TensorAccessor) both refresh their backing address from the tensor args on a cache hit.
    // (This replaces the legacy legacy slot-0 patch + borrowed-address rebuild.)
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    ProgramRunArgs params;
    params.tensor_args = {{INPUT, tensor_args.input_tensor.mesh_tensor()}, {OUTPUT, tensor_return_value.mesh_tensor()}};
    return params;
}

}  // namespace ttnn::prim
