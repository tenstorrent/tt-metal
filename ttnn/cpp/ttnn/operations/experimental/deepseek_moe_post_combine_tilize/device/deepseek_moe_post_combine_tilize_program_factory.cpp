// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_device_operation.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

ttnn::device_operation::ProgramArtifacts
DeepseekMoEPostCombineTilizeDeviceOperation::DeepseekMoEPostCombineTilizeProgramFactory::create_program_artifacts(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& tensor_return_value) {
    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName TILIZE_INPUT{"tilize_input"};
    const DFBSpecName TILIZE_OUTPUT{"tilize_output"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    /*
     * Tensors
     */
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    uint32_t input_row_page_size = static_cast<uint32_t>(input_tensor.buffer()->page_size());
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    const ttnn::Tensor& output_tensor = tensor_return_value;
    const uint32_t output_tile_page_size = static_cast<uint32_t>(output_tensor.buffer()->page_size());
    const auto& output_shape = output_tensor.padded_shape();

    const auto& input = input_tensor.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();

    /*
     * Shard spec
     */
    const auto output_nd_shard_spec = output_tensor.memory_config().nd_shard_spec().value();
    const uint32_t output_shard_width = output_nd_shard_spec.shard_shape[-1];
    const uint32_t output_shard_width_tiles = output_shard_width / tt::constants::TILE_WIDTH;
    const uint32_t output_shard_width_bytes = output_shard_width * output_tensor.element_size();

    const CoreRangeSet op_cores = output_nd_shard_spec.grid;

    uint32_t upper_dims = 1;
    for (uint32_t dim = 0; dim < input_rank - 1; ++dim) {
        upper_dims *= input_shape[dim];
    }

    const uint32_t output_num_shards_wide = output_shape[-1] / output_shard_width;
    const uint32_t output_num_shards_high = upper_dims / output_nd_shard_spec.shard_shape[-2];

    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    /*
     * Tensor parameters
     */
    const TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    // The output is reached only as the backing memory of the tilize output DFB (borrowed_from below).
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    /*
     * DFBs
     */
    const DataflowBufferSpec tilize_input_dfb{
        .unique_id = TILIZE_INPUT,
        .entry_size = output_shard_width_bytes,
        .num_entries = tt::constants::TILE_HEIGHT,
        .data_format_metadata = data_format,
    };

    // The output DFB is backed by the sharded output tensor. Borrowing it from the output
    // TensorParameter (never an address) is what lets the framework re-peg it on a cache hit.
    const DataflowBufferSpec tilize_output_dfb{
        .unique_id = TILIZE_OUTPUT,
        .entry_size = output_tile_page_size,
        .num_entries = output_shard_width_tiles,
        .data_format_metadata = data_format,
        .borrowed_from = OUTPUT,
    };

    /*
     * Kernels
     */

    // reader
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
            "deepseek_moe_post_combine_tilize_reader.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O2},
        .dfb_bindings = {{
            .dfb_spec_name = TILIZE_INPUT,
            .accessor_name = "tilize_input",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .compile_time_args =
            {
                {"input_row_page_size", input_row_page_size},
                {"bytes_to_read_per_row", output_shard_width_bytes},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"intra_row_byte_offset", "row_page_offset"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // compute
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
            "deepseek_moe_post_combine_tilize_compute.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                {
                    .dfb_spec_name = TILIZE_INPUT,
                    .accessor_name = "tilize_input",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                {
                    .dfb_spec_name = TILIZE_OUTPUT,
                    .accessor_name = "tilize_output",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"num_tiles", output_shard_width_tiles},
            },
        .hw_config = ComputeHardwareConfig{},
    };

    // writer
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
            "deepseek_moe_post_combine_tilize_writer.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O2},
        .dfb_bindings = {{
            .dfb_spec_name = TILIZE_OUTPUT,
            .accessor_name = "tilize_output",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .compile_time_args =
            {
                {"num_tiles", output_shard_width_tiles},
            },
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelRunArgs reader_run_args{.kernel = READER};

    bool is_row_major_shard_orientation = output_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    std::vector<tt::tt_metal::CoreCoord> cores =
        corerange_to_cores(op_cores, std::nullopt, is_row_major_shard_orientation);
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        // reader
        uint32_t intra_row_byte_offset;
        uint32_t row_page_offset;
        if (is_row_major_shard_orientation) {
            intra_row_byte_offset = (i % output_num_shards_wide) * output_shard_width_bytes;
            row_page_offset = (i / output_num_shards_wide) * tt::constants::TILE_HEIGHT;
        } else {
            intra_row_byte_offset = (i / output_num_shards_high) * output_shard_width_bytes;
            row_page_offset = (i % output_num_shards_high) * tt::constants::TILE_HEIGHT;
        }
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"intra_row_byte_offset", intra_row_byte_offset}, {"row_page_offset", row_page_offset}});
    }

    ProgramSpec spec{
        .name = "deepseek_moe_post_combine_tilize",
        .kernels = {std::move(reader), std::move(compute), std::move(writer)},
        .dataflow_buffers = {tilize_input_dfb, tilize_output_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "deepseek_moe_post_combine_tilize",
            .kernels = {READER, COMPUTE, WRITER},
            .target_nodes = op_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args)};
    run_args.tensor_args = {{INPUT, input}, {OUTPUT, output}};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim
