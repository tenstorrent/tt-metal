// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_tiled_sharded_input_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/padded_slice/device/padded_slice_utils.hpp"

namespace ttnn::experimental::prim {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::corerange_to_cores;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::datatype_to_dataformat_converter;
using tt::tt_metal::Layout;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::TensorMemoryLayout;
using ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor;

using tt::tt_metal::experimental::AdvancedKernelRunArgs;
using tt::tt_metal::experimental::ConsumerOf;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProducerOf;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorArgument;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {

const TensorParamName TILE_SHARDED_INPUT{"input"};
const TensorParamName TILE_SHARDED_OUTPUT{"output"};
const DFBSpecName TILE_SHARDED_INPUT_DFB{"input"};
const KernelSpecName TILE_SHARDED_READER{"reader"};
const KernelSpecName TILE_SHARDED_WRITER{"writer"};

SliceWriteRuntimeArgs get_slice_write_runtime_args_tiled_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores) {
    auto actual_input_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < actual_input_shape.rank(); ++i) {
        actual_input_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }
    const auto output_shape = output_tensor.padded_shape();
    const uint32_t tile_size_bytes = tt::tile_size(datatype_to_dataformat_converter(input_tensor.dtype()));
    const auto shard_spec = input_tensor.shard_spec().value();
    const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input_tensor);

    const uint32_t num_dims = static_cast<uint32_t>(actual_input_shape.rank());
    std::vector<uint32_t> num_output_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_input_tiles_per_dim(num_dims);
    std::vector<uint32_t> accumulated_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> size_till_end(num_dims);

    num_input_tiles_per_dim[0] = tt::div_up(actual_input_shape[-1], TILE_WIDTH * num_cores_channels);
    num_input_tiles_per_dim[1] = tt::div_up(actual_input_shape[-2], TILE_HEIGHT);
    num_output_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH) - num_input_tiles_per_dim[0];
    num_output_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) - num_input_tiles_per_dim[1];
    num_output_tiles_per_dim[1] *= tt::div_up(output_shape[-1], TILE_WIDTH);

    const uint32_t num_tiles_per_channel = num_input_tiles_per_dim[0];
    accumulated_total_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH);
    accumulated_total_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) * accumulated_total_tiles_per_dim[0];
    const uint32_t output_channel_tiles = accumulated_total_tiles_per_dim[0];
    accumulated_input_total_tiles_per_dim[0] = num_input_tiles_per_dim[0];
    accumulated_input_total_tiles_per_dim[1] = num_input_tiles_per_dim[1] * accumulated_input_total_tiles_per_dim[0];
    for (uint32_t i = 2; i < num_dims; ++i) {
        const uint32_t num_unpadded_dim = actual_input_shape[-(i + 1)];
        const uint32_t num_total_dim = output_shape[-(i + 1)];
        const uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_tiles_per_dim[i - 1];
        num_input_tiles_per_dim[i] = num_unpadded_dim;
        num_output_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_tiles_per_dim[i] = num_total_dim * accumulated_total_tiles_per_dim[i - 1];
        accumulated_input_total_tiles_per_dim[i] = num_unpadded_dim * accumulated_input_total_tiles_per_dim[i - 1];
    }

    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);
    TT_FATAL(
        shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Shard Height {} should be a multiple of tile height",
        shard_spec.shape[0]);

    const uint32_t num_tiles_nhw_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    const uint32_t start_offset =
        ttnn::operations::data_movement::get_tiled_start_offset(output_tensor, output_tensor_start);
    SliceWriteRuntimeArgs result(cores.size());
    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        } else if (is_width_sharded) {
            core_h_index = 0;
            core_w_index = core_index;
        }

        const uint32_t num_sticks_read = core_h_index * num_tiles_nhw_per_core;
        const uint32_t width_offset = core_w_index * num_tiles_per_channel;
        const uint32_t channels_tiles_this_core = std::min(output_channel_tiles - width_offset, num_tiles_per_channel);
        id_per_dim[0] = 0;
        uint32_t unpadded_written = num_sticks_read;
        uint32_t start_id = start_offset + width_offset;
        int max_num_tiles_this_core = 0;
        for (uint32_t j = 1; j < num_dims; ++j) {
            id_per_dim[j] = unpadded_written % num_input_tiles_per_dim[j];
            if (j == num_dims - 1 && unpadded_written == num_input_tiles_per_dim[j]) {
                // Handle edge case where last dimension is completely written.
                id_per_dim[j] = num_input_tiles_per_dim[j];
            }
            unpadded_written /= num_input_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_tiles_per_dim[j - 1];
            size_till_end[j] = num_input_tiles_per_dim[j] - id_per_dim[j] - (j == 1 ? 0 : 1);
            max_num_tiles_this_core += size_till_end[j] * accumulated_input_total_tiles_per_dim[j - 1];
        }

        const uint32_t num_tiles_this_core = std::min<uint32_t>(
            num_tiles_nhw_per_core * num_tiles_per_channel, std::max<int>(max_num_tiles_this_core, 0));
        log_trace(
            tt::LogOp,
            "Start ID: {}, Start ID per dim: {}, Size till end: {}, Channel Tiles: {}, Max Tiles: {}, Num Tiles: {} "
            "for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            channels_tiles_this_core,
            max_num_tiles_this_core,
            num_tiles_this_core,
            core);
        ReaderKernelArgs reader_args = {num_tiles_this_core};
        WriterKernelArgs writer_args = {
            tile_size_bytes,
            tile_size_bytes,
            tile_size_bytes,
            num_dims,
            start_id,
            num_tiles_this_core,
            num_tiles_this_core,
            1};
        writer_args.insert(writer_args.end(), num_input_tiles_per_dim.begin(), num_input_tiles_per_dim.end());
        writer_args.insert(writer_args.end(), num_output_tiles_per_dim.begin(), num_output_tiles_per_dim.end());
        writer_args.insert(writer_args.end(), id_per_dim.begin(), id_per_dim.end());
        writer_args.push_back(num_tiles_per_channel - channels_tiles_this_core);
        result[core_index] = {std::move(reader_args), std::move(writer_args)};
        ++core_index;
    }
    return result;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceWriteTiledShardedInputProgramFactory::create_program_artifacts(
    const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto input_shape = input.logical_shape();
    const auto input_padded_shape = input.padded_shape();
    const auto output_shape = output.logical_shape();
    const tt::DataFormat input_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t tile_size_bytes = tt::tile_size(input_format);

    TT_FATAL(input.dtype() == output.dtype(), "Input & output should have the same dtype");
    TT_FATAL(operation_attributes.slice_start[-1] == 0, "Slice write expects last-dimension start to be 0");
    TT_FATAL(
        operation_attributes.slice_start[-2] % TILE_HEIGHT == 0,
        "Slice write expects second-last-dimension start to be tile aligned");
    TT_FATAL(input_padded_shape[-2] % TILE_HEIGHT == 0, "Input height must be tile aligned");
    TT_FATAL(input.layout() == Layout::TILE && output.layout() == Layout::TILE, "Input and output must be tiled");
    TT_FATAL(input.shard_spec().has_value(), "Input tensor should be sharded");
    TT_FATAL(input_format == output_format, "Input & output should have the same data format");

    const auto shard_spec = input.shard_spec().value();
    TT_FATAL(shard_spec.shape[0] % TILE_HEIGHT == 0, "Shard height must be tile aligned");
    const CoreRangeSet input_cores = shard_spec.grid;
    const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);
    const uint32_t num_tiles_height_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    const uint32_t num_tiles_channel_per_core = shard_spec.shape[1] / TILE_WIDTH;
    const uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input);
    const bool unpad_input_width = num_tiles_channel_per_core * TILE_WIDTH * num_cores_channels > output_shape[-1];

    ProgramSpec spec;
    spec.name = "slice_write_tiled_sharded_input";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TILE_SHARDED_INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = TILE_SHARDED_OUTPUT, .spec = output.tensor_spec()},
    };
    spec.dataflow_buffers = {DataflowBufferSpec{
        .unique_id = TILE_SHARDED_INPUT_DFB,
        .entry_size = tile_size_bytes,
        .num_entries = num_tiles_height_per_core * num_tiles_channel_per_core,
        .data_format_metadata = input_format,
        .borrowed_from = TILE_SHARDED_INPUT,
    }};
    KernelSpec reader{
        .unique_id = TILE_SHARDED_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
            "slice_write_reader_sharded.cpp",
        .dfb_bindings = {ProducerOf(TILE_SHARDED_INPUT_DFB, "input")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch(), true),
    };
    KernelSpec writer{
        .unique_id = TILE_SHARDED_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
            "slice_write_writer_interleaved.cpp",
        .dfb_bindings = {ConsumerOf(TILE_SHARDED_INPUT_DFB, "input")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = TILE_SHARDED_OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"output_byte_offset",
                  "output_stick_size",
                  "input_stick_size",
                  "stick_size_offset",
                  "num_dims",
                  "start_id",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier",
                  "padding_width_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch(), true),
        .advanced_options = {.num_runtime_varargs = 3 * static_cast<uint32_t>(input_shape.rank())},
    };
    writer.compile_time_args = {{"unpad_input_width", unpad_input_width ? 1U : 0U}};
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {WorkUnitSpec{
        .name = "main", .kernels = {TILE_SHARDED_READER, TILE_SHARDED_WRITER}, .target_nodes = input_cores}};

    const auto all_runtime_args = get_slice_write_runtime_args_tiled_sharded_input(
        input, output, operation_attributes.slice_start, operation_attributes.slice_end, iter_cores);
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = TILE_SHARDED_READER};
    KernelRunArgs writer_run{.kernel = TILE_SHARDED_WRITER};
    const uint32_t num_varargs = 3 * static_cast<uint32_t>(input_shape.rank());
    for (uint32_t i = 0; i < iter_cores.size(); ++i) {
        const auto& core = iter_cores[i];
        const auto& r = all_runtime_args[i].first;
        const auto& w = all_runtime_args[i].second;
        AddRuntimeArgsForNode(reader_run.runtime_arg_values, core, {{"num_units", r[0]}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"output_byte_offset", 0},
             {"output_stick_size", w[0]},
             {"input_stick_size", w[1]},
             {"stick_size_offset", w[2]},
             {"num_dims", w[3]},
             {"start_id", w[4]},
             {"num_sticks_per_core", w[5]},
             {"num_sticks_per_core_read", w[6]},
             {"num_read_per_barrier", w[7]},
             {"padding_width_units", w[8 + num_varargs]}});
        writer_run.advanced_options.runtime_varargs.emplace(
            core, AdvancedKernelRunArgs::Varargs(w.begin() + 8, w.begin() + 8 + num_varargs));
    }
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args.emplace(TILE_SHARDED_INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(TILE_SHARDED_OUTPUT, TensorArgument{output.mesh_tensor()});
    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
