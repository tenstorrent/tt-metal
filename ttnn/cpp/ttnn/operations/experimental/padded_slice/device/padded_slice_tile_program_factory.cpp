// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_tile_program_factory.hpp"
#include "padded_slice_utils.hpp"

#include <optional>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <vector>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

const uint32_t cb_buffer_size = 4;

static std::vector<std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>>
get_padded_slice_runtime_args_tile_sharded_output(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& actual_output_shape,
    const std::vector<CoreCoord>& cores,
    uint32_t max_num_tiles_per_row,
    bool is_non_aligned) {
    auto input_padded_shape = input_tensor.padded_shape();
    auto input_shape = input_tensor.logical_shape();
    auto output_shard_spec = output_tensor.shard_spec().value();
    auto output_shard_shape = output_shard_spec.shape;

    auto num_cores_total = cores.size();

    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    bool is_width_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    uint32_t num_cores_channels =
        ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor(output_tensor);
    const uint32_t input_num_tiles_per_channel = tt::div_up(input_padded_shape[3], tt::constants::TILE_WIDTH);

    uint32_t num_tiles_per_channel = tt::div_up(input_num_tiles_per_channel, num_cores_channels);
    TT_FATAL(
        num_tiles_per_channel == tt::div_up(output_shard_shape[1], tt::constants::TILE_WIDTH),
        "Number of tiles per channel {} should be equal to number of output shard width in tiles {}",
        num_tiles_per_channel,
        tt::div_up(output_shard_shape[1], tt::constants::TILE_WIDTH));

    [[maybe_unused]] uint32_t output_row_size_bytes = output_shard_shape[1] * output_tensor.element_size();
    uint32_t output_row_size_elems = output_shard_shape[1];

    uint32_t input_channels_num_elems = input_padded_shape[3];

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    std::vector<uint32_t> num_output_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_input_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> accumulated_total_sticks_per_dim(num_dims);

    num_output_tiles_per_dim[0] = num_tiles_per_channel;
    num_output_tiles_per_dim[1] = (tt::round_up(output_tensor_start[-2] + actual_output_shape[-2], TILE_HEIGHT) -
                                   tt::round_down(output_tensor_start[-2], TILE_HEIGHT)) /
                                  TILE_HEIGHT;

    log_debug(tt::LogOp, "Output Start : {}, Output Shape : {}", output_tensor_start, actual_output_shape);

    accumulated_total_tiles_per_dim[0] = tt::div_up(actual_output_shape[-1], TILE_WIDTH);
    accumulated_total_tiles_per_dim[1] = tt::div_up(input_shape[-2], TILE_HEIGHT) * accumulated_total_tiles_per_dim[0];

    num_input_tiles_per_dim[0] = tt::div_up(input_padded_shape[-1], TILE_WIDTH) - num_output_tiles_per_dim[0];
    num_input_tiles_per_dim[1] =
        (tt::div_up(input_shape[-2], TILE_HEIGHT) - num_output_tiles_per_dim[1]) * accumulated_total_tiles_per_dim[0];

    num_output_sticks_per_dim[0] = 1;
    num_input_sticks_per_dim[0] = 0;
    accumulated_total_sticks_per_dim[0] = 1;

    for (int32_t i = 2; i < num_dims; i++) {
        uint32_t num_output_dim = actual_output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        log_debug(tt::LogOp, "i = {}, num_output_dim: {}, num_total_dim: {}", i, num_output_dim, num_total_dim);
        uint32_t num_input_dim = (num_total_dim - num_output_dim) * accumulated_total_tiles_per_dim[i - 1];
        num_output_tiles_per_dim[i] = num_output_dim;
        num_input_tiles_per_dim[i] = num_input_dim;
        accumulated_total_tiles_per_dim[i] = num_total_dim * accumulated_total_tiles_per_dim[i - 1];
    }

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_output_dim = actual_output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_input_dim = (num_total_dim - num_output_dim) * accumulated_total_sticks_per_dim[i - 1];
        num_output_sticks_per_dim[i] = num_output_dim;
        num_input_sticks_per_dim[i] = num_input_dim;
        accumulated_total_sticks_per_dim[i] = num_total_dim * accumulated_total_sticks_per_dim[i - 1];
    }

    for (int i = 0; i < num_dims; i++) {
        log_debug(
            tt::LogOp,
            "i = {}, num_output_tiles_per_dim: {}, num_input_tiles_per_dim: {}, accumulated_total_tiles_per_dim: {}",
            i,
            num_output_tiles_per_dim[i],
            num_input_tiles_per_dim[i],
            accumulated_total_tiles_per_dim[i]);
    }

    for (int i = 0; i < num_dims; i++) {
        log_debug(
            tt::LogOp,
            "i = {}, num_output_sticks_per_dim: {}, num_input_sticks_per_dim: {}, accumulated_total_per_dim: {}",
            i,
            num_output_sticks_per_dim[i],
            num_input_sticks_per_dim[i],
            accumulated_total_sticks_per_dim[i]);
    }
    const auto num_tiles_per_full_row = num_output_tiles_per_dim[1] * max_num_tiles_per_row;

    std::vector<uint32_t> common_reader_kernel_args = {
        num_dims,
        0,  // input_start_id
        0,  // num_tiles_per_core
        max_num_tiles_per_row,
        0  // num_tiles_per_row_this_core
    };

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_tiles_per_dim.begin(), num_output_tiles_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_tiles_per_dim.begin(), num_input_tiles_per_dim.end());

    std::vector<std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(
        num_cores_total);

    const auto num_sticks_per_core = output_shard_spec.shape[0];
    [[maybe_unused]] uint32_t start_offset =
        ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, output_tensor_start);
    log_debug(tt::LogOp, "Start Offset: {}", start_offset);
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
        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;

        const uint32_t num_sticks_written_start = core_h_index * num_sticks_per_core;
        const uint32_t num_sticks_written_end = (core_h_index + 1) * num_sticks_per_core;

        const uint32_t width_offset_elems = core_w_index * output_row_size_elems;
        int this_core_output_channels_end_elem = width_offset_elems + output_row_size_elems;

        uint32_t output_channels_padding_elems =
            std::max<int>(this_core_output_channels_end_elem - input_channels_num_elems, 0);

        const uint32_t width_offset_start_tile = width_offset_elems / TILE_WIDTH;
        const uint32_t width_offset_end_tile = std::min(
            tt::div_up(std::min(width_offset_elems + output_row_size_elems, input_channels_num_elems), TILE_WIDTH),
            input_num_tiles_per_channel);
        const uint32_t this_core_num_tiles_per_channel = width_offset_end_tile - width_offset_start_tile;
        const uint32_t misalignment_bytes = width_offset_elems % TILE_WIDTH * output_tensor.element_size();

        if (!is_non_aligned && output_channels_padding_elems == 0) {
            TT_FATAL(
                this_core_num_tiles_per_channel == max_num_tiles_per_row,
                "If padded_slice uses aligned reads, then all cores must read the same number of tiles per row. Core "
                "{} reads {} tiles, expected {}",
                core,
                this_core_num_tiles_per_channel,
                max_num_tiles_per_row);
        }
        reader_kernel_args[4] = this_core_num_tiles_per_channel;
        reader_kernel_args[5] = max_num_tiles_per_row;
        reader_kernel_args[9] = tt::div_up(input_padded_shape[-1], TILE_WIDTH) - max_num_tiles_per_row;

        std::vector<uint32_t> start_index_per_dim(num_dims);
        std::vector<uint32_t> end_index_per_dim(num_dims);

        uint32_t output_written_start = num_sticks_written_start / num_output_sticks_per_dim[0];
        uint32_t output_written_end = num_sticks_written_end / num_output_sticks_per_dim[0];

        for (uint32_t j = 0; j < num_dims; j++) {
            start_index_per_dim[j] =
                (j == num_dims - 1) ? output_written_start : output_written_start % num_output_sticks_per_dim[j];
            output_written_start = output_written_start / num_output_sticks_per_dim[j];

            end_index_per_dim[j] =
                (j == num_dims - 1) ? output_written_end : output_written_end % num_output_sticks_per_dim[j];
            output_written_end = output_written_end / num_output_sticks_per_dim[j];
        }

        // If this core's start location is beyond the output tensor's end, we need to clamp it to the end.
        if (start_index_per_dim[num_dims - 1] >= actual_output_shape[0]) {
            start_index_per_dim[num_dims - 1] = actual_output_shape[0];
            for (uint32_t j = 1; j < num_dims - 1; j++) {
                start_index_per_dim[j] = 0;
            }
        }

        // If this core's end location is beyond the output tensor's end, we need to clamp it to the end.
        if (end_index_per_dim[num_dims - 1] >= actual_output_shape[0]) {
            end_index_per_dim[num_dims - 1] = actual_output_shape[0];
            for (uint32_t j = 1; j < num_dims - 1; j++) {
                end_index_per_dim[j] = 0;
            }
        }
        std::vector<uint32_t> start_index_in_input_per_dim(num_dims);
        std::vector<uint32_t> end_index_in_input_per_dim(num_dims);

        for (uint32_t index = 0; index < num_dims; index++) {
            start_index_in_input_per_dim[index] =
                start_index_per_dim[num_dims - index - 1] + output_tensor_start[index];
            end_index_in_input_per_dim[index] = end_index_per_dim[num_dims - index - 1] + output_tensor_start[index];
        }
        std::ranges::reverse(end_index_per_dim);
        std::ranges::reverse(start_index_per_dim);
        uint32_t input_start_id = ttnn::operations::data_movement::get_tiled_start_offset(
            input_tensor, ttnn::Shape(start_index_in_input_per_dim));
        [[maybe_unused]] uint32_t input_end_id = ttnn::operations::data_movement::get_tiled_start_offset(
            input_tensor, ttnn::Shape(end_index_in_input_per_dim), true);
        ttnn::operations::data_movement::get_tiled_start_offset(actual_output_shape, ttnn::Shape(start_index_per_dim));
        ttnn::operations::data_movement::get_tiled_start_offset(
            actual_output_shape, ttnn::Shape(end_index_per_dim), true);

        int32_t num_full_rows = ((end_index_per_dim[0] - start_index_per_dim[0]) * actual_output_shape[1]) +
                                end_index_per_dim[1] - start_index_per_dim[1];

        if (start_index_per_dim[2] != 0) {
            num_full_rows--;
        }
        uint32_t num_tiles_this_core = num_full_rows * num_tiles_per_full_row;

        num_tiles_this_core += ((tt::round_up(end_index_in_input_per_dim[num_dims - 2], TILE_HEIGHT) -
                                 tt::round_down(output_tensor_start[num_dims - 2], TILE_HEIGHT)) /
                                TILE_HEIGHT) *
                               max_num_tiles_per_row;

        if (start_index_per_dim[2] != 0) {
            num_tiles_this_core += ((tt::round_up(output_tensor_start[-2] + actual_output_shape[-2], TILE_HEIGHT) -
                                     tt::round_down(start_index_in_input_per_dim[num_dims - 2], TILE_HEIGHT)) /
                                    TILE_HEIGHT) *
                                   max_num_tiles_per_row;
        }
        num_full_rows = std::max(num_full_rows, 0);
        log_trace(
            tt::LogOp,
            "For Core {}, Input Start ID {}, End ID {}, Output Start Coord: {}, End Coord : {}, Input Start Coord: {}, "
            "End Coord "
            ": {}, Num Full Rows "
            ": {}, Num Tiles : {}"
            " This Core Num Tiles Per Channel: {}",
            core,
            input_start_id,
            input_end_id,
            start_index_per_dim,
            end_index_per_dim,
            start_index_in_input_per_dim,
            end_index_in_input_per_dim,
            num_full_rows,
            num_tiles_this_core,
            this_core_num_tiles_per_channel);

        uint32_t addr_offset = 1;
        reader_kernel_args[addr_offset++] = input_start_id + width_offset_start_tile;
        reader_kernel_args[addr_offset++] = num_tiles_this_core;
        auto reversed_start_index = start_index_per_dim;
        std::ranges::reverse(reversed_start_index);
        auto reversed_tile_start_index = reversed_start_index;
        reversed_tile_start_index[0] /= TILE_WIDTH;
        reversed_tile_start_index[1] =
            tt::round_down(reversed_tile_start_index[1] + output_tensor_start[2], TILE_HEIGHT) -
            tt::round_down(output_tensor_start[-2], TILE_HEIGHT);
        reversed_tile_start_index[1] /= TILE_HEIGHT;
        std::vector<uint32_t> reversed_output_start_in_input(num_dims);
        std::vector<uint32_t> reversed_output_end(num_dims);

        for (uint32_t index = 0; index < num_dims; index++) {
            reversed_output_start_in_input[index] = output_tensor_start[num_dims - index - 1];
            reversed_output_end[index] = actual_output_shape[num_dims - index - 1];
        }

        reader_kernel_args.insert(
            reader_kernel_args.end(), reversed_tile_start_index.begin(), reversed_tile_start_index.end());

        std::vector<uint32_t> compute_kernel_args = {
            num_tiles_this_core / max_num_tiles_per_row,  // number of tiles to read
        };

        log_trace(
            tt::LogOp,
            "Core = {}, width_offset elems = {} to {}, tiles = {} to {}, input_channels_num_elems = {}, "
            "output_channels_padding = {}",
            core,
            core_w_index * output_row_size_elems,
            (core_w_index + 1) * output_row_size_elems,
            width_offset_start_tile,
            width_offset_end_tile,
            input_channels_num_elems,
            output_channels_padding_elems);

        std::vector<uint32_t> writer_kernel_args = {
            num_tiles_this_core,
            max_num_tiles_per_row,
            num_sticks_per_core,
            output_channels_padding_elems,
            misalignment_bytes};
        writer_kernel_args.insert(writer_kernel_args.end(), reversed_start_index.begin(), reversed_start_index.end());
        writer_kernel_args.insert(
            writer_kernel_args.end(), reversed_output_start_in_input.begin(), reversed_output_start_in_input.end());
        writer_kernel_args.insert(writer_kernel_args.end(), reversed_output_end.begin(), reversed_output_end.end());
        ret_val[core_index] = {reader_kernel_args, compute_kernel_args, writer_kernel_args};
        core_index++;
    }
    return ret_val;
}

using tt::tt_metal::experimental::AdvancedKernelRunArgs;
using tt::tt_metal::experimental::ComputeGen1Config;
using tt::tt_metal::experimental::ConsumerOf;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProducerOf;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::ScratchpadBinding;
using tt::tt_metal::experimental::ScratchpadSpec;
using tt::tt_metal::experimental::ScratchpadSpecName;
using tt::tt_metal::experimental::TensorArgument;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

const TensorParamName TILE_INPUT{"input"};
const TensorParamName TILE_OUTPUT{"output"};
const DFBSpecName TILE_INPUT_DFB{"input"};
const DFBSpecName TILE_UNTILIZED_DFB{"untilized"};
const DFBSpecName TILE_OUTPUT_DFB{"output"};
const ScratchpadSpecName TILE_PADDING_SCRATCH{"padding_scratch"};
const KernelSpecName TILE_READER{"reader"};
const KernelSpecName TILE_COMPUTE{"compute"};
const KernelSpecName TILE_WRITER{"writer"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
using namespace CMAKE_UNIQUE_NAMESPACE;
}  // namespace

ttnn::device_operation::ProgramArtifacts PaddedSliceTileProgramFactory::create_program_artifacts(
    const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.padded_slice_start;
    const auto& output_tensor_end = operation_attributes.padded_slice_end;

    const ttnn::Shape output_shape = output.logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    const ttnn::Shape& input_padded_shape = a.padded_shape();
    TT_FATAL(
        input_padded_shape.rank() == 4, "Input tensor must be rank 4 for padded_slice operation with tiled inputs");
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    TT_FATAL(
        input_padded_shape[3] % tt::constants::TILE_WIDTH == 0,
        "Input tensor channel dimension must be divisible by TILE_WIDTH for padded_slice operation with tiled inputs");
    const uint32_t input_num_tiles_per_channel = input_padded_shape[3] / tt::constants::TILE_WIDTH;

    TT_FATAL(output.is_sharded(), "Output Tensor must be sharded.");
    const auto output_shard_spec = output.shard_spec().value();
    const uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();
    const uint32_t output_row_size_elems = output_shard_spec.shape[1];

    const CoreRangeSet total_cores = output_shard_spec.grid;
    const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);

    const uint32_t num_cores_channels =
        ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor(output);
    uint32_t max_num_tiles_per_row = 0;
    for (uint32_t channel_index = 0; channel_index < num_cores_channels; channel_index++) {
        const uint32_t width_offset_elems = channel_index * output_row_size_elems;
        const uint32_t width_offset_start_tile = width_offset_elems / TILE_WIDTH;
        const uint32_t width_offset_end_tile = tt::div_up(width_offset_elems + output_row_size_elems, TILE_WIDTH);
        max_num_tiles_per_row = std::max(max_num_tiles_per_row, width_offset_end_tile - width_offset_start_tile);
    }
    const uint32_t num_tiles_per_channel = tt::div_up(input_num_tiles_per_channel, num_cores_channels);
    const uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");
    TT_FATAL(
        output.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    const uint32_t dst_buffer_alignment = ::hal::get_l1_alignment();
    TT_FATAL(
        output_row_size_bytes % dst_buffer_alignment == 0,
        "Output row size {} must be aligned to the destination buffer {} alignment {}",
        output_row_size_bytes,
        output.buffer()->buffer_type(),
        dst_buffer_alignment);
    // Input is tiled, so channels are always TILE_WIDTH-aligned. The non-aligned copy is needed
    // when the output alignment is smaller than TILE_WIDTH * element_size.
    const uint32_t alignment = TILE_WIDTH * output.element_size();
    const bool is_non_aligned = output_row_size_bytes % alignment != 0;

    log_debug(
        tt::LogOp,
        "num_tiles_per_channel: {}, max_num_tiles_per_row: {}",
        num_tiles_per_channel,
        max_num_tiles_per_row);

    ProgramSpec spec;
    spec.name = "padded_slice_tile";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = TILE_INPUT, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = TILE_OUTPUT, .spec = output.tensor_spec()},
    };
    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = TILE_INPUT_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = cb_buffer_size * max_num_tiles_per_row,
            .data_format_metadata = input_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = TILE_UNTILIZED_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = cb_buffer_size * max_num_tiles_per_row,
            .data_format_metadata = output_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = TILE_OUTPUT_DFB,
            .entry_size = output_row_size_bytes,
            .num_entries = num_output_sticks_per_core,
            .data_format_metadata = output_cb_data_format,
            .borrowed_from = TILE_OUTPUT,
        },
    };
    spec.scratchpads = {
        // We need only a single row to hold the padding, and reuse it.
        ScratchpadSpec{.unique_id = TILE_PADDING_SCRATCH, .size_per_node = output_row_size_bytes},
    };

    const uint32_t rank = static_cast<uint32_t>(input_padded_shape.rank());
    const KernelSpec reader{
        .unique_id = TILE_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
            "padded_slice_reader_tiled_interleaved_start_id.cpp",
        .dfb_bindings = {ProducerOf(TILE_INPUT_DFB, "input")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = TILE_INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"num_tiles_per_row", max_num_tiles_per_row}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_dims",
                  "start_id",
                  "num_tiles_per_core",
                  "num_tiles_per_barrier",
                  "num_tiles_per_row_this_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
        .advanced_options = {.num_runtime_varargs = 3 * rank},
    };
    const KernelSpec compute{
        .unique_id = TILE_COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/"
            "pack_untilize.cpp",
        // Legacy ComputeConfig defaults to O3; Metal 2.0 CompilerOptions defaults to O2.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {ConsumerOf(TILE_INPUT_DFB, "src"),
             tt::tt_metal::experimental::DFBBinding{
                 .dfb_spec_name = TILE_UNTILIZED_DFB,
                 .accessor_name = "untilize_out0",
                 .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
                 .accessor_aliases = {"untilize_out1"}}},
        .compile_time_args = {{"tiles_per_row", max_num_tiles_per_row}, {"block_size", 1}},
        .runtime_arg_schema = {.runtime_arg_names = {"total_blocks"}},
        .hw_config = ComputeGen1Config{.enable_32_bit_dest = false},
    };
    const KernelSpec writer{
        .unique_id = TILE_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
            "writer_unary_sharded_padded_tiled.cpp",
        .dfb_bindings =
            {ConsumerOf(TILE_UNTILIZED_DFB, "untilized"),
             ProducerOf(TILE_OUTPUT_DFB, "output"),
             ConsumerOf(TILE_OUTPUT_DFB, "output")},
        .scratchpad_bindings = {ScratchpadBinding{
            .scratchpad_spec_name = TILE_PADDING_SCRATCH, .accessor_name = "padding"}},
        .compile_time_args =
            {{"is_non_aligned", is_non_aligned},
             {"num_dims", rank},
             {"output_elem_size", output.element_size()},
             {"output_row_size_bytes", output_row_size_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"total_num_tiles",
                  "num_tiles_per_read",
                  "num_sticks_this_core",
                  "padded_channels_elems",
                  "misalignment"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
        .advanced_options = {.num_runtime_varargs = 3 * rank},
    };

    spec.kernels = {reader, compute, writer};
    spec.work_units = {
        WorkUnitSpec{.name = "main", .kernels = {TILE_READER, TILE_COMPUTE, TILE_WRITER}, .target_nodes = total_cores}};

    const auto all_runtime_args = get_padded_slice_runtime_args_tile_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores, max_num_tiles_per_row, is_non_aligned);

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = TILE_READER};
    KernelRunArgs compute_run{.kernel = TILE_COMPUTE};
    KernelRunArgs writer_run{.kernel = TILE_WRITER};
    for (uint32_t i = 0; i < iter_cores.size(); ++i) {
        const auto& core = iter_cores[i];
        const auto& r = std::get<0>(all_runtime_args[i]);
        const auto& c = std::get<1>(all_runtime_args[i]);
        const auto& w = std::get<2>(all_runtime_args[i]);

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_dims", r[0]},
             {"start_id", r[1]},
             {"num_tiles_per_core", r[2]},
             {"num_tiles_per_barrier", r[3]},
             {"num_tiles_per_row_this_core", r[4]}});
        reader_run.advanced_options.runtime_varargs.emplace(
            core, AdvancedKernelRunArgs::Varargs(r.begin() + 5, r.end()));

        AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"total_blocks", c[0]}});

        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"total_num_tiles", w[0]},
             {"num_tiles_per_read", w[1]},
             {"num_sticks_this_core", w[2]},
             {"padded_channels_elems", w[3]},
             {"misalignment", w[4]}});
        writer_run.advanced_options.runtime_varargs.emplace(
            core, AdvancedKernelRunArgs::Varargs(w.begin() + 5, w.end()));
    }

    run_args.kernel_run_args = {std::move(reader_run), std::move(compute_run), std::move(writer_run)};
    run_args.tensor_args.emplace(TILE_INPUT, TensorArgument{a.mesh_tensor()});
    run_args.tensor_args.emplace(TILE_OUTPUT, TensorArgument{output.mesh_tensor()});

    return {.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
