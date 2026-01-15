// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_tile_program_factory.hpp"
#include "padded_slice_utils.hpp"

#include "hostdevcommon/kernel_structs.h"
#include "optional"
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <vector>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::padded_slice::program {

// Circular buffer indices
const uint32_t cb_buffer_size = 4;
const uint32_t cb_input_index = 0;
const uint32_t cb_untilized_index = 1;
const uint32_t cb_output_index = 2;
const uint32_t cb_padding_index = 3;

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

    uint32_t num_cores_channels = detail::get_num_cores_channels_from_sharded_tensor(output_tensor);
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

    uint32_t start_addr = input_tensor.buffer()->address();
    std::vector<uint32_t> common_reader_kernel_args = {
        start_addr,  // read from nearest aligned address
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
        reader_kernel_args[5] = this_core_num_tiles_per_channel;
        reader_kernel_args[6] = max_num_tiles_per_row;
        reader_kernel_args[10] = tt::div_up(input_padded_shape[-1], TILE_WIDTH) - max_num_tiles_per_row;

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

        uint32_t addr_offset = 2;
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

PaddedSliceTileProgramFactory::cached_program_t PaddedSliceTileProgramFactory::create(
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
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    TT_FATAL(
        input_padded_shape[3] % tt::constants::TILE_WIDTH == 0,
        "Input tensor channel dimension must be divisible by TILE_WIDTH for padded_slice operation with tiled inputs");
    uint32_t num_tiles_per_channel = input_padded_shape[3] / tt::constants::TILE_WIDTH;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    TT_FATAL(output.is_sharded(), "Output Tensor must be sharded.");
    auto output_shard_spec = output.shard_spec().value();

    uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();
    uint32_t output_row_size_elems = output_shard_spec.shape[1];

    CoreRangeSet total_cores = output.shard_spec().value().grid;
    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);

    uint32_t num_cores_channels = detail::get_num_cores_channels_from_sharded_tensor(output);
    uint32_t max_num_tiles_per_row = 0;
    for (uint32_t channel_index = 0; channel_index < num_cores_channels; channel_index++) {
        const uint32_t width_offset_elems = channel_index * output_row_size_elems;
        const uint32_t width_offset_start_tile = width_offset_elems / TILE_WIDTH;
        const uint32_t width_offset_end_tile = tt::div_up(width_offset_elems + output_row_size_elems, TILE_WIDTH);
        const uint32_t this_core_num_tiles_per_channel = width_offset_end_tile - width_offset_start_tile;
        max_num_tiles_per_row = std::max(max_num_tiles_per_row, this_core_num_tiles_per_channel);
    }
    num_tiles_per_channel = tt::div_up(num_tiles_per_channel, num_cores_channels);

    [[maybe_unused]] uint32_t num_tiles_height_per_core =
        tt::div_up(output_shard_spec.shape[0], tt::constants::TILE_HEIGHT);
    uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    TT_FATAL(
        dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    uint32_t max_read_size = 4096;

    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    TT_FATAL(
        output_row_size_bytes % dst_buffer_alignment == 0,
        "Output row size {} must be aligned to the destination buffer {} alignment {}",
        output_row_size_bytes,
        output.buffer()->buffer_type(),
        dst_buffer_alignment);
    // Input is tiled, and so channels would always be aligned to TILE_WIDTH.
    // So the non aligned copy is needed if the output alignment is less than TILE_WIDTH * element_size.
    auto alignment = TILE_WIDTH * output.element_size();
    auto is_non_aligned = false;
    if (output_row_size_bytes % alignment) {
        is_non_aligned = true;
    }

    tt::tt_metal::create_cb(
        cb_input_index,
        program,
        total_cores,
        input_single_tile_size,
        cb_buffer_size * max_num_tiles_per_row,
        input_cb_data_format);

    tt::tt_metal::create_cb(
        cb_untilized_index,
        program,
        total_cores,
        output_single_tile_size,
        cb_buffer_size * max_num_tiles_per_row,
        output_cb_data_format);

    log_debug(
        tt::LogOp,
        "output_row_size_bytes: {}, num_output_sticks_per_core: {}",
        output_row_size_bytes,
        num_output_sticks_per_core);

    auto cb_output_tuple = tt::tt_metal::create_cb(
        cb_output_index,
        program,
        total_cores,
        output_row_size_bytes,
        num_output_sticks_per_core,
        output_cb_data_format,
        output.buffer());

    tt::tt_metal::create_cb(
        cb_padding_index,
        program,
        total_cores,
        output_row_size_bytes,
        1,  // We need only a single row to hold the padding, and reuse it.
        output_cb_data_format);
    log_debug(
        tt::LogOp,
        "num_tiles_height_per_core: {}, num_tiles_per_channel: {}, max_num_tiles_per_row: {}",
        num_tiles_height_per_core,
        num_tiles_per_channel,
        max_num_tiles_per_row);
    std::vector<uint32_t> compute_args = {
        cb_input_index,         // src0_cb_index
        cb_untilized_index,     // untilized_cb_index
        cb_untilized_index,     // untilized_cb_index
        max_num_tiles_per_row,  // per_block_ntiles
        1                       // block_size_height_ntiles
    };

    const std::string compute_kernel =
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp";

    auto untilize_compute_kernel_id = CreateKernel(
        program, compute_kernel, total_cores, ComputeConfig{.fp32_dest_acc_en = false, .compile_args = compute_args});

    std::vector<uint32_t> writer_compile_time_args_vec = {
        cb_untilized_index,
        cb_output_index,
        cb_padding_index,
        (std::uint32_t)is_non_aligned,
        input_padded_shape.rank() /* == 4*/,
        output.element_size(),
        output_row_size_bytes};

    std::vector<uint32_t> reader_compile_time_args_vec = {max_num_tiles_per_row};
    tt::tt_metal::TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args_vec);
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
        "padded_slice_reader_tiled_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
        "writer_unary_sharded_padded_tiled.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    auto all_runtime_args = get_padded_slice_runtime_args_tile_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores, max_num_tiles_per_row, is_non_aligned);

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, std::get<0>(all_runtime_args[i]));
        tt::tt_metal::SetRuntimeArgs(program, untilize_compute_kernel_id, core, std::get<1>(all_runtime_args[i]));
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::get<2>(all_runtime_args[i]));
        i++;
    }

    shared_variables_t shared_vars{
        /* unary_reader_kernel_id = */ unary_reader_kernel_id,
        /* unary_writer_kernel_id = */ unary_writer_kernel_id,
        /* untilize_compute_kernel_id = */ untilize_compute_kernel_id,
        /* output_tensor_start = */ output_tensor_start,
        /* actual_output_shape = */ actual_output_shape,
        /* compute_with_storage_grid_size = */ compute_with_storage_grid_size,
        /* max_read_size = */ max_read_size,
        /* max_num_tiles_per_row = */ max_num_tiles_per_row,
        /* iter_cores = */ iter_cores,
        /* cb_output_tuple = */ cb_output_tuple};
    return cached_program_t{std::move(program), std::move(shared_vars)};
}

void PaddedSliceTileProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PaddedSliceParams& /*operation_attributes*/,
    const PaddedSliceInputs& tensor_args,
    Tensor& output) {
    auto& shared_vars = cached_program.shared_variables;
    const auto& src_tensor = tensor_args.input;
    auto& dst_tensor = output;
    TT_FATAL(dst_tensor.is_sharded(), "Output tensor must be sharded");
    UpdateDynamicCircularBufferAddress(
        cached_program.program, std::get<1>(shared_vars.cb_output_tuple), *dst_tensor.buffer());
    uint32_t output_row_size_bytes = dst_tensor.shard_spec()->shape[1] * dst_tensor.element_size();
    auto alignment = TILE_WIDTH * dst_tensor.element_size();
    auto is_non_aligned = output_row_size_bytes % alignment;
    auto all_runtime_args = get_padded_slice_runtime_args_tile_sharded_output(
        src_tensor,
        dst_tensor,
        shared_vars.output_tensor_start,
        shared_vars.actual_output_shape,
        shared_vars.iter_cores,
        shared_vars.max_num_tiles_per_row,
        is_non_aligned);

    uint32_t i = 0;
    for (const auto& core : shared_vars.iter_cores) {
        tt::tt_metal::SetRuntimeArgs(
            cached_program.program, shared_vars.unary_reader_kernel_id, core, std::get<0>(all_runtime_args[i]));
        tt::tt_metal::SetRuntimeArgs(
            cached_program.program, shared_vars.untilize_compute_kernel_id, core, std::get<1>(all_runtime_args[i]));
        tt::tt_metal::SetRuntimeArgs(
            cached_program.program, shared_vars.unary_writer_kernel_id, core, std::get<2>(all_runtime_args[i]));
        i++;
    }
}

}  // namespace ttnn::operations::experimental::padded_slice::program
