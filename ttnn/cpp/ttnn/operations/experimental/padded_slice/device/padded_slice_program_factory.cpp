// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "optional"
#include "tt-metalium/assert.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"
#include "ttnn/operations/math.hpp"
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <vector>

#include "padded_slice_op.hpp"
#include "padded_slice_rm_multi_core_program_factory.hpp"
#include "padded_slice_tile_multi_core_program_factory.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

const uint32_t cb_buffer_size = 4;
const uint32_t cb_input_index = 0;
const uint32_t cb_untilized_index = 1;
const uint32_t cb_output_index = 2;

namespace ttnn::operations::experimental::detail {

uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor) {
    auto shard_spec = tensor.shard_spec().value();
    auto core_grid = shard_spec.grid;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    uint32_t num_cores_channels = 1;
    if (tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        if (rm_orientation) {
            num_cores_channels = core_grid.bounding_box().grid_size().x;
        } else {
            num_cores_channels = core_grid.bounding_box().grid_size().y;
        }
    } else if (tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_channels = core_grid.num_cores();
    }
    return num_cores_channels;
}
static std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
get_padded_slice_runtime_args_rm_sharded_output(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& actual_output_shape,
    const std::vector<CoreCoord>& cores,
    uint32_t max_read_size) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.logical_shape();
    auto output_shard_spec = output_tensor.shard_spec().value();
    auto output_shard_shape = output_shard_spec.shape;

    auto num_cores_total = cores.size();

    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(output_tensor);
    uint32_t input_page_size = input_shape[-1] * input_tensor.element_size();
    uint32_t input_row_size_bytes = input_shape[-1] * input_tensor.element_size() / num_cores_channels;

    uint32_t output_row_size_bytes = output_shard_shape[1] * input_tensor.element_size();
    uint32_t output_row_size_elems = output_shard_shape[1];

    log_debug(
        tt::LogOp, "input_row_size_bytes: {}, output_row_size_bytes: {}", input_row_size_bytes, output_row_size_bytes);
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_output_sticks_per_dim[0] = 1;
    num_input_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    log_debug(tt::LogOp, "Output Shape : {}, Input Shape : {}", actual_output_shape, input_shape);
    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_output_dim = actual_output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_input_dim = (num_total_dim - num_output_dim) * accumulated_total_per_dim[i - 1];
        num_output_sticks_per_dim[i] = num_output_dim;
        num_input_sticks_per_dim[i] = num_input_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    for (int i = 0; i < num_dims; i++) {
        log_debug(
            tt::LogOp,
            "i = {}, num_output_sticks_per_dim: {}, num_input_sticks_per_dim: {}, accumulated_total_per_dim: {}",
            i,
            num_output_sticks_per_dim[i],
            num_input_sticks_per_dim[i],
            accumulated_total_per_dim[i]);
    }
    using namespace tt::tt_metal::experimental;
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    uint32_t output_row_size_bytes_offset = tt::round_up(output_row_size_bytes, alignment);
    uint32_t start_addr = input_tensor.buffer()->address();
    std::vector<uint32_t> common_reader_kernel_args = {
        start_addr + begins_bytes - misalignment,  // read from nearest aligned address
        input_page_size,
        input_row_size_bytes,
        output_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    const auto num_sticks_per_core = output_shard_spec.shape[0];

    log_debug(tt::LogOp, "num_stick_per_core: {}", num_sticks_per_core);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);

    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        }

        const uint32_t num_sticks_written = core_h_index * num_sticks_per_core;
        const uint32_t width_offset = core_w_index * output_row_size_bytes_offset;

        id_per_dim[0] = num_sticks_written % num_output_sticks_per_dim[0];
        uint32_t output_written = num_sticks_written / num_output_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = output_written % num_output_sticks_per_dim[j];
            output_written = output_written / num_output_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[0] += width_offset;

        uint32_t addr_offset = 5;
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = num_sticks_per_core;
        reader_kernel_args[addr_offset++] = num_sticks_per_core;
        reader_kernel_args[addr_offset] = num_sticks_per_core;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> writer_kernel_args = {
            num_sticks_per_core, output_row_size_elems, input_row_size_bytes, output_row_size_bytes};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

static std::vector<std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>>
get_padded_slice_runtime_args_tile_sharded_output(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& actual_output_shape,
    const std::vector<CoreCoord>& cores,
    uint32_t max_read_size) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_padded_shape = input_tensor.get_padded_shape();
    auto input_shape = input_tensor.get_logical_shape();
    auto output_shard_spec = output_tensor.shard_spec().value();
    auto output_shard_shape = output_shard_spec.shape;

    auto num_cores_total = cores.size();

    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    uint32_t num_cores_channels = 1;
    auto total_cores = output_shard_spec.grid;
    if (is_block_sharded) {
        if (rm_orientation) {
            num_cores_channels = total_cores.bounding_box().grid_size().x;
        } else {
            num_cores_channels = total_cores.bounding_box().grid_size().y;
        }
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    uint32_t output_row_size_bytes = output_shard_shape[1] * input_tensor.element_size();
    uint32_t output_row_size_elems = output_shard_shape[1];

    uint32_t num_tiles_per_channel = tt::div_up(input_padded_shape[3], tt::constants::TILE_WIDTH);
    num_tiles_per_channel = num_tiles_per_channel / num_cores_channels;
    TT_FATAL(
        num_tiles_per_channel == tt::div_up(output_shard_shape[1], tt::constants::TILE_WIDTH),
        "Number of tiles per channel {} should be equal to number of output shard width in tiles {}",
        num_tiles_per_channel,
        tt::div_up(output_shard_shape[1], tt::constants::TILE_WIDTH));
    uint32_t num_tiles_height_per_core = tt::div_up(output_shard_spec.shape[0], tt::constants::TILE_HEIGHT);
    uint32_t num_tiles_per_core = num_tiles_per_channel * num_tiles_height_per_core;

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    std::vector<uint32_t> num_output_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_input_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> accumulated_total_sticks_per_dim(num_dims);

    num_output_tiles_per_dim[0] = tt::div_up(actual_output_shape[-1], TILE_WIDTH) / num_cores_channels;
    num_output_tiles_per_dim[1] = (tt::round_up(output_tensor_start[-2] + actual_output_shape[-2], TILE_HEIGHT) -
                                   tt::round_down(output_tensor_start[-2], TILE_HEIGHT)) /
                                  TILE_HEIGHT;

    log_debug(tt::LogOp, "Output Start : {}, Output Shape : {}", output_tensor_start, actual_output_shape);

    accumulated_total_tiles_per_dim[0] = tt::div_up(actual_output_shape[-1], TILE_WIDTH);
    accumulated_total_tiles_per_dim[1] = tt::div_up(input_shape[-2], TILE_HEIGHT) * accumulated_total_tiles_per_dim[0];

    num_input_tiles_per_dim[0] = tt::div_up(actual_output_shape[-1], TILE_WIDTH) - num_output_tiles_per_dim[0];
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
    uint32_t start_addr = input_tensor.buffer()->address();
    std::vector<uint32_t> common_reader_kernel_args = {
        start_addr,  // read from nearest aligned address
        num_dims,
        0,  // input_start_id
        0,  // num_tiles_per_core
        num_tiles_per_channel};

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_tiles_per_dim.begin(), num_output_tiles_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_tiles_per_dim.begin(), num_input_tiles_per_dim.end());

    std::vector<std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(
        num_cores_total);

    const auto num_sticks_per_core = output_shard_spec.shape[0];
    const auto num_tiles_per_full_row = num_output_tiles_per_dim[1] * num_output_tiles_per_dim[0];
    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, output_tensor_start);
    log_debug(tt::LogOp, "Start Offset: {}", start_offset);
    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        }

        const uint32_t num_sticks_written_start = core_h_index * num_sticks_per_core;
        const uint32_t num_sticks_written_end = (core_h_index + 1) * num_sticks_per_core;

        const uint32_t width_offset = core_w_index * num_tiles_per_channel;
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
        uint32_t input_end_id = ttnn::operations::data_movement::get_tiled_start_offset(
            input_tensor, ttnn::Shape(end_index_in_input_per_dim), true);
        uint32_t output_start_id = ttnn::operations::data_movement::get_tiled_start_offset(
            actual_output_shape, ttnn::Shape(start_index_per_dim));
        uint32_t output_end_id = ttnn::operations::data_movement::get_tiled_start_offset(
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
                               num_output_tiles_per_dim[0];

        if (start_index_per_dim[2] != 0) {
            num_tiles_this_core += ((tt::round_up(output_tensor_start[-2] + actual_output_shape[-2], TILE_HEIGHT) -
                                     tt::round_down(start_index_in_input_per_dim[num_dims - 2], TILE_HEIGHT)) /
                                    TILE_HEIGHT) *
                                   num_output_tiles_per_dim[0];
        }
        if (num_full_rows < 0) {
            num_full_rows = 0;
            num_tiles_this_core = 0;
        }
        log_debug(
            tt::LogOp,
            "For Core {}, Input Start ID {}, End ID {}, Output Start Coord: {}, End Coord : {}, Input Start Coord: {}, "
            "End Coord "
            ": {}, Num Full Rows "
            ": {}, Num Tiles : {}",
            core,
            input_start_id,
            input_end_id,
            start_index_per_dim,
            end_index_per_dim,
            start_index_in_input_per_dim,
            end_index_in_input_per_dim,
            num_full_rows,
            num_tiles_this_core);

        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;

        uint32_t addr_offset = 2;
        reader_kernel_args[addr_offset++] = input_start_id + width_offset;
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
            num_tiles_this_core / num_tiles_per_channel,  // number of tiles to read
        };

        std::vector<uint32_t> writer_kernel_args = {num_tiles_this_core, num_tiles_per_channel, num_sticks_per_core};
        writer_kernel_args.insert(writer_kernel_args.end(), reversed_start_index.begin(), reversed_start_index.end());
        writer_kernel_args.insert(
            writer_kernel_args.end(), reversed_output_start_in_input.begin(), reversed_output_start_in_input.end());
        writer_kernel_args.insert(writer_kernel_args.end(), reversed_output_end.begin(), reversed_output_end.end());

        ret_val[core_index] = {reader_kernel_args, compute_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

operation::ProgramWithCallbacks padded_slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {
    bool has_step = false;
    for (int i = 0; i < step.size(); i++) {
        if (step[i] != 1) {
            has_step = true;
            break;
        }
    }
    TT_FATAL(!has_step, "Padded Slice with Stride is not supported yet");
    TT_FATAL(
        output.is_sharded(),
        "Output must be sharded for the padded_slice operation. Use slice for non-sharded outputs");
    TT_FATAL(!a.is_sharded(), " Sharded input is not supported for padded_slice operation");
    if (a.layout() == Layout::ROW_MAJOR) {
        return padded_slice_rm_multi_core(a, output, output_tensor_start, output_tensor_end);
    } else if (a.layout() == Layout::TILE) {
        return padded_slice_tile_multi_core(a, output, output_tensor_start, output_tensor_end);
    } else {
        TT_THROW("Unsupported layout for padded_slice operation: {}", a.layout());
    }
    return {};
}

}  // namespace ttnn::operations::experimental::detail
