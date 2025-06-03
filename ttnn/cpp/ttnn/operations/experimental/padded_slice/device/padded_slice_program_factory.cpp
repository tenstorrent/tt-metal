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
#include <cstdint>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

#include "padded_slice_op.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

const uint32_t cb_buffer_size = 3;
const uint32_t cb_input_index = 0;
const uint32_t cb_untilized_index = 1;
const uint32_t cb_output_index = 2;

namespace ttnn::operations::experimental::detail {

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
    auto output_shape = output_tensor.logical_shape();
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
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_sticks_per_core != 0) {
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, output_row_size_bytes_offset, max_read_size);
        num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
    }
    log_debug(
        tt::LogOp,
        "num_stick_per_core: {}, num_stick_per_core_read: {}, num_read_per_barrier: {}",
        num_sticks_per_core,
        num_sticks_per_core_read,
        num_read_per_barrier);

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

static operation::ProgramWithCallbacks padded_slice_rm_multi_core(
    const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    const ttnn::Shape output_shape = output.logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    TT_FATAL(output.is_sharded(), "Output Tensor must be sharded.");
    auto output_shard_spec = output.shard_spec().value();

    bool is_block_sharded = output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();

    CoreRangeSet total_cores = output.shard_spec().value().grid;
    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);
    uint32_t num_cores_total = total_cores.num_cores();

    uint32_t num_cores_channels = 1;
    if (is_block_sharded) {
        if (rm_orientation) {
            num_cores_channels = total_cores.bounding_box().grid_size().x;
        } else {
            num_cores_channels = total_cores.bounding_box().grid_size().y;
        }
    }

    bool pad_output_row = false;

    TT_FATAL(
        a.logical_shape()[3] % num_cores_channels == 0,
        "Input tensor should be divisible by number of cores in channel dimension");
    uint32_t input_row_size_bytes = a.logical_shape()[-1] * a.element_size();
    input_row_size_bytes = input_row_size_bytes / num_cores_channels;

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(
        dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    uint32_t src_stick_size = input_row_size_bytes;
    uint32_t dst_stick_size = output_row_size_bytes;

    uint32_t src0_cb_index = 0;
    uint32_t temp_pad_cb_index = 1;
    uint32_t max_read_size = 4096;

    auto src_buffer_alignment = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    // if begins is not aligned then we need to pad the cb size, so that we can read from the nearest aligned address
    uint32_t begins_bytes = output_tensor_start[-1] * a.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    uint32_t cb_page_size = tt::round_up(output_row_size_bytes, alignment);

    CBHandle cb_src0, cb_temp_padded_row;
    uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_sticks_per_core * output_row_size_bytes, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, output_row_size_bytes)
            .set_globally_allocated_address(*output.buffer());
    cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    if (output_row_size_bytes > input_row_size_bytes) {
        pad_output_row = true;
        tt::tt_metal::CircularBufferConfig cb_temp_pad_config =
            tt::tt_metal::CircularBufferConfig(1 * output_row_size_bytes, {{temp_pad_cb_index, cb_data_format}})
                .set_page_size(temp_pad_cb_index, output_row_size_bytes);
    }

    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index};

    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_is_dram, misalignment};
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
        "padded_slice_reader_rm_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id;
    if (pad_output_row) {
        writer_compile_time_args_vec.push_back((std::uint32_t)temp_pad_cb_index);
        unary_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
            "writer_unary_sharded_padded_rm.cpp",
            total_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));
    } else {
        unary_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
            total_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));
    }

    auto all_runtime_args = get_padded_slice_runtime_args_rm_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores, max_read_size);

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
        i++;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           output_tensor_start,
                                           actual_output_shape,
                                           compute_with_storage_grid_size,
                                           max_read_size,
                                           iter_cores,
                                           cb_src0](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        TT_FATAL(dst_tensor.is_sharded(), "Output tensor must be sharded");
        UpdateDynamicCircularBufferAddress(program, cb_src0, *dst_tensor.buffer());

        auto all_runtime_args = get_padded_slice_runtime_args_rm_sharded_output(
            src_tensor, dst_tensor, output_tensor_start, actual_output_shape, iter_cores, max_read_size);

        uint32_t i = 0;
        for (const auto& core : iter_cores) {
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            i++;
        }

    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

static std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
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
    auto output_shape = output_tensor.get_logical_shape();
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
    tt::log_debug(
        "num_tiles_per_core: {}, num_tiles_per_channel: {}, num_tiles_height_per_core: {}",
        num_tiles_per_core,
        num_tiles_per_channel,
        num_tiles_height_per_core);

    tt::log_debug(
        "Tiled Input input_tile_size: {}, output_pagesize_bytes: {}", input_single_tile_size, output_row_size_bytes);
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_output_sticks_per_dim[0] = tt::div_up(actual_output_shape[-1], TILE_WIDTH);
    num_output_sticks_per_dim[1] = tt::div_up(actual_output_shape[-2], TILE_HEIGHT);

    num_input_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = num_output_sticks_per_dim[0];
    accumulated_total_per_dim[1] = num_output_sticks_per_dim[0] * num_output_sticks_per_dim[1];

    tt::log_debug("Output Shape : {}, Input Shape : {}", actual_output_shape, input_shape);
    for (int32_t i = 2; i < num_dims; i++) {
        uint32_t num_output_dim = actual_output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_input_dim = (num_total_dim - num_output_dim) * accumulated_total_per_dim[i - 1];
        num_output_sticks_per_dim[i] = num_output_dim;
        num_input_sticks_per_dim[i] = num_input_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    for (int i = 0; i < num_dims; i++) {
        tt::log_debug(
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
        num_dims,
        0,
        num_tiles_per_core,
        num_tiles_per_channel};

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    const auto num_sticks_per_core = output_shard_spec.shape[0];
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_sticks_per_core != 0) {
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, output_row_size_bytes_offset, max_read_size);
        num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;
    }
    tt::log_debug(
        "num_stick_per_core: {}, num_stick_per_core_read: {}, num_read_per_barrier: {}",
        num_sticks_per_core,
        num_sticks_per_core_read,
        num_read_per_barrier);

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, output_tensor_start);
    tt::log_debug("Start Offset: {}", start_offset);
    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        }

        const uint32_t num_sticks_written = core_h_index * num_sticks_per_core;
        const uint32_t num_tiles_written = num_tiles_height_per_core * core_h_index * num_tiles_per_channel;

        const uint32_t width_offset = core_w_index * output_row_size_bytes_offset;

        id_per_dim[0] = num_tiles_written % num_output_sticks_per_dim[0];
        uint32_t output_written = num_tiles_written / num_output_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = output_written % num_output_sticks_per_dim[j];
            output_written = output_written / num_output_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        tt::log_debug("For Core {}, Start Offset: {}", core, start_id);

        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[0] += width_offset;

        uint32_t addr_offset = 2;
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> writer_kernel_args = {num_tiles_per_core, num_tiles_per_channel};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

static operation::ProgramWithCallbacks padded_slice_tile_multi_core(
    const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    const ttnn::Shape output_shape = output.get_logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    const ttnn::Shape input_padded_shape = a.get_padded_shape();
    TT_FATAL(
        input_padded_shape.rank() == 4, "Input tensor must be rank 4 for padded_slice operation with tiled inputs");
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

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

    bool is_block_sharded = output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();

    CoreRangeSet total_cores = output.shard_spec().value().grid;
    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);
    uint32_t num_cores_total = total_cores.num_cores();

    uint32_t num_cores_channels = 1;
    if (is_block_sharded) {
        if (rm_orientation) {
            num_cores_channels = total_cores.bounding_box().grid_size().x;
        } else {
            num_cores_channels = total_cores.bounding_box().grid_size().y;
        }
        TT_FATAL(
            num_tiles_per_channel % num_cores_channels == 0,
            "Number of tiles in channel dimension {} must be divisible by num_cores_channels {} for padded_slice "
            "operation with tiled inputs",
            num_tiles_per_channel,
            num_cores_channels);
        num_tiles_per_channel = num_tiles_per_channel / num_cores_channels;
    }

    bool pad_output_row = false;
    uint32_t num_tiles_height_per_core = tt::div_up(output_shard_spec.shape[0], tt::constants::TILE_HEIGHT);
    uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(
        dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    uint32_t dst_stick_size = output_row_size_bytes;

    uint32_t src0_cb_index = 0;
    uint32_t temp_pad_cb_index = 1;
    uint32_t max_read_size = 4096;

    auto src_buffer_alignment = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    // if begins is not aligned then we need to pad the cb size, so that we can read from the nearest aligned address
    uint32_t begins_bytes = output_tensor_start[-1] * a.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    uint32_t cb_page_size = tt::round_up(output_row_size_bytes, alignment);

    auto cb_input_tuple = tt::tt_metal::create_cb(
        cb_input_index,
        program,
        total_cores,
        input_single_tile_size,
        cb_buffer_size * num_tiles_per_channel,
        input_cb_data_format);

    auto cb_untilized_tuple = tt::tt_metal::create_cb(
        cb_untilized_index,
        program,
        total_cores,
        output_single_tile_size,
        cb_buffer_size * num_tiles_per_channel,
        output_cb_data_format);

    tt::log_debug(
        "output_row_size_bytes: {}, num_output_sticks_per_core: {}", output_row_size_bytes, num_output_sticks_per_core);
    auto cb_output_tuple = tt::tt_metal::create_cb(
        2,
        program,
        total_cores,
        output_row_size_bytes,
        num_output_sticks_per_core,
        output_cb_data_format,
        output.buffer());

    tt::log_debug(
        tt::LogOp,
        "num_tiles_height_per_core: {}, num_tiles_per_channel: {}",
        num_tiles_height_per_core,
        num_tiles_per_channel);
    std::vector<uint32_t> compute_args = {
        num_tiles_height_per_core,  // per_core_block_cnt
        num_tiles_per_channel,      // per_block_ntiles
        cb_input_index,             // src0_cb_index
        cb_untilized_index          // untilized_cb_index
    };
    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_channel > MAX_PACK_UNTILIZE_WIDTH || a.get_dtype() == DataType::UINT16) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }

    auto untilize_compute_kernel_id = CreateKernel(
        program, compute_kernel, total_cores, ComputeConfig{.fp32_dest_acc_en = false, .compile_args = compute_args});

    std::vector<uint32_t> writer_compile_time_args_vec = {cb_untilized_index, cb_output_index};

    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_is_dram, misalignment};
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
        a, output, output_tensor_start, actual_output_shape, iter_cores, max_read_size);

    uint32_t i = 0;
    tt::log_debug("Reader args : {}", all_runtime_args[i].first);
    tt::log_debug("Writer args : {}", all_runtime_args[i].second);
    for (const auto& core : iter_cores) {
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
        i++;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           output_tensor_start,
                                           actual_output_shape,
                                           compute_with_storage_grid_size,
                                           max_read_size,
                                           iter_cores,
                                           cb_output_tuple](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        TT_FATAL(dst_tensor.is_sharded(), "Output tensor must be sharded");
        UpdateDynamicCircularBufferAddress(program, std::get<1>(cb_output_tuple), *dst_tensor.buffer());

        auto all_runtime_args = get_padded_slice_runtime_args_tile_sharded_output(
            src_tensor, dst_tensor, output_tensor_start, actual_output_shape, iter_cores, max_read_size);

        uint32_t i = 0;
        for (const auto& core : iter_cores) {
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            i++;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
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
