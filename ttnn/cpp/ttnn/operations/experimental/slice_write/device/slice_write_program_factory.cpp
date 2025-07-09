// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

#include "slice_write_op.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"
#include "ttnn/operations/experimental/padded_slice/device/padded_slice_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::detail {

const uint32_t cb_input_index = 0;

using SliceWriteRuntimeArgs = std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>;
static SliceWriteRuntimeArgs get_slice_write_runtime_args_rm(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    TT_FATAL(
        input_tensor.element_size() == output_tensor.element_size(),
        "Input & output should have the same element size");
    TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Input & output should have the same dtype");

    uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    uint32_t input_row_size_bytes = input_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    std::string unpadded_sticks_str = "";
    for (auto& i : num_input_sticks_per_dim) {
        unpadded_sticks_str += std::to_string(i) + ", ";
    }
    std::string padded_sticks_str = "";
    for (auto& i : num_output_sticks_per_dim) {
        padded_sticks_str += std::to_string(i) + ", ";
    }
    std::string accumulated_str = "";
    for (auto& i : accumulated_total_per_dim) {
        accumulated_str += std::to_string(i) + ", ";
    }

    using namespace tt::tt_metal::experimental;
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_buffer_alignment);
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    std::vector<uint32_t> common_writer_kernel_args = {
        output_buffer->address() + output_tensor_start[-1] * output_tensor.element_size(),
        output_row_size_bytes,
        input_row_size_bytes,
        input_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};

    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());
    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(output_tensor, output_tensor_start);

    for (uint32_t i = 0, num_sticks_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + (32 - num_sticks_per_core % 32) % 32;
            num_sticks_per_core_read =
                tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, input_row_size_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }
        id_per_dim[0] = num_sticks_read % num_input_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_read / num_input_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_input_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        std::vector<uint32_t> writer_kernel_args = common_writer_kernel_args;

        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_sticks_per_core;
        writer_kernel_args[addr_offset++] = num_sticks_per_core_read;
        writer_kernel_args[addr_offset] = num_read_per_barrier;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> reader_kernel_args = {
            input_buffer->address(),
            input_row_size_bytes,
            input_row_size_bytes_offset,
            num_sticks_per_core,
            num_sticks_per_core_read,
            num_read_per_barrier,
            num_sticks_read,
            0};
        num_sticks_read += num_sticks_per_core;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

static SliceWriteRuntimeArgs get_slice_write_runtime_args_rm_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores,
    uint32_t max_read_size) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < input_shape.rank(); i++) {
        input_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }
    log_debug(tt::LogOp, "Slice Write Input Shape: {}", input_shape);
    auto output_shape = output_tensor.logical_shape();
    log_debug(tt::LogOp, "Slice Write Output Shape: {}", output_shape);

    TT_FATAL(
        input_tensor.element_size() == output_tensor.element_size(),
        "Input & output should have the same element size");
    TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Input & output should have the same dtype");

    TT_FATAL(input_tensor.shard_spec().has_value(), "Input tensor should be sharded");

    auto shard_spec = input_tensor.shard_spec().value();
    auto input_cores = shard_spec.grid;
    auto input_shard_shape = shard_spec.shape;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input_tensor);

    uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    TT_FATAL(
        output_row_size_bytes % num_cores_channels == 0,
        "Output row size {} should be divisible by num_cores_channels {}",
        output_row_size_bytes,
        num_cores_channels);
    uint32_t input_row_size_bytes = input_shard_shape[1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> size_till_end(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;
    accumulated_input_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        accumulated_input_total_per_dim[i] = num_unpadded_dim * accumulated_input_total_per_dim[i - 1];
    }

    std::string unpadded_sticks_str = "";
    for (auto& i : num_input_sticks_per_dim) {
        unpadded_sticks_str += std::to_string(i) + ", ";
    }
    std::string padded_sticks_str = "";
    for (auto& i : num_output_sticks_per_dim) {
        padded_sticks_str += std::to_string(i) + ", ";
    }
    std::string accumulated_str = "";
    for (auto& i : accumulated_total_per_dim) {
        accumulated_str += std::to_string(i) + ", ";
    }
    log_debug(tt::LogOp, "Slice Write Accumulated Sticks: {}", accumulated_str);
    log_debug(tt::LogOp, "Slice Write Unpadded Sticks: {}", unpadded_sticks_str);
    log_debug(tt::LogOp, "Slice Write Padded Sticks: {}", padded_sticks_str);
    log_debug(tt::LogOp, "Accumulated Input : {}", accumulated_input_total_per_dim);

    using namespace tt::tt_metal::experimental;
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_buffer_alignment);
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    log_debug(tt::LogOp, "Output Buffer adddress: {}", output_buffer->address());
    std::vector<uint32_t> common_writer_kernel_args = {
        output_buffer->address() + output_tensor_start[-1] * output_tensor.element_size(),
        output_row_size_bytes,
        input_row_size_bytes,
        input_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};

    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());
    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());

    auto num_cores_total = cores.size();

    auto total_num_input_sticks = input_tensor.physical_volume() / input_shape[-1];
    const auto num_sticks_per_core = shard_spec.shape[0];
    // issue more reads before calling barrier
    const uint32_t num_sticks_per_core_read =
        tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, input_row_size_bytes_offset, max_read_size);
    const uint32_t num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;

    log_debug(
        tt::LogOp,
        "num_sticks_per_core = {}, num_sticks_per_core_read = {}, num_read_per_barrier = {}",
        num_sticks_per_core,
        num_sticks_per_core_read,
        num_read_per_barrier);
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(output_tensor, output_tensor_start);
    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        }
        const uint32_t num_sticks_read = core_h_index * num_sticks_per_core;
        const uint32_t width_offset = core_w_index * input_row_size_bytes;

        id_per_dim[0] = num_sticks_read % num_input_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_read / num_input_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        uint32_t max_num_sticks_this_core = 0;
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_input_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
            size_till_end[j] = output_tensor_end[-1 - j] - output_tensor_start[-1 - j] - id_per_dim[j] - 1;
            max_num_sticks_this_core += size_till_end[j] * accumulated_input_total_per_dim[j - 1];
        }

        std::vector<uint32_t> writer_kernel_args = common_writer_kernel_args;
        writer_kernel_args[0] += width_offset;

        uint32_t num_sticks_this_core = std::min(num_sticks_per_core, max_num_sticks_this_core + 1);

        log_debug(
            tt::LogOp,
            "Start ID: {}, Start ID per dim : {} , Size till end : {} Num Sticks: {} for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            num_sticks_this_core,
            core);
        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset] = num_read_per_barrier;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> reader_kernel_args = {num_sticks_per_core};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

static operation::ProgramWithCallbacks slice_write_rm_sharded_input_multi_core(
    const Tensor& input,
    const Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = input.device();

    auto input_shape = input.logical_shape();
    auto output_shape = output.logical_shape();

    uint32_t num_unpadded_sticks = input.physical_volume() / input_shape[-1];

    TT_FATAL(input.shard_spec().has_value(), "Input tensor should be sharded");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "Input tensor should be height or block sharded");
    bool is_height_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    bool is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    auto shard_spec = input.shard_spec().value();
    auto input_cores = shard_spec.grid;
    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    log_debug(tt::LogOp, "Input cores = {}", input_cores);
    log_debug(tt::LogOp, "Input shard spec = {}", shard_spec);

    auto total_num_input_sticks = input.physical_volume() / input_shape[-1];
    auto num_input_sticks_per_core = shard_spec.shape[0];

    uint32_t output_row_size_bytes = output_shape[-1] * output.element_size();
    uint32_t input_row_size_bytes = shard_spec.shape[1] * input.element_size();

    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_buffer_alignment);

    uint32_t max_read_size = 4096;

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = tt::CBIndex::c_0;

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    TT_FATAL(
        input_cb_data_format == output_cb_data_format,
        "Input & output should have the same data format, {} , {}",
        input_cb_data_format,
        output_cb_data_format);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_sticks_per_core * input_row_size_bytes_offset, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, input_row_size_bytes_offset)
            .set_globally_allocated_address(*input.buffer());

    auto input_cb_handle = tt::tt_metal::CreateCircularBuffer(program, input_cores, cb_src0_config);

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        input_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
        "slice_write_writer_interleaved.cpp",
        input_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    const auto iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);

    auto all_runtime_args = get_slice_write_runtime_args_rm_sharded_input(
        input, output, output_tensor_start, output_tensor_end, iter_cores, max_read_size);

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
        i++;
    }

    auto override_runtime_args_callback = [iter_cores,
                                           unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           output_tensor_start,
                                           output_tensor_end,
                                           max_read_size,
                                           input_cb_handle](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto& src_tensor = input_tensors.at(0);
        const auto& dst_tensor = output_tensors.at(0);

        UpdateDynamicCircularBufferAddress(program, input_cb_handle, *src_tensor.buffer());

        auto all_runtime_args = get_slice_write_runtime_args_rm_sharded_input(
            src_tensor, dst_tensor, output_tensor_start, output_tensor_end, iter_cores, max_read_size);

        uint32_t i = 0;
        for (const auto& core : iter_cores) {
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            i++;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

static SliceWriteRuntimeArgs get_slice_write_runtime_args_tiled_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto actual_input_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < actual_input_shape.rank(); i++) {
        actual_input_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }
    auto output_shape = output_tensor.padded_shape();
    log_debug(tt::LogOp, "Slice Write Output Shape: {}", output_shape);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    auto shard_spec = input_tensor.shard_spec().value();
    auto input_cores = shard_spec.grid;
    auto input_shard_shape = shard_spec.shape;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    bool is_width_sharded = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input_tensor);

    uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    TT_FATAL(
        output_row_size_bytes % num_cores_channels == 0,
        "Output row size {} should be divisible by num_cores_channels {}",
        output_row_size_bytes,
        num_cores_channels);
    uint32_t input_row_size_bytes = input_shard_shape[1] * input_tensor.element_size();

    uint32_t num_tiles_per_channel = tt::div_up(output_shape[3], tt::constants::TILE_WIDTH);
    num_tiles_per_channel = num_tiles_per_channel / num_cores_channels;

    std::uint32_t num_dims = static_cast<std::uint32_t>(actual_input_shape.rank());
    std::vector<uint32_t> num_output_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_input_tiles_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> size_till_end(num_dims);

    num_input_tiles_per_dim[0] = actual_input_shape[-1] / (TILE_WIDTH * num_cores_channels);
    num_input_tiles_per_dim[1] = tt::div_up(actual_input_shape[-2], TILE_HEIGHT);

    num_output_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH) - num_input_tiles_per_dim[0];
    num_output_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) - num_input_tiles_per_dim[1];
    num_output_tiles_per_dim[1] *= tt::div_up(output_shape[-1], TILE_WIDTH);

    log_debug(
        tt::LogOp,
        "Output Start : {}, Output End : {}, Actual Input Shape : {}, \n Input Shape : {}, Output Shape : {}",
        output_tensor_start,
        output_tensor_end,
        actual_input_shape,
        input_shape,
        output_shape);

    accumulated_total_tiles_per_dim[0] = tt::div_up(output_shape[-1], TILE_WIDTH);
    accumulated_total_tiles_per_dim[1] = tt::div_up(output_shape[-2], TILE_HEIGHT) * accumulated_total_tiles_per_dim[0];

    accumulated_input_total_tiles_per_dim[0] = actual_input_shape[-1] / TILE_WIDTH;
    accumulated_input_total_tiles_per_dim[1] =
        (actual_input_shape[-2] / TILE_HEIGHT) * accumulated_input_total_tiles_per_dim[0];
    for (int32_t i = 2; i < num_dims; i++) {
        uint32_t num_unpadded_dim = actual_input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_tiles_per_dim[i - 1];
        num_input_tiles_per_dim[i] = num_unpadded_dim;
        num_output_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_tiles_per_dim[i] = num_total_dim * accumulated_total_tiles_per_dim[i - 1];
        accumulated_input_total_tiles_per_dim[i] = num_unpadded_dim * accumulated_input_total_tiles_per_dim[i - 1];
    }

    log_debug(
        tt::LogOp,
        "Slice Write Input Tiles {}, Output Tiles {}, Acc Output Tiles {}, Acc Input Tiles {}",
        num_input_tiles_per_dim,
        num_output_tiles_per_dim,
        accumulated_total_tiles_per_dim,
        accumulated_input_total_tiles_per_dim);

    using namespace tt::tt_metal::experimental;
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_buffer_alignment);
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    log_debug(tt::LogOp, "Output Buffer adddress: {}", output_buffer->address());
    std::vector<uint32_t> common_writer_kernel_args = {
        output_buffer->address(),
        input_single_tile_size,
        input_single_tile_size,
        input_single_tile_size,
        num_dims,
        0,
        0,
        0,
        0};

    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_input_tiles_per_dim.begin(), num_input_tiles_per_dim.end());
    common_writer_kernel_args.insert(
        common_writer_kernel_args.end(), num_output_tiles_per_dim.begin(), num_output_tiles_per_dim.end());

    auto num_cores_total = cores.size();

    TT_FATAL(
        shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Shard Height {} should be a multiple of tile height",
        shard_spec.shape[0]);
    const auto num_tiles_nhw_per_core = shard_spec.shape[0] / TILE_HEIGHT;

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(output_tensor, output_tensor_start);
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

        id_per_dim[0] = 0;
        uint32_t unpadded_written = num_sticks_read;
        uint32_t start_id = id_per_dim[0] + start_offset + width_offset;
        uint32_t max_num_tiles_this_core = 0;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_input_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_tiles_per_dim[j - 1];
            size_till_end[j] = num_input_tiles_per_dim[j] - id_per_dim[j] - ((j == 1) ? 0 : 1);
            max_num_tiles_this_core += size_till_end[j] * accumulated_input_total_tiles_per_dim[j - 1];
        }
        std::vector<uint32_t> writer_kernel_args = common_writer_kernel_args;

        uint32_t num_tiles_this_core =
            std::min(num_tiles_nhw_per_core * num_tiles_per_channel, max_num_tiles_this_core);

        log_debug(
            tt::LogOp,
            "Start ID: {}, Start ID per dim : {} , Size till end : {}, Max Tiles: {}, Num Tiles: {} for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            max_num_tiles_this_core,
            num_tiles_this_core,
            core);
        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_tiles_this_core;
        writer_kernel_args[addr_offset++] = num_tiles_this_core;
        writer_kernel_args[addr_offset] = 1;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> reader_kernel_args = {num_tiles_this_core};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

static operation::ProgramWithCallbacks slice_write_tiled_sharded_input_multi_core(
    const Tensor& input,
    const Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = input.device();
    const auto& output_padded_shape = output.padded_shape();
    const auto& input_padded_shape = input.padded_shape();

    auto input_shape = input.logical_shape();
    auto output_shape = output.logical_shape();
    auto actual_input_shape = input_shape;
    for (int index = 0; index < input_shape.rank(); index++) {
        actual_input_shape[index] = output_tensor_end[index] - output_tensor_start[index];
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_cb_data_format);

    log_debug(tt::LogOp, "Slice Write Input Shape : {} ,Actual Input Shape: {}", input_shape, input_shape);
    TT_FATAL(input.dtype() == output.dtype(), "Input & output should have the same dtype");
    TT_FATAL(output_tensor_start[-1] == 0, "Slice write expects output start for the last dimension to be 0");
    TT_FATAL(
        output_tensor_start[-2] % TILE_HEIGHT == 0,
        "Slice write expects output start for the second last dimension to be a multiple of tile height");

    TT_FATAL(
        input_padded_shape[-2] % TILE_HEIGHT == 0,
        "Slice write expects input shape for the second last dimension to be a multiple of tile height");

    TT_FATAL(
        input.layout() == Layout::TILE,
        "Slice write expects input tensor to be in TILE layout, got {}",
        input.layout());
    TT_FATAL(
        output.layout() == Layout::TILE,
        "Slice write expects output tensor to be in TILE layout, got {}",
        output.layout());

    TT_FATAL(input.shard_spec().has_value(), "Input tensor should be sharded");

    bool is_height_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    bool is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    bool is_width_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    auto shard_spec = input.shard_spec().value();
    auto input_cores = shard_spec.grid;
    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    log_debug(tt::LogOp, "Input cores = {}", input_cores);
    log_debug(tt::LogOp, "Input shard spec = {}", shard_spec);

    TT_FATAL(
        shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Slice write needs tiled inputs, where the shard height {} is a multiple of tile height {}",
        shard_spec.shape[0],
        TILE_HEIGHT);

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = tt::CBIndex::c_0;

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    uint32_t num_tiles_height_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_tiles_channel_per_core = shard_spec.shape[1] / TILE_HEIGHT;

    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(input);

    TT_FATAL(
        num_tiles_channel_per_core * TILE_WIDTH * num_cores_channels == output_padded_shape[-1],
        "Number of tiles per channel {} * tile width {} * num_cores_channels {} should be equal to output shape "
        "last dimension {}",
        num_tiles_channel_per_core,
        TILE_WIDTH,
        num_cores_channels,
        output_shape[-1]);

    TT_FATAL(
        input_cb_data_format == output_cb_data_format,
        "Input & output should have the same data format, {} , {}",
        input_cb_data_format,
        output_cb_data_format);

    auto cb_input_tuple = tt::tt_metal::create_cb(
        cb_input_index,
        program,
        input_cores,
        input_single_tile_size,
        num_tiles_height_per_core * num_tiles_channel_per_core,
        input_cb_data_format,
        input.buffer());

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        input_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
        "slice_write_writer_interleaved.cpp",
        input_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    const auto iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);

    auto all_runtime_args = get_slice_write_runtime_args_tiled_sharded_input(
        input, output, output_tensor_start, output_tensor_end, iter_cores);

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
        i++;
    }

    auto override_runtime_args_callback = [iter_cores,
                                           unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           output_tensor_start,
                                           output_tensor_end,
                                           cb_input_tuple](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto& src_tensor = input_tensors.at(0);
        auto& dst_tensor = output_tensors.at(0);

        UpdateDynamicCircularBufferAddress(program, std::get<1>(cb_input_tuple), *src_tensor.buffer());

        auto all_runtime_args = get_slice_write_runtime_args_tiled_sharded_input(
            src_tensor, dst_tensor, output_tensor_start, output_tensor_end, iter_cores);

        uint32_t i = 0;
        for (const auto& core : iter_cores) {
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            i++;
        }
    };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

static operation::ProgramWithCallbacks slice_write_rm_interleaved_multi_core(
    const Tensor& input,
    const Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = input.device();
    const auto& output_padded_shape = output.padded_shape();
    const auto& input_padded_shape = input.padded_shape();

    uint32_t num_unpadded_sticks = input.physical_volume() / input.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    uint32_t output_row_size_bytes = output_padded_shape[-1] * output.element_size();
    uint32_t input_row_size_bytes = input_padded_shape[-1] * input.element_size();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t src_stick_size = input_row_size_bytes;
    uint32_t dst_stick_size = output_row_size_bytes;

    uint32_t max_read_size = 4096;

    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    // if begins is not aligned then we need to pad the cb size, so that we can read from the nearest aligned address
    uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t cb_page_size = tt::round_up(input_row_size_bytes, alignment);

    uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                                         : num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_input_pages != 0) {
        // Round up num_input_pages so that it takes the max of both core groups.
        auto num_input_pages_pad32 = tt::round_up(num_input_pages, 32);
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_input_pages_pad32, cb_page_size, max_read_size);
        num_read_per_barrier = num_input_pages_pad32 / num_sticks_per_core_read;
    }
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_read_per_barrier * 2 * cb_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_cb_index, src0_is_dram};
    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
        "slice_write_reader_interleaved.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
        "slice_write_writer_interleaved.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    auto all_runtime_args = get_slice_write_runtime_args_rm(
        input,
        output,
        output_tensor_start,
        num_cores_total,
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        max_read_size);

    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
    }

    auto override_runtime_args_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, compute_with_storage_grid_size, max_read_size](
            const void* operation,
            const Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            const auto& src_tensor = input_tensors.at(0);
            const auto& dst_tensor = output_tensors.at(0);
            uint32_t num_cores_x = compute_with_storage_grid_size.x;
            uint32_t num_cores_y = compute_with_storage_grid_size.y;
            uint32_t num_cores_total = num_cores_x * num_cores_y;
            uint32_t num_unpadded_sticks = src_tensor.physical_volume() / src_tensor.padded_shape()[-1];
            auto
                [num_cores,
                 all_cores,
                 core_group_1,
                 core_group_2,
                 num_sticks_per_core_group_1,
                 num_sticks_per_core_group_2] =
                    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

            const auto tensor_start =
                static_cast<const ttnn::operations::experimental::SliceWriteDeviceOperation*>(operation)->slice_start;
            auto all_runtime_args = get_slice_write_runtime_args_rm(
                src_tensor,
                dst_tensor,
                tensor_start,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                core_group_2,
                num_sticks_per_core_group_1,
                num_sticks_per_core_group_2,
                max_read_size);

            for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; i++) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};
                SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
                SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks slice_write_multi_core(
    const Tensor& a,
    const Tensor& output,
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
    TT_FATAL(!output.is_sharded(), "Sharded output is not supported for slice_write operation");
    TT_FATAL(!has_step, "Step is not supported for slice_write operation");
    if (a.is_sharded()) {  // Supports Height & Block Sharding
        if (a.layout() == Layout::ROW_MAJOR) {
            return slice_write_rm_sharded_input_multi_core(a, output, output_tensor_start, output_tensor_end);
        } else if (a.layout() == Layout::TILE) {
            return slice_write_tiled_sharded_input_multi_core(a, output, output_tensor_start, output_tensor_end);
        } else {
            TT_THROW("Unsupported output memory layout for slice_write operation");
        }
    } else if (!a.is_sharded()) {
        return slice_write_rm_interleaved_multi_core(a, output, output_tensor_start, output_tensor_end);
    }
    TT_THROW("Unsupport input memory layout for slice_write operation");
}

}  // namespace ttnn::operations::experimental::detail
