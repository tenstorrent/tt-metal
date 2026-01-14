// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_rm_sharded_input_program_factory.hpp"

#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::slice_write::program {

namespace {

SliceWriteRuntimeArgs get_slice_write_runtime_args_rm_sharded_input(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const std::vector<CoreCoord>& cores,
    uint32_t max_read_size) {
    auto* output_buffer = output_tensor.buffer();
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

    uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    uint32_t input_row_size_bytes = input_shard_shape[1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<int> size_till_end(num_dims);

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

    std::string unpadded_sticks_str;
    for (auto& i : num_input_sticks_per_dim) {
        unpadded_sticks_str += std::to_string(i) + ", ";
    }
    std::string padded_sticks_str;
    for (auto& i : num_output_sticks_per_dim) {
        padded_sticks_str += std::to_string(i) + ", ";
    }
    std::string accumulated_str;
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
        output_buffer->address() + (output_tensor_start[-1] * output_tensor.element_size()),
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
    SliceWriteRuntimeArgs ret_val(num_cores_total);

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
        int max_num_sticks_this_core = 0;
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_sticks_per_dim[j];
            if (j == num_dims - 1 && unpadded_written == num_input_sticks_per_dim[j]) {
                // Handle edge case where last dimension is completely written
                id_per_dim[j] = num_input_sticks_per_dim[j];
            }
            unpadded_written = unpadded_written / num_input_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
            size_till_end[j] = output_tensor_end[-1 - j] - output_tensor_start[-1 - j] - id_per_dim[j] - 1;
            max_num_sticks_this_core += size_till_end[j] * accumulated_input_total_per_dim[j - 1];
        }

        uint32_t this_input_row_size_bytes = std::min(input_row_size_bytes, output_row_size_bytes - width_offset);
        WriterKernelArgs writer_kernel_args = common_writer_kernel_args;
        writer_kernel_args[0] += width_offset;
        writer_kernel_args[2] = this_input_row_size_bytes;

        uint32_t num_sticks_this_core =
            std::min<uint32_t>(num_sticks_per_core, std::max<int>(max_num_sticks_this_core + 1, 0));

        log_trace(
            tt::LogOp,
            "Start ID: {}, Start ID per dim : {} , Size till end : {} Num Sticks: {}, this_input_row_size_bytes: {} "
            "for Core: {}",
            start_id,
            id_per_dim,
            size_till_end,
            num_sticks_this_core,
            this_input_row_size_bytes,
            core);
        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset] = num_read_per_barrier;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        ReaderKernelArgs reader_kernel_args = {num_sticks_per_core};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}
}  // namespace

SliceWriteRMShardedInputProgramFactory::cached_program_t SliceWriteRMShardedInputProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& output_tensor_start = operation_attributes.slice_start;
    const auto& output_tensor_end = operation_attributes.slice_end;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto input_shape = input.logical_shape();
    auto output_shape = output.logical_shape();

    TT_FATAL(input.shard_spec().has_value(), "Input tensor should be sharded");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
        "Input tensor should be height or block sharded");
    auto shard_spec = input.shard_spec().value();
    auto input_cores = shard_spec.grid;
    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    log_debug(tt::LogOp, "Input cores = {}", input_cores);
    log_debug(tt::LogOp, "Input shard spec = {}", shard_spec);

    auto num_input_sticks_per_core = shard_spec.shape[0];

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

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, 0};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args_vec);

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

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .iter_cores = iter_cores,
            .unary_reader_kernel_id = unary_reader_kernel_id,
            .unary_writer_kernel_id = unary_writer_kernel_id,
            .output_tensor_start = output_tensor_start,
            .output_tensor_end = output_tensor_end,
            .max_read_size = max_read_size,
            .input_cb_handle = input_cb_handle});
}

void SliceWriteRMShardedInputProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const auto& src_tensor = tensor_args.input;
    const auto& dst_tensor = tensor_return_value;

    UpdateDynamicCircularBufferAddress(
        cached_program.program, cached_program.shared_variables.input_cb_handle, *src_tensor.buffer());

    auto all_runtime_args = get_slice_write_runtime_args_rm_sharded_input(
        src_tensor,
        dst_tensor,
        cached_program.shared_variables.output_tensor_start,
        cached_program.shared_variables.output_tensor_end,
        cached_program.shared_variables.iter_cores,
        cached_program.shared_variables.max_read_size);

    uint32_t i = 0;
    for (const auto& core : cached_program.shared_variables.iter_cores) {
        tt::tt_metal::SetRuntimeArgs(
            cached_program.program,
            cached_program.shared_variables.unary_reader_kernel_id,
            core,
            all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(
            cached_program.program,
            cached_program.shared_variables.unary_writer_kernel_id,
            core,
            all_runtime_args[i].second);
        i++;
    }
}

}  // namespace ttnn::operations::experimental::slice_write::program
