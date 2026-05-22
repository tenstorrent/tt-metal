// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_rm_sharded_input_program_factory.hpp"

#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ProgramDescriptor SliceWriteRMShardedInputProgramFactory::create_descriptor(
    const SliceWriteParams& operation_attributes, const SliceWriteInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& output_tensor_start = operation_attributes.slice_start;
    const auto& output_tensor_end = operation_attributes.slice_end;

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
    bool is_block_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

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
    std::vector<int> size_till_end(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    std::vector<uint32_t> accumulated_input_total_per_dim(num_dims);
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;
    accumulated_input_total_per_dim[0] = 1;

    // Override input_shape per-dim using slice extents for per-dim arithmetic below.
    auto input_shape_for_dims = input_shape;
    for (uint32_t i = 0; i < input_shape_for_dims.rank(); i++) {
        input_shape_for_dims[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = input_shape_for_dims[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        accumulated_input_total_per_dim[i] = num_unpadded_dim * accumulated_input_total_per_dim[i - 1];
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    TT_FATAL(
        input_cb_data_format == output_cb_data_format,
        "Input & output should have the same data format, {} , {}",
        input_cb_data_format,
        output_cb_data_format);

    ProgramDescriptor desc;
    // Sharded CB bound to the input buffer (was set_globally_allocated_address).
    CBDescriptor cb_src0_desc{
        .total_size = num_input_sticks_per_core * input_row_size_bytes_offset,
        .core_ranges = input_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_row_size_bytes_offset,
        }}},
        .buffer = input.buffer(),
    };
    desc.cbs.push_back(std::move(cb_src0_desc));

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, 0};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args_vec);

    KernelDescriptor reader_desc{
        .kernel_source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = input_cores,
        .compile_time_args = std::move(reader_compile_time_args),
        .config = ReaderConfigDescriptor{},
    };

    KernelDescriptor writer_desc{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
            "slice_write_writer_interleaved.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = input_cores,
        .compile_time_args = std::move(writer_compile_time_args_vec),
        .config = WriterConfigDescriptor{},
    };

    const auto iter_cores = corerange_to_cores(input_cores, std::nullopt, rm_orientation);

    // Per-core arg computation. We inline rather than calling a helper since the helper used
    // raw pair-of-vector form; here we build kernel descriptors directly.
    auto* output_buffer = output.buffer();

    auto input_shape_for_args = input.logical_shape();
    for (uint32_t i = 0; i < input_shape_for_args.rank(); i++) {
        input_shape_for_args[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    uint32_t output_row_size_bytes = output_shape[-1] * input.element_size();
    uint32_t input_row_size_bytes_local = shard_spec.shape[1] * input.element_size();
    TT_FATAL(
        output_tensor_start[-1] == 0,
        "slice_write expects output start for the last dimension to be 0. Got {}",
        output_tensor_start[-1]);

    std::vector<uint32_t> common_writer_kernel_args = {
        output_buffer->address() + (output_tensor_start[-1] * output.element_size()),
        output_row_size_bytes,
        input_row_size_bytes_local,
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

    const auto num_sticks_per_core = shard_spec.shape[0];
    const uint32_t num_sticks_per_core_read =
        tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core, input_row_size_bytes_offset, max_read_size);
    const uint32_t num_read_per_barrier = num_sticks_per_core / num_sticks_per_core_read;

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(output, output_tensor_start);
    uint32_t core_index = 0;
    for (const auto& core : iter_cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        }
        const uint32_t num_sticks_read = core_h_index * num_sticks_per_core;
        const uint32_t width_offset = core_w_index * input_row_size_bytes_local;

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

        uint32_t this_input_row_size_bytes = std::min(input_row_size_bytes_local, output_row_size_bytes - width_offset);
        std::vector<uint32_t> writer_kernel_args = common_writer_kernel_args;
        writer_kernel_args[0] += width_offset;
        writer_kernel_args[2] = this_input_row_size_bytes;

        uint32_t num_sticks_this_core =
            std::min<uint32_t>(num_sticks_per_core, std::max<int>(max_num_sticks_this_core + 1, 0));

        uint32_t addr_offset = 5;
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset++] = num_sticks_this_core;
        writer_kernel_args[addr_offset] = num_read_per_barrier;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_sticks_per_core});
        writer_desc.runtime_args.emplace_back(core, std::move(writer_kernel_args));
        core_index++;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::experimental::prim
