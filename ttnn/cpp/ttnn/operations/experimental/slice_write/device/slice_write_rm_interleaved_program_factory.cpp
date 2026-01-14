// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_write_rm_interleaved_program_factory.hpp"

#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "slice_write_device_operation_types.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::slice_write::program {

namespace {

SliceWriteRuntimeArgs get_slice_write_runtime_args_rm(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& stride,
    uint32_t num_cores_total,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    TT_FATAL(
        input_tensor.element_size() == output_tensor.element_size(),
        "Input & output should have the same element size");
    TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Input & output should have the same dtype");

    uint32_t output_row_size_bytes = output_shape[-1] * input_tensor.element_size();
    uint32_t input_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    bool strided = std::any_of(stride.cbegin(), stride.cend(), [](int val) { return val != 1; });

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> rev_stride(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_input_sticks_per_dim[0] = 1;
    num_output_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;
    rev_stride[0] = stride[num_dims - 1];

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = input_shape[-(i + 1)];
        uint32_t num_total_dim = output_shape[-(i + 1)];
        rev_stride[i] = stride[num_dims - (i + 1)];
        uint32_t num_padded_dim;
        if (strided) {
            uint32_t dims_traversed = (rev_stride[i] * (num_unpadded_dim - 1));
            uint32_t num_dims_to_skip = (num_total_dim - dims_traversed);
            num_padded_dim = num_dims_to_skip * accumulated_total_per_dim[i - 1];
        } else {
            num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        }

        num_input_sticks_per_dim[i] = num_unpadded_dim;
        num_output_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
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

    std::string rev_stride_str;
    for (auto& i : rev_stride) {
        rev_stride_str += std::to_string(i) + ", ";
    }

    using namespace tt::tt_metal::experimental;
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    uint32_t input_row_size_bytes_offset = tt::round_up(input_row_size_bytes, src_buffer_alignment);

    std::vector<uint32_t> common_writer_kernel_args = {
        output_buffer->address(),
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

    SliceWriteRuntimeArgs ret_val(num_cores_total);

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
            auto num_sticks_per_core_pad32 = num_sticks_per_core + ((32 - num_sticks_per_core % 32) % 32);
            num_sticks_per_core_read =
                tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, input_row_size_bytes, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }
        id_per_dim[0] = num_sticks_read %
                        num_input_sticks_per_dim[0];  // if num_input_sticks_per_dim[0] is always 1, this is always 0
        uint32_t unpadded_written =
            num_sticks_read /
            num_input_sticks_per_dim[0];  // if num_input_sticks_per_dim[0] is always 1, this is always num_sticks_read?
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_input_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_input_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1] * rev_stride[j];
        }
        std::vector<uint32_t> writer_kernel_args = common_writer_kernel_args;

        uint32_t addr_offset = 5;  // output buffer addr, output_row_size_bytes, input_row_size_bytes, num_dims
        writer_kernel_args[addr_offset++] = start_id;
        writer_kernel_args[addr_offset++] = num_sticks_per_core;
        writer_kernel_args[addr_offset++] = num_sticks_per_core_read;
        writer_kernel_args[addr_offset] = num_read_per_barrier;
        writer_kernel_args.insert(writer_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());
        writer_kernel_args.insert(writer_kernel_args.end(), rev_stride.begin(), rev_stride.end());

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
}  // namespace

SliceWriteRMInterleavedProgramFactory::cached_program_t SliceWriteRMInterleavedProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& output_tensor_start = operation_attributes.slice_start;
    const auto& stride = operation_attributes.step;

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input.device();
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

    uint32_t input_row_size_bytes = input_padded_shape[-1] * input.element_size();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

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
    uint32_t page_alignment_offset = begins_bytes % src_buffer_alignment;

    // reader defines
    std::map<std::string, std::string> reader_defines;
    if (page_alignment_offset != 0) {
        reader_defines["LAST_DIM"] = "1";
    }

    const uint32_t src0_cb_index = tt::CBIndex::c_0;  // cb for reading in input
    const uint32_t dst0_cb_index = tt::CBIndex::c_1;  // cb for reading in output pages for last dim striding
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
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    std::map<std::string, std::string> writer_defines;
    if (stride[-1] != 1) {
        writer_defines["LAST_DIM_STRIDED"] = "1";
        uint32_t output_row_size_bytes = input_padded_shape[-1] * input.element_size();
        cb_page_size = tt::round_up(output_row_size_bytes, alignment);
        tt::tt_metal::CircularBufferConfig cb_dst0_config =
            tt::tt_metal::CircularBufferConfig(
                num_read_per_barrier * 2 * cb_page_size,
                {{dst0_cb_index, cb_data_format}})  // input/output data_formats should be the same
                .set_page_size(dst0_cb_index, cb_page_size);
        tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_dst0_config);
    }

    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_cb_index, page_alignment_offset};
    std::vector<uint32_t> writer_compile_time_args_vec = {
        (std::uint32_t)src0_cb_index,
        page_alignment_offset,
        (std::uint32_t)dst0_cb_index,
        begins_bytes,
        output.element_size()};
    tt::tt_metal::TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args_vec);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args_vec);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
        "slice_write_reader_interleaved.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec, reader_defines));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/slice_write/device/kernels/dataflow/"
        "slice_write_writer_interleaved_strided.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec, writer_defines));

    auto all_runtime_args = get_slice_write_runtime_args_rm(
        input,
        output,
        output_tensor_start,
        stride,
        num_cores_total,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        max_read_size);

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
    }

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .unary_reader_kernel_id = unary_reader_kernel_id,
            .unary_writer_kernel_id = unary_writer_kernel_id,
            .compute_with_storage_grid_size = compute_with_storage_grid_size,
            .max_read_size = max_read_size});
}

void SliceWriteRMInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const auto& src_tensor = tensor_args.input;
    const auto& dst_tensor = tensor_return_value;
    uint32_t num_cores_x = cached_program.shared_variables.compute_with_storage_grid_size.x;
    uint32_t num_cores_y = cached_program.shared_variables.compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    uint32_t num_unpadded_sticks = src_tensor.physical_volume() / src_tensor.padded_shape()[-1];
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(
            cached_program.shared_variables.compute_with_storage_grid_size, num_unpadded_sticks);

    const auto& tensor_start = operation_attributes.slice_start;
    const auto& stride = operation_attributes.step;
    auto all_runtime_args = get_slice_write_runtime_args_rm(
        src_tensor,
        dst_tensor,
        tensor_start,
        stride,
        num_cores_total,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        cached_program.shared_variables.max_read_size);

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        SetRuntimeArgs(
            cached_program.program,
            cached_program.shared_variables.unary_reader_kernel_id,
            core,
            all_runtime_args[i].first);
        SetRuntimeArgs(
            cached_program.program,
            cached_program.shared_variables.unary_writer_kernel_id,
            core,
            all_runtime_args[i].second);
    }
}

}  // namespace ttnn::operations::experimental::slice_write::program
