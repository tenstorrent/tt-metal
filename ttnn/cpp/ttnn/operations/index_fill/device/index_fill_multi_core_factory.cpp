// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "index_fill_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt::tt_metal::detail;

namespace ttnn::operations::index_fill {
IndexFillOperation::MultiCore::cached_program_t IndexFillOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& index = tensor_args.index;
    const Tensor& input = tensor_args.input;
    uint32_t dim = operation_attributes.dim;

    auto dtype = input.dtype();

    const auto input_shape = input.logical_shape();
    const auto n = input_shape.rank();

    uint32_t num_rows_to_fill_per_index = 1;
    for (int i = n - 2; i > dim; i--) {
        num_rows_to_fill_per_index *= input_shape[i];
    }

    auto fill_value_ = operation_attributes.value;
    uint32_t fill_value{};
    switch (dtype) {
        case DataType::BFLOAT16:
            fill_value = pack_two_bfloat16_into_uint32({bfloat16(std::get<float>(fill_value_)), bfloat16(0.0f)});
            break;
        case DataType::FLOAT32: fill_value = std::bit_cast<uint32_t>(std::get<float>(fill_value_)); break;
        case DataType::INT32: fill_value = static_cast<uint32_t>(std::get<int>(fill_value_)); break;
        default: TT_FATAL(false, "Unsupported datatype"); break;
    }

    auto num_rows = input.physical_volume() / input.padded_shape()[-1];
    Program program{};
    IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    auto input_data_format = datatype_to_dataformat_converter(dtype);
    auto index_data_format = datatype_to_dataformat_converter(index.dtype());

    uint32_t input_page_size = input.buffer()->aligned_page_size();
    uint32_t index_page_size = index.buffer()->aligned_page_size();
    uint32_t output_page_size = output.buffer()->aligned_page_size();

    auto src_cb_index = CBIndex::c_0;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(input_page_size, {{src_cb_index, input_data_format}})
            .set_page_size(src_cb_index, input_page_size));

    auto index_cb_index = CBIndex::c_1;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(index_page_size, {{index_cb_index, index_data_format}})
            .set_page_size(index_cb_index, index_page_size));

    // Create Kernels
    // reader
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src_cb_index,
        (std::uint32_t)index_cb_index,
        (std::uint32_t)(dim == n - 1),
        (std::uint32_t)index.physical_volume(),
        (std::uint32_t)input_page_size,
        (std::uint32_t)index_page_size,
        (std::uint32_t)input.element_size(),
        (std::uint32_t)input.padded_shape()[-1]};
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(index.buffer()).append_to(reader_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/index_fill/device/kernels/reader_index_fill.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_page_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/index_fill/device/kernels/writer_index_fill.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    uint32_t unit_offset = 0;
    uint32_t num_cores_group_1 = core_group_1.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);
    for (uint32_t i = 0; i < cores.size(); i++) {
        const auto& core = cores[i];
        uint32_t num_rows_per_core = i < num_cores_group_1 ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {input.buffer()->address(),
             index.buffer()->address(),
             fill_value,
             unit_offset,
             num_rows_per_core,
             num_rows_to_fill_per_index,
             input_shape[dim]});
        SetRuntimeArgs(program, writer_kernel_id, core, {output.buffer()->address(), num_rows_per_core, unit_offset});

        unit_offset += num_rows_per_core;
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, num_cores_y}};
}

void IndexFillOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto src_buffer = tensor_args.input.buffer()->address();
    auto index_buffer = tensor_args.index.buffer()->address();
    auto output_buffer = output.buffer()->address();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer;
            runtime_args[1] = index_buffer;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_buffer;
        }
    }
}

}  // namespace ttnn::operations::index_fill
