// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "index_fill_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt::tt_metal::detail;

union datatype {
    uint32_t u32;
    float f32;
} u_fill_value;

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

    auto fill_value = operation_attributes.value;
    if (std::holds_alternative<int>(fill_value)) {
        u_fill_value.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        u_fill_value.f32 = std::get<float>(fill_value);
    }

    auto num_rows = input.physical_volume() / input.logical_shape()[-1];
    Program program{};
    IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    auto input_data_format = datatype_to_dataformat_converter(dtype);
    auto index_data_format = datatype_to_dataformat_converter(index.dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());

    uint32_t input_unit_size = input.logical_shape()[-1] * input.element_size();
    uint32_t rounded_input_unit_size = round_up_to_mul32(input_unit_size);

    uint32_t index_unit_size = index.physical_volume() * index.element_size();
    uint32_t rounded_index_unit_size = round_up_to_mul32(index_unit_size);

    uint32_t output_unit_size = output.logical_shape()[-1] * output.element_size();
    uint32_t rounded_output_unit_size = round_up_to_mul32(output_unit_size);

    auto src_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src_config =
        CircularBufferConfig(rounded_input_unit_size, {{src_cb_index, input_data_format}})
            .set_page_size(src_cb_index, rounded_input_unit_size);
    auto cb_src = CreateCircularBuffer(program, all_cores, cb_src_config);
    std::map<string, string> reader_defines;

    switch (dtype) {
        case DataType::BFLOAT16: reader_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: reader_defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: reader_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: TT_FATAL(false, "Unsupported datatype"); break;
    }

    auto index_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_index_config =
        CircularBufferConfig(rounded_index_unit_size, {{index_cb_index, index_data_format}})
            .set_page_size(index_cb_index, rounded_index_unit_size);
    auto cb_index = CreateCircularBuffer(program, all_cores, cb_index_config);

    auto dst_cb_index = CBIndex::c_16;
    CircularBufferConfig dst_cb_config =
        CircularBufferConfig(rounded_output_unit_size, {{dst_cb_index, output_data_format}})
            .set_page_size(dst_cb_index, rounded_output_unit_size);
    auto cb_dst = CreateCircularBuffer(program, all_cores, dst_cb_config);

    bool in_is_dram = input.buffer()->is_dram();
    bool index_is_dram = index.buffer()->is_dram();
    bool out_is_dram = output.buffer()->is_dram();

    // Create Kernels
    // reader
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)in_is_dram,
        (std::uint32_t)index_is_dram,
        (std::uint32_t)src_cb_index,
        (std::uint32_t)index_cb_index,
        (std::uint32_t)(dim == n - 1),
        (std::uint32_t)index.physical_volume()};

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/index_fill/device/kernels/reader_index_fill.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)out_is_dram};

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
             u_fill_value.u32,
             input_unit_size,
             index_unit_size,
             unit_offset,
             num_rows_per_core,
             num_rows_to_fill_per_index,
             input_shape[dim]});
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output.buffer()->address(), num_rows_per_core, unit_offset, output_unit_size});

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
