// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t num_pages(const ttnn::Tensor& input_tensor) {
    const auto& padded_shape = input_tensor.get_logical_shape();
    return padded_shape.volume() / padded_shape[-1];
}

uint32_t page_size(const ttnn::Tensor& input_tensor) {
    const auto& padded_shape = input_tensor.get_logical_shape();  // in anticipation of RM padding
    return padded_shape[-1] * input_tensor.element_size();
}

std::vector<uint32_t> get_row_strides(const ttnn::SimpleShape& shape) {
    std::vector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    strides[shape.rank() - 2] = 1;
    for (int i = shape.rank() - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
}  // namespace detail

PermuteDeviceOperation::SingleCore::cached_program_t PermuteDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t input_rm_page_size = detail::page_size(input_tensor);

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t output_rm_page_size = detail::page_size(tensor_return_value);

    uint32_t num_input_pages = detail::num_pages(input_tensor);

    tt::tt_metal::Device* device = input_tensor.device();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_pages_to_read = 1;

    CoreRange core({0, 0}, {0, 0});
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_pages_to_read * input_rm_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_rm_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t N = operation_attributes.dims.size();
    uint32_t num_rows = input_tensor.volume() / input_tensor.get_logical_shape()[-1];

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram, N, input_rm_page_size, num_rows};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/reader_permute_interleaved_rm.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)dst_is_dram, N, output_rm_page_size, num_rows};
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/writer_permute_interleaved_rm.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {src_buffer->address()};

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

    auto input_shape_view = input_tensor.get_logical_shape().view();
    auto output_strides = detail::get_row_strides(output_tensor.get_logical_shape());  // in anticipation of RM padding

    std::vector<uint32_t> writer_runtime_args = {dst_buffer->address()};
    writer_runtime_args.insert(writer_runtime_args.end(), input_shape_view.begin(), input_shape_view.end());
    writer_runtime_args.insert(
        writer_runtime_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
    writer_runtime_args.insert(writer_runtime_args.end(), output_strides.begin(), output_strides.end());

    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id}};
}

void PermuteDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement
