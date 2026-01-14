// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::data_movement::pad::program {
PadRmReaderWriterProgramFactory::cached_program_t PadRmReaderWriterProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    Program program{};

    auto output_shape = operation_attributes.output_padded_shape;

    uint32_t unpadded_row_size_nbytes = tensor_args.input.padded_shape()[3] * tensor_args.input.element_size();
    uint32_t padded_row_size_nbytes =
        output_shape[3] * tensor_args.input.element_size();  // Assuming output is same datatype as input
    TT_ASSERT(
        unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    // construct const buffer with the pad_value
    MeshDevice* device = tensor_args.input.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * tensor_args.input.element_size();
    auto pad_value_const_buffer =
        tt::tt_metal::HostBuffer(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    const Tensor pad_value_const_tensor =
        Tensor(
            std::move(pad_value_const_buffer),
            ttnn::Shape({1, 1, 1, pad_value_const_buffer_size}),
            DataType::BFLOAT16,
            Layout::ROW_MAJOR)
            .to_device(device, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    CoreRange cores({0, 0}, {0, 0});
    uint32_t cb_id = tt::CBIndex::c_0;
    uint32_t cb_npages = 16;  // multibuffering
    uint32_t cb_pagesize =
        tt::round_up(padded_row_size_nbytes, std::max(src0_buffer->alignment(), tt::constants::TILE_WIDTH));
    tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_npages * cb_pagesize, {{cb_id, in_df}}).set_page_size(cb_id, cb_pagesize);
    tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);

    std::vector<uint32_t> reader_ct_args = {unpadded_row_size_nbytes, padded_row_size_nbytes};
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*dst_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(*pad_value_const_tensor.buffer()).append_to(reader_ct_args);
    const std::vector<uint32_t>& writer_ct_args = reader_ct_args;

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({0, float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(0.0f), bfloat16(pad_value)});
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
        cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
        cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));
    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;

#if 0
    {
        log_debug(tt::LogOp, "src0_buffer_addr: {}", src0_buffer->address());
        log_debug(tt::LogOp, "dst_buffer_addr: {}", dst_buffer->address());
        log_debug(tt::LogOp, "a.shape[0]: {}", a.padded_shape()[0]);
        log_debug(tt::LogOp, "out.shape[0]: {}", output_shape[0]);
        log_debug(tt::LogOp, "a.shape[1]: {}", a.padded_shape()[1]);
        log_debug(tt::LogOp, "out.shape[1]: {}", output_shape[1]);
        log_debug(tt::LogOp, "a.shape[2]: {}", a.padded_shape()[2]);
        log_debug(tt::LogOp, "out.shape[2]: {}", output_shape[2]);
        log_debug(tt::LogOp, "s.shape[3]: {}", a.padded_shape()[3]);
        log_debug(tt::LogOp, "out.shape[3]: {}", output_shape[3]);
        log_debug(tt::LogOp, "unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug(tt::LogOp, "padded_row_size_nbytes: {}", padded_row_size_nbytes);
        log_debug(tt::LogOp, "padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug(tt::LogOp, "pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        log_debug(tt::LogOp, "pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug(tt::LogOp, "packed_pad_value: {}", packed_pad_value);
    }
#endif

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    const std::array reader_rt_args = {
        src0_buffer->address(),
        dst_buffer->address(),
        a.padded_shape()[0],
        output_shape[0],
        a.padded_shape()[1],
        output_shape[1],
        a.padded_shape()[2],
        output_shape[2],
        a.padded_shape()[3],
        output_shape[3],
        unpadded_row_size_nbytes,
        padded_row_size_nbytes,
        padded_row_diff_size_nbytes,
        pad_value_const_tensor_addr,
        pad_value_const_buffer_nbytes,
        packed_pad_value,
        start_src_stick_id,
        start_dst_stick_id,
        std::uint32_t{0},
        std::uint32_t{0},
        std::uint32_t{0},
        output_shape[2],
        a.padded_shape()[2],
        unpadded_row_size_nbytes,
        padded_row_size_nbytes,
        std::uint32_t{0},
        output.padded_shape()[0]};
    const auto& writer_rt_args = reader_rt_args;
    tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, reader_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, writer_rt_args);

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_id}};
}

void PadRmReaderWriterProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    CoreCoord core = {0, 0};
    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        runtime_args[1] = dst_buffer->address();
    }
    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        runtime_args[1] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::data_movement::pad::program
