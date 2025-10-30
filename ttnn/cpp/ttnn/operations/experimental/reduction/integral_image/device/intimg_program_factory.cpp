// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_device_operation.hpp"
#include "intimg_program_factory.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <iostream>

namespace ttnn::operations::experimental::reduction {

IntImgProgramFactory::cached_program_t IntImgProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args.input_tensor};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    constexpr uint32_t block_depth = 32;

    Program program{};

    // IDevice* device{input_tensor.device()};

    auto src_buffer{input_tensor.buffer()};
    auto dst_buffer{output_tensor.buffer()};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(input_tensor.dtype())};
    const bool fp32_dest_acc_en{
        (dst_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Int32) ||
        (dst_cb_data_format == DataFormat::UInt32)};

    const auto tile_spec = input_tensor.tensor_spec().tile();

    const auto core_range_set = CoreRangeSet{{{0, 0}, {0, 0}}};
    create_cb(program, input_tensor.dtype(), IntImgCB::START, core_range_set, 4);
    create_cb(program, input_tensor.dtype(), IntImgCB::INPUT, core_range_set, 4);
    create_cb(program, input_tensor.dtype(), IntImgCB::ACC, core_range_set, 4);
    create_cb(program, input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_0, core_range_set, 32);
    create_cb(program, input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_1, core_range_set, 32);
    create_cb(program, input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_2, core_range_set, 32);
    create_cb(program, input_tensor.dtype(), IntImgCB::CUMSUM_STAGE_3, core_range_set, 32);
    create_cb(
        program,
        input_tensor.dtype(),
        IntImgCB::OUTPUT,
        core_range_set,
        32);  // TODO(jbbieniekTT): temporary change from 2t to 32t
    create_cb(program, input_tensor.dtype(), IntImgCB::AXIS_2_BUFFER, core_range_set, 4);
    create_cb(program, input_tensor.dtype(), IntImgCB::AXIS_3_BUFFER_0, core_range_set, 32);
    create_cb(program, input_tensor.dtype(), IntImgCB::AXIS_3_BUFFER_1, core_range_set, 32);

    std::vector<uint32_t> compute_compile_time_args{
        static_cast<uint32_t>(IntImgCB::START),
        static_cast<uint32_t>(IntImgCB::INPUT),
        static_cast<uint32_t>(IntImgCB::ACC),
        static_cast<uint32_t>(IntImgCB::CUMSUM_STAGE_0),
        static_cast<uint32_t>(IntImgCB::CUMSUM_STAGE_1),
        static_cast<uint32_t>(IntImgCB::CUMSUM_STAGE_2),
        static_cast<uint32_t>(IntImgCB::CUMSUM_STAGE_3),
        static_cast<uint32_t>(IntImgCB::OUTPUT),
        static_cast<uint32_t>(IntImgCB::AXIS_2_BUFFER),
        static_cast<uint32_t>(IntImgCB::AXIS_3_BUFFER_0),
        static_cast<uint32_t>(IntImgCB::AXIS_3_BUFFER_1),
        tile_spec.get_height(),
        tile_spec.get_width(),
        block_depth,
        input_shape[3],
        input_shape[2],
        input_shape[1],
        input_shape[0]};
    auto dataflow_compile_time_args = compute_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(dataflow_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(dataflow_compile_time_args);
    const ReaderDataMovementConfig reader_config{dataflow_compile_time_args};
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = false,
        .compile_args = compute_compile_time_args,
        .defines = {}};
    const WriterDataMovementConfig writer_config{dataflow_compile_time_args};

    auto reader_kernel_id{create_kernel(program, KERNEL_PATHS[0], core_range_set, reader_config)};
    auto compute_kernel_id{create_kernel(program, KERNEL_PATHS[1], core_range_set, compute_config)};
    auto writer_kernel_id{create_kernel(program, KERNEL_PATHS[2], core_range_set, writer_config)};

    SetRuntimeArgs(program, reader_kernel_id, core_range_set, {src_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core_range_set, {dst_buffer->address()});

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id}};
}

void IntImgProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    auto input_buffer_address = tensor_args.input_tensor.buffer()->address();
    auto output_buffer_address = tensor_return_value.buffer()->address();
    const auto core = CoreCoord{0, 0};
    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
    reader_runtime_args[0] = input_buffer_address;
    writer_runtime_args[0] = output_buffer_address;
}

CBHandle IntImgProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const IntImgCB& intimg_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    const uint32_t cb_id{static_cast<uint32_t>(intimg_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{tt::tile_size(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle IntImgProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

}  // namespace ttnn::operations::experimental::reduction
