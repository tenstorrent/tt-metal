// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_device_operation.hpp"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/util.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::reduction {
CumprodDeviceOperation::SingleCoreCumprodProgramFactory::cached_program_t
CumprodDeviceOperation::SingleCoreCumprodProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::detail;

    const auto& input_tensor{tensor_args.input_tensor};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.get_padded_shape()};
    const auto& dim{operation_attributes.dim};

    Program program{};

    IDevice* device{input_tensor.device()};

    auto src_buffer{input_tensor.buffer()};
    auto dst_buffer{output_tensor.buffer()};

    constexpr CoreCoord core{1, 1};

    auto cb_src{create_cb(program, input_tensor.get_dtype(), CumprodCB::SRC, core)};
    auto cb_acc{create_cb(program, output_tensor.get_dtype(), CumprodCB::ACC, core)};
    auto cb_dst{create_cb(program, output_tensor.get_dtype(), CumprodCB::DST, core)};

    const uint32_t src_is_dram{src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0};
    const uint32_t dst_is_dram{dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(output_tensor.get_dtype())};
    const bool fp32_dest_acc_en{
        (dst_cb_data_format == tt::DataFormat::Float32) || (dst_cb_data_format == tt::DataFormat::Int32) ||
        (dst_cb_data_format == tt::DataFormat::UInt32)};
    const uint32_t height_tiles{input_shape[2] / constants::TILE_HEIGHT};
    const uint32_t width_tiles{input_shape[3] / constants::TILE_WIDTH};

    const std::vector<uint32_t> compile_args{
        src_is_dram,
        dst_is_dram,
        static_cast<uint32_t>(CumprodCB::SRC),
        static_cast<uint32_t>(CumprodCB::ACC),
        static_cast<uint32_t>(CumprodCB::DST),
        input_shape[0],
        input_shape[1],
        height_tiles,
        width_tiles};

    const ReaderDataMovementConfig reader_config{compile_args};
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = false,
        .compile_args = compile_args};
    const WriterDataMovementConfig writer_config{compile_args};

    auto cumprod_reader_kernel_id{create_kernel(program, core, CumprodCB::SRC, reader_config)};
    auto cumprod_compute_sc_kernel_id{create_kernel(program, core, CumprodCB::ACC, compute_config)};
    auto cumprod_writer_kernel_id{create_kernel(program, core, CumprodCB::DST, writer_config)};

    return {
        std::move(program),
        {.cumprod_reader_kernel_id = cumprod_reader_kernel_id,
         .cumprod_compute_kernel_id = cumprod_compute_sc_kernel_id,
         .cumprod_writer_kernel_id = cumprod_writer_kernel_id}};
}

void CumprodDeviceOperation::SingleCoreCumprodProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program{cached_program.program};
    const auto& cumprod_reader_kernel_id{cached_program.shared_variables.cumprod_reader_kernel_id};
    const auto& cumprod_compute_kernel_id{cached_program.shared_variables.cumprod_compute_kernel_id};
    const auto& cumprod_writer_kernel_id{cached_program.shared_variables.cumprod_writer_kernel_id};

    auto src_buffer{tensor_args.input_tensor.buffer()};
    auto dst_buffer{tensor_return_value.buffer()};

    constexpr CoreCoord core{1, 1};
    auto& reader_runtime_args{GetRuntimeArgs(program, cumprod_reader_kernel_id, core)};
    reader_runtime_args[0] = src_buffer->address();
    reader_runtime_args[1] = dst_buffer->address();
    auto& compute_runtime_args{GetRuntimeArgs(program, cumprod_reader_kernel_id, core)};
    compute_runtime_args[0] = src_buffer->address();
    compute_runtime_args[1] = dst_buffer->address();
    auto& writer_runtime_args{GetRuntimeArgs(program, cumprod_reader_kernel_id, core)};
    writer_runtime_args[0] = src_buffer->address();
    writer_runtime_args[1] = dst_buffer->address();
}

CBHandle CumprodDeviceOperation::SingleCoreCumprodProgramFactory::create_cb(
    Program& program, const DataType& dtype, const CumprodCB& cumprod_cb, const CoreCoord& core) {
    using tt::tt_metal::detail::TileSize;
    const uint32_t cb_id{static_cast<uint32_t>(cumprod_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{TileSize(cb_data_format)};
    const auto cb_config{CircularBufferConfig{CB_NUM_TILES * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core, cb_config);
}

KernelHandle CumprodDeviceOperation::SingleCoreCumprodProgramFactory::create_kernel(
    Program& program,
    const CoreCoord& core,
    const CumprodCB& cumprod_cb,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    const uint32_t cb_id{static_cast<uint32_t>(cumprod_cb)};
    const auto& path_to_kernel{KERNEL_PATHS[cb_id]};

    auto kernel_id{CreateKernel(program, path_to_kernel, core, config)};
    // TODO(jbbieniekTT): make sure about this
    // SetRuntimeArgs(program, kernel_id, core, runtime_args);

    return kernel_id;
}

}  // namespace ttnn::operations::experimental::reduction
