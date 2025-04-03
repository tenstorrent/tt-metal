// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "cumsum_device_operation.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/data_types.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::reduction {

CumSumDeviceOperation::SingleCore::cached_program_t CumSumDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Scaffold: No 'real' program for now

    const auto& input_tensor = tensor_args.input_tensor;

    using namespace tt;
    using namespace tt::tt_metal;

    IDevice* device = input_tensor.device();
    CommandQueue& cq = device->command_queue();

    Program program;

    constexpr CoreCoord core{0, 0};

    const uint32_t single_tile_size = input_tensor.element_size() * 1024;
    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    uint32_t src_bank_id = 0;
    uint32_t dst_bank_id = 0;

    // Write input tensor data into DRAM (?)
    // Is this needed in this function ?
    EnqueueWriteBuffer(cq, src_dram_buffer, input_tensor.buffer(), false);

    constexpr uint32_t src_cb_index = CBIndex::c_0;

    DataFormat in_df = datatype_to_dataformat_converter(input_tensor.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(operation_attributes.dtype);

    CircularBufferConfig cb_config =
        CircularBufferConfig(single_tile_size, {{src_cb_index, in_df}}).set_page_size(src_cb_index, single_tile_size);
    CBHandle cb_src = CreateCircularBuffer(program, core, cb_config);

    KernelHandle cumsum_reader_handle_id = CreateKernel(
        program,
        "ttnn/cpp/operations/experimental/reduction/cumsum/kernels/dataflow/cumsum_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(
        program,
        cumsum_reader_handle_id,
        core,
        {src_dram_buffer->address(), dst_dram_buffer->address(), src_bank_id, dst_bank_id});

    return {std::move(program), {}};
}

void CumSumDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::experimental::reduction
