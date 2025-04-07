// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <numeric>
#include <random>
#include "cumsum_device_operation.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "tt-metalium/command_queue.hpp"
#include "tt-metalium/data_types.hpp"
#include "tt-metalium/device.hpp"
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
    tensor_return_value_t& output_tensor) {
    // Scaffold: No 'real' program for now

    using namespace tt;

    Program program;
    const auto& input_tensor = tensor_args.input_tensor;

    IDevice* device = input_tensor.device();
    CommandQueue& cq = device->command_queue();

    constexpr CoreCoord core{0, 0};
    constexpr uint32_t TILE_SIZE = 1024;

    TT_FATAL(input_tensor.get_dtype() == output_tensor.get_dtype(), "Type conversion not supported yet");

    TT_FATAL(output_tensor.get_dtype() == DataType::FLOAT32, "Only float32 data type supported for now");

    TT_FATAL(output_tensor.get_layout() == Layout::TILE, "Only supported layout is TILE");

    const uint32_t single_tile_size = sizeof(float) * TILE_SIZE;

    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig sram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> tmp_sram_buffer = CreateBuffer(sram_config);

    uint32_t src_bank_id = 0;
    uint32_t dst_bank_id = 0;

    constexpr uint32_t cb_in_index = CBIndex::c_0;
    constexpr uint32_t cb_out_index = CBIndex::c_1;

    DataFormat in_df = DataFormat::Float32;
    DataFormat out_df = DataFormat::Float32;

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(single_tile_size, {{cb_in_index, in_df}}).set_page_size(cb_in_index, single_tile_size);

    CBHandle cb_in = CreateCircularBuffer(program, core, cb_in_config);

    KernelHandle cumsum_reader_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/cumsum/device/kernels/dataflow/cumsum_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    uint32_t num_tiles = output_tensor.volume() / TILE_SIZE;  // TODO: Not sure

    SetRuntimeArgs(
        program,
        cumsum_reader_handle_id,
        core,
        {
            input_tensor.buffer()->address(),
            output_tensor.buffer()->address(),
            tmp_sram_buffer->address(),
            src_bank_id,
            dst_bank_id,
            num_tiles,
        });

    return {std::move(program), {}};
}

void CumSumDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations::experimental::reduction
