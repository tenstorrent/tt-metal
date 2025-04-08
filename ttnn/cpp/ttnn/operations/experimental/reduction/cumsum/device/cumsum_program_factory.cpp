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

    // Device setup
    Program program;
    IDevice* device = output_tensor.device();

    // Parameters setup

    const auto& input_tensor = tensor_args.input_tensor;

    constexpr CoreCoord core{0, 0};
    constexpr uint32_t TILE_SIZE = 1024;

    const auto& tensor_shape = input_tensor.get_padded_shape();
    const uint32_t tensor_rank = tensor_shape.rank();
    int32_t dim = operation_attributes.dim;
    if (dim < 0) {  // Handle cases where dim is negative
        dim += tensor_rank;
    }

    TT_FATAL(input_tensor.get_dtype() == output_tensor.get_dtype(), "Type conversion not supported yet");

    TT_FATAL(output_tensor.get_dtype() == DataType::FLOAT32, "Only float32 data type supported for now");

    TT_FATAL(output_tensor.get_layout() == Layout::TILE, "Only supported layout is TILE");

    TT_FATAL(input_tensor.get_padded_shape().rank() >= 3, "Only support 4D tensor");

    TT_FATAL(
        input_tensor.get_padded_shape()[tensor_rank - 1] % 32 == 0 &&
            input_tensor.get_padded_shape()[tensor_rank - 2] % 32 == 0,
        "Input tensor must have padding");

    TT_FATAL(
        output_tensor.get_padded_shape()[tensor_rank - 1] % 32 == 0 &&
            output_tensor.get_padded_shape()[tensor_rank - 2] % 32 == 0,
        "Output tensor must have padding");

    TT_FATAL(
        input_tensor.buffer()->size() == input_tensor.volume() * sizeof(float),
        "Input tensor size does not match expected volume");

    TT_FATAL(input_tensor.get_logical_volume() > 0, "Input must not be empty");

    TT_ASSERT(dim >= 0, "dim argument must be positive");

    TT_FATAL(dim + 2 < tensor_rank, "cumsum on x and y axes not supported (dim = {}, rank = {})", dim, tensor_rank);

    printf("Input size = %ld, input volume = %u\n", input_tensor.buffer()->size(), input_tensor.volume());
    printf(
        "Input buffer addr = %u, Output buffer addr = %u\n",
        input_tensor.buffer()->address(),
        output_tensor.buffer()->address());

    // Buffer setup
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

    // Parameters setup

    const auto& input_shape = input_tensor.get_padded_shape();
    const uint32_t input_dim = input_shape.rank();

    // for dim != rank-1 && dim != rank-2

    // TOOD: If dim == x-axis or y-axis then transform both input and output

    uint32_t num_tiles = output_tensor.volume() / TILE_SIZE;  // TODO: Not sure
    const uint32_t xy_volume = input_shape[input_dim - 1] * input_shape[input_dim - 2];  // W * H
    const uint32_t num_tiles_per_row = input_shape[dim];      // each row contains N independent tiles
    const uint32_t num_rows = num_tiles / num_tiles_per_row;  // total number of rows in tensor
    const uint32_t HtWt = xy_volume / TILE_SIZE;              // padded shape => xy_volume is multiple of tile_size

    uint32_t high_dims = 1;
    uint32_t low_dims = 1;

    for (int i = dim + 1; i + 2 < tensor_rank; i++) {
        high_dims *= tensor_shape[i];
    }
    for (int i = 0; i < dim; i++) {
        low_dims *= tensor_shape[i];
    }

    SetRuntimeArgs(
        program,
        cumsum_reader_handle_id,
        core,
        {
            input_tensor.buffer()->address(),
            output_tensor.buffer()->address(),
            tmp_sram_buffer->address(),
            num_rows,
            num_tiles_per_row,
            high_dims,
            low_dims,
            HtWt,
        });

    return {std::move(program), {}};
}

void CumSumDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dtype = operation_attributes.dtype;

    TT_FATAL(false, "false");

    TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32, "Unsupported input data type");

    TT_FATAL(dtype == DataType::FLOAT32, "Unsupported data type");
}

}  // namespace ttnn::operations::experimental::reduction
