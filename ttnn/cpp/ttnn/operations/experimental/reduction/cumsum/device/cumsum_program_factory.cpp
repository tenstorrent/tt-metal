// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <numeric>
#include <random>
#include "cumsum_device_operation.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer.hpp"
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
    const auto& input_dtype = input_tensor.dtype();
    const auto& output_dtype = output_tensor.dtype();

    constexpr CoreCoord core{0, 0};
    constexpr uint32_t TILE_SIZE = 1024;

    const auto& tensor_shape = input_tensor.get_padded_shape();
    const uint32_t tensor_rank = tensor_shape.rank();
    int32_t dim = operation_attributes.dim;
    if (dim < 0) {  // Handle cases where dim is negative
        dim += tensor_rank;
    }

    // 1) If 2D or 1D => convert to 3D. Then finally convert back after processing
    // 2) If dim is x or y axis => permute dim with axis=0 and then do operation on dim=0
    // Then permute back matching axes

    TT_FATAL(input_dtype == output_dtype, "Type conversion not supported yet");

    TT_FATAL(
        output_dtype == DataType::FLOAT32 || output_dtype == DataType::INT32 || output_dtype == DataType::BFLOAT16,
        "Only float32 and int32 data type supported for now");

    TT_FATAL(output_tensor.get_layout() == Layout::TILE, "Only supported layout is TILE");

    TT_FATAL(input_tensor.get_padded_shape().rank() >= 3, "Only support 3D tensor and above");

    TT_FATAL(
        input_tensor.buffer()->size() == input_tensor.volume() * input_tensor.element_size(),
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
    const uint32_t single_tile_size = output_tensor.element_size() * TILE_SIZE;

    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig sram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> tmp_sram_buffer = CreateBuffer(sram_config);

    uint32_t src_bank_id = 0;
    uint32_t dst_bank_id = 0;

    constexpr uint32_t cb_in_index = CBIndex::c_0;
    constexpr uint32_t cb_out_index = CBIndex::c_1;
    constexpr uint32_t cb_zero_index = CBIndex::c_16;
    constexpr uint32_t cb_intermed_index = CBIndex::c_24;

    DataFormat in_df = datatype_to_dataformat_converter(output_dtype);

    DataFormat out_df = in_df;

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(single_tile_size, {{cb_in_index, in_df}}).set_page_size(cb_in_index, single_tile_size);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(single_tile_size, {{cb_out_index, out_df}}).set_page_size(cb_out_index, single_tile_size);

    CircularBufferConfig cb_zero_config = CircularBufferConfig(single_tile_size, {{cb_zero_index, out_df}})
                                              .set_page_size(cb_zero_index, single_tile_size);

    CircularBufferConfig cb_intermed_config = CircularBufferConfig(single_tile_size, {{cb_intermed_index, out_df}})
                                                  .set_page_size(cb_intermed_index, single_tile_size);

    CreateCircularBuffer(program, core, cb_in_config);
    CreateCircularBuffer(program, core, cb_out_config);
    CreateCircularBuffer(program, core, cb_zero_config);
    CreateCircularBuffer(program, core, cb_intermed_config);

    KernelHandle cumsum_reader_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/cumsum/device/kernels/dataflow/cumsum_reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle cumsum_writer_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/cumsum/device/kernels/dataflow/cumsum_writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {};
    KernelHandle cumsum_compute_handle_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/cumsum/device/kernels/compute/cumsum_compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args});

    // Parameters setup
    const auto& input_shape = input_tensor.get_padded_shape();
    const uint32_t input_dim = input_shape.rank();

    uint32_t num_tiles = output_tensor.volume() / TILE_SIZE;
    const uint32_t xy_volume = input_shape[input_dim - 1] * input_shape[input_dim - 2];  // W * H
    const uint32_t num_tiles_per_row = input_shape[dim];      // each row contains N independent tiles
    const uint32_t num_rows = num_tiles / num_tiles_per_row;  // total number of rows in tensor
    const uint32_t HtWt = xy_volume / TILE_SIZE;              // padded shape => xy_volume is multiple of tile_size

    uint32_t PHi = 1;
    uint32_t PLo = 1;

    for (int i = dim + 1; i + 2 < tensor_rank; i++) {
        PHi *= tensor_shape[i];
    }
    for (int i = 0; i < dim; i++) {
        PLo *= tensor_shape[i];
    }

    SetRuntimeArgs(
        program,
        cumsum_reader_handle_id,
        core,
        {
            input_tensor.buffer()->address(),
            tmp_sram_buffer->address(),
            num_rows,
            num_tiles_per_row,
            PHi,
            PLo,
            HtWt,
        });

    SetRuntimeArgs(
        program,
        cumsum_writer_handle_id,
        core,
        {
            output_tensor.buffer()->address(),
            num_rows,
            num_tiles_per_row,
            PHi,
            PLo,
            HtWt,
        });

    SetRuntimeArgs(
        program,
        cumsum_compute_handle_id,
        core,
        {
            PHi * PLo * HtWt,
            num_tiles_per_row,
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

    const auto& input_dtype = input_tensor.get_dtype();
    TT_FATAL(input_dtype == DataType::FLOAT32 || input_dtype == DataType::INT32, "Unsupported input data type");

    TT_FATAL(dtype == DataType::FLOAT32 || dtype == DataType::INT32, "Unsupported data type");
}

}  // namespace ttnn::operations::experimental::reduction
