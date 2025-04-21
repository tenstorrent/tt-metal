// ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_binary_scalar_row_major_program_factory.cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise_binary_scalar_row_major_program_factory.hpp"

#include <cmath>  // For std::ceil

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"  // For detail::TileSize
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_device_operation.hpp"  // Include for tensor_args_t etc.

namespace ttnn::operations::binary {

using namespace tt::tt_metal;
using namespace tt::constants;

// Basic validation - ensure input is RowMajor and scalar is provided
void EltwiseBinaryScalarRowMajor::validate(const tensor_args_t& tensor_args) const {
    const auto& input_tensor = tensor_args.input_tensor_a;
    TT_FATAL(
        input_tensor.get_layout() == Layout::ROW_MAJOR,
        "Input tensor must be ROW_MAJOR for EltwiseBinaryScalarRowMajor.");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device.");
    // Assuming scalar is passed via operation_attributes, validation happens there or during create/override.
}

// Output shape is same as input shape
std::vector<tt::tt_metal::Shape> EltwiseBinaryScalarRowMajor::compute_output_shapes(
    const tensor_args_t& tensor_args) const {
    return {tensor_args.input_tensor_a.get_shape()};
}

// Create output tensor with same shape/layout/dtype as input
std::vector<Tensor> EltwiseBinaryScalarRowMajor::create_output_tensors(const tensor_args_t& tensor_args) const {
    const auto& input_tensor = tensor_args.input_tensor_a;
    // Use input memory config unless overridden
    // Note: If tensor_args.output_tensor exists, ttnn might handle allocation? Check ttnn patterns.
    // Assuming direct creation for now.
    return {create_device_tensor(
        compute_output_shapes(tensor_args)[0],
        input_tensor.get_dtype(),
        Layout::ROW_MAJOR,  // Keep RowMajor!
        input_tensor.device(),
        tensor_args.output_mem_config.value_or(input_tensor.memory_config()))};
}

// --- create function ---
EltwiseBinaryScalarRowMajor::cached_program_t EltwiseBinaryScalarRowMajor::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor_a;
    auto& output_tensor = tensor_return_value;  // Assumes output tensor is created/provided correctly

    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR, "Input must be RowMajor.");
    TT_FATAL(operation_attributes.scalar.has_value(), "Scalar value must be provided for this operation.");
    TT_FATAL(
        operation_attributes.binary_op_type == BinaryOpType::ADD,
        "This factory currently only supports ADD. Extend if needed.");  // Example limitation

    // Get device, shape, dtype etc.
    Device* device = input_tensor.device();
    const auto& shape = input_tensor.get_shape();  // Use logical shape for element count
    DataType dtype = input_tensor.get_dtype();
    TT_FATAL(dtype == DataType::BFLOAT16, "This RowMajor kernel currently supports BFLOAT16 only.");

    uint32_t num_elements = shape.volume();
    uint32_t element_size = tt::tt_metal::detail::DataTypeSize(dtype);  // Should be 2 for BF16

    // Use single core for simplicity
    CoreCoord core = {0, 0};  // Target core 0,0
    // TODO: Extend to multi-core: would need work splitting logic here.

    // L1 Buffer Allocation (choose sizes carefully)
    // We need space for at least one chunk for reader->compute and one for compute->writer.
    // Let's use a chunk size related to tile size for simplicity, although not strictly necessary.
    // TODO: Make chunk size configurable or dynamically calculated based on L1 size.
    constexpr uint32_t TILE_ELEMENTS = TILE_HW;  // 32*32 = 1024
    uint32_t chunk_num_elements = TILE_ELEMENTS;
    uint32_t chunk_size_bytes = chunk_num_elements * element_size;

    // Need enough L1 for src data chunk and dst data chunk used by compute
    uint32_t l1_buffer_size = chunk_size_bytes;  // Size for one chunk
    uint32_t src_l1_addr = L1_UNRESERVED_BASE;
    uint32_t dst_l1_addr = src_l1_addr + l1_buffer_size;
    TT_FATAL((dst_l1_addr + l1_buffer_size) <= device->l1_size_per_core(), "Insufficient L1 size for buffers.");

    // Create Program
    Program program{};

    // Create Kernels
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_row_major.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,  // NOC0 for DRAM reads
            .noc = NOC::RISCV_1_default});

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary_scalar_row_major.cpp",
        core,
        ComputeConfig{});  // Default compute config

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_row_major.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,  // NOC1 for DRAM writes
            .noc = NOC::RISCV_0_default});

    // Runtime arguments will be set by override_runtime_arguments

    // Return cached program info
    return cached_program_t{
        .program = std::move(program),
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id};
}

// --- override_runtime_arguments function ---
void EltwiseBinaryScalarRowMajor::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor_a;
    auto& output_tensor = tensor_return_value;

    Device* device = input_tensor.device();
    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer must be allocated!");

    const auto& shape = input_tensor.get_shape();
    uint32_t num_elements = shape.volume();
    DataType dtype = input_tensor.get_dtype();
    uint32_t element_size = tt::tt_metal::detail::DataTypeSize(dtype);

    // Use the same chunk size logic as in create
    constexpr uint32_t TILE_ELEMENTS = TILE_HW;
    uint32_t chunk_num_elements = TILE_ELEMENTS;
    uint32_t chunk_size_bytes = chunk_num_elements * element_size;

    // Get L1 addresses calculated in 'create' (or recalculate if dynamic)
    uint32_t src_l1_addr = L1_UNRESERVED_BASE;              // Must match 'create'
    uint32_t dst_l1_addr = src_l1_addr + chunk_size_bytes;  // Must match 'create'

    // Pack the scalar value (float -> uint32)
    float scalar_f32 = operation_attributes.scalar.value();
    uint32_t scalar_packed = *reinterpret_cast<uint32_t*>(&scalar_f32);

    // Core coordination - single core for now
    CoreCoord core = {0, 0};

    // Set Runtime Arguments for each kernel
    // Reader Args: src_dram_addr, dst_l1_addr, total_num_elements, chunk_num_elements
    SetRuntimeArgs(
        cached_program.program,
        cached_program.reader_kernel_id,
        core,
        {src_buffer->address(), src_l1_addr, num_elements, chunk_num_elements});

    // Compute Args: src_addr, dst_addr, num_elements, scalar_value_uint32
    SetRuntimeArgs(
        cached_program.program,
        cached_program.compute_kernel_id,
        core,
        {src_l1_addr, dst_l1_addr, num_elements, scalar_packed});

    // Writer Args: src_l1_addr, dst_dram_addr, total_num_elements, chunk_num_elements
    SetRuntimeArgs(
        cached_program.program,
        cached_program.writer_kernel_id,
        core,
        {dst_l1_addr, dst_buffer->address(), num_elements, chunk_num_elements});
}

}  // namespace ttnn::operations::binary
