// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/wavelet_tensor_validation.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::prim::wavelet_tensor_validation {

void validate_device_tensor(const Tensor& tensor, const char* tensor_name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be a device tensor", tensor_name);
    TT_FATAL(
        tensor.is_allocated() && tensor.buffer() != nullptr, "{} must have an allocated device buffer", tensor_name);
    TT_FATAL(tensor.device() != nullptr, "{} has no device", tensor_name);
    TT_FATAL(tensor.device()->num_devices() == 1, "{} must be placed on exactly one physical device", tensor_name);
    TT_FATAL(tensor.dtype() == DataType::FLOAT32, "{} must have FLOAT32 dtype", tensor_name);
}

void validate_input_memory_config(const MemoryConfig& memory_config, const char* tensor_name) {
    const bool supported_buffer = memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM ||
                                  memory_config.buffer_type() == tt::tt_metal::BufferType::L1;
    TT_FATAL(
        memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED && supported_buffer &&
            !memory_config.is_sharded(),
        "{} must use INTERLEAVED memory with DRAM or L1 storage; sharded inputs are unsupported",
        tensor_name);
}

void validate_output_memory_config(const MemoryConfig& memory_config, const char* operation_name) {
    TT_FATAL(
        memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM && !memory_config.is_sharded(),
        "{} supports only DRAM-interleaved outputs in its first TTNN version",
        operation_name);
}

void validate_preallocated_output_placement(
    const Tensor& output, const tt::tt_metal::distributed::MeshDevice* expected_device, const char* output_name) {
    validate_output_memory_config(output.memory_config(), output_name);
    TT_FATAL(output.device() == expected_device, "{} must be on the same device as the inputs", output_name);
}

void validate_same_device(
    const Tensor& tensor, const tt::tt_metal::distributed::MeshDevice* expected_device, const char* error_message) {
    TT_FATAL(tensor.device() == expected_device, "{}", error_message);
}

void validate_distinct_buffers(const Tensor& lhs, const Tensor& rhs, const char* error_message) {
    TT_FATAL(lhs.buffer() != rhs.buffer(), "{}", error_message);
}

}  // namespace ttnn::prim::wavelet_tensor_validation
