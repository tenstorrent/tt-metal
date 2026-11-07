// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed {

class MeshDevice;

}  // namespace tt::tt_metal::distributed

namespace ttnn::prim::wavelet_tensor_validation {

void validate_device_tensor(const Tensor& tensor, const char* tensor_name);

void validate_input_memory_config(const MemoryConfig& memory_config, const char* tensor_name);

void validate_output_memory_config(const MemoryConfig& memory_config, const char* operation_name);

void validate_preallocated_output_placement(
    const Tensor& output, const tt::tt_metal::distributed::MeshDevice* expected_device, const char* output_name);

void validate_same_device(
    const Tensor& tensor, const tt::tt_metal::distributed::MeshDevice* expected_device, const char* error_message);

void validate_distinct_buffers(const Tensor& lhs, const Tensor& rhs, const char* error_message);

}  // namespace ttnn::prim::wavelet_tensor_validation
