// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

#include "tt_metal/common/constants.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_device_operation.hpp"

namespace tt {

namespace tt_metal {

using unary_tensor_op_t = Tensor(const Tensor& a);
using binary_tensor_op_t = Tensor(const Tensor& a, const Tensor& b);

// Note: inline doesn't allow pybind to work well so we keep few function not inlined.


// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor mac(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mac(
    const Tensor& a, float b, float c, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// compute polyval by Horner's rule
Tensor polyval(
    const Tensor& input_tensor,
    std::vector<float> coeffs,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor rfloor_div(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


// lerp(input, end, weight) = start + weight * (end - start), weight is float
Tensor lerp(
    const Tensor& input_a,
    const Tensor& input_b,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// lerp(input, end, weight) = start + weight * (end - start), weight is tensor
Tensor lerp(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor scatter(
    const Tensor& input_a,
    const Tensor& input_b,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor= std::nullopt);

Tensor zeros_like(
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor= std::nullopt);

// on-device tensor creation 1s like @reference_tensor
Tensor ones_like(
    const Tensor& reference_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation with value like @reference_tensor
Tensor full_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor= std::nullopt);
Tensor full_like(
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<Tensor> output_tensor= std::nullopt);

// on-device tensor creation 0s with shape
Tensor empty(
    const Shape shape,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 0s with shape
Tensor zeros(
    const Shape shape,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation 1s with shape
Tensor ones(
    const Shape shape,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor arange(
    int32_t start,
    int32_t end,
    int32_t step = 1,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation with shape and filled with value
Tensor full(
    const Shape shape,
    float value,
    DataType data_type = DataType::BFLOAT16,
    Layout layout = Layout::ROW_MAJOR,
    Device* device = nullptr,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// machine epsilon
Tensor eps(
    const Shape shape,
    Layout layout,
    Device* device,
    const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// on-device tensor creation with shape and filled with value
Tensor sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config);

}  // namespace tt_metal

}  // namespace tt
