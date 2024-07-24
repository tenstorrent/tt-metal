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

Tensor outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


}  // namespace tt_metal

}  // namespace tt
