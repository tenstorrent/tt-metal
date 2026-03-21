// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::complex_binary {

ComplexTensor add(
    const ComplexTensor& input_a, const ComplexTensor& input_b, const tt::tt_metal::MemoryConfig& output_mem_config);
ComplexTensor subtract(
    const ComplexTensor& input_a, const ComplexTensor& input_b, const tt::tt_metal::MemoryConfig& output_mem_config);
ComplexTensor multiply(
    const ComplexTensor& input_a, const ComplexTensor& input_b, const tt::tt_metal::MemoryConfig& output_mem_config);
ComplexTensor divide(
    const ComplexTensor& input_a, const ComplexTensor& input_b, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::complex_binary
